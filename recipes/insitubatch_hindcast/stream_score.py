"""Streaming vs dense hindcast scoring: interleave verification with the model rollout.

The Earth2Studio pattern (`fetch_data -> map_coords -> create_iterator`) materializes the whole
`(init, lead)` verification grid up front (its `recipes/eval` even has a predownload sentinel,
because live fetch is slow). That dense tensor is a *slow-IO shortcut*, not a requirement: the
model's `create_iterator` already STREAMS the forecast lead-by-lead, and scoring is pointwise per
`(init, lead)`. So the verification never needs to be dense -- read each lead's ground truth as the
rollout produces it, accumulate RMSE, discard.

Three modes over one hindcast campaign (N inits x L leads x V vars), scored vs ERA5 with Persistence:
  e2s    = E2S WB2 fetch_data (redundant per-(time,var) reads) -> dense grid -> roll out + score.
  dense  = insitubatch, batch_size=N -> one dense materialization (the E2S shape, deduped reads).
  stream = insitubatch, batch_size=W -> windowed stream; roll out + score each window inline, discard.

Reports wall, peak RSS (getrusage), and RMSE-per-lead (must match across modes = correctness).
`dense` vs `stream` isolates the *materialization* axis (same reads, same backend): peak memory
N*L vs W*L. `e2s` adds the redundant-read wall of the status quo.
"""

import argparse
import resource
import time
from collections import OrderedDict

import numpy as np
import torch

from earth2studio.data.insitu import InSituForecastFeed, decode_cf_time
from earth2studio.models.px import Persistence
from earth2studio.utils.coords import map_coords
from insitubatch import fsspec_store

URL = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
VAR_MAP = {"t2m": "2m_temperature", "u10m": "10m_u_component_of_wind", "v10m": "10m_v_component_of_wind"}
DT = np.timedelta64(6, "h")


def anon_store():
    return fsspec_store(URL, token="anon", access="read_only")  # noqa: S106


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # ru_maxrss is KB on Linux


class RmseAccumulator:
    """Per-lead running MSE over (init, var, lat, lon); sqrt at the end."""

    def __init__(self):
        self.sse: dict[int, float] = {}
        self.n: dict[int, int] = {}

    def update(self, lead_h: int, pred: torch.Tensor, truth: torch.Tensor):
        e = (pred - truth).float()
        self.sse[lead_h] = self.sse.get(lead_h, 0.0) + float((e * e).sum())
        self.n[lead_h] = self.n.get(lead_h, 0) + e.numel()

    def table(self):
        return {h: (self.sse[h] / self.n[h]) ** 0.5 for h in sorted(self.sse)}


def build_feed(variables, start, n_init, leads_h, batch_size):
    leads = np.array([np.timedelta64(h, "h") for h in leads_h])  # includes 0 (the IC)
    return InSituForecastFeed(
        anon_store(), variables=variables, var_map={v: VAR_MAP[v] for v in variables},
        lead_times=leads, sample_range=(start, start + n_init),
        batch_size=batch_size, transpose_inner=True,
    )


def make_model(variables, feed):
    domain = OrderedDict([("lat", feed.lat), ("lon", feed.lon)])
    return Persistence(variable=variables, domain_coords=domain, history=1, dt=DT)


def score_window(model, x_all, coords_all, leads_h, nsteps, acc):
    """Roll Persistence out over one window and score each lead vs the pre-read verification slice.

    ``x_all`` is (W, L+1, V, lat, lon) with lead index 0 = IC and index k = verification at leads_h[k].
    """
    ic_x = x_all[:, 0:1]
    ic_coords = OrderedDict(
        [("time", coords_all["time"]), ("lead_time", coords_all["lead_time"][0:1]),
         ("variable", coords_all["variable"]), ("lat", coords_all["lat"]), ("lon", coords_all["lon"])]
    )
    ic_x, ic_coords = map_coords(ic_x, ic_coords, model.input_coords())
    for step, (fx, fcoords) in enumerate(model.create_iterator(ic_x, ic_coords)):
        if step == 0:
            continue  # step 0 is the IC (lead 0); score forecast leads only
        acc.update(leads_h[step], fx[:, -1], x_all[:, step])
        if step == nsteps:
            break


def run_insitu(variables, start, n_init, leads_h, nsteps, batch_size):
    feed = build_feed(variables, start, n_init, leads_h, batch_size)
    model = make_model(variables, feed)
    acc = RmseAccumulator()
    for x_all, coords_all in feed:  # one batch per window (stream) or the whole campaign (dense)
        score_window(model, x_all, coords_all, leads_h, nsteps, acc)
    decodes = feed.dataset.cache_misses
    feed.dataset.close()
    return acc, decodes


def run_e2s(variables, start, n_init, leads_h, nsteps):
    from earth2studio.data.wb2 import WB2ERA5_121x240

    src = WB2ERA5_121x240(cache=False, verbose=False)
    g_store = anon_store()
    import zarr

    g = zarr.open_group(store=g_store, mode="r")
    attrs = dict(g["time"].attrs)
    times = decode_cf_time(np.asarray(g["time"][:]), attrs["units"], attrs.get("calendar", "standard"))
    inits = times[start : start + n_init]
    feed = build_feed(variables, start, n_init, leads_h, batch_size=n_init)  # for coords/model only
    model = make_model(variables, feed)
    feed.dataset.close()

    # Dense predownload: fetch every (init, lead) valid time (redundant reads), materialize.
    lat = np.asarray(g["latitude"][:]).astype(np.float32)
    lon = np.asarray(g["longitude"][:]).astype(np.float32)
    dense = np.empty((n_init, len(leads_h), len(variables), len(lat), len(lon)), dtype=np.float32)
    for i, it in enumerate(inits):
        # Realistic per-init verification fetch: all leads for this init in one call.
        valids = [(np.datetime64(it) + np.timedelta64(h, "h")).astype("datetime64[s]").astype("O")
                  for h in leads_h]
        da = src(valids, list(variables))  # (L+1, V, lat, lon) xr.DataArray
        dense[i] = np.asarray(da.values)
    x_all = torch.from_numpy(dense)
    coords_all = OrderedDict(
        [("time", inits), ("lead_time", np.array([np.timedelta64(h, "h") for h in leads_h], dtype="timedelta64[ns]")),
         ("variable", np.asarray(variables)), ("lat", lat), ("lon", lon)]
    )
    acc = RmseAccumulator()
    score_window(model, x_all, coords_all, leads_h, nsteps, acc)
    return acc, n_init * len(leads_h) * len(variables)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["stream", "dense", "e2s"], required=True)
    p.add_argument("--vars", nargs="+", default=["t2m", "u10m", "v10m"])
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--n-init", type=int, default=120)
    p.add_argument("--n-leads", type=int, default=40)
    p.add_argument("--window", type=int, default=8)
    args = p.parse_args()

    leads_h = [6 * k for k in range(args.n_leads + 1)]  # 0, 6, .., n_leads*6  (0 = IC)
    nsteps = args.n_leads

    t0 = time.perf_counter()
    if args.mode == "e2s":
        acc, reads = run_e2s(args.vars, args.start, args.n_init, leads_h, nsteps)
    else:
        bs = args.window if args.mode == "stream" else args.n_init
        acc, reads = run_insitu(args.vars, args.start, args.n_init, leads_h, nsteps, bs)
    wall = time.perf_counter() - t0

    rmse = acc.table()
    print(f"mode={args.mode}  grid={args.n_init}x{args.n_leads}x{len(args.vars)}  window={args.window}")
    print(f"  wall        : {wall:.2f} s")
    print(f"  peak RSS    : {peak_rss_gb():.2f} GB")
    print(f"  reads/decodes: {reads}")
    print(f"  RMSE @ 24h/120h/240h: "
          f"{rmse.get(24, float('nan')):.3f} / {rmse.get(120, float('nan')):.3f} / "
          f"{rmse.get(min(240, nsteps*6), float('nan')):.3f}")


if __name__ == "__main__":
    main()

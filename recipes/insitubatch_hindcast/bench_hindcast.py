"""Before/after hindcast verification-read benchmark on WB2 / ARCO ERA5.

Scenario: score an ``N_init x len(leads)`` forecast grid against ERA5. Every ``(init, lead)``
needs ERA5 at ``valid = init + lead``; consecutive init times share valid times, so the
requested reads collapse onto far fewer stored chunks.

BEFORE  = Earth2Studio's ERA5 source (``fetch`` gathers one read per (time, variable), no
          dedup) -- the realistic per-init eval fetch.
AFTER   = insitubatch InSituForecastFeed over the init window with the leads as shift views;
          each shared chunk is decoded exactly once (``dataset.cache_misses``).

Both read the SAME store over gcsfs anon, so the delta isolates insitubatch's
dedup + bounded prefetch (not obstore-vs-gcsfs).

Two regimes:
  wb2  = 240x121 6-hourly, chunks=(8,240,121): fat time-chunk -> high dedup ratio, tiny
         fields -> wall gated by concurrency (wall speedup << read reduction).
  arco = 721x1440 1-hourly, chunks=(1,721,1440): chunk-1 -> dedup = pure valid-time overlap,
         4MB fields -> genuinely IO-bound (wall speedup tracks read reduction).
"""

import argparse
import time

import numpy as np
from insitubatch import fsspec_store

from earth2studio.data.insitu import InSituForecastFeed, decode_cf_time

# store id -> (url, before source class, inner (H,W), transpose store(lon,lat)->(lat,lon))
STORES = {
    "wb2": {
        "url": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        "before": "earth2studio.data.wb2:WB2ERA5_121x240",
        "field_bytes": 240 * 121 * 4,
        "chunk_steps": 8,
        "transpose_inner": True,
    },
    "arco": {
        "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        "before": "earth2studio.data.arco:ARCO",
        "field_bytes": 721 * 1440 * 4,
        "chunk_steps": 1,
        "transpose_inner": False,
    },
}
VAR_MAP = {
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
}


def anon_store(url):
    return fsspec_store(url, token="anon", access="read_only")  # noqa: S106


def load_before_cls(spec):
    mod, _, name = spec.partition(":")
    import importlib

    return getattr(importlib.import_module(mod), name)


def run_after(cfg, variables, start, n_init, leads_h, batch_size, max_inflight):
    leads = np.array([np.timedelta64(h, "h") for h in leads_h])
    feed = InSituForecastFeed(
        anon_store(cfg["url"]),
        variables=variables,
        var_map={v: VAR_MAP[v] for v in variables},
        lead_times=leads,
        sample_range=(start, start + n_init),
        batch_size=batch_size,
        max_inflight=max_inflight,
        transpose_inner=cfg["transpose_inner"],
    )
    t0 = time.perf_counter()
    n_rows = 0
    for x, _coords in feed:
        n_rows += x.shape[
            0
        ]  # gather returns eager numpy; no touch needed to force decode
    wall = time.perf_counter() - t0
    feed.dataset.close()
    return {
        "wall_s": wall,
        "init_rows": n_rows,
        "chunk_decodes": feed.dataset.cache_misses,
        "resident_peak": feed.dataset.resident_peak,
    }


def run_before(before_cls, variables, init_times64, leads_h):
    src = before_cls(cache=False, verbose=False)
    init_dt = init_times64.astype("datetime64[s]").astype("O")
    leads_td = [np.timedelta64(h, "h") for h in leads_h]
    t0 = time.perf_counter()
    for it in init_dt:
        valid = [
            (np.datetime64(it) + td).astype("datetime64[s]").astype("O")
            for td in leads_td
        ]
        src(valid, list(variables))  # realistic per-init verification fetch
    return {"wall_s": time.perf_counter() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", choices=list(STORES), default="wb2")
    p.add_argument("--vars", nargs="+", default=["t2m"])
    p.add_argument("--start", type=int, default=1000)
    p.add_argument("--n-init", type=int, default=24)
    p.add_argument("--lead-step-h", type=int, default=6)
    p.add_argument("--max-lead-h", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--skip-before", action="store_true")
    args = p.parse_args()

    cfg = STORES[args.store]
    leads_h = list(range(args.lead_step_h, args.max_lead_h + 1, args.lead_step_h))
    requested = args.n_init * len(leads_h) * len(args.vars)

    import zarr

    g = zarr.open_group(store=anon_store(cfg["url"]), mode="r")
    attrs = dict(g["time"].attrs)
    times64 = decode_cf_time(
        np.asarray(g["time"][:]), attrs["units"], attrs.get("calendar", "standard")
    )
    init_times64 = times64[args.start : args.start + args.n_init]

    print(
        f"[{args.store}] grid: {args.n_init} inits x {len(leads_h)} leads x {len(args.vars)} vars "
        f"= {requested} requested field-reads ({requested*cfg['field_bytes']/1e9:.2f} GB naive)"
    )
    print(
        f"leads: {args.lead_step_h}h..{args.max_lead_h}h ; vars: {args.vars} ; repeats: {args.repeats}"
    )

    def med3(w):
        w = sorted(w)
        return w[len(w) // 2], w[0], w[-1]

    before_cls = load_before_cls(cfg["before"])
    after_walls, before_walls = [], []
    decodes = resident = None
    for r in range(args.repeats):
        a = run_after(
            cfg,
            args.vars,
            args.start,
            args.n_init,
            leads_h,
            args.batch_size,
            args.max_inflight,
        )
        after_walls.append(a["wall_s"])
        decodes, resident = a["chunk_decodes"], a["resident_peak"]
        if not args.skip_before:
            before_walls.append(
                run_before(before_cls, args.vars, init_times64, leads_h)["wall_s"]
            )
        print(
            f"  repeat {r+1}/{args.repeats}: after={after_walls[-1]:.2f}s"
            + (f"  before={before_walls[-1]:.2f}s" if before_walls else "")
        )

    dedup = requested / decodes
    a_med, a_lo, a_hi = med3(after_walls)
    print("\n=== AFTER (insitubatch, gcsfs anon) ===")
    print(f"  wall (med/min/max): {a_med:.2f} / {a_lo:.2f} / {a_hi:.2f} s")
    print(
        f"  chunk decodes  : {decodes}  ({decodes*cfg['field_bytes']*cfg['chunk_steps']/1e9:.2f} GB)"
    )
    print(
        f"  dedup ratio    : {dedup:.1f}x  ({requested} requested -> {decodes} decoded)"
    )
    print(f"  resident peak  : {resident} chunks")
    if before_walls:
        b_med, b_lo, b_hi = med3(before_walls)
        print("\n=== BEFORE (E2S fetch, gcsfs anon, cache off) ===")
        print(f"  wall (med/min/max): {b_med:.2f} / {b_lo:.2f} / {b_hi:.2f} s")
        print("\n=== HEADLINE (medians) ===")
        print(
            f"  speedup        : {b_med/a_med:.1f}x wall ({b_med:.2f}s -> {a_med:.2f}s)"
        )
        print(f"  read reduction : {dedup:.1f}x fewer chunk decodes")


if __name__ == "__main__":
    main()

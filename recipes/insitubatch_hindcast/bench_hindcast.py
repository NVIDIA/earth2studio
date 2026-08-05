# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Before/after hindcast verification-read benchmark on WB2 / ARCO ERA5.

Scenario: score an ``N_init x len(leads)`` forecast grid against ERA5. Every ``(init, lead)``
needs ERA5 at ``valid = init + lead``; consecutive init times share valid times, so the
requested reads collapse onto far fewer stored chunks.

BEFORE  = Earth2Studio's ERA5 source (``fetch`` gathers one read per (time, variable), no
          dedup) -- the realistic per-init eval fetch.
AFTER   = insitubatch InSituForecastFeed over the init window with the leads as shift views;
          each shared chunk is decoded exactly once (``dataset.cache_misses``).

The BEFORE leg has two configurations, and they measure different things:

  --before-cache OFF (default here)  Earth2Studio's per-source cache disabled, so every
      redundant read goes back to the cloud. Symmetric with the feed, which holds no local
      byte cache either -- but NOT how a stock Earth2Studio run behaves.
  --before-cache ON                  the source's own default (``cache=True``), which wraps
      the store in ``LocalCachingStore``. Redundant reads then hit local disk instead of the
      network. That cache sits at the ``Store.get`` level and holds *compressed* buffers, so
      zarr still decodes a fat chunk once per requested step either way -- the de-dup ratio
      is unchanged, only the fetch component of the wall moves.

Report both. The cached run is the honest status quo for wall-clock; the uncached run
isolates read elimination from local-cache effects.

Both read the SAME store over obstore anon, so the delta isolates insitubatch's
dedup + bounded prefetch rather than the storage backend. Earth2Studio's zarr sources
moved to obstore in #955; the feed uses ``obstore_store`` to match. Swap both to
``fsspec_store(url, token="anon")`` to re-run the comparison over gcsfs -- the de-dup
ratio is backend-independent, only the wall moves.

Two regimes:
  wb2  = 240x121 6-hourly, chunks=(8,240,121): fat time-chunk -> high dedup ratio, tiny
         fields -> wall gated by concurrency (wall speedup << read reduction).
  arco = 721x1440 1-hourly, chunks=(1,721,1440): chunk-1 -> dedup = pure valid-time overlap,
         4MB fields -> genuinely IO-bound (wall speedup tracks read reduction).
"""

import argparse
import os
import shutil
import tempfile
import time
from typing import Any

import numpy as np
from insitubatch import obstore_store

from earth2studio.data.insitu import InSituForecastFeed, decode_cf_time

# store id -> (url, before source class, inner (H,W), transpose store(lon,lat)->(lat,lon))
STORES: dict[str, dict[str, Any]] = {
    "wb2": {
        "url": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        "before": "earth2studio.data.wb2:WB2ERA5_121x240",
        "field_bytes": 240 * 121 * 4,
        "chunk_steps": 8,
        "transpose_inner": True,
        "start": 1000,  # 1959-09-20; WB2 axis begins 1959
    },
    "arco": {
        "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
        "before": "earth2studio.data.arco:ARCO",
        "field_bytes": 721 * 1440 * 4,
        "chunk_steps": 1,
        "transpose_inner": False,
        "start": 1051896,  # 2020-01-01; ARCO axis begins 1900 but data only from 1940
    },
}
VAR_MAP = {
    "t2m": "2m_temperature",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
}


def anon_store(url: str) -> Any:
    return obstore_store(url, skip_signature=True)


def load_before_cls(spec: str) -> Any:
    mod, _, name = spec.partition(":")
    import importlib

    return getattr(importlib.import_module(mod), name)


def run_after(
    cfg: dict[str, Any],
    variables: list[str],
    start: int,
    n_init: int,
    leads_h: list[int],
    batch_size: int,
    max_inflight: int,
) -> dict[str, Any]:
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
    saw_finite = False
    for x, _coords in feed:
        n_rows += x.shape[0]  # decode is forced by the gather; no touch needed
        saw_finite = saw_finite or bool(x.isfinite().any())
    wall = time.perf_counter() - t0
    feed.dataset.close()
    if not saw_finite:
        raise ValueError(
            f"every field read from {cfg['url']} was fill/NaN: the window at "
            f"start={start} is outside the store's populated range. Pass a valid --start."
        )
    return {
        "wall_s": wall,
        "init_rows": n_rows,
        "chunk_decodes": feed.dataset.cache_misses,
        "resident_peak": feed.dataset.resident_peak,
    }


BEFORE_CACHE_DIRNAME = "insitubatch_bench_e2s_cache"


def before_cache_root() -> str:
    """Bench-owned Earth2Studio cache root for the ``--before-cache`` leg.

    Deliberately not the user's real cache (``~/.cache/earth2studio`` or
    ``$EARTH2STUDIO_CACHE``): this directory is wiped before every repeat so
    each repeat measures a cold-cache campaign pass.
    """
    return os.path.join(tempfile.gettempdir(), BEFORE_CACHE_DIRNAME)


def wipe_before_cache() -> None:
    """Empty the bench-owned cache so the next repeat starts cold."""
    root = before_cache_root()
    if os.path.basename(root) != BEFORE_CACHE_DIRNAME:
        raise RuntimeError(f"refusing to wipe unexpected cache path: {root}")
    shutil.rmtree(root, ignore_errors=True)


def run_before(
    before_cls: Any,
    variables: list[str],
    init_times64: Any,
    leads_h: list[int],
    cache: bool,
) -> dict[str, Any]:
    src = before_cls(cache=cache, verbose=False)
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", choices=list(STORES), default="wb2")
    p.add_argument("--vars", nargs="+", default=["t2m"])
    p.add_argument("--start", type=int, default=None)  # default: per-store (see STORES)
    p.add_argument("--n-init", type=int, default=24)
    p.add_argument("--lead-step-h", type=int, default=6)
    p.add_argument("--max-lead-h", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--skip-before", action="store_true")
    p.add_argument(
        "--before-cache",
        action="store_true",
        help="run the BEFORE leg with Earth2Studio's per-source cache ON (its own "
        "default), so redundant reads hit local disk instead of the cloud. Decode "
        "still repeats per requested step either way. Uses a bench-owned cache "
        "directory, wiped before each repeat so every repeat is a cold-cache pass.",
    )
    args = p.parse_args()

    if args.before_cache:
        # Redirect Earth2Studio's cache to a bench-owned directory before any source
        # is constructed. DATA_CACHE takes precedence downstream, so set both.
        os.environ["EARTH2STUDIO_CACHE"] = before_cache_root()
        os.environ["EARTH2STUDIO_DATA_CACHE"] = before_cache_root()

    cfg = STORES[args.store]
    start = args.start if args.start is not None else cfg["start"]
    leads_h = list(range(args.lead_step_h, args.max_lead_h + 1, args.lead_step_h))
    requested = args.n_init * len(leads_h) * len(args.vars)

    import zarr

    g = zarr.open_group(store=anon_store(cfg["url"]), mode="r")
    attrs = dict(g["time"].attrs)
    times64 = decode_cf_time(
        np.asarray(g["time"][:]), attrs["units"], attrs.get("calendar", "standard")
    )
    init_times64 = times64[start : start + args.n_init]

    print(
        f"[{args.store}] start={start} ({init_times64[0]}) ; grid: {args.n_init} inits x {len(leads_h)} leads x {len(args.vars)} vars "
        f"= {requested} requested field-reads ({requested*cfg['field_bytes']/1e9:.2f} GB naive)"
    )
    print(
        f"leads: {args.lead_step_h}h..{args.max_lead_h}h ; vars: {args.vars} ; repeats: {args.repeats}"
    )

    def med3(w: list[float]) -> tuple[float, float, float]:
        w = sorted(w)
        return w[len(w) // 2], w[0], w[-1]

    before_cls = load_before_cls(cfg["before"])
    after_walls: list[float] = []
    before_walls: list[float] = []
    decodes: Any = None
    resident: Any = None
    for r in range(args.repeats):
        a = run_after(
            cfg,
            args.vars,
            start,
            args.n_init,
            leads_h,
            args.batch_size,
            args.max_inflight,
        )
        after_walls.append(a["wall_s"])
        decodes, resident = a["chunk_decodes"], a["resident_peak"]
        if not args.skip_before:
            if args.before_cache:
                wipe_before_cache()  # every repeat is a cold-cache pass
            before_walls.append(
                run_before(
                    before_cls, args.vars, init_times64, leads_h, args.before_cache
                )["wall_s"]
            )
        print(
            f"  repeat {r+1}/{args.repeats}: after={after_walls[-1]:.2f}s"
            + (f"  before={before_walls[-1]:.2f}s" if before_walls else "")
        )

    dedup = requested / decodes
    a_med, a_lo, a_hi = med3(after_walls)
    print("\n=== AFTER (insitubatch, obstore anon) ===")
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
        cache_note = (
            "cache ON (E2S default; cold per repeat, redundant reads hit local disk)"
            if args.before_cache
            else "cache OFF (not the E2S default; every redundant read goes to the cloud)"
        )
        print(f"\n=== BEFORE (E2S fetch, obstore anon, {cache_note}) ===")
        print(f"  wall (med/min/max): {b_med:.2f} / {b_lo:.2f} / {b_hi:.2f} s")
        print("\n=== HEADLINE (medians) ===")
        print(
            f"  speedup        : {b_med/a_med:.1f}x wall ({b_med:.2f}s -> {a_med:.2f}s)"
        )
        print(f"  read reduction : {dedup:.1f}x fewer chunk decodes")


if __name__ == "__main__":
    main()

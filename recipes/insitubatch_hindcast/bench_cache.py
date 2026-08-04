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

"""Cross-run persistent-cache benchmark for the insitubatch verification feed.

Scoring a hindcast campaign is rarely a one-shot: the *same* ERA5 verification set is read
again every time another model (or another checkpoint / hyperparameter) is scored against
it. Earth2Studio's eval recipe handles this with a ``predownload.py`` sentinel -- a separate
step that materializes a dense local copy before the run.

``InSituForecastFeed(cache_dir=...)`` replaces that: the first run decodes each shared chunk
once (the dedup win) AND persists it to local disk; a second run over the same store reads
those chunks back as ``cache_hits`` instead of re-fetching the cloud -- no predownload step,
no reshard, and only the chunks actually touched. Because reanalysis is static the cache
never goes stale.

This measures the second-run win: same verification window, run COLD (empty cache) then WARM
(cache populated), over obstore anon.
"""

import argparse
import os
import shutil
import time
from typing import Any

import numpy as np
from insitubatch import obstore_store

from earth2studio.data.insitu import InSituForecastFeed

STORES: dict[str, dict[str, Any]] = {
    "wb2": {
        "url": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        "transpose_inner": True,
        "start": 1000,  # 1959-09-20; WB2 axis begins 1959
    },
    "arco": {
        "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
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


def run(
    cfg: dict[str, Any],
    variables: list[str],
    start: int,
    n_init: int,
    leads_h: list[int],
    batch_size: int,
    max_inflight: int,
    cache_dir: str,
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
        cache_dir=cache_dir,
        transpose_inner=cfg["transpose_inner"],
    )
    t0 = time.perf_counter()
    saw_finite = False
    for x, _coords in feed:
        saw_finite = saw_finite or bool(x.isfinite().any())
    wall = time.perf_counter() - t0
    hits, misses = feed.dataset.cache_hits, feed.dataset.cache_misses
    feed.dataset.close()
    if not saw_finite:
        raise ValueError(
            f"every field read from {cfg['url']} was fill/NaN: the window at "
            f"start={start} is outside the store's populated range. Pass a valid --start."
        )
    return {"wall_s": wall, "hits": hits, "misses": misses}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--store", choices=list(STORES), default="wb2")
    p.add_argument("--vars", nargs="+", default=["t2m", "u10m", "v10m"])
    p.add_argument("--start", type=int, default=None)  # default: per-store (see STORES)
    p.add_argument("--n-init", type=int, default=48)
    p.add_argument("--lead-step-h", type=int, default=6)
    p.add_argument("--max-lead-h", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--cache-dir", default="/tmp/insitu_cache_bench")  # noqa: S108
    args = p.parse_args()

    cfg = STORES[args.store]
    start = args.start if args.start is not None else cfg["start"]
    leads_h = list(range(args.lead_step_h, args.max_lead_h + 1, args.lead_step_h))
    requested = args.n_init * len(leads_h) * len(args.vars)
    # Only ever wipe a subdirectory this benchmark owns -- --cache-dir may be a real
    # cache root, and the cold leg requires starting empty.
    cache_dir = os.path.join(args.cache_dir, "bench_cold_warm", args.store)
    shutil.rmtree(cache_dir, ignore_errors=True)  # start cold

    print(
        f"[{args.store}] start={start} ; {args.n_init} inits x {len(leads_h)} leads x {len(args.vars)} vars "
        f"= {requested} requested field-reads ; cache_dir={cache_dir}"
    )

    common = (
        cfg,
        args.vars,
        start,
        args.n_init,
        leads_h,
        args.batch_size,
        args.max_inflight,
    )
    cold = run(*common, cache_dir)
    warm = run(*common, cache_dir)

    print("\n=== COLD (empty cache: fetch + decode + persist) ===")
    print(
        f"  wall: {cold['wall_s']:.2f} s ; misses={cold['misses']} hits={cold['hits']}"
    )
    print("\n=== WARM (cache populated: local-disk hits) ===")
    print(
        f"  wall: {warm['wall_s']:.2f} s ; misses={warm['misses']} hits={warm['hits']}"
    )
    print("\n=== HEADLINE ===")
    print(
        f"  cross-run speedup : {cold['wall_s'] / warm['wall_s']:.1f}x ({cold['wall_s']:.2f}s -> {warm['wall_s']:.2f}s)"
    )
    print(f"  cloud fetches     : {cold['misses']} cold -> {warm['misses']} warm")


if __name__ == "__main__":
    main()

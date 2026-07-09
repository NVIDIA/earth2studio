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
(cache populated), over gcsfs anon.
"""

import argparse
import shutil
import time

import numpy as np
from insitubatch import fsspec_store

from earth2studio.data.insitu import InSituForecastFeed

STORES = {
    "wb2": {
        "url": "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr",
        "transpose_inner": True,
    },
    "arco": {
        "url": "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
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


def run(cfg, variables, start, n_init, leads_h, batch_size, max_inflight, cache_dir):
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
    for _x, _coords in feed:
        pass
    wall = time.perf_counter() - t0
    hits, misses = feed.dataset.cache_hits, feed.dataset.cache_misses
    feed.dataset.close()
    return {"wall_s": wall, "hits": hits, "misses": misses}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--store", choices=list(STORES), default="wb2")
    p.add_argument("--vars", nargs="+", default=["t2m", "u10m", "v10m"])
    p.add_argument("--start", type=int, default=1000)
    p.add_argument("--n-init", type=int, default=48)
    p.add_argument("--lead-step-h", type=int, default=6)
    p.add_argument("--max-lead-h", type=int, default=240)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-inflight", type=int, default=32)
    p.add_argument("--cache-dir", default="/tmp/insitu_cache_bench")  # noqa: S108
    args = p.parse_args()

    cfg = STORES[args.store]
    leads_h = list(range(args.lead_step_h, args.max_lead_h + 1, args.lead_step_h))
    requested = args.n_init * len(leads_h) * len(args.vars)
    shutil.rmtree(args.cache_dir, ignore_errors=True)  # start cold

    print(
        f"[{args.store}] {args.n_init} inits x {len(leads_h)} leads x {len(args.vars)} vars "
        f"= {requested} requested field-reads ; cache_dir={args.cache_dir}"
    )

    common = (
        cfg,
        args.vars,
        args.start,
        args.n_init,
        leads_h,
        args.batch_size,
        args.max_inflight,
    )
    cold = run(*common, args.cache_dir)
    warm = run(*common, args.cache_dir)

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

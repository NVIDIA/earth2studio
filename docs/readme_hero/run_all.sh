#!/usr/bin/env bash
# Regenerate the README hero image (hero.png, hero_2700.png).
#
# Usage (from the repo root):
#   export EUMETSAT_CONSUMER_KEY=... EUMETSAT_CONSUMER_SECRET=...   # fetch_meteosat.py only
#   export E2S_HERO_DIR=hero_out                                    # optional, default ./hero_out
#   bash docs/readme_hero/run_all.sh [fetch|plot|compose]           # default: all stages
#
# Requirements:
#   * An environment with earth2studio and the extras for SFNO, StormScope
#     (GOES/MRMS/Meteosat), earth2bufr (NNJA obs), plus matplotlib, cartopy,
#     scipy and pillow. A CUDA GPU is needed for fetch_forecast/storm/meteosat.
#   * Network access (GFS, ERA5 ARCO, NNJA, GOES, Himawari on public buckets,
#     model weights from Hugging Face, MTG FCI from the EUMETSAT Data Store).
#   * The EUMETSAT Data Store only keeps a rolling window of recent data, so
#     update the init time in fetch_meteosat.py (and the label in compose.py)
#     if the default time can no longer be downloaded.
#
# Stages: fetch_* download data / run models and write .npz files, plot_*
# render one black-background panel each, compose.py assembles the final image.
# Large downloads: GOES full disk ~1.3 GB, Himawari ~0.9 GB, MTG ~2-3 min/frame.
set -euo pipefail
ROOT="$PWD"
cd "$(dirname "$0")"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export E2S_HERO_DIR="${E2S_HERO_DIR:-$ROOT/hero_out}"
stage="${1:-all}"

if [[ $stage == all || $stage == fetch ]]; then
  for s in obs forecast storm goes himawari gfs meteosat era5 jet; do
    python "fetch_${s}.py"
  done
fi
if [[ $stage == all || $stage == plot ]]; then
  for s in globe obs storm storm_rgb meteosat satellite era5 jet; do
    python "plot_${s}.py"
  done
fi
if [[ $stage == all || $stage == compose ]]; then
  python compose.py
fi

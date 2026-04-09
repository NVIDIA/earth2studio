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

"""Validation script: compare JPSS_CRIS (raw L1 from NOAA S3) with CrIS FSR
observations stored in the UFS GEFS-v13 replay archive (UFSObsSat).

The two data sources use different file formats (HDF5 vs. NetCDF), different
variable layouts and possibly different preprocessing.  This script fetches a
short time window from both sources for the same satellite and plots
distribution-level statistics (histogram, scatter by channel) to verify that
the JPSS_CRIS wrapper is reading physically plausible radiance values and
that the observation magnitudes are consistent.

Usage
-----
    python scripts/validate_jpss_cris.py [--time 2024-06-01T12:00] [--sat n20]

Requires the ``data`` extras: ``pip install earth2studio[data]``
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from earth2studio.data import JPSS_CRIS
from earth2studio.data.ufs import UFSObsSat


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--time",
        type=str,
        default="2024-06-01T12:00",
        help="ISO timestamp for the comparison window (UTC)",
    )
    parser.add_argument(
        "--sat",
        type=str,
        default="n20",
        choices=["n20", "n21", "npp"],
        help="JPSS satellite short-name",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=30,
        help="Time tolerance in minutes",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="validate_cris.png",
        help="Output figure path",
    )
    return parser.parse_args()


def fetch_jpss_cris(t: datetime, sat: str, tol_min: int) -> pd.DataFrame:
    """Fetch CrIS L1 spectral radiance from NOAA S3."""
    logger.info(f"Fetching JPSS_CRIS for {sat} at {t} ±{tol_min}min")
    ds = JPSS_CRIS(
        satellites=[sat],
        time_tolerance=timedelta(minutes=tol_min),
        cache=True,
        verbose=True,
    )
    return ds(t, ["crisfsr"])


def fetch_ufs_cris(t: datetime, tol_min: int) -> pd.DataFrame:
    """Fetch CrIS FSR observations from the UFS replay archive."""
    logger.info(f"Fetching UFSObsSat (crisfsr) at {t} ±{tol_min}min")
    ds = UFSObsSat(
        time_tolerance=timedelta(minutes=tol_min),
        cache=True,
        verbose=True,
    )
    return ds(t, ["crisfsr"])


def _summarise(df: pd.DataFrame, label: str) -> None:
    """Print a quick statistical summary."""
    if df.empty:
        logger.warning(f"{label}: empty DataFrame")
        return
    obs = df["observation"]
    logger.info(
        f"{label}: {len(df)} rows, "
        f"obs min={obs.min():.4f}, max={obs.max():.4f}, "
        f"mean={obs.mean():.4f}, std={obs.std():.4f}"
    )
    if "channel_index" in df.columns:
        ch = df["channel_index"]
        logger.info(
            f"  channel_index range: [{ch.min()}, {ch.max()}], unique: {ch.nunique()}"
        )


def plot_comparison(
    df_jpss: pd.DataFrame,
    df_ufs: pd.DataFrame,
    out_path: str,
) -> None:
    """Create a comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: histogram of all observations
    ax = axes[0, 0]
    if not df_jpss.empty:
        ax.hist(
            df_jpss["observation"].values,
            bins=100,
            alpha=0.6,
            label="JPSS_CRIS (L1)",
            density=True,
        )
    if not df_ufs.empty:
        ax.hist(
            df_ufs["observation"].values,
            bins=100,
            alpha=0.6,
            label="UFS CrIS-FSR",
            density=True,
        )
    ax.set_xlabel("Radiance / Observation value")
    ax.set_ylabel("Density")
    ax.set_title("Observation distribution")
    ax.legend()

    # Panel 2: mean observation per channel
    ax = axes[0, 1]
    if not df_jpss.empty and "channel_index" in df_jpss.columns:
        ch_mean = df_jpss.groupby("channel_index")["observation"].mean()
        ax.plot(ch_mean.index, ch_mean.values, ".", markersize=1, label="JPSS_CRIS")
    if not df_ufs.empty and "channel_index" in df_ufs.columns:
        ch_mean_ufs = df_ufs.groupby("channel_index")["observation"].mean()
        ax.plot(ch_mean_ufs.index, ch_mean_ufs.values, ".", markersize=1, label="UFS")
    ax.set_xlabel("Channel index")
    ax.set_ylabel("Mean observation")
    ax.set_title("Mean observation per channel")
    ax.legend()

    # Panel 3: spatial scatter for JPSS_CRIS (first 10k points)
    ax = axes[1, 0]
    if not df_jpss.empty:
        subset = df_jpss.head(10000)
        sc = ax.scatter(
            subset["lon"],
            subset["lat"],
            c=subset["observation"],
            s=0.5,
            cmap="viridis",
        )
        plt.colorbar(sc, ax=ax, label="Radiance")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("JPSS_CRIS spatial coverage (first 10k)")

    # Panel 4: spatial scatter for UFS CrIS-FSR
    ax = axes[1, 1]
    if not df_ufs.empty:
        subset = df_ufs.head(10000)
        sc = ax.scatter(
            subset["lon"],
            subset["lat"],
            c=subset["observation"],
            s=0.5,
            cmap="viridis",
        )
        plt.colorbar(sc, ax=ax, label="Observation")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("UFS CrIS-FSR spatial coverage (first 10k)")

    fig.suptitle("JPSS_CRIS vs UFS CrIS-FSR Validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info(f"Saved comparison figure to {out_path}")
    plt.close(fig)


def main() -> None:
    """Run JPSS_CRIS vs UFS CrIS-FSR validation."""
    args = _parse_args()
    t = datetime.fromisoformat(args.time)

    df_jpss = fetch_jpss_cris(t, args.sat, args.tolerance)
    _summarise(df_jpss, "JPSS_CRIS")

    df_ufs = fetch_ufs_cris(t, args.tolerance)
    _summarise(df_ufs, "UFS CrIS-FSR")

    if df_jpss.empty and df_ufs.empty:
        logger.error("Both sources returned empty data. Nothing to compare.")
        return

    plot_comparison(df_jpss, df_ufs, args.out)

    # Quick sanity checks
    if not df_jpss.empty:
        obs = df_jpss["observation"]
        assert (obs > 0).all(), "Negative radiance detected in JPSS_CRIS"  # noqa: S101
        assert (  # noqa: S101
            obs < 500
        ).all(), "Unreasonably large radiance in JPSS_CRIS"
        logger.info("JPSS_CRIS sanity checks PASSED")

    if not df_ufs.empty and not df_jpss.empty:
        jpss_mean = df_jpss["observation"].mean()
        ufs_mean = df_ufs["observation"].mean()
        ratio = jpss_mean / ufs_mean if ufs_mean > 0 else float("inf")
        logger.info(
            f"Mean ratio JPSS/UFS: {ratio:.4f} "
            f"(JPSS={jpss_mean:.4f}, UFS={ufs_mean:.4f})"
        )
        if 0.01 < ratio < 100:
            logger.info("Mean ratio is within plausible range")
        else:
            logger.warning(
                "Mean ratio is out of expected range — units or calibration may differ"
            )


if __name__ == "__main__":
    main()

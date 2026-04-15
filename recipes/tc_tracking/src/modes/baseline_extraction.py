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

import os
import urllib.request

import numpy as np
import pandas as pd
import tropycal.tracks as tropytracks
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

from earth2studio.data import fetch_data
from src.data.tc_hunt_data_utils import DataSourceManager, load_heights
from src.tc_hunt_utils import great_circle_distance, run_with_rank_ordered_execution
from src.tempest_extremes import TempestExtremes

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)


def ensure_ibtracs(path: str) -> str:
    """Ensure the IBTrACS CSV exists at *path*, downloading it if necessary.

    The full IBTrACS v04r01 dataset (~300 MB) is downloaded from NCEI on
    first use and cached at the specified location for subsequent runs.

    Parameters
    ----------
    path : str
        Local file path where the IBTrACS CSV should reside.

    Returns
    -------
    str
        Absolute path to the IBTrACS CSV file.
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(
        f"IBTrACS file not found at {path} — downloading from NCEI (~300 MB)..."
    )
    urllib.request.urlretrieve(IBTRACS_URL, path)  # noqa: S310
    logger.success(f"IBTrACS downloaded to {path}")
    return path


def extract_from_historic_data(
    cfg: DictConfig,
    ic: np.datetime64,
    n_steps: int,
    time_step: np.timedelta64,
    vars: list[str],
    data_source_mngr: DataSourceManager,
) -> pd.DataFrame:
    """Fetch reanalysis data and extract TC tracks with TempestExtremes.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    ic : np.datetime64
        Initial condition (start time).
    n_steps : int
        Number of time steps to extract.
    time_step : np.timedelta64
        Spacing between time steps.
    vars : list[str]
        Variable names required by TempestExtremes.
    data_source_mngr : DataSourceManager
        Manager providing the correct data source for the time range.

    Returns
    -------
    pd.DataFrame
        Extracted TC tracks from the reanalysis data.
    """
    times = np.arange(np.timedelta64(0, "h"), n_steps * time_step, time_step)
    data_source = data_source_mngr.select_data_source(ic + times)

    xx, coords = fetch_data(
        data_source,
        time=[ic],
        lead_time=times,
        variable=vars,
        device="cpu",
    )
    xx = xx.unsqueeze(0)
    coords["ensemble"] = np.array([0])
    coords.move_to_end("ensemble", last=False)

    heights, height_coords = (
        load_heights(cfg.cyclone_tracking.orography_path)
        if "orography_path" in cfg.cyclone_tracking
        else (None, None)
    )

    cyclone_tracking = TempestExtremes(
        detect_cmd=cfg.cyclone_tracking.detect_cmd,
        stitch_cmd=cfg.cyclone_tracking.stitch_cmd,
        input_vars=cfg.cyclone_tracking.vars,
        batch_size=1,
        n_steps=n_steps - 1,
        time_step=time_step,
        lats=coords["lat"],
        lons=coords["lon"],
        static_vars=heights,
        static_coords=height_coords,
        store_dir=cfg.store_dir,
        keep_raw_data=cfg.cyclone_tracking.get("keep_raw_data", False),
        print_te_output=cfg.cyclone_tracking.get("print_te_output", False),
        scratch_dir=cfg.cyclone_tracking.get("scratch_dir", None),
    )
    cyclone_tracking.record_state(xx, coords)
    cyclone_tracking()

    hist_tracks = pd.read_csv(cyclone_tracking.track_files[0])
    hist_tracks.columns = hist_tracks.columns.str.strip()
    os.remove(cyclone_tracking.track_files[0])

    return hist_tracks


def extract_from_ibtracs(
    cfg: DictConfig,
    ibtracs: tropytracks.TrackDataset,
    case: str,
    time_step: np.timedelta64,
) -> tuple[pd.DataFrame, np.datetime64, int]:
    """Query IBTrACS for a named storm and derive the extraction window.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object containing ``cases.<case>.year`` and
        ``cases.<case>.basin``.
    ibtracs : tropytracks.TrackDataset
        Pre-loaded IBTrACS dataset.
    case : str
        Storm name (must match an IBTrACS entry).
    time_step : np.timedelta64
        Time step for extraction (typically 6 h).

    Returns
    -------
    tuple[pd.DataFrame, np.datetime64, int]
        IBTrACS storm data filtered to the extraction window, the
        initial condition time (rounded down to the nearest 6 h), and
        the number of time steps to extract.
    """
    logger.info(
        f"extracting storm {case} ({cfg.cases[case].year}, "
        f"{cfg.cases[case].basin}) from IBTrACS"
    )
    storm = ibtracs.get_storm((case, cfg.cases[case].year))
    storm.time = storm.time.astype(np.datetime64)

    ib_storm = pd.DataFrame(
        {
            "time": storm.time,
            "year": storm.time.astype("datetime64[Y]").astype(int) + 1970,
            "month": storm.time.astype("datetime64[M]").astype(int) % 12 + 1,
            "day": np.array([int(pd.Timestamp(tt).day) for tt in storm.time]),
            "hour": storm.time.astype("datetime64[h]").astype(int) % 24,
            "lat": storm.lat,
            "lon": storm.lon,
            "wind_speed": storm.vmax,
            "mslp": storm.mslp,
            "type": storm.type,
        }
    )

    ic = storm.time[0]

    # Ensure ic is at 00h, 06h, 12h, or 18h
    ic_datetime = pd.to_datetime(ic)
    hour = ic_datetime.hour
    if hour % 6 != 0:
        hours_to_subtract = hour % 6
        ic = ic - np.timedelta64(hours_to_subtract, "h")
        ic = ic.astype("datetime64[h]")

    n_steps = (storm.time[-1] - ic) // time_step
    if (storm.time[-1] - ic) % time_step != 0:
        n_steps += 1

    ib_storm = ib_storm[
        ib_storm["time"].isin(np.arange(ic, ic + n_steps * time_step, time_step))
    ]

    return ib_storm, ic, int(n_steps)


def match_tracks(
    ib_storm: pd.DataFrame, hist_tracks: pd.DataFrame, case: str
) -> pd.DataFrame:
    """Match extracted tracks against IBTrACS to identify the target storm.

    Iterates over IBTrACS observation times and compares against each
    detected track. A match is declared when a track point is within
    300 km of the IBTrACS position.

    Parameters
    ----------
    ib_storm : pd.DataFrame
        IBTrACS storm observations.
    hist_tracks : pd.DataFrame
        All tracks extracted by TempestExtremes.
    case : str
        Storm name (for logging).

    Returns
    -------
    pd.DataFrame
        The matched track, or an empty DataFrame if no match is found.
    """
    times = np.array(
        [
            pd.to_datetime(
                f"{hist_tracks['year'].iloc[jj]}-"
                f"{int(hist_tracks.iloc[jj]['month']):02d}-"
                f"{int(hist_tracks.iloc[jj]['day']):02d} "
                f"{int(hist_tracks.iloc[jj]['hour']):02d}:00:00"
            )
            for jj in range(len(hist_tracks))
        ]
    )
    hist_tracks.insert(0, "time", times)

    n_tracks = hist_tracks["track_id"].nunique()

    matched_track = None
    for time in ib_storm["time"]:
        lat_ib = ib_storm.loc[ib_storm["time"] == time, "lat"].item()
        lon_ib = ib_storm.loc[ib_storm["time"] == time, "lon"].item()

        for track_id in range(n_tracks):
            track = hist_tracks[hist_tracks["track_id"] == track_id]
            if (track["time"] == time).any():
                lat_track = track.loc[track["time"] == time, "lat"].item()
                lon_track = track.loc[track["time"] == time, "lon"].item()
                dist = great_circle_distance(lat_ib, lon_ib, lat_track, lon_track)
                if dist < 300000:
                    matched_track = track_id
                    logger.info(
                        f"matched track {matched_track} for storm {case} "
                        f"with distance {dist / 1000:.2f} km"
                    )
                    break

        if matched_track is not None:
            break

    return hist_tracks[hist_tracks["track_id"] == matched_track].reset_index(drop=True)


def add_ibtracs_data(
    matched_track: pd.DataFrame, ib_storm: pd.DataFrame
) -> pd.DataFrame:
    """Merge IBTrACS observations into the matched track.

    Parameters
    ----------
    matched_track : pd.DataFrame
        Track extracted from reanalysis data.
    ib_storm : pd.DataFrame
        IBTrACS storm observations.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with both reanalysis and IBTrACS fields.
    """
    matched_track = pd.merge(
        matched_track, ib_storm, on="time", how="right", suffixes=("", "_ib")
    ).reset_index(drop=True)
    matched_track.rename(columns={"mslp": "msl_ib"}, inplace=True)

    datetime_cols = ["year", "month", "day", "hour"]
    for col in datetime_cols:
        if col in matched_track.columns:
            matched_track[col] = matched_track[col].astype("Int64")

    matched_track["track_id"] = 0

    matched_track.loc[matched_track["lon_ib"] < 0, "lon_ib"] = (
        360 + matched_track.loc[matched_track["lon_ib"] < 0, "lon_ib"]
    )

    return matched_track


def write_track_to_csv(
    matched_track: pd.DataFrame, case: str, store_dir: str, basin: str
) -> None:
    """Write the merged track to a CSV reference file.

    Parameters
    ----------
    matched_track : pd.DataFrame
        Merged track data.
    case : str
        Storm name.
    store_dir : str
        Output directory.
    basin : str
        Ocean basin identifier.
    """
    matched_track.drop(columns=["i", "j", "time"], inplace=True)

    matched_track = matched_track[
        [
            "track_id",
            "year_ib",
            "month_ib",
            "day_ib",
            "hour_ib",
            "lon",
            "lat",
            "msl",
            "wind_speed",
            "lon_ib",
            "lat_ib",
            "msl_ib",
            "wind_speed_ib",
            "type",
        ]
    ]

    matched_track.rename(
        columns={
            "year_ib": "year",
            "month_ib": "month",
            "day_ib": "day",
            "hour_ib": "hour",
        },
        inplace=True,
    )

    # IBTrACS reports MSLP in hPa; convert to Pa for consistency
    matched_track["msl_ib"] = matched_track["msl_ib"] * 100

    csv_name = f"reference_track_{case}_{matched_track['year'].iloc[0]}_{basin}.csv"
    matched_track.to_csv(os.path.join(store_dir, csv_name), index=False)


def extract_baseline(
    cfg: DictConfig,
    time_step: np.timedelta64 = np.timedelta64(6, "h"),
    vars: list[str] | None = None,
) -> None:
    """Extract TC reference tracks from reanalysis using IBTrACS ground truth.

    For each case defined in the configuration, queries IBTrACS to determine
    when the storm was active, fetches the corresponding reanalysis data,
    runs TempestExtremes to extract all TC tracks, and matches the result
    against IBTrACS to identify the target storm.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object.
    time_step : np.timedelta64, optional
        Time step for the reanalysis extraction, by default 6 h.
    vars : list[str] | None, optional
        Variables required by TempestExtremes. Defaults to
        ``['msl', 'z300', 'z500', 'u10m', 'v10m']``.
    """
    if vars is None:
        vars = ["msl", "z300", "z500", "u10m", "v10m"]

    DistributedManager.initialize()

    ibtracs_path = run_with_rank_ordered_execution(
        ensure_ibtracs, cfg.ibtracs_source_data
    )

    ibtracs = tropytracks.TrackDataset(
        basin="all",
        source="ibtracs",
        ibtracs_mode="jtwc_neumann",
        ibtracs_url=ibtracs_path,
    )

    data_source_mngr = DataSourceManager(cfg)

    for case in cfg.cases:
        ib_storm, ic, n_steps = extract_from_ibtracs(cfg, ibtracs, case, time_step)

        hist_tracks = extract_from_historic_data(
            cfg=cfg,
            ic=ic,
            n_steps=n_steps,
            time_step=time_step,
            vars=vars,
            data_source_mngr=data_source_mngr,
        )

        matched_track = match_tracks(ib_storm, hist_tracks, case)

        matched_track = add_ibtracs_data(matched_track, ib_storm)

        write_track_to_csv(matched_track, case, cfg.store_dir, cfg.cases[case].basin)

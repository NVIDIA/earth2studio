# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cyclone tracks of WeatherNext 2 Cyclones' built-in tracker.

WeatherNext 2 Cyclones (WN2) predicts cyclones directly: next to the atmospheric
fields, the network outputs gridded cyclone fields learnt from IBTrACS, which
its own tracker turns into a track table while the model rolls out. No fields
have to be recorded for it, unlike for TempestExtremes (TE).

:class:`WN2TrackWriter` writes that table once a member is done, in the layout
of TE's track files, so both trackers can be read, matched and plotted by the
same code:

==============  ==============================================================
Column          Content
==============  ==============================================================
``track_id``    Index of the track within the file, starting at 0
``year`` ...    Valid time, as ``year``, ``month``, ``day``, ``hour``
``i``, ``j``    Longitude and latitude index of the nearest 0.25 degree point
``lon``         Longitude of the centre in degrees east, 0 to 360
``lat``         Latitude of the centre in degrees north
``msl``         Minimum sea-level pressure in Pa (TE: grid-point minimum)
``wind_speed``  Maximum 1-minute sustained 10 m wind in m/s (TE: maximum
                instantaneous grid-point wind)
``height``      Not predicted, always NaN (TE: terrain height)
further         Storm size, see :data:`SIZE_COLUMNS`; not part of TE's files
==============  ==============================================================
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.tc_hunt_utils import track_file_name

if TYPE_CHECKING:
    from earth2studio.run import WorkItem

# name of the model in the config, and of its tracker's output directory
WN2_MODEL = "wn2"
WN2_TRACK_DIR = "cyclone_tracks_wn2"

# columns of a TempestExtremes track file, in order, and how StitchNodes
# formats them: integers, positions with six decimals, values in scientific
# notation
TE_COLUMNS = {
    "track_id": "%d",
    "year": "%d",
    "month": "%d",
    "day": "%d",
    "hour": "%d",
    "i": "%d",
    "j": "%d",
    "lon": "%.6f",
    "lat": "%.6f",
    "msl": "%e",
    "wind_speed": "%e",
    "height": "%e",
}

# WN2 reports wind radii for the thresholds 34, 50 and 64 kt; the track file
# names them by the threshold in m/s: 17.5, 25.7 and 32.9 m/s
RADIUS_THRESHOLDS = {"34": "17", "50": "26", "64": "33"}

# storm size columns of the WN2 track table, and their names in the track file
SIZE_COLUMNS = {
    "radius_of_maximum_winds_km": "rmw_km",
    **{
        f"radius_{native}_knot_winds_{quadrant}_km": f"r{ms}_{quadrant}_km"
        for native, ms in RADIUS_THRESHOLDS.items()
        for quadrant in ("ne", "se", "sw", "nw")
    },
}

GRID_SPACING_DEG = 0.25


def validate_wn2_tracking(cfg: DictConfig) -> None:
    """Check that the run can use WN2's built-in tracker.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration object

    Raises
    ------
    ValueError
        If ``wn2_tracking`` is configured for another model than ``wn2``, or
        for more than one member per batch
    """
    if "wn2_tracking" not in cfg:
        return
    if cfg.model != WN2_MODEL:
        raise ValueError(
            f"wn2_tracking is the tracker built into WeatherNext 2 Cyclones and "
            f"needs model '{WN2_MODEL}', got model '{cfg.model}'. Other models "
            "do not predict the cyclone fields it reads: track them with "
            "TempestExtremes (cyclone_tracking) instead."
        )
    if cfg.batch_size != 1:
        raise ValueError(
            "wn2_tracking needs batch_size 1, as the WN2 tracker handles one "
            f"member per model run, got batch_size {cfg.batch_size}"
        )


def wn2_tracks_to_te_frame(tracks: pd.DataFrame) -> pd.DataFrame:
    """Convert a WN2 track table to the layout of a TempestExtremes track file.

    Parameters
    ----------
    tracks : pd.DataFrame
        Track table of one member, as returned by
        ``WeatherNext2Cyclones.cyclone_tracks``. Needs ``track_id``,
        ``valid_time``, ``lat``, ``lon`` and the wrapper's metric columns
        ``tcmsl`` (Pa) and ``tcw10m`` (m/s); may be empty

    Returns
    -------
    pd.DataFrame
        One row per track point with the columns of :data:`TE_COLUMNS`,
        followed by those of :data:`SIZE_COLUMNS` the table provides. Tracks
        are ordered by their first valid time and numbered from 0, their
        points ordered by time
    """
    size_columns = {
        native: name for native, name in SIZE_COLUMNS.items() if native in tracks
    }
    columns = [*TE_COLUMNS, *size_columns.values()]
    if tracks.empty:
        return pd.DataFrame(columns=columns)

    tracks = tracks.reset_index(drop=True)
    time = pd.to_datetime(tracks["valid_time"])
    lon = tracks["lon"].to_numpy(dtype=float) % 360.0
    lat = tracks["lat"].to_numpy(dtype=float)
    n_lon = round(360.0 / GRID_SPACING_DEG)
    frame = pd.DataFrame(
        {
            "track_id": _renumber_by_first_time(tracks["track_id"], time),
            "year": time.dt.year,
            "month": time.dt.month,
            "day": time.dt.day,
            "hour": time.dt.hour,
            "i": np.rint(lon / GRID_SPACING_DEG).astype(int) % n_lon,
            "j": np.rint((90.0 - lat) / GRID_SPACING_DEG).astype(int),
            "lon": lon,
            "lat": lat,
            "msl": tracks["tcmsl"].to_numpy(dtype=float),
            "wind_speed": tracks["tcw10m"].to_numpy(dtype=float),
            "height": np.nan,
            **{name: tracks[native] for native, name in size_columns.items()},
            "_time": time,
        }
    )
    frame = frame.sort_values(["track_id", "_time"], kind="stable")
    return frame[columns].reset_index(drop=True)


def _renumber_by_first_time(track_id: pd.Series, time: pd.Series) -> np.ndarray:
    """Number the tracks from 0, ordered by first valid time, then by native id."""
    first = time.groupby(track_id.to_numpy()).min()
    order = sorted(first.index, key=lambda native: (first[native], str(native)))
    number = {native: n for n, native in enumerate(order)}
    return track_id.map(number).to_numpy()


def write_te_style_csv(
    frame: pd.DataFrame, path: str, header: Mapping[str, Any] | None = None
) -> None:
    """Write tracks as a TempestExtremes-style CSV, with an optional comment header.

    Formatted as StitchNodes does, see :data:`TE_COLUMNS`; further columns in
    scientific notation.

    Parameters
    ----------
    frame : pd.DataFrame
        Tracks, see :func:`wn2_tracks_to_te_frame`
    path : str
        File to write
    header : Mapping[str, Any] | None, optional
        Written as ``# key: value`` lines above the column names, by default
        None. Readers skip them with ``pd.read_csv(..., comment="#")``
    """
    comments = "".join(f"# {key}: {value}\n" for key, value in (header or {}).items())
    np.savetxt(
        path,
        frame.to_numpy(dtype=float),
        fmt=[TE_COLUMNS.get(column, "%e") for column in frame.columns],
        delimiter=", ",
        header=comments + ", ".join(frame.columns),
        comments="",
        encoding="utf-8",
    )


class WN2TrackWriter:
    """Write the tracks of WN2's built-in tracker, one file per member.

    A :class:`~src.ensemble.runner.BatchObserver` of the runner: the model
    tracks cyclones while it rolls out, so once a batch (one member) is done,
    its track table is complete and written to
    ``<store_dir>/cyclone_tracks_wn2/``, under the name TempestExtremes uses
    for the member in ``cyclone_tracks_te/``.

    Parameters
    ----------
    model : Any
        ``WeatherNext2Cyclones`` loaded with ``track_cyclones=True``
    store_dir : str
        Output directory of the run
    batch_size : int
        Batch size of the run, part of the track file names
    header : Mapping[str, Any] | None, optional
        Run description for the header of every file, e.g. model and
        checkpoint, by default None

    Raises
    ------
    ValueError
        If the model does not track cyclones
    """

    def __init__(
        self,
        model: Any,
        store_dir: str,
        batch_size: int,
        header: Mapping[str, Any] | None = None,
    ) -> None:
        if not getattr(model, "track_cyclones", False):
            raise ValueError("the model has to be loaded with track_cyclones=True")
        self.model = model
        self.batch_size = batch_size
        self.out_dir = os.path.join(store_dir, WN2_TRACK_DIR)
        self.header = {
            **(header or {}),
            "tracker": "WeatherNext 2 Cyclones built-in tracker",
            "wind_speed": "maximum 1-minute sustained 10 m wind [m/s]",
            "msl": "minimum sea-level pressure [Pa]",
        }
        os.makedirs(self.out_dir, exist_ok=True)

    @classmethod
    def from_config(cls, cfg: DictConfig, model: Any) -> WN2TrackWriter:
        """Writer of a run, with the model and its settings in the file header.

        Parameters
        ----------
        cfg : DictConfig
            Hydra configuration object of the run
        model : Any
            The run's ``WeatherNext2Cyclones``, loaded with
            ``track_cyclones=True``

        Returns
        -------
        WN2TrackWriter
            Writer to ``<store_dir>/cyclone_tracks_wn2/``
        """
        model_kwargs = (
            OmegaConf.to_container(cfg.model_kwargs, resolve=True)
            if "model_kwargs" in cfg
            else {}
        )
        header = {"model": cfg.model, **model_kwargs}
        return cls(model, cfg.store_dir, cfg.batch_size, header)

    def attempt_started(self, item: WorkItem, attempt: int, seed: int) -> None:
        """Nothing to discard: the model resets its tracks with every rollout."""

    def batch_finished(self, item: WorkItem, seed: int | None, attempts: int) -> None:
        """Write the member's tracks, unless the batch failed.

        Parameters
        ----------
        item : WorkItem
            The batch, of one member
        seed : int | None
            Seed of the stable attempt, None if the batch failed
        attempts : int
            Number of attempts made
        """
        if seed is None:
            return
        (member,) = item.member_ids
        frame = wn2_tracks_to_te_frame(self.model.cyclone_tracks)
        if frame.empty:
            logger.warning(
                f"WN2 tracker found no cyclone in member {member} of {item.time}"
            )
        name = track_file_name(item.time, member, seed, self.batch_size)
        write_te_style_csv(frame, os.path.join(self.out_dir, name), self.header)

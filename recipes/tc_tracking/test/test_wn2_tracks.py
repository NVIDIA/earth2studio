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

"""Tests of the WN2 track files, which need neither the model nor a GPU.

Run from the recipe directory: ``pytest test/test_wn2_tracks.py``.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.plotting.data_handling import (  # noqa: E402
    extract_tracks,
    extract_tracks_from_file,
)
from src.ensemble.wn2_tracking import (  # noqa: E402
    TE_COLUMNS,
    WN2_TRACK_DIR,
    WN2TrackWriter,
    validate_wn2_tracking,
    wn2_tracks_to_te_frame,
    write_te_style_csv,
)
from src.tc_hunt_utils import parse_track_file_name, track_file_name  # noqa: E402

IC = np.datetime64("2025-10-24T00:00:00")
KNOTS_TO_MS = 0.514444


def wn2_table() -> pd.DataFrame:
    """Track table as the model returns it: two storms, rows not ordered."""
    late = pd.DataFrame(
        {
            "track_id": "b",
            "valid_time": pd.date_range("2025-10-25 00:00", periods=3, freq="6h"),
            "lat": [10.1, 10.6, 11.2],
            "lon": [-40.0, -41.1, -42.3],
            "minimum_sea_level_pressure_hpa": [1002.0, 1000.0, 998.0],
            "maximum_sustained_wind_speed_knots": [30.0, 35.0, 40.0],
            "radius_of_maximum_winds_km": [80.0, 70.0, 60.0],
            "radius_34_knot_winds_ne_km": [150.0, 160.0, 170.0],
        }
    )
    early = pd.DataFrame(
        {
            "track_id": "a",
            "valid_time": pd.date_range("2025-10-24 06:00", periods=2, freq="6h"),
            "lat": [15.8, 15.9],
            "lon": [284.7, 284.9],
            "minimum_sea_level_pressure_hpa": [990.0, 985.0],
            "maximum_sustained_wind_speed_knots": [50.0, 60.0],
            "radius_of_maximum_winds_km": [40.0, 35.0],
            "radius_34_knot_winds_ne_km": [120.0, 130.0],
        }
    )
    tracks = pd.concat([late, early.iloc[::-1]], ignore_index=True)
    # metric columns the Earth2Studio wrapper adds
    tracks["tcmsl"] = tracks["minimum_sea_level_pressure_hpa"] * 100.0
    tracks["tcw10m"] = tracks["maximum_sustained_wind_speed_knots"] * KNOTS_TO_MS
    return tracks


def test_frame_has_te_layout_and_units():
    frame = wn2_tracks_to_te_frame(wn2_table())

    # radii named by their wind threshold in m/s: 34 kt is 17.5 m/s
    assert list(frame.columns) == [*TE_COLUMNS, "rmw_km", "r17_ne_km"]
    # tracks numbered by first valid time, points ordered by time
    assert frame["track_id"].tolist() == [0, 0, 1, 1, 1]
    assert frame["hour"].tolist() == [6, 12, 0, 6, 12]
    assert frame["day"].tolist() == [24, 24, 25, 25, 25]
    # Pa and m/s, longitudes in 0 to 360
    assert frame["msl"].iloc[0] == pytest.approx(99000.0)
    assert frame["wind_speed"].iloc[0] == pytest.approx(50.0 * KNOTS_TO_MS)
    assert frame["lon"].iloc[2] == pytest.approx(320.0)
    # nearest point of the 0.25 degree grid, latitudes from north to south
    assert frame[["i", "j"]].iloc[0].tolist() == [1139, 297]
    assert frame["height"].isna().all()


def test_empty_table_gives_empty_frame_with_columns():
    frame = wn2_tracks_to_te_frame(pd.DataFrame())
    assert frame.empty
    assert list(frame.columns) == list(TE_COLUMNS)


def test_file_is_read_like_a_tempest_extremes_file(tmp_path):
    frame = wn2_tracks_to_te_frame(wn2_table())
    path = tmp_path / track_file_name(IC, 3, 42, 1)
    write_te_style_csv(frame, str(path), {"model": "wn2", "checkpoint_member": 1})

    lines = path.read_text().splitlines()
    assert lines[0] == "# model: wn2"
    assert lines[2].split(", ")[: len(TE_COLUMNS)] == list(TE_COLUMNS)
    # formatted as StitchNodes does
    assert lines[3].startswith("0, 2025, 10, 24, 6, 1139, 297, 284.700000, 15.800000, ")
    assert lines[3].split(", ")[9:12] == ["9.900000e+04", "2.572220e+01", "nan"]

    tracks = extract_tracks_from_file(str(path))
    assert tracks["time"].iloc[0] == pd.Timestamp("2025-10-24 06:00")
    assert tracks["wind_speed"].iloc[0] == pytest.approx(50.0 * KNOTS_TO_MS, abs=1e-5)
    assert "i" not in tracks

    (member,) = extract_tracks(str(tmp_path))
    assert member["member"] == 3
    assert member["ic"] == pd.Timestamp("2025-10-24 00:00")


def test_writer_writes_one_file_per_stable_member(tmp_path):
    model = SimpleNamespace(track_cyclones=True, cyclone_tracks=wn2_table())
    writer = WN2TrackWriter(model, str(tmp_path), batch_size=1, header={"model": "wn2"})
    item = SimpleNamespace(time=IC, member_ids=(5,))

    writer.batch_finished(item, seed=None, attempts=1)  # failed batch: no file
    out_dir = tmp_path / WN2_TRACK_DIR
    assert list(out_dir.iterdir()) == []

    writer.batch_finished(item, seed=11, attempts=1)
    (path,) = out_dir.iterdir()
    assert path.name == "tracks_2025-10-24T00:00:00_mem_0005_seed_11_bs_1.csv"
    assert parse_track_file_name(str(path)) == (pd.Timestamp(IC), 5, 11)
    assert len(extract_tracks_from_file(str(path))) == 5


@pytest.mark.parametrize(
    "extra, header",
    [
        ({}, {"model": "wn2"}),  # model_kwargs is optional
        ({"model_kwargs": {"checkpoint_member": 1}}, {"checkpoint_member": 1}),
    ],
)
def test_writer_from_config(tmp_path, extra, header):
    cfg = OmegaConf.create(
        {"model": "wn2", "store_dir": str(tmp_path), "batch_size": 1, **extra}
    )
    writer = WN2TrackWriter.from_config(cfg, SimpleNamespace(track_cyclones=True))
    assert header.items() <= writer.header.items()


def test_writer_needs_a_tracking_model(tmp_path):
    with pytest.raises(ValueError, match="track_cyclones"):
        WN2TrackWriter(SimpleNamespace(track_cyclones=False), str(tmp_path), 1)


@pytest.mark.parametrize(
    "cfg, error",
    [
        ({"model": "wn2", "batch_size": 1, "wn2_tracking": {}}, None),
        ({"model": "fcn3", "batch_size": 2}, None),  # not configured: no check
        ({"model": "fcn3", "batch_size": 1, "wn2_tracking": {}}, "needs model 'wn2'"),
        ({"model": "aifs", "batch_size": 1, "wn2_tracking": {}}, "needs model 'wn2'"),
        ({"model": "wn2", "batch_size": 2, "wn2_tracking": {}}, "batch_size 1"),
    ],
)
def test_guard(cfg, error):
    cfg = OmegaConf.create(cfg)
    if error is None:
        validate_wn2_tracking(cfg)
    else:
        with pytest.raises(ValueError, match=error):
            validate_wn2_tracking(cfg)

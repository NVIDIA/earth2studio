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


import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import xarray as xr

from earth2studio.data import NNJAObsConv, NNJAObsSat, NNJAObsSatwnd
from earth2studio.data.utils_ncep import (
    _NCEP_PREPBUFR_WIDE_SCHEMA,
    _sat_wide_schema,
    to_dataframe,
)
from earth2studio.models.da.healda_v2 import (
    NLAT,
    NLON,
    HealDAv2,
    channel_to_e2s,
)

# ---------- Constants ----------

CHANNELS = ["U1000", "Z500", "T10", "tas", "pres_msl", "sst"]
E2S_VARIABLES = ["u1000", "z500", "t10", "t2m", "msl", "sst"]
CYCLE = np.datetime64("2024-01-05T00:00:00", "ns")
ELRC = 6_371_000.0


# ---------- Mock analysis model ----------


class PhooDAModel:
    """Stands in for healda.inference.DAModel."""

    def __init__(self, device="cpu"):
        self.net = torch.nn.Linear(1, 1)
        self.device = torch.device(device)
        self.channels = list(CHANNELS)
        self.satellite_sensors = ("atms", "amsua", "cris")
        self.ir_channels = {"cris": [19, 24]}
        self.calls: list[dict] = []

    def to(self, device):
        self.device = torch.device(device)
        self.net.to(self.device)
        return self

    def run_analysis(self, analysis_times, **tables):
        self.calls.append({"times": analysis_times, **tables})
        return torch.randn(
            len(analysis_times), len(self.channels), NLAT, NLON, device=self.device
        )


def _build_model(device="cpu") -> HealDAv2:
    return HealDAv2(PhooDAModel(device))


def _gpsro_df(request_time, n_levels=3):
    """Wide NNJAObsConv rows for one occultation: bending angles plus refractivity."""
    t = pd.Timestamp(request_time[0]) - pd.Timedelta(minutes=30)
    heights = np.arange(0.0, 40_001.0, 500.0)
    common = {
        "time": t,
        "class": "GPSRO",
        "type": 750,
        "station": "07500027",
        "radius_curvature": ELRC,
        "geoid_undulation": 10.0,
    }
    levels = pd.DataFrame(
        {
            **common,
            "lat": np.linspace(-9.0, -10.0, n_levels),
            "lon": np.linspace(290.0, 291.0, n_levels),
            "elev": np.linspace(2_500.0, 25_000.0, n_levels),
            "observation_gps": np.linspace(0.02, 0.001, n_levels),
            "quality_gps": 0,
        }
    )
    profile = pd.DataFrame(
        {
            **common,
            "lat": -10.5,
            "lon": 289.75,
            "elev": heights,
            "observation_gps_refractivity": 300.0 * np.exp(-heights / 7_000.0),
            "quality_gps_refractivity": 0,
        }
    )
    df = pd.concat([levels, profile], ignore_index=True)
    df["cycle_time"] = pd.Timestamp(request_time[0])
    df.attrs = {"request_time": request_time}
    return df


def _satwnd_df(request_time):
    """Wide NNJAObsSatwnd rows: two winds."""
    t = pd.Timestamp(request_time[0])
    winds = pd.DataFrame(
        {
            "time": [t - pd.Timedelta(hours=3), t + pd.Timedelta(hours=1)],
            "lat": [10.0, 12.0],
            "lon": [70.0, 250.0],
            "pres": [30_000.0, 50_000.0],
            "satellite_id": [473, 270],
            "subset": ["NC005024", "NC005030"],
            "wind_method": [1.0, 1.0],
            "wind_method_local": [np.nan, np.nan],
            "height_method": [np.nan, np.nan],
            "satellite_za": [20.0, 30.0],
            "quality": [np.nan, np.nan],
        }
    )
    df = winds.assign(observation_u=[5.0, -3.0], observation_v=[1.0, 2.0])
    df["cycle_time"] = pd.Timestamp(request_time[0])
    df.attrs = {"request_time": request_time}
    return df


def _wide_df(schema, request_time, **columns):
    """One-row frame of ``schema`` holding ``columns``; the other columns are null."""
    table = pa.table(
        {
            name: pa.array([value], schema.field(name).type)
            for name, value in columns.items()
        }
    )
    df = to_dataframe(table, schema, "test")
    df.attrs["request_time"] = request_time
    return df


def _prepbufr_df(request_time):
    """Wide NNJAObsConv PrepBUFR row: a radiosonde's surface level."""
    t = pd.Timestamp(request_time[0]) - pd.Timedelta(hours=1)
    return _wide_df(
        _NCEP_PREPBUFR_WIDE_SCHEMA,
        request_time,
        time=t,
        report_time=t,
        cycle_time=pd.Timestamp(request_time[0]),
        station="72403",
        type=120,
        lat=38.98,
        lon=282.53,
        report_lat=38.98,
        report_lon=282.53,
        station_elev=88.0,
        level_cat=0,
        pres=100_000.0,
        elev=88.0,
        pressure_quality=2,
        observation_pres=100_000.0,
        quality_pres=2,
        observation_t=288.45,
        quality_t=2,
    )


def _amsua_df(request_time):
    """Wide NNJAObsSat AMSU-A row: one footprint over the 15 channels."""
    return _wide_df(
        _sat_wide_schema("amsua"),
        request_time,
        time=pd.Timestamp(request_time[0]) - pd.Timedelta(hours=1),
        cycle_time=pd.Timestamp(request_time[0]),
        lat=10.0,
        lon=20.0,
        satellite="n19",
        scan_position=5,
        scan_line=1,
        satellite_za=30.0,
        solza=60.0,
        quality=[0] * 15,
        observation_amsua=[200.0 + channel for channel in range(15)],
    )


def test_channel_names_follow_the_earth2studio_vocabulary():
    assert [channel_to_e2s(c) for c in CHANNELS] == E2S_VARIABLES
    assert channel_to_e2s("100u") == "u100m"
    assert channel_to_e2s("tcw") == "tcw"


@pytest.mark.parametrize(
    "request_time",
    [
        np.array([CYCLE]),
        np.array([CYCLE, CYCLE + np.timedelta64(6, "h")]),
    ],
)
def test_healda_v2_call(request_time):
    model = _build_model()
    gpsro = _gpsro_df(request_time)
    satwnd = _satwnd_df(request_time)

    out = model(None, gpsro, satwnd)

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "variable", "lat", "lon")
    assert out.shape == (len(request_time), len(CHANNELS), NLAT, NLON)
    assert list(out.coords["variable"].values) == E2S_VARIABLES
    assert np.all(out.coords["time"].values == request_time)
    assert out.coords["lat"].values[0] == 90.0 and out.coords["lat"].values[-1] == -90.0
    assert out.coords["lon"].values[-1] < 360.0

    (call,) = model._model.calls
    assert list(call["times"]) == list(pd.DatetimeIndex(request_time))
    # The adapters keyed the rows by NCEP cycle for the healda loaders.
    assert pd.Timestamp(CYCLE) in call["gpsro_tables"]
    assert pd.Timestamp(CYCLE) in call["satwnd_tables"]
    assert "prepbufr_tables" not in call


def test_healda_v2_called_like_the_eval_pipeline():
    # recipes/eval fetches one frame per input_coords() schema, None for a disabled
    # slot, and calls model(*frames)
    model = _build_model()
    request_time = np.array([CYCLE])
    frame_of = {
        ("u", "v", "q", "t", "pres"): _prepbufr_df(request_time),
        ("u", "v"): _satwnd_df(request_time),
        ("amsua",): _amsua_df(request_time),
    }
    schemas = model.input_coords()
    frames = [frame_of.get(tuple(schema["variable"])) for schema in schemas]

    out = model(*frames)

    assert out.shape == (1, len(CHANNELS), NLAT, NLON)
    (call,) = model._model.calls
    assert set(call) == {
        "times",
        "prepbufr_tables",
        "satwnd_tables",
        "satellite_tables",
    }
    cycle = pd.Timestamp(CYCLE)
    assert call["prepbufr_tables"][cycle].num_rows == 1
    assert call["satellite_tables"]["amsua"][cycle].num_rows == 1


def test_healda_v2_omitted_streams_are_not_passed():
    model = _build_model()
    request_time = np.array([CYCLE])
    out = model(None, None, _satwnd_df(request_time))
    assert out.shape == (1, len(CHANNELS), NLAT, NLON)
    (call,) = model._model.calls
    assert set(call) == {"times", "satwnd_tables"}


def test_healda_v2_empty_sensor_frame_is_omitted():
    model = _build_model()
    request_time = np.array([CYCLE])
    schema = _sat_wide_schema("amsua")
    amsua = to_dataframe(schema.empty_table(), schema, "test")
    amsua.attrs["request_time"] = request_time
    # (prepbufr, gpsro, satwnd, atms, amsua, cris)
    model(None, None, _satwnd_df(request_time), None, amsua)
    (call,) = model._model.calls
    assert set(call) == {"times", "satwnd_tables"}


def test_healda_v2_accepts_cudf():
    cudf = pytest.importorskip("cudf")
    model = _build_model()
    request_time = np.array([CYCLE])
    frame = cudf.from_pandas(_satwnd_df(request_time))
    frame.attrs = {"request_time": request_time}
    out = model(None, None, frame)
    assert out.shape[0] == 1


def test_healda_v2_call_missing_request_time():
    model = _build_model()
    df = _satwnd_df(np.array([CYCLE]))
    df.attrs = {}
    with pytest.raises(ValueError, match="request_time"):
        model(None, None, df)


def test_healda_v2_call_mismatched_request_times():
    model = _build_model()
    later = np.array([CYCLE + np.timedelta64(6, "h")])
    with pytest.raises(ValueError, match="different 'request_time'"):
        model(None, _gpsro_df(np.array([CYCLE])), _satwnd_df(later))


def test_healda_v2_rejects_long_frames():
    model = _build_model()
    long = _satwnd_df(np.array([CYCLE])).rename(
        columns={"observation_u": "observation"}
    )
    with pytest.raises(ValueError, match='layout="wide"'):
        model(None, None, long)


def test_healda_v2_call_no_inputs():
    model = _build_model()
    with pytest.raises(ValueError, match="At least one"):
        model()


def test_healda_v2_call_empty_frames_runs_without_observations():
    model = _build_model()
    request_time = np.array([CYCLE])
    empty = _satwnd_df(request_time).iloc[:0].copy()
    empty.attrs = {"request_time": request_time}
    out = model(None, None, empty)
    assert out.shape == (1, len(CHANNELS), NLAT, NLON)
    assert np.isfinite(out.values).all()
    assert [set(call) for call in model._model.calls] == [{"times"}]


def test_healda_v2_coords():
    model = _build_model()
    prepbufr, gpsro, satwnd, atms, amsua, cris = model.input_coords()
    assert [list(schema["variable"]) for schema in model.input_coords()] == [
        ["u", "v", "q", "t", "pres"],
        ["gps", "gps_refractivity"],
        ["u", "v"],
        ["atms_antenna_temperature"],
        ["amsua"],
        ["cris"],
    ]
    # Each stream has its own columns
    assert "detector" in cris and "detector" not in atms
    assert "radius_curvature" in gpsro
    assert "report_time" in prepbufr and "report_time" not in gpsro
    assert all("scan_angle" not in schema for schema in model.input_coords())
    with pytest.raises(ValueError, match="at most 6 frames"):
        model(*[None] * 6, _satwnd_df(np.array([CYCLE])))
    coords = model.output_coords(*model.input_coords(), request_time=np.array([CYCLE]))
    assert list(coords.dims) == ["time", "variable", "lat", "lon"]
    assert coords.shape == (1, len(CHANNELS), NLAT, NLON)
    assert list(coords.coords["variable"].values) == E2S_VARIABLES


def test_healda_v2_default_sources():
    model = _build_model()
    conv, gpsro, satwnd, *sats = model.default_sources()
    assert isinstance(conv, NNJAObsConv) and gpsro is conv
    assert isinstance(satwnd, NNJAObsSatwnd)
    assert all(isinstance(sat, NNJAObsSat) for sat in sats)
    assert len(model.default_sources()) == len(model.input_coords())


def test_healda_v2_rng():
    model = _build_model()
    assert model.stochastic is False
    model.set_rng(0)


def test_healda_v2_sensor_indices():
    assert _build_model().sensor_indices == {"cris": [19, 24]}


@pytest.mark.package
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_healda_v2_package():
    package = HealDAv2.load_default_package()
    model = HealDAv2.load_model(package).to("cuda:0")
    request_time = np.array([CYCLE])

    out = model(None, _gpsro_df(request_time), _satwnd_df(request_time))

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "variable", "lat", "lon")
    assert out.shape[2:] == (NLAT, NLON)
    assert np.all(out.coords["time"].values == request_time)

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
import pytest
import torch
import xarray as xr

from earth2studio.models.da.healda_v2 import NLAT, NLON, HealDAv2, channel_to_e2s

# ---------- Constants ----------

CHANNELS = ["U1000", "Z500", "T10", "tas", "pres_msl", "sst"]
E2S_VARIABLES = ["u1000", "z500", "t10", "t2m", "msl", "sst"]
CYCLE = np.datetime64("2024-01-05T00:00:00", "ns")
ELRC = 6_371_000.0


# ---------- Mock analysis model ----------


class PhooAnalysisModel:
    """Stands in for healda.inference.AnalysisModel."""

    def __init__(self, device="cpu"):
        self.net = torch.nn.Linear(1, 1)
        self.device = torch.device(device)
        self.channels = list(CHANNELS)
        self.calls: list[dict] = []

    def analyze(self, analysis_times, *, gpsro_tables=None, satwnd_tables=None):
        self.calls.append(
            {
                "times": analysis_times,
                "gpsro": gpsro_tables,
                "satwnd": satwnd_tables,
            }
        )
        return torch.randn(
            len(analysis_times), len(self.channels), NLAT, NLON, device=self.device
        )


def _build_model(device="cpu") -> HealDAv2:
    return HealDAv2(PhooAnalysisModel(device))


def _gpsro_df(request_time, n_levels=3):
    """Raw NNJAObsConv rows for one occultation: bending angles plus refractivity."""
    t = pd.Timestamp(request_time[0]) - pd.Timedelta(minutes=30)
    heights = np.arange(0.0, 40_001.0, 500.0)
    common = {
        "time": t,
        "type": 750,
        "station": "07500027",
        "quality": 0.0,
        "radius_curvature": ELRC,
        "geoid_undulation": 10.0,
    }
    levels = pd.DataFrame(
        {
            **common,
            "lat": np.linspace(-9.0, -10.0, n_levels),
            "lon": np.linspace(290.0, 291.0, n_levels),
            "elev": np.linspace(2_500.0, 25_000.0, n_levels),
            "observation": np.linspace(0.02, 0.001, n_levels),
            "variable": "gps",
        }
    )
    profile = pd.DataFrame(
        {
            **common,
            "lat": -10.5,
            "lon": 289.75,
            "elev": heights,
            "observation": 300.0 * np.exp(-heights / 7_000.0),
            "variable": "gps_refractivity",
        }
    )
    df = pd.concat([levels, profile], ignore_index=True)
    df.attrs = {"request_time": request_time}
    return df


def _satwnd_df(request_time):
    """Raw NNJAObsSatwnd rows: two winds as u and v components."""
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
    u = winds.assign(observation=[5.0, -3.0], variable="u")
    v = winds.assign(observation=[1.0, 2.0], variable="v")
    df = pd.concat([u, v], ignore_index=True)
    df.attrs = {"request_time": request_time}
    return df


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

    out = model(gpsro_obs=gpsro, satwnd_obs=satwnd)

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
    assert call["gpsro"] is not None and call["satwnd"] is not None
    assert pd.Timestamp(CYCLE) in call["gpsro"]
    assert pd.Timestamp(CYCLE) in call["satwnd"]


def test_healda_v2_single_stream():
    model = _build_model()
    request_time = np.array([CYCLE])
    out = model(satwnd_obs=_satwnd_df(request_time))
    assert out.shape == (1, len(CHANNELS), NLAT, NLON)
    (call,) = model._model.calls
    assert call["gpsro"] is None


def test_healda_v2_accepts_frames_with_to_pandas():
    class HostFrame:
        def __init__(self, df):
            self._df = df
            self.attrs = df.attrs

        def to_pandas(self):
            return self._df

    model = _build_model()
    request_time = np.array([CYCLE])
    out = model(gpsro_obs=HostFrame(_gpsro_df(request_time)))
    assert out.shape[0] == 1


def test_healda_v2_call_missing_request_time():
    model = _build_model()
    df = _satwnd_df(np.array([CYCLE]))
    df.attrs = {}
    with pytest.raises(ValueError, match="request_time"):
        model(satwnd_obs=df)


def test_healda_v2_call_no_inputs():
    model = _build_model()
    with pytest.raises(ValueError, match="At least one"):
        model()


def test_healda_v2_call_empty_frames_returns_nan():
    model = _build_model()
    request_time = np.array([CYCLE])
    empty = _satwnd_df(request_time).iloc[:0].copy()
    empty.attrs = {"request_time": request_time}
    out = model(satwnd_obs=empty)
    assert out.shape == (1, len(CHANNELS), NLAT, NLON)
    assert np.all(np.isnan(out.values))
    assert model._model.calls == []


def test_healda_v2_generator():
    model = _build_model()
    request_time = np.array([CYCLE])
    gen = model.create_generator()
    assert gen.send(None) is None
    da = gen.send((_gpsro_df(request_time), None))
    assert isinstance(da, xr.DataArray)
    da = gen.send((None, _satwnd_df(request_time)))
    assert da.shape == (1, len(CHANNELS), NLAT, NLON)
    with pytest.raises(ValueError, match="At least one"):
        gen.send((None, None))
    gen.close()


def test_healda_v2_coords():
    model = _build_model()
    assert model.init_coords() is None
    gpsro_schema, satwnd_schema = model.input_coords()
    for schema in (gpsro_schema, satwnd_schema):
        for field in ("time", "lat", "lon", "observation", "variable"):
            assert field in schema
    assert list(gpsro_schema["variable"]) == ["gps", "gps_refractivity"]
    assert list(satwnd_schema["variable"]) == ["u", "v"]
    (coords,) = model.output_coords(
        model.input_coords(), request_time=np.array([CYCLE])
    )
    assert list(coords) == ["time", "variable", "lat", "lon"]
    assert len(coords["lat"]) == NLAT and len(coords["lon"]) == NLON


def test_healda_v2_to_moves_the_pipeline_device():
    model = _build_model()
    model.to("cpu")
    assert model.device == torch.device("cpu")
    assert model._model.device == torch.device("cpu")


@pytest.mark.package
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_healda_v2_package():
    package = HealDAv2.load_default_package()
    model = HealDAv2.load_model(package, device="cuda:0")
    request_time = np.array([CYCLE])

    out = model(gpsro_obs=_gpsro_df(request_time), satwnd_obs=_satwnd_df(request_time))

    assert isinstance(out, xr.DataArray)
    assert out.dims == ("time", "variable", "lat", "lon")
    assert out.shape[2:] == (NLAT, NLON)
    assert np.all(out.coords["time"].values == request_time)

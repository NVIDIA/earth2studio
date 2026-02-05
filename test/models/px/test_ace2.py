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

from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.data.ace2 import ACE_GRID_LAT, ACE_GRID_LON
from earth2studio.models.px.ace2 import (
    ACE2ERA5,
    _cftime_to_npdatetime64,
    _npdatetime64_to_cftime,
)
from earth2studio.utils import handshake_dim


class PhooOutput:
    prediction: dict = {}


class PhooStepper(torch.nn.Module):
    prognostic_names = [
        "Q2m",
        "northward_wind_2",
        "northward_wind_6",
        "northward_wind_3",
        "air_temperature_3",
        "air_temperature_5",
        "air_temperature_1",
        "northward_wind_0",
        "eastward_wind_2",
        "surface_temperature",
        "eastward_wind_6",
        "northward_wind_5",
        "eastward_wind_5",
        "eastward_wind_4",
        "eastward_wind_7",
        "air_temperature_6",
        "specific_total_water_4",
        "air_temperature_2",
        "specific_total_water_1",
        "UGRD10m",
        "specific_total_water_7",
        "specific_total_water_5",
        "northward_wind_1",
        "specific_total_water_3",
        "VGRD10m",
        "PRESsfc",
        "air_temperature_0",
        "specific_total_water_2",
        "eastward_wind_0",
        "air_temperature_4",
        "eastward_wind_1",
        "eastward_wind_3",
        "northward_wind_4",
        "specific_total_water_6",
        "northward_wind_7",
        "TMP2m",
        "air_temperature_7",
        "specific_total_water_0",
    ]
    _input_only_names = [
        "DSWRFtoa",
        "HGTsfc",
        "global_mean_co2",
        "land_fraction",
        "ocean_fraction",
        "sea_ice_fraction",
    ]
    out_names = [
        "PRESsfc",
        "surface_temperature",
        "air_temperature_0",
        "air_temperature_1",
        "air_temperature_2",
        "air_temperature_3",
        "air_temperature_4",
        "air_temperature_5",
        "air_temperature_6",
        "air_temperature_7",
        "specific_total_water_0",
        "specific_total_water_1",
        "specific_total_water_2",
        "specific_total_water_3",
        "specific_total_water_4",
        "specific_total_water_5",
        "specific_total_water_6",
        "specific_total_water_7",
        "eastward_wind_0",
        "eastward_wind_1",
        "eastward_wind_2",
        "eastward_wind_3",
        "eastward_wind_4",
        "eastward_wind_5",
        "eastward_wind_6",
        "eastward_wind_7",
        "northward_wind_0",
        "northward_wind_1",
        "northward_wind_2",
        "northward_wind_3",
        "northward_wind_4",
        "northward_wind_5",
        "northward_wind_6",
        "northward_wind_7",
        "LHTFLsfc",
        "SHTFLsfc",
        "PRATEsfc",
        "ULWRFsfc",
        "ULWRFtoa",
        "DLWRFsfc",
        "DSWRFsfc",
        "USWRFsfc",
        "USWRFtoa",
        "tendency_of_total_water_path_due_to_advection",
        "TMP850",
        "h500",
        "TMP2m",
        "Q2m",
        "UGRD10m",
        "VGRD10m",
    ]

    def __init__(self):
        super().__init__()
        self.register_buffer("device_buffer", torch.empty(0))

    def predict_paired(self, ic, forcing_batch):
        # https://github.com/ai2cm/ace/blob/a40f16bb4d985cdcd3234b1a540b4dd764b933db/fme/ace/data_loading/batch_data.py#L23
        batch, _, lat, lon = ic._data.data[self.prognostic_names[0]].shape
        output = PhooOutput()
        output.prediction = {
            key: torch.randn(batch, 1, lat, lon, device=self.device_buffer.device)
            for key in self.out_names
        }
        return output, None


@pytest.mark.parametrize("device", ["cuda:0"])
def test_ACE2ERA5_call(device):
    torch.cuda.empty_cache()
    # Use a single timestamp; forcing source will handle needed adjustments
    time = np.array([np.datetime64("2001-01-01T00:00")])

    forcing_source = Random({"lat": ACE_GRID_LAT, "lon": ACE_GRID_LON})
    p = ACE2ERA5(PhooStepper(), forcing_source).to(device)

    # Build a Random data source over the model grid
    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Validate shape and coords handshake
    assert out.shape[0] == len(time)
    assert out.shape[1] == 1  # one lead time step
    assert out.shape[3] == len(p.lat)
    assert out.shape[4] == len(p.lon)
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
    np.testing.assert_array_equal(out_coords["lat"], ACE_GRID_LAT)
    np.testing.assert_array_equal(out_coords["lon"], ACE_GRID_LON)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_ACE2ERA5_iter(batch, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2001-01-01T00:00")])

    forcing_source = Random({"lat": ACE_GRID_LAT, "lon": ACE_GRID_LON})
    p = ACE2ERA5(PhooStepper(), forcing_source).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to the front
    x = x.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch)})
    coords.move_to_end("batch", last=False)

    p_iter = p.create_iterator(x, coords)

    # First yield returns the first forecast step
    out, out_coords = next(p_iter)
    assert len(out.shape) == 6
    assert out.shape[0] == batch
    assert out_coords["lead_time"][0] == np.timedelta64(0, "h")

    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["batch"] == np.arange(batch)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
        np.testing.assert_array_equal(out_coords["lat"], ACE_GRID_LAT)
        np.testing.assert_array_equal(out_coords["lon"], ACE_GRID_LON)
        if i > 2:
            break


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_ace2era5_package(device):
    torch.cuda.empty_cache()
    model = ACE2ERA5.load_model(ACE2ERA5.load_default_package())

    # Use a single timestamp; forcing source will handle needed adjustments
    time = np.array([np.datetime64("2001-01-01T00:00")])

    p = model.to(device)

    # Build a Random data source over the model grid
    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Validate shape and coords handshake
    assert out.shape[0] == len(time)
    assert out.shape[1] == 1  # one lead time step
    assert out.shape[3] == len(p.lat)
    assert out.shape[4] == len(p.lon)
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


def test_time_conversion_helpers_roundtrip():
    dt_in = np.array(
        [
            np.datetime64("2001-01-02T03:04:05"),
            np.datetime64("2020-12-31T23:59:59"),
        ],
        dtype="datetime64[s]",
    )

    cftime_arr = _npdatetime64_to_cftime(dt_in)
    assert cftime_arr.shape == dt_in.shape
    dt_out = _cftime_to_npdatetime64(cftime_arr)
    np.testing.assert_array_equal(dt_out, dt_in)

    # Multi-dimensional shape preserved
    dt2 = dt_in.reshape(1, 2)
    cf2 = _npdatetime64_to_cftime(dt2)
    assert cf2.shape == (1, 2)
    dt2_back = _cftime_to_npdatetime64(cf2)
    np.testing.assert_array_equal(dt2_back, dt2)

    # Out-of-range years become NaT in reverse conversion
    try:
        import cftime  # local import for test
    except Exception:
        pytest.skip("cftime not available")
    cf_bad = np.array(
        [
            cftime.DatetimeProlepticGregorian(3000, 1, 1, 0, 0, 0),  # beyond supported
            cftime.DatetimeProlepticGregorian(2005, 1, 1, 0, 0, 0),  # valid
        ],
        dtype=object,
    )
    dt_bad = _cftime_to_npdatetime64(cf_bad)
    assert np.isnat(dt_bad[0])
    assert dt_bad[1] == np.datetime64("2005-01-01T00:00:00")

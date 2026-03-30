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
import xarray as xr

from earth2studio.data import Random, fetch_data
from earth2studio.data.ace2 import ACE_GRID_LAT, ACE_GRID_LON
from earth2studio.models.px.samudrace import SamudrACE
from earth2studio.utils import handshake_dim

pytest.importorskip("fme")

# Atmosphere variable lists matching SamudrACE coupled config
ATM_PROGNOSTIC = sorted(
    ["PRESsfc", "surface_temperature", "TMP2m", "Q2m", "UGRD10m", "VGRD10m"]
    + [f"air_temperature_{k}" for k in range(8)]
    + [f"specific_total_water_{k}" for k in range(8)]
    + [f"eastward_wind_{k}" for k in range(8)]
    + [f"northward_wind_{k}" for k in range(8)]
)

ATM_INPUT_ONLY = sorted(
    [
        "DSWRFtoa",
        "HGTsfc",
        "lake_fraction",
        "land_fraction",
        "ocean_fraction",
        "sea_ice_fraction",
    ]
)

ATM_OUT = ATM_PROGNOSTIC + [
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
    "eastward_surface_wind_stress",
    "northward_surface_wind_stress",
    "TMP850",
    "h500",
]

OCEAN_PROGNOSTIC = sorted(
    ["sst", "zos", "HI", "ocean_sea_ice_fraction"]
    + [f"thetao_{d}" for d in range(19)]
    + [f"so_{d}" for d in range(19)]
    + [f"uo_{d}" for d in range(19)]
    + [f"vo_{d}" for d in range(19)]
)

OCEAN_INPUT_ONLY = sorted(
    [
        "DLWRFsfc",
        "DSWRFsfc",
        "ULWRFsfc",
        "USWRFsfc",
        "LHTFLsfc",
        "SHTFLsfc",
        "PRATEsfc",
        "eastward_surface_wind_stress",
        "northward_surface_wind_stress",
        "land_fraction",
    ]
)

OCEAN_OUT = OCEAN_PROGNOSTIC

# External forcing variables that the wrapper reads from the forcing file
# (everything in ATM_INPUT_ONLY that is not coupled or from ocean output)
_EXTERNAL_FORCING = ["DSWRFtoa", "HGTsfc", "lake_fraction", "land_fraction"]


class PhooOutput:
    """Mock output from predict_paired."""

    prediction: dict = {}


class PhooStepper(torch.nn.Module):
    """Mock single-component stepper matching the FME Stepper interface."""

    def __init__(
        self,
        prognostic_names: list,
        input_only_names: list,
        out_names: list,
    ):
        super().__init__()
        self.prognostic_names = prognostic_names
        self._input_only_names = input_only_names
        self.out_names = out_names
        self.register_buffer("device_buffer", torch.empty(0))

    def predict_paired(self, ic, forcing_batch):
        first_key = self.prognostic_names[0]
        batch, _, lat, lon = ic._data.data[first_key].shape
        device = self.device_buffer.device
        output = PhooOutput()
        output.prediction = {
            key: torch.randn(batch, 1, lat, lon, device=device)
            for key in self.out_names
        }
        # Return updated state as PrognosticState for ocean updates
        import cftime
        from fme.ace.data_loading.batch_data import BatchData, PrognosticState

        state_data = {
            key: torch.randn(batch, 1, lat, lon, device=device)
            for key in self.prognostic_names
        }

        time_da = xr.DataArray(
            np.array(
                [
                    [cftime.DatetimeProlepticGregorian(2001, 1, 1, 0, 0, 0)]
                    for _ in range(batch)
                ],
                dtype=object,
            ),
            dims=["sample", "time"],
        )
        bd = BatchData.new_on_device(
            data=state_data, time=time_da, horizontal_dims=["lat", "lon"]
        )
        return output, PrognosticState(bd)


class PhooCoupledOceanFractionConfig:
    """Mock ocean fraction prediction config."""

    sea_ice_fraction_name = "ocean_sea_ice_fraction"
    land_fraction_name = "land_fraction"
    sea_ice_fraction_name_in_atmosphere = "sea_ice_fraction"


class PhooCoupledConfig:
    """Mock coupled stepper config."""

    sst_name = "sst"
    ocean_fraction_prediction = PhooCoupledOceanFractionConfig()


class PhooCoupledStepper:
    """Mock CoupledStepper matching the interface used by SamudrACE wrapper."""

    def __init__(self):
        self.atmosphere = PhooStepper(ATM_PROGNOSTIC, ATM_INPUT_ONLY, ATM_OUT)
        self.ocean = PhooStepper(OCEAN_PROGNOSTIC, OCEAN_INPUT_ONLY, OCEAN_OUT)
        self._config = PhooCoupledConfig()

    def to(self, device):
        self.atmosphere = self.atmosphere.to(device)
        self.ocean = self.ocean.to(device)
        return self


@pytest.fixture(scope="module")
def dummy_forcing_ds(tmp_path_factory):
    """Create a minimal forcing xarray Dataset for mock tests.

    Contains DSWRFtoa (time-varying, 3 days of 6h steps) plus static fields
    HGTsfc, land_fraction, and lake_fraction.
    """
    n_lat = len(ACE_GRID_LAT)
    n_lon = len(ACE_GRID_LON)
    # 12 time steps covering 3 days at 6h cadence
    import cftime

    times = [
        cftime.DatetimeNoLeap(2001, 1, d + 1, h)
        for d in range(3)
        for h in (0, 6, 12, 18)
    ]

    ds = xr.Dataset()
    ds["time"] = xr.DataArray(times, dims=["time"])
    ds["lat"] = xr.DataArray(ACE_GRID_LAT, dims=["lat"])
    ds["lon"] = xr.DataArray(ACE_GRID_LON, dims=["lon"])

    # Time-varying field
    ds["DSWRFtoa"] = xr.DataArray(
        np.random.randn(len(times), n_lat, n_lon).astype(np.float32),
        dims=["time", "lat", "lon"],
    )
    # Static fields
    for name in ("HGTsfc", "land_fraction", "lake_fraction"):
        ds[name] = xr.DataArray(
            np.random.randn(n_lat, n_lon).astype(np.float32),
            dims=["lat", "lon"],
        )

    return ds


@pytest.mark.parametrize("device", ["cuda:0"])
def test_samudrace_call(dummy_forcing_ds, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2001-01-01T00:00")])

    coupled = PhooCoupledStepper().to(device)
    p = SamudrACE(coupled).to(device)
    # Inject dummy forcing dataset to bypass HuggingFace download
    p._forcing_ds = dummy_forcing_ds

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

    assert out.shape[0] == len(time)
    assert out.shape[1] == 1
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
def test_samudrace_iter(dummy_forcing_ds, batch, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2001-01-01T00:00")])

    coupled = PhooCoupledStepper().to(device)
    p = SamudrACE(coupled).to(device)
    # Inject dummy forcing dataset to bypass HuggingFace download
    p._forcing_ds = dummy_forcing_ds

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    x = x.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch)})
    coords.move_to_end("batch", last=False)

    p_iter = p.create_iterator(x, coords)

    # First yield returns IC
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
def test_samudrace_package(device):
    torch.cuda.empty_cache()
    model = SamudrACE.load_model(SamudrACE.load_default_package())

    time = np.array([np.datetime64("2001-01-01T00:00")])

    p = model.to(device)

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

    assert out.shape[0] == len(time)
    assert out.shape[1] == 1
    assert out.shape[3] == len(p.lat)
    assert out.shape[4] == len(p.lon)
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)

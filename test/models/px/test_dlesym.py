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


from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.px.dlesym as dlesym_src
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import DLESyM, DLESyMLatLon
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import OptionalDependencyFailure

EARTH2GRID_AVAILABLE = dlesym_src.earth2grid is not None


@pytest.fixture(autouse=True)
def optional_backend(monkeypatch):
    if dlesym_src.insolation is None:
        monkeypatch.delitem(
            OptionalDependencyFailure.failures, dlesym_src.__file__, raising=False
        )
        monkeypatch.setattr(
            dlesym_src,
            "insolation",
            lambda times, lat, lon: np.zeros(
                (len(times), *lat.shape), dtype=np.float32
            ),
        )

    if not EARTH2GRID_AVAILABLE:

        class Regrid(torch.nn.Module):
            def __init__(self, source, target):
                super().__init__()
                self.target = target

            def forward(self, x):
                flat = x.flatten(1)
                indexes = torch.linspace(
                    0,
                    flat.shape[1] - 1,
                    int(np.prod(self.target.shape)),
                    device=x.device,
                ).long()
                return flat[:, indexes].reshape(x.shape[0], *self.target.shape)

        monkeypatch.setattr(
            dlesym_src,
            "earth2grid",
            SimpleNamespace(
                healpix=SimpleNamespace(
                    HEALPIX_PAD_XY="xy",
                    Grid=lambda level, **kw: SimpleNamespace(shape=(12 * 4**level,)),
                ),
                latlon=SimpleNamespace(
                    equiangular_lat_lon_grid=lambda h, w: SimpleNamespace(shape=(h, w))
                ),
                get_regridder=Regrid,
            ),
        )


class PhooAtmosModel(torch.nn.Module):
    """Mock atmosphere model for testing."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = len(dlesym_src._ATMOS_OUTPUT_TIMES)
        self.input_time_dim = len(dlesym_src._ATMOS_INPUT_TIMES)

    def forward(self, in_list):
        x = in_list[0]
        b, t = x.shape[:2]
        return torch.ones(b, t, self.output_time_dim, *x.shape[3:], device=x.device)


class PhooOceanModel(torch.nn.Module):
    """Mock ocean model for testing."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = len(dlesym_src._OCEAN_OUTPUT_TIMES)
        self.input_time_dim = len(dlesym_src._OCEAN_INPUT_TIMES)

    def forward(self, in_list):
        x = in_list[0]
        b, t = x.shape[:2]
        return torch.ones(b, t, self.output_time_dim, *x.shape[3:], device=x.device)


def build_dlesym_model(device, nside=64, type="hpx"):
    """Build a DLESyM prognostic model with mock atmos/ocean models.

    Parameters
    ----------
    device : torch.device
        The device to build the model on.
    nside : int
        The nside of the HEALPix grid.
    type : str
        The type of model to build. Set to "hpx" for HEALPix grid or "ll" for lat/lon grid.

    Returns
    -------
    model : DLESyM or DLESyMLatLon
        The DLESyM model.
    """
    # A local generator, not the global np.random state: this helper is shared by
    # the conformance test, and drawing from the global generator made whether the
    # checker's P16 aliasing probe actually observed a value change depend on
    # whatever earlier test in the suite last touched np.random, i.e. test order.
    rng = np.random.default_rng(0)
    hpx_lat = rng.standard_normal((12, nside, nside))
    hpx_lon = rng.standard_normal((12, nside, nside))
    center = np.zeros((1, 1, 1, 9, 1, 1, 1))  # 9 variables total
    scale = np.ones((1, 1, 1, 9, 1, 1, 1))
    atmos_constants = rng.standard_normal((12, 2, nside, nside))
    ocean_constants = rng.standard_normal((12, 2, nside, nside))

    atmos_input_times = dlesym_src._ATMOS_INPUT_TIMES
    ocean_input_times = dlesym_src._OCEAN_INPUT_TIMES
    atmos_output_times = dlesym_src._ATMOS_OUTPUT_TIMES
    ocean_output_times = dlesym_src._OCEAN_OUTPUT_TIMES

    atmos_variables = dlesym_src._ATMOS_VARIABLES
    ocean_variables = dlesym_src._OCEAN_VARIABLES
    atmos_coupling_variables = dlesym_src._ATMOS_COUPLING_VARIABLES
    ocean_coupling_variables = dlesym_src._OCEAN_COUPLING_VARIABLES

    model_constructor = DLESyM if type == "hpx" else DLESyMLatLon

    model = model_constructor(
        atmos_model=PhooAtmosModel(),
        ocean_model=PhooOceanModel(),
        hpx_lat=hpx_lat,
        hpx_lon=hpx_lon,
        nside=nside,
        center=center,
        scale=scale,
        atmos_constants=atmos_constants,
        ocean_constants=ocean_constants,
        atmos_input_times=atmos_input_times,
        ocean_input_times=ocean_input_times,
        atmos_output_times=atmos_output_times,
        ocean_output_times=ocean_output_times,
        atmos_variables=atmos_variables,
        ocean_variables=ocean_variables,
        atmos_coupling_variables=atmos_coupling_variables,
        ocean_coupling_variables=ocean_coupling_variables,
    ).to(device)

    return model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("grid_type", ["hpx", "ll"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_forward(device, grid_type, batch_size):
    """Test DLESyM forward pass with mock models."""

    if grid_type == "ll" and device == "cpu":
        pytest.skip("Lat/lon regridding is slow on CPU")

    nside = 64
    model = build_dlesym_model(device, type=grid_type)

    spatial_dims = (12, nside, nside) if grid_type == "hpx" else (721, 1440)

    # Create test input
    time = np.array([np.datetime64("2020-01-01T00:00")])
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x = torch.randn(
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        *spatial_dims,
        device=device,
    )

    output_vars = model.output_coords(model.input_coords())["variable"]

    # Test forward pass
    in_coords = model.input_coords()
    assert isinstance(in_coords, xr.DataArray)
    assert in_coords.data.nbytes == 0
    in_coords = coord_array_like(
        in_coords, {"batch": np.arange(batch_size), "time": time}
    )
    field = from_torch(x, in_coords, name="state")
    field.encoding = {"test": "dlesym"}
    if grid_type == "hpx":
        assert field.attrs["ordering"] == "xy"
        assert field.attrs["origin"] == "north"
        assert field.attrs["clockwise"] is True
        assert "earth2studio_grid_id" not in field.attrs
        for key, value in {
            "origin": "south",
            "clockwise": False,
            "ordering": "nested",
            "layout": "flat",
            "level": 0,
            "nside": 1,
            "type": "LatLonGrid",
            "topology": "rectilinear",
            "dims": ["hpx"],
            "shape": [12 * nside**2],
        }.items():
            bad = field.assign_attrs({key: value})
            with pytest.raises(ValueError, match="HEALPix"):
                model.output_coords(bad)
        bad = field.copy(deep=False)
        bad.attrs.pop("origin")
        with pytest.raises(ValueError, match="HEALPix"):
            model(bad)
    else:
        field = field.assign_coords(terrain=(("lat", "lon"), np.ones(spatial_dims)))
        field.terrain.attrs["units"] = "m"
    output = model(field)
    output_coords = output.coords
    assert output.name == field.name and output.encoding == field.encoding
    expected_coords = model.output_coords(field)
    if grid_type == "ll":
        xr.testing.assert_identical(output.terrain, field.terrain)
    assert output.shape == (
        batch_size,
        len(time),
        len(dlesym_src._ATMOS_OUTPUT_TIMES),
        len(output_vars),
        *spatial_dims,
    )
    for key in output_coords:
        np.testing.assert_array_equal(output_coords[key], expected_coords[key])
    assert np.all(output_coords["lead_time"] == dlesym_src._ATMOS_OUTPUT_TIMES)

    # Test retrieving valid outputs
    atmos_outputs = model.retrieve_valid_atmos_outputs(output)
    atmos_coords = atmos_outputs.coords
    assert atmos_outputs.shape == (
        batch_size,
        len(time),
        len(dlesym_src._ATMOS_OUTPUT_TIMES),
        len(dlesym_src._ATMOS_VARIABLES),
        *spatial_dims,
    )
    assert np.all(atmos_coords["lead_time"] == dlesym_src._ATMOS_OUTPUT_TIMES)

    ocean_outputs = model.retrieve_valid_ocean_outputs(output)
    ocean_coords = ocean_outputs.coords
    assert ocean_outputs.shape == (
        batch_size,
        len(time),
        len(dlesym_src._OCEAN_OUTPUT_TIMES),
        len(dlesym_src._OCEAN_VARIABLES),
        *spatial_dims,
    )
    assert np.all(ocean_coords["lead_time"] == dlesym_src._OCEAN_OUTPUT_TIMES)


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_latlon_regridding(device, batch_size):
    """Test DLESyMLatLon regridding functionality."""
    if not EARTH2GRID_AVAILABLE:
        pytest.skip("Round-trip accuracy requires real Earth2Grid")
    nside = 64
    model = build_dlesym_model(device, type="ll")

    # Test basic coordinate conversion
    ll_coords = model.input_coords()
    hpx_coords = model.coords_to_hpx(ll_coords)
    for coord in ["lat", "lon"]:
        assert coord not in hpx_coords.dims
        assert coord in ll_coords.dims
    for coord in ["face", "height", "width"]:
        assert coord in hpx_coords.dims
        assert coord not in ll_coords.dims

    # Test regridding
    time = np.array([np.datetime64("2020-01-01T00:00")])
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x_ll = torch.randn(
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        721,  # lat
        1440,  # lon
        device=device,
    )

    # Test round-trip regridding
    in_coords = model.input_coords()
    in_coords = coord_array_like(
        in_coords, {"batch": np.arange(batch_size), "time": time}
    )
    x_hpx = model.to_hpx(x_ll)
    assert x_hpx.shape == (
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        12,
        nside,
        nside,
    )
    x_ll_roundtrip = model.to_ll(x_hpx)

    # Round-trip regridding error is more sensitive to fine-scale detail
    # so here we just check that the mean error is less than 1e-2
    diff = x_ll - x_ll_roundtrip
    assert diff.mean().item() < 1e-2


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("grid_type", ["hpx", "ll"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_iterator(device, grid_type, batch_size):
    """Test DLESyM iterator functionality."""

    if grid_type == "ll" and device == "cpu":
        pytest.skip("Lat/lon regridding is slow on CPU")

    nside = 64
    model = build_dlesym_model(device, type=grid_type, nside=nside)
    spatial_dims = (12, nside, nside) if grid_type == "hpx" else (721, 1440)
    # Create test input
    time = np.array([np.datetime64("2020-01-01T00:00")])
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x = torch.randn(
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        *spatial_dims,
        device=device,
    )

    output_vars = model.output_coords(model.input_coords())["variable"]

    # Test iterator
    in_coords = model.input_coords()
    in_coords = coord_array_like(
        in_coords, {"batch": np.arange(batch_size), "time": time}
    )
    field = from_torch(x, in_coords)
    field = field.rename(batch="member").drop_vars("member")
    field.name = "history"
    field.encoding = {"test": "history"}
    if grid_type == "ll":
        field = field.assign_coords(terrain=(("lat", "lon"), np.ones(spatial_dims)))
        field.terrain.attrs["units"] = "m"
    events = []

    def front(state):
        if grid_type == "ll":
            assert "terrain" not in state.coords
        assert state.dims[0] == "member" and "member" not in state.coords
        events.append("front")
        state.data[...] += 1
        return state

    def rear(state):
        if grid_type == "ll":
            assert "terrain" not in state.coords
            state.attrs["rear"] = "retained"
        assert state.dims[0] == "member"
        events.append("rear")
        return state

    model.front_hook, model.rear_hook = front, rear
    original = field.copy(deep=True)
    iterator = model.create_iterator(field)

    # First yield should be initial condition
    initial_x = next(iterator)
    xr.testing.assert_identical(initial_x, field.isel(lead_time=slice(-1, None)))
    saved = initial_x.copy(deep=True)
    assert events == []

    # Test a few steps
    coupler_step = dlesym_src._ATMOS_OUTPUT_TIMES[-1]
    for i in range(3):
        x = next(iterator)
        if grid_type == "ll":
            xr.testing.assert_identical(x.terrain, field.terrain)
            assert x.attrs["rear"] == "retained"
        coords = x.coords
        assert x.shape == (
            batch_size,
            len(time),
            len(dlesym_src._ATMOS_OUTPUT_TIMES),
            len(output_vars),
            *spatial_dims,
        )
        assert np.all(
            coords["lead_time"] == dlesym_src._ATMOS_OUTPUT_TIMES + coupler_step * i
        )
    xr.testing.assert_identical(initial_x, saved)
    xr.testing.assert_identical(field, original)
    assert events == ["front", "rear"] * 3


def test_dlesym_conformance():
    model = build_dlesym_model("cpu", nside=8, type="hpx")
    check_prognostic_contract(model)
    model.atmos_variables = [
        "ttr03" if v == "z500" else v for v in model.atmos_variables
    ]
    signature = model.input_coords()
    assert signature["variable"].values[0] == "ttr:sum:-2h:1h"
    assert "ttr:sum:-2h:1h" in signature.attrs["earth2studio_statistics"]


def test_dlesym_latlon_conformance():
    model = build_dlesym_model("cpu", nside=8, type="ll")
    check_prognostic_contract(model)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_dlesym_package(device):
    torch.cuda.empty_cache()
    model = DLESyM.load_model(DLESyM.load_default_package())
    model = model.to(device)

    nside = 64
    spatial_dims = (12, nside, nside)
    batch_size = 1

    time = np.array([np.datetime64("2020-01-01T00:00")])
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x = torch.randn(
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        *spatial_dims,
        device=device,
    )

    # Test forward pass
    in_coords = model.input_coords()
    in_coords = coord_array_like(
        in_coords, {"batch": np.arange(batch_size), "time": time}
    )
    output = model(from_torch(x, in_coords))
    output_coords = output.coords
    expected_coords = model.output_coords(in_coords)
    assert output.shape == (
        batch_size,
        len(time),
        len(dlesym_src._ATMOS_OUTPUT_TIMES),
        len(expected_coords["variable"]),
        *spatial_dims,
    )
    for key in output_coords:
        np.testing.assert_array_equal(output_coords[key], expected_coords[key])
    assert np.all(output_coords["lead_time"] == dlesym_src._ATMOS_OUTPUT_TIMES)

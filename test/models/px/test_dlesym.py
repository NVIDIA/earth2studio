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


import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.px.dlesym as dlesym_src
from earth2studio.coupling import Connector, CoupledRunner
from earth2studio.grids import HEALPixGrid
from earth2studio.io import KVBackend
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import DLESyM, DLESyMAtmos, DLESyMLatLon, DLESyMOcean
from earth2studio.run import Pipeline, PrognosticRunner, WorkItem, work_items
from earth2studio.utils.coords import coord_array, coord_array_like, handshake_metadata
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


class CoupledAtmosModel(PhooAtmosModel):
    """Mock atmosphere core whose output depends on its state and SST coupling."""

    def forward(self, in_list):
        state, _, _, coupling = in_list
        # state: (batch, face, lead, variable, h, w); coupling: (step, batch, variable, face, h, w)
        last = state[:, :, -1:]
        sst = coupling.mean(dim=(0, 2))[:, :, None, None]
        leads = torch.arange(self.output_time_dim, device=state.device)
        return last + 0.1 * sst + 0.01 * leads.view(1, 1, -1, 1, 1, 1)


class NoisyAtmosModel(CoupledAtmosModel):
    """Coupled atmosphere core that also depends on conditional layer norm noise."""

    integration_steps = 2

    def forward(self, in_list, conditions_cln=None):
        out = super().forward(in_list)
        assert conditions_cln is not None and len(conditions_cln) == 2
        assert all(c.shape == (in_list[0].shape[0], 3) for c in conditions_cln)
        noise = torch.stack(conditions_cln).mean(dim=(0, 2))
        return out + 0.05 * noise.view(-1, 1, 1, 1, 1, 1)


class CoupledOceanModel(PhooOceanModel):
    """Mock ocean core whose output depends on its state and window-mean coupling."""

    def forward(self, in_list):
        state, _, _, coupling = in_list
        # coupling: (1, batch, window-major variables, face, h, w)
        weights = torch.arange(1, coupling.shape[2] + 1, device=state.device)
        forced = (coupling[0] * weights.view(1, -1, 1, 1, 1)).sum(dim=1)
        leads = torch.arange(self.output_time_dim, device=state.device)
        return (
            state[:, :, -1:]
            + 0.5 * forced[:, :, None, None]
            + 0.01 * leads.view(1, 1, -1, 1, 1, 1)
        )


class NoisyOceanModel(CoupledOceanModel):
    """Coupled ocean core that also depends on conditional layer norm noise."""

    integration_steps = 1

    def forward(self, in_list, conditions_cln=None):
        out = super().forward(in_list)
        assert conditions_cln is not None and len(conditions_cln) == 1
        noise = conditions_cln[0].mean(dim=1)
        return out + 0.05 * noise.view(-1, 1, 1, 1, 1, 1)


def build_dlesym_model(
    device,
    nside=64,
    type="hpx",
    atmos_model=None,
    ocean_model=None,
    scaled=False,
    use_cln=False,
):
    """Build a DLESyM prognostic model with mock atmos/ocean models.

    Parameters
    ----------
    device : torch.device
        The device to build the model on.
    nside : int
        The nside of the HEALPix grid.
    type : str
        The type of model to build. Set to "hpx" for HEALPix grid or "ll" for lat/lon grid.
    atmos_model, ocean_model : torch.nn.Module, optional
        Component cores, by default the constant mocks.
    scaled : bool
        Draw nontrivial normalization constants instead of zero center, unit scale.
    use_cln : bool
        Sample conditional layer norm noise (three conditions) for each core call.

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
    if scaled:
        center = rng.uniform(-1.0, 1.0, center.shape)
        scale = rng.uniform(0.5, 2.0, scale.shape)
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
        atmos_model=atmos_model if atmos_model is not None else PhooAtmosModel(),
        ocean_model=ocean_model if ocean_model is not None else PhooOceanModel(),
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
        use_cln=use_cln,
        condition_shape=3 if use_cln else None,
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
        assert (
            field.attrs["earth2studio_grid_id"] == "healpix-l6-xy-north-clockwise-face"
        )
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
            with pytest.raises(ValueError, match="metadata"):
                handshake_metadata(bad, in_coords, (key,))
        bad = field.copy(deep=False)
        bad.attrs.pop("origin")
        with pytest.raises(ValueError, match="metadata"):
            handshake_metadata(bad, in_coords, ("origin",))
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
            xr.testing.assert_identical(state.terrain, field.terrain)
        assert state.dims[0] == "member" and "member" not in state.coords
        events.append("front")
        state.data[...] += 1
        return state

    def rear(state):
        if grid_type == "ll":
            xr.testing.assert_identical(state.terrain, field.terrain)
            state.attrs["rear"] = "retained"
        assert state.dims[0] == "member"
        events.append("rear")
        return state

    model.front_hook, model.rear_hook = front, rear
    original = field.copy(deep=True)
    iterator = model.create_iterator(field)

    assert events == []

    # Test a few steps
    coupler_step = dlesym_src._ATMOS_OUTPUT_TIMES[-1]
    for i in range(3):
        x = next(iterator)
        if i == 0:
            first, saved = x, x.copy(deep=True)
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
    saved.data += 1
    xr.testing.assert_identical(first, saved)
    xr.testing.assert_identical(field, original)
    assert events == ["rear", "front", "rear", "front", "rear"]


def test_dlesym_conformance():
    model = build_dlesym_model("cpu", nside=8, type="hpx")
    check_prognostic_contract(model)


def test_dlesym_latlon_conformance():
    model = build_dlesym_model("cpu", nside=8, type="ll")
    check_prognostic_contract(model)


class HEALPixFill:
    """Data source of deterministic fields on DLESyM's native grid.

    Each variable is offset by its index in the fused input, drifts with time and
    carries a fixed spatial pattern, so wrong frames or variables are detectable.
    """

    def __init__(self, model):
        signature = DLESyM.input_coords(model)
        self.variables = list(signature["variable"].values)
        self.grid = HEALPixGrid(
            int(np.log2(model.nside)),
            ordering="xy",
            layout="face",
            xy_origin="north",
            xy_clockwise=True,
        )
        self.spatial = {
            dim: signature.coords[dim].values for dim in ("face", "height", "width")
        }
        pattern = np.linspace(0.0, 0.1, int(np.prod(self.grid.shape)), dtype=np.float32)
        self.pattern = pattern.reshape(self.grid.shape)

    def __call__(self, time, variable):
        time = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variable = np.atleast_1d(np.asarray(variable))
        hours = (time - np.datetime64("2020-01-01")) / np.timedelta64(1, "h")
        index = np.array([self.variables.index(v) for v in variable], dtype=np.float32)
        values = (
            1.0
            + index[None, :, None, None, None]
            + 1e-3 * hours.astype(np.float32)[:, None, None, None, None]
            + self.pattern[None, None]
        )
        coords = coord_array(
            ("time", "variable", "face", "height", "width"),
            {"time": time, "variable": variable, **self.spatial},
            grid=self.grid,
        )
        return from_torch(torch.from_numpy(values.astype(np.float32)), coords)

    async def fetch(self, time, variable):
        return self(time, variable)


def hours(coords):
    """Lead times of a signature or dataset in whole hours."""
    return list(coords["lead_time"].values.astype("timedelta64[h]").astype(int))


def _split_model():
    return build_dlesym_model(
        "cpu",
        nside=8,
        atmos_model=CoupledAtmosModel(),
        ocean_model=CoupledOceanModel(),
        scaled=True,
    )


def test_dlesym_split_declarations():
    model = _split_model()
    atmos, ocean = model.split()
    assert isinstance(atmos, DLESyMAtmos) and isinstance(ocean, DLESyMOcean)
    assert atmos.model is model and ocean.model is model

    assert hours(atmos.input_coords()) == [-18, -12, -6, 0]
    assert list(atmos.input_coords()["variable"].values) == dlesym_src._ATMOS_VARIABLES
    assert hours(atmos.forcing_coords()) == [0]
    assert list(atmos.forcing_coords()["variable"].values) == ["sst"]
    assert hours(atmos.output_coords(atmos.input_coords())) == list(range(6, 97, 6))

    assert hours(ocean.input_coords()) == [-48, 0]
    assert list(ocean.input_coords()["variable"].values) == ["sst"]
    # Window means valid at the ocean's output times, over the atmosphere frames
    # of the same cycle: [T - 42h, T + 6h) is 6h..48h for T = 48h.
    assert hours(ocean.forcing_coords()) == [48, 96]
    assert list(ocean.forcing_coords()["variable"].values) == [
        "z1000:mean:-42h:+6h",
        "ws10m:mean:-42h:+6h",
    ]
    assert hours(ocean.output_coords(ocean.input_coords())) == [48, 96]
    native = DLESyM.input_coords(model)
    for component in (atmos, ocean):
        for signature in (component.input_coords(), component.forcing_coords()):
            assert signature.data.nbytes == 0
            assert signature.dims == native.dims
            assert signature.attrs["nside"] == native.attrs["nside"]


def test_dlesym_split_conformance():
    atmos, ocean = build_dlesym_model("cpu", nside=8).split()
    check_prognostic_contract(atmos)
    check_prognostic_contract(ocean)


def _coupled(model, source):
    atmos, ocean = model.split()
    return CoupledRunner(
        {"atmos": atmos, "ocean": ocean},
        [
            Connector("ocean", "atmos", time="hold"),
            Connector("atmos", "ocean"),
        ],
        sources={"atmos": source, "ocean": source},
    )


def test_dlesym_split_schedule():
    model = _split_model()
    plan = _coupled(model, HEALPixFill(model)).describe(np.timedelta64(192, "h"))
    assert "atmos  stateful    every 96h" in plan
    assert "ocean  stateful    every 96h" in plan
    assert "ocean -> atmos forcing[0] (hold): sst" in plan
    assert (
        "atmos -> ocean forcing[0] (exact): z1000:mean:-42h:+6h, ws10m:mean:-42h:+6h"
        in plan
    )
    # The atmosphere initializes from the ocean's initial SST and the ocean
    # reduces the atmosphere's first cycle. The next atmosphere cycle is forced
    # at the previous cycle's lead times, held from the ocean's earlier frames:
    # the +96h frame it uses is the ocean's +96h output.
    assert "atmos  initialize" in plan and "[forcing[0] <- ocean at +0h]" in plan
    assert "ocean  initialize" in plan
    assert "forcing[0] <- atmos at +48h, +96h from +6h..+96h" in plan
    assert "atmos  step" in plan and "+96h from +0h..+96h]" in plan
    assert "forcing[0] <- atmos at +144h, +192h from +102h..+192h" in plan


def test_dlesym_split_reproduces_fused():
    model = _split_model()
    source = HEALPixFill(model)
    item = WorkItem(np.datetime64("2020-06-01T00"), np.timedelta64(288, "h"))

    fused = [
        step["forecast"] for step in PrognosticRunner(model, source).run_item(item)
    ]
    streams: dict[str, list[xr.DataArray]] = {}
    for step in _coupled(model, source).run_item(item):
        for name, x in step.items():
            streams.setdefault(name, []).append(x)

    # The initial condition then three cycles, in every stream.
    assert len(fused) == 4
    assert len(streams["atmos"]) == 4 and len(streams["ocean"]) == 4
    xr.testing.assert_allclose(
        streams["atmos"][0].drop_vars("lead_time"),
        fused[0].sel(variable=dlesym_src._ATMOS_VARIABLES).drop_vars("lead_time"),
    )
    xr.testing.assert_allclose(
        streams["ocean"][0].drop_vars("lead_time"),
        fused[0].sel(variable=["sst"]).drop_vars("lead_time"),
    )
    for k in range(1, 4):
        atmos = model.retrieve_valid_atmos_outputs(fused[k])
        ocean = model.retrieve_valid_ocean_outputs(fused[k])
        xr.testing.assert_allclose(streams["atmos"][k], atmos, rtol=1e-5, atol=1e-5)
        xr.testing.assert_allclose(streams["ocean"][k], ocean, rtol=1e-5, atol=1e-5)
    # The coupling mattered: the ocean's SST changed the fused result.
    assert not np.allclose(
        fused[1].sel(variable="z500").values, fused[3].sel(variable="z500").values
    )


def _noisy_model():
    return build_dlesym_model(
        "cpu",
        nside=8,
        atmos_model=NoisyAtmosModel(),
        ocean_model=NoisyOceanModel(),
        scaled=True,
        use_cln=True,
    )


def test_dlesym_split_stochastic_conformance():
    atmos, ocean = _noisy_model().split()
    assert atmos.stochastic and ocean.stochastic
    check_prognostic_contract(atmos)
    check_prognostic_contract(ocean)


def test_dlesym_split_components_draw_independent_seeded_noise():
    model = _noisy_model()
    source = HEALPixFill(model)
    item = WorkItem(np.datetime64("2020-06-01T00"), np.timedelta64(192, "h"))

    def run(seed):
        runner = _coupled(model, source)
        runner.seed = seed
        streams = {}
        for step in runner.run_item(item):
            for name, x in step.items():
                streams.setdefault(name, []).append(x)
        return streams

    first, second, other = run(0), run(0), run(1)
    for name in ("atmos", "ocean"):
        for a, b in zip(first[name], second[name]):
            xr.testing.assert_identical(a, b)
        assert not np.allclose(first[name][1].values, other[name][1].values)
    # The components are seeded apart from the fused model and from each other,
    # so a coupled run is another realization rather than the fused run.
    fused = [s["forecast"] for s in PrognosticRunner(model, source).run_item(item)]
    assert not np.allclose(
        first["atmos"][1].values,
        model.retrieve_valid_atmos_outputs(fused[1]).values,
    )


def test_dlesym_split_step_replays_from_state():
    model = _noisy_model()
    atmos, ocean = model.split()
    source = HEALPixFill(model)
    t0 = np.datetime64("2020-06-01T00")
    runner = _coupled(model, source)
    chunks = [step for step in runner.run_item(WorkItem(t0, np.timedelta64(96, "h")))]
    y_atmos = [s["atmos"] for s in chunks if "atmos" in s][-1]
    y_ocean = [s["ocean"] for s in chunks if "ocean" in s][-1]
    sst = y_ocean.isel(lead_time=[-1]).assign_coords(
        lead_time=y_atmos["lead_time"].values[-1:]
    )
    atmos.set_rng(7)
    first, state = atmos.initialize(
        y_atmos.isel(lead_time=slice(-4, None)), sst.copy(deep=True)
    )

    def held(y):  # the same SST, relabeled at the chunk's final lead time
        return sst.assign_coords(lead_time=y["lead_time"].values[-1:])

    one, next_state = atmos.step(first, held(first), state)
    again, _ = atmos.step(first, held(first), state)
    xr.testing.assert_identical(one, again)
    assert not torch.equal(state, next_state)
    # Continuing advances the stream: the next step differs from a replay.
    two, _ = atmos.step(one, held(one), next_state)
    assert not np.allclose(two.values, one.values)


def test_dlesym_split_rejects_mismatched_forcing():
    atmos, _ = _split_model().split()
    t0 = np.datetime64("2020-06-01T00")
    x = from_torch(
        torch.zeros(1, 1, 4, 8, 12, 8, 8),
        coord_array_like(
            atmos.input_coords(), {"batch": np.arange(1), "time": np.array([t0])}
        ),
    )
    good = from_torch(
        torch.zeros(1, 1, 1, 1, 12, 8, 8),
        coord_array_like(
            atmos.forcing_coords(), {"batch": np.arange(1), "time": np.array([t0])}
        ),
    )
    atmos(x, good)
    with pytest.raises(ValueError):
        atmos(
            x,
            good.transpose(
                "batch", "time", "lead_time", "variable", "width", "height", "face"
            ),
        )
    with pytest.raises(ValueError, match="lacks"):
        atmos(
            x, good.assign_coords(lead_time=good["lead_time"] + np.timedelta64(6, "h"))
        )
    with pytest.raises(ValueError):
        atmos(x, good.assign_coords(variable=["z500"]))
    with pytest.raises(ValueError, match="valid for times"):
        atmos(x, good.assign_coords(time=good["time"] + np.timedelta64(1, "D")))
    with pytest.raises(ValueError, match="must have dimensions"):
        atmos(x, good.isel(lead_time=0, drop=True))
    # No time coordinate on the forcing is fine; it is read from the state.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        atmos(x, good.isel(time=0, drop=True))


def test_dlesym_ocean_requires_even_atmosphere_spacing():
    model = build_dlesym_model("cpu", nside=8)
    model.atmos_output_times = np.array([6, 12, 24, 48, 72, 96], dtype="timedelta64[h]")
    with pytest.raises(ValueError, match="evenly spaced"):
        DLESyMOcean(model)


def test_dlesym_ocean_requires_aligned_output_times():
    model = build_dlesym_model("cpu", nside=8)
    model.ocean_output_times = np.array([24, 96], dtype="timedelta64[h]")
    with pytest.raises(ValueError, match="one offset"):
        DLESyMOcean(model)


def test_dlesym_split_under_pipeline():
    model = _split_model()
    source = HEALPixFill(model)
    atmos_io, ocean_io = KVBackend(), KVBackend()
    Pipeline(
        _coupled(model, source),
        output={"atmos": atmos_io, "ocean": ocean_io},
        device="cpu",
    ).run(work_items([np.datetime64("2020-06-01T00")], np.timedelta64(192, "h")))
    atmos_ds, ocean_ds = atmos_io.to_xarray(), ocean_io.to_xarray()
    assert hours(atmos_ds) == list(range(0, 193, 6))
    assert hours(ocean_ds) == [0, 48, 96, 144, 192]
    assert set(atmos_ds.data_vars) == set(dlesym_src._ATMOS_VARIABLES)
    assert set(ocean_ds.data_vars) == {"sst"}
    assert atmos_ds["ws10m"].dims == ("time", "lead_time", "face", "height", "width")
    assert not np.isnan(ocean_ds["sst"].values).any()


def test_dlesym_latlon_to_native():
    model = build_dlesym_model("cpu", nside=8, type="ll")
    signature = model.input_coords()
    variables = list(signature["variable"].values)
    time = np.array([np.datetime64("2020-06-01T00"), np.datetime64("2020-06-01T06")])
    coords = coord_array(
        ("time", "variable", "lat", "lon"),
        {"time": time, "variable": variables, "lat": model.lat, "lon": model.lon},
    )
    values = (
        np.random.default_rng(0)
        .standard_normal((len(time), len(variables), len(model.lat), len(model.lon)))
        .astype(np.float32)
    )
    values[:, variables.index("sst"), :10] = np.nan  # land
    x = from_torch(torch.from_numpy(values), coords)
    native = model.to_native(x)
    assert native.dims == ("time", "variable", "face", "height", "width")
    assert list(native["variable"].values) == (
        dlesym_src._ATMOS_VARIABLES + dlesym_src._OCEAN_VARIABLES
    )
    assert native.sizes["face"] == 12 and native.sizes["height"] == 8
    assert not np.isnan(native.sel(variable="sst").values).any()
    np.testing.assert_array_equal(native["time"].values, time)
    # The native array fits the split components' and fused model's input grid.
    handshake_metadata(
        native, DLESyM.input_coords(model), ("nside", "ordering", "origin")
    )


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

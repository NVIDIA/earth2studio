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


import asyncio
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
from earth2studio.models.px import DLESyM, DLESyMAtmos, DLESyMOcean
from earth2studio.models.px.dlesym import DLESyMInputSource
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
                    LatLonGrid=lambda lat, lon: SimpleNamespace(
                        shape=(len(lat), len(lon))
                    )
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


class DiagnosticAtmosModel(CoupledAtmosModel):
    """Coupled atmosphere core with one extra, diagnostic output channel.

    The channel sweeps negative to positive values so both branches of the
    ``tp06`` inverse transform are exercised.
    """

    def forward(self, in_list):
        out = super().forward(in_list)
        sweep = torch.linspace(-2.0, 2.0, out.shape[-1], device=out.device)
        return torch.cat([out, out[:, :, :, :1] * 0 + sweep], dim=3)


def build_dlesym_model(
    device,
    nside=64,
    atmos_model=None,
    ocean_model=None,
    scaled=False,
    use_cln=False,
    atmos_diagnostic_variables=None,
):
    """Build a DLESyM prognostic model with mock atmos/ocean models.

    Parameters
    ----------
    device : torch.device
        The device to build the model on.
    nside : int
        The nside of the HEALPix grid.
    atmos_model, ocean_model : torch.nn.Module, optional
        Component cores, by default the constant mocks.
    scaled : bool
        Draw nontrivial normalization constants instead of zero center, unit scale.
    use_cln : bool
        Sample conditional layer norm noise (three conditions) for each core call.
    atmos_diagnostic_variables : list[str], optional
        Atmospheric diagnostic output variables.

    Returns
    -------
    model : DLESyM
        The DLESyM model.
    """
    # A local generator, not the global np.random state: this helper is shared by
    # the conformance test, and drawing from the global generator made whether the
    # checker's P16 aliasing probe actually observed a value change depend on
    # whatever earlier test in the suite last touched np.random, i.e. test order.
    rng = np.random.default_rng(0)
    hpx_lat = rng.standard_normal((12, nside, nside))
    hpx_lon = rng.standard_normal((12, nside, nside))
    n_vars = 9 + len(atmos_diagnostic_variables or [])
    center = np.zeros((1, 1, 1, n_vars, 1, 1, 1))
    scale = np.ones((1, 1, 1, n_vars, 1, 1, 1))
    if scaled:
        center = rng.uniform(-1.0, 1.0, center.shape)
        scale = rng.uniform(0.5, 2.0, scale.shape)
    atmos_constants = rng.standard_normal((12, 2, nside, nside))
    ocean_constants = rng.standard_normal((12, 2, nside, nside))

    return DLESyM(
        atmos_model=atmos_model if atmos_model is not None else PhooAtmosModel(),
        ocean_model=ocean_model if ocean_model is not None else PhooOceanModel(),
        hpx_lat=hpx_lat,
        hpx_lon=hpx_lon,
        nside=nside,
        center=center,
        scale=scale,
        atmos_constants=atmos_constants,
        ocean_constants=ocean_constants,
        atmos_input_times=dlesym_src._ATMOS_INPUT_TIMES,
        ocean_input_times=dlesym_src._OCEAN_INPUT_TIMES,
        atmos_output_times=dlesym_src._ATMOS_OUTPUT_TIMES,
        ocean_output_times=dlesym_src._OCEAN_OUTPUT_TIMES,
        atmos_variables=dlesym_src._ATMOS_VARIABLES,
        ocean_variables=dlesym_src._OCEAN_VARIABLES,
        atmos_coupling_variables=dlesym_src._ATMOS_COUPLING_VARIABLES,
        ocean_coupling_variables=dlesym_src._OCEAN_COUPLING_VARIABLES,
        atmos_diagnostic_variables=atmos_diagnostic_variables,
        use_cln=use_cln,
        condition_shape=3 if use_cln else None,
    ).to(device)


def build_components(
    model, atmos_cls=DLESyMAtmos, ocean_cls=DLESyMOcean, atmos_extra=None, **overrides
):
    """Build standalone components equivalent to a fused mock ``model``.

    Each component gets its own core and the slices of the fused model's
    normalization that it uses, as ``load_model`` reads them from a package.
    ``overrides`` replace constructor arguments of both components and
    ``atmos_extra`` adds arguments of an atmosphere subclass.
    """
    outputs = (
        model.atmos_variables
        + model.atmos_diagnostic_variables
        + model.ocean_variables
        + model.ocean_diagnostic_variables
    )
    center = model.center.reshape(-1).cpu().numpy()
    scale = model.scale.reshape(-1).cpu().numpy()

    def stats(variables):
        index = [outputs.index(v) for v in variables]
        return {"center": center[index], "scale": scale[index]}

    def coupling(variables):
        index = [outputs.index(v) for v in variables]
        return {"coupling_center": center[index], "coupling_scale": scale[index]}

    shared = dict(
        hpx_lat=model.hpx_lat,
        hpx_lon=model.hpx_lon,
        nside=model.nside,
        use_cln=model.use_cln,
        condition_shape=model.condition_shape,
    )
    atmos = dict(
        core=model.atmos_model,
        **stats(model.atmos_variables + model.atmos_diagnostic_variables),
        **coupling(model.atmos_coupling_variables),
        constants=model.atmos_constants.cpu().numpy(),
        input_times=model.atmos_input_times,
        output_times=model.atmos_output_times,
        variables=model.atmos_variables,
        coupling_variables=model.atmos_coupling_variables,
        diagnostic_variables=model.atmos_diagnostic_variables,
        **shared,
    )
    ocean = dict(
        core=model.ocean_model,
        **stats(model.ocean_variables + model.ocean_diagnostic_variables),
        **coupling(model.ocean_coupling_variables),
        constants=model.ocean_constants.cpu().numpy(),
        input_times=model.ocean_input_times,
        output_times=model.ocean_output_times,
        atmos_output_times=model.atmos_output_times,
        variables=model.ocean_variables,
        coupling_variables=model.ocean_coupling_variables,
        diagnostic_variables=model.ocean_diagnostic_variables,
        **shared,
    )
    atmos.update({k: v for k, v in overrides.items() if k in atmos})
    ocean.update({k: v for k, v in overrides.items() if k in ocean})
    return atmos_cls(**atmos, **(atmos_extra or {})), ocean_cls(**ocean)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_forward(device, batch_size):
    """Test DLESyM forward pass with mock models."""

    nside = 64
    model = build_dlesym_model(device)
    spatial_dims = (12, nside, nside)

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
    assert field.attrs["ordering"] == "xy"
    assert field.attrs["origin"] == "north"
    assert field.attrs["clockwise"] is True
    assert field.attrs["earth2studio_grid_id"] == "healpix-l6-xy-north-clockwise-face"
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
    output = model(field)
    output_coords = output.coords
    assert output.name == field.name and output.encoding == field.encoding
    expected_coords = model.output_coords(field)
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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_iterator(device, batch_size):
    """Test DLESyM iterator functionality."""

    nside = 64
    model = build_dlesym_model(device, nside=nside)
    spatial_dims = (12, nside, nside)
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
    events = []

    def front(state):
        assert state.dims[0] == "member" and "member" not in state.coords
        events.append("front")
        state.data[...] += 1
        return state

    def rear(state):
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
    model = build_dlesym_model("cpu", nside=8)
    check_prognostic_contract(model)


class HEALPixFill:
    """Data source of deterministic fields on DLESyM's native grid.

    Each variable is offset by its index in the fused input, drifts with time and
    carries a fixed spatial pattern, so wrong frames or variables are detectable.
    """

    def __init__(self, model):
        signature = model.input_coords()
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


def _coupled_model():
    return build_dlesym_model(
        "cpu",
        nside=8,
        atmos_model=CoupledAtmosModel(),
        ocean_model=CoupledOceanModel(),
        scaled=True,
    )


def test_dlesym_component_declarations():
    model = _coupled_model()
    atmos, ocean = build_components(model)
    # Standalone models: each holds its own core and no fused model.
    assert atmos.core is model.atmos_model and ocean.core is model.ocean_model
    for component in (atmos, ocean):
        assert not any(isinstance(m, DLESyM) for m in component.modules())
        assert [name for name, _ in component.named_children()] == ["core"]

    assert hours(atmos.input_coords()) == [-18, -12, -6, 0]
    assert list(atmos.input_coords()["variable"].values) == dlesym_src._ATMOS_VARIABLES
    assert hours(atmos.forcing_coords()) == [0]
    assert list(atmos.forcing_coords()["variable"].values) == ["sst"]
    assert hours(atmos.output_coords(atmos.input_coords())) == list(range(6, 97, 6))

    assert hours(ocean.input_coords()) == [-48, 0]
    assert list(ocean.input_coords()["variable"].values) == ["sst"]
    # Window means valid at the ocean's output times, over the atmosphere frames
    # of the same cycle: the right-closed (T - 48h, T] is 6h..48h for T = 48h.
    assert hours(ocean.forcing_coords()) == [48, 96]
    assert list(ocean.forcing_coords()["variable"].values) == [
        "z1000:mean:48h",
        "ws10m:mean:48h",
    ]
    assert hours(ocean.output_coords(ocean.input_coords())) == [48, 96]
    native = model.input_coords()
    for component in (atmos, ocean):
        for signature in (component.input_coords(), component.forcing_coords()):
            assert signature.data.nbytes == 0
            assert signature.dims == native.dims
            assert signature.attrs["nside"] == native.attrs["nside"]


def test_dlesym_component_conformance():
    atmos, ocean = build_components(build_dlesym_model("cpu", nside=8))
    check_prognostic_contract(atmos)
    check_prognostic_contract(ocean)


def test_dlesym_component_rejects_mismatched_normalization():
    model = build_dlesym_model("cpu", nside=8)
    with pytest.raises(ValueError, match="'coupling_center' has 3 entries"):
        build_components(model, coupling_center=np.zeros(3))
    with pytest.raises(ValueError, match="condition_shape"):
        build_components(model, use_cln=True, condition_shape=None)


def _coupled(components, source):
    atmos, ocean = components
    return CoupledRunner(
        {"atmos": atmos, "ocean": ocean},
        [
            Connector("ocean", "atmos", time="hold"),
            Connector("atmos", "ocean"),
        ],
        sources={"atmos": source, "ocean": source},
    )


def _streams(runner, item):
    streams: dict[str, list[xr.DataArray]] = {}
    for step in runner.run_item(item):
        for name, x in step.items():
            streams.setdefault(name, []).append(x)
    return streams


def test_dlesym_coupled_schedule():
    model = _coupled_model()
    runner = _coupled(build_components(model), HEALPixFill(model))
    plan = runner.describe(np.timedelta64(192, "h"))
    assert "atmos  stateful    every 96h" in plan
    assert "ocean  stateful    every 96h" in plan
    assert "ocean -> atmos forcing[0] (hold): sst" in plan
    assert "atmos -> ocean forcing[0] (exact): z1000:mean:48h, ws10m:mean:48h" in plan
    # The atmosphere initializes from the ocean's initial SST and the ocean
    # reduces the atmosphere's first cycle. The next atmosphere cycle is forced
    # at the previous cycle's lead times, held from the ocean's earlier frames:
    # the +96h frame it uses is the ocean's +96h output.
    assert "atmos  initialize" in plan and "[forcing[0] <- ocean at +0h]" in plan
    assert "ocean  initialize" in plan
    assert "forcing[0] <- atmos at +48h, +96h from +6h..+96h" in plan
    assert "atmos  step" in plan and "+96h from +0h..+96h]" in plan
    assert "forcing[0] <- atmos at +144h, +192h from +102h..+192h" in plan


def _assert_coupled_reproduces_fused(model, components, source, variables):
    """Three coupled cycles of ``components`` match the fused ``model``."""
    item = WorkItem(np.datetime64("2020-06-01T00"), np.timedelta64(288, "h"))
    fused = [
        step["forecast"] for step in PrognosticRunner(model, source).run_item(item)
    ]
    streams = _streams(_coupled(components, source), item)

    # Three cycles in every stream, after the initial condition where the
    # stream publishes one: a model whose public input variables differ from its
    # outputs, such as ``ttr`` converted to ``rlut``, publishes forecasts only.
    for stream in (fused, streams["atmos"], streams["ocean"]):
        assert len(stream) in (3, 4)
    if len(fused) == 4 and len(streams["atmos"]) == 4:
        xr.testing.assert_allclose(
            streams["atmos"][0].drop_vars("lead_time"),
            fused[0].sel(variable=variables).drop_vars("lead_time"),
        )
    assert len(streams["ocean"]) == 4
    if len(fused) == 4:
        xr.testing.assert_allclose(
            streams["ocean"][0].drop_vars("lead_time"),
            fused[0].sel(variable=["sst"]).drop_vars("lead_time"),
        )
    fused = fused[-3:]
    for k in range(3):
        atmos = model.retrieve_valid_atmos_outputs(fused[k])
        ocean = model.retrieve_valid_ocean_outputs(fused[k])
        xr.testing.assert_allclose(streams["atmos"][k - 3], atmos, rtol=1e-5, atol=1e-5)
        xr.testing.assert_allclose(streams["ocean"][k - 3], ocean, rtol=1e-5, atol=1e-5)
    # The coupling mattered: the ocean's SST changed the fused result.
    assert not np.allclose(
        fused[0].sel(variable="z500").values, fused[2].sel(variable="z500").values
    )


def test_dlesym_coupled_reproduces_fused():
    model = _coupled_model()
    _assert_coupled_reproduces_fused(
        model, build_components(model), HEALPixFill(model), dlesym_src._ATMOS_VARIABLES
    )


def test_dlesym_tp06_in_meters():
    """The log-transformed precipitation diagnostic is returned in meters."""
    model = build_dlesym_model(
        "cpu",
        nside=8,
        atmos_model=DiagnosticAtmosModel(),
        ocean_model=CoupledOceanModel(),
        scaled=True,
        atmos_diagnostic_variables=["tp06"],
    )
    source = HEALPixFill(model)
    item = WorkItem(np.datetime64("2020-06-01T00"), np.timedelta64(96, "h"))
    fused = [s["forecast"] for s in PrognosticRunner(model, source).run_item(item)][1]
    tp06 = fused.sel(variable="tp06").values
    index = list(fused["variable"].values).index("tp06")
    log = (
        np.linspace(-2.0, 2.0, 8, dtype=np.float32)
        * model.scale.reshape(-1)[index].item()
        + model.center.reshape(-1)[index].item()
    )
    expected = np.expm1(np.clip(log, 0, None)) * 1e-8
    assert (expected == 0).any() and (expected > 0).any()
    np.testing.assert_allclose(
        tp06, np.broadcast_to(expected, tp06.shape), rtol=1e-5, atol=0
    )
    # The diagnostic is not an input, and the atmosphere component converts it too.
    assert "tp06" not in model.input_coords()["variable"].values
    atmos = _streams(_coupled(build_components(model), source), item)["atmos"][1]
    xr.testing.assert_allclose(
        atmos, model.retrieve_valid_atmos_outputs(fused), rtol=1e-5, atol=0
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


def test_dlesym_component_stochastic_conformance():
    atmos, ocean = build_components(_noisy_model())
    assert atmos.stochastic and ocean.stochastic
    check_prognostic_contract(atmos)
    check_prognostic_contract(ocean)


def test_dlesym_components_draw_independent_seeded_noise():
    model = _noisy_model()
    source = HEALPixFill(model)
    item = WorkItem(np.datetime64("2020-06-01T00"), np.timedelta64(192, "h"))

    def run(seed):
        runner = _coupled(build_components(model), source)
        runner.seed = seed
        return _streams(runner, item)

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


def test_dlesym_component_step_replays_from_state():
    model = _noisy_model()
    atmos, ocean = build_components(model)
    source = HEALPixFill(model)
    t0 = np.datetime64("2020-06-01T00")
    runner = _coupled((atmos, ocean), source)
    chunks = [*runner.run_item(WorkItem(t0, np.timedelta64(96, "h")))]
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


def test_dlesym_component_rejects_mismatched_forcing():
    atmos, _ = build_components(_coupled_model())
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
    uneven = np.array([6, 12, 24, 48, 72, 96], dtype="timedelta64[h]")
    with pytest.raises(ValueError, match="evenly spaced"):
        build_components(model, atmos_output_times=uneven)


def test_dlesym_ocean_requires_aligned_output_times():
    model = build_dlesym_model("cpu", nside=8)
    model.ocean_output_times = np.array([24, 96], dtype="timedelta64[h]")
    with pytest.raises(ValueError, match="one offset"):
        build_components(model)


def test_dlesym_coupled_under_pipeline():
    model = _coupled_model()
    source = HEALPixFill(model)
    atmos_io, ocean_io = KVBackend(), KVBackend()
    Pipeline(
        _coupled(build_components(model), source),
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


class LatLonFill:
    """Lat/lon data source of spatially uniform base variables.

    Each field is a constant set by its variable and drifting with time, so the
    regridded and derived values are known exactly; ``sst`` has a band of NaNs
    standing in for land. ``t2m`` is the latitude and ``tcwv`` the cosine of the
    longitude, which pin down the orientation of the regridding. The variables
    of every request are recorded.
    """

    VALUES = {"u10m": 3.0, "v10m": 4.0, "z300": 9000.0, "z700": 3000.0, "sst": 290.0}

    def __init__(self, nlat=37, nlon=72, ascending=False):
        self.lat = np.linspace(90, -90, nlat)
        if ascending:
            self.lat = self.lat[::-1].copy()
        self.lon = np.linspace(0, 360, nlon, endpoint=False)
        self.requests = []

    def __call__(self, time, variable):
        time = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variable = [str(v) for v in np.atleast_1d(np.asarray(variable))]
        self.requests.append(variable)
        hours = (time - np.datetime64("2020-01-01")) / np.timedelta64(1, "h")
        base = np.array([self.VALUES.get(v, 1.0) for v in variable], dtype=np.float32)
        values = (
            base[None, :, None, None]
            + 1e-3 * hours.astype(np.float32)[:, None, None, None]
            + np.zeros((len(self.lat), len(self.lon)), dtype=np.float32)
        )
        if "sst" in variable:
            values[:, variable.index("sst"), 5:10, 3:20] = np.nan
        if "t2m" in variable:
            values[:, variable.index("t2m")] = self.lat[:, None]
        if "tcwv" in variable:
            values[:, variable.index("tcwv")] = np.cos(np.deg2rad(self.lon))
        return xr.DataArray(
            values.astype(np.float32),
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                "lat": self.lat,
                "lon": self.lon,
            },
        )


@pytest.mark.parametrize("ascending", [False, True])
def test_dlesym_input_source(ascending):
    model = build_dlesym_model("cpu", nside=8)
    fill = LatLonFill(ascending=ascending)
    source = DLESyMInputSource(fill, nside=8)
    time = np.array(["2020-01-01T00", "2020-01-01T06"], dtype="datetime64[ns]")
    variables = ["ws10m", "z500", "tau300-700", "sst", "t2m", "tcwv"]
    native = source(time, variables)

    # Only the base variables the request needs are fetched, once.
    assert fill.requests == [
        ["u10m", "v10m", "z500", "z300", "z700", "sst", "t2m", "tcwv"]
    ]
    assert native.dims == ("time", "variable", "face", "height", "width")
    assert list(native["variable"].values) == variables
    np.testing.assert_array_equal(native["time"].values, time)
    assert native.shape == (2, 6, 12, 8, 8)
    # The native array fits the fused model's and the components' input grid.
    handshake_metadata(native, model.input_coords(), ("nside", "ordering", "origin"))

    if not EARTH2GRID_AVAILABLE:
        return
    # Uniform fields regrid to themselves; derived variables and the SST fill
    # are computed before regridding. The second time drifts by 6e-3.
    drift = np.array([0.0, 6e-3])[:, None, None, None]

    def check(name, expected, **tolerance):
        actual = native.sel(variable=name).values
        expected = np.broadcast_to(expected, actual.shape)
        np.testing.assert_allclose(actual, expected, **tolerance)

    check("ws10m", np.hypot(3 + drift, 4 + drift), rtol=1e-5)
    check("z500", 1.0 + drift, rtol=1e-5)
    check("tau300-700", 6000.0, rtol=1e-6)
    check("sst", 290.0 + drift, rtol=1e-6)
    # Fields that vary in space land on the HEALPix pixels at their own
    # latitude and longitude, whichever way the source orders its latitudes.
    grid = source._grid
    hpx_lat = np.asarray(grid.lat).reshape(12, 8, 8)
    hpx_lon = np.asarray(grid.lon).reshape(12, 8, 8)
    check("t2m", hpx_lat, atol=1e-3)
    check("tcwv", np.cos(np.deg2rad(hpx_lon)), atol=5e-3)


def test_dlesym_input_source_fetch_and_validation():
    model = build_dlesym_model("cpu", nside=8)
    time = np.array(["2020-01-01T00"], dtype="datetime64[ns]")

    class AsyncFill(LatLonFill):
        async def fetch(self, time, variable):
            self.awaited = True
            return self(time, variable)

    # fetch awaits an async source and falls back to calling a synchronous one.
    for fill in (LatLonFill(), AsyncFill()):
        source = DLESyMInputSource(fill, nside=8)
        fetched = asyncio.run(source.fetch(time, ["sst", "ws10m"]))
        xr.testing.assert_identical(fetched, source(time, ["sst", "ws10m"]))
        assert getattr(fill, "awaited", False) is isinstance(fill, AsyncFill)

    with pytest.raises(ValueError, match="regular lat/lon grid"):
        DLESyMInputSource(HEALPixFill(model), nside=8)(time, ["sst"])


def test_dlesym_default_sources():
    model = build_dlesym_model("cpu", nside=8)
    # The fused model and its components recommend ARCO ERA5 prepared as native
    # inputs; a component leaves its forcing slot to a connector.
    recommended = model.default_sources()
    assert isinstance(recommended, DLESyMInputSource) and recommended.nside == 8
    assert type(recommended.source).__name__ == "ARCO_ERA5"
    for component in build_components(model):
        history, forcing = component.default_sources()
        assert isinstance(history, DLESyMInputSource) and history.nside == 8
        assert forcing is None


def test_dlesym_runs_from_a_latlon_source():
    model = _coupled_model()
    fill = LatLonFill()
    source = DLESyMInputSource(fill, nside=8)
    item = WorkItem(np.datetime64("2020-06-01T00"), np.timedelta64(96, "h"))

    fused = [s["forecast"] for s in PrognosticRunner(model, source).run_item(item)]
    assert len(fused) == 2 and fused[1].sizes["lead_time"] == 16
    assert not np.isnan(fused[0].sel(variable="sst").values).any()

    fill.requests.clear()
    streams = _streams(_coupled(build_components(model), source), item)
    assert len(streams["atmos"]) == 2 and len(streams["ocean"]) == 2
    assert streams["atmos"][1].sizes["lead_time"] == 16
    assert not np.isnan(streams["ocean"][0].values).any()
    # Each component fetches only the base variables of its own history.
    atmos_base = {"z500", "z300", "z700", "z1000", "t2m", "tcwv", "t850", "z250"}
    atmos_base |= {"u10m", "v10m"}
    assert {frozenset(request) for request in fill.requests} == {
        frozenset(atmos_base),
        frozenset({"sst"}),
    }
    if EARTH2GRID_AVAILABLE:
        xr.testing.assert_allclose(
            streams["atmos"][1],
            model.retrieve_valid_atmos_outputs(fused[1]),
            rtol=1e-5,
            atol=1e-5,
        )


class FakePackage:
    """A package directory on disk, standing in for a downloaded model package."""

    def __init__(self, root):
        self.root = root
        self.resolved = []

    def resolve(self, file_path):
        self.resolved.append(file_path)
        return str(self.root / file_path)


def write_package(root, config, version_path=None, nside=8):
    """Write a DLESyM package's config and constant fields under ``root``."""
    rng = np.random.default_rng(1)
    for name in ("hpx_lat", "hpx_lon", "land_sea_mask", "topography"):
        np.save(root / f"{name}.npy", rng.standard_normal((12, nside, nside)))
    directory = root
    if version_path is not None:
        directory = root / version_path
        directory.mkdir(parents=True)
        (root / "config.yaml").write_text(
            f"versions:\n  v1.1:\n    path: {version_path}\n"
        )
    (directory / "config.yaml").write_text(config)
    return FakePackage(root)


def mock_checkpoints(
    monkeypatch, atmos_core=CoupledAtmosModel, ocean_core=CoupledOceanModel
):
    """Load mock cores in place of package checkpoints; returns the loaded paths."""
    omegaconf = pytest.importorskip("omegaconf")
    loaded = []

    def from_checkpoint(path):
        loaded.append(path)
        return atmos_core() if "atmos_model" in path else ocean_core()

    monkeypatch.setattr(dlesym_src, "OmegaConf", omegaconf.OmegaConf)
    monkeypatch.setattr(
        dlesym_src, "Module", SimpleNamespace(from_checkpoint=from_checkpoint)
    )
    return loaded


_PACKAGE_CONFIG = """
io:
  atmos_variables: ["z500", "tau300-700", "z1000", "t2m", "tcwv", "t850", "z250", "ws10m", "ttr-3h"]
  ocean_variables: ["sst"]
  atmos_coupling_variables: ["sst"]
  ocean_coupling_variables: ["z1000", "ws10m"]
  atmos_input_times: [-18, -12, -6, 0]
  ocean_input_times: [-48, 0]
  atmos_output_times: [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
  ocean_output_times: [48, 96]
  atmos_constants: [land_sea_mask, topography]
  ocean_constants: [land_sea_mask]
models:
  atmos_model_checkpoints: [atmos_model_0.mdlus]
  ocean_model_checkpoints: [ocean_model_0.mdlus]
data:
  nside: 8
  scaling:
    z500: {mean: 0.5, std: 1.5}
    tau300-700: {mean: -0.5, std: 0.5}
    z1000: {mean: 0.25, std: 2.0}
    t2m: {mean: 1.0, std: 0.75}
    tcwv: {mean: -1.0, std: 1.25}
    t850: {mean: 0.1, std: 0.9}
    z250: {mean: -0.2, std: 1.1}
    ws10m: {mean: 0.3, std: 0.6}
    ttr-3h: {mean: -0.7, std: 1.7}
    sst: {mean: 0.8, std: 1.3}
"""


def test_dlesym_components_load_from_package(tmp_path, monkeypatch):
    """Components load their own core and constants from the fused package."""
    package = write_package(tmp_path, _PACKAGE_CONFIG, "checkpoints/v1.1")
    loaded = mock_checkpoints(monkeypatch)
    root = "checkpoints/v1.1"

    model = DLESyM.load_model(package)
    assert [p.removeprefix(str(tmp_path) + "/") for p in loaded] == [
        f"{root}/atmos/atmos_model_0.mdlus",
        f"{root}/ocean/ocean_model_0.mdlus",
    ]
    # Package names are mapped onto the lexicon.
    assert model.atmos_variables[-1] == "ttr03"

    loaded.clear()
    package.resolved.clear()
    ocean = DLESyMOcean.load_model(package)
    # The ocean alone loads neither the atmosphere core nor its constants.
    assert [p.removeprefix(str(tmp_path) + "/") for p in loaded] == [
        f"{root}/ocean/ocean_model_0.mdlus"
    ]
    assert "topography.npy" not in package.resolved
    assert ocean.constants.shape == (12, 1, 8, 8)
    # Its forcing is normalized with the atmosphere's output constants.
    assert ocean.coupling_variables == ["z1000", "ws10m"]
    np.testing.assert_allclose(ocean.coupling_center.numpy(), [0.25, 0.3])
    np.testing.assert_allclose(ocean.coupling_scale.numpy(), [2.0, 0.6])
    np.testing.assert_allclose(ocean.center.numpy(), [0.8])

    loaded.clear()
    atmos = DLESyMAtmos.load_model(package)
    assert [p.removeprefix(str(tmp_path) + "/") for p in loaded] == [
        f"{root}/atmos/atmos_model_0.mdlus"
    ]
    assert atmos.variables == model.atmos_variables
    np.testing.assert_allclose(atmos.coupling_center.numpy(), [0.8])
    np.testing.assert_allclose(atmos.coupling_scale.numpy(), [1.3])
    assert isinstance(atmos, DLESyMAtmos) and isinstance(ocean, DLESyMOcean)

    _assert_coupled_reproduces_fused(
        model, (atmos, ocean), HEALPixFill(model), model.atmos_variables
    )

    with pytest.raises(ValueError, match="single atmos checkpoint"):
        DLESyMAtmos.load_model(package, atmos_model_idx=1)
    with pytest.raises(ValueError, match="Unknown DLESyM checkpoint version"):
        DLESyMOcean.load_model(package, version="v9")


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

    # The precipitation diagnostic uses the lexicon name and physical units (m)
    assert "tp06" in output_coords["variable"]
    tp06, _ = (
        model.retrieve_valid_atmos_outputs(output).sel(variable="tp06").e2s.to_torch()
    )
    assert tp06.min() >= 0


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_dlesym_component_package(device):
    """The packaged components run one coupled cycle on the native grid."""
    torch.cuda.empty_cache()
    package = DLESyMAtmos.load_default_package()
    atmos = DLESyMAtmos.load_model(package).to(device)
    ocean = DLESyMOcean.load_model(package).to(device)
    fused = SimpleNamespace(
        nside=atmos.nside,
        input_coords=lambda: coord_array_like(
            atmos.input_coords(),
            {"variable": np.array(atmos.variables + ocean.variables)},
        ),
    )
    runner = _coupled((atmos, ocean), HEALPixFill(fused))
    streams = _streams(
        runner, WorkItem(np.datetime64("2020-06-01T00"), np.timedelta64(96, "h"))
    )
    assert hours(streams["atmos"][1]) == list(range(6, 97, 6))
    assert hours(streams["ocean"][1]) == [48, 96]
    assert "tp06" in streams["atmos"][1]["variable"].values
    assert not np.isnan(streams["ocean"][1].values).any()

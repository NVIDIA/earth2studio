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

import sys
from copy import deepcopy
from datetime import datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import xarray as xr

from earth2studio.coupling import CoupledRunner
from earth2studio.data import fetch_dataframe
from earth2studio.io import ZarrBackend
from earth2studio.models.da.aida import (
    AIDA,
    IASI_CHANNELS,
    AIDAWeatherMesh5C,
    _pack,
    _reports,
    _schema,
)
from earth2studio.models.px.weathermesh5c import WeatherMesh5C, _iterator
from earth2studio.run import Pipeline, work_items
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyError,
    OptionalDependencyFailure,
)


@pytest.fixture
def core_model(monkeypatch):
    from weathermesh5c.inference import SURFACE_VARIABLES, Inference, ProductTransform

    mesh = SimpleNamespace(lats=[1.0, 0.0], lons=np.arange(3.0), levels=[500])

    class FakeCore(torch.nn.Module):
        def move_assets(self, device):
            pass

    core = FakeCore()
    core.anl_encoders = core.decoders = [SimpleNamespace(mesh=mesh)]
    core.obs_encoders = [SimpleNamespace(meshes=[SimpleNamespace(string_id="sfc")])]
    monkeypatch.setattr(ProductTransform, "from_model", lambda model: None)

    class FakeNumerics:
        def prepare_sst(self, values, lat, lon):
            return values

        def prepare_background(self, fields, ocean):
            return np.asarray(fields.surface["167_2t"].mean() + ocean.mean())

        def prepare_observations(self, source, hours, valid_time, *, max_observations):
            values = np.concatenate([values for _, values in hours])
            return torch.as_tensor(np.nanmean(values[:, 6]), device=self.device)

        def encode(self, background, time):
            return torch.as_tensor(background, device=self.device).reshape(1, 1).float()

        def assimilate(self, background, time, cycles):
            return self.encode(background, time) + sum(
                obs["adpsfc"].sum() for obs in cycles.values()
            )

        def step(self, latent):
            return latent + 0.125

        def fork(self, latent, *, hour, seed):
            generator = torch.Generator(device=self.device).manual_seed(seed)
            return latent + torch.randn(
                latent.shape, device=self.device, generator=generator
            )

        def decode(self, latent, *, pressure=True):
            count = len(self.variables if pressure else SURFACE_VARIABLES)
            return latent[:, None, :, None].expand(-1, count, 2, 3).cpu().numpy().copy()

    for name, method in vars(FakeNumerics).items():
        if callable(method):
            monkeypatch.setattr(Inference, name, method)
    return core


@pytest.fixture
def model(core_model):
    return WeatherMesh5C(core_model, variables=("t2m", "u10m", "tp:sum:3h"))


@pytest.fixture
def fields(model, monkeypatch):
    monkeypatch.setattr(
        model._runtime,
        "prepare_sst",
        lambda values, lat, lon: np.full((2, 3), values[0, 0], dtype=np.float32),
    )
    output = []
    for signature in model.input_coords():
        layout = coord_array_like(
            signature,
            {"batch": [0], "time": np.array(["2026-10-28T09"], dtype="datetime64[ns]")},
        )
        values = np.zeros(layout.shape, dtype=np.float32)
        for i, variable in enumerate(layout.coords["variable"].values):
            value = 0.001 if variable.startswith("q") else 280.0
            if variable == "msl":
                value = 101300
            if variable == "lsm":
                value = 0
            values[..., i, :, :] = value
        output.append(from_torch(torch.from_numpy(values), layout))
    return tuple(output)


def standard_atmosphere(signature, time="2026-01-01T00", device="cpu"):
    """Horizontally uniform standard-atmosphere fields for a declared input slot."""
    layout = coord_array_like(
        signature, {"batch": [0], "time": np.array([time], dtype="datetime64[ns]")}
    )
    values = np.zeros(layout.shape, dtype=np.float32)
    surface = {"t2m": 288.0, "d2m": 283.0, "skt": 288.0, "sst": 290.0, "msl": 101325.0}
    for i, name in enumerate(map(str, layout.coords["variable"].values)):
        if name[0] in "ztq" and name[1:].isdigit():
            level = int(name[1:])
            height = 44330.8 * (1 - (level / 1013.25) ** 0.190263)
            value = {
                "z": 9.80665 * height,
                "t": max(216.65, 288.15 - 0.0065 * height),
                "q": 0.01 * (level / 1000) ** 3,
            }[name[0]]
        else:
            value = surface.get(name, 0.0)
        values[..., i, :, :] = value
    return from_torch(torch.from_numpy(values).to(device), layout)


def test_missing_runtime_reports_installation_extra(monkeypatch):
    from earth2studio.models.da import aida

    monkeypatch.setattr(OptionalDependencyFailure, "failures", {})
    monkeypatch.setitem(sys.modules, "weathermesh5c.inference", None)
    spec = spec_from_file_location("_aida_without_runtime", aida.__file__)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    for model_type in (module.AIDA, module.AIDAWeatherMesh5C):
        with pytest.raises(OptionalDependencyError):
            model_type(torch.nn.Identity())
        with pytest.raises(OptionalDependencyError):
            model_type.load_model(None)


def inputs(fields, steps=3):
    return tuple(
        x.assign_coords(lead_time=[np.timedelta64(-3 * steps, "h")]) for x in fields
    )


def observations():
    return pd.DataFrame(
        {
            **{field: np.nan for field in _schema("adpsfc") if field != "variable"},
            "time": np.array(
                ["2026-10-28T03", "2026-10-28T06", "2026-10-28T09"],
                dtype="datetime64[ns]",
            ),
            "lat": 1.0,
            "lon": 359.0,
            "lon_native": -1.0,
            "class": "ADPSFC",
            "subset": 1,
            "observation_t": 285.0,
            "observation_u": 2.0,
            "observation_v": 3.0,
        }
    )


def test_aida_analysis_and_observation_effect(core_model, fields):
    aida = AIDA(core_model, variables=("t2m",))
    x, sst = inputs(fields)
    frame = observations()
    before = frame.copy(deep=True)
    np.testing.assert_array_equal(_reports("adpsfc", frame)["lon"], [-1.0] * 3)
    y = aida(x, sst, surface=frame)
    assert y.lead_time.values[0] == np.timedelta64(0, "h")
    assert np.isfinite(y.values).all()
    changed = frame.copy()
    changed["observation_t"] += 5
    assert not np.array_equal(y, aida(x, sst, surface=changed))
    pd.testing.assert_frame_equal(frame, before)
    assert not np.array_equal(y, aida(x, sst))


@pytest.mark.parametrize(
    "model_type, steps",
    [(AIDA, 2), (AIDA, 3), (AIDAWeatherMesh5C, 2), (AIDAWeatherMesh5C, 3)],
)
def test_aida_requires_backgrounds_at_declared_offset(
    core_model, fields, model_type, steps
):
    aida = model_type(core_model, assimilation_steps=steps, variables=("t2m",))
    valid = inputs(fields, steps)
    expected_lead = np.timedelta64(0 if model_type is AIDA else 3, "h")
    assert aida.output_coords(*valid).lead_time.values[0] == expected_lead
    assert aida(*valid, surface=observations()).lead_time.values[0] == expected_lead
    for slot in range(2):
        wrong = list(valid)
        wrong[slot] = wrong[slot].assign_coords(lead_time=[np.timedelta64(0, "h")])
        with pytest.raises(ValueError, match="lead_time"):
            aida.output_coords(*wrong)
        with pytest.raises(ValueError, match="lead_time"):
            aida(*wrong, surface=observations())


def test_coupled_fork_and_replay(core_model, fields):
    coupled = AIDAWeatherMesh5C(core_model, fork_hour=12, variables=("t2m",))
    assert not isinstance(coupled, AIDA)
    x, sst = inputs(fields)
    frame = observations()
    coupled.set_rng(123)
    y, state = coupled.initialize(x, sst, surface=frame)
    assert y.lead_time.values[0] == np.timedelta64(3, "h")
    before_rng = torch.random.get_rng_state()
    for _ in range(3):
        previous, snapshot = y.copy(deep=True), deepcopy(state)
        y, state = coupled.step(y, state)
        replay, replay_state = coupled.step(previous, snapshot)
        xr.testing.assert_identical(y, replay)
        assert torch.equal(state.latent, replay_state.latent)
    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert state.position == 1
    coupled.set_rng(999)
    xr.testing.assert_identical(coupled.step(previous, snapshot)[0], y)
    coupled.set_rng(123)
    it = coupled.create_iterator(x, sst, surface=frame)
    for _ in range(4):
        z = next(it)
    xr.testing.assert_identical(z, y)
    it.close()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_iasi_uses_raw_radiance_and_channel_axis(device):
    if device == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        pytest.importorskip("cudf")
    radiance = np.arange(2 * len(IASI_CHANNELS), dtype=np.float64).reshape(2, -1) / 1e5
    frame = pd.DataFrame(
        {
            **{field: 0.0 for field in _schema("mtiasi") if field != "variable"},
            "time": np.array(
                ["2026-10-28T00", "2026-10-28T01"], dtype="datetime64[ns]"
            ),
            "observation_iasi_radiance": list(radiance),
            "scan_position": 1,
            "footprint_quality": pd.array([None, 0], dtype="uint32[pyarrow]"),
        }
    )
    array = pa.array(list(radiance), type=pa.list_(pa.float64(), len(IASI_CHANNELS)))
    frame["observation_iasi_radiance"] = pd.Series(
        pa.chunked_array([array[:1], array[1:]]), dtype=pd.ArrowDtype(array.type)
    )
    frame.attrs["channels"] = {"iasi": list(IASI_CHANNELS)}
    runtime = SimpleNamespace(
        observation_sources=("mtiasi",),
        pack_raw_observations=lambda source, columns: columns,
    )

    def pack():
        received = fetch_dataframe(
            lambda time, variable, fields: frame.copy(),
            frame["time"].to_numpy(),
            np.array(["iasi_radiance"]),
            device=torch.device(device),
        )
        return _pack(runtime, "mtiasi", received)

    columns = pack()
    np.testing.assert_array_equal(columns["radiance"], radiance)
    np.testing.assert_array_equal(columns["quality"], [np.nan, 0])
    frame.attrs["channels"]["iasi"].reverse()
    with pytest.raises(ValueError, match="native selected radiance"):
        pack()


def test_gpsro_pairs_report_levels_not_height_coordinates():
    frame = pd.DataFrame(
        {
            "time": np.repeat(np.datetime64("2026-10-28T09"), 4),
            "lat": [10.0, 11.0, 12.0, 13.0],
            "lon": 20.0,
            "report": 42,
            "profile_level": [0, 1, 0, 1],
            "confidence": [100.0, 100.0, None, None],
            "ionosphere_corrected": [True, False, True, True],
            "elev": [1200.0, 2200.0, 1000.0, 2000.0],
            "observation_gps": [0.01, 0.009, None, None],
            "observation_gps_refractivity": [None, None, 300.0, 250.0],
        }
    )
    columns = _reports("gpsro", pd.concat((frame, frame.assign(report=11))))
    np.testing.assert_array_equal(columns["elev"], [1000.0, 1000.0])
    np.testing.assert_array_equal(columns["lat"], [10.0, 10.0])
    np.testing.assert_array_equal(columns["gps_refractivity"], [300.0, 300.0])
    np.testing.assert_array_equal(columns["profile_id"], [0, 1])
    with pytest.raises(ValueError, match="profile_level"):
        _reports("gpsro", frame.drop(columns="profile_level"))
    duplicate = pd.concat(
        (
            frame.assign(cycle_time=np.datetime64("2026-10-28T06")),
            frame.assign(cycle_time=np.datetime64("2026-10-28T12")),
        )
    )
    np.testing.assert_array_equal(_reports("gpsro", duplicate)["profile_id"], [0, 1])


def test_default_sources_request_all_assimilation_windows(core_model):
    sources = AIDA(core_model).default_sources()
    analysis_time = datetime(2026, 9, 21, 9)
    tasks = sources[2]._create_tasks([analysis_time], ["mhs"])
    assert [task.datetime_file for task in tasks] == [
        analysis_time.replace(hour=hour) for hour in (0, 6, 12)
    ]
    assert min(task.datetime_min for task in tasks) == analysis_time - timedelta(
        hours=7, minutes=30
    )
    assert max(task.datetime_max for task in tasks) == analysis_time + timedelta(
        hours=1, minutes=30
    )
    assert sources[8] is None


@pytest.mark.parametrize("model_type", [AIDA, AIDAWeatherMesh5C])
def test_load_model_forwards_options(monkeypatch, core_model, model_type):
    from earth2studio.models.da import aida

    monkeypatch.setattr(aida, "_load_core", lambda package: core_model)
    options = {"fork_hour": None} if model_type is AIDAWeatherMesh5C else {}
    model = model_type.load_model(
        None, ("t2m",), assimilation_steps=2, max_observations=17, **options
    )
    analysis = model.analysis if options else model
    assert analysis.assimilation_steps == 2
    assert analysis.max_observations == 17
    assert model.variables == ("t2m",)
    assert not model.training
    if options:
        assert model.fork_hour is None


def test_coupled_runner_batches_two_branches_per_background(
    core_model, fields, monkeypatch
):

    class Background:
        calls = []

        def __call__(self, time, lead_time, variable, *, member=None):
            self.calls.append(list(member))
            x = (
                fields[0]
                .isel(batch=0, drop=True)
                .assign_coords(time=time, lead_time=lead_time)
            )
            return xr.concat(
                [x + float(m) for m in member], dim="ensemble"
            ).assign_coords(ensemble=member)

    class Ocean:
        def __call__(self, time, variable):
            return (
                fields[1].isel(batch=0, lead_time=0, drop=True).assign_coords(time=time)
            )

    class Surface:
        SCHEMA = pa.schema([])

        def __call__(self, time, variable, fields=None):
            return observations()

    weather = Background()
    coupled = AIDAWeatherMesh5C(core_model, fork_hour=12, variables=("t2m",))
    monkeypatch.setattr(coupled, "default_sources", lambda: (None,) * 11)
    runner = CoupledRunner(
        {"weather": coupled},
        sources={"weather": (weather, Ocean(), *[None] * 8, Surface())},
        source_member=lambda m: m // 2 + 1,
    )
    items = work_items(
        [np.datetime64("2026-10-28T09")],
        np.timedelta64(12, "h"),
        range(4),
        member_batch=2,
    )
    store = ZarrBackend()
    Pipeline(runner, output={"weather": store}, device="cpu").run(items)
    assert weather.calls == [[1], [2]]
    values = store.to_xarray()["t2m"]
    np.testing.assert_array_equal(
        values.lead_time, np.arange(3, 13, 3).astype("timedelta64[h]")
    )
    np.testing.assert_array_equal(values.ensemble, np.arange(4))
    forecast = (
        values.isel(time=0, lat=0, lon=0).transpose("ensemble", "lead_time").values
    )
    np.testing.assert_array_equal(forecast[0, :3], forecast[1, :3])
    np.testing.assert_array_equal(forecast[2, :3], forecast[3, :3])
    assert forecast[0, 0] != forecast[2, 0]
    at_fork = values.sel(lead_time=np.timedelta64(12, "h"))
    assert np.all(abs(at_fork.sel(ensemble=0) - at_fork.sel(ensemble=1)) > 0)


class _GriddedAIDAWeatherMesh5C(AIDAWeatherMesh5C):
    """Expose only the IFS and SST slots; every observation slot stays empty.

    The prognostic contract harness accepts DataArray slots only.
    """

    def input_coords(self):
        return super().input_coords()[:2]

    def default_sources(self):
        return super().default_sources()[:2]

    def output_coords(self, background, sst):
        return super().output_coords(background, sst)

    def __call__(self, background, sst):
        return self.initialize(background, sst)[0]

    def initialize(self, background, sst):
        return super().initialize(background, sst)

    def create_iterator(self, background, sst):
        yield from _iterator(self, background, sst)


@pytest.mark.parametrize("fork_hour", [None, 12])
def test_aida_weathermesh_conformance(core_model, fork_hour):
    from earth2studio.models.conformance import _evaluate_prognostic

    model = _GriddedAIDAWeatherMesh5C(
        core_model, fork_hour=fork_hour, variables=("t2m",)
    )
    assert model.stochastic is (fork_hour is not None)
    report = _evaluate_prognostic(model, nsteps=4)
    # P10: hook edits cannot update the native latent and raise (as WeatherMesh5C).
    # P6: AIDA requires backgrounds at exactly -3 * assimilation_steps hours and
    # rejects rebased lead times. The harness plans later steps by rebasing the
    # inputs, so the rollout, seeding and replay checks (P9, P13, P19, and P14
    # when stochastic) fail at planning rather than in the model.
    expected = {"P6", "P9", "P10", "P13", "P19"}
    if model.stochastic:
        expected.add("P14")
    assert {v.partition(":")[0] for v in report.violations} == expected


@pytest.mark.package
@pytest.mark.timeout(900)
def test_aida_weathermesh_package():
    if not torch.cuda.is_available():
        pytest.skip("AIDA package test requires CUDA")

    model = AIDAWeatherMesh5C.load_model(
        AIDAWeatherMesh5C.load_default_package(),
        variables=("t2m", "z500"),
        fork_hour=12,
    ).to("cuda:0")
    x, sst = (
        standard_atmosphere(s, "2026-10-28T09", device="cuda:0")
        for s in model.input_coords()[:2]
    )
    model.set_rng(0)
    y, state = model.initialize(x, sst, surface=observations())
    assert y.shape == (1, 1, 1, 2, 720, 1440)
    assert y.lead_time.values[0] == np.timedelta64(3, "h")
    for _ in range(3):
        y, state = model.step(y, state)
    # The learned fork is applied at +12h.
    assert y.lead_time.values[0] == np.timedelta64(12, "h")
    assert state.position == 1
    values = y.e2s.as_numpy()
    assert np.isfinite(values.values).all()
    assert 200 < float(values.sel(variable="t2m").mean()) < 330
    assert 45000 < float(values.sel(variable="z500").mean()) < 60000

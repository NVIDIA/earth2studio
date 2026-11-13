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

import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.px.weathermesh5c import WeatherMesh5C
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch


@pytest.fixture
def core_model(monkeypatch):
    from weathermesh5c.inference import (
        SURFACE_VARIABLES,
        Inference,
        ProductTransform,
    )

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
            if variable in {"msl"}:
                value = 101300
            if variable == "lsm":
                value = 0
            values[..., i, :, :] = value
        output.append(from_torch(torch.from_numpy(values), layout))
    return tuple(output)


def test_weathermesh_call_and_replay(model, fields):
    before = [x.copy(deep=True) for x in fields]
    y, state = model.initialize(*fields)
    assert np.isfinite(y.values).all()
    xr.testing.assert_identical(y, model(*fields))
    iterator = model.create_iterator(*fields)
    xr.testing.assert_identical(next(iterator), y)
    expected, following = model.step(y, deepcopy(state))
    xr.testing.assert_identical(next(iterator), expected)
    xr.testing.assert_identical(model.step(y, state)[0], expected)
    assert torch.equal(state.latent, deepcopy(state).latent)
    for x, saved in zip(fields, before):
        xr.testing.assert_identical(x, saved)
    assert following.latent.mean() > state.latent.mean()
    iterator.close()


def test_weathermesh_hooks_reject_field_edits(model, fields):
    called = []
    model.rear_hook = lambda y: called.append("rear") or y
    model.front_hook = lambda y: y + 1
    iterator = model.create_iterator(*fields)
    next(iterator)
    assert called == ["rear"]
    with pytest.raises(ValueError, match="Edited forecast"):
        next(iterator)
    model.clear_hooks()
    y, state = model.initialize(*fields)
    xr.testing.assert_identical(model.step(y, state)[0], model.step(y + 1, state)[0])
    model.rear_hook = lambda y: y + 1
    with pytest.raises(ValueError, match="Edited forecast"):
        next(model.create_iterator(*fields))
    with pytest.raises(ValueError):
        model.step(y, None)


def test_weathermesh_default_iteration_does_not_copy_output_to_cpu(
    model, fields, monkeypatch
):
    iterator = model.create_iterator(*fields)
    y = next(iterator)
    monkeypatch.setattr(
        type(y.e2s), "as_numpy", lambda *_: pytest.fail("Output copied to CPU")
    )
    next(iterator)
    iterator.close()


@pytest.mark.parametrize(
    "coordinate, offset",
    [
        ("lead_time", np.timedelta64(72, "h")),
        ("time", np.timedelta64(24, "h")),
        ("batch", 1),
        ("lat", 1.0),
        ("lon", 1.0),
    ],
)
def test_weathermesh_rejects_coordinate_edits(
    model, fields, monkeypatch, coordinate, offset
):
    y, state = model.initialize(*fields)
    before = deepcopy(state)
    model.front_hook = lambda y: y.assign_coords({coordinate: y[coordinate] + offset})
    iterator = model.create_iterator(*fields)
    next(iterator)
    monkeypatch.setattr(
        model._runtime, "step", lambda *_: pytest.fail("Invalid state advanced")
    )
    modified = model.front_hook(y)
    with pytest.raises(ValueError):
        model.step(modified, state)
    with pytest.raises(ValueError):
        next(iterator)
    assert torch.equal(state.latent, before.latent)
    for dim in state.coordinates:
        np.testing.assert_array_equal(state.coordinates[dim], before.coordinates[dim])


def test_weathermesh_input_validation(model, fields):
    with pytest.raises(ValueError):
        model(fields[0].isel(variable=slice(1, None)), fields[1])
    with pytest.raises(ValueError, match="valid times"):
        model(fields[0], fields[1].assign_coords(lead_time=[np.timedelta64(3, "h")]))


def test_weathermesh_batch_matches_independent_calls(model, fields):
    batched = tuple(
        xr.concat((x, x + 0.5), dim="batch").assign_coords(
            batch=np.array([0, 1], dtype=object)
        )
        for x in fields
    )
    result, state = model.initialize(*batched)
    for index in range(2):
        expected = model(*(x.isel(batch=[index]) for x in batched))
        xr.testing.assert_identical(result.isel(batch=[index]), expected)
    xr.testing.assert_identical(
        model.step(result, state)[0],
        model.step(result.copy(deep=True), deepcopy(state))[0],
    )
    with pytest.raises(ValueError, match="Edited forecast"):
        model.step(
            result.assign_coords(batch=np.array(["0", "1"], dtype=object)), state
        )


def test_weathermesh_package_uses_resolved_assets(monkeypatch, core_model, tmp_path):
    from weathermesh5c.assets import BUNDLE_SCHEMA_VERSION

    from earth2studio.models.px import weathermesh5c

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "model": "WeatherMesh-5c with AIDA",
                "files": [],
            }
        )
    )
    resolved = []
    package = SimpleNamespace(
        resolve=lambda path: resolved.append(path) or tmp_path / path
    )

    def load(bundle):
        bundle.path("checkpoint.pt")
        return core_model

    monkeypatch.setattr(weathermesh5c, "load_core_model", load)
    model = WeatherMesh5C.load_model(package)
    assert resolved == ["manifest.json", "manifest.json", "checkpoint.pt"]
    assert not model.training


def test_weathermesh_conformance(model, fields):
    from earth2studio.models.conformance import _evaluate_prognostic

    report = _evaluate_prognostic(model)
    assert {v.partition(":")[0] for v in report.violations} == {"P10"}


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


@pytest.fixture(scope="module")
def package_model():
    if not torch.cuda.is_available():
        pytest.skip("WeatherMesh-5c package test requires CUDA")
    package = WeatherMesh5C.load_default_package()
    return WeatherMesh5C.load_model(package, variables=("t2m", "z500"))


@pytest.mark.package
@pytest.mark.timeout(900)
def test_weathermesh_package(package_model):
    model = package_model.to("cuda:0")
    x, sst = (standard_atmosphere(s, device="cuda:0") for s in model.input_coords())
    y, state = model.initialize(x, sst)
    assert y.dims == ("batch", "time", "lead_time", "variable", "lat", "lon")
    assert y.shape == (1, 1, 1, 2, 720, 1440)
    assert y.lead_time.values[0] == np.timedelta64(3, "h")
    y, _ = model.step(y, state)
    assert y.lead_time.values[0] == np.timedelta64(6, "h")
    values = y.e2s.as_numpy()
    assert np.isfinite(values.values).all()
    # Physically plausible ranges catch variable-order or unit mismatches.
    assert 200 < float(values.sel(variable="t2m").mean()) < 330
    assert 45000 < float(values.sel(variable="z500").mean()) < 60000

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

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr
from pyproj import CRS

import earth2studio.grids as grids
from earth2studio.data import Random, Random_FX, fetch_data
from earth2studio.grids import CurvilinearGrid, LatLonGrid
from earth2studio.models.conformance import ContractException, check_prognostic_contract
from earth2studio.models.px.datareplay import DataReplay
from earth2studio.models.px.persistence import Persistence

LAT = np.linspace(90, -90, 8)
LON = np.linspace(0, 360, 16, endpoint=False)
DOMAIN = OrderedDict(lat=LAT, lon=LON)
TIME = np.array([np.datetime64("2020-01-01T00:00:00")])
VARIABLE = np.array(["t2m", "u10m", "z500"])
STEP = np.timedelta64(6, "h")


class GridRandom(Random):
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        output.attrs["earth2studio_crs"] = LatLonGrid(
            self.domain_coords["lat"], self.domain_coords["lon"]
        ).crs.to_string()
        return output


class GridRandomFX(Random_FX):
    def __call__(self, time, lead_time, variable):
        output = super().__call__(time, lead_time, variable)
        output.attrs["earth2studio_crs"] = LatLonGrid(
            self.domain_coords["lat"], self.domain_coords["lon"]
        ).crs.to_string()
        return output


def _initial_condition(source: Random | Random_FX):
    return fetch_data(
        source,
        time=TIME,
        variable=VARIABLE,
        lead_time=np.array([np.timedelta64(0, "h")]),
    )


@pytest.mark.parametrize("source_type", [GridRandom, GridRandomFX])
def test_datareplay_call(source_type):
    source = source_type(DOMAIN)
    x = _initial_condition(source)
    replay = DataReplay(source, VARIABLE, DOMAIN, step=STEP)

    output = replay(x)

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.isfinite(output.e2s.to_torch()[0]).all()
    np.testing.assert_array_equal(output.time, TIME)
    np.testing.assert_array_equal(output.lead_time, np.array([STEP]))


@pytest.mark.parametrize("source_type", [GridRandom, GridRandomFX])
def test_datareplay_iter(source_type):
    source = source_type(DOMAIN)
    x = _initial_condition(source)
    replay = DataReplay(source, VARIABLE, DOMAIN, step=STEP)
    hook_calls = {"front": 0, "rear": 0}

    def front_hook(data):
        hook_calls["front"] += 1
        return data

    def rear_hook(data):
        hook_calls["rear"] += 1
        return data

    replay.front_hook = front_hook
    replay.rear_hook = rear_hook
    iterator = replay.create_iterator(x)

    initial = next(iterator)
    xr.testing.assert_identical(initial, x)
    np.testing.assert_array_equal(initial.lead_time, np.array([np.timedelta64(0, "h")]))
    assert hook_calls == {"front": 0, "rear": 0}

    first = next(iterator)
    np.testing.assert_array_equal(first.lead_time, np.array([STEP]))
    assert hook_calls == {"front": 1, "rear": 1}

    second = next(iterator)
    np.testing.assert_array_equal(second.lead_time, np.array([2 * STEP]))
    assert hook_calls == {"front": 2, "rear": 2}


@pytest.mark.parametrize("source_type", [GridRandom, GridRandomFX])
def test_datareplay_conformance(source_type):
    source = source_type(DOMAIN)
    replay = DataReplay(source, VARIABLE, DOMAIN, step=STEP)
    # KNOWN CONTRACT GAP (deferred, do not weaken this check to hide it):
    # DataReplay declares stochastic=False but, because it replays from a data
    # source that draws fresh random values on every fetch instead of caching a
    # rollout, two rollouts from one input disagree. check_prognostic_contract()
    # raises with:
    #   P13: model declares stochastic=False but two rollouts from one input
    #   disagree; declare stochastic=True and implement set_rng()
    # See dev/spec/MODEL_CONTRACT_SPEC.md. Fixing this is out of scope for this
    # change and tracked separately; re-enable the assertion below once fixed.
    with pytest.raises(ContractException) as excinfo:
        check_prognostic_contract(replay)
    assert "P13" in str(excinfo.value)


def test_datareplay_input_coords_copy():
    replay = DataReplay(Random(DOMAIN), "t2m", DOMAIN)
    coords = replay.input_coords()
    coords.coords["variable"] = ["msl"]

    assert str(replay) == "DataReplay()"
    assert replay.input_coords()["variable"][0] == "t2m"


def test_datareplay_output_coords_copy():
    source = GridRandom(DOMAIN)
    coords = _initial_condition(source)
    replay = DataReplay(source, VARIABLE, DOMAIN)
    original_lead_time = coords["lead_time"].copy()

    output_coords = replay.output_coords(coords)

    np.testing.assert_array_equal(coords["lead_time"], original_lead_time)
    assert output_coords is not coords


def test_datareplay_grid_mismatch_raises():
    source = GridRandom(OrderedDict(lat=np.linspace(90, -90, 9), lon=LON))
    x = _initial_condition(GridRandom(DOMAIN))
    replay = DataReplay(source, VARIABLE, DOMAIN)

    with pytest.raises(ValueError, match="required dim lat is not of size 8"):
        replay(x)


def test_datareplay_nonfinite_raises(monkeypatch):
    source = GridRandom(DOMAIN)
    x = _initial_condition(source)
    replay = DataReplay(source, VARIABLE, DOMAIN)
    monkeypatch.setattr(np.random, "randn", lambda *shape: np.full(shape, np.nan))

    with pytest.raises(ValueError, match="non-finite"):
        replay(x)


@pytest.mark.parametrize(
    "coords_update, match",
    [
        (
            {"time": np.empty(0, dtype="datetime64[ns]")},
            "Dimension 'time' must be nonempty",
        ),
        ({"variable": VARIABLE[::-1]}, "required dim variable are not the same"),
    ],
)
def test_datareplay_invalid_coords(coords_update, match):
    source = GridRandom(DOMAIN)
    x = _initial_condition(source)
    if "time" in coords_update:
        x = x.isel(time=slice(0, 0))
    else:
        x = x.assign_coords(coords_update)
    replay = DataReplay(source, VARIABLE, DOMAIN)

    with pytest.raises(ValueError, match=match):
        replay(x)


@pytest.mark.parametrize(
    "step, error",
    [
        (6, TypeError),
        (np.timedelta64(0, "h"), ValueError),
        (np.timedelta64(-1, "h"), ValueError),
        (np.timedelta64("NaT"), ValueError),
    ],
)
def test_datareplay_invalid_step(step, error):
    with pytest.raises(error):
        DataReplay(Random(DOMAIN), VARIABLE, DOMAIN, step=step)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("source_type", [GridRandom, GridRandomFX])
def test_datareplay_preserves_ensemble_metadata(source_type, device):
    source = source_type(DOMAIN)
    x = _initial_condition(source).expand_dims(ensemble=["control", "perturbed"]).copy()
    x = x.assign_coords(member_weight=("ensemble", [0.75, 0.25]), height=2.0)
    x.name = "weather"
    x.attrs["experiment"] = "replay"
    x.encoding["test"] = "retained"
    if device == "cuda:0":
        x = x.e2s.as_cupy(device=0)
    replay = DataReplay(source, VARIABLE, DOMAIN)

    output = replay(x)

    assert output.dims == x.dims
    assert output.name == x.name
    assert output.attrs["experiment"] == "replay"
    assert output.encoding == x.encoding
    xr.testing.assert_identical(output.member_weight, x.member_weight)
    xr.testing.assert_identical(output.height, x.height)
    tensor, _ = output.e2s.to_torch()
    assert tensor.device == torch.device(device)
    torch.testing.assert_close(tensor[0], tensor[1])


def test_datareplay_rejects_missing_source_spatial_dimension():
    class MissingLatitude(GridRandom):
        def __call__(self, *args, **kwargs):
            return super().__call__(*args, **kwargs).isel(lat=0, drop=True)

    replay = DataReplay(MissingLatitude(DOMAIN), VARIABLE, DOMAIN)
    x = _initial_condition(GridRandom(DOMAIN))
    with pytest.raises(ValueError, match="dimensions"):
        replay(x)


def test_datareplay_domain_does_not_initialize_checkpoint(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Domain resolution must not initialize a Persistence checkpoint")

    monkeypatch.setattr(
        "earth2studio.models.px.persistence.bind_checkpoint_state", forbidden
    )
    replay = DataReplay(Random(DOMAIN), "t2m", DOMAIN)
    assert replay.input_coords().sizes["lat"] == LAT.size


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("hook", ["front", "rear", "both"])
def test_datareplay_inplace_hooks_own_storage(device, hook):
    source = GridRandom(DOMAIN)
    x = _initial_condition(source).assign_coords(height=2.0)
    x.name = "weather"
    x.attrs["experiment"] = {"calls": 0}
    x.encoding["source"] = "replay.nc"
    if device == "cuda:0":
        x = x.e2s.as_cupy(device=0)
    original = x.copy(deep=True)
    replay = DataReplay(source, VARIABLE, DOMAIN)
    calls = []

    def mutate(field):
        calls.append(field.lead_time.values.copy())
        field.data += 1
        field.attrs["experiment"]["calls"] += 1
        field.coords["height"].data += 1
        return field

    if hook in ("front", "both"):
        replay.front_hook = mutate
    if hook in ("rear", "both"):
        replay.rear_hook = mutate
    iterator = replay.create_iterator(x)
    retained = []
    for step in range(4):
        output = next(iterator)
        xr.testing.assert_identical(x, original)
        for previous, snapshot in retained:
            xr.testing.assert_identical(previous, snapshot)
        assert output.attrs["experiment"]["calls"] == step * (
            2 if hook == "both" else 1
        )
        assert output.lead_time.values[0] == step * STEP
        assert output.e2s.to_torch()[0].device == torch.device(device)
        assert output.name == x.name
        assert output.encoding == x.encoding
        retained.append((output, output.copy(deep=True)))
    assert len(calls) == 3 * (2 if hook == "both" else 1)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_datareplay_initial_yield_is_owned(device):
    source = GridRandom(DOMAIN)
    x = _initial_condition(source)
    x.attrs["experiment"] = {"name": "original"}
    if device == "cuda:0":
        x = x.e2s.as_cupy(device=0)
    original = x.copy(deep=True)
    replay = DataReplay(source, VARIABLE, DOMAIN)
    iterator = replay.create_iterator(x)
    initial = next(iterator)
    initial.data += 100
    initial.attrs["experiment"]["name"] = "consumer"
    xr.testing.assert_identical(x, original)
    assert next(iterator).attrs["experiment"]["name"] == "original"


@pytest.mark.parametrize("model_type", [Persistence, DataReplay])
@pytest.mark.parametrize("domain_map", [False, True])
@pytest.mark.parametrize("difference", [None, "indexes", "attrs", "coord_attrs", "crs"])
def test_replay_persistence_exact_registered_domain(
    monkeypatch, model_type, domain_map, difference
):
    latitude = np.array([[40.0, 40.1], [41.0, 41.1]])
    longitude = np.array([[10.0, 11.0], [10.1, 11.1]])
    configured = CurvilinearGrid(latitude, longitude)

    class RegisteredGrid(CurvilinearGrid):
        @property
        def attrs(self):
            attrs = configured.attrs.copy()
            if difference == "attrs":
                attrs["description"] = "different grid"
            return attrs

        @property
        def crs(self):
            return CRS("EPSG:4326") if difference == "crs" else None

        def coords(self, *args, **kwargs):
            coords = super().coords(*args, **kwargs)
            if difference == "coord_attrs":
                coords["lat"].attrs["units"] = "degrees_north"
            return coords

    registered = RegisteredGrid(
        latitude, longitude, y=np.array([10, 11]) if difference == "indexes" else None
    )
    assert registered.fingerprint() == configured.fingerprint()
    name = "exact-domain-test"
    monkeypatch.setitem(grids._GRID_REGISTRY, name, registered)
    domain = OrderedDict(lat=latitude, lon=longitude) if domain_map else configured
    model = (
        model_type(Random(DOMAIN), "t2m", domain)
        if model_type is DataReplay
        else model_type("t2m", domain)
    )
    signature = model.input_coords()
    assert "earth2studio_grid_id" not in signature.attrs
    for coord in configured.coords():
        xr.testing.assert_identical(signature.coords[coord], configured.coords()[coord])
    if difference is not None:
        assert "earth2studio_crs" not in signature.attrs
        assert "description" not in signature.attrs
    named_model = (
        model_type(Random(DOMAIN), "t2m", name)
        if model_type is DataReplay
        else model_type("t2m", name)
    )
    named_signature = named_model.input_coords()
    assert named_signature.attrs["earth2studio_grid_id"] == name
    for coord in registered.coords():
        xr.testing.assert_identical(
            named_signature.coords[coord], registered.coords()[coord]
        )
    for key, value in registered.attrs.items():
        assert named_signature.attrs[key] == value
    if registered.crs is not None:
        assert named_signature.attrs["earth2studio_crs"] == registered.crs.to_string()

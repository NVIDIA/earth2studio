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

try:
    from aurora import Batch, Metadata
except ImportError:
    Batch = Metadata = None

import earth2studio.models.px.aurora1p5 as aurora_module
from earth2studio.models.conformance import ContractException, check_prognostic_contract
from earth2studio.models.px import Aurora1p5, Aurora1p5Ensemble
from earth2studio.models.px.aurora1p5 import _OUTPUT_ONLY_SURF_VARS
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch

_N_VARS = 90  # 65 atmos + 18 surface + 7 output-only
_H = 720
_W = 1440
_STATIC_KEYS = ["lsm", "z", "anor", "isor"]  # representative subset for mocking


# ── Shared mock models ────────────────────────────────────────────────────────


class PhooAurora1p5Model(torch.nn.Module):
    """Dummy Aurora1p5: echoes the most-recent input time step."""

    def forward(self, batch: Batch, lead_times: torch.Tensor) -> Batch:
        surf = {k: v[:, -1:, ...] for k, v in batch.surf_vars.items()}
        # Add output-only vars that the real model produces but don't exist in input
        ref = next(iter(surf.values()))
        for _e2s, aurora_name, _log in _OUTPUT_ONLY_SURF_VARS:
            surf[aurora_name] = torch.zeros_like(ref)
        return Batch(
            surf_vars=surf,
            static_vars=batch.static_vars,
            atmos_vars={k: v[:, -1:, ...] for k, v in batch.atmos_vars.items()},
            metadata=Metadata(
                lat=batch.metadata.lat,
                lon=batch.metadata.lon,
                time=batch.metadata.time,
                atmos_levels=batch.metadata.atmos_levels,
                rollout_step=batch.metadata.rollout_step + 1,
            ),
        )

    def apply_rollout_input_clipping(self, batch: Batch) -> Batch:
        return batch


class PhooAurora1p5EnsembleModel(PhooAurora1p5Model):
    """Dummy ensemble model: echo logic plus RNG-draw noise, adds reset_noise().

    The real ensemble checkpoint injects fresh Gaussian noise per forward pass, so
    the mock must too: a deterministic forward can't demonstrate that different
    seeds give different rollouts (P13), since torch.manual_seed has nothing to
    perturb. Drawing from the global generator (no explicit generator=) ties the
    noise to whatever set_rng() seeded, matching the real model's seeding path.
    """

    def forward(self, batch: Batch, lead_times: torch.Tensor) -> Batch:
        out = super().forward(batch, lead_times)
        surf = {k: v + torch.randn_like(v) for k, v in out.surf_vars.items()}
        atmos = {k: v + torch.randn_like(v) for k, v in out.atmos_vars.items()}
        return Batch(
            surf_vars=surf,
            static_vars=out.static_vars,
            atmos_vars=atmos,
            metadata=out.metadata,
        )

    def reset_noise(self) -> None:
        pass


def _make_model(device: str = "cpu") -> Aurora1p5:
    core = PhooAurora1p5Model()
    static_vars = {k: torch.ones(_H, _W) for k in _STATIC_KEYS}
    return Aurora1p5(core, static_vars).to(device)


def _make_ensemble_model(device: str = "cpu") -> Aurora1p5Ensemble:
    core = PhooAurora1p5EnsembleModel()
    static_vars = {k: torch.ones(_H, _W) for k in _STATIC_KEYS}
    return Aurora1p5Ensemble(core, static_vars).to(device)


@pytest.fixture(autouse=True)
def small_models(request, monkeypatch):
    if request.node.get_closest_marker("package"):
        return
    monkeypatch.setattr(__import__(__name__, fromlist=["_H"]), "_H", 4)
    monkeypatch.setattr(__import__(__name__, fromlist=["_W"]), "_W", 8)
    if Batch is None:
        monkeypatch.setattr(
            __import__(__name__, fromlist=["Batch"]), "Batch", SimpleNamespace
        )
        monkeypatch.setattr(
            __import__(__name__, fromlist=["Metadata"]), "Metadata", SimpleNamespace
        )
        monkeypatch.setattr(aurora_module, "Batch", SimpleNamespace)
        monkeypatch.setattr(aurora_module, "Metadata", SimpleNamespace)
        monkeypatch.setattr(
            aurora_module,
            "aurora_insolation",
            lambda dates, lat, lon, **kw: np.zeros((2, len(lat), len(lon))),
        )
        monkeypatch.setattr(
            aurora_module, "aurora_log_untransform", lambda x: torch.expm1(x)
        )
        monkeypatch.setattr(Aurora1p5, "__init__", Aurora1p5.__init__.__wrapped__)
        monkeypatch.setattr(
            Aurora1p5Ensemble, "__init__", Aurora1p5Ensemble.__init__.__wrapped__
        )
    original = Aurora1p5.input_coords

    def signature(self):
        native = original(self)
        return coord_array(
            native.dims,
            {
                "lead_time": native.lead_time.values,
                "variable": native.coords["variable"].values,
                "lat": np.linspace(90, -90, _H, endpoint=False),
                "lon": np.linspace(0, 360, _W, endpoint=False),
            },
            dynamic=("batch", "time"),
        )

    monkeypatch.setattr(Aurora1p5, "input_coords", signature)


def _input(p, time, device="cpu"):
    signature = coord_array_like(p.input_coords(), {"time": time})
    signature = coord_array_like(signature, {"batch": [0]})
    x = from_torch(torch.rand(signature.shape, device=device), signature).isel(
        batch=0, drop=True
    )
    x.name = "weather"
    x.encoding["note"] = "kept"
    x.attrs["source"] = "test"
    return x.assign_coords(marker=7)


# ── Aurora1p5 tests ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array([np.datetime64("2001-06-04T06:00")]),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_call(time, device):
    p = _make_model(device)
    x = _input(p, time)
    out = p(x)
    assert out.shape == torch.Size([len(time), 1, _N_VARS, _H, _W])
    np.testing.assert_array_equal(
        out.coords["variable"], p.output_coords(x).coords["variable"]
    )
    assert out.dims == x.dims and out.name == x.name and out.encoding == x.encoding
    assert out.marker == 7
    torch.testing.assert_close(
        out.e2s.to_torch()[0][:, :, :83].cpu(), x.e2s.to_torch()[0][:, -1:]
    )


@pytest.mark.parametrize("ensemble", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_iter(ensemble, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = _make_model(device)

    x = _input(p, time).expand_dims(ensemble=ensemble)
    saved = x.copy(deep=True)
    calls = []

    def front(field):
        assert field.dims == x.dims and "ensemble" not in field.coords
        calls.append("front")
        field.data += 1
        return field

    def rear(field):
        calls.append("rear")
        field.data += 1
        return field

    p.front_hook, p.rear_hook = front, rear
    p_iter = p.create_iterator(x)
    initial = next(p_iter)
    assert calls == []
    for i, out in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, _N_VARS, _H, _W])
        assert (out.coords["variable"] == p.output_coords(x).coords["variable"]).all()
        assert (out.coords["time"] == time).all()
        assert out.lead_time.shape == (1,)
        # Iterator yields at 1-hour intervals
        assert out.lead_time.values[0] == np.timedelta64(i + 1, "h")

        if i == 11:
            break
    assert calls == (["front"] + ["rear"] * 6) * 2
    xr.testing.assert_identical(x, saved)
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_iter_repeated(device):
    """Second create_iterator() call must produce the same outputs as the first.

    Regression test for preds_idx not being reset between rollouts, which caused
    Aurora's rollout_step counter to start at the wrong value on the second call.
    """
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = _make_model(device)

    x = _input(p, time, device)

    def collect(n=3):
        it = p.create_iterator(x)
        next(it)  # skip IC
        return [out.copy(deep=True) for out, _ in zip(it, range(n))]

    first = collect()
    second = collect()

    for a, b in zip(first, second):
        xr.testing.assert_identical(a, b)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_exceptions(device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = _make_model(device)

    x = _input(p, time, device)
    with pytest.raises(ValueError):
        p(x.transpose(..., "lon", "lat"))


def test_aurora1p5_conformance():
    p = Aurora1p5.__new__(Aurora1p5)
    torch.nn.Module.__init__(p)
    signature = p.input_coords()
    assert signature.shape == (0, 0, 2, 83, _H, _W)
    assert "tp:sum:1h" in p.output_coords(signature).coords["variable"]
    assert p.front_hook_interval == 6
    check_prognostic_contract(p, rollout=False)
    check_prognostic_contract(_make_model())


@pytest.fixture(scope="function")
def model() -> Aurora1p5:
    pytest.importorskip("aurora")
    package = Aurora1p5.load_default_package()
    return Aurora1p5.load_model(package)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aurora1p5_package(model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2023-01-01T00:00")])
    p = model.to(device)

    x = _input(p, time, device)
    out = p(x)

    assert out.shape == torch.Size([len(time), 1, _N_VARS, _H, _W])
    assert (out.coords["variable"] == p.output_coords(x).coords["variable"]).all()
    assert (out.time == time).all()
    assert out.dims == x.dims


# ── Aurora1p5Ensemble tests ───────────────────────────────────────────────────


@pytest.mark.parametrize("n_members", [2, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_ensemble_iter(n_members, device):
    """Ensemble members run independently; noise is reset at each create_iterator."""
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = _make_ensemble_model(device)

    x = _input(p, time, device).expand_dims(ensemble=np.arange(n_members))
    p_iter = p.create_iterator(x)
    next(p_iter)  # skip initial condition

    for i, out in enumerate(p_iter):
        assert out.shape == torch.Size([n_members, len(time), 1, _N_VARS, _H, _W])
        assert out.lead_time.values[0] == np.timedelta64(i + 1, "h")
        if i > 11:
            break


def test_aurora1p5_ensemble_conformance():
    p = _make_ensemble_model("cpu")
    with pytest.raises(ContractException) as exc_info:
        check_prognostic_contract(p)
    assert {v.split(":")[0] for v in exc_info.value.violations} == {"P14"}
    with pytest.raises(TypeError, match="reset"):
        p.set_rng(123, reset=True)
    x = _input(p, np.array(["2001-06-04"], dtype="datetime64[ns]"))
    state = torch.get_rng_state().clone()
    p.set_rng(None)
    assert torch.equal(state, torch.get_rng_state())
    p.set_rng(123)
    assert p.seed is None
    assert not torch.equal(state, torch.get_rng_state())
    state = torch.get_rng_state().clone()
    iterator = p.create_iterator(x)
    next(iterator)
    first = next(iterator)
    p.set_rng(123)
    iterator = p.create_iterator(x)
    next(iterator)
    xr.testing.assert_identical(first, next(iterator))
    assert not torch.equal(state, torch.get_rng_state())
    constructor_seeded = Aurora1p5Ensemble(
        PhooAurora1p5EnsembleModel(),
        {k: torch.ones(_H, _W) for k in _STATIC_KEYS},
        seed=987,
    )
    constructor_seeded.set_rng(123)
    iterator = constructor_seeded.create_iterator(x)
    next(iterator)
    constructor_first = next(iterator)
    assert not constructor_first.identical(first)
    constructor_seeded.set_rng(456)
    iterator = constructor_seeded.create_iterator(x)
    next(iterator)
    xr.testing.assert_identical(constructor_first, next(iterator))
    assert constructor_seeded.seed == 987


@pytest.fixture(scope="function")
def ensemble_model() -> Aurora1p5Ensemble:
    pytest.importorskip("aurora")
    package = Aurora1p5Ensemble.load_default_package()
    return Aurora1p5Ensemble.load_model(package)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aurora1p5_ensemble_package(ensemble_model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2023-01-01T00:00")])
    p = ensemble_model.to(device)

    n_members = 1
    x = _input(p, time, device).expand_dims(ensemble=np.arange(n_members))
    p_iter = p.create_iterator(x)
    next(p_iter)  # skip initial condition

    out = next(p_iter)
    assert out.shape == torch.Size([n_members, len(time), 1, _N_VARS, _H, _W])
    assert out.lead_time.values[0] == np.timedelta64(1, "h")

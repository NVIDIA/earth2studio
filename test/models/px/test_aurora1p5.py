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
from collections.abc import Iterable

import numpy as np
import pytest
import torch

try:
    from aurora import Batch, Metadata
except ImportError:
    pytest.importorskip("aurora")

from earth2studio.data import Random, fetch_data
from earth2studio.models import px
from earth2studio.models.px import Aurora1p5, Aurora1p5Ensemble
from earth2studio.models.px import aurora1p5 as aurora_module
from earth2studio.models.px.aurora1p5 import _OUTPUT_ONLY_SURF_VARS
from earth2studio.utils import handshake_dim

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
    """Dummy ensemble model: same echo logic, adds noise-cache stubs."""

    def __init__(self) -> None:
        super().__init__()
        self.noise_accumulation_calls: list[int] = []

    def reset_noise(self) -> None:
        pass

    def set_noise_accumulation(self, n: int = 0) -> None:
        self.noise_accumulation_calls.append(n)


def _make_model(device: str = "cpu") -> Aurora1p5:
    core = PhooAurora1p5Model()
    static_vars = {k: torch.ones(_H, _W) for k in _STATIC_KEYS}
    return Aurora1p5(core, static_vars).to(device)


def _make_ensemble_model(device: str = "cpu") -> Aurora1p5Ensemble:
    core = PhooAurora1p5EnsembleModel()
    static_vars = {k: torch.ones(_H, _W) for k in _STATIC_KEYS}
    return Aurora1p5Ensemble(core, static_vars).to(device)


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

    assert out.shape == torch.Size([len(time), 1, _N_VARS, _H, _W])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize("ensemble", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_iter(ensemble, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = _make_model(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    next(p_iter)  # Skip initial condition
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, _N_VARS, _H, _W])
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"].shape == (1,)
        # Iterator yields at 1-hour intervals
        assert out_coords["lead_time"][0] == np.timedelta64(i + 1, "h")

        if i > 11:
            break


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_iter_repeated(device):
    """Second create_iterator() call must produce the same outputs as the first.

    Regression test for preds_idx not being reset between rollouts, which caused
    Aurora's rollout_step counter to start at the wrong value on the second call.
    """
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = _make_model(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    def collect(n=3):
        it = p.create_iterator(x, coords)
        next(it)  # skip IC
        return [out.clone() for (out, _coords), _ in zip(it, range(n))]

    first = collect()
    second = collect()

    for a, b in zip(first, second):
        assert torch.equal(
            a, b
        ), "create_iterator() results differ across repeated calls"


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(_H)}),
        OrderedDict({"lat": np.random.randn(_H), "phoo": np.random.randn(_W)}),
        OrderedDict({"lat": np.random.randn(_H), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_exceptions(dc, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = _make_model(device)

    r = Random(dc)
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.fixture(scope="function")
def model() -> Aurora1p5:
    package = Aurora1p5.load_default_package()
    return Aurora1p5.load_model(package)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aurora1p5_package(model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2023-01-01T00:00")])
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

    assert out.shape == torch.Size([len(time), 1, _N_VARS, _H, _W])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


# ── Aurora1p5Ensemble tests ───────────────────────────────────────────────────


@pytest.mark.parametrize("n_members", [2, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora1p5_ensemble_iter(n_members, device):
    """Ensemble members run independently; noise is reset at each create_iterator."""
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = _make_ensemble_model(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    x = x.unsqueeze(0).repeat(n_members, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(n_members)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)
    next(p_iter)  # skip initial condition

    for i, (out, out_coords) in enumerate(p_iter):
        assert out.shape == torch.Size([n_members, len(time), 1, _N_VARS, _H, _W])
        assert out_coords["lead_time"][0] == np.timedelta64(i + 1, "h")
        if i > 11:
            break


@pytest.mark.parametrize(
    "model_name,expected_n",
    [("Aurora1p5Ensemble", 6), ("Aurora1p5Ensemble_6h", 1)],
)
def test_aurora1p5_ensemble_noise_accumulation_cache_size(model_name, expected_n):
    """create_iterator sizes the noise cache to the sub-steps per AR cycle and
    disables accumulation again once the iterator is closed."""
    core = PhooAurora1p5EnsembleModel()
    static_vars = {k: torch.ones(_H, _W) for k in _STATIC_KEYS}
    p = getattr(px, model_name)(core, static_vars)

    time = np.array([np.datetime64("1993-04-05T00:00")])
    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device="cpu")

    p_iter = p.create_iterator(x, coords)
    next(p_iter)  # initial condition; runs the pre-yield setup in create_iterator

    assert core.noise_accumulation_calls == [expected_n]

    p_iter.close()
    assert core.noise_accumulation_calls == [expected_n, 0]


_VARIANTS = [
    ("Aurora1p5", 1, False),
    ("Aurora1p5Ensemble", 1, True),
    ("Aurora1p5_6h", 6, False),
    ("Aurora1p5Ensemble_6h", 6, True),
]


@pytest.mark.parametrize("model_name,step,ensemble", _VARIANTS)
def test_aurora1p5_fixed_cadence(model_name, step, ensemble):
    assert hasattr(px, model_name)
    model_cls = getattr(px, model_name)
    assert model_cls.__bases__ == (aurora_module._Aurora,)
    core = PhooAurora1p5EnsembleModel() if ensemble else PhooAurora1p5Model()
    p = model_cls(core, {})
    for target in (p._input_coords, p._output_coords):
        target["lat"] = np.linspace(90, -90, 4, endpoint=False)
        target["lon"] = np.linspace(0, 360, 8, endpoint=False)
    coords = p.input_coords()
    coords["batch"] = np.arange(2)
    coords["time"] = np.array([np.datetime64("2023-01-01T00:00")])
    coords["lead_time"] = np.array([6, 12], dtype="timedelta64[h]")
    x = torch.zeros(2, 1, 2, 83, 4, 8)
    calls = []

    original_forward = core.forward

    def forward(batch, lead_times):
        calls.append((batch.metadata.rollout_step, lead_times[0].item()))
        return original_forward(batch, lead_times)

    core.forward = forward
    out, out_coords = p(x, coords)
    assert out.shape == (2, 1, 1, 90, 4, 8)
    assert out_coords["lead_time"][0] == np.timedelta64(12 + step, "h")
    assert calls == [(0, step)]
    calls.clear()

    iterator = p.create_iterator(x, coords)
    initial, initial_coords = next(iterator)
    assert initial_coords["lead_time"][0] == np.timedelta64(12, "h")
    assert torch.isnan(initial[..., 83:, :, :]).all()
    for i in range(12 // step):
        out, out_coords = next(iterator)
        assert out_coords["lead_time"][0] == np.timedelta64(12 + (i + 1) * step, "h")
        assert out.shape == (2, 1, 1, 90, 4, 8)
    iterator.close()
    assert calls == [(cycle, h) for cycle in range(2) for h in range(step, 7, step)]
    if ensemble:
        assert core.noise_accumulation_calls == [6 // step, 0]
    np.testing.assert_array_equal(
        coords["lead_time"], np.array([6, 12], dtype="timedelta64[h]")
    )
    with pytest.raises(TypeError):
        model_cls(core, {}, lead_time_stride_hours=step)


@pytest.mark.parametrize("model_name,step,ensemble", _VARIANTS)
def test_aurora1p5_variant_load_model(model_name, step, ensemble, monkeypatch):
    assert hasattr(px, model_name)
    model_cls = getattr(px, model_name)
    calls = []
    core = PhooAurora1p5EnsembleModel() if ensemble else PhooAurora1p5Model()

    def load(package, aurora_cls, checkpoint):
        calls.append((package, aurora_cls, checkpoint))
        return core, {}

    monkeypatch.setattr(aurora_module, "_load_aurora1p5_from_package", load)
    package = object()
    p = model_cls.load_model(package)
    assert type(p) is model_cls
    expected_core = (
        aurora_module.Aurora1p5Ensemble_model
        if ensemble
        else aurora_module.Aurora1p5_model
    )
    checkpoint = (
        "aurora-0.25-v1.5-ensemble.ckpt" if ensemble else "aurora-0.25-v1.5.ckpt"
    )
    assert calls == [(package, expected_core, checkpoint)]
    assert p._output_coords["lead_time"][0] == np.timedelta64(step, "h")
    with pytest.raises(TypeError):
        model_cls.load_model(package, lead_time_stride_hours=step)


@pytest.fixture(scope="function")
def ensemble_model() -> Aurora1p5Ensemble:
    package = Aurora1p5Ensemble.load_default_package()
    return Aurora1p5Ensemble.load_model(package)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aurora1p5_ensemble_package(ensemble_model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2023-01-01T00:00")])
    p = ensemble_model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    n_members = 1
    x = x.unsqueeze(0).repeat(n_members, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(n_members)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)
    next(p_iter)  # skip initial condition

    out, out_coords = next(p_iter)
    assert out.shape == torch.Size([n_members, len(time), 1, _N_VARS, _H, _W])
    assert out_coords["lead_time"][0] == np.timedelta64(1, "h")

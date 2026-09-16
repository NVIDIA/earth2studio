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
from earth2studio.models.conformance import ContractException, check_prognostic_contract
from earth2studio.models.px import Aurora1p5, Aurora1p5Ensemble
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


def test_aurora1p5_conformance():
    """Check the mock Aurora1p5 model against the Earth2Studio model contract.

    Not fully conformant. P16 fails: successive yields from create_iterator()
    alias a shared buffer, so an earlier yield's values change once a later step
    is produced. This is a real violation (not a documented spec deviation); the
    assertion below pins the exact message rather than asserting a clean pass, so
    this test stays green while feeding the "fix wrappers" follow-up rather than
    hiding the gap. Do not weaken this further without fixing the wrapper.
    """
    p = _make_model("cpu")
    with pytest.raises(ContractException) as excinfo:
        check_prognostic_contract(p)
    assert excinfo.value.violations == [
        "P16: yield 0 changed after later steps were produced, so the yields "
        "alias one buffer; a caller holding a yield across steps — an async IO "
        "write, a resume buffer — reads the wrong values"
    ]


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


def test_aurora1p5_ensemble_conformance():
    """Check the mock Aurora1p5Ensemble model against the Earth2Studio model contract.

    Not fully conformant, expected:
    - P14 (x2): Aurora1p5Ensemble.set_rng() is a bare torch.manual_seed(seed) with
      no torch.random.fork_rng() wrapper, so both the seeding and stepping halves
      of RNG isolation fail. This is the documented known deviation in
      dev/spec/MODEL_CONTRACT_SPEC.md's "RNG isolation" section.
    - P16: the same shared-buffer aliasing bug as the base Aurora1p5 class (see
      test_aurora1p5_conformance).
    The assertion pins the exact messages rather than a clean pass, so this test
    stays green while feeding the "fix wrappers" follow-up.
    """
    p = _make_ensemble_model("cpu")
    with pytest.raises(ContractException) as excinfo:
        check_prognostic_contract(p)
    assert excinfo.value.violations == [
        "P14: set_rng() left the global RNG state perturbed, which silently "
        "reseeds every other consumer in the process — a second model in a "
        "cascade, a perturbation method, a dataloader. Seed a local "
        "torch.Generator, or confine global seeding to a torch.random.fork_rng() "
        "block",
        "P16: yield 0 changed after later steps were produced, so the yields "
        "alias one buffer; a caller holding a yield across steps — an async IO "
        "write, a resume buffer — reads the wrong values",
        "P14: stepping a seeded model left the global RNG state perturbed, which "
        "silently reseeds every other consumer in the process — a second model "
        "in a cascade, a perturbation method, a dataloader. Seed a local "
        "torch.Generator, or confine global seeding to a torch.random.fork_rng() "
        "block",
    ]


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

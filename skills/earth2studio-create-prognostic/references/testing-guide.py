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

"""Native prognostic test patterns to adapt inside test/models/px/test_<model>.py.

Supply the existing model fixture with mock weights. Retain call, iterator,
exceptions, conformance and package cases; avoid parallel migration suites or
redundant matrices. Package tests stay marked package and use real cached weights
only when explicitly enabled. Never replace core numerical assertions with shape
checks alone. Stochastic mocks must exercise the wrapper's real seeding mechanism.
"""

import inspect

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils.coords import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


def make_input(model, device: str = "cpu") -> xr.DataArray:
    """Concretize the configured signature without materializing its placeholder."""
    signature = model.input_coords()
    replacements = {"batch": [0]}
    if "time" in signature.attrs.get("earth2studio_dynamic_dims", ()):
        replacements["time"] = np.array(["2024-01-01"], dtype="datetime64[ns]")
    signature = coord_array_like(signature, replacements)
    return from_torch(torch.randn(signature.shape, device=device), signature)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_model_call(model, device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model.to(device)
    x = make_input(model, device)
    before = x.copy(deep=True)
    output = model(x)
    handshake_dataarray(output, model.output_coords(x))
    xr.testing.assert_identical(x.e2s.as_numpy(), before.e2s.as_numpy())
    # Add model-specific normalization, ordering and numerical assertions here.


def test_model_iter(model):
    x = make_input(model)
    if model.stochastic:
        model.set_rng(42)
    expected, state = model.initialize(x)
    if model.stochastic:
        model.set_rng(42)
    iterator = model.create_iterator(x)
    first = next(iterator)
    saved = first.copy(deep=True)
    xr.testing.assert_identical(first, expected)
    handshake_dataarray(first, model.output_coords(x))
    prediction = next(iterator)
    expected_next, _ = model.step(expected, state=state)
    xr.testing.assert_identical(prediction, expected_next)
    next(iterator)
    xr.testing.assert_identical(first, saved)
    iterator.close()


def test_model_exceptions(model):
    x = make_input(model)
    with pytest.raises(ValueError):
        model(x.transpose(*reversed(x.dims)))


def test_model_conformance(model):
    # P1/P11: check the required public members, including sources and forcing.
    # Structural protocol checks do not validate signatures or runtime behavior.
    assert isinstance(model, PrognosticModel)
    assert isinstance(model.stochastic, bool)
    # Probes forecasts-only iteration, explicit state/replay, slots and forcing.
    # Keep numerical wrapper tests too; generic probes cannot validate core math.
    skipped = check_prognostic_contract(model)
    expected = (
        [] if model.stochastic else ["P14: model does not declare itself stochastic"]
    )
    assert skipped == expected


def test_model_signatures(model):
    # P24: also document the wrapper's intended fixed signature explicitly.
    for name in ("__call__", "initialize", "step", "create_iterator"):
        parameters = inspect.signature(getattr(model, name)).parameters
        assert all(
            p.kind != inspect.Parameter.VAR_POSITIONAL for p in parameters.values()
        )
        if name == "step":
            assert parameters["state"].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD


# Forced/multi-slot cases: supply input slots then full forcing windows to
# initialize/create_iterator. Compare send(new_forcing) with step(*outputs,
# *dynamic_forcing, state=state); normalize a single array to (array,) first.
# Cover static forcing omitted after initialization, missing/wrong forcing,
# additional available frames, complete multi-lead chunks, and source slot order.
# Hook tests: rear runs on first forecast; front starts only before step; both
# results feed recurrence. In-place front edits intentionally change prior yields.
# Seeded tests: compare reset rollouts, differing seeds, reset=False continuation,
# global RNG before/after, and replay unaffected by model.set_rng mid-rollout.

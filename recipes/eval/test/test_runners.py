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

"""EnsembleRunner: member batching and per-member perturbation."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from src.runners.ensemble import EnsembleRunner

from earth2studio.io import KVBackend
from earth2studio.perturbation import Gaussian
from earth2studio.run import Pipeline, PrognosticRunner, work_items

T0 = np.datetime64("2024-01-01T00:00:00")
H6 = np.timedelta64(6, "h")


def _steps(runner, item):
    return [step["forecast"] for step in runner.run_item(item)]


def test_deterministic_item_matches_prognostic_runner(prognostic, data_source) -> None:
    item = work_items([T0], 2 * H6)[0]
    torch.manual_seed(0)
    ours = _steps(EnsembleRunner(prognostic, data_source), item)
    torch.manual_seed(0)
    theirs = _steps(PrognosticRunner(prognostic, data_source), item)
    assert len(ours) == len(theirs) == 3
    for a, b in zip(ours, theirs):
        assert a.dims == b.dims
        np.testing.assert_array_equal(a.values, b.values)


def test_ensemble_item_carries_members_on_leading_axis(prognostic, data_source) -> None:
    item = work_items([T0], 2 * H6, members=[3, 5], member_batch=2)[0]
    steps = _steps(EnsembleRunner(prognostic, data_source), item)
    assert len(steps) == 3
    for step in steps:
        assert step.dims[0] == "ensemble"
        np.testing.assert_array_equal(step["ensemble"].values, [3, 5])
    # Without a perturbation every member is the same persistence forecast.
    np.testing.assert_array_equal(
        steps[-1].isel(ensemble=0).values, steps[-1].isel(ensemble=1).values
    )


def _perturbations() -> list:
    """Gaussian always; SphericalGaussian only when its optional extra imports."""
    cases = [Gaussian(noise_amplitude=0.1)]
    try:
        from earth2studio.perturbation import SphericalGaussian

        cases.append(SphericalGaussian(noise_amplitude=0.1))
    except Exception:  # noqa: BLE001, S110 - optional dependency group missing
        pass
    return cases


@pytest.mark.parametrize("perturbation", _perturbations())
def test_perturbation_is_per_member_and_batch_invariant(
    prognostic, data_source, perturbation
) -> None:
    runner = EnsembleRunner(prognostic, data_source, perturbation=perturbation, seed=7)
    batched = _steps(
        runner, work_items([T0], H6, members=[0, 1, 2], member_batch=3)[0]
    )[-1]
    singles = [
        _steps(runner, item)[-1]
        for item in work_items([T0], H6, members=[0, 1, 2], member_batch=1)
    ]
    for m, single in enumerate(singles):
        np.testing.assert_allclose(
            batched.isel(ensemble=m).values, single.isel(ensemble=0).values
        )
    # Members differ from one another.
    assert not np.allclose(
        batched.isel(ensemble=0).values, batched.isel(ensemble=1).values
    )


def test_pipeline_writes_members_into_one_store(prognostic, data_source) -> None:
    store = KVBackend()
    runner = EnsembleRunner(prognostic, data_source, perturbation=Gaussian(0.1))
    items = work_items([T0, T0 + np.timedelta64(1, "D")], H6, members=4, member_batch=2)
    Pipeline(runner, output={"forecast": store}, device="cpu").run(items)
    assert tuple(store.dims["t2m"]) == ("ensemble", "time", "lead_time", "lat", "lon")
    assert store["t2m"].shape[:3] == (4, 2, 2)
    assert not torch.isnan(store["t2m"]).any()

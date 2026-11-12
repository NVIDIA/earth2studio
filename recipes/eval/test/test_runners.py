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

"""The ``forecast`` runner: the library's PrognosticRunner wired by the recipe.

Member batching, per-member perturbation and seeding are library behaviour
(EXECUTION_CONTRACT X5 and X6).  These tests pin the layout the recipe's
stores, online scorer and campaign configs rely on, and the builder's wiring.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import open_dict
from src.runners import forecast

from earth2studio.io import KVBackend
from earth2studio.models.px import Persistence
from earth2studio.perturbation import Gaussian
from earth2studio.run import Pipeline, PrognosticRunner, work_items
from earth2studio.run.work import member_seed
from test.conftest import VARIABLES

T0 = np.datetime64("2024-01-01T00:00:00")
H6 = np.timedelta64(6, "h")


class StochasticPersistence(Persistence):
    """Persistence that declares itself stochastic and records its seeds."""

    stochastic = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seeds: list[int] = []

    def set_rng(self, seed: int, reset: bool = True) -> None:
        self.seeds.append(int(seed))


def _steps(runner, item):
    return [step["forecast"] for step in runner.run_item(item)]


def test_runner_spec_resolves_names_targets_and_rejects_the_rest(base_cfg) -> None:
    from omegaconf import OmegaConf
    from src.runners import is_window_mean, runner_spec

    assert runner_spec(OmegaConf.create({}))["_target_"] == "src.runners.forecast"
    assert runner_spec(OmegaConf.create({"runner": "stormscope"})) == {
        "_target_": "src.runners.stormscope.stormscope"
    }
    spec = runner_spec(
        OmegaConf.create({"runner": {"name": "window_mean", "window": "30D"}})
    )
    assert spec == {"_target_": "src.runners.seasonal.window_mean", "window": "30D"}
    assert is_window_mean(OmegaConf.create({"runner": {"name": "window_mean"}}))
    assert not is_window_mean(base_cfg)
    custom = {"_target_": "my_pkg.my_runner", "option": 1}
    assert runner_spec(OmegaConf.create({"runner": custom})) == custom
    with pytest.raises(ValueError, match="Unknown built-in runner"):
        runner_spec(OmegaConf.create({"runner": {"name": "windowmean"}}))
    with pytest.raises(ValueError, match="name or a mapping"):
        runner_spec(OmegaConf.create({"runner": ["forecast"]}))
    with pytest.raises(ValueError, match="needs 'name' or '_target_'"):
        runner_spec(OmegaConf.create({"runner": {"window": "30D"}}))


def test_builder_wires_the_library_runner(base_cfg) -> None:
    runner = forecast(base_cfg)
    assert isinstance(runner, PrognosticRunner)
    assert runner.seed == 42
    assert runner.perturbation is None  # ensemble_size=1 ignores the block
    assert runner.source_roles["data"] is runner.source
    with open_dict(base_cfg):
        base_cfg.ensemble_size = 2
        base_cfg.perturbation = {
            "_target_": "earth2studio.perturbation.Gaussian",
            "noise_amplitude": 0.1,
        }
    ensemble = forecast(base_cfg)
    assert isinstance(ensemble.perturbation, Gaussian)


def test_ensemble_item_carries_members_on_leading_axis(prognostic, data_source) -> None:
    item = work_items([T0], 2 * H6, members=[3, 5], member_batch=2)[0]
    steps = _steps(PrognosticRunner(prognostic, data_source), item)
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
    runner = PrognosticRunner(
        prognostic, data_source, perturbation=perturbation, seed=7
    )
    batched = _steps(
        runner, work_items([T0], H6, members=[0, 1, 2], member_batch=3)[0]
    )[-1]
    singles = [
        _steps(runner, item)[-1]
        for item in work_items([T0], H6, members=[0, 1, 2], member_batch=1)
    ]
    for m, single in enumerate(singles):
        # One member runs without the axis; Pipeline adds it when storing.
        assert "ensemble" not in single.dims
        np.testing.assert_allclose(batched.isel(ensemble=m).values, single.values)
    # Members differ from one another.
    assert not np.allclose(
        batched.isel(ensemble=0).values, batched.isel(ensemble=1).values
    )


def test_pipeline_writes_members_into_one_store(prognostic, data_source) -> None:
    store = KVBackend()
    runner = PrognosticRunner(prognostic, data_source, perturbation=Gaussian(0.1))
    items = work_items([T0, T0 + np.timedelta64(1, "D")], H6, members=4, member_batch=2)
    Pipeline(runner, output={"forecast": store}, device="cpu").run(items)
    assert tuple(store.dims["t2m"]) == ("ensemble", "time", "lead_time", "lat", "lon")
    assert store["t2m"].shape[:3] == (4, 2, 2)
    assert not torch.isnan(store["t2m"]).any()


def test_stochastic_batch_is_seeded_once_from_its_first_member(
    data_source, small_grid
) -> None:
    model = StochasticPersistence(variable=VARIABLES, domain_coords=small_grid)
    runner = PrognosticRunner(model, data_source, seed=7)
    assert runner.supports_member_batching
    steps = _steps(runner, work_items([T0], H6, members=[0, 1], member_batch=2)[0])
    assert steps[-1].sizes["ensemble"] == 2
    assert model.seeds == [member_seed(7, T0, 0)]
    # member_batch=1 seeds every member alone, so each reproduces whatever the others.
    model.seeds.clear()
    for item in work_items([T0], H6, members=[0, 1], member_batch=1):
        assert len(_steps(runner, item)) == 2
    assert model.seeds == [member_seed(7, T0, 0), member_seed(7, T0, 1)]

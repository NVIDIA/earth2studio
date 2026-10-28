# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

from functools import partial

import cbottle
import pytest
import torch
from cbottle.datasets.base import BatchInfo
from cbottle.diffusion_samplers import edm_sampler, edm_sampler_from_sigma
from cbottle.inference import (
    CBottle3d,
    DistilledSuperResolutionModel,
    MixtureOfExpertsDenoiser,
)
from torch import Tensor

from earth2studio.models._cbottle import (
    DistilledSuperResolutionSampler,
    SuperResolutionSampler,
    infill_cbottle,
    sample_cbottle,
)
from earth2studio.models._diffusion import SOLVERS
from earth2studio.models.dx.cbottle_infill import CBottleInfill
from earth2studio.models.dx.cbottle_sr import CHANNEL_TO_VARIABLE, CBottleSR


class ScalarDenoiser(torch.nn.Module):
    sigma_min = 0.002
    sigma_max = 80.0

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        return torch.tanh(x / (1 + sigma.square()))

    def round_sigma(self, sigma: Tensor) -> Tensor:
        return sigma


@pytest.fixture(scope="module")
def core_model() -> CBottle3d:
    config = cbottle.config.models.ModelConfigV1()
    config.model_channels = 4
    config.label_dim = 1024
    config.out_channels = 45
    config.condition_channels = 1
    config.level = 2
    net = MixtureOfExpertsDenoiser(
        [cbottle.models.get_model(config)],
        (),
        batch_info=BatchInfo(CBottleInfill.output_variables),
    ).eval()
    return CBottle3d(net, num_steps=3, sigma_max=3, channels_last=False, device="cuda")


def make_batch(core: CBottle3d, batch_size: int = 2) -> dict[str, Tensor]:
    pixels = core.net.domain.numel()
    labels = torch.zeros(batch_size, 1024, device="cuda")
    labels[:, 0] = 1
    return {
        "target": torch.randn(batch_size, 45, 1, pixels, device="cuda"),
        "condition": torch.randn(batch_size, 1, 1, pixels, device="cuda"),
        "labels": labels,
        "second_of_day": torch.full((batch_size, 1), 3600.0, device="cuda"),
        "day_of_year": torch.full((batch_size, 1), 100.0, device="cuda"),
    }


class TestConsistency:
    @pytest.mark.parametrize(
        "batch_size,time_stepper,dtype,scale_latents",
        [
            (1, "heun", torch.float32, False),
            (3, "heun", torch.float64, False),
            (3, "euler", torch.float32, False),
            (2, "heun", torch.float32, True),
        ],
    )
    def test_native_sampler(
        self, batch_size, time_stepper, dtype, scale_latents
    ) -> None:
        net = ScalarDenoiser().cuda()
        latents = torch.randn(batch_size, 3, 2, 7, device="cuda", dtype=dtype)
        settings = dict(num_steps=5, sigma_min=0.02, sigma_max=3)
        evaluations = []

        def record_evaluation(module, args, output):
            evaluations.append(args[1].clone())

        handle = net.register_forward_hook(record_evaluation)
        torch.manual_seed(27)
        if scale_latents:
            expected = edm_sampler(net, latents, **settings)
        else:
            expected = edm_sampler_from_sigma(
                net, latents, time_stepper=time_stepper, **settings
            )
        expected_next_noise = torch.randn_like(expected)
        native_evaluations = len(evaluations)
        evaluations.clear()
        torch.manual_seed(27)
        actual = sample_cbottle(
            net,
            latents,
            time_stepper=time_stepper,
            scale_latents=scale_latents,
            **settings,
        )
        actual_next_noise = torch.randn_like(actual)
        handle.remove()
        if time_stepper == "heun":
            assert native_evaluations == 2 * settings["num_steps"] - 1
            assert len(evaluations) == 2 * settings["num_steps"]
            # PhysicsNeMo evaluates its masked terminal corrector at the last sigma.
            torch.testing.assert_close(evaluations[-1], evaluations[-2])
        else:
            assert native_evaluations == len(evaluations) == settings["num_steps"]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            actual_next_noise, expected_next_noise, atol=0, rtol=0
        )

    @pytest.mark.parametrize("distilled", [False, True])
    def test_sr_default(self, distilled) -> None:
        model_type = (
            DistilledSuperResolutionSampler if distilled else SuperResolutionSampler
        )
        helper = model_type(
            ScalarDenoiser(),
            BatchInfo(["tas"]),
            hpx_level=2,
            hpx_lr_level=1,
            patch_size=4,
            overlap_size=1,
            sigma_max=3,
            num_steps=4,
        ).cuda()
        latents = torch.randn(2, 1, 12, device="cuda")
        torch.manual_seed(9)
        if distilled:
            expected = DistilledSuperResolutionModel._sample(
                helper, helper.net, latents
            )
        else:
            expected = edm_sampler(helper.net, latents, num_steps=4, sigma_max=3)
        expected_next_noise = torch.randn_like(expected)
        torch.manual_seed(9)
        actual = helper._sample(helper.net, latents)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.randn_like(actual), expected_next_noise, atol=0, rtol=0
        )

    @pytest.mark.parametrize("missing", [False, True])
    def test_observed_infill(self, core_model, missing) -> None:
        batch = make_batch(core_model)
        if missing:
            batch["target"][:, 1::2] = torch.nan
        observed_inputs = []

        def record_observed_input(module, args):
            observed_inputs.append(args[0][:, ::2, :, :16].clone())

        handle = core_model.net.register_forward_pre_hook(record_observed_input)
        torch.manual_seed(31)
        expected, _ = core_model.infill(batch, bf16=False)
        if not missing:
            expected = expected[1]
        native_observed_inputs = observed_inputs.copy()
        observed_inputs.clear()
        expected_next_noise = torch.randn_like(expected)
        torch.manual_seed(31)
        actual, _ = infill_cbottle(core_model, batch, bf16=False)
        handle.remove()
        assert len(observed_inputs) == len(native_observed_inputs) + int(missing)
        # A one-ULP sigma change must not select a different observation level.
        for actual_input, expected_input in zip(
            observed_inputs, native_observed_inputs
        ):
            torch.testing.assert_close(actual_input, expected_input, atol=0, rtol=0)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.randn_like(actual), expected_next_noise, atol=0, rtol=0
        )

    def test_observed_infill_off_grid_sigma(self, core_model) -> None:
        batch = make_batch(core_model)
        batch["target"][:, 1::2] = torch.nan
        evaluations = []

        def record_sigma(module, args):
            evaluations.append(args[1].clone())

        handle = core_model.net.register_forward_pre_hook(record_sigma)
        infill_cbottle(core_model, batch, options={"alpha": 0.5}, bf16=False)
        handle.remove()
        # Half-step Heun intentionally evaluates between the first two grid levels.
        midpoint = evaluations[0] + 0.5 * (evaluations[2] - evaluations[0])
        torch.testing.assert_close(evaluations[1], midpoint, atol=0, rtol=0)
        assert evaluations[2] < evaluations[1] < evaluations[0]

    @pytest.mark.parametrize("guided", [False, True])
    def test_core_sampler_hook(self, core_model, guided) -> None:
        batch = make_batch(core_model)

        def guidance(data, logits, x, denoised, sigma):
            return sigma * torch.tanh(x)

        settings = dict(
            seed=13,
            bf16=False,
            guidance_fn=guidance if guided else None,
            guidance_scale=0.01,
        )
        torch.manual_seed(18)
        expected, _ = core_model._sample_with_latents(batch, **settings)
        expected_next_noise = torch.randn_like(expected)
        torch.manual_seed(18)
        actual, _ = core_model._sample_with_latents(
            batch, sampler=partial(sample_cbottle), **settings
        )
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.randn_like(actual), expected_next_noise, atol=0, rtol=0
        )


class TestDiagnostics:
    @pytest.mark.parametrize("solver", tuple(SOLVERS))
    def test_solver_alternatives(self, solver) -> None:
        net = ScalarDenoiser().cuda()
        latents = torch.randn(2, 3, 2, 7, device="cuda")
        options = {"S_churn": 0.5} if solver.startswith("edm_stochastic_") else {}
        torch.manual_seed(2)
        actual = sample_cbottle(
            net, latents, solver=solver, options=options, num_steps=5, sigma_max=3
        )
        assert actual.shape == latents.shape
        assert torch.isfinite(actual).all()
        torch.manual_seed(2)
        repeated = sample_cbottle(
            net, latents, solver=solver, options=options, num_steps=5, sigma_max=3
        )
        torch.testing.assert_close(actual, repeated, atol=0, rtol=0)


class TestConstructor:
    @pytest.mark.parametrize("solver", ["heun", "edm_stochastic_heun", "dpmpp_2m"])
    def test_step_count_options_rejected(self, solver):
        with pytest.raises(ValueError, match="num_steps.*solver_options"):
            sample_cbottle(
                ScalarDenoiser(),
                torch.ones(1, 1, 2, 2),
                solver=solver,
                num_steps=4,
                options={"num_steps": 8},
            )

    def test_supplied_sr_helper_preserved(self) -> None:
        helper = DistilledSuperResolutionModel(
            ScalarDenoiser(),
            BatchInfo(list(CHANNEL_TO_VARIABLE)),
            hpx_level=6,
            hpx_lr_level=6,
            patch_size=4,
            overlap_size=1,
            window_function="uniform",
        )
        model = CBottleSR(helper, lat_lon=False)
        assert model.sr_model is helper
        with pytest.raises(TypeError, match="CBottleSR.load_model"):
            CBottleSR(helper, lat_lon=False, solver="euler")


class TestCompile:
    def test_compiled_predictor(self) -> None:
        net = ScalarDenoiser().cuda()
        latents = torch.randn(2, 3, 2, 7, device="cuda")
        settings = dict(num_steps=4, sigma_max=3)
        torch.manual_seed(42)
        expected = sample_cbottle(net, latents, **settings)
        compiled = torch.compile(net, fullgraph=True)
        torch.manual_seed(42)
        actual = sample_cbottle(compiled, latents, **settings)
        torch.testing.assert_close(actual, expected)
        with torch._dynamo.config.patch(error_on_recompile=True):
            torch.manual_seed(42)
            repeated = sample_cbottle(compiled, latents, **settings)
        torch.testing.assert_close(repeated, actual)

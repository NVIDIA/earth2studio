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

import pytest
import torch
from physicsnemo.diffusion.noise_schedulers import (
    EDMNoiseScheduler,
    RectifiedFlowNoiseScheduler,
)
from physicsnemo.diffusion.samplers import sample
from torch import Tensor

from earth2studio.models._diffusion import solver_options


class ResidualModel(torch.nn.Module):
    sigma_min = 0.0
    sigma_max = 1000.0

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        sigma: Tensor,
        class_labels: Tensor | None = None,
    ) -> Tensor:
        return 0.1 * torch.tanh(x) + 0.03 * condition[:, :1]

    def round_sigma(self, sigma: Tensor | float) -> Tensor:
        return torch.as_tensor(sigma)


class ClampedSigmaResidualModel(ResidualModel):
    def round_sigma(self, sigma: Tensor | float) -> Tensor:
        # A supported noise floor must not replace the clean sample endpoint.
        return torch.as_tensor(sigma).clamp_min(0.05)


def predict_x0(x: Tensor, t: Tensor) -> Tensor:
    sigma = t.reshape(-1, 1, 1, 1)  # (B, 1, 1, 1)
    return torch.tanh(x / (1 + sigma.square()))


@pytest.mark.parametrize(
    "scheduler,solver",
    [
        (EDMNoiseScheduler(sigma_max=3), "exponential_euler"),
        (EDMNoiseScheduler(sigma_max=3), "dpmpp_2m"),
        (EDMNoiseScheduler(sigma_max=3), "dpmpp_2m_unic2"),
        (EDMNoiseScheduler(sigma_max=3), "edm_stochastic_exponential_euler"),
        (RectifiedFlowNoiseScheduler(t_max=0.99), "exponential_euler"),
        (RectifiedFlowNoiseScheduler(t_max=0.99), "dpmpp_2m"),
        (RectifiedFlowNoiseScheduler(t_max=0.99), "dpmpp_2m_unic2"),
    ],
)
class TestConsistency:
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_solver_configuration(self, scheduler, solver, device):
        device = torch.device(device)
        x = torch.randn(2, 3, 4, 5, dtype=torch.float64, device=device)
        denoiser = scheduler.get_denoiser(x0_predictor=predict_x0)
        bias, bias_int, slope = scheduler.get_linear_denoiser("x0")
        explicit = dict(bias_fn=bias, bias_int_fn=bias_int, slope_fn=slope)
        if solver in ("dpmpp_2m", "dpmpp_2m_unic2"):
            explicit["lambda_fn"] = lambda t: scheduler.snr(t).log()
        options = solver_options(scheduler, solver)
        torch.manual_seed(17)
        expected = sample(denoiser, x, scheduler, 8, solver, solver_options=explicit)
        torch.manual_seed(17)
        actual = sample(denoiser, x, scheduler, 8, solver, solver_options=options)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)
        torch.manual_seed(17)
        repeated = sample(denoiser, x, scheduler, 8, solver, solver_options=options)
        torch.testing.assert_close(repeated, actual)


class TestConstructor:
    @pytest.mark.parametrize("solver", ["euler", "heun", "edm_stochastic_heun"])
    def test_explicit_options(self, solver):
        supplied = {"alpha": 0.5} if solver != "euler" else {}
        assert solver_options(EDMNoiseScheduler(), solver, options=supplied) == supplied

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("num_steps", [3, 8])
    def test_stochastic_options(self, device, num_steps):
        scheduler = EDMNoiseScheduler(sigma_max=3)
        denoiser = scheduler.get_denoiser(x0_predictor=predict_x0)
        x = torch.ones(2, 3, 4, 5, dtype=torch.float64, device=device)
        supplied = {"S_churn": 0.4, "S_noise": 1.1}
        configured = solver_options(
            scheduler,
            "edm_stochastic_heun",
            options=supplied,
            num_steps=num_steps,
            defaults={"S_churn": 0.1},
        )
        torch.manual_seed(17)
        expected = sample(
            denoiser,
            x,
            scheduler,
            num_steps,
            "edm_stochastic_heun",
            solver_options=supplied | {"num_steps": num_steps},
        )
        torch.manual_seed(17)
        actual = sample(
            denoiser,
            x,
            scheduler,
            num_steps,
            "edm_stochastic_heun",
            solver_options=configured,
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "solver", ["unknown_solver", "euler_maruyama", "stochastic_runge_kutta"]
    )
    def test_unknown_solver(self, solver):
        with pytest.raises(ValueError, match=solver):
            solver_options(EDMNoiseScheduler(), solver)

    @pytest.mark.parametrize(
        "solver", ["euler", "heun", "edm_stochastic_heun", "dpmpp_2m"]
    )
    @pytest.mark.parametrize("value", [None, 4, 18])
    def test_step_count_options_rejected(self, solver, value):
        with pytest.raises(ValueError, match="num_steps.*solver_options"):
            solver_options(
                EDMNoiseScheduler(), solver, options={"num_steps": value}, num_steps=4
            )

    @pytest.mark.parametrize("value", [0, -1, 2.5, True])
    def test_invalid_step_count(self, value):
        with pytest.raises(ValueError, match="positive integer"):
            solver_options(EDMNoiseScheduler(), "euler", num_steps=value)

    def test_stochastic_defaults_do_not_leak_to_other_methods(self):
        assert solver_options(
            EDMNoiseScheduler(),
            "heun",
            num_steps=4,
            defaults={"S_churn": 10.0},
            options={"alpha": 0.5},
        ) == {"alpha": 0.5}


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("residual_type", [ResidualModel, ClampedSigmaResidualModel])
@pytest.mark.parametrize(
    "sampler_type,solver,mean_conditioning",
    [
        ("deterministic", "euler", False),
        ("deterministic", "heun", False),
        ("deterministic", "euler", True),
        ("deterministic", "heun", True),
        ("stochastic", None, False),
        ("stochastic", None, True),
    ],
)
def test_corrdiff_legacy_parity(
    device, residual_type, sampler_type, solver, mean_conditioning
):
    from functools import partial

    from physicsnemo.diffusion.generate.legacy_generate import diffusion_step
    from physicsnemo.diffusion.samplers.legacy_deterministic_sampler import (
        deterministic_sampler,
    )
    from physicsnemo.diffusion.samplers.legacy_stochastic_sampler import (
        stochastic_sampler,
    )

    from earth2studio.models.dx.corrdiff import CorrDiff

    model = CorrDiff(
        ["t2m"],
        ["t2m"],
        residual_type(),
        ResidualModel(),
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
        number_of_steps=8,
        solver=solver,
        sampler_type=sampler_type if solver is None else None,
        hr_mean_conditioning=mean_conditioning and sampler_type == "stochastic",
    ).to(device)
    condition = torch.randn(2, 1, 8, 8, device=device)
    mean = torch.randn(1, 1, 8, 8, device=device) if mean_conditioning else None
    seeds = [5, 29]
    legacy = (
        partial(deterministic_sampler, num_steps=8, solver=solver)
        if sampler_type == "deterministic"
        else partial(stochastic_sampler, num_steps=8)
    )
    calls = []
    sigma_shapes = []

    def record_condition(module, args, output):
        calls.append((args[0].shape[0], args[1].clone()))
        sigma_shapes.append(args[2].shape)

    handle = model.residual_model.register_forward_hook(record_condition)
    expected = diffusion_step(
        model.residual_model,
        legacy,
        (8, 8),
        1,
        [seeds],
        condition,
        1,
        torch.device(device),
        mean_hr=mean,
    )
    legacy_calls = calls.copy()
    calls.clear()
    sigma_shapes.clear()
    with torch.inference_mode():
        actual = model._sample_residual(condition, seeds, mean)
    handle.remove()
    with torch.inference_mode():
        repeated = model._sample_residual(condition, seeds, mean)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(repeated, actual, atol=0, rtol=0)
    if sampler_type == "deterministic":
        assert sigma_shapes and all(shape == torch.Size([]) for shape in sigma_shapes)
        # PhysicsNeMo Heun evaluates its masked terminal corrector.
        assert len(calls) == len(legacy_calls) + (solver == "heun")
        assert all(batch_size == len(seeds) for batch_size, _ in calls)
        for _, observed_condition in calls:
            torch.testing.assert_close(observed_condition, legacy_calls[0][1])
    else:
        assert all(batch_size == 1 for batch_size, _ in calls)
        evaluations_per_member = len(calls) // len(seeds)
        for index, (_, observed_condition) in enumerate(calls):
            member = index // evaluations_per_member
            torch.testing.assert_close(
                observed_condition, legacy_calls[0][1][member : member + 1]
            )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "solver",
    [
        "exponential_euler",
        "edm_stochastic_exponential_euler",
        "dpmpp_2m",
        "dpmpp_2m_unic2",
    ],
)
def test_corrdiff_independent_members(device, solver, monkeypatch):
    import earth2studio.models.dx.corrdiff as corrdiff_module
    from earth2studio.models.dx.corrdiff import CorrDiff

    model = CorrDiff(
        ["t2m"],
        ["t2m"],
        ResidualModel(),
        ResidualModel(),
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        torch.linspace(0, 1, 8),
        torch.zeros(1),
        torch.ones(1),
        torch.zeros(1),
        torch.ones(1),
        number_of_steps=8,
        solver=solver,
        hr_mean_conditioning=False,
        solver_options=(
            {"S_churn": 0.2} if solver.startswith("edm_stochastic") else None
        ),
    ).to(device)
    condition = torch.randn(2, 1, 8, 8, device=device)
    batch_sizes = []

    def record_batch_size(module, args, output):
        batch_sizes.append(args[0].shape[0])

    handle = model.residual_model.register_forward_hook(record_batch_size)
    with torch.inference_mode():
        forward = model._sample_residual(condition, [5, 29])
        reverse = model._sample_residual(condition.flip(0), [29, 5]).flip(0)
    handle.remove()
    assert torch.isfinite(forward).all()
    torch.testing.assert_close(forward, reverse, atol=0, rtol=0)
    expected_batch_size = 1 if solver.startswith("edm_stochastic_") else 2
    assert batch_sizes and all(size == expected_batch_size for size in batch_sizes)
    if expected_batch_size == 2:
        observed_times = []

        def record_times(module, args, output):
            observed_times.append(args[2].clone())

        times = torch.tensor([0.25, 0.5], device=device, dtype=torch.float64)

        def sample_nonuniform_times(denoiser, latents, *args, **kwargs):
            return denoiser(latents, times)

        # Alternative solver callbacks must keep distinct member noise levels.
        monkeypatch.setattr(corrdiff_module, "sample", sample_nonuniform_times)
        handle = model.residual_model.register_forward_hook(record_times)
        with torch.inference_mode():
            model._sample_residual(condition, [5, 29])
        handle.remove()
        assert len(observed_times) == 1
        torch.testing.assert_close(observed_times[0], times, atol=0, rtol=0)

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

import math

import numpy as np
import pytest
import torch
from physicsnemo.diffusion import Denoiser
from physicsnemo.diffusion.samplers import (
    DPMPlusPlus2M,
    EulerSolver,
    HeunSolver,
    Solver,
)
from torch import Tensor

from earth2studio.models.nn.atlas import StochasticInterpolant


def _velocity(
    I: Tensor,  # noqa: E741
    t: Tensor,
    condition: Tensor,
    c_S: Tensor | None = None,
    c_O: Tensor | None = None,
    c_I: Tensor | None = None,
) -> Tensor:
    t = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    if c_I is not None:
        c_I = c_I.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        c_O = c_O.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        c_S = c_S.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        return c_O * torch.tanh(c_I * I + condition + t) + c_S * condition
    return torch.tanh(I + condition + t)


def _reference_drift(
    interpolant: StochasticInterpolant,
    x: Tensor,
    t: Tensor,
    condition: Tensor,
    initial: Tensor,
    correction: bool,
    g: Tensor,
) -> Tensor:
    c_S = interpolant.alpha_dot(t)
    c_O = torch.sqrt(interpolant.beta_dot(t) ** 2 + interpolant.sigma_dot(t) ** 2 * t)
    c_I = 1.0 / torch.sqrt(
        interpolant.alpha(t) ** 2
        + interpolant.beta(t) ** 2
        + interpolant.sigma(t) ** 2 * t
    )
    drift = _velocity(x, t, condition, c_S, c_O, c_I)
    if correction:
        t = t.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        alpha, beta, sigma = (
            interpolant.alpha(t),
            interpolant.beta(t),
            interpolant.sigma(t),
        )
        beta_dot = interpolant.beta_dot(t)
        A = 1.0 / (t * sigma * (beta_dot * sigma - beta * interpolant.sigma_dot(t)))
        c = (
            beta_dot * x
            + (beta * interpolant.alpha_dot(t) - beta_dot * alpha) * initial
        )
        score = A * (beta * drift - c)
        drift = drift + 0.5 * (torch.pow(g, 2) - torch.pow(sigma, 2)) * score
    return drift


def _reference_step(
    interpolant: StochasticInterpolant,
    method: str,
    x: Tensor,
    t: Tensor,
    index: int,
    condition: Tensor,
    initial: Tensor,
) -> Tensor:
    t0 = t[:, index].view(-1, 1)  # (B, 1)
    t1 = t[:, index + 1].view(-1, 1)  # (B, 1)
    dt = t1 - t0
    h = dt.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    amplitude = interpolant.sigma if index == 0 else interpolant.g
    correction = index != 0 and interpolant.compute_correction
    g0 = amplitude(t0).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    if method == "em":
        drift = _reference_drift(interpolant, x, t0, condition, initial, correction, g0)
        x = x + h * drift
        return x + torch.sqrt(h) * g0 * interpolant.noise_sampler(
            x.shape, device=x.device
        )
    if method == "rk_roberts":
        g1 = amplitude(t1).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        sign = np.random.binomial(n=1, p=0.5, size=1).item() - 0.5
        noise = torch.sqrt(h) * interpolant.noise_sampler(x.shape, device=x.device)
        drift = _reference_drift(interpolant, x, t0, condition, initial, correction, g0)
        y1 = h * drift + g0 * (noise - torch.sqrt(h) * sign)
        drift = _reference_drift(
            interpolant, x + y1, t1, condition, initial, correction, g1
        )
        y2 = h * drift + g1 * (noise + torch.sqrt(h) * sign)
        return x + 0.5 * (y1 + y2)
    if method == "rk_rossler_a1":
        c, beta1, beta2 = (1.0, 0.0), (1.0, 0.0), (-1.0, 1.0)
    else:
        c, beta1, beta2 = (1.0 / 3.0, 1.0), (0.0, 1.0), (-1.5, 1.5)
    t_stage = t0 + 0.75 * dt
    g1 = amplitude(t0 + c[0] * dt).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    g2 = amplitude(t0 + c[1] * dt).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    g3 = amplitude(t_stage).view(-1, 1, 1, 1)  # (B, 1, 1, 1)
    noise = torch.sqrt(h) * interpolant.noise_sampler(x.shape, device=x.device)
    auxiliary = 0.5 * (
        noise
        + (1.0 / math.sqrt(3.0))
        * torch.sqrt(h)
        * interpolant.noise_sampler(x.shape, device=x.device)
    )
    drift1 = _reference_drift(interpolant, x, t0, condition, initial, correction, g0)
    stage = x + 0.75 * h * drift1 + 1.5 * g1 * auxiliary
    drift2 = _reference_drift(
        interpolant, stage, t_stage, condition, initial, correction, g3
    )
    return (
        x
        + (1.0 / 3.0) * h * drift1
        + (2.0 / 3.0) * h * drift2
        + (beta1[0] * noise + beta2[0] * auxiliary) * g1
        + (beta1[1] * noise + beta2[1] * auxiliary) * g2
    )


def _reference_sample(
    interpolant: StochasticInterpolant,
    x: Tensor,
    t: Tensor,
    condition: Tensor,
    keep_every: int | None,
    method: str,
) -> Tensor:
    initial = x.clone()
    out = (
        torch.zeros(
            (x.shape[0], (t.shape[1] - 1) // keep_every, *x.shape[1:]), device=x.device
        )
        if keep_every is not None
        else None
    )
    for index in range(t.shape[1] - 1):
        x = _reference_step(interpolant, method, x, t, index, condition, initial)
        if keep_every is not None and (index + 1) % keep_every == 0:
            out[:, (index + 1) // keep_every - 1] = x
    return x if out is None else out


def _diffusion(t: Tensor) -> Tensor:
    return 0.4 * (1 - t)


class TestConsistency:
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize(
        "method,correction,keep_every,dtype,distinct_times",
        [
            ("em", False, None, torch.float32, False),
            ("em", True, 1, torch.float64, True),
            ("rk_roberts", False, 2, torch.float32, False),
            ("rk_roberts", True, 10, torch.float64, True),
            ("rk_rossler_a1", False, None, torch.float32, True),
            ("rk_rossler_a1", True, 3, torch.float64, False),
            ("rk_rossler_a2", False, 1, torch.float32, False),
            ("rk_rossler_a2", True, None, torch.float64, True),
        ],
    )
    def test_native_values_and_rng(
        self, device, method, correction, keep_every, dtype, distinct_times
    ):
        interpolant = StochasticInterpolant(
            sample_method=method,
            g=_diffusion if correction else None,
            beta="beta_quad" if correction else "beta_linear",
        )
        x = torch.linspace(-0.5, 0.5, 48, device=device, dtype=dtype).reshape(
            2, 2, 3, 4
        )
        # (B, C, H, W)
        condition = x.square()
        t = torch.linspace(0, 0.9, 5, device=device).repeat(2, 1)
        if distinct_times:
            t[1] = t[1] ** 2
        np.random.seed(23)
        torch.manual_seed(17)
        expected = _reference_sample(interpolant, x, t, condition, keep_every, method)
        expected_numpy = np.random.random(4)
        expected_torch = torch.randn(4, device=device)

        np.random.seed(23)
        torch.manual_seed(17)
        actual = interpolant.sample(
            _velocity,
            x,
            t=t,
            cond={"condition": condition},
            compute_normalization=True,
            keep_every=keep_every,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.random(4), expected_numpy)
        torch.testing.assert_close(
            torch.randn(4, device=device), expected_torch, rtol=0, atol=0
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize(
        "solver", ["euler", "heun", EulerSolver, HeunSolver, DPMPlusPlus2M]
    )
    def test_custom_solver_sequential_trajectories(self, device, solver):
        interpolant = StochasticInterpolant()
        x = torch.ones(2, 2, 3, 4, device=device)
        created = []

        def factory(denoiser: Denoiser) -> Solver:
            method = {"euler": EulerSolver, "heun": HeunSolver}.get(solver, solver)
            instance = method(denoiser)
            created.append(instance)
            return instance

        for offset in (0.1, 0.7):
            condition = torch.full_like(x, offset)
            actual = interpolant.sample(
                _velocity,
                x,
                steps=4,
                cond={"condition": condition},
                compute_normalization=True,
                solver=factory,
            )
            fresh = StochasticInterpolant()
            expected = fresh.sample(
                _velocity,
                x,
                steps=4,
                cond={"condition": condition},
                compute_normalization=True,
                solver=solver,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.isfinite(actual).all()
            x = actual
        assert len(created) == 2
        assert created[0] is not created[1]

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_direct_solver_instance(self, device):
        x = torch.zeros(2, 2, 3, 4, device=device)

        def rhs(x: Tensor, t: Tensor) -> Tensor:
            return torch.ones_like(x)

        solver = EulerSolver(rhs)
        result = StochasticInterpolant().sample(
            _velocity,
            x,
            steps=4,
            solver=solver,
        )
        torch.testing.assert_close(result, torch.ones_like(x), rtol=0, atol=0)


class TestValidation:
    def test_distinct_custom_time_grid(self):
        x = torch.zeros(2, 2, 3, 4)
        t = torch.tensor([[0.0, 0.2, 1.0], [0.0, 0.3, 1.0]])
        with pytest.raises(ValueError, match="shared batch time grid"):
            StochasticInterpolant().sample(_velocity, x, t=t, solver="euler")

    def test_invalid_retention_interval(self):
        with pytest.raises(ValueError, match="keep_every must be positive"):
            StochasticInterpolant().sample(_velocity, torch.zeros(1, 2), keep_every=0)

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

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from physicsnemo.diffusion import PredictorType
    from physicsnemo.diffusion.noise_schedulers import (
        EDMNoiseScheduler,
        LinearGaussianNoiseScheduler,
    )
    from physicsnemo.diffusion.samplers.samplers import SOLVERS as _PNM_SOLVERS
except ImportError:
    OptionalDependencyFailure("corrdiff")
    PredictorType = None
    EDMNoiseScheduler = None
    LinearGaussianNoiseScheduler = None
    _PNM_SOLVERS = {}

# These wrappers integrate decreasing diffusion-time grids.
SOLVERS = {
    name: _PNM_SOLVERS[name]
    for name in (
        "euler",
        "heun",
        "edm_stochastic_euler",
        "edm_stochastic_heun",
        "exponential_euler",
        "edm_stochastic_exponential_euler",
        "dpmpp_2m",
        "dpmpp_2m_unic2",
    )
    if name in _PNM_SOLVERS
}


@check_optional_dependencies()
def solver_options(
    scheduler: LinearGaussianNoiseScheduler,
    solver: str,
    prediction_type: PredictorType = "x0",
    options: Mapping[str, Any] | None = None,
    *,
    num_steps: int | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Configure a PhysicsNeMo solver for the model's noise schedule.

    Parameters
    ----------
    scheduler : LinearGaussianNoiseScheduler
        Noise schedule that supplies the diffusion coefficients.
    solver : str
        Solver name accepted by ``physicsnemo.diffusion.samplers.sample``.
    prediction_type : PredictorType, optional
        Prediction target used to train the model, such as clean data for "x0",
        by default "x0"
    options : Mapping[str, Any] | None, optional
        Options passed to the solver to customize its behavior and override
        default settings. Set the step count through the wrapper, not through
        this mapping, by default None
    num_steps : int | None, optional
        Sampling step count supplied by the wrapper. Also controls stochastic
        noise scaling when the selected method uses it, by default None
    defaults : Mapping[str, Any] | None, optional
        Existing wrapper settings. The helper applies applicable stochastic
        noise settings before explicit options, by default None

    Returns
    -------
    dict[str, Any]
        Solver constructor options with schedule coefficients and overrides.

    Examples
    --------
    >>> from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    >>> options = solver_options(EDMNoiseScheduler(), "dpmpp_2m")
    >>> sorted(options)
    ['bias_fn', 'bias_int_fn', 'lambda_fn', 'slope_fn']
    """
    if solver not in SOLVERS:
        raise ValueError(f"Unknown solver {solver!r}; choose from {tuple(SOLVERS)}.")
    if options is not None and "num_steps" in options:
        raise ValueError(
            "Do not pass 'num_steps' in solver_options; use the wrapper's "
            "sampling step-count argument instead."
        )
    if num_steps is not None and (type(num_steps) is not int or num_steps < 1):
        raise ValueError(f"num_steps must be a positive integer, got {num_steps!r}.")
    configured: dict[str, Any] = {}
    if solver.startswith("edm_stochastic_"):
        if defaults is not None:
            configured.update(
                (key, defaults[key])
                for key in ("S_churn", "S_min", "S_max", "S_noise")
                if key in defaults
            )
        if num_steps is not None:
            configured["num_steps"] = num_steps
    if solver.startswith("edm_stochastic_") and not isinstance(
        scheduler, EDMNoiseScheduler
    ):
        configured.update(
            sigma_fn=scheduler.sigma,
            sigma_inv_fn=scheduler.sigma_inv,
            diffusion_fn=scheduler.diffusion,
        )
        if solver == "edm_stochastic_exponential_euler":
            configured["alpha_fn"] = scheduler.alpha
    if solver in (
        "exponential_euler",
        "edm_stochastic_exponential_euler",
        "dpmpp_2m",
        "dpmpp_2m_unic2",
    ):
        bias, bias_int, slope = scheduler.get_linear_denoiser(prediction_type)
        configured.update(bias_fn=bias, bias_int_fn=bias_int, slope_fn=slope)
        if solver in ("dpmpp_2m", "dpmpp_2m_unic2"):
            configured["lambda_fn"] = lambda t: scheduler.snr(t).log()
    if options is not None:
        configured.update(options)
    return configured


@contextmanager
def seeded_rng(generator: torch.Generator | None) -> Iterator[None]:
    """Continue a member's noise stream without changing the caller's RNG state."""
    if generator is None:
        yield
        return
    device = generator.device
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        if device.type == "cuda":
            torch.cuda.set_rng_state(generator.get_state(), device)
        else:
            torch.set_rng_state(generator.get_state())
        yield

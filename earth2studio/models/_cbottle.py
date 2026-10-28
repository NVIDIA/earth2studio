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

from collections.abc import Callable
from typing import Any, Protocol, cast

import torch
from cbottle.diffusion_samplers import edm_sampler_steps
from cbottle.inference import (
    CBottle3d,
    Coords,
    DistilledSuperResolutionModel,
    SuperResolutionModel,
    _build_labels,
)
from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
from physicsnemo.diffusion.samplers import sample

from earth2studio.models._diffusion import solver_options as configure_solver


class _ScalarDenoiser(Protocol):
    sigma_min: float
    sigma_max: float
    round_sigma: Callable[[torch.Tensor], torch.Tensor]

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor: ...


def sample_cbottle(
    net: _ScalarDenoiser,
    latents: torch.Tensor,
    *,
    solver: str | None = None,
    options: dict[str, Any] | None = None,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    time_stepper: str = "heun",
    randn_like: Callable = torch.randn_like,
    scale_latents: bool = False,
    distilled: bool = False,
) -> torch.Tensor:
    """Adapt C-Bottle's scalar noise interface to PhysicsNeMo sampling."""
    if randn_like is not torch.randn_like:
        raise ValueError("C-Bottle sampling requires torch.randn_like.")
    sigma_max_requested = sigma_max
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    scheduler = EDMNoiseScheduler(sigma_min=sigma_min, sigma_max=sigma_max)
    if distilled:
        if solver is None:
            raise ValueError(
                "Distilled PhysicsNeMo sampling requires an explicit solver."
            )
        times = torch.tensor(
            [sigma_max, sigma_max_requested / 80 * 1.5, 0],
            dtype=latents.dtype,
            device=latents.device,
        )
        num_steps = 2
        selected = solver
    else:
        times = scheduler.timesteps(
            num_steps,
            device=latents.device,
            dtype=torch.float64,
        )
        times = torch.cat((net.round_sigma(times[:-1]), times[-1:]))
        # C-Bottle consumes one noise draw per step even when churn is zero.
        selected = solver or f"edm_stochastic_{time_stepper}"

    def predict_x0(
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        return net(x, t[0]).to(torch.float64)

    initial = latents.to(torch.float64)
    if scale_latents:
        initial = initial * times[0]
    configured = configure_solver(
        scheduler,
        selected,
        num_steps=num_steps,
        options=options,
    )
    return sample(
        scheduler.get_denoiser(x0_predictor=predict_x0),
        initial,
        scheduler,
        num_steps,
        solver=selected,
        time_steps=times,
        solver_options=configured,
    )


def infill_cbottle(
    model: CBottle3d,
    batch: dict[str, torch.Tensor],
    *,
    solver: str | None = None,
    options: dict[str, Any] | None = None,
    bf16: bool = True,
) -> tuple[torch.Tensor, Coords]:
    """Preserve C-Bottle's observed-pixel path while sampling with PhysicsNeMo."""
    batch = model._move_to_device(batch)
    images = batch["target"]
    condition = batch["condition"]
    if model.channels_last:
        condition = condition.to(memory_format=torch.channels_last)
        if images.ndim == 4:
            images = images.to(memory_format=torch.channels_last)
    if not torch.isnan(images).any():
        return model._post_process(images), Coords(model.batch_info, model.output_grid)
    labels = _build_labels(batch["labels"], "icon")
    mask = ~torch.isnan(images)
    times = torch.unique_consecutive(
        torch.tensor(
            edm_sampler_steps(
                sigma_max=model.sigma_max,
                num_steps=model.num_steps,
                sigma_min=model.sigma_min,
            ),
            device=images.device,
        )
    ).flip(0)
    dt = torch.diff(times).float()
    dt = dt.view(-1, 1, 1, 1, 1)  # (steps, 1, 1, 1, 1)
    increments = (
        torch.randn((len(dt), *images.shape), dtype=images.dtype, device=images.device)
        * dt
    )
    observed = images + torch.cumsum(increments, dim=0)
    scheduler = EDMNoiseScheduler(
        sigma_min=max(model.sigma_min, model.net.sigma_min),
        sigma_max=min(model.sigma_max, model.net.sigma_max),
    )
    sampler_times = model.net.round_sigma(
        scheduler.timesteps(model.num_steps, device=images.device, dtype=torch.float64)[
            :-1
        ]
    )
    # Bound subtraction/addition roundoff at the largest sampling level.
    sigma_tolerance = 4 * torch.finfo(sampler_times.dtype).eps * sampler_times[0]

    def predict_x0(
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Heun can reconstruct a grid endpoint with a few rounding bits changed.
        # Restore the native GPU level before the discrete observation lookup;
        # the CPU observation grid can itself differ from the GPU sampling grid.
        nearest_sigma = sampler_times[(sampler_times - t).abs().argmin()]
        t = torch.where((t - nearest_sigma).abs() <= sigma_tolerance, nearest_sigma, t)
        index = torch.searchsorted(times[1:], t).clamp(max=len(observed) - 1)
        return model.net(
            torch.where(mask, observed[index], x),
            t,
            labels,
            condition=condition,
            second_of_day=batch["second_of_day"],
            day_of_year=batch["day_of_year"],
        ).out

    denoiser = cast(_ScalarDenoiser, predict_x0)
    denoiser.round_sigma = model.net.round_sigma
    denoiser.sigma_min = model.net.sigma_min
    denoiser.sigma_max = model.net.sigma_max
    latents = torch.randn_like(images) * model.sigma_max
    with torch.no_grad(), torch.autocast("cuda", enabled=bf16, dtype=torch.bfloat16):
        output = sample_cbottle(
            denoiser,
            latents,
            solver=solver,
            options=options,
            sigma_min=model.sigma_min,
            sigma_max=model.sigma_max,
            num_steps=model.num_steps,
        )
    return model._post_process(output), Coords(model.batch_info, model.output_grid)


class SuperResolutionSampler(SuperResolutionModel):
    """Use PhysicsNeMo integration with C-Bottle's patch assembly."""

    solver: str | None = None
    solver_options: dict[str, Any] | None = None

    def _sample(
        self,
        denoiser: _ScalarDenoiser,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        return sample_cbottle(
            denoiser,
            latents,
            solver=self.solver,
            options=self.solver_options,
            num_steps=self.num_steps,
            sigma_max=self.sigma_max,
            scale_latents=True,
        )


class DistilledSuperResolutionSampler(DistilledSuperResolutionModel):
    """Keep native distilled sampling unless the caller selects a PhysicsNeMo solver."""

    solver: str | None = None
    solver_options: dict[str, Any] | None = None

    def _sample(
        self,
        denoiser: _ScalarDenoiser,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if self.solver is None:
            if self.solver_options:
                raise ValueError("Distilled solver_options require an explicit solver.")
            return super()._sample(denoiser, latents)
        return sample_cbottle(
            denoiser,
            latents,
            solver=self.solver,
            options=self.solver_options,
            sigma_max=self.sigma_max,
            scale_latents=True,
            distilled=True,
        )

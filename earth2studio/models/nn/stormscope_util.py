# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import Any

import torch
import torch.nn as nn

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from physicsnemo.experimental.models.dit.dit import DiT as PNM_DiT
except ImportError:
    OptionalDependencyFailure("stormscope")
    PNM_DiT = None  # type: ignore[assignment]

# Items copied from research repository; to be upstreamed to physicsnemo
# TODO: Remove once upstreamed


@check_optional_dependencies()
class EDMPrecond(torch.nn.Module):
    def __init__(
        self,
        label_dim=0,  # Number of class labels, 0 = unconditional.
        use_fp16=False,  # Execute the underlying model at FP16 precision?
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
        model=None,  # instance of the model to be used
        return_logvar=False,
        logvar_channels=128,
        output_channels=30,
        dropout: bool = False,
        sigma_max_dropout: float = 1000.0,
        sigma_min_dropout: float = 0.002,
        dropout_function_type: str = "sigmoid",
        p_max: float = 0.9,
        p_min: float = 0.1,
        x_offset: float = 15.0,
        slope: float = 6.0,
    ):
        super().__init__()
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model
        self.return_logvar = return_logvar
        if self.return_logvar:
            raise NotImplementedError(
                "logvar_fourier and logvar_linear are not implemented"
            )

        if dropout:
            self.noise_dependent_dropout = dropout
            self.sigma_max_dropout = torch.tensor(sigma_max_dropout)
            self.sigma_min_dropout = torch.tensor(sigma_min_dropout)
            self.dropout_function_type = dropout_function_type
            self.p_max = p_max
            self.p_min = p_min
            self.x_offset = x_offset
            self.slope = slope
            print(f"sigma_max_dropout: {self.sigma_max_dropout}")
            print(f"sigma_min_dropout: {self.sigma_min_dropout}")
            print(f"dropout_function_type: {self.dropout_function_type}")
            print(f"p_max: {self.p_max}")
            print(f"p_min: {self.p_min}")
            print(f"x_offset: {self.x_offset}")
            print(f"slope: {self.slope}")
        else:
            self.noise_dependent_dropout = False

    def forward(
        self,
        x,
        sigma,
        condition,
        class_labels=None,
        return_logvar=False,
        force_fp32=False,
        training=False,
        **model_kwargs,
    ):
        x = x.to(torch.float32)
        """
        p_dropout is the dropout probability for the model.
        class DropoutConfig:
        dropout: bool = False
        sigma_max_dropout: float = 200.0
        sigma_min_dropout: float = 0.002
        dropout_function_type: str = "sigmoid"
        p_max: float = 0.9
        p_min: float = 0.1
        x_offset: float = 15.0
        slope: float = 6.0
        """

        p_dropout = 0
        if self.noise_dependent_dropout:
            if self.dropout_function_type == "sigmoid":
                x_offset = torch.tensor(
                    self.x_offset
                )  # this is the point where the dropout probability is 0.5
                log_sigma = torch.log(sigma)
                log_offset = torch.log(x_offset)
                sigmoid = 1 / (
                    1 + torch.exp(-self.slope * (log_sigma - log_offset))
                )  # slope is the steepness of the sigmoid function for S curve
                p_dropout = (
                    self.p_min + (self.p_max - self.p_min) * sigmoid
                )  # p_min is the minimum dropout probability, p_max is the maximum dropout probability
            else:
                log_sigma_range = torch.log(self.sigma_max_dropout) - torch.log(
                    self.sigma_min_dropout
                )
                # Clamp sigma to [0, max_sigma] to ensure r is in [0, 1]
                r = (
                    torch.clamp(
                        torch.log(sigma) - torch.log(self.sigma_min_dropout),
                        max=log_sigma_range,
                    )
                    / log_sigma_range
                )
                p_dropout = (self.p_max - self.p_min) * r + self.p_min
        else:
            p_dropout = None

        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else (
                torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = c_in * x

        if condition is not None:
            arg = torch.cat([arg, condition], dim=1)
        # now we have added the p_dropout probability to the model.
        F_x = self.model(
            (arg).to(dtype),
            c_noise.flatten(),
            p_dropout=p_dropout,
            training=training,
            **model_kwargs,
        )

        D_x = c_skip * x + c_out * F_x.to(torch.float32)

        if return_logvar:
            logvar = (
                self.logvar_linear(self.logvar_fourier(c_noise.flatten()))
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            return D_x, logvar  # u(sigma) in Equation 21

        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class DropInDiT(nn.Module):
    """
    Wrapper that exposes the old DiT API while delegating to PhysicsNeMo DiT.

    Forward signature matches `models.diffusion_transformer.DiT.forward`:
        forward(x, time_step_cond=None, label_cond=None, points=None, p_dropout=None, training=False)
    """

    def __init__(
        self,
        pnm: PNM_DiT,
    ):
        super().__init__()
        self.pnm = pnm

        # Cache tokenizer geometry for NAT2D latent_hw
        self._input_size: tuple[int, int] = tuple(int(x) for x in pnm.input_size)
        self._patch_size: tuple[int, int] = tuple(int(x) for x in pnm.patch_size)

    @torch.no_grad()
    def _compute_latent_hw(self, x: torch.Tensor) -> tuple[int, int]:
        h, w = int(x.shape[-2]), int(x.shape[-1])
        ph, pw = self._patch_size
        return h // ph, w // pw

    def forward(
        self,
        x: torch.Tensor,
        time_step_cond: torch.Tensor | None = None,
        label_cond: torch.Tensor | None = None,
        points: torch.Tensor | None = None,
        p_dropout: float | torch.Tensor | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        # time_step_cond required by PNM; default to zeros if None
        if time_step_cond is None:
            time_step_cond = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        # PNM DiT uses a combined condition vector c = t (+ optional extra condition embedding)
        condition = None

        # Always provide NAT latent_hw
        latent_hw = self._compute_latent_hw(x)
        attn_kwargs: dict[str, Any] | None = {"latent_hw": latent_hw}

        # Note: points / cross-attention are not supported in PhysicsNeMo DiT so we ignore them
        out = self.pnm(
            x=x,
            t=time_step_cond,
            condition=condition,
            p_dropout=p_dropout,
            attn_kwargs=attn_kwargs,
        )
        return out

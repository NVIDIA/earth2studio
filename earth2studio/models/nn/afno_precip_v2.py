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

import physicsnemo
import torch
import torch.nn.functional as F
from physicsnemo.models.afno import AFNO


class PeriodicPad2d(torch.nn.Module):
    """
    pad longitudinal (left-right) circular
    and pad latitude (top-bottom) with zeros
    """

    def __init__(self, pad_width: int):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        # pad top and bottom zeros
        out = F.pad(
            out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0
        )
        return out


class PrecipNet_v2(physicsnemo.Module):
    def __init__(
        self,
        inp_shape: tuple,
        patch_size: tuple,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        depth: int,
        num_blocks: int,
        *args: tuple,
        **kwargs: dict,
    ):
        super().__init__()
        backbone = AFNO(
            inp_shape,
            in_channels,
            out_channels,
            patch_size,
            embed_dim,
            depth,
            num_blocks,
        )

        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

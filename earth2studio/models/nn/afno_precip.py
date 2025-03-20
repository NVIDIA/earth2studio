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
    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x):
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        out = F.pad(
            out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0
        )
        return out


class PrecipNet(physicsnemo.Module):
    def __init__(
        self,
        inp_shape,
        in_channels,
        out_channels,
        patch_size=(8, 8),
        embed_dim=768,
    ):
        super().__init__()
        self.backbone = AFNO(
            inp_shape=inp_shape,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=12,
            mlp_ratio=4.0,
            drop_rate=0.0,
            num_blocks=8,
        )
        self.ppad = PeriodicPad2d(1)
        self.conv = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True
        )
        self.act = torch.nn.ReLU()
        self.eps = 1e-5

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

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
from functools import partial

import physicsnemo  # noqa: F401 for docs
import torch
import torch.nn as nn
from physicsnemo.models.afno.afno import Block
from physicsnemo.models.module import Module

Tensor = torch.Tensor


class WrapConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        # self.kernel_size = kernel_size
        padding = (0, 0)
        super().__init__(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

    def pad(self, x, pad_size):
        left_padding = x[..., -pad_size[1][0] :]
        right_padding = x[..., : pad_size[1][1]]
        x = torch.concat([left_padding, x, right_padding], dim=-1)

        upper_padding = torch.flip(x[:, :, -pad_size[0][0] :, :], dims=[-2])
        lower_padding = torch.flip(x[:, :, : pad_size[0][1], :], dims=[-2])
        x = torch.concat([lower_padding, x, upper_padding], dim=-2)
        return x

    def forward(self, x, pad_size):
        if pad_size is not None:
            x = self.pad(x, pad_size)
        return super().forward(x)


class WrapConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_kernel_size: int,
        spatial_kernel_size: int,
        **kwargs,
    ):
        padding = (0, 0, 0)
        kernel_size = (time_kernel_size, spatial_kernel_size, spatial_kernel_size)
        super().__init__(
            in_channels, out_channels, kernel_size, padding=padding, **kwargs
        )

    def pad(self, x, pad_size):
        left_padding = x[..., -pad_size[1][0] :]
        right_padding = x[..., : pad_size[1][1]]
        x = torch.concat([left_padding, x, right_padding], dim=-1)

        upper_padding = torch.flip(x[:, :, :, -pad_size[0][0] :, :], dims=[-2])
        lower_padding = torch.flip(x[:, :, :, : pad_size[0][1], :], dims=[-2])
        x = torch.concat([lower_padding, x, upper_padding], dim=-2)
        return x

    def forward(self, x, pad_size):
        if pad_size is not None:
            x = self.pad(x, pad_size)
        return super().forward(x)


class PatchEmbed(nn.Module):
    """Patch embedding layer

    Converts 2D/3D patch into a 1D vector for input to AFNO

    Parameters
    ----------
    dim : int
        Input dimension [2D or 3D]
    in_channels : int
        Number of input channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        patch_size: list[int] = [16, 16],
        embed_dim: int = 256,
    ):
        super().__init__()
        if dim not in (2, 3):
            raise ValueError("dim should be 2 or 3")

        self.dim = dim
        self.patch_size = patch_size

        if self.dim == 2:
            self.proj = WrapConv2d(
                in_channels, embed_dim, kernel_size=patch_size[0], stride=patch_size
            )
        elif self.dim == 3:
            self.proj = nn.Sequential()
            self.proj = WrapConv3d(
                in_channels,
                embed_dim,
                time_kernel_size=patch_size[0],
                spatial_kernel_size=patch_size[1],
                stride=patch_size,
            )
            self.flat = nn.Flatten(start_dim=1, end_dim=2)

    def forward(self, x: Tensor, pad_size: list[list[int]]) -> Tensor:
        x = self.proj(x, pad_size)
        if self.dim == 3:
            x = self.flat(x)
        return x.flatten(2).transpose(1, 2)


class SolarRadiationNet(Module):
    """Adaptive Fourier neural operator (AFNO) model.

    Note
    ----
    AFNO is a model that is designed for 2D images only.

    Parameters
    ----------
    inp_shape : List[int]
        Input image dimensions [height, width]
    in_channels : int
        Number of input channels
    out_channels: int
        Number of output channels
    patch_size : List[int], optional
        Size of image patches, by default [16, 16]
    embed_dim : int, optional
        Embedded channel size, by default 256
    depth : int, optional
        Number of AFNO layers, by default 4
    mlp_ratio : float, optional
        Ratio of layer MLP latent variable size to input feature size, by default 4.0
    drop_rate : float, optional
        Drop out rate in layer MLPs, by default 0.0
    num_blocks : int, optional
        Number of blocks in the block-diag frequency weight matrices, by default 16
    sparsity_threshold : float, optional
        Sparsity threshold (softshrink) of spectral features, by default 0.01
    hard_thresholding_fraction : float, optional
        Threshold for limiting number of modes used [0,1], by default 1

    Example
    -------
    >>> model = modulus.models.afno.AFNO(
    ...     inp_shape=[32, 32],
    ...     in_channels=2,
    ...     out_channels=1,
    ...     patch_size=(8, 8),
    ...     embed_dim=16,
    ...     depth=2,
    ...     num_blocks=2,
    ... )
    >>> input = torch.randn(32, 2, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 1, 32, 32])

    Note
    ----
    Reference: Guibas, John, et al. "Adaptive fourier neural operators:
    Efficient token mixers for transformers." arXiv preprint arXiv:2111.13587 (2021).
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        patch_size: list[int] = [16, 16],
        pad_size: list[list[int]] = None,
        embed_dim: int = 256,
        depth: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.01,
        hard_thresholding_fraction: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        if self.dim not in (2, 3):
            raise ValueError("inp_shape should be a list of length 2 or 3")
        if self.dim not in (2, 3):
            raise ValueError("patch_size should be a list of length 2 or 3")

        self.in_chans = in_channels
        self.out_chans = out_channels
        self.dim = dim
        self.patch_size = patch_size
        self.pad_size = pad_size
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = num_blocks
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            dim=dim,
            in_channels=self.in_chans,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim=embed_dim,
                    num_blocks=self.num_blocks,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    norm_layer=norm_layer,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction,
                )
                for i in range(depth)
            ]
        )

        self.head = nn.Linear(
            embed_dim,
            self.out_chans * self.patch_size[-2] * self.patch_size[-1],
            bias=False,
        )

        # torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init model weights"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def compute_pad_size(self, x: torch.Tensor) -> list:
        H = x.shape[-2]
        W = x.shape[-1]
        if H % self.patch_size[-2] == 0:
            h_pad_size = [self.patch_size[-2] // 2, self.patch_size[-2] // 2]
        else:
            h_pad_size = [self.patch_size[-2] // 2, self.patch_size[-2] // 2 + 1]
        if W % self.patch_size[-1] == 0:
            w_pad_size = [self.patch_size[-1] // 2, self.patch_size[-1] // 2]
        else:
            w_pad_size = [self.patch_size[-1] // 2, self.patch_size[-1] // 2 + 1]
        return (h_pad_size, w_pad_size)

    def forward_features(self, x: Tensor) -> Tensor:
        """Forward pass of core AFNO"""
        B = x.shape[0]
        H = x.shape[-2]
        W = x.shape[-1]

        if self.pad_size == "auto":
            pad_size = self.compute_pad_size(x)
        else:
            pad_size = self.pad_size

        if pad_size is not None:
            h = (H + pad_size[0][0] + pad_size[0][1]) // self.patch_size[-2]
            w = (W + pad_size[1][0] + pad_size[1][1]) // self.patch_size[-1]
        else:
            h = H // self.patch_size[-2]
            w = W // self.patch_size[-1]
        x = self.patch_embed(x, pad_size)
        x = x.reshape(B, h, w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)
        return x, pad_size

    def forward(self, x: Tensor) -> Tensor:
        s = x.shape
        x, pad_size = self.forward_features(x)
        x = self.head(x)
        # Correct tensor shape back into [B, C, H, W]
        # [b h w (p1 p2 c_out)]
        x = x.view(list(x.shape[:-1]) + [self.patch_size[-2], self.patch_size[-1], -1])
        # [b h w p1 p2 c_out]
        x = torch.permute(x, (0, 5, 1, 3, 2, 4))
        # # [b c_out, h, p1, w, p2]
        if pad_size is not None:
            x = x.reshape(
                list(x.shape[:2])
                + [
                    s[-2] + pad_size[0][0] + pad_size[0][1],
                    s[-1] + pad_size[1][0] + pad_size[1][1],
                ]
            )
            x = x[
                :, :, pad_size[0][0] : -pad_size[0][1], pad_size[1][0] : -pad_size[1][1]
            ]
        else:
            x = x.reshape(list(x.shape[:2]) + [s[-2], s[-1]])
        # # [b c_out, (h*p1), (w*p2)]
        return x

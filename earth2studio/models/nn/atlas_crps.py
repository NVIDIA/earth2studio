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

from functools import partial

import torch
import torch.nn as nn

from earth2studio.models.nn.atlas import (
    DiTBlock,
    FinalLayer,
    FourierEmbedder,
    PatchPad,
    PatchUnpad,
    SInterpolantDownsampleProcessor,
    modulate,
    validate_patch_size,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import einops
    from timm.models.vision_transformer import Mlp, PatchEmbed
except ImportError:
    OptionalDependencyFailure("atlas")
    einops = None


class CRPSDownsampleProcessor(SInterpolantDownsampleProcessor):
    """Downsampling processor that also builds the low-resolution conditioning.

    The CRPS model consumes the cosine zenith angle and static invariant channels at
    low resolution, whereas :meth:`preprocess_input` appends them to the
    high-resolution state. Those channels are therefore downsampled from the
    high-resolution tensor rather than computed on the low-resolution grid, which is
    the convention the model was trained with.
    """

    def preprocess_conditioning(
        self, high_res: torch.Tensor, low_res: torch.Tensor
    ) -> torch.Tensor:
        """Append the downsampled auxiliary channels to the low-resolution state.

        Parameters
        ----------
        high_res : torch.Tensor
            High-resolution state with the auxiliary channels appended, as returned
            by :meth:`preprocess_input`.
        low_res : torch.Tensor
            Low-resolution normalized state, as returned by :meth:`preprocess_input`.

        Returns
        -------
        torch.Tensor
            Low-resolution conditioning of shape (batch, state + auxiliary channels,
            lat, lon).
        """
        auxiliary = high_res[:, self.normalizer_in.mean.shape[1] :]
        auxiliary = self.intep(auxiliary, self.downsample_grid_shape)
        return torch.cat([low_res, auxiliary], dim=1)


class EnsembleDiTBlock(DiTBlock):
    """DiT block with an additive noise term in the adaLN modulation.

    The noise term is what generates ensemble spread in a CRPS trained model:
    a single noise vector is broadcast over all tokens of a sample and shifts the
    per-block modulation.

    Parameters
    ----------
    hidden_dim : int
        Token embedding dimension.
    num_heads : int
        Number of attention heads.
    mlp_ratio : float, optional
        MLP hidden dimension multiplier, by default 4.0
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        **block_kwargs: bool,
    ) -> None:
        super().__init__(
            hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, **block_kwargs
        )
        self.noise_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        )

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, z: torch.Tensor
    ) -> torch.Tensor:
        """Apply the block to tokens `x` conditioned on `c` and noise `z`."""
        modulation = self.adaLN_modulation(c) + self.noise_modulation(z)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class Conv2dReflectYCircularX(nn.Module):
    """Convolution with circular longitude padding and reflected latitude padding.

    Parameters
    ----------
    channels : int
        Number of input and output channels.
    kernel_size : int, optional
        Square kernel size, by default 3
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.pad_x = nn.CircularPad2d((pad, pad, 0, 0))
        self.pad_y = nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve `x` of shape (batch, channels, lat, lon)."""
        return self.conv(self.pad_y(self.pad_x(x)))


@check_optional_dependencies()
class CRPSLatentDiT(torch.nn.Module):
    """CRPS trained latent diffusion transformer with two input streams.

    Predicts a normalized low-resolution residual from a history state (`x_1`) and a
    current state augmented with cosine zenith angle and static channels (`x_2`).
    Both streams are patch embedded with a shared patch grid, concatenated along the
    embedding dimension and processed by noise conditioned adaLN-Zero DiT blocks.

    Parameters
    ----------
    input_shape_1 : tuple[int, int], optional
        Spatial shape of the first input stream, by default (181, 360)
    input_shape_2 : tuple[int, int], optional
        Spatial shape of the second input stream, by default (181, 360)
    input_channels_1 : int, optional
        Channels of the first input stream, by default 75
    input_channels_2 : int, optional
        Channels of the second input stream, by default 79
    embed_dim_1 : int, optional
        Token embedding dimension of the first stream, by default 832
    embed_dim_2 : int, optional
        Token embedding dimension of the second stream, by default 2496
    num_patches : tuple[int, int], optional
        Number of patches in latitude and longitude, by default (91, 120)
    output_channels : int, optional
        Number of predicted channels, by default 75
    depth : int, optional
        Number of transformer blocks, by default 12
    num_heads : int, optional
        Number of attention heads, by default 13
    mlp_ratio : float, optional
        MLP hidden dimension multiplier, by default 4.0
    qk_norm : bool, optional
        Normalize queries and keys, by default False
    bfloat_cast : bool, optional
        Compute attention in bfloat16, by default True
    """

    def __init__(
        self,
        input_shape_1: tuple[int, int] = (181, 360),
        input_shape_2: tuple[int, int] = (181, 360),
        input_channels_1: int = 75,
        input_channels_2: int = 79,
        embed_dim_1: int = 832,
        embed_dim_2: int = 2496,
        num_patches: tuple[int, int] = (91, 120),
        output_channels: int = 75,
        depth: int = 12,
        num_heads: int = 13,
        mlp_ratio: float = 4.0,
        qk_norm: bool = False,
        bfloat_cast: bool = True,
    ) -> None:
        super().__init__()

        self.output_channels = output_channels

        patch_size_1, self.num_patches, latent_shape_1 = validate_patch_size(
            input_shape_1, n_patch=num_patches
        )
        patch_size_2, _, latent_shape_2 = validate_patch_size(
            input_shape_2, n_patch=num_patches
        )
        self.patch_size_1 = patch_size_1

        self.pad_1 = PatchPad(input_shape_1, latent_shape_1)
        self.pad_2 = PatchPad(input_shape_2, latent_shape_2)
        self.unpad = PatchUnpad(latent_shape_1, input_shape_1)

        self.x_embedder_1 = PatchEmbed(
            latent_shape_1, patch_size_1, input_channels_1, embed_dim_1, bias=True
        )
        self.x_embedder_2 = PatchEmbed(
            latent_shape_2, patch_size_2, input_channels_2, embed_dim_2, bias=True
        )
        self.pos_embed_1 = nn.Parameter(
            torch.zeros(1, self.x_embedder_1.num_patches, embed_dim_1),
            requires_grad=False,
        )
        self.pos_embed_2 = nn.Parameter(
            torch.zeros(1, self.x_embedder_2.num_patches, embed_dim_2),
            requires_grad=False,
        )

        embed_dim = embed_dim_1 + embed_dim_2
        self.t_embedder = FourierEmbedder(embed_dim)
        self.noise_mlp = Mlp(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            act_layer=partial(nn.GELU, approximate="tanh"),
            drop=0,
        )
        self.blocks = nn.ModuleList(
            [
                EnsembleDiTBlock(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                    bfloat_cast=bfloat_cast,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            embed_dim, patch_size_1[0] * patch_size_1[1] * output_channels
        )
        self.conv_head = Conv2dReflectYCircularX(output_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Fold tokens of shape (batch, patch, patch_dim) back onto the latent grid."""
        x = x.reshape(
            x.shape[0],
            self.num_patches[0],
            self.num_patches[1],
            self.patch_size_1[0],
            self.patch_size_1[1],
            self.output_channels,
        )
        return einops.rearrange(x, "b n1 n2 p1 p2 c -> b c (n1 p1) (n2 p2)")

    def forward(
        self,
        x_1: torch.Tensor,
        x_2: torch.Tensor,
        t: torch.Tensor | None = None,
        z_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict a normalized residual from the two input streams.

        Parameters
        ----------
        x_1 : torch.Tensor
            History state of shape (batch, input_channels_1, lat, lon).
        x_2 : torch.Tensor
            Augmented current state of shape (batch, input_channels_2, lat, lon).
        t : torch.Tensor, optional
            Conditioning time of shape (batch,). If None, ones are used, by default None
        z_noise : torch.Tensor, optional
            Ensemble noise of shape (batch, embed_dim). If None, standard normal noise
            is sampled, by default None

        Returns
        -------
        torch.Tensor
            Predicted residual of shape (batch, output_channels, lat, lon).
        """
        e_1 = self.x_embedder_1(self.pad_1(x_1)) + self.pos_embed_1
        e_2 = self.x_embedder_2(self.pad_2(x_2)) + self.pos_embed_2
        x = torch.cat((e_1, e_2), dim=-1)

        if t is None:
            t = torch.ones(x.shape[0], 1, device=x.device).view(-1)
        cond = self.t_embedder(t)

        if z_noise is None:
            z_noise = torch.randn_like(cond)
        z_noise = self.noise_mlp(z_noise)

        for block in self.blocks:
            x = block(x, cond, z_noise)

        x = self.final_layer(x, cond)
        x = self.unpatchify(x)
        x = self.conv_head(x)
        return self.unpad(x)

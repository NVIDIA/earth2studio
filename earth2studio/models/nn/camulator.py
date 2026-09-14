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

# Source: https://github.com/NCAR/miles-credit
# credit/models/camulator.py and credit/boundary_padding.py
# Apache-2.0, Copyright NSF NCAR Machine Integration and Learning for Earth
# Systems (MILES). Ported to plain PyTorch (no einops); module names and
# parameter layout are kept identical so CREDIT checkpoints load directly.

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def apply_spectral_norm(model: nn.Module) -> None:
    """Apply spectral normalization to every Conv2d/Linear/ConvTranspose2d layer,
    skipping the identity-initialised ``sharp`` convolutions, exactly as CREDIT does.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
            if "sharp" in name:
                continue
            nn.utils.spectral_norm(module)


class EarthPadding:
    """Spherical boundary padding: latitude is padded across the poles with the
    180-degree-rolled, flipped field; longitude is padded circularly.

    Parameters
    ----------
    pad_lat : tuple[int, int]
        Padding rows added at the south and north edges (first and last rows).
    pad_lon : tuple[int, int]
        Padding columns added at the west and east edges.
    """

    def __init__(self, pad_lat: tuple[int, int], pad_lon: tuple[int, int]):
        self.pad_lat = tuple(pad_lat)
        self.pad_lon = tuple(pad_lon)

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        if any(p > 0 for p in self.pad_lat):
            xroll = torch.roll(x, shifts=x.shape[-1] // 2, dims=-1)
            top = torch.flip(xroll[..., : self.pad_lat[0], :], (-2,))
            bot = torch.flip(xroll[..., -self.pad_lat[1] :, :], (-2,))
            x = torch.cat([top, x, bot], dim=-2)
        if any(p > 0 for p in self.pad_lon):
            x = F.pad(x, (self.pad_lon[0], self.pad_lon[1], 0, 0, 0, 0), mode="circular")
        return x

    def unpad(self, x: torch.Tensor) -> torch.Tensor:
        if any(p > 0 for p in self.pad_lat):
            end = -self.pad_lat[1] if self.pad_lat[1] > 0 else None
            x = x[..., self.pad_lat[0] : end, :]
        if any(p > 0 for p in self.pad_lon):
            end = -self.pad_lon[1] if self.pad_lon[1] > 0 else None
            x = x[..., :, self.pad_lon[0] : end]
        return x


class CubeEmbedding(nn.Module):
    """3D patch embedding. Constructed by CREDIT even when unused (patch size 1) so
    the parameters exist in the checkpoint; kept for state-dict compatibility."""

    def __init__(
        self,
        img_size: tuple[int, int, int],
        patch_size: tuple[int, int, int],
        in_chans: int,
        embed_dim: int,
    ):
        super().__init__()
        self.patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x)
        x = x.reshape(B, self.embed_dim, -1).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x.squeeze(2)


class UpBlockPS(nn.Module):
    """Pixel-shuffle upsampling block with an identity-initialised sharpening branch
    and a residual conv stack."""

    def __init__(
        self, in_ch: int, out_ch: int, num_groups: int, scale: int = 2, num_residuals: int = 2
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * scale**2, 3, stride=1, padding=1)
        self.ps = nn.PixelShuffle(scale)
        self.sharp = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        nn.init.xavier_normal_(self.sharp.weight)
        nn.init.zeros_(self.sharp.bias)
        blk: list[nn.Module] = []
        for _ in range(num_residuals):
            blk += [
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.GroupNorm(num_groups, out_ch),
                nn.SiLU(),
            ]
        self.b = nn.Sequential(*blk)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ps(self.conv(x))
        x = x + self.sharp(x)
        sc = x
        x = self.b(x)
        return x + sc


class CrossEmbedLayer(nn.Module):
    """Multi-kernel strided convolution embedding (CrossFormer cross-scale embed)."""

    def __init__(
        self, dim_in: int, dim_out: int, kernel_sizes: Sequence[int], stride: int = 2
    ):
        super().__init__()
        kernel_sizes = sorted(kernel_sizes)
        num_scales = len(kernel_sizes)
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]
        self.convs = nn.ModuleList([])
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv2d(
                    dim_in, dim_scale, kernel, stride=stride, padding=(kernel - stride) // 2
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([conv(x) for conv in self.convs], dim=1)


class DynamicPositionBias(nn.Module):
    """MLP producing a relative-position attention bias from 2D offsets."""

    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class LayerNorm(nn.Module):
    """Channel layer norm for (B, C, H, W) tensors (CREDIT/CrossFormer variant)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mult, dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Attention(nn.Module):
    """Windowed self-attention. ``short`` attends within local windows of size
    ``window_size``; ``long`` attends across a strided grid with stride
    ``window_size`` (CrossFormer short/long distance attention)."""

    def __init__(
        self,
        dim: int,
        attn_type: str,
        window_size: int,
        dim_head: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        if attn_type not in {"short", "long"}:
            raise ValueError("attention type must be one of short or long")
        heads = dim // dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.attn_type = attn_type
        self.window_size = window_size

        self.norm = LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)
        self.dpb = DynamicPositionBias(dim // 4)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))  # (2, w, w)
        grid = grid.permute(1, 2, 0).reshape(-1, 2)  # (w*w, 2)
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim=-1)
        self.register_buffer("rel_pos_indices", rel_pos_indices, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, d, height, width = x.shape
        wsz = self.window_size
        heads = self.heads

        x = self.norm(x)

        h, w = height // wsz, width // wsz
        if self.attn_type == "short":
            # b d (h s1) (w s2) -> (b h w) d s1 s2
            x = x.view(b, d, h, wsz, w, wsz).permute(0, 2, 4, 1, 3, 5)
        else:
            # b d (l1 h) (l2 w) -> (b h w) d l1 l2
            x = x.view(b, d, wsz, h, wsz, w).permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(b * h * w, d, wsz, wsz)

        q, k, v = self.to_qkv(x).chunk(3, dim=1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # b (h d) x y -> b h (x y) d
            return t.reshape(t.shape[0], heads, -1, wsz * wsz).transpose(-1, -2)

        q, k, v = map(split_heads, (q, k, v))
        q = q * self.scale

        sim = torch.einsum("bhid,bhjd->bhij", q, k)

        pos = torch.arange(-wsz, wsz + 1, device=x.device)
        rel_pos = torch.stack(torch.meshgrid(pos, pos, indexing="ij"))
        rel_pos = rel_pos.permute(1, 2, 0).reshape(-1, 2)
        biases = self.dpb(rel_pos.float())
        sim = sim + biases[self.rel_pos_indices]

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        # b h (x y) d -> b (h d) x y
        out = out.transpose(-1, -2).reshape(out.shape[0], -1, wsz, wsz)
        out = self.to_out(out)

        out = out.view(b, h, w, d, wsz, wsz)
        if self.attn_type == "short":
            # (b h w) d s1 s2 -> b d (h s1) (w s2)
            out = out.permute(0, 3, 1, 4, 2, 5).reshape(b, d, height, width)
        else:
            # (b h w) d l1 l2 -> b d (l1 h) (l2 w)
            out = out.permute(0, 3, 4, 1, 5, 2).reshape(b, d, height, width)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        local_window_size: int,
        global_window_size: int,
        depth: int = 4,
        dim_head: int = 32,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, "short", local_window_size, dim_head, attn_dropout),
                        FeedForward(dim, dropout=ff_dropout),
                        Attention(dim, "long", global_window_size, dim_head, attn_dropout),
                        FeedForward(dim, dropout=ff_dropout),
                    ]
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for short_attn, short_ff, long_attn, long_ff in self.layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x
        return x


class CamulatorNet(nn.Module):
    """CAMulator CrossFormer encoder-decoder (CREDIT ``Camulator`` without the
    post-block). Input and output tensors are channel-first with a singleton time
    dimension: ``(batch, channels, 1, lat, lon)``.

    Parameters
    ----------
    image_height : int
        Number of latitude grid points.
    image_width : int
        Number of longitude grid points.
    channels : int
        Number of 3D prognostic variables.
    surface_channels : int
        Number of 2D prognostic variables.
    input_only_channels : int
        Number of input-only (forcing/static) channels.
    output_only_channels : int
        Number of output-only (diagnostic) channels.
    levels : int
        Number of vertical levels per 3D variable.
    dim : tuple[int, ...]
        Hidden dimension of the four encoder stages.
    depth : tuple[int, ...]
        Number of transformer blocks per stage.
    dim_head : int
        Attention head dimension.
    global_window_size : tuple[int, ...]
        Long-distance attention stride per stage.
    local_window_size : int
        Short-distance attention window size.
    cross_embed_kernel_sizes : tuple[tuple[int, ...], ...]
        Kernel sizes of the cross-scale embedding per stage.
    cross_embed_strides : tuple[int, ...]
        Downsampling stride per stage.
    use_spectral_norm : bool
        Apply spectral normalization (required to load CREDIT checkpoints).
    interp : bool
        Bilinearly resize the decoder output to the input grid.
    pad_lat : tuple[int, int]
        Spherical padding rows (south, north).
    pad_lon : tuple[int, int]
        Spherical padding columns (west, east).
    """

    def __init__(
        self,
        image_height: int = 192,
        image_width: int = 288,
        channels: int = 4,
        surface_channels: int = 2,
        input_only_channels: int = 6,
        output_only_channels: int = 17,
        levels: int = 32,
        dim: tuple[int, ...] = (256, 512, 1024, 2048),
        depth: tuple[int, ...] = (2, 2, 18, 2),
        dim_head: int = 32,
        global_window_size: tuple[int, ...] = (4, 4, 2, 1),
        local_window_size: int = 3,
        cross_embed_kernel_sizes: tuple[tuple[int, ...], ...] = (
            (4, 8, 16, 32),
            (2, 4),
            (2, 4),
            (2, 4),
        ),
        cross_embed_strides: tuple[int, ...] = (2, 2, 2, 2),
        use_spectral_norm: bool = True,
        interp: bool = True,
        pad_lat: tuple[int, int] = (48, 48),
        pad_lon: tuple[int, int] = (48, 48),
    ):
        super().__init__()
        if not (len(dim) == len(depth) == len(global_window_size) == 4):
            raise ValueError("dim, depth and global_window_size must have 4 stages")
        if not (len(cross_embed_kernel_sizes) == len(cross_embed_strides) == 4):
            raise ValueError("cross_embed_kernel_sizes and strides must have 4 stages")

        self.image_height = image_height
        self.image_width = image_width
        self.use_interp = interp
        self.padding = EarthPadding(pad_lat, pad_lon)
        self.use_padding = any(p > 0 for p in (*pad_lat, *pad_lon))

        self.input_channels = channels * levels + surface_channels + input_only_channels
        self.output_channels = channels * levels + surface_channels + output_only_channels

        dims = [self.input_channels, *dim]
        self.layers = nn.ModuleList([])
        for (dim_in, dim_out), num_layers, global_wsize, kernel_sizes, stride in zip(
            zip(dims[:-1], dims[1:]),
            depth,
            global_window_size,
            cross_embed_kernel_sizes,
            cross_embed_strides,
        ):
            cel = CrossEmbedLayer(dim_in, dim_out, kernel_sizes, stride)
            transformer = Transformer(
                dim_out,
                local_window_size=local_window_size,
                global_window_size=global_wsize,
                depth=num_layers,
                dim_head=dim_head,
            )
            self.layers.append(nn.ModuleList([cel, transformer]))

        # Present in CREDIT checkpoints even though patch size 1 never uses it.
        self.cube_embedding = CubeEmbedding(
            (1, image_height, image_width), (1, 1, 1), self.input_channels, dim[0]
        )

        last_dim = dim[-1]
        self.up_block1 = UpBlockPS(last_dim, last_dim // 2, dim[0])
        self.up_block2 = UpBlockPS(2 * (last_dim // 2), last_dim // 4, dim[0])
        self.up_block3 = UpBlockPS(2 * (last_dim // 4), last_dim // 8, dim[0])
        scale = 2
        self.up_block4 = nn.Sequential(
            nn.Conv2d(2 * (last_dim // 8), self.output_channels * scale**2, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(self.output_channels, self.output_channels, 3, padding=1),
        )

        if use_spectral_norm:
            apply_spectral_norm(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_padding:
            x = self.padding.pad(x)
        x = x.squeeze(2)

        encodings = []
        for cel, transformer in self.layers:
            x = cel(x)
            x = transformer(x)
            encodings.append(x)

        x = self.up_block1(x)
        x = torch.cat([x, encodings[2]], dim=1)
        x = self.up_block2(x)
        x = torch.cat([x, encodings[1]], dim=1)
        x = self.up_block3(x)
        x = torch.cat([x, encodings[0]], dim=1)
        x = self.up_block4(x)

        if self.use_padding:
            x = self.padding.unpad(x)
        if self.use_interp:
            x = F.interpolate(
                x, size=(self.image_height, self.image_width), mode="bilinear"
            )
        return x.unsqueeze(2)

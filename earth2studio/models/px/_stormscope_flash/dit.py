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

"""Inference-only DiT compatible with the bundled StormScope Flash checkpoints."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from earth2studio.models.px._stormscope_flash.rope import RoPE2D
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from natten.functional import na2d  # type: ignore[import-not-found, import-untyped]
except (ImportError, OSError):
    OptionalDependencyFailure("stormscope-flash")
    na2d = None


def _modulate(
    x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class MLP(nn.Module):
    """Checkpoint-compatible transformer MLP."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        act_layer: Callable[[], nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.fwd = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=True),
            act_layer(),
            nn.Identity(),
            nn.Linear(hidden_features, out_features, bias=True),
            nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the checkpoint's two-layer feed-forward projection."""
        return self.fwd(x)


class Attention(nn.Module):
    """Full or two-dimensional neighborhood self-attention."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        qkv_bias: bool,
        qk_norm: bool,
        attn_kernel: int,
        rope: RoPE2D | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.rope = rope
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_drop_rate = 0.0
        self.attn_mask = None
        self.attn_kernel = attn_kernel
        self.num_register_tokens = 0
        self.use_te = False
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Identity()

    @check_optional_dependencies()
    def forward(
        self,
        x: torch.Tensor,
        per_batch_attn_mask: torch.Tensor | None = None,
        latent_hw: tuple[int, int] | None = None,
        invalid_token_mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply normalized rotary attention with optional learned missing tokens."""
        attn_mask = (
            per_batch_attn_mask if per_batch_attn_mask is not None else self.attn_mask
        )
        batch, tokens, channels = x.shape
        if invalid_token_mask is not None and mask_token is not None:
            x = torch.where(invalid_token_mask[..., None], mask_token.to(x), x)
        qkv = (
            self.qkv(x)
            .reshape(batch, tokens, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        input_dtype = q.dtype
        q = self.q_norm(q).to(input_dtype)
        k = self.k_norm(k).to(input_dtype)
        if self.rope is not None:
            if latent_hw is None:
                raise ValueError("rotary attention requires latent_hw")
            q, k = self.rope(q, k, *latent_hw)

        if self.attn_kernel == -1:
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                attended = F.scaled_dot_product_attention(
                    q.to(torch.float32),
                    k.to(torch.float32),
                    v.to(torch.float32),
                    dropout_p=self.attn_drop_rate,
                    attn_mask=attn_mask,
                    scale=self.scale,
                )
            attended = attended.transpose(1, 2).reshape(batch, tokens, channels)
            attended = attended.to(input_dtype)
        elif self.attn_kernel > 0:
            if latent_hw is None:
                raise ValueError("neighborhood attention requires latent_hw")
            if na2d is None:
                raise ImportError(
                    "StormScope Flash requires natten; "
                    "install earth2studio[stormscope]"
                )
            height, width = latent_hw
            if tokens != height * width:
                raise ValueError("spatial token count does not match latent_hw")
            q = q.permute(0, 2, 1, 3).reshape(
                batch, height, width, self.num_heads, self.head_dim
            )
            k = k.permute(0, 2, 1, 3).reshape(
                batch, height, width, self.num_heads, self.head_dim
            )
            v = v.permute(0, 2, 1, 3).reshape(
                batch, height, width, self.num_heads, self.head_dim
            )
            attended = na2d(q, k, v, kernel_size=self.attn_kernel)
            attended = attended.reshape(batch, tokens, channels)
        else:
            raise ValueError("attn_kernel must be -1 or a positive integer")

        return self.proj_drop(self.proj(attended))


class DiTBlock(nn.Module):
    """Checkpoint-compatible adaptive-LayerNorm DiT block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_norm: bool,
        attn_kernel: int,
        rope: RoPE2D | None = None,
    ) -> None:
        super().__init__()
        self.attn_mask = None
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_kernel=attn_kernel,
            rope=rope,
        )
        self.num_ada_ln = 6
        self.drop_path = nn.Identity()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            dim,
            hidden_dim,
            dim,
            act_layer=partial(nn.GELU, approximate="tanh"),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, self.num_ada_ln * dim, bias=True)
        )
        self.adaLN_modulation_cross_scale_shift = None
        self.adaLN_modulation_cross_gate = None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        kv: torch.Tensor | None = None,
        latent_hw: tuple[int, int] | None = None,
        p_dropout: float | torch.Tensor | None = None,
        training: bool = False,
        invalid_token_mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply adaptive attention and MLP residual updates for inference."""
        if kv is not None:
            raise ValueError("the StormScope Flash DiT does not use cross attention")
        if training or p_dropout not in (None, 0, 0.0):
            raise ValueError("the packaged Flash DiT supports inference only")

        shift, scale, gate, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            cond
        ).chunk(self.num_ada_ln, dim=1)
        attended = self.attn(
            _modulate(self.norm1(x), scale, shift),
            latent_hw=latent_hw,
            invalid_token_mask=invalid_token_mask,
            mask_token=mask_token,
        )
        gate = gate.unsqueeze(1)
        gate_mlp = gate_mlp.unsqueeze(1)
        if self.attn.rope is None:
            gate, gate_mlp = gate.float(), gate_mlp.float()
        x = x + self.drop_path(gate * attended)
        mlp = self.mlp(_modulate(self.norm2(x), scale_mlp, shift_mlp))
        return x + self.drop_path(gate_mlp * mlp)


class DiTLastLayer(nn.Module):
    """Adaptive normalization and final patch projection."""

    def __init__(
        self, hidden_size: int, patch_size: tuple[int, int], out_chans: int
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_chans)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward_features(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply the final adaptive normalization before interval-head projection."""
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        return _modulate(self.norm(x), scale, shift)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Project patches in FP32, then restore the incoming feature dtype."""
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.linear(x.to(torch.float32))
        return x.to(input_dtype)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Normalize and project the final patch features."""
        return self.project(self.forward_features(x, cond))


class TimeStepEmbed(nn.Module):
    """Sinusoidal EDM noise embedding used by the trained DiT."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embed_dim: int = 256,
        max_period: float = 10000.0,
    ) -> None:
        super().__init__()
        self.frequency_embed_dim = frequency_embed_dim
        self.hidden_size = hidden_size
        self.max_period = max_period
        half = frequency_embed_dim // 2
        powers = torch.arange(half, dtype=torch.float32) / half
        self.freqs = nn.Parameter(
            torch.exp(-math.log(max_period) * powers), requires_grad=False
        )
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """Embed each noise coordinate using the stored sinusoidal frequencies."""
        arguments = torch.outer(time, self.freqs)
        embedding = torch.cat((torch.sin(arguments), torch.cos(arguments)), dim=1)
        return self.mlp(embedding)


class PatchEmbed(nn.Module):
    """Convolutional non-overlapping patch projection."""

    def __init__(
        self,
        height: int,
        width: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        if height % patch_size or width % patch_size:
            raise ValueError("image dimensions must be divisible by patch_size")
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.num_patches = (height // patch_size) * (width // patch_size)
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project image patches in FP32 and restore the input dtype."""
        input_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = self.proj(x.to(torch.float32))
        return x.to(input_dtype)


class DiT(nn.Module):
    """Lean inference implementation matching the validated StormScope Flash DiT."""

    def __init__(
        self,
        *,
        height: int,
        width: int,
        patch_size: int,
        in_chans: int,
        base_out_chans: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        frequency_embed_dim: int = 256,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        learn_sigma: bool = False,
        num_classes: int = 0,
        class_dropout_prob: float = 0.1,
        pos_embedding_type: str = "rotary",
        is_conditional: bool = True,
        use_skip_connection: bool = False,
        use_concat_skip_connection: bool = False,
        num_input_time_steps: int = 1,
        num_output_time_steps: int = 1,
        attn_mask_type: str | None = None,
        attn_kernel: int = 49,
        point_channels: int = 0,
        use_fused_layernorm: bool = False,
        p_dropout: float | None = None,
        grad_checkpoint_blocks: int = 0,
        alternate_attn: bool = False,
        use_transformer_engine: bool = False,
        num_register_tokens: int = 0,
        rope_theta: float = 10000.0,
        use_nan_mask_tokens: bool = False,
    ) -> None:
        super().__init__()
        unsupported = {
            "learn_sigma": learn_sigma,
            "num_classes": num_classes,
            "use_skip_connection": use_skip_connection,
            "use_concat_skip_connection": use_concat_skip_connection,
            "attn_mask_type": attn_mask_type,
            "point_channels": point_channels,
            "use_fused_layernorm": use_fused_layernorm,
            "p_dropout": p_dropout,
            "grad_checkpoint_blocks": grad_checkpoint_blocks,
            "use_transformer_engine": use_transformer_engine,
            "num_register_tokens": num_register_tokens,
        }
        enabled = [
            name for name, value in unsupported.items() if value not in (0, None, False)
        ]
        if enabled:
            raise ValueError(
                "unsupported StormScope Flash DiT options: " + ", ".join(enabled)
            )
        if not is_conditional or pos_embedding_type != "rotary":
            raise ValueError("StormScope Flash requires a conditional rotary DiT")

        self._is_conditional = is_conditional
        self._learn_sigma = False
        self._out_chans = base_out_chans
        self._in_chans = in_chans
        self._patch_size = patch_size
        self._height = height
        self._width = width
        self._embed_dim = embed_dim
        self._depth = depth
        self._num_heads = num_heads
        self._mlp_ratio = mlp_ratio
        self._qkv_bias = qkv_bias
        self._qk_norm = qk_norm
        self._num_classes = num_classes
        self._class_dropout_prob = class_dropout_prob
        self._frequency_embed_dim = frequency_embed_dim
        self._pos_embedding_type = pos_embedding_type
        self._use_fused_layernorm = False
        self._grad_checkpoint_blocks = 0
        self._alternate_attn = alternate_attn
        self._use_transformer_engine = False
        self._num_register_tokens = 0
        self._register_tokens = None
        self._use_skip_connection = False
        self._use_concat_skip_connection = False
        self._num_input_time_steps = num_input_time_steps
        self._num_output_time_steps = num_output_time_steps
        self._embed_points = None

        self._patch_emb = PatchEmbed(height, width, patch_size, in_chans, embed_dim)
        self._pos_emb = None
        self._rope = RoPE2D(embed_dim // num_heads, rope_theta)
        self._use_nan_mask_tokens = use_nan_mask_tokens
        self.register_buffer("invalid_token_mask_flat", None, persistent=False)
        self._label_emb = 0.0
        self._time_step_emb = TimeStepEmbed(embed_dim, frequency_embed_dim)
        self.attn_mask = None
        middle = depth // 2
        self._blocks = nn.ModuleList(
            [
                DiTBlock(
                    embed_dim,
                    num_heads,
                    rope=self._rope,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    attn_kernel=(
                        -1
                        if alternate_attn and index in (0, middle, depth - 1)
                        else attn_kernel
                    ),
                )
                for index in range(depth)
            ]
        )
        self._nan_mask_tokens = nn.ParameterDict(
            {
                str(i): nn.Parameter(torch.zeros(1, 1, embed_dim))
                for i, block in enumerate(self._blocks)
                if use_nan_mask_tokens and cast(DiTBlock, block).attn.attn_kernel > 0
            }
        )
        if use_nan_mask_tokens and not self._nan_mask_tokens:
            raise ValueError("NaN mask tokens require neighborhood attention")
        self._final_layer = DiTLastLayer(
            embed_dim, (patch_size, patch_size), base_out_chans
        )

    invalid_token_mask_flat: torch.Tensor | None

    def set_image_size(self, height: int, width: int) -> None:
        """Set this instance's rotary grid before forecasting or compiling."""
        if min(height, width) < 200 or height % 4 or width % 4:
            raise ValueError(
                "Flash image dimensions must be >=200 and divisible by four"
            )
        self._height, self._width = height, width
        self._patch_emb.height, self._patch_emb.width = height, width
        self._patch_emb.num_patches = (height // self._patch_size) * (
            width // self._patch_size
        )
        self.invalid_token_mask_flat = None

    def set_nan_pixel_mask(self, pixel_mask: torch.Tensor) -> None:
        """Set missing conditioning pixels, marking every touched patch invalid.

        Parameters
        ----------
        pixel_mask : torch.Tensor
            Boolean mask with shape [height, width] or [batch, height, width].
        """
        if not self._use_nan_mask_tokens:
            raise RuntimeError("this DiT does not use NaN mask tokens")
        if pixel_mask.ndim not in (2, 3) or pixel_mask.shape[-2:] != (
            self._height,
            self._width,
        ):
            raise ValueError("pixel mask must match the model image dimensions")
        if pixel_mask.ndim == 2:
            pixel_mask = pixel_mask[None]
        self.invalid_token_mask_flat = (
            F.max_pool2d(
                pixel_mask.to(
                    device=self._patch_emb.proj.weight.device, dtype=torch.float32
                )[:, None],
                self._patch_size,
                self._patch_size,
            )[:, 0]
            .flatten(1)
            .bool()
        )

    def prepare_patch_tokens(
        self, patch: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Flatten a projected image and add the learned positional embedding."""

        height, width = patch.shape[-2:]
        tokens = patch.flatten(2).transpose(1, 2)
        if self._pos_emb is not None:
            if tokens.shape[1] != self._pos_emb.shape[1]:
                raise ValueError(
                    "input image does not match the trained positional grid"
                )
            tokens = tokens + self._pos_emb
        return tokens, (height, width)

    def prepare_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Embed an image and return flattened tokens with their patch-grid shape."""
        return self.prepare_patch_tokens(self._patch_emb(x))

    def _split_patch_projection(
        self,
        x: torch.Tensor,
        *,
        channel_start: int,
        channel_stop: int,
        include_bias: bool,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        projection = self._patch_emb.proj
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            projected = F.conv2d(
                x.to(torch.float32),
                projection.weight[:, channel_start:channel_stop].to(torch.float32),
                (
                    projection.bias.to(torch.float32)
                    if include_bias and projection.bias is not None
                    else None
                ),
                stride=projection.stride,
                padding=projection.padding,
                dilation=projection.dilation,
                groups=projection.groups,
            )
        return projected.to(output_dtype)

    def prepare_condition_patch(
        self,
        condition: torch.Tensor,
        *,
        state_channels: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Project static condition channels once for a Flash trajectory."""

        if condition.ndim != 4:
            raise ValueError(
                "condition must have shape [batch, channel, height, width]"
            )
        expected = self._in_chans - state_channels
        if (
            state_channels <= 0
            or state_channels >= self._in_chans
            or condition.shape[1] != expected
        ):
            raise ValueError(
                f"condition has {condition.shape[1]} channels, expected {expected}"
            )
        return self._split_patch_projection(
            condition,
            channel_start=state_channels,
            channel_stop=self._in_chans,
            include_bias=False,
            output_dtype=output_dtype,
        )

    def forward_features_from_tokens(
        self,
        tokens: torch.Tensor,
        latent_hw: tuple[int, int],
        time_step_cond: torch.Tensor | None,
        *,
        training: bool = False,
    ) -> torch.Tensor:
        """Run the shared transformer on prepared tokens at the given noise level."""
        if training:
            raise ValueError("the packaged Flash DiT supports inference only")
        if time_step_cond is None:
            time_step_cond = torch.zeros(
                tokens.shape[0], device=tokens.device, dtype=tokens.dtype
            )
        condition = self._time_step_emb(time_step_cond) + self._label_emb
        for index, block in enumerate(self._blocks):
            tokens = block(
                tokens,
                condition,
                latent_hw=latent_hw,
                p_dropout=None,
                training=False,
                invalid_token_mask=self.invalid_token_mask_flat,
                mask_token=(
                    self._nan_mask_tokens[str(index)]
                    if str(index) in self._nan_mask_tokens
                    else None
                ),
            )
        return self._final_layer.forward_features(tokens, condition)

    def forward_features(
        self,
        x: torch.Tensor,
        time_step_cond: torch.Tensor | None = None,
        *,
        label_cond: torch.Tensor | None = None,
        points: Any | None = None,
        p_dropout: float | torch.Tensor | None = None,
        training: bool = False,
    ) -> torch.Tensor:
        """Return shared features for the parallel interval heads."""
        if (
            label_cond is not None
            or points is not None
            or p_dropout not in (None, 0, 0.0)
        ):
            raise ValueError(
                "the StormScope Flash DiT does not use labels, points, or dropout"
            )
        tokens, latent_hw = self.prepare_tokens(x)
        return self.forward_features_from_tokens(
            tokens, latent_hw, time_step_cond, training=training
        )

    def forward_features_with_condition_patch(
        self,
        state_input: torch.Tensor,
        condition_patch: torch.Tensor,
        time_step_cond: torch.Tensor | None = None,
        *,
        training: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        """Run trunk features with a cached condition patch contribution."""

        state_patch = self._split_patch_projection(
            state_input,
            channel_start=0,
            channel_stop=state_input.shape[1],
            include_bias=True,
            output_dtype=state_input.dtype,
        )
        if state_patch.shape != condition_patch.shape:
            raise ValueError("state and condition patch projections must match")
        tokens, latent_hw = self.prepare_patch_tokens(state_patch + condition_patch)
        return self.forward_features_from_tokens(
            tokens, latent_hw, time_step_cond, training=training
        )

    def unpatchify(
        self, tokens: torch.Tensor, height: int | None = None, width: int | None = None
    ) -> torch.Tensor:
        """Restore patch tokens to a channel-first image."""

        height = height or self._height
        width = width or self._width
        patch_height = height // self._patch_size
        patch_width = width // self._patch_size
        batch = tokens.shape[0]
        tokens = tokens.reshape(
            batch,
            patch_height,
            patch_width,
            self._patch_size,
            self._patch_size,
            self._out_chans,
        )
        return tokens.permute(0, 5, 1, 3, 2, 4).reshape(
            batch, self._out_chans, height, width
        )

    def forward(
        self, x: torch.Tensor, time_step_cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return the checkpoint-compatible image prediction before preconditioning."""
        height, width = x.shape[-2:]
        features = self.forward_features(x, time_step_cond)
        return self.unpatchify(self._final_layer.project(features), height, width)


__all__ = ["DiT"]

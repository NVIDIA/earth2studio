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

import torch
from torch import Tensor, nn


def rope(pos: Tensor, dim: int, theta: float) -> Tensor:
    """Build real-valued rotation matrices for one positional axis.

    Parameters
    ----------
    pos : Tensor
        Floating position IDs with shape (..., tokens).
    dim : int
        Positive, even rotary dimension.
    theta : float
        Positive frequency base.

    Returns
    -------
    Tensor
        Float32 rotations with shape (..., tokens, dim / 2, 2, 2).

    Notes
    -----
    Uses the FLUX real-valued rotation convention:
    https://github.com/black-forest-labs/flux/blob/main/src/flux/math.py
    """
    if dim < 2 or dim % 2 or theta <= 0:
        raise ValueError("RoPE requires an even positive dimension and positive theta")
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    angles = torch.einsum("...n,d->...nd", pos, omega)
    rotations = torch.stack(
        [torch.cos(angles), -torch.sin(angles), torch.sin(angles), torch.cos(angles)],
        dim=-1,
    )
    return rotations.reshape(*rotations.shape[:-1], 2, 2).float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate [batch, heads, tokens, head_dim] queries and keys in float32.

    Parameters
    ----------
    xq, xk : Tensor
        Query and key tensors with the same shape.
    freqs_cis : Tensor
        Rotation matrices broadcastable over batch and heads.

    Returns
    -------
    tuple[Tensor, Tensor]
        Rotated queries and keys, preserving their original dtypes.
    """
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    q = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    k = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return q.reshape_as(xq).type_as(xq), k.reshape_as(xk).type_as(xk)


class RoPE2D(nn.Module):
    """Parameter-free axial row/column rotations for a rectangular latent grid.

    Parameters
    ----------
    head_dim : int
        Attention head dimension, divisible by four.
    theta : float, optional
        Frequency base, by default 10000.0
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim < 4 or head_dim % 4 or theta <= 0:
            raise ValueError("head_dim must be divisible by four and theta positive")
        self.head_dim = head_dim
        self.axis_dim = head_dim // 2
        self.theta = theta

    def build_pe(self, h: int, w: int, h_offset: int, device: torch.device) -> Tensor:
        """Build rotations using global row IDs and zero-based column IDs.

        Parameters
        ----------
        h, w : int
            Local latent grid dimensions.
        h_offset : int
            Global row index of the first local row.
        device : torch.device
            Device for the generated tensor.

        Returns
        -------
        Tensor
            Rotations of shape [1, 1, h*w, head_dim/2, 2, 2].
        """
        rows = torch.arange(h_offset, h_offset + h, device=device, dtype=torch.float32)
        cols = torch.arange(w, device=device, dtype=torch.float32)
        row_pe = rope(rows[:, None].expand(h, w).reshape(-1), self.axis_dim, self.theta)
        col_pe = rope(cols[None, :].expand(h, w).reshape(-1), self.axis_dim, self.theta)
        return torch.cat([row_pe, col_pe], dim=-3).unsqueeze(0).unsqueeze(0)

    def forward(
        self, q: Tensor, k: Tensor, h: int, w: int, h_offset: int = 0
    ) -> tuple[Tensor, Tensor]:
        """Apply axial rotations to queries and keys on the given latent grid."""
        return apply_rope(q, k, self.build_pe(h, w, h_offset, q.device))

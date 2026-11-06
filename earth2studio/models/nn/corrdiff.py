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

"""Shared helpers for the CorrDiff downscaling wrappers: curvilinear-footprint
selection and binding of a crop-size agnostic RoPE / NATTEN PhysicsNeMo ``DiT`` to a
regional view (:class:`~earth2studio.models.base.DomainAdaptable`)."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

try:
    from physicsnemo.nn.module.rope import build_axial_rope_cos_sin_2d
except ImportError:  # pragma: no cover - optional dependency
    build_axial_rope_cos_sin_2d = None


def points_in_grid_footprint(
    plat: np.ndarray, plon: np.ndarray, lat2d: np.ndarray, lon2d: np.ndarray
) -> np.ndarray:
    """Whether geographic points fall inside a (curvilinear) grid footprint.

    Projected grids (rotated pole, Lambert conformal, ...) have curved edges in
    latitude / longitude, so their rectangular bounds include points outside the
    actual grid. This checks whether each point lies inside the polygon formed by
    the boundary-cell centers; points within about half a cell of the outer edge
    may therefore be excluded. ``plon`` and ``lon2d`` must share one longitude
    convention and the footprint must not cross its wrap meridian.

    Parameters
    ----------
    plat, plon : np.ndarray
        Query point latitudes / longitudes (any shape, broadcast together).
    lat2d, lon2d : np.ndarray
        The target grid's 2D [lat, lon] coordinate arrays.

    Returns
    -------
    np.ndarray
        Boolean mask (shape of ``plat``), True where the point is inside.
    """
    ring_lat = np.concatenate(
        [lat2d[0, :], lat2d[:, -1], lat2d[-1, ::-1], lat2d[::-1, 0]]
    ).astype(np.float64)
    ring_lon = np.concatenate(
        [lon2d[0, :], lon2d[:, -1], lon2d[-1, ::-1], lon2d[::-1, 0]]
    ).astype(np.float64)
    plat = np.atleast_1d(plat).astype(np.float64)
    plon = np.atleast_1d(plon).astype(np.float64)
    inside = np.zeros(plat.shape, dtype=bool)
    n = ring_lat.size
    j = n - 1
    # Test whether each point is inside the boundary polygon by counting where a
    # horizontal ray from it crosses the boundary: an odd number means inside, an
    # even number means outside. Toggling `inside` records this odd/even result. The
    # small offset avoids division by zero for horizontal edges.
    for i in range(n):
        yi, yj = ring_lat[i], ring_lat[j]
        xi, xj = ring_lon[i], ring_lon[j]
        straddle = (yi > plat) != (yj > plat)
        x_int = (xj - xi) * (plat - yi) / (yj - yi + 1e-30) + xi
        inside ^= straddle & (plon < x_int)
        j = i
    return inside


def snap_to_patch(lo: int, hi: int, n: int, patch: int) -> tuple[int, int]:
    """Snap a run-window extent ``[lo, hi)`` along one axis to a multiple of ``patch``.

    A DiT's patch detokenizer reconstructs ``(extent // patch) * patch`` rows, so an
    extent that is not a multiple of the patch size would come back short. The
    window grows OUTWARD into the surrounding grid (real neighbouring cells),
    bounded by ``[0, n)``: ``hi`` first, then ``lo``. It only shrinks when ``n``
    itself is not a multiple of ``patch`` and there is no room left to grow, which
    callers detect as a negative trim. No-op for ``patch == 1``.

    Parameters
    ----------
    lo, hi : int
        Half-open window along the axis.
    n : int
        Length of the axis.
    patch : int
        Patch size along the axis.

    Returns
    -------
    tuple[int, int]
        The snapped ``(lo, hi)``.
    """
    need = (-(hi - lo)) % patch
    if need:
        add = min(need, n - hi)
        hi += add
        need -= add
        add = min(need, lo)
        lo -= add
        need -= add
        if need:  # axis not a multiple of patch and no room -> shrink
            hi -= (hi - lo) % patch
    return lo, hi


def find_dit(
    network: torch.nn.Module | None, max_depth: int = 4
) -> torch.nn.Module | None:
    """Locate the DiT inside a (possibly wrapped) network.

    Walks ``model`` / ``dit`` / ``module`` attributes through preconditioners and
    condition wrappers until it finds a module with a ``tokenizer``, a
    ``detokenizer`` and ``blocks`` (the PhysicsNeMo ``DiT`` interface).

    Parameters
    ----------
    network : torch.nn.Module | None
        Network or wrapper around one (``None`` yields ``None``).
    max_depth : int, optional
        How many wrapper levels to descend, by default 4

    Returns
    -------
    torch.nn.Module | None
        The DiT, or None when the network holds no DiT (e.g. a test double).
    """
    node: torch.nn.Module | None = network
    for _ in range(max_depth + 1):
        if node is None:
            return None
        if all(hasattr(node, a) for a in ("tokenizer", "detokenizer", "blocks")):
            return node
        node = next(
            (
                getattr(node, a)
                for a in ("model", "dit", "module")
                if isinstance(getattr(node, a, None), torch.nn.Module)
            ),
            None,
        )
    return None


def dit_grid_constraints(
    dit: torch.nn.Module | None,
) -> tuple[tuple[int, int], int]:
    """Patch size and minimum run-domain extent of a DiT.

    Parameters
    ----------
    dit : torch.nn.Module | None
        The DiT (``None`` for networks without one).

    Returns
    -------
    tuple[tuple[int, int], int]
        ``((patch_h, patch_w), min_cells)``: the run grid must be a multiple of the
        patch size per axis, and at least ``min_cells`` per side. For NATTEN
        attention the (dilated) neighbourhood kernel of every block must fit the
        latent grid, so ``min_cells = max(attn_kernel * dilation) * patch``;
        otherwise one patch. ``((1, 1), 1)`` without a DiT.
    """
    if dit is None:
        return (1, 1), 1
    ph, pw = (int(v) for v in dit.patch_size)

    def _largest(value: Any) -> int:
        return (
            max(int(v) for v in value)
            if isinstance(value, (tuple, list))
            else int(value)
        )

    kernel = 0
    for module in dit.modules():
        attn_kernel = getattr(module, "attn_kernel", None)
        if attn_kernel is None:
            continue
        dilation = getattr(module, "na2d_kwargs", {}).get("dilation", 1)
        kernel = max(kernel, _largest(attn_kernel) * _largest(dilation))
    return (ph, pw), max(kernel * max(ph, pw), ph, pw)


def dit_latent_grid(dit: torch.nn.Module, H: int, W: int) -> tuple[int, int]:
    """Latent (post-patchify) grid of an ``(H, W)`` field of a DiT.

    Parameters
    ----------
    dit : torch.nn.Module
        PhysicsNeMo ``DiT``.
    H, W : int
        Pixel grid; must be a multiple of the patch size.

    Returns
    -------
    tuple[int, int]
        ``(H // patch_h, W // patch_w)``.
    """
    ph, pw = dit.patch_size
    if H % ph or W % pw:
        raise ValueError(
            f"field shape {(H, W)} must be a multiple of the network's patch size {(ph, pw)}"
        )
    return H // ph, W // pw


def dit_rope_tables(
    dit: torch.nn.Module, latent_hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE cos / sin tables of a ``natten2d_rope`` DiT for a latent grid.

    On the construction grid these are the DiT's own cached tables; elsewhere they
    are built call-locally, so the DiT's state is not modified.

    Parameters
    ----------
    dit : torch.nn.Module
        PhysicsNeMo ``DiT`` with rotary embeddings.
    latent_hw : tuple[int, int]
        Latent grid from :func:`dit_latent_grid`.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(cos, sin)`` tables of shape ``(h, w, head_dim)``.
    """
    if latent_hw == dit_latent_grid(dit, *dit.input_size):
        return dit.rope(None)
    if build_axial_rope_cos_sin_2d is None:
        raise ImportError("physicsnemo is required to build RoPE tables")
    cos, sin = build_axial_rope_cos_sin_2d(
        *latent_hw,
        dit.rope.head_dim,
        theta=dit.rope.theta,
        device=dit.rope.rope_cos.device,
    )
    return cos.to(dit.rope.rope_cos.dtype), sin.to(dit.rope.rope_sin.dtype)


@contextmanager
def dit_bound_to(
    dit: torch.nn.Module | None, H: int, W: int
) -> Iterator[dict[str, Any]]:
    """Bind a PhysicsNeMo ``DiT`` to an ``(H, W)`` field for one sampling call.

    The generic ``DiT.forward`` reads the detokenizer's patch counts and, for
    NATTEN backends, the attention latent grid and the cached RoPE tables from the
    module, so a field of another size needs that state rebound. This binds it on
    entry, yields the ``attn_kwargs`` to pass to the forward (``latent_hw`` for
    NATTEN backends, nothing for global attention), and puts the original state
    back on exit, including the original RoPE tensors, so results on the
    construction grid are bit-identical afterwards. Callers sharing one network
    must therefore not run concurrently. ``None`` yields ``{}`` and binds nothing.

    Parameters
    ----------
    dit : torch.nn.Module | None
        The DiT, or ``None`` for networks without one.
    H, W : int
        Pixel grid of the field; must be a multiple of the patch size.

    Yields
    ------
    dict[str, Any]
        Forward-time ``attn_kwargs`` for the bound grid.
    """
    if dit is None:
        yield {}
        return
    latent_hw = dit_latent_grid(dit, H, W)
    detok = dit.detokenizer
    saved_patches = (detok.h_patches, detok.w_patches)
    # NATTEN backends carry the latent grid in the forward-time attention kwargs
    natten_kwargs: dict[str, Any] | None = getattr(dit, "attn_kwargs_forward", None)
    if not isinstance(natten_kwargs, dict) or "latent_hw" not in natten_kwargs:
        natten_kwargs = None
    saved_latent = natten_kwargs["latent_hw"] if natten_kwargs is not None else None
    rope = dit.rope if getattr(dit, "_is_rope", False) else None
    saved_rope = (rope.rope_cos, rope.rope_sin, rope._latent_hw) if rope else None
    try:
        detok.h_patches, detok.w_patches = latent_hw
        if natten_kwargs is not None:
            natten_kwargs["latent_hw"] = latent_hw
        if rope is not None:
            rope(latent_hw)  # rebuilds the cos / sin tables only when the grid changed
        yield {"latent_hw": latent_hw} if natten_kwargs is not None else {}
    finally:
        detok.h_patches, detok.w_patches = saved_patches
        if natten_kwargs is not None:
            natten_kwargs["latent_hw"] = saved_latent
        if rope is not None and saved_rope is not None:
            rope.rope_cos, rope.rope_sin, rope._latent_hw = saved_rope

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
from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Literal

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch

Hook = Callable[[xr.DataArray], xr.DataArray]


def initial_output(x: xr.DataArray, output_coords: xr.DataArray) -> xr.DataArray:
    """Return the last input frame on the output structure, padding absent values.

    Coordinates must match exactly to copy values: this does not interpolate,
    diagnose missing variables, or advance the model. The returned field owns its
    data and metadata; the input history remains available for internal rollout.
    """
    signature = coord_array_like(
        output_coords, {"lead_time": x.lead_time.values[-1:]}
    ).copy(deep=True)
    source = x.isel(lead_time=slice(-1, None))
    tensor, _ = source.e2s.to_torch()
    missing_dims = set(source.dims) - set(signature.dims)
    compatible = all(source.sizes[d] == 1 for d in missing_dims)
    # Equal index axes do not identify equal physical curvilinear grids.
    for name in ("lat", "lon"):
        if name in source.coords and name in signature.coords:
            if source.coords[name].ndim > 1 or signature.coords[name].ndim > 1:
                compatible &= source.coords[name].variable.equals(
                    signature.coords[name].variable
                )
    if compatible:
        source = source.squeeze(list(missing_dims), drop=True)
        for dim in signature.dims:
            if dim not in source.dims:
                if signature.sizes[dim] != 1:
                    compatible = False
                    break
                source = source.expand_dims(
                    {
                        dim: (
                            signature.coords[dim].values
                            if dim in signature.coords
                            else 1
                        )
                    }
                )
    if not compatible:
        tensor = tensor.new_full(signature.shape, float("nan"))
    else:
        source = source.transpose(*signature.dims)
        tensor, _ = source.e2s.to_torch()
        for axis, dim in enumerate(signature.dims):
            if dim in source.coords and dim in signature.coords:
                indices = source.get_index(dim).get_indexer(
                    signature.coords[dim].values
                )
            elif source.sizes[dim] == signature.sizes[dim]:
                indices = np.arange(source.sizes[dim])
            else:
                indices = np.full(signature.sizes[dim], -1)
            if np.array_equal(indices, np.arange(source.sizes[dim])):
                continue
            if source.sizes[dim] == 0:
                tensor = tensor.new_full(signature.shape, float("nan"))
                break
            index = torch.as_tensor(indices.clip(min=0), device=tensor.device)
            tensor = tensor.index_select(axis, index)
            mask_shape = [1] * tensor.ndim
            mask_shape[axis] = len(indices)
            mask = torch.as_tensor(indices < 0, device=tensor.device).reshape(
                mask_shape
            )
            tensor = tensor.masked_fill(mask, float("nan"))
    backend: Literal["numpy", "cupy", "torch"] = (
        "numpy"
        if isinstance(x.data, np.ndarray)
        else "cupy" if x.e2s.is_cupy else "torch"
    )
    result = from_torch(tensor.clone(), signature, backend=backend)
    result.encoding = deepcopy(x.encoding)
    return result


class PrognosticMixin:
    """DataArray iterator hooks around core advances and forecast outputs.

    Hooks take and return one DataArray in the original leading dimensions.
    Hooks belong to the iterator, which is the only path that owns a rollout loop.
    ``__call__`` is the single-step primitive and does not apply them: a caller
    holding a single step can transform the DataArray itself, whereas nothing outside
    ``create_iterator`` can reach the state fed back between steps.

    ``front_hook``/``rear_hook`` are single callable slots, not a registration
    list: a caller with more than one transformation to apply composes them into
    one function and assigns that, so the order they run in is visible at the
    assignment site rather than spread across every place that registered one.
    """

    #: Whether the model draws randomness during a rollout. Stochastic models must
    #: implement ``set_rng(seed, reset=True)``; see ``dev/spec/MODEL_CONTRACT_SPEC.md``.
    stochastic: bool = False

    #: Number of forecast outputs produced by each front-hook/core advance.
    #: Rear hooks run for every output; multi-output cores declare their cadence.
    front_hook_interval: int = 1

    @staticmethod
    def _default_hook(x: xr.DataArray) -> xr.DataArray:
        return x

    # Typed as the Hook signature so assigning a plain function — the normal use
    # of this slot — type-checks instead of looking like a method override.
    front_hook: Hook = _default_hook
    rear_hook: Hook = _default_hook

    def clear_hooks(self) -> None:
        """Remove every registered hook, restoring the default pass-through."""
        for name in ("front_hook", "rear_hook"):
            if name in vars(self):
                delattr(self, name)

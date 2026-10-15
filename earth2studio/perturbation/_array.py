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

"""Private array helpers shared by perturbation methods"""

from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.cupy import _get_cupy


def as_host_or_cupy(x: xr.DataArray) -> xr.DataArray:
    """Ensure a DataArray is backed by NumPy or CuPy (not Torch) data

    Parameters
    ----------
    x : xr.DataArray
        Input DataArray

    Returns
    -------
    xr.DataArray
        Input if already NumPy/CuPy backed, otherwise a converted DataArray
    """
    if isinstance(x.variable._data, np.ndarray) or x.e2s.is_cupy:
        return x
    return x.e2s.to_backend("auto")


def array_module(x: xr.DataArray) -> Any:
    """Array module (numpy or cupy) matching the storage backing a DataArray

    Parameters
    ----------
    x : xr.DataArray
        NumPy or CuPy backed DataArray

    Returns
    -------
    Any
        ``numpy`` for host data, ``cupy`` for GPU data
    """
    return _get_cupy() if x.e2s.is_cupy else np


def amplitude_like(amplitude: torch.Tensor, xp: Any, dtype: Any) -> Any:
    """Convert a noise amplitude tensor to an array of the requested module and dtype

    Parameters
    ----------
    amplitude : torch.Tensor
        Noise amplitude, must be broadcastable with the data
    xp : Any
        Array module (numpy or cupy)
    dtype : Any
        Data type of the array the amplitude is applied to

    Returns
    -------
    Any
        Array in the requested module
    """
    return xp.asarray(amplitude.detach().cpu().numpy().astype(dtype))


def randn(xp: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    """Standard normal samples using numpy or cupy

    Parameters
    ----------
    xp : Any
        Array module (numpy or cupy)
    shape : tuple[int, ...]
        Shape of the output array
    dtype : Any
        Floating point data type of the output, float32 or float64

    Returns
    -------
    Any
        Array of standard normal noise
    """
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        dtype = np.dtype(np.float32)
    if xp is np:
        return np.random.default_rng().standard_normal(shape, dtype=dtype)
    return xp.random.standard_normal(shape, dtype=dtype)

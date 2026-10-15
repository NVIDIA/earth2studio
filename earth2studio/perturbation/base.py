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

from typing import Protocol, runtime_checkable

import xarray as xr


# --8<-- [start:perturbation-interface]
@runtime_checkable
class Perturbation(Protocol):
    """Perturbation interface

    A perturbation is a callable that takes a single DataArray holding the state to
    perturb and returns a DataArray of the same shape, dimensions and coordinates. The
    input is not modified in place: methods return a new array or the unmodified input.
    """

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Apply perturbation method to input DataArray

        Parameters
        ----------
        x : xr.DataArray
            xarray DataArray intended to apply perturbation on, with NumPy, CuPy or
            Torch array data. Coordinates and dimensions describe the array

        Returns
        -------
        xr.DataArray
            Perturbed DataArray with the same dimensions, coordinates and attributes
            as the input
        """
        pass


# --8<-- [end:perturbation-interface]

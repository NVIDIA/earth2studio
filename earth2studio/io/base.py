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

from typing import Any, Protocol, runtime_checkable

import torch
import xarray as xr

from earth2studio.utils.type import CoordSystem


# --8<-- [start:io-backend-interface]
@runtime_checkable
class IOBackend(Protocol):
    """Interface for a DataArray IO backend.

    See ``dev/spec/IO_SPEC.md`` for the full contract.
    """

    def add_array(self, template: xr.DataArray) -> None:
        """Create storage for the arrays a template describes.

        Parameters
        ----------
        template : xr.DataArray
            Concrete coordinate signature; field values are never read. Each
            ``variable`` label names one array over the remaining dimensions;
            without a ``variable`` dimension, the template's name does.
        """
        pass

    def write(self, x: xr.DataArray) -> None:
        """Write a field at the positions its coordinate labels identify.

        Parameters
        ----------
        x : xr.DataArray
            NumPy-, CuPy- or Torch-backed field holding a subset of the store's
            labels along each dimension. It is never modified.
        """
        pass

    def flush(self) -> None:
        """Block until every earlier write is visible to readers of the store."""
        pass

    def close(self) -> None:
        """Flush and release resources; later writes raise."""
        pass


# --8<-- [end:io-backend-interface]


@runtime_checkable
class _LegacyIOBackend(Protocol):
    """Tensor and coordinate-dictionary IO interface of unmigrated backends.

    Temporary: removed once every backend implements :class:`IOBackend`.
    """

    def add_array(
        self, coords: CoordSystem, array_name: str | list[str], **kwargs: dict[str, Any]
    ) -> None:
        """Add arrays with the given coordinates."""
        pass

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """Write tensors to the named arrays."""
        pass

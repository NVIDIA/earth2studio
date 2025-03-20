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

from typing import Any, Protocol, runtime_checkable

import torch

from earth2studio.utils.type import CoordSystem


@runtime_checkable
class IOBackend(Protocol):
    """Interface for a generic IO backend."""

    def __init__(
        self,
    ) -> None:
        pass

    def add_array(
        self, coords: CoordSystem, array_name: str | list[str], **kwargs: dict[str, Any]
    ) -> None:
        """
        Add an array with `array_name` to the existing IO backend object.

        Parameters
        ----------
        coords : OrderedDict
            Ordered dictionary of representing the dimensions and coordinate data
            of x.
        array_name : str
            Name of the arrays that will be initialized with coordinates as dimensions.
        kwargs : dict[str, Any], optional
            Optional keyword arguments that will be passed to the IO backend constructor.
        """
        pass

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """
        Write data to the current backend using the passed array_name.

        Parameters
        ----------
        x : torch.Tensor | list[torch.Tensor]
            Tensor(s) to be written to zarr store.
        coords : OrderedDict
            Coordinates of the passed data.
        array_name : str | list[str]
            Name(s) of the array(s) that will be written to.
        """
        pass

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

"""
Prognostic Model Wrapper Skeleton Template

This file demonstrates the required structure for Earth2Studio prognostic model
wrappers. Copy this template and replace TODO comments with actual implementations.

Method ordering is canonical and must be preserved:
1. __init__ — constructor
2. input_coords — input coordinate system
3. output_coords — output coordinate system (decorated @batch_coords())
4. load_default_package — classmethod returning default Package
5. load_model — classmethod loading model from package
6. to — device management (optional, only if non-PyTorch state exists)
7. Private/support methods (e.g., _prepare_input, _normalize, etc.)
8. __call__ — single-step forward (decorated @batch_func())
9. _default_generator — batch-decorated generator (decorated @batch_func())
10. create_iterator — public time-integration entry point
"""

from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

# Optional dependency imports (try/except pattern)
try:
    import optional_package
except ImportError:
    OptionalDependencyFailure("model-name")
    optional_package = None

VARIABLES = [...]  # List of variable names from E2STUDIO_VOCAB


class ModelName(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """One-line description.

    Extended description of the model, its source,
    and any relevant details.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core model instance
    ...additional params...

    Note
    ----
    For more information see: <link to paper/repo>
    """

    # 1. Constructor
    def __init__(self, core_model, ...):
        super().__init__()
        # TODO: Initialize model
        self.register_buffer("device_buffer", torch.empty(0))
        pass

    # 2. Input coordinates
    def input_coords(self) -> CoordSystem:
        # TODO: Define input coordinates
        pass

    # 3. Output coordinates
    @batch_coords()
    def output_coords(
        self,
        input_coords: CoordSystem,
    ) -> CoordSystem:
        # TODO: Define output coordinates
        pass

    # 4. Default package location
    @classmethod
    def load_default_package(cls) -> Package:
        # TODO: Default checkpoint location
        pass

    # 5. Load model from package
    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        # TODO: Load model from package
        pass

    # 6. Device management (optional — only needed for non-PyTorch state)
    def to(
        self,
        device: torch.device | str,
    ) -> PrognosticModel:
        # TODO: Device management
        pass

    # 7. Private/support methods go here
    # e.g., _load_checkpoint(), _prepare_input(), etc.

    # 8. Single step forward
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        # TODO: Single step forward
        pass

    # 9. Batch-decorated generator
    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # TODO: Yield initial condition, then loop
        pass

    # 10. Public iterator entry point
    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # TODO: Setup, then yield from self._default_generator(x, coords)
        pass

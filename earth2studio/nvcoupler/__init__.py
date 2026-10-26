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

"""nvcoupler: couple Earth2Studio models on one clock.

A :class:`Coupler` advances ordinary prognostic and diagnostic models together,
serving each model's declared forcing from the others' outputs through
:class:`Connector` declarations. It is a :class:`~earth2studio.run.Runner`, so
``Pipeline`` drives coupled runs like any other. Design follows NUOPC/ESMF:
models advance in a run sequence derived from their cadences, and exchanges
are matched by variable label, regridded between grids, and reduced over time
windows when a model declares a qualified label such as ``"z1000:mean:48h"``.
"""

from earth2studio.nvcoupler.connector import Connector
from earth2studio.nvcoupler.coupler import Coupler
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.sequence import RunSequence, Slot, derive_sequence
from earth2studio.nvcoupler.vertical import (
    HybridLevels,
    PressureLevels,
    to_pressure_levels,
)

__all__ = [
    "Connector",
    "Coupler",
    "CouplingError",
    "HybridLevels",
    "PressureLevels",
    "RunSequence",
    "Slot",
    "derive_sequence",
    "to_pressure_levels",
]

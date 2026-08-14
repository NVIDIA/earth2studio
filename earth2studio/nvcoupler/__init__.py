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

"""nvcoupler: NUOPC/ESMF-inspired coupling framework for ML inference.

Couples independent AI Earth-system components (prognostic models, data
sources, mediators) through Fields exchanged by standard name, with
per-component cadences, connector regridding, and a configurable run
sequence. See earth2studio/nvcoupler/driver.py for the entry point.
"""

from .api import couple, coupled, describe, describe_html
from .clock import Clock
from .component import (
    CallableComponent,
    Component,
    ConditioningKwargAdapter,
    DataComponent,
    DiagnosticComponent,
    Exchange,
    ExtraTensorAdapter,
    ImportAdapter,
    PrognosticComponent,
    VariableOverwriteAdapter,
)
from .config import from_yaml, to_yaml
from .connector import Connector
from .dictionary import (
    DEFAULT_DICTIONARY,
    CellMethod,
    FieldDictionary,
    FieldEntry,
)
from .dlesym_split import (
    DLESYM_DICTIONARY,
    build_dlesym_driver,
    split_dlesym,
)
from .driver import Driver
from .errors import (
    AmbiguousCouplingError,
    CadenceError,
    CouplingError,
    IncompatibleFieldError,
    SequenceError,
    UnitsMismatchError,
    UnknownFieldError,
    UnmatchedImportError,
    VerticalMismatchError,
)
from .field import Field, State
from .mediator import AccumulationMediator, Mediator, TrailingAverageMediator
from .sequence import (
    ConnectAction,
    MediateAction,
    RunAction,
    RunSequence,
    Slot,
    derive_sequence,
    parse_run_sequence,
)
from .vertical import HybridLevels, PressureLevels

__all__ = [
    "Clock",
    "CallableComponent",
    "DataComponent",
    "DiagnosticComponent",
    "couple",
    "coupled",
    "describe",
    "describe_html",
    "from_yaml",
    "to_yaml",
    "split_dlesym",
    "build_dlesym_driver",
    "DLESYM_DICTIONARY",
    "Component",
    "ConditioningKwargAdapter",
    "Connector",
    "Driver",
    "Exchange",
    "ExtraTensorAdapter",
    "HybridLevels",
    "ImportAdapter",
    "AccumulationMediator",
    "Mediator",
    "PressureLevels",
    "PrognosticComponent",
    "TrailingAverageMediator",
    "VariableOverwriteAdapter",
    "ConnectAction",
    "MediateAction",
    "RunAction",
    "RunSequence",
    "Slot",
    "derive_sequence",
    "parse_run_sequence",
    "DEFAULT_DICTIONARY",
    "CellMethod",
    "FieldDictionary",
    "FieldEntry",
    "Field",
    "State",
    "AmbiguousCouplingError",
    "CadenceError",
    "CouplingError",
    "IncompatibleFieldError",
    "SequenceError",
    "UnitsMismatchError",
    "UnknownFieldError",
    "UnmatchedImportError",
    "VerticalMismatchError",
]

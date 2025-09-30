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

from collections.abc import Callable
from typing import Any

import numpy as np

from earth2studio.lexicon.base import LexiconType


def _to_float32(array: Any) -> Any:
    return array.astype(np.float32)


class MODISFireLexicon(metaclass=LexiconType):
    """Lexicon exposing MODIS Thermal Anomalies daily fields."""

    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        "fire_mask": ("fire_mask", _to_float32),
        "max_frp": ("max_frp", _to_float32),
        "qa": ("qa", _to_float32),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in MODIS Fire lexicon")
        return cls.VOCAB[val]

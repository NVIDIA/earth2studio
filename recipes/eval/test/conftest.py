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

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from earth2studio.data import Random
from earth2studio.models.px import Persistence

# Ensure the recipe root is importable so ``from src.…`` works.
_RECIPE_ROOT = Path(__file__).resolve().parents[1]
if str(_RECIPE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RECIPE_ROOT))

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
VARIABLES = ["t2m", "z500"]


@pytest.fixture()
def small_domain() -> OrderedDict:
    return OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})


@pytest.fixture()
def prognostic(small_domain) -> Persistence:
    return Persistence(variable=VARIABLES, domain_coords=small_domain)


@pytest.fixture()
def data_source(small_domain) -> Random:
    return Random(domain_coords=small_domain)


@pytest.fixture()
def base_cfg(tmp_path) -> OmegaConf:
    return OmegaConf.create(
        {
            "project": "test_eval",
            "run_id": "unit",
            "start_times": ["2024-01-01 00:00:00"],
            "nsteps": 2,
            "ensemble_size": 1,
            "random_seed": 42,
            "model": {"architecture": "earth2studio.models.px.Persistence"},
            "data_source": {"_target_": "earth2studio.data.Random"},
            "output": {
                "path": str(tmp_path / "outputs"),
                "variables": list(VARIABLES),
                "overwrite": True,
                "thread_writers": 0,
                "chunks": {"time": 1, "lead_time": 1},
            },
        }
    )

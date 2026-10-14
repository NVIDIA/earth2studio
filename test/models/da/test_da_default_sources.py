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

import inspect

import pytest

from earth2studio.data import ARCO_ERA5, HRRR, GHCNHourly, UFSObsConv, UFSObsSat
from earth2studio.models.da import (
    CorrDiffCosmoEra5SDA,
    HealDA,
    InterpEquirectangular,
    StormCastSDA,
)


@pytest.mark.parametrize(
    "model_cls,expected",
    [
        (StormCastSDA, (HRRR, GHCNHourly)),
        (CorrDiffCosmoEra5SDA, (ARCO_ERA5, GHCNHourly)),
        (HealDA, (UFSObsConv, UFSObsSat)),
        (InterpEquirectangular, None),
    ],
)
def test_default_sources(model_cls, expected):
    # default_sources does not touch model state, so skip heavy construction
    model = object.__new__(model_cls)
    sources = model.default_sources()
    assert sources == expected
    if expected is not None:
        # Classes must be returned uninstantiated
        assert all(inspect.isclass(s) for s in sources)

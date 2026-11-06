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

"""Fixtures for coupling tests."""

import pytest
from test_coupling_toys import ATMOS_GRID, OCEAN_GRID, Fill


@pytest.fixture
def ic_sources() -> dict[str, Fill]:
    """Initial conditions of the atmosphere/ocean toy: z1000 = 0, sst = 2."""
    return {
        "atmos": Fill(ATMOS_GRID, {"z1000": 0.0, "sst": 2.0}),
        "ocean": Fill(OCEAN_GRID, {"sst": 2.0, "z1000": 0.0}),
    }

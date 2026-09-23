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

"""Variables and metadata published by SFNO.

This list is specific to SFNO and is passed to the DSX coordinator. The shared
``dsx`` code does not define it. SFNO is a global model, and its surface fields
already use east and north directions, so no wind rotation is needed. It
publishes 2 m temperature and 10 m eastward and northward wind. Unlike
StormCast, it does not calculate surface humidity.

Each entry links a DSX ``{variable}`` name to its unit, CF standard name, and
height above ground. The variable names must be allowed by the contract's
``{variable}`` enum. ``test/sfno/test_sfno_workflow.py`` checks this.
"""

from __future__ import annotations

from typing import Any

# variable -> {unit, standardName, heightMeters}
# SFNO publishes temperature and two earth-relative wind components.
VARIABLES: dict[str, dict[str, Any]] = {
    "Temperature": {"unit": "K", "standardName": "air_temperature", "heightMeters": 2},
    "WindU": {"unit": "m s-1", "standardName": "eastward_wind", "heightMeters": 10},
    "WindV": {"unit": "m s-1", "standardName": "northward_wind", "heightMeters": 10},
}

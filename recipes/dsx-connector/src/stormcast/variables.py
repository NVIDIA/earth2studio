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

"""Variables and metadata published by StormCast.

This list is specific to StormCast and is passed to the DSX coordinator. The
shared ``dsx`` code does not define it.

Each entry links a DSX ``{variable}`` name to its unit, CF standard name, and
height above ground. Variable names must be allowed by the contract's
``{variable}`` enum. ``test/stormcast/test_pipeline.py`` checks this.
"""

from __future__ import annotations

from typing import Any

# variable -> {unit, standardName, heightMeters}. ``standardName``/``heightMeters`` anchor the
# human-friendly wire name to CF Conventions without putting CF names on the wire.
VARIABLES: dict[str, dict[str, Any]] = {
    "Temperature": {"unit": "K", "standardName": "air_temperature", "heightMeters": 2},
    "RelativeHumidity": {
        "unit": "percent",
        "standardName": "relative_humidity",
        "heightMeters": 2,
    },
    "WetBulb": {"unit": "K", "standardName": "wet_bulb_temperature", "heightMeters": 2},
    "WindU": {"unit": "m s-1", "standardName": "eastward_wind", "heightMeters": 10},
    "WindV": {"unit": "m s-1", "standardName": "northward_wind", "heightMeters": 10},
}

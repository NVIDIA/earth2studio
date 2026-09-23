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

"""Define the shared format used to pass forecasts from model workflows to DSX publishing code.

``ForecastSeries`` is the connector's internal format for one site and weather variable.
``member_values`` contains one list per ensemble member, with one value for each lead time. One
member represents a deterministic forecast; multiple members represent an ensemble.

The DSX contract adapter turns ``ForecastSeries`` into messages that follow the DSX format. For
ensembles, it calculates summary statistics and can include each member's values. This means model
workflows do not need to build DSX messages themselves.

Metadata about sites and variables is stored separately because it follows its own publishing path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ForecastSeries:
    """Forecast values for one site and weather variable.

    A series contains one or more ensemble members. Each member has exactly one value for each lead
    time. A series with one member is a deterministic forecast.

    Fields cannot be reassigned after construction because the dataclass uses ``frozen=True``.
    However, the nested lists can still be changed and should be treated as read-only. The
    publisher copies their values when it queues a message.

    Descriptive metadata such as units, coordinates, standard names, and measurement heights is
    supplied separately.

    Parameters
    ----------
    site_id : str
        Site identifier included in the MQTT topic.
    variable : str
        DSX weather-variable name allowed by the contract.
    init_ms : int
        Initial-condition time as Unix epoch milliseconds.
    lead_seconds : list[int]
        Forecast times measured in seconds after ``init_ms``.
    member_values : list[list[float | None]]
        One list of values per ensemble member. Each list must match ``lead_seconds`` in length.
        ``None`` and non-finite numbers are treated as missing when a message is built.
    model : str
        Identifier of the model that produced the forecast.
    """

    site_id: str
    variable: str
    init_ms: int
    lead_seconds: list[int]
    member_values: list[list[float | None]]
    model: str

    def __post_init__(self) -> None:
        """Check that every ensemble member matches the forecast lead times.

        Raises
        ------
        ValueError
            If there are no members or a member does not contain exactly one value per lead time.
        """
        if not self.member_values:
            raise ValueError(
                f"ForecastSeries for {self.site_id}/{self.variable} must contain at least one member"
            )
        lead_count = len(self.lead_seconds)
        for member_index, member in enumerate(self.member_values):
            if len(member) != lead_count:
                raise ValueError(
                    f"member {member_index} has {len(member)} values but the forecast has "
                    f"{lead_count} lead times "
                    f"for {self.site_id}/{self.variable}"
                )

    @property
    def member_count(self) -> int:
        """Return the number of ensemble members."""
        return len(self.member_values)

    @property
    def is_probabilistic(self) -> bool:
        """True if more than one member is present."""
        return len(self.member_values) > 1

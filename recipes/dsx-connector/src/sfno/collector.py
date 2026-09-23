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

"""Collect SFNO output and convert it into ``ForecastSeries`` objects.

SFNO provides temperature and earth-relative wind fields on a regular latitude/longitude grid. The
collector maps ``t2m`` to ``Temperature``, ``u10m`` to ``WindU``, and ``v10m`` to ``WindV``. These
variables need no wind rotation or humidity calculation. Site extraction reads only the configured
grid cells from each field, so an entire field does not need to be copied from the GPU.

Earth2Studio sends deterministic or ensemble output to :meth:`SFNOCollector.write`. The collector
buffers values by site, lead time, and ensemble member. :meth:`SFNOCollector.collect` then groups
them into one :class:`~src.shared.forecast.ForecastSeries` per site and variable. It leaves the
buffer unchanged; the workflow clears it only after the complete batch has been validated and
staged successfully.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..shared.forecast import ForecastSeries
from ..shared.lead_time import lead_seconds_from_coords
from ..shared.site_extraction import SiteExtractor

# Map SFNO output names to forecast variable names.
_VAR_MAP = {"t2m": "Temperature", "u10m": "WindU", "v10m": "WindV"}

# Variables the workflow requests from SFNO and the collector accepts.
SOURCE_VARS = list(_VAR_MAP)


class SFNOCollector:
    """Buffer SFNO output and build one ``ForecastSeries`` per site and variable.

    Earth2Studio uses this collector as an output backend during inference and calls its
    :meth:`add_array` and :meth:`write` methods. The collector does not need to inherit from an
    Earth2Studio class or import Torch.

    Parameters
    ----------
    extractor : SiteExtractor
        Extracts configured site values from each two-dimensional model field.
    model_id : str
        Model identifier stored in every resulting ``ForecastSeries``.
    init_ms : int, optional
        Forecast initial-condition time in Unix epoch milliseconds. :meth:`begin_cycle` updates it
        for each cycle.
    """

    def __init__(
        self, extractor: SiteExtractor, model_id: str, init_ms: int = 0
    ) -> None:
        self.extractor = extractor
        self.model_id = model_id
        # Store source values by site, lead time in seconds, and ensemble-member index.
        self._buffer: dict[tuple[str, int, int], dict[str, float | None]] = defaultdict(
            dict
        )
        self.begin_cycle(init_ms)

    # --- IOBackend protocol ---------------------------------------------------
    def add_array(
        self, coords: dict[str, np.ndarray], array_name: str | np.ndarray, **kwargs: Any
    ) -> None:
        """Accept Earth2Studio's array-creation callback without creating persistent storage."""
        pass

    def write(
        self, x: Any, coords: dict[str, np.ndarray], array_name: str | np.ndarray
    ) -> None:
        """Extract and buffer site values from one Earth2Studio output write.

        Parameters
        ----------
        x : Any
            One field or a list of fields represented by NumPy arrays or Torch tensors.
        coords : CoordSystem
            Coordinates containing one ``lead_time`` and, for an ensemble, member indices under
            ``ensemble``.
        array_name : np.ndarray | str
            One variable name per field.

        Raises
        ------
        KeyError
            If ``coords`` does not contain ``lead_time``.
        ValueError
            If the lead time, ensemble-member coordinates, field count, field shape, or duplicate
            detection is invalid.
        """
        fields = x if isinstance(x, list) else [x]
        variable_names = (
            [array_name] if isinstance(array_name, str) else list(array_name)
        )
        if len(fields) != len(variable_names):
            raise ValueError(
                f"received {len(fields)} field(s) but {len(variable_names)} variable name(s)"
            )

        lead_seconds = lead_seconds_from_coords(coords)
        # Deterministic output has no ensemble coordinate and is treated as member zero.
        raw_member_ids = coords.get("ensemble")
        if raw_member_ids is None:
            member_ids = [0]
        else:
            member_array = np.asarray(raw_member_ids)
            if member_array.size == 0 or not np.issubdtype(
                member_array.dtype, np.integer
            ):
                raise ValueError(
                    "ensemble coordinates must contain at least one integer member index"
                )
            member_ids = [int(member_id) for member_id in member_array.ravel()]
            if len(set(member_ids)) != len(member_ids):
                raise ValueError(
                    "ensemble coordinates contain duplicate member indices"
                )

        for member_id in member_ids:
            if not (0 <= member_id < self._member_count):
                raise ValueError(
                    f"ensemble member index {member_id} out of range "
                    f"[0, {self._member_count})"
                )

        for field, variable_name in zip(fields, variable_names, strict=True):
            variable_name = str(variable_name)
            if variable_name not in SOURCE_VARS:
                continue
            member_fields = self._reshape_members(field, len(member_ids))
            for member_position, member_id in enumerate(member_ids):
                site_values = self.extractor.extract(member_fields[member_position])
                for site_id, value in site_values.items():
                    buffered_values = self._buffer[(site_id, lead_seconds, member_id)]
                    if variable_name in buffered_values:
                        raise ValueError(
                            f"duplicate write for member {member_id} site {site_id} "
                            f"lead {lead_seconds} variable {variable_name}"
                        )
                    buffered_values[variable_name] = value

    @staticmethod
    def _reshape_members(field: Any, member_count: int) -> Any:
        """Validate and reshape model output to ``(member, lat, lon)``.

        Earth2Studio output may include leading batch, time, lead-time, and ensemble dimensions.
        The site extractor needs one two-dimensional latitude/longitude field per member. This
        method verifies that the leading dimensions contain the expected number of members and
        reshapes them into a single member axis. The result remains on the input's current device.

        Raises
        ------
        ValueError
            If the input has fewer than two dimensions or its leading dimensions do not contain the
            expected number of members.
        """
        shape = tuple(field.shape)
        if len(shape) < 2:
            raise ValueError(
                f"expected a field with at least 2 dimensions, got shape {shape}"
            )
        if int(np.prod(shape[:-2])) != member_count:
            raise ValueError(
                f"expected {member_count} member(s) in the leading dimensions, "
                f"got shape {shape}"
            )
        return field.reshape(member_count, shape[-2], shape[-1])

    # --- cycle / collect ------------------------------------------------------
    def begin_cycle(self, init_ms: int, member_count: int = 1) -> None:
        """Prepare the collector for a new forecast cycle.

        This updates the initial-condition time and expected ensemble size, then removes values
        buffered for the previous cycle. One member represents a deterministic forecast.

        Parameters
        ----------
        init_ms : int
            Forecast initial-condition time in Unix epoch milliseconds.
        member_count : int, optional
            Number of ensemble members expected for the cycle. The default is one.

        Raises
        ------
        ValueError
            If ``init_ms`` is not an integer or ``member_count`` is not a positive integer.
        """
        if isinstance(init_ms, bool) or not isinstance(init_ms, int):
            raise ValueError("init_ms must be an integer")
        if (
            isinstance(member_count, bool)
            or not isinstance(member_count, int)
            or member_count < 1
        ):
            raise ValueError("member_count must be a positive integer")

        self.init_ms = init_ms
        self._member_count = member_count
        self._buffer.clear()

    def clear_buffer(self) -> None:
        """Remove buffered values after the workflow stages a cycle successfully."""
        self._buffer.clear()

    def collect(self) -> list[ForecastSeries]:
        """Build one ``ForecastSeries`` for each buffered site and variable.

        Lead times are sorted, and ensemble members are placed in member-index order. The method
        leaves the buffer unchanged so the workflow can clear it only after staging succeeds.

        Returns
        -------
        list[ForecastSeries]
            Forecast series built from the buffered values.

        Raises
        ------
        ValueError
            If an ensemble member is missing or members do not contain the same lead times.
        """
        # grouped_values[site][variable][member][lead_time] = value
        grouped_values: dict[str, dict[str, dict[int, dict[int, float | None]]]] = (
            defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
        for (
            site_id,
            lead_seconds,
            member_id,
        ), source_values in self._buffer.items():
            for source_name, variable_name in _VAR_MAP.items():
                if source_name in source_values:
                    value = source_values[source_name]
                    grouped_values[site_id][variable_name][member_id][lead_seconds] = (
                        None if value is None else float(value)
                    )

        expected_member_ids = list(range(self._member_count))
        forecasts: list[ForecastSeries] = []
        for site_id, variables in grouped_values.items():
            for variable_name, values_by_member in variables.items():
                present_member_ids = sorted(values_by_member)
                if present_member_ids != expected_member_ids:
                    raise ValueError(
                        f"expected members 0..{self._member_count - 1} for "
                        f"{site_id}/{variable_name}, got {present_member_ids}"
                    )

                reference_leads = set(values_by_member[0])
                for member_id in expected_member_ids[1:]:
                    member_leads = set(values_by_member[member_id])
                    if member_leads != reference_leads:
                        raise ValueError(
                            f"ensemble members have different lead times for "
                            f"{site_id}/{variable_name}: member 0 has "
                            f"{sorted(reference_leads)}, member {member_id} has "
                            f"{sorted(member_leads)}"
                        )

                leads = sorted(reference_leads)
                forecasts.append(
                    ForecastSeries(
                        site_id=site_id,
                        variable=variable_name,
                        init_ms=self.init_ms,
                        lead_seconds=leads,
                        member_values=[
                            [values_by_member[member_id][lead] for lead in leads]
                            for member_id in expected_member_ids
                        ],
                        model=self.model_id,
                    )
                )
        return forecasts

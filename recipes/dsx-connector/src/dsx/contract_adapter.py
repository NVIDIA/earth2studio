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

"""Convert internal forecast data into DSX topics and payload dictionaries.

This module contains pure conversion functions. It does not connect to the broker or depend on
model-specific code. A payload is the content of an MQTT message. This module represents it as a
Python dictionary, which the publisher later encodes as JSON.

Its two public functions are:

- ``series_to_bundle`` converts a deterministic or ensemble forecast into a forecast topic and
  payload.
- ``site_metadata_message`` converts site and variable information into a metadata topic and
  payload.

``publisher.py`` publishes the resulting metadata messages as retained messages.

The adapter replaces non-finite values with ``null`` and verifies that every value array matches
the number of forecast lead times.

For deterministic forecasts, ``values`` contains the single member and ``memberCount`` is 1.
Ensemble forecasts include the mean in ``values``, population standard deviation, minimum,
maximum, selected percentiles, and member count. Raw members are included only when requested.

Each payload uses new lists and dictionaries, so later changes to the source forecast cannot alter
a queued message.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..shared.forecast import ForecastSeries

# Percentiles included in ensemble summaries.
_PERCENTILES = (10, 50, 90)
_DECIMAL_PLACES = 3

# Each workflow defines its own variable metadata. src/dsx/coordinator.py passes that metadata to
# this module when building DSX messages. Keeping the catalog outside the DSX core allows different
# workflows to publish different variable sets.


def _clean(value: float | None) -> float | None:
    """Round finite values to three decimal places and replace non-finite values with None.

    JSON encodes None as ``null`` and does not support NaN or infinity.
    """
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else round(value, _DECIMAL_PLACES)


def _finite(value: float | None) -> float | None:
    """Replace non-finite values with None while preserving precision for calculations."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _clean_values_for_leads(
    values: list[float | None],
    lead_seconds: list[int],
    field_name: str,
    site: str,
    variable: str,
) -> list[float | None]:
    """Require one value per forecast lead, then clean each value."""
    if len(values) != len(lead_seconds):
        raise ValueError(
            f"{field_name} length {len(values)} != leadSeconds length {len(lead_seconds)} for "
            f"{site}/{variable}"
        )
    return [_clean(value) for value in values]


def _reduce_ensemble(
    member_values: list[list[float | None]], lead_count: int
) -> dict[str, Any]:
    """Calculate ensemble summaries for each forecast lead.

    Statistics use full-precision member values and are rounded afterward. If any member is missing
    or non-finite at a lead, every summary for that lead is ``null``. This prevents calculations
    from silently using fewer ensemble members.
    """
    values: list[float | None] = [None] * lead_count
    standard_deviation: list[float | None] = [None] * lead_count
    minimum: list[float | None] = [None] * lead_count
    maximum: list[float | None] = [None] * lead_count
    percentiles: dict[str, list[float | None]] = {
        f"p{percentile}": [None] * lead_count for percentile in _PERCENTILES
    }
    for lead_index in range(lead_count):
        column = [_finite(member[lead_index]) for member in member_values]
        if any(value is None for value in column):
            continue  # any invalid member -> all summaries null at this lead
        member_array = np.asarray(column, dtype=float)
        values[lead_index] = round(float(member_array.mean()), _DECIMAL_PLACES)
        standard_deviation[lead_index] = round(
            float(member_array.std(ddof=0)), _DECIMAL_PLACES
        )
        minimum[lead_index] = round(float(member_array.min()), _DECIMAL_PLACES)
        maximum[lead_index] = round(float(member_array.max()), _DECIMAL_PLACES)
        for percentile, percentile_value in zip(
            _PERCENTILES,
            np.percentile(member_array, _PERCENTILES, method="linear"),
        ):
            percentiles[f"p{percentile}"][lead_index] = round(
                float(percentile_value), _DECIMAL_PLACES
            )
    return {
        "values": values,
        "standardDeviation": standard_deviation,
        "minimum": minimum,
        "maximum": maximum,
        "percentiles": percentiles,
    }


def series_to_bundle(
    series: ForecastSeries,
    topic_prefix: str,
    product: str,
    pub_ms: int,
    include_members: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Convert a :class:`~src.shared.forecast.ForecastSeries` into a DSX forecast topic and payload.

    For a deterministic forecast, ``values`` contains the cleaned member values and
    ``memberCount`` is 1. Ensemble summary fields are omitted.

    For an ensemble forecast, ``values`` contains the mean. The payload also includes population
    standard deviation, minimum, maximum, selected percentiles, and member count. Raw member values
    are included only when requested.

    All lists in the payload are copied so later changes to the source series cannot alter a queued
    message.

    Parameters
    ----------
    series : ForecastSeries
        Forecast for one site and variable.
    topic_prefix : str
        Beginning of the MQTT topic, such as ``Weather/v1/PUB``.
    product : str
        Product identifier included in the topic.
    pub_ms : int
        Initial publication time in Unix epoch milliseconds. The publisher updates this value
        immediately before sending.
    include_members : bool, optional
        Whether to include raw ensemble member values, by default False

    Returns
    -------
    tuple[str, dict[str, Any]]
        MQTT topic and forecast payload dictionary.

    Raises
    ------
    ValueError
        If the series has no members or a member does not contain one value per forecast lead.
    """
    site_id = series.site_id
    variable = series.variable
    member_count = series.member_count
    if member_count == 0:
        raise ValueError(
            f"ForecastSeries for {site_id}/{variable} has no members "
            "(empty member_values)"
        )
    lead_seconds = list(series.lead_seconds)
    payload: dict[str, Any] = {
        "initTime": series.init_ms,
        "leadSeconds": lead_seconds,
    }
    if member_count == 1:
        payload["values"] = _clean_values_for_leads(
            series.member_values[0],
            lead_seconds,
            "values",
            site_id,
            variable,
        )
    else:
        for member_index, member in enumerate(series.member_values):
            if len(member) != len(lead_seconds):
                raise ValueError(
                    f"member {member_index} length {len(member)} != leadSeconds length "
                    f"{len(lead_seconds)} for {site_id}/{variable}"
                )
        payload.update(_reduce_ensemble(series.member_values, len(lead_seconds)))
        if include_members:
            payload["members"] = [
                [_clean(value) for value in member] for member in series.member_values
            ]
    payload["memberCount"] = member_count
    payload["publicationTime"] = pub_ms
    payload["model"] = series.model
    topic = f"{topic_prefix}/Forecast/{product}/{site_id}/{variable}"
    return topic, payload


def site_metadata_message(
    site: dict[str, Any],
    variable: str,
    spec: dict[str, Any],
    topic_prefix: str,
    product: str,
    model_id: str,
    horizon_seconds: int,
    cadence_seconds: int,
    inputs: list[dict[str, str]],
) -> tuple[str, dict[str, Any]]:
    """Build a DSX metadata topic and payload for one site and variable.

    The metadata describes the site, variable, model, forecast horizon, and update schedule. The
    publisher sends it as a retained message at startup and during heartbeats.

    Parameters
    ----------
    site : dict[str, Any]
        Site configuration containing ``id``, ``lat``, and ``lon``.
    variable : str
        Forecast variable name defined by the DSX contract.
    spec : dict[str, Any]
        Metadata for the variable. ``unit`` is required; ``standardName`` and ``heightMeters`` are
        optional.
    topic_prefix : str
        Beginning of the MQTT topic, such as ``Weather/v1/PUB``.
    product : str
        Product identifier included in the topic.
    model_id : str
        Identifier of the model that produced the forecast.
    horizon_seconds : int
        Maximum forecast lead time in seconds.
    cadence_seconds : int
        Expected number of seconds between forecast cycles.
    inputs : list[dict[str, str]]
        Input data sources and the role of each source in producing the forecast.

    Returns
    -------
    tuple[str, dict[str, Any]]
        MQTT topic and metadata payload dictionary.

    Raises
    ------
    ValueError
        If the variable metadata does not provide a non-empty unit.
    """
    unit = spec.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError(
            f"variable {variable!r} catalog entry is missing a non-empty 'unit' "
            "(the retained Metadata unit is required to interpret the values)"
        )
    unit = unit.strip()
    metadata: dict[str, Any] = {
        "site": site["id"],
        "variable": variable,
        "unit": unit,
        "model": model_id,
        "lat": float(site["lat"]),
        # Normalize either common longitude convention to the contract's [-180, 180) range.
        "lon": ((float(site["lon"]) + 180.0) % 360.0) - 180.0,
        "horizonSeconds": horizon_seconds,
        "issueCadenceSeconds": cadence_seconds,
        # Copy input records so later configuration changes cannot alter this payload.
        "inputs": [dict(input_item) for input_item in inputs],
        "description": f"{variable} forecast for {site['id']} from {model_id}.",
    }
    if "standardName" in spec:
        metadata["standardName"] = spec["standardName"]
    if "heightMeters" in spec:
        metadata["heightMeters"] = spec["heightMeters"]
    topic = f"{topic_prefix}/Metadata/{product}/{site['id']}/{variable}"
    return topic, metadata

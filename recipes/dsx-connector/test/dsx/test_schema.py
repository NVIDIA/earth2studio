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


"""DSX weather schema and topic validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.dsx.schema import WeatherSchema


@pytest.fixture(scope="module")
def schema() -> WeatherSchema:
    return WeatherSchema.load()


@pytest.mark.parametrize("contents", ["", "- not\n- a\n- dictionary\n"])
def test_load_rejects_yaml_without_top_level_dictionary(
    tmp_path: Path, contents: str
) -> None:
    schema_path = tmp_path / "weather.yaml"
    schema_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="dictionary at its top level"):
        WeatherSchema.load(schema_path)


def test_variable_enum_consistent(schema: WeatherSchema) -> None:
    d = schema.doc
    enums = [
        schema.channel_enum("forecast", "variable"),
        schema.channel_enum("metadata", "variable"),
        d["components"]["messages"]["ForecastMetadataMessage"]["payload"]["properties"][
            "variable"
        ]["enum"],
    ]
    assert all(e == enums[0] for e in enums)
    assert "WetBulb" in enums[0]


def test_bad_percentile_key_rejected(schema: WeatherSchema) -> None:
    from jsonschema.exceptions import ValidationError  # type: ignore

    with pytest.raises(ValidationError):
        schema.validate_payload(
            "ForecastBundleMessage",
            {
                "initTime": 1,
                "leadSeconds": [0],
                "values": [1.0],
                "publicationTime": 2,
                "model": "sfno",
                "memberCount": 2,
                "percentiles": {"pXX": [1.0]},
            },
        )


@pytest.mark.parametrize(
    ("message_name", "payload"),
    [
        (
            "ForecastBundleMessage",
            {
                "initTime": 1,
                "leadSeconds": [0],
                "values": [20.0],
                "memberCount": 1,
                "publicationTime": 2,
                "model": "",
            },
        ),
        (
            "ForecastMetadataMessage",
            {
                "site": "dc-omaha-1",
                "variable": "Temperature",
                "unit": "K",
                "model": "",
            },
        ),
    ],
)
def test_empty_model_id_rejected(
    schema: WeatherSchema, message_name: str, payload: dict
) -> None:
    from jsonschema.exceptions import ValidationError  # type: ignore

    with pytest.raises(ValidationError):
        schema.validate_payload(message_name, payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("lat", -90.1), ("lat", 90.1), ("lon", -180.1), ("lon", 180)],
)
def test_metadata_location_range_enforced(
    schema: WeatherSchema, field: str, value: float
) -> None:
    from jsonschema.exceptions import ValidationError  # type: ignore

    payload = {
        "site": "dc-omaha-1",
        "variable": "Temperature",
        "unit": "K",
        "model": "stormcast-conus",
        "lat": 41.26,
        "lon": -95.94,
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        schema.validate_payload("ForecastMetadataMessage", payload)


def test_topic_matching(schema: WeatherSchema) -> None:
    m = schema.match_topic(
        "forecast", "Weather/v1/PUB/Forecast/conus-site-weather/dc-omaha-1/WetBulb"
    )
    assert m == {
        "product": "conus-site-weather",
        "site": "dc-omaha-1",
        "variable": "WetBulb",
    }
    assert schema.match_topic("forecast", "BMS/v1/PUB/x/y") is None
    # {variable} outside the enum must not match (mirrors the DSX bus validator).
    assert (
        schema.match_topic(
            "forecast", "Weather/v1/PUB/Forecast/conus-site-weather/s/NotAVariable"
        )
        is None
    )


def test_static_topic_match_returns_empty_parameter_dictionary() -> None:
    schema = WeatherSchema(
        {"channels": {"health": {"address": "Weather/v1/PUB/Health"}}}
    )
    assert schema.match_topic("health", "Weather/v1/PUB/Health") == {}


def test_product_slug_pattern_enforced(schema: WeatherSchema) -> None:
    # {product} is a lowercase slug (x-pattern in the contract); bad slugs must not match.
    good = "Weather/v1/PUB/Forecast/conus-site-weather/s/WetBulb"
    assert schema.match_topic("forecast", good) is not None
    for bad in (
        "Conus",
        "a.b",
        "a_b",
        "-lead",
        "a--b",
    ):  # case/dot/underscore/hyphen rules
        topic = f"Weather/v1/PUB/Forecast/{bad}/s/WetBulb"
        assert (
            schema.match_topic("forecast", topic) is None
        ), f"{bad!r} should be rejected"
    # The Weather contract limits product slugs to 63 characters.
    at63, at64 = "a" * 63, "a" * 64
    assert (
        schema.match_topic("forecast", f"Weather/v1/PUB/Forecast/{at63}/s/WetBulb")
        is not None
    )
    assert (
        schema.match_topic("forecast", f"Weather/v1/PUB/Forecast/{at64}/s/WetBulb")
        is None
    )

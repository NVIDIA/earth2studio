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

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import RoutedObsSource, fetch_dataframe


class PhooObsSource:
    """Minimal DataFrameSource returning one row per requested variable."""

    def __init__(self, schema: pa.Schema, tag: str):
        self.SCHEMA = schema
        self.tag = tag
        self.calls: list[tuple] = []

    def __call__(self, time, variable, fields=None):
        self.calls.append((time, list(variable), fields))
        columns = fields if fields is not None else self.SCHEMA.names
        rows = [
            {
                name: (v if name == "variable" else self.tag)
                for name in columns
                if name != "observation"
            }
            | {"observation": 1.0}
            for v in variable
        ]
        return pd.DataFrame(rows)[list(columns)]


SCHEMA_A = pa.schema(
    [
        pa.field("time", pa.timestamp("ns")),
        pa.field("lat", pa.float32()),
        pa.field("lon", pa.float32()),
        pa.field("observation", pa.float32()),
        pa.field("variable", pa.string()),
        pa.field("only_in_a", pa.float32()),
    ]
)
SCHEMA_B = pa.schema(
    [
        pa.field("time", pa.timestamp("ns")),
        pa.field("lat", pa.float32()),
        pa.field("lon", pa.float32()),
        pa.field("observation", pa.float32()),
        pa.field("variable", pa.string()),
        pa.field("only_in_b", pa.string()),
    ]
)

TIME = np.array([np.datetime64("2024-01-01T12:00:00")])


def _build_routed():
    src_a = PhooObsSource(SCHEMA_A, "a")
    src_b = PhooObsSource(SCHEMA_B, "b")
    routed = RoutedObsSource({("atms", "mhs"): src_a, ("iasi", "airs"): src_b})
    return routed, src_a, src_b


def test_common_schema():
    routed, _, _ = _build_routed()
    assert routed.SCHEMA.names == ["time", "lat", "lon", "observation", "variable"]
    assert routed.variables == ["atms", "mhs", "iasi", "airs"]


def test_routing_dispatch():
    routed, src_a, src_b = _build_routed()
    df = routed(TIME, ["atms", "iasi", "mhs"])

    # Each backend called once with only its own variables
    assert len(src_a.calls) == 1
    assert src_a.calls[0][1] == ["atms", "mhs"]
    assert len(src_b.calls) == 1
    assert src_b.calls[0][1] == ["iasi"]

    # Concatenated result covers all requested variables on common columns
    assert sorted(df["variable"]) == ["atms", "iasi", "mhs"]
    assert list(df.columns) == routed.SCHEMA.names


def test_single_backend_only():
    routed, src_a, src_b = _build_routed()
    df = routed(TIME, "atms")
    assert len(src_a.calls) == 1
    assert len(src_b.calls) == 0
    assert list(df["variable"]) == ["atms"]


def test_fields_passthrough():
    routed, src_a, _ = _build_routed()
    df = routed(TIME, ["atms"], fields=["time", "observation", "variable"])
    assert src_a.calls[0][2] == ["time", "observation", "variable"]
    assert list(df.columns) == ["time", "observation", "variable"]


def test_unknown_variable_raises():
    routed, _, _ = _build_routed()
    with pytest.raises(ValueError, match="no route"):
        routed(TIME, ["bogus"])


def test_duplicate_route_raises():
    src = PhooObsSource(SCHEMA_A, "a")
    with pytest.raises(ValueError, match="more than one source"):
        RoutedObsSource({"atms": src, ("atms", "mhs"): src})


def test_empty_routes_raises():
    with pytest.raises(ValueError, match="At least one route"):
        RoutedObsSource({})


def test_disjoint_schemas_raise():
    src_a = PhooObsSource(pa.schema([pa.field("x", pa.float32())]), "a")
    src_b = PhooObsSource(pa.schema([pa.field("y", pa.float32())]), "b")
    with pytest.raises(ValueError, match="no common schema"):
        RoutedObsSource({"atms": src_a, "iasi": src_b})


def test_fetch_dataframe_integration():
    routed, _, _ = _build_routed()
    df = fetch_dataframe(routed, TIME, np.array(["atms", "iasi"]))
    assert np.all(df.attrs["request_time"] == TIME)
    assert sorted(df["variable"]) == ["atms", "iasi"]

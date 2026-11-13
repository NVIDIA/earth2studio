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

"""``map_field`` selects within a source and refuses to extrapolate beyond it."""

from collections import OrderedDict

import numpy as np
import pytest
import xarray as xr

from earth2studio.run import map_field


def _field(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    values = np.arange(len(lat) * len(lon), dtype=np.float32).reshape(
        1, len(lat), len(lon)
    )
    return xr.DataArray(
        values,
        dims=("variable", "lat", "lon"),
        coords={"variable": ["t2m"], "lat": lat, "lon": lon},
    )


GLOBAL_LAT = np.linspace(90, -90, 181)
GLOBAL_LON = np.arange(0, 360, 1.0)


@pytest.mark.parametrize(
    "lat, lon",
    [
        (GLOBAL_LAT[:-1], GLOBAL_LON),  # 721-vs-720 style: drop the south pole
        (GLOBAL_LAT[::-1], GLOBAL_LON),  # flipped latitude
        (np.arange(60, 20, -1.0), np.arange(200, 300, 1.0)),  # crop
        (GLOBAL_LAT, GLOBAL_LON + 0.4),  # offset grid, within one gap of the edge
    ],
)
def test_covered_targets_select(lat: np.ndarray, lon: np.ndarray) -> None:
    field = _field(GLOBAL_LAT, GLOBAL_LON)
    mapped = map_field(field, OrderedDict(lat=lat, lon=lon))
    np.testing.assert_array_equal(mapped.lat, lat)
    np.testing.assert_array_equal(mapped.lon, lon)
    expected = field.sel(lat=lat, lon=lon, method="nearest")
    np.testing.assert_array_equal(mapped.values, expected.values)


def test_regional_source_on_global_target_raises() -> None:
    field = _field(np.arange(60, 20, -1.0), np.arange(200, 300, 1.0))
    with pytest.raises(ValueError, match="beyond the source's range"):
        map_field(field, OrderedDict(lat=GLOBAL_LAT, lon=GLOBAL_LON))


def test_longitude_convention_mismatch_raises() -> None:
    field = _field(GLOBAL_LAT, np.arange(-180, 180, 1.0))
    with pytest.raises(ValueError, match="0..360 versus -180..180"):
        map_field(field, OrderedDict(lon=GLOBAL_LON))


def test_coarser_source_repeats_neighbouring_values() -> None:
    # Resolution is not checked: a coarser source is selected by nearest label.
    field = _field(GLOBAL_LAT, GLOBAL_LON)
    mapped = map_field(field, OrderedDict(lon=np.arange(0, 360, 0.25)))
    assert mapped.sizes["lon"] == 1440


def test_single_label_source_requires_exact_labels() -> None:
    field = _field(np.array([45.0]), GLOBAL_LON)
    assert map_field(field, OrderedDict(lat=np.array([45.0]))).sizes["lat"] == 1
    with pytest.raises(ValueError, match="single label"):
        map_field(field, OrderedDict(lat=np.array([44.0])))


def test_missing_variable_names_the_missing_labels() -> None:
    field = _field(GLOBAL_LAT, GLOBAL_LON)
    with pytest.raises(
        ValueError, match=r"Missing labels for coordinate variable: \['ws10m'\]"
    ):
        map_field(field, OrderedDict(variable=np.array(["t2m", "ws10m"])))


def test_unsigned_source_labels_are_covered() -> None:
    # Unsigned labels must not wrap in the coverage bounds or distances.
    field = _field(np.array([0, 10], dtype=np.uint32), GLOBAL_LON)
    mapped = map_field(field, OrderedDict(lat=np.array([5.0, 10.0])))
    np.testing.assert_array_equal(mapped["lat"], [5.0, 10.0])
    with pytest.raises(ValueError, match="beyond the source's range"):
        map_field(field, OrderedDict(lat=np.array([25.0])))

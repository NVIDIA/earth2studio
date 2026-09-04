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
import pytest

from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.points import PointSet


def test_default_labels_are_integer_index():
    p = PointSet(lat=np.array([1.0, 2.0, 3.0]), lon=np.array([4.0, 5.0, 6.0]))
    assert len(p) == 3
    assert np.array_equal(p.labels(), np.arange(3))
    assert list(p.grid_coords()) == ["point"]
    assert np.array_equal(p.grid_coords()["point"], np.arange(3))


def test_names_become_labels():
    p = PointSet(
        lat=np.array([1.0, 2.0]), lon=np.array([3.0, 4.0]), names=("a", "b")
    )
    assert np.array_equal(p.labels(), np.array(["a", "b"]))


def test_mismatched_lat_lon_length_raises():
    with pytest.raises(CouplingError, match="same length"):
        PointSet(lat=np.array([1.0, 2.0]), lon=np.array([3.0]))


def test_non_1d_raises():
    with pytest.raises(CouplingError, match="1-D"):
        PointSet(lat=np.array([[1.0, 2.0]]), lon=np.array([[3.0, 4.0]]))


def test_empty_raises():
    with pytest.raises(CouplingError, match="at least one point"):
        PointSet(lat=np.array([]), lon=np.array([]))


def test_names_length_mismatch_raises():
    with pytest.raises(CouplingError, match="names has"):
        PointSet(lat=np.array([1.0, 2.0]), lon=np.array([3.0, 4.0]), names=("a",))


def test_signature_stable_and_distinguishes_locations():
    p1 = PointSet(lat=np.array([1.0]), lon=np.array([2.0]))
    p2 = PointSet(lat=np.array([1.0]), lon=np.array([2.0]))
    p3 = PointSet(lat=np.array([1.0]), lon=np.array([3.0]))
    assert p1.signature() == p2.signature()
    assert p1.signature() != p3.signature()

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

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from earth2studio.data import ARCO_ERA5, HRRR, GHCNHourly, UFSObsConv, UFSObsSat
from earth2studio.models.da import (
    CorrDiffCosmoEra5SDA,
    HealDA,
    InterpEquirectangular,
    StormCastSDA,
)
from earth2studio.utils.time import normalize_time_tolerance


@pytest.fixture
def fake_stations(monkeypatch):
    """Avoid network access when looking up GHCN stations."""
    calls = []

    def get_stations_bbox(bbox):
        calls.append(bbox)
        return ["USW00000001"]

    monkeypatch.setattr(
        GHCNHourly, "get_stations_bbox", staticmethod(get_stations_bbox)
    )
    return calls


def test_default_sources_stormcast_sda(fake_stations):
    # default_sources only reads the domain and tolerance, skip heavy construction
    model = object.__new__(StormCastSDA)
    model.lat = np.array([[30.0, 40.0], [31.0, 41.0]])
    model.lon = np.array([[250.0, 260.0], [251.0, 261.0]])
    model._tolerance = normalize_time_tolerance(timedelta(minutes=15))
    hrrr, ghcn = model.default_sources()
    assert isinstance(hrrr, HRRR)
    assert isinstance(ghcn, GHCNHourly)
    assert ghcn.stations == ["USW00000001"]
    assert fake_stations == [(30.0, -110.0, 41.0, -99.0)]


def test_default_sources_corrdiff_cosmo_era5_sda(fake_stations):
    model = object.__new__(CorrDiffCosmoEra5SDA)
    model.model = SimpleNamespace(
        lat_output_numpy=np.array([[20.0, 30.0]]),
        lon_output_numpy=np.array([[100.0, 110.0]]),
    )
    model._tolerance = normalize_time_tolerance(timedelta(minutes=30))
    era5, ghcn = model.default_sources()
    assert isinstance(era5, ARCO_ERA5)
    assert isinstance(ghcn, GHCNHourly)
    assert ghcn.stations == ["USW00000001"]
    assert fake_stations == [(20.0, 100.0, 30.0, 110.0)]


def test_default_sources_healda():
    model = object.__new__(HealDA)
    conv, sat = model.default_sources()
    assert isinstance(conv, UFSObsConv)
    assert isinstance(sat, UFSObsSat)


def test_default_sources_interp():
    model = object.__new__(InterpEquirectangular)
    assert model.default_sources() is None

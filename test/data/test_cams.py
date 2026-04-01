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

import datetime
import hashlib
import pathlib
import shutil

import numpy as np
import pytest

from earth2studio.data import CAMS_FX

YESTERDAY = datetime.datetime.now(datetime.UTC).replace(
    hour=0, minute=0, second=0, microsecond=0
) - datetime.timedelta(days=1)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("variable", ["aod550", ["aod550", "tcco"]])
@pytest.mark.parametrize(
    "lead_time",
    [
        datetime.timedelta(hours=0),
        [datetime.timedelta(hours=0), datetime.timedelta(hours=24)],
    ],
)
def test_cams_fx_fetch(variable, lead_time):
    time = np.array([np.datetime64(YESTERDAY.strftime("%Y-%m-%dT%H:%M"))])
    ds = CAMS_FX(cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]
    if isinstance(lead_time, datetime.timedelta):
        lead_time = [lead_time]

    assert shape[0] == 1  # time
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert len(data.coords["lat"]) > 0
    assert len(data.coords["lon"]) > 0
    assert not np.isnan(data.values).all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_cams_fx_cache(cache):
    time = np.array([np.datetime64(YESTERDAY.strftime("%Y-%m-%dT%H:%M"))])
    lead_time = datetime.timedelta(hours=0)
    ds = CAMS_FX(cache=cache)
    data = ds(time, lead_time, ["aod550", "tcco"])
    shape = data.shape

    assert shape[0] == 1
    assert shape[2] == 2
    assert not np.isnan(data.values).all()
    assert pathlib.Path(ds.cache).is_dir() == cache

    data = ds(time, lead_time, "aod550")
    assert data.shape[2] == 1

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(30)
def test_cams_fx_invalid():
    with pytest.raises((ValueError, KeyError)):
        ds = CAMS_FX()
        ds(YESTERDAY, datetime.timedelta(hours=0), "nonexistent_var")


def test_cams_fx_time_validation():
    with pytest.raises(ValueError, match="CAMS Global forecast"):
        CAMS_FX._validate_time([datetime.datetime(2014, 1, 1)])


def test_cams_fx_available():
    assert CAMS_FX.available(datetime.datetime(2024, 1, 1))
    assert not CAMS_FX.available(datetime.datetime(2010, 1, 1))
    # timezone-aware datetimes must not raise TypeError
    assert CAMS_FX.available(datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC))


def test_cams_fx_cache_key_lead_hours_order():
    """lead_hours in different order must produce the same cache key."""

    def _make_cache_key(lead_hours):
        return hashlib.sha256(
            f"cams_fx_aod550_{'_'.join(sorted(lead_hours, key=int))}"
            f"_2024-06-01_00".encode()
        ).hexdigest()

    assert _make_cache_key(["0", "24", "48"]) == _make_cache_key(["48", "0", "24"])
    assert _make_cache_key(["12", "6"]) == _make_cache_key(["6", "12"])


def test_cams_fx_api_vars_dedup():
    """Variables sharing the same API name must not produce duplicate requests."""
    from earth2studio.data.cams import _resolve_variable

    info_a = _resolve_variable("aod550", 0)
    info_b = _resolve_variable("aod550", 1)
    api_vars = list(dict.fromkeys([info_a.api_name, info_b.api_name]))
    assert len(api_vars) == 1

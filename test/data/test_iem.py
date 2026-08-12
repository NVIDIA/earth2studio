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

import asyncio
import json
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import IEM_ASOS
from earth2studio.lexicon import IEM_ASOSLexicon

_TEST_TIME = datetime(2026, 7, 20, 0, 54)
_PARSED_CSV = b"""station,valid,lon,lat,elevation,tmpf,dwpf,relh,drct,sknt,p01i,mslp,gust,skyc1,skyc2,skyc3,skyc4
DSM,2026-07-20 00:54,-93.6531,41.5339,294.0,83.0,65.0,54.71,110.0,8.0,0.0,1013.1,12.0,FEW,SCT,,
"""

_STATION_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": "DFW",
            "properties": {
                "sid": "DFW",
                "elevation": 182.0,
                "sname": "Dallas/Fort Worth",
                "network": "TX_ASOS",
                "country": "US",
                "online": True,
            },
            "geometry": {"type": "Point", "coordinates": [-97.038, 32.8968]},
        },
        {
            "type": "Feature",
            "id": "JFK",
            "properties": {
                "sid": "JFK",
                "elevation": 7.0,
                "sname": "New York/JF Kennedy",
                "network": "NY_ASOS",
                "country": "US",
                "online": True,
            },
            "geometry": {"type": "Point", "coordinates": [-73.7622, 40.6386]},
        },
        {
            "type": "Feature",
            "id": "SEA",
            "properties": {"sid": "SEA"},
            "geometry": {"type": "Point", "coordinates": ["bad", 47.4447]},
        },
        {
            "type": "Feature",
            "id": "NONE",
            "properties": {"sid": "NONE"},
            "geometry": None,
        },
    ],
}


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_iem_asos_fetch():
    source = IEM_ASOS(
        stations="DSM",
        time_tolerance=timedelta(minutes=2),
        cache=False,
        verbose=False,
    )
    frame = source(_TEST_TIME, ["t2m", "d2m", "ws10m", "msl"])

    assert list(frame.columns) == source.SCHEMA.names
    assert set(frame["variable"]) == {"t2m", "d2m", "ws10m", "msl"}
    assert set(frame["station"]) == {"DSM"}
    assert frame["observation"].notna().all()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize("cache", [True, False])
def test_iem_asos_cache(cache):
    source = IEM_ASOS(
        stations="DSM",
        time_tolerance=timedelta(minutes=2),
        cache=cache,
        verbose=False,
    )
    first = source(_TEST_TIME, "t2m")
    second = source(_TEST_TIME, "t2m")

    assert not first.empty
    assert first.equals(second)
    assert pd.io.common.file_exists(source.cache) is cache


@pytest.mark.timeout(15)
def test_iem_asos_call_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    async def fake_fetch_array(self, task):
        return _PARSED_CSV

    monkeypatch.setattr(IEM_ASOS, "fetch_array", fake_fetch_array)
    source = IEM_ASOS(
        stations="DSM",
        time_tolerance=np.timedelta64(0, "m"),
        cache=True,
        verbose=False,
    )
    variables = list(IEM_ASOSLexicon.VOCAB)
    frame = source(_TEST_TIME, variables)

    assert list(frame.columns) == source.SCHEMA.names
    assert set(frame["variable"]) == set(variables)
    assert "metar" not in frame.columns
    assert frame["lat"].dtype == np.float32
    assert frame["lon"].dtype == np.float32
    assert frame["elev"].dtype == np.float32
    assert frame["observation"].dtype == np.float32
    assert frame["lon"].iat[0] == pytest.approx(266.3469, abs=1.0e-4)
    assert frame.attrs["source"] == source.SOURCE_ID

    tasks = source._create_tasks([_TEST_TIME], variables)
    query = parse_qs(urlparse(tasks[0].remote_url).query)
    assert "metar" not in query.get("data", [])
    assert "all" not in query.get("data", [])
    assert query["station"] == ["DSM"]
    assert set(query["data"]) == {
        "tmpf",
        "dwpf",
        "relh",
        "sknt",
        "drct",
        "gust",
        "p01i",
        "mslp",
        "skyc1",
        "skyc2",
        "skyc3",
        "skyc4",
    }


def test_iem_asos_exceptions():
    source = IEM_ASOS(stations="DSM", cache=False, verbose=False)

    with pytest.raises(KeyError, match="not found in IEM_ASOSLexicon"):
        source._create_tasks([_TEST_TIME], ["invalid"])
    with pytest.raises(ValueError, match="earlier than"):
        source._validate_time([datetime(1899, 12, 31, 23, 59)])
    with pytest.raises(ValueError, match="Invalid IEM report types"):
        IEM_ASOS(report_types=(2,))
    with pytest.raises(ValueError, match="at least one"):
        IEM_ASOS(report_types=())
    with pytest.raises(KeyError, match="not found"):
        source.resolve_fields(["observation", "invalid"])
    with pytest.raises(TypeError, match="expected"):
        source.resolve_fields(pa.schema([pa.field("time", pa.string())]))


def test_iem_asos_available():
    assert IEM_ASOS.available(_TEST_TIME)
    assert IEM_ASOS.available(np.datetime64("2026-07-20T00:54"))
    assert not IEM_ASOS.available(datetime(1899, 12, 31, 23, 59))


@pytest.mark.asyncio
async def test_iem_asos_request_rate_limit_allows_concurrency(tmp_path, monkeypatch):
    """Request starts are throttled without serializing response downloads."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    request_starts: list[float] = []
    counters = {"active": 0, "max_active": 0}

    class FakeResponse:
        status = 200

        def raise_for_status(self) -> None:
            pass

        async def read(self) -> bytes:
            return _PARSED_CSV

    class FakeGetCM:
        async def __aenter__(self):
            request_starts.append(asyncio.get_running_loop().time())
            counters["active"] += 1
            counters["max_active"] = max(counters["max_active"], counters["active"])
            await asyncio.sleep(0.03)
            counters["active"] -= 1
            return FakeResponse()

        async def __aexit__(self, *args) -> None:
            pass

    class FakeSession:
        def get(self, url):
            return FakeGetCM()

    source = IEM_ASOS(cache=True, verbose=False)
    source.REQUEST_INTERVAL_SECONDS = 0.01
    source._session = FakeSession()
    tasks = source._create_tasks([_TEST_TIME, _TEST_TIME + timedelta(days=1)], ["t2m"])

    await asyncio.gather(*(source.fetch_array(task) for task in tasks))

    assert len(request_starts) == 2
    assert request_starts[1] - request_starts[0] >= source.REQUEST_INTERVAL_SECONDS
    assert counters["max_active"] == 2


def test_iem_asos_station_bbox(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    downloads = []

    class FakeResponse:
        status = 200

        def raise_for_status(self) -> None:
            pass

        async def read(self) -> bytes:
            return json.dumps(_STATION_GEOJSON).encode()

    class FakeGetCM:
        def __init__(self, url: str) -> None:
            downloads.append(url)

        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, *args) -> None:
            pass

    class FakeClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            pass

        def get(self, url: str):
            return FakeGetCM(url)

    monkeypatch.setattr(
        "earth2studio.data.iem.aiohttp.ClientSession", FakeClientSession
    )

    metadata = IEM_ASOS.get_station_metadata()
    assert list(metadata.columns) == [
        "ID",
        "LAT",
        "LON",
        "ELEV",
        "NAME",
        "NETWORK",
        "COUNTRY",
        "ONLINE",
    ]
    assert metadata["ID"].tolist() == ["DFW", "JFK", "SEA"]
    assert IEM_ASOS.get_stations_bbox((32.0, -98.0, 34.0, -96.0)) == ["DFW"]
    assert IEM_ASOS.get_stations_bbox((32.0, 262.0, 34.0, 264.0)) == ["DFW"]
    assert IEM_ASOS.get_stations_bbox((32.8968, -97.038, 32.8968, -97.038)) == ["DFW"]
    assert downloads == [
        "https://mesonet.agron.iastate.edu/geojson/network.py?network=AZOS"
    ]

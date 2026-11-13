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
import pathlib
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import AIFS_ENS_FX, AIFS_FX, IFS, IFS_ENS, IFS_ENS_FX, IFS_FX
from earth2studio.data import ecmwf as ecmwf_module


@pytest.mark.parametrize(
    "source_class, model",
    [
        (IFS, "ifs"),
        (IFS_FX, "ifs"),
        (IFS_ENS, "ifs"),
        (IFS_ENS_FX, "ifs"),
        (AIFS_FX, "aifs-single"),
        (AIFS_ENS_FX, "aifs-ens"),
    ],
)
@pytest.mark.parametrize("source", ["aws", "google"])
def test_opendata_client_kwargs(source_class, model, source, monkeypatch):
    client_factory = Mock(return_value=Mock())
    monkeypatch.setattr(ecmwf_module.opendata, "Client", client_factory)

    source_class(
        source=source,
        client_kwargs={"preserve_request_order": True},
    )

    client_factory.assert_called_once_with(
        source=source,
        model=model,
        preserve_request_order=True,
    )


def test_accumulation_alias_tasks_use_previous_lead_time():
    ds = object.__new__(AIFS_FX)
    time = [datetime(2025, 12, 31, 0)]
    lead_time = [timedelta(hours=12)]

    tasks = asyncio.run(ds._create_tasks(time, lead_time, ["sf06", "ro06", "t2m"]))

    assert tasks[0].variable == "sf"
    assert tasks[0].lead_time == timedelta(hours=12)
    assert tasks[0].previous_lead_time == timedelta(hours=6)
    assert tasks[1].variable == "rowe"
    assert tasks[1].previous_lead_time == timedelta(hours=6)
    assert tasks[2].variable == "2t"
    assert tasks[2].previous_lead_time is None


def test_accumulation_alias_tasks_reject_short_lead_time():
    ds = object.__new__(IFS_FX)

    with pytest.raises(ValueError, match="too short"):
        asyncio.run(
            ds._create_tasks(
                [datetime(2025, 12, 31, 0)],
                [timedelta(hours=0)],
                ["sf06"],
            )
        )


def _ensemble_source(source_class, monkeypatch, member=1):
    """Build an ECMWF ensemble source whose member fetches return member ids."""
    monkeypatch.setattr(ecmwf_module.opendata, "Client", Mock(return_value=Mock()))
    ds = source_class(member=member, cache=False, verbose=False)
    calls = []

    async def fetch_members(time, lead_time, variable, fc_type, members):
        calls.append((fc_type, list(members)))
        shape = (len(time), len(lead_time), len(variable))
        coords = {"time": time, "lead_time": lead_time, "variable": variable}
        if fc_type != "pf":
            (value,) = members
            return xr.DataArray(
                np.full(shape, float(value)),
                dims=("time", "lead_time", "variable"),
                coords=coords,
            )
        values = np.broadcast_to(np.array(members, dtype=float), (*shape, len(members)))
        return xr.DataArray(
            values,
            dims=("time", "lead_time", "variable", "ensemble"),
            coords={**coords, "ensemble": members},
        )

    ds._fetch_members = fetch_members
    return ds, calls


def test_ifs_ens_fetches_requested_members_by_forecast_type(monkeypatch):
    ds, calls = _ensemble_source(IFS_ENS, monkeypatch)
    da = ds(datetime(2025, 1, 1), ["t2m"], member=[2, 0, 1])

    assert calls == [("cf", [0]), ("pf", [2, 1])]
    assert da.dims[0] == "ensemble"
    np.testing.assert_array_equal(da["ensemble"], [2, 0, 1])
    np.testing.assert_array_equal(da.isel(time=0, variable=0), [2, 0, 1])


def test_ifs_ens_without_member_returns_its_member(monkeypatch):
    ds, calls = _ensemble_source(IFS_ENS, monkeypatch, member=3)
    da = ds(datetime(2025, 1, 1), ["t2m"])

    assert calls == [("pf", [3])]
    assert "ensemble" not in da.dims
    np.testing.assert_array_equal(da.values, [[3.0]])


@pytest.mark.parametrize("source_class", [IFS_ENS_FX, AIFS_ENS_FX])
def test_ensemble_forecasts_fetch_requested_members(source_class, monkeypatch):
    ds, calls = _ensemble_source(source_class, monkeypatch)
    da = asyncio.run(
        ds.fetch([datetime(2025, 1, 1)], [timedelta(hours=6)], ["t2m"], member=4)
    )

    assert calls == [("pf", [4])]
    assert da.dims[0] == "ensemble"
    np.testing.assert_array_equal(da["ensemble"], [4])


def test_ensemble_rejects_negative_members(monkeypatch):
    ds, _ = _ensemble_source(IFS_ENS, monkeypatch)
    with pytest.raises(ValueError, match="ensemble members"):
        ds(datetime(2025, 1, 1), ["t2m"], member=[-1])


def test_grib_request_uses_requested_members(monkeypatch, tmp_path):
    monkeypatch.setattr(ecmwf_module, "datasource_cache_root", lambda: str(tmp_path))
    ds, _ = _ensemble_source(IFS_ENS, monkeypatch)
    pathlib.Path(ds.cache).mkdir(parents=True)

    def retrieve(request):
        pathlib.Path(request["target"]).touch()

    ds.client.retrieve = Mock(side_effect=retrieve)
    for fc_type, members in (("pf", [2, 5]), ("cf", [0])):
        asyncio.run(
            ds._download_ifs_grib_cached(
                datetime(2025, 1, 1),
                timedelta(hours=0),
                "2t",
                "sfc",
                "",
                fc_type=fc_type,
                members=members,
            )
        )
    perturbed, control = (call.args[0] for call in ds.client.retrieve.call_args_list)
    assert (perturbed["type"], perturbed["number"]) == ("pf", [2, 5])
    assert control["type"] == "cf" and "number" not in control
    assert perturbed["target"] != control["target"]


def now6h():
    """Get closest 6 hour timestamp"""
    nt = datetime.now()
    delta_hr = nt.hour % 6
    return datetime(nt.year, nt.month, nt.day, nt.hour - delta_hr)


@pytest.mark.live(timeout=30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
    ],
)
@pytest.mark.parametrize("variable", [["tcwv", "sp"]])
def test_ifs_fetch(time, variable):
    ds = IFS()
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))

    ds = IFS_FX()
    data_fx = ds(time, timedelta(hours=0), variable)
    shape = data_fx.shape

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert np.allclose(data.values, data_fx.values[:, 0])

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.live(timeout=30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "sp"]])
@pytest.mark.parametrize("member", [2])
def test_ifs_ens_fetch(time, variable, member):
    ds = IFS_ENS(cache=False, member=member)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))

    ds = IFS_ENS_FX(cache=False, member=member)
    data_fx = ds(time, timedelta(hours=0), variable)
    shape = data_fx.shape

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()
    assert np.allclose(data.values, data_fx.values[:, 0])


@pytest.mark.live(timeout=30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(hours=12),
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        [
            timedelta(hours=3),
            timedelta(hours=6),
        ],
    ],
)
@pytest.mark.parametrize("variable", [["tcwv", "sp"]])
def test_ifs_ens_fx_fetch(time, lead_time, variable):
    ds = IFS_ENS_FX(cache=False)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.live(timeout=30)
@pytest.mark.parametrize(
    "lead_time",
    [
        [
            timedelta(hours=6),
            timedelta(hours=12),
        ],
    ],
)
@pytest.mark.parametrize("variable", [["t2m", "tcc", "tp"]])
@pytest.mark.parametrize("member", [1])
def test_aifs_ens_fx_fetch(lead_time, variable, member):
    time = now6h() - timedelta(hours=12)
    ds = AIFS_ENS_FX(cache=False, member=member)
    data = ds(time, lead_time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(lead_time, timedelta):
        lead_time = [lead_time]

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(lead_time)
    assert shape[2] == len(variable)
    assert shape[3] == 721
    assert shape[4] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.live(timeout=30)
@pytest.mark.parametrize(
    "time",
    [
        np.array([now6h() - timedelta(days=2)], dtype="datetime64"),
    ],
)
@pytest.mark.parametrize("variable", [["u10m", "tcwv"]])
@pytest.mark.parametrize("cache", [True])
def test_ifs_cache(time, variable, cache):
    ds = IFS(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cache or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(30)
@pytest.mark.parametrize("data_source", [IFS, IFS_ENS])
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("variable", ["msl"])
def test_ifs_time_available(data_source, time, variable):
    with pytest.raises(ValueError):
        ds = IFS(source="ecmwf")
        ds(time, ["msl"])

        ds = IFS_FX(source="ecmwf")
        ds(time, [timedelta(hours=0)], ["msl"])


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "lead_time",
    [
        timedelta(hours=1),
        timedelta(hours=147),
        timedelta(hours=366),
    ],
)
def test_ifs_leadtime_available(lead_time):
    with pytest.raises(ValueError):
        time = now6h() - timedelta(days=2)
        ds = IFS_FX(source="ecmwf")
        ds(time, lead_time, ["msl"])

        ds = IFS_ENS_FX(source="ecmwf")
        ds(time, lead_time, ["msl"])


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        now6h() - timedelta(days=1, minutes=1),
        datetime(year=1993, month=4, day=5),
    ],
)
@pytest.mark.parametrize("lead_time", [timedelta(hours=0)])
@pytest.mark.parametrize("variable", ["msl"])
def test_aifs_time_available(time, lead_time, variable):
    with pytest.raises(ValueError):
        ds = AIFS_FX(source="ecmwf")
        ds(time, lead_time, variable)

    with pytest.raises(ValueError):
        ds = AIFS_ENS_FX(source="ecmwf")
        ds(time, lead_time, variable)

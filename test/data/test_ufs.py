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

import pathlib
import shutil
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data import UFSObsConv, UFSObsSat


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
    ],
)
@pytest.mark.parametrize("variable", [["t"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ufsobsconv_cache(time, variable, cache):
    ds = UFSObsConv(
        time_tolerance=timedelta(hours=1),
        cache=cache,
        verbose=False,
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert pathlib.Path(ds.cache).is_dir() == cache

    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_ufsobsconv_schema_fields():
    time = datetime(year=2024, month=1, day=1, hour=0)
    tol = timedelta(hours=1)

    ds = UFSObsConv(time_tolerance=tol, cache=False, verbose=False)

    df_full = ds(time, ["t"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    subset_fields = ["time", "lat", "lon", "observation", "variable"]
    df_subset = ds(time, ["t"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_ufsobsconv_exceptions():
    ds = UFSObsConv(
        time_tolerance=timedelta(hours=1),
        cache=False,
        verbose=False,
    )

    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["invalid_variable"])

    with pytest.raises(KeyError):
        ds(
            datetime(2024, 1, 1),
            ["t"],
            fields=["observation", "variable", "invalid_field"],
        )

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["t2m"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        ds(datetime(2024, 1, 1), ["t2m"], fields=wrong_type_schema)


def test_ufsobsconv_tolerance_conversion():
    ds_timedelta = UFSObsConv(
        time_tolerance=timedelta(hours=1), cache=False, verbose=False
    )
    assert ds_timedelta._tolerance_lower == timedelta(hours=-1)
    assert ds_timedelta._tolerance_upper == timedelta(hours=1)

    ds_numpy = UFSObsConv(
        time_tolerance=np.timedelta64(1, "h"), cache=False, verbose=False
    )
    assert ds_numpy._tolerance_lower == timedelta(hours=-1)
    assert ds_numpy._tolerance_upper == timedelta(hours=1)

    # Asymmetric tolerance tuple
    ds_asym = UFSObsConv(
        time_tolerance=(np.timedelta64(-3, "h"), np.timedelta64(1, "h")),
        cache=False,
        verbose=False,
    )
    assert ds_asym._tolerance_lower == timedelta(hours=-3)
    assert ds_asym._tolerance_upper == timedelta(hours=1)


@pytest.mark.parametrize("cls", [UFSObsConv, UFSObsSat])
def test_ufsobs_missing_file_warns_not_raises(cls, caplog):
    """A missing diag file warns and returns rather than aborting the fetch.

    GSI archives have gaps (e.g. an absent GNSS-RO ``gps`` cycle or a
    decommissioned satellite platform); both obs sources must tolerate
    them so a bulk request spanning many cycles is not derailed by one
    missing object.
    """
    ds = cls(cache=False, verbose=False)
    key = "2024/03/2024033018/gsi/diag_conv_gps_ges.2024033018_control.nc4"

    # Must not raise (previously the conventional source raised here).
    ds._handle_missing_file(key)
    assert "not found" in caplog.text


@pytest.mark.parametrize("cls", [UFSObsConv, UFSObsSat])
def test_ufsobs_all_files_missing_returns_empty(cls):
    """When every file is skipped, an empty schema-shaped frame is returned.

    Guards against ``pd.concat([])`` raising "No objects to concatenate"
    when a whole request's worth of diag files is absent — consumers get
    a well-formed empty DataFrame to apply their own handling to.
    """
    ds = cls(cache=False, verbose=False)
    schema = ds.resolve_fields(None)

    # No async tasks => no frames compiled => empty result.
    df = ds._compile_dataframe([], ["t"], schema)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert list(df.columns) == schema.names


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
        [datetime(year=2024, month=1, day=1, hour=6)],
    ],
)
@pytest.mark.parametrize(
    "variable, satellites, tol",
    [
        (["atms"], ["npp"], timedelta(hours=1)),
        (["mhs"], ["metop-a", "metop-b"], timedelta(hours=2)),
        (["airs"], ["aqua"], timedelta(hours=1)),
    ],
)
def test_ufsobssat_fetch(time, variable, satellites, tol):
    ds = UFSObsSat(
        time_tolerance=tol, satellites=satellites, cache=False, verbose=False
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert set(df["variable"].unique()).issubset(set(variable))
    assert "observation" in df.columns
    assert "satellite" in df.columns

    if not isinstance(time, (list, np.ndarray)):
        time = [time]

    if not df.empty:
        time_union = pd.DataFrame({"time": np.zeros(df.shape[0])}).astype("bool")
        for t in time:
            df_times = df["time"]
            min_time = t - tol
            max_time = t + tol
            time_union["time"] = time_union["time"] | (
                df_times.ge(min_time) & df_times.le(max_time)
            )
        assert time_union["time"].all()
        assert set(df["satellite"].unique()).issubset(set(satellites))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=1, day=1, hour=0),
    ],
)
@pytest.mark.parametrize("variable", [["atms"]])
@pytest.mark.parametrize("cache", [True, False])
def test_ufsobssat_cache(time, variable, cache):
    ds = UFSObsSat(
        time_tolerance=timedelta(hours=1),
        satellites=["npp"],
        cache=cache,
        verbose=False,
    )
    df = ds(time, variable)

    assert list(df.columns) == ds.SCHEMA.names
    assert pathlib.Path(ds.cache).is_dir() == cache

    df = ds(time, variable)
    assert list(df.columns) == ds.SCHEMA.names

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_ufsobssat_schema_fields():
    time = datetime(year=2024, month=1, day=1, hour=0)
    tol = timedelta(hours=1)

    ds = UFSObsSat(time_tolerance=tol, satellites=["npp"], cache=False, verbose=False)

    df_full = ds(time, ["atms"], fields=None)
    assert list(df_full.columns) == ds.SCHEMA.names

    subset_fields = ["time", "lat", "lon", "satellite", "observation", "variable"]
    df_subset = ds(time, ["atms"], fields=subset_fields)
    assert list(df_subset.columns) == subset_fields


def test_ufsobssat_exceptions():
    ds = UFSObsSat(
        time_tolerance=timedelta(hours=1),
        satellites=["npp"],
        cache=False,
        verbose=False,
    )

    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["invalid_variable"])

    with pytest.raises(KeyError):
        ds(
            datetime(2024, 1, 1),
            ["atms"],
            fields=["observation", "variable", "invalid_field"],
        )

    invalid_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("nonexistent", pa.float32()),
        ]
    )
    with pytest.raises(KeyError):
        ds(datetime(2024, 1, 1), ["atms"], fields=invalid_schema)

    wrong_type_schema = pa.schema(
        [
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
            pa.field("time", pa.string()),
        ]
    )
    with pytest.raises(TypeError):
        ds(datetime(2024, 1, 1), ["atms"], fields=wrong_type_schema)

    # Test satellites
    with pytest.raises(ValueError, match="Invalid satellite"):
        UFSObsSat(satellites=["invalid_sat"])

    with pytest.raises(ValueError, match="Invalid satellite"):
        UFSObsSat(satellites=["npp", "invalid_sat"])

    ds = UFSObsSat(cache=False, verbose=False)
    assert set(ds.satellites) == ds.VALID_SATELLITES

    ds = UFSObsSat(satellites=["npp", "n20"], cache=False, verbose=False)
    assert ds.satellites == ["npp", "n20"]


def test_gsi_cache_path():
    ds = UFSObsConv(cache=True, verbose=False)
    path1 = ds.cache_path("s3://bucket/file.nc4")
    path2 = ds.cache_path("s3://bucket/file.nc4", byte_offset=100)
    path3 = ds.cache_path("s3://bucket/file.nc4", byte_offset=100, byte_length=200)

    assert path1 != path2
    assert path2 != path3
    assert all(p.startswith(ds.cache) for p in [path1, path2, path3])


def test_hours_since_to_datetime_bit_exact():
    """The numpy fast path must be bit-exact with the pandas conversion it
    replaced (``pd.to_timedelta(values, unit="h") + origin``), including
    negative offsets, float32 noise near hour boundaries and NaN -> NaT."""
    import earth2studio.data.ufs as ufs_module

    rng = np.random.default_rng(0)
    vals = np.concatenate(
        [
            rng.uniform(-6, 6, 20000),
            np.array([0.0, -3.0, 2.9999999, -0.0000001, 5.5, np.nan]),
        ]
    ).astype(np.float32)
    origin = datetime(2024, 1, 1, 6)

    expected = (pd.to_timedelta(vals, unit="h") + origin).values
    got = ufs_module._hours_since_to_datetime(vals, origin)

    assert got.dtype == expected.dtype
    np.testing.assert_array_equal(got.view(np.int64), expected.view(np.int64))


def test_compile_dataframe_groups_tasks_by_file(monkeypatch, tmp_path):
    """Tasks sharing a file and window (e.g. u and v in ``diag_conv_uv``)
    must be decoded in one group, and the result must preserve task order."""
    import earth2studio.data.ufs as ufs_module

    ds = UFSObsConv(cache=False, verbose=False, decode_workers=1)
    schema = ds.resolve_fields(["time", "lat", "lon", "observation", "variable"])

    fake_file = tmp_path / "diag_conv_uv.nc4"
    fake_file.touch()
    monkeypatch.setattr(ds, "cache_path", lambda key: str(fake_file))

    t0 = datetime(2024, 1, 1)
    tmin, tmax = t0 - timedelta(hours=1), t0 + timedelta(hours=1)

    def make_task(gsi_obs_name, e2s_obs_name):
        return ufs_module._GSIAsyncTask(
            datetime_file=t0,
            datetime_min=tmin,
            datetime_max=tmax,
            gsi_obs_key="key",
            gsi_modifier=lambda df: df,
            gsi_obs_name=gsi_obs_name,
            e2s_obs_name=e2s_obs_name,
        )

    tasks = [make_task("u_Observation", "u"), make_task("v_Observation", "v")]

    group_sizes = []

    def fake_decode(cls, local_path, group, column_map, channel_fields):
        group_sizes.append(len(group))
        return [
            pd.DataFrame(
                {
                    "time": [t0],
                    "lat": [0.0],
                    "lon": [0.0],
                    "observation": [1.0],
                    "variable": [task.e2s_obs_name],
                }
            )
            for task in group
        ]

    monkeypatch.setattr(ufs_module, "_decode_gsi_group", fake_decode)
    df = ds._compile_dataframe(tasks, ["u", "v"], schema)

    assert group_sizes == [2]  # one decode call served both tasks
    assert list(df["variable"]) == ["u", "v"]
    assert df.attrs["source"] == ds.SOURCE_ID


def test_compile_dataframe_broken_pool_falls_back_to_serial(
    monkeypatch, tmp_path, caplog
):
    """If decode workers die during bootstrap (e.g. an unguarded __main__
    with the spawn start method), the fetch must degrade to serial decode
    with a warning instead of raising."""
    from concurrent.futures.process import BrokenProcessPool

    import earth2studio.data.ufs as ufs_module

    ds = UFSObsConv(cache=False, verbose=False, decode_workers=4)
    schema = ds.resolve_fields(["time", "lat", "lon", "observation", "variable"])

    fake_file = tmp_path / "diag_conv_t.nc4"
    fake_file.touch()
    monkeypatch.setattr(ds, "cache_path", lambda key: str(fake_file))

    t0 = datetime(2024, 1, 1)
    task = ufs_module._GSIAsyncTask(
        datetime_file=t0,
        datetime_min=t0 - timedelta(hours=1),
        datetime_max=t0 + timedelta(hours=1),
        gsi_obs_key="key",
        gsi_modifier=lambda df: df,
        gsi_obs_name="t_Observation",
        e2s_obs_name="t",
    )
    # Two tasks in distinct groups so the parallel path is taken
    task2 = ufs_module._GSIAsyncTask(
        datetime_file=t0,
        datetime_min=t0 - timedelta(hours=2),
        datetime_max=t0 + timedelta(hours=2),
        gsi_obs_key="key2",
        gsi_modifier=lambda df: df,
        gsi_obs_name="t_Observation",
        e2s_obs_name="t",
    )

    class BrokenPool:
        def submit(self, *args, **kwargs):
            raise BrokenProcessPool("worker died during bootstrap")

        def shutdown(self, wait=True):
            pass

    monkeypatch.setattr(ds, "_get_decode_pool", lambda: BrokenPool())

    def fake_decode(cls, local_path, group, column_map, channel_fields):
        return [
            pd.DataFrame(
                {
                    "time": [t0],
                    "lat": [0.0],
                    "lon": [0.0],
                    "observation": [1.0],
                    "variable": [t.e2s_obs_name],
                }
            )
            for t in group
        ]

    monkeypatch.setattr(ufs_module, "_decode_gsi_group", fake_decode)

    df = ds._compile_dataframe([task, task2], ["t"], schema)
    assert len(df) == 2
    assert ds._decode_pool is None
    assert "Falling back to serial decode" in caplog.text

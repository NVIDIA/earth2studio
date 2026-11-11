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

"""predownload.py caches the runner's requests into stores main.py can read."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import open_dict
from predownload import (
    download_gridded_store,
    plan_frame_stores,
    plan_gridded_stores,
    plan_verification_store,
    run,
)
from src.campaign import build_items
from src.data import (
    FrameRequest,
    PredownloadedForecastSource,
    PredownloadedFrameSource,
    PredownloadedSource,
    frame_filename,
    frame_store_path,
    open_predownloaded,
    resolve_source,
)
from src.output import sentinel_path, store_path
from src.runners import build_runner

from earth2studio.data import fetch_data
from earth2studio.grids import LatLonGrid, resolve_regridder, same_geometry
from earth2studio.run import WorkItem
from earth2studio.run.runner import DataRequest
from test.conftest import SMALL_LAT, SMALL_LON, VARIABLES, SeededRandom

TIMES = ["2024-01-01 00:00:00", "2024-01-01 06:00:00"]


def _open(cfg, name: str) -> xr.Dataset:
    return xr.open_zarr(store_path(cfg, name), consolidated=False)


@pytest.fixture()
def cfg(base_cfg):
    with open_dict(base_cfg):
        base_cfg.start_times = list(TIMES)
    return base_cfg


def test_plan_gridded_stores_groups_by_role(cfg) -> None:
    runner = build_runner(cfg, prefer_predownloaded=False)
    items = build_items(cfg, runner)
    stores = plan_gridded_stores(cfg, runner, items)
    assert [s.role for s in stores] == ["data"]
    assert stores[0].variables == VARIABLES
    np.testing.assert_array_equal(
        stores[0].times, np.array(TIMES, dtype="datetime64[ns]")
    )
    assert stores[0].source is runner.source_roles["data"]


def test_plan_skips_bring_your_own_initial_conditions(cfg) -> None:
    with open_dict(cfg):
        cfg.ic_source = {
            "_target_": "earth2studio.data.Random",
            "domain_coords": {"lat": SMALL_LAT.tolist(), "lon": SMALL_LON.tolist()},
        }
    runner = build_runner(cfg, prefer_predownloaded=False)
    assert plan_gridded_stores(cfg, runner, build_items(cfg, runner)) == []


def test_plan_verification_store_covers_every_valid_time(cfg) -> None:
    runner = build_runner(cfg, prefer_predownloaded=False)
    items = build_items(cfg, runner)
    assert plan_verification_store(cfg, runner, items) is None
    with open_dict(cfg):
        cfg.predownload.verification.enabled = True
    store = plan_verification_store(cfg, runner, items)
    assert store is not None and store.role == "verification"
    assert store.variables == VARIABLES
    expected = np.unique(
        np.array(TIMES, dtype="datetime64[ns]")[:, None]
        + np.arange(3)[None, :] * np.timedelta64(6, "h")
    )
    np.testing.assert_array_equal(store.times, expected)
    assert same_geometry(store.target_grid, LatLonGrid(SMALL_LAT, SMALL_LON))
    # A bring-your-own verification source needs no cache.
    with open_dict(cfg):
        cfg.verification_source = dict(cfg.data_source)
    assert plan_verification_store(cfg, runner, items) is None


def test_verification_falls_back_to_the_runner_source(cfg) -> None:
    runner = build_runner(cfg, prefer_predownloaded=False)
    items = build_items(cfg, runner)
    with open_dict(cfg):
        cfg.predownload.verification.enabled = True
        cfg.data_source = None
    runner.source_roles = {}
    assert plan_verification_store(cfg, runner, items) is None
    runner.verification_source = runner.source
    store = plan_verification_store(cfg, runner, items)
    assert store is not None and store.source is runner.source


def test_run_writes_data_store_and_sentinel(cfg) -> None:
    run(cfg)
    ds = _open(cfg, "data.zarr")
    assert set(ds.data_vars) == set(VARIABLES)
    assert ds["t2m"].dims == ("time", "lat", "lon")
    assert ds["t2m"].shape == (2, len(SMALL_LAT), len(SMALL_LON))
    np.testing.assert_array_equal(
        ds["time"].values, np.array(TIMES, dtype="datetime64[ns]")
    )
    assert not np.isnan(ds["t2m"].values).any()
    assert sentinel_path(cfg).exists()
    assert not Path(store_path(cfg, "verification.zarr")).exists()


def test_predownloaded_store_feeds_fetch_data(cfg) -> None:
    run(cfg)
    source = PredownloadedSource(store_path(cfg, "data.zarr"))
    assert sorted(source.variables) == sorted(VARIABLES)
    x = fetch_data(
        source, time=np.array(TIMES[:1], dtype="datetime64[ns]"), variable=VARIABLES
    )
    assert x.dims == ("time", "lead_time", "variable", "lat", "lon")
    assert x.shape == (1, 1, 2, len(SMALL_LAT), len(SMALL_LON))
    np.testing.assert_array_equal(x["lat"].values, SMALL_LAT)
    # The inference entry point prefers the cached store.
    runner = build_runner(cfg)
    assert isinstance(runner.source_roles["data"], PredownloadedSource)


def test_verification_store_is_separate_and_regridded(cfg) -> None:
    with open_dict(cfg):
        cfg.predownload.verification.enabled = True
        cfg.output.regrid = {
            "grid": {
                "lat": {"start": 90.0, "stop": -90.0, "num": 2},
                "lon": {"start": 0.0, "stop": 180.0, "num": 2},
            },
            "method": "nearest",
        }
    run(cfg)
    data = _open(cfg, "data.zarr")
    verification = _open(cfg, "verification.zarr")
    assert data["t2m"].shape[-2:] == (len(SMALL_LAT), len(SMALL_LON))
    assert verification["t2m"].shape == (4, 2, 2)
    np.testing.assert_array_equal(verification["lat"].values, [90.0, -90.0])
    assert set(verification.data_vars) == set(VARIABLES)
    source = PredownloadedSource(store_path(cfg, "verification.zarr"))
    assert source.time_step == np.timedelta64(6, "h")


def test_verification_lands_on_a_coarser_forecast_grid_without_regrid(cfg) -> None:
    # The model runs on every other point of the source grid (UCast at 1.5
    # degrees under 0.25 degree reanalysis); verification follows the forecast.
    from earth2studio.grids import list_grids, register_grid

    if "latlon-coarse" not in list_grids():
        register_grid("latlon-coarse", LatLonGrid(SMALL_LAT[::2], SMALL_LON[::2]))
    with open_dict(cfg):
        cfg.predownload.verification.enabled = True
        cfg.model.domain_coords = "latlon-coarse"
    run(cfg)
    verification = _open(cfg, "verification.zarr")
    np.testing.assert_array_equal(verification["lat"].values, SMALL_LAT[::2])
    np.testing.assert_array_equal(verification["lon"].values, SMALL_LON[::2])
    # The initial-condition store keeps the source grid; fetch_data maps it
    # onto the model grid at run time.
    data = _open(cfg, "data.zarr")
    assert data["t2m"].shape[-2:] == (len(SMALL_LAT), len(SMALL_LON))


def test_verification_store_uses_the_configured_regrid_method(cfg) -> None:
    target = LatLonGrid(np.linspace(60.0, -60.0, 2), np.linspace(10.0, 190.0, 2))
    with open_dict(cfg):
        cfg.predownload.verification.enabled = True
        cfg.output.regrid = {
            "grid": {
                "lat": {"start": 60.0, "stop": -60.0, "num": 2},
                "lon": {"start": 10.0, "stop": 190.0, "num": 2},
            },
            "method": "bilinear",
        }
    run(cfg)
    verification = _open(cfg, "verification.zarr")
    time = np.array(TIMES[:1], dtype="datetime64[ns]")
    source = SeededRandom(domain_coords={"lat": SMALL_LAT, "lon": SMALL_LON})
    live = fetch_data(source, time=time, variable=np.array(VARIABLES))
    live = live.isel(time=0, lead_time=0)
    small = LatLonGrid(SMALL_LAT, SMALL_LON)
    bilinear = resolve_regridder(small, target, "bilinear")(live)
    nearest = resolve_regridder(small, target, "nearest")(live)
    stored = verification["t2m"].sel(time=time[0]).values
    np.testing.assert_allclose(stored, bilinear.sel(variable="t2m").values, rtol=1e-5)
    assert not np.allclose(stored, nearest.sel(variable="t2m").values)


class _FakeForecast:
    """Forecast source whose values encode the initial time and the lead."""

    def __call__(self, time, lead_time, variable) -> xr.DataArray:
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        leads = np.atleast_1d(np.asarray(lead_time, dtype="timedelta64[ns]"))
        variables = np.atleast_1d(np.asarray(variable))
        hours = leads.astype("timedelta64[h]").astype(float)
        origin = np.datetime64("2024-01-01", "ns")
        days = (times - origin).astype("timedelta64[h]").astype(float) / 24.0
        shape = (len(times), len(leads), len(variables), len(SMALL_LAT), len(SMALL_LON))
        data = np.zeros(shape, dtype="float32")
        data += days[:, None, None, None, None] * 100.0
        data += hours[None, :, None, None, None]
        return xr.DataArray(
            data,
            dims=("time", "lead_time", "variable", "lat", "lon"),
            coords={
                "time": times,
                "lead_time": leads,
                "variable": variables,
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            },
        )


class _ForecastRunner:
    """Runner stub binding one forecast source under the role ``fx``."""

    step = np.timedelta64(6, "h")
    leads = np.array([0, 6], dtype="timedelta64[h]")

    def __init__(self, source: _FakeForecast) -> None:
        self.source = source
        self.source_roles = {"fx": source}

    def data_requests(self, item: WorkItem):
        return (
            DataRequest(
                self.source, np.array([item.time]), np.array(VARIABLES), self.leads
            ),
        )


def test_forecast_source_is_cached_per_lead_time(cfg) -> None:
    source = _FakeForecast()
    runner = _ForecastRunner(source)
    items = build_items(cfg, runner)
    stores = plan_gridded_stores(cfg, runner, items)
    assert [s.role for s in stores] == ["fx"]
    store = stores[0]
    np.testing.assert_array_equal(store.times, np.array(TIMES, dtype="datetime64[ns]"))
    np.testing.assert_array_equal(
        store.lead_times, _ForecastRunner.leads.astype("timedelta64[ns]")
    )

    download_gridded_store(cfg, store, overwrite=False)
    ds = _open(cfg, "fx.zarr")
    assert ds["t2m"].dims == ("time", "lead_time", "lat", "lon")
    assert ds["t2m"].shape == (2, 2, len(SMALL_LAT), len(SMALL_LON))

    cached = open_predownloaded(store_path(cfg, "fx.zarr"))
    assert isinstance(cached, PredownloadedForecastSource)
    time = np.array(TIMES[1:], dtype="datetime64[ns]")
    live = fetch_data(
        source, time=time, variable=np.array(VARIABLES), lead_time=store.lead_times
    )
    back = fetch_data(
        cached, time=time, variable=np.array(VARIABLES), lead_time=store.lead_times
    )
    assert tuple(back.dims) == tuple(live.dims)
    np.testing.assert_array_equal(back.values, live.values)
    # The cached value at (06:00, lead 6h) is the forecast (0.25 days * 100 + 6h),
    # not the analysis at the 12:00 valid time (0.5 days * 100).
    assert float(back.isel(time=0, lead_time=1, variable=0, lat=0, lon=0)) == 31.0
    assert isinstance(resolve_source(cfg, "fx"), PredownloadedForecastSource)


def test_resume_skips_finished_times_and_overwrite_restarts(cfg, monkeypatch) -> None:
    import predownload as module

    run(cfg)
    first = _open(cfg, "data.zarr")["t2m"].values.copy()
    calls: list[np.datetime64] = []
    original = module._fetch_field

    def counting(store, time):
        calls.append(time)
        return original(store, time)

    monkeypatch.setattr(module, "_fetch_field", counting)
    run(cfg)
    assert calls == [], "resume must not fetch finished times"
    np.testing.assert_array_equal(_open(cfg, "data.zarr")["t2m"].values, first)

    with open_dict(cfg):
        cfg.predownload.overwrite = True
    run(cfg)
    # The write loop reuses the probe fetch that built the template.
    assert len(calls) == len(TIMES)
    assert not np.isnan(_open(cfg, "data.zarr")["t2m"].values).any()


def test_changed_campaign_requires_overwrite(cfg) -> None:
    run(cfg)
    with open_dict(cfg):
        cfg.start_times = TIMES + ["2024-01-02 00:00:00"]
    with pytest.raises(ValueError, match="predownload.overwrite=true"):
        run(cfg)


class _FrameRunner:
    """Runner stub exposing observation-frame requests only."""

    source_roles: dict = {}
    step = np.timedelta64(6, "h")

    def __init__(self, frames: pd.DataFrame | None) -> None:
        self.frames = frames

    def data_requests(self, item: WorkItem):
        return ()

    def frame_requests(self, item: WorkItem):
        return (
            FrameRequest(
                "obs",
                self,
                np.array([item.time]),
                np.array(["t2m"]),
                np.array(["lat", "lon", "value"]),
            ),
        )

    def __call__(self, time, variable, fields=None):
        if self.frames is None:
            return pd.DataFrame(
                {c: pd.Series(dtype=float) for c in ("lat", "lon", "value")}
            )
        return self.frames.copy()


def test_plan_frame_stores_and_download(cfg, tmp_path) -> None:
    frames = pd.DataFrame({"lat": [0.0, 10.0], "lon": [5.0, 15.0], "value": [1.0, 2.0]})
    runner = _FrameRunner(frames)
    items = build_items(cfg, runner)
    stores = plan_frame_stores(cfg, runner, items)
    assert [s.name for s in stores] == ["obs"]
    assert len(stores[0].times) == len(TIMES)
    with open_dict(cfg):
        cfg.predownload.observations.enabled = False
    assert plan_frame_stores(cfg, runner, items) == []
    with open_dict(cfg):
        cfg.predownload.observations.enabled = True

    from predownload import download_frame_store

    download_frame_store(cfg, stores[0], overwrite=False)
    store_dir = Path(frame_store_path(cfg.output.path, "obs"))
    for t in stores[0].times:
        assert (store_dir / frame_filename(t)).exists()
    cached = PredownloadedFrameSource(str(store_dir))
    df = cached(np.array(TIMES[:1], dtype="datetime64[ns]"), ["t2m"], fields=["value"])
    assert list(df["value"]) == [1.0, 2.0]


def test_empty_frames_are_stored_with_a_warning(cfg) -> None:
    from predownload import download_frame_store

    runner = _FrameRunner(None)
    items = build_items(cfg, runner)
    store = plan_frame_stores(cfg, runner, items)[0]
    download_frame_store(cfg, store, overwrite=False)
    store_dir = Path(frame_store_path(cfg.output.path, "obs"))
    assert len(pd.read_parquet(store_dir / frame_filename(store.times[0]))) == 0

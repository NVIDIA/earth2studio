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

"""Scorecard pipeline variants: the regional pipeline and its grid crop.

Uses a stand-in model with StormCast's surface (HRRR window coordinates,
2-D latitude/longitude, a ``conditioning_data_source`` hook) and a fake
full-grid source, so the tests need no weights and no network access.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf
from scorecard.utils.pipelines import (
    RegionalForecastPipeline,
    SubgridSource,
    _conditioning_window,
)
from src.data import PredownloadedSource

# A small "full" limited-area grid and the model window inside it.
FULL_Y = np.arange(6, dtype=float) * 3000.0
FULL_X = np.arange(8, dtype=float) * 3000.0
WIN_Y = FULL_Y[1:4]
WIN_X = FULL_X[2:6]
VARIABLES = ["a", "b"]


def _latlon(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat = 35.0 + y[:, None] / 3000.0 + 0 * x[None, :]
    lon = 260.0 + x[None, :] / 3000.0 + 0 * y[:, None]
    return lat, lon


class FullGridSource:
    """Fake HRRR: returns the whole grid with 2-D lat/lon coordinates."""

    def __init__(self, cache: bool = False, verbose: bool = False) -> None:
        self.calls: list = []

    def __call__(self, time, variable) -> xr.DataArray:
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = [str(v) for v in np.atleast_1d(variable)]
        self.calls.append((times, variables))
        lat, lon = _latlon(FULL_Y, FULL_X)
        data = np.random.default_rng(0).standard_normal(
            (len(times), len(variables), len(FULL_Y), len(FULL_X))
        )
        return xr.DataArray(
            data,
            dims=["time", "variable", "hrrr_y", "hrrr_x"],
            coords={
                "time": times,
                "variable": variables,
                "hrrr_y": FULL_Y,
                "hrrr_x": FULL_X,
                "lat": (("hrrr_y", "hrrr_x"), lat),
                "lon": (("hrrr_y", "hrrr_x"), lon),
            },
        )


class LeadTaggedSource(FullGridSource):
    """A source that tags analyses with a zero lead time."""

    def __call__(self, time, variable) -> xr.DataArray:
        da = super().__call__(time, variable)
        return da.expand_dims(lead_time=[np.timedelta64(0, "h")], axis=1)


class GlobalSource:
    """Fake ERA5 on the full 0.25 degree grid."""

    def __init__(self, cache: bool = False, verbose: bool = False) -> None:
        pass

    def __call__(self, time, variable) -> xr.DataArray:
        from scorecard.utils.baselines import ERA5_LAT, ERA5_LON

        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = [str(v) for v in np.atleast_1d(variable)]
        data = np.zeros((len(times), len(variables), len(ERA5_LAT), len(ERA5_LON)))
        return xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": times,
                "variable": variables,
                "lat": ERA5_LAT,
                "lon": ERA5_LON,
            },
        )


class FakeStormCast(torch.nn.Module):
    """Stand-in with StormCast's coordinate surface and conditioning hook."""

    conditioning_variables = np.array(["u10m", "t2m"])

    def __init__(self) -> None:
        super().__init__()
        self.hrrr_y = WIN_Y
        self.hrrr_x = WIN_X
        self.lat, self.lon = _latlon(WIN_Y, WIN_X)
        self.conditioning_data_source = None

    def input_coords(self):
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(VARIABLES),
                "hrrr_y": self.hrrr_y,
                "hrrr_x": self.hrrr_x,
            }
        )

    def output_coords(self, input_coords):
        out = self.input_coords()
        out["lead_time"] = np.array([np.timedelta64(1, "h")])
        return out

    @classmethod
    def load_default_package(cls):
        return None

    @classmethod
    def load_model(cls, package, conditioning_data_source=None, sampler_steps=18):
        model = cls()
        model.conditioning_data_source = conditioning_data_source
        return model


@pytest.fixture(autouse=True)
def _single_process(monkeypatch):
    """Model loading goes through ``run_on_rank0_first``, which needs the
    distributed manager; these are single-process unit tests.  The fixture
    also clears the data cache variable the pipeline sets."""
    monkeypatch.setattr(
        "src.models.run_on_rank0_first", lambda func, *a, **kw: func(*a, **kw)
    )
    monkeypatch.delenv("EARTH2STUDIO_DATA_CACHE", raising=False)


SOURCE = "test.test_scorecard_pipelines.FullGridSource"


def _cfg(
    tmp_path,
    nsteps: int = 2,
    separate_verification: bool = True,
    live: bool = False,
):
    verification = {"enabled": not live, "source": None}
    if separate_verification and not live:
        verification["source"] = {"_target_": SOURCE}
    return OmegaConf.create(
        {
            "model": {
                "architecture": "test.test_scorecard_pipelines.FakeStormCast",
                "load_args": {"conditioning_data_source": None},
            },
            "start_times": ["2025-05-19 00:00:00"],
            "nsteps": nsteps,
            "ensemble_size": 1,
            "random_seed": 0,
            "data_source": {"_target_": SOURCE},
            "ic_source": {"_target_": SOURCE} if live else None,
            "verification_source": {"_target_": SOURCE} if live else None,
            "predownload": {"overwrite": False, "verification": verification},
            "output": {"path": str(tmp_path / "run"), "variables": ["a"]},
            "scoring": {"variables": ["a"], "lat_weights": False},
        }
    )


class TestSubgridSource:
    def test_crops_to_window_and_drops_2d_coords(self):
        inner = FullGridSource()
        source = SubgridSource(inner, OrderedDict({"hrrr_y": WIN_Y, "hrrr_x": WIN_X}))
        da = source(np.datetime64("2025-05-19T00:00"), VARIABLES)
        assert da.shape == (1, 2, len(WIN_Y), len(WIN_X))
        np.testing.assert_array_equal(da.hrrr_y.values, WIN_Y)
        np.testing.assert_array_equal(da.hrrr_x.values, WIN_X)
        assert "lat" not in da.coords and "lon" not in da.coords
        # Values are the window of the full grid, not an interpolation.
        full = inner(np.datetime64("2025-05-19T00:00"), VARIABLES)
        np.testing.assert_array_equal(da.values, full.values[:, :, 1:4, 2:6])

    def test_nearest_tolerates_float_noise_and_ignores_non_spatial(self):
        ref = OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES),
                "hrrr_y": WIN_Y + 1e-6,
                "hrrr_x": WIN_X - 1e-6,
            }
        )
        da = SubgridSource(FullGridSource(), ref)(np.datetime64("2025-05-19"), "a")
        np.testing.assert_array_equal(da.hrrr_y.values, WIN_Y)
        assert list(da["variable"].values) == ["a"]

    def test_singleton_lead_time_is_squeezed(self):
        ref = OrderedDict({"hrrr_y": WIN_Y, "hrrr_x": WIN_X})
        da = SubgridSource(LeadTaggedSource(), ref)(np.datetime64("2025-05-19"), "a")
        assert da.dims == ("time", "variable", "hrrr_y", "hrrr_x")


class TestConditioningWindow:
    def test_window_covers_domain_with_margin(self):
        lat, lon = _latlon(WIN_Y, WIN_X)
        lats, lons = _conditioning_window(lat, lon, 2.0)
        assert lats[0] >= lat.max() + 2.0 - 0.25 and lats[-1] <= lat.min() - 2.0 + 0.25
        assert lons[0] <= lon.min() - 2.0 + 0.25 and lons[-1] >= lon.max() + 2.0 - 0.25
        assert np.all(np.diff(lats) < 0)  # ERA5 order, north to south


class TestRegionalForecastPipeline:
    def test_predownload_stores(self, tmp_path):
        pipeline = RegionalForecastPipeline(
            conditioning_source=GlobalSource(), conditioning_margin_deg=2.0
        )
        stores = {s.name: s for s in pipeline.predownload_stores(_cfg(tmp_path))}
        assert set(stores) == {"data", "verification", "conditioning"}

        for name in ("data", "verification"):
            store = stores[name]
            assert isinstance(store.source, SubgridSource)
            np.testing.assert_array_equal(store.spatial_ref["hrrr_y"], WIN_Y)
            da = store.source(store.times[0], store.variables)
            assert da.shape[-2:] == (len(WIN_Y), len(WIN_X))
        assert stores["data"].variables == VARIABLES
        assert stores["verification"].variables == ["a"]
        # Truth for every output tick of a 2-step rollout: IC, +1 h, +2 h.
        assert len(stores["verification"].times) == 3

        cond = stores["conditioning"]
        assert cond.role == "conditioning"
        assert cond.variables == ["u10m", "t2m"]
        # Conditioning at the input time of each step: IC and IC + 1 h.
        assert list(cond.times) == [
            np.datetime64("2025-05-19T00:00:00"),
            np.datetime64("2025-05-19T01:00:00"),
        ]
        lat, lon = _latlon(WIN_Y, WIN_X)
        assert cond.spatial_ref["lat"].max() >= lat.max() + 1.75
        assert cond.spatial_ref["lon"].min() <= lon.min() - 1.75
        da = cond.source(cond.times[0], cond.variables)
        assert da.shape[-2:] == (
            len(cond.spatial_ref["lat"]),
            len(cond.spatial_ref["lon"]),
        )

    def test_predownload_stores_merged_verification(self, tmp_path):
        pipeline = RegionalForecastPipeline(conditioning_source=GlobalSource())
        cfg = _cfg(tmp_path, separate_verification=False)
        stores = {s.name: s for s in pipeline.predownload_stores(cfg)}
        assert set(stores) == {"data", "conditioning"}
        assert isinstance(stores["data"].source, SubgridSource)

    def test_no_crop_keeps_raw_sources(self, tmp_path):
        pipeline = RegionalForecastPipeline(crop_to_model_grid=False)
        stores = {s.name: s for s in pipeline.predownload_stores(_cfg(tmp_path))}
        assert set(stores) == {"data", "verification"}
        assert isinstance(stores["data"].source, FullGridSource)

    def test_instantiates_conditioning_source_from_config(self, tmp_path):
        pipeline = RegionalForecastPipeline(
            conditioning_source={
                "_target_": "test.test_scorecard_pipelines.GlobalSource"
            }
        )
        stores = {s.name: s for s in pipeline.predownload_stores(_cfg(tmp_path))}
        assert "conditioning" in stores

    def test_live_mode_declares_nothing_and_attaches_the_source(self, tmp_path):
        source = GlobalSource()
        pipeline = RegionalForecastPipeline(
            conditioning_source=source, conditioning_mode="live"
        )
        cfg = _cfg(tmp_path, live=True)
        assert pipeline.predownload_stores(cfg) == []
        pipeline.setup(cfg, torch.device("cpu"))
        assert pipeline.prognostic.conditioning_data_source is source

    def test_live_mode_instantiates_a_configured_source_once(self, tmp_path):
        pipeline = RegionalForecastPipeline(
            conditioning_source={
                "_target_": "test.test_scorecard_pipelines.GlobalSource"
            },
            conditioning_mode="live",
        )
        cfg = _cfg(tmp_path, live=True)
        pipeline.setup(cfg, torch.device("cpu"))
        first = pipeline.prognostic.conditioning_data_source
        assert isinstance(first, GlobalSource)
        pipeline.setup(cfg, torch.device("cpu"))
        assert pipeline.prognostic.conditioning_data_source is first

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError, match="conditioning_mode"):
            RegionalForecastPipeline(conditioning_mode="stream")

    def test_setup_attaches_predownloaded_conditioning(self, tmp_path):
        cfg = _cfg(tmp_path)
        pipeline = RegionalForecastPipeline(conditioning_source=GlobalSource())
        with pytest.raises(FileNotFoundError, match="conditioning store"):
            pipeline.setup(cfg, torch.device("cpu"))

        store = tmp_path / "run" / "conditioning.zarr"
        xr.Dataset(
            {v: (("time", "lat", "lon"), np.zeros((1, 3, 4))) for v in ("u10m", "t2m")},
            coords={
                "time": np.array(["2025-05-19T00:00"], dtype="datetime64[ns]"),
                "lat": np.array([40.0, 39.75, 39.5]),
                "lon": np.array([260.0, 260.25, 260.5, 260.75]),
            },
        ).to_zarr(store, mode="w")
        pipeline.setup(cfg, torch.device("cpu"))
        source = pipeline.prognostic.conditioning_data_source
        assert isinstance(source, PredownloadedSource)
        da = source(np.datetime64("2025-05-19T00:00"), ["u10m"])
        assert da.dims == ("time", "variable", "lat", "lon")

    def test_live_verification_source_is_cropped(self, tmp_path):
        cfg = _cfg(tmp_path, live=True)
        pipeline = RegionalForecastPipeline()
        pipeline.setup(cfg, torch.device("cpu"))
        source = pipeline.verification_source(cfg)
        assert isinstance(source, SubgridSource)
        da = source([np.datetime64("2025-05-19T01:00")], ["a"])
        assert da.dims == ("time", "variable", "hrrr_y", "hrrr_x")
        assert da.shape[-2:] == (len(WIN_Y), len(WIN_X))

    def test_live_verification_source_without_setup_inspects_the_model(self, tmp_path):
        cfg = _cfg(tmp_path, live=True)
        source = RegionalForecastPipeline().verification_source(cfg)
        da = source(np.datetime64("2025-05-19T01:00"), "a")
        assert da.shape[-2:] == (len(WIN_Y), len(WIN_X))

    def test_live_verification_source_not_cropped_when_disabled(self, tmp_path):
        cfg = _cfg(tmp_path, live=True)
        source = RegionalForecastPipeline(crop_to_model_grid=False).verification_source(
            cfg
        )
        assert isinstance(source, FullGridSource)

    def test_isolated_data_cache_per_process(self, tmp_path):
        cfg = _cfg(tmp_path, live=True)
        pipeline = RegionalForecastPipeline()
        pipeline.setup(cfg, torch.device("cpu"))
        cache = os.environ.get("EARTH2STUDIO_DATA_CACHE")
        assert cache and os.path.isdir(cache) and "e2s-data-cache-" in cache
        # Idempotent: a second call keeps the same directory.
        pipeline.setup(cfg, torch.device("cpu"))
        assert os.environ["EARTH2STUDIO_DATA_CACHE"] == cache

    def test_isolation_can_be_disabled(self, tmp_path):
        cfg = _cfg(tmp_path, live=True)
        RegionalForecastPipeline(isolate_data_cache=False).setup(
            cfg, torch.device("cpu")
        )
        assert "EARTH2STUDIO_DATA_CACHE" not in os.environ

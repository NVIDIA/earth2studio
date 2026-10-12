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

"""Tests for window-mean (subseasonal-to-seasonal) scoring.

Covers :class:`src.data.WindowMeanSource`, :class:`src.regrid.LinearRegridder`
and :class:`src.pipelines.seasonal.WindowMeanForecastPipeline`.  The
pipeline runs a stub prognostic whose state grows by one per step and a
data source whose fields equal the valid time in hours, so every window
mean has a closed-form value.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf
from src.data import WindowMeanSource
from src.pipelines import WindowMeanForecastPipeline, build_pipeline
from src.regrid import LinearRegridder, RegriddedSource
from src.work import WorkItem

from earth2studio.utils.coords import CoordSystem

LAT = np.linspace(90.0, -90.0, 37)  # 5 degrees
LON = np.arange(0.0, 360.0, 5.0)
VARIABLES = ["t2m", "z500"]
HOUR = np.timedelta64(1, "h")
EPOCH = np.datetime64("2024-01-01T00:00", "ns")
_RANK0 = "src.models.run_on_rank0_first"


def _hours(*values: int) -> np.ndarray:
    return (np.array(values) * HOUR).astype("timedelta64[ns]")


class _HourSource:
    """DataSource whose fields are constant: the valid time in hours since 2024-01-01.

    Records every request so tests can check which samples a wrapper asked for.
    """

    def __init__(self, nan_at: str | None = None) -> None:
        self.calls: list[np.ndarray] = []
        self._nan_at = None if nan_at is None else np.datetime64(nan_at, "ns")

    def __call__(self, time: Any, variable: Any) -> xr.DataArray:
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = (
            [variable] if isinstance(variable, str) else [str(v) for v in variable]
        )
        self.calls.append(times.copy())
        hours = ((times - EPOCH) / HOUR).astype(np.float64)[:, None, None, None]
        data = np.broadcast_to(
            hours, (len(times), len(variables), len(LAT), len(LON))
        ).copy()
        if self._nan_at is not None:
            data[times == self._nan_at] = np.nan
        return xr.DataArray(
            data,
            dims=("time", "variable", "lat", "lon"),
            coords={"time": times, "variable": variables, "lat": LAT, "lon": LON},
        )

    async def fetch(self, time: Any, variable: Any) -> xr.DataArray:
        return self(time, variable)


# ---------------------------------------------------------------------------
# WindowMeanSource
# ---------------------------------------------------------------------------


class TestWindowMeanSource:
    def test_mean_over_trailing_window(self):
        source = _HourSource()
        mean = WindowMeanSource(source, "24h", "6h")
        out = mean(np.datetime64("2024-01-02T00"), VARIABLES)
        assert out.dims == ("time", "variable", "lat", "lon")
        assert out.time.values[0] == np.datetime64("2024-01-02T00", "ns")
        # Samples at 06, 12, 18 and 24 hours: the window excludes its start.
        np.testing.assert_allclose(out.values, 15.0)
        np.testing.assert_array_equal(
            np.concatenate(source.calls), EPOCH + _hours(6, 12, 18, 24)
        )

    def test_chunks_requests_to_the_wrapped_source(self):
        source = _HourSource()
        mean = WindowMeanSource(source, "24h", "1h", chunk_size=5)
        out = mean([np.datetime64("2024-01-02T00")], "t2m")
        assert [len(c) for c in source.calls] == [5, 5, 5, 5, 4]
        np.testing.assert_allclose(out.values, 12.5)  # mean of 1..24

    def test_several_times_concatenate_in_request_order(self):
        mean = WindowMeanSource(_HourSource(), "24h", "6h")
        times = np.array(
            [np.datetime64("2024-01-03T00"), np.datetime64("2024-01-02T00")]
        )
        out = mean(times, VARIABLES)
        assert out.shape == (2, 2, len(LAT), len(LON))
        np.testing.assert_array_equal(out.time.values, times.astype("datetime64[ns]"))
        np.testing.assert_allclose(out.isel(time=0).values, 39.0)
        np.testing.assert_allclose(out.isel(time=1).values, 15.0)

    def test_start_alignment_gives_calendar_day_means(self):
        source = _HourSource()
        mean = WindowMeanSource(source, "24h", "6h", align="start")
        out = mean(np.datetime64("2024-01-02T00"), "t2m")
        # Samples at 00, 06, 12 and 18 UTC of January 2nd: hours 24..42.
        np.testing.assert_allclose(out.values, 33.0)
        np.testing.assert_array_equal(
            np.concatenate(source.calls), EPOCH + _hours(24, 30, 36, 42)
        )

    def test_rejects_unknown_alignment(self):
        with pytest.raises(ValueError, match="align"):
            WindowMeanSource(_HourSource(), "24h", "6h", align="middle")

    def test_composes_daily_means_into_window_means(self):
        source = _HourSource()
        daily = WindowMeanSource(source, "24h", "6h", align="start")
        weekly = WindowMeanSource(daily, "48h", "24h")
        out = weekly(np.datetime64("2024-01-03T00"), "t2m")
        # The days January 2 and 3 (hours 24..42 and 48..66): mean 45.
        np.testing.assert_allclose(out.values, 45.0)
        np.testing.assert_array_equal(
            np.concatenate(source.calls), EPOCH + _hours(*range(24, 67, 6))
        )

    def test_rejects_a_source_that_drops_times(self):
        class _Dropping(_HourSource):
            def __call__(self, time: Any, variable: Any) -> xr.DataArray:
                return super().__call__(time, variable).isel(time=slice(1, None))

        mean = WindowMeanSource(_Dropping(), "24h", "6h")
        with pytest.raises(ValueError, match="requested times"):
            mean(np.datetime64("2024-01-02T00"), "t2m")

    def test_nan_sample_propagates(self):
        mean = WindowMeanSource(_HourSource(nan_at="2024-01-01T12"), "24h", "6h")
        out = mean(np.datetime64("2024-01-02T00"), "t2m")
        assert np.isnan(out.values).all()

    @pytest.mark.parametrize(
        "window, cadence",
        [("25h", "6h"), ("0h", "6h"), ("24h", "0h"), ("3h", "6h")],
    )
    def test_rejects_window_not_a_multiple_of_cadence(self, window, cadence):
        with pytest.raises(ValueError):
            WindowMeanSource(_HourSource(), window, cadence)

    def test_async_fetch_matches_call(self):
        mean = WindowMeanSource(_HourSource(), "12h", "6h")
        t = np.datetime64("2024-01-02T00")
        xr.testing.assert_equal(mean(t, "t2m"), asyncio.run(mean.fetch(t, "t2m")))


# ---------------------------------------------------------------------------
# LinearRegridder
# ---------------------------------------------------------------------------


class TestLinearRegridder:
    def test_coincident_targets_subsample(self):
        regridder = LinearRegridder(LAT, LON, LAT[::2], LON[::2])
        x = torch.randn(3, len(LAT), len(LON))
        y = regridder.apply(x, spatial_dims=("lat", "lon"))
        torch.testing.assert_close(y, x[:, ::2, ::2])

    def test_linear_field_is_reproduced_between_points(self):
        target_lat = np.array([87.5, 0.0, -42.5])
        target_lon = np.array([2.5, 180.0, 352.5])
        regridder = LinearRegridder(LAT, LON, target_lat, target_lon)
        field = torch.as_tensor(2.0 * LAT[:, None] + 0.5 * LON[None, :])
        y = regridder.apply(field, spatial_dims=("lat", "lon"))
        expected = 2.0 * target_lat[:, None] + 0.5 * target_lon[None, :]
        np.testing.assert_allclose(y.numpy(), expected)

    def test_longitude_wraps_on_a_global_grid(self):
        regridder = LinearRegridder(LAT, LON, np.array([0.0]), np.array([357.5]))
        x = torch.zeros(len(LAT), len(LON))
        x[:, 0] = 10.0
        x[:, -1] = 20.0
        y = regridder.apply(x, spatial_dims=("lat", "lon"))
        assert y.item() == pytest.approx(15.0)  # halfway between 355 and 360

    def test_nan_neighbour_does_not_spread_into_coincident_points(self):
        regridder = LinearRegridder(LAT, LON, LAT[::2], LON[::2])
        x = torch.arange(len(LAT) * len(LON), dtype=torch.float32)
        x = x.view(len(LAT), len(LON))
        x[:, 1::2] = float("nan")  # every odd source column is land
        y = regridder.apply(x, spatial_dims=("lat", "lon"))
        torch.testing.assert_close(y, x[::2, ::2])
        between = LinearRegridder(LAT, LON, LAT[:1], np.array([2.5]))
        assert torch.isnan(between.apply(x, spatial_dims=("lat", "lon"))).all()

    def test_inexact_grid_coordinates_still_subsample(self):
        lon = np.arange(0.0, 360.0, 0.1)  # 0.1 degree steps are not exact floats
        regridder = LinearRegridder(LAT, lon, LAT, np.arange(0.0, 360.0, 1.0))
        x = torch.zeros(len(LAT), len(lon))
        x[:, 1::2] = float("nan")
        y = regridder.apply(x, spatial_dims=("lat", "lon"))
        torch.testing.assert_close(y, torch.zeros(len(LAT), 360))

    def test_target_outside_latitude_range_raises(self):
        lat_720 = np.linspace(90.0, -89.75, 720)
        with pytest.raises(ValueError, match="outside"):
            LinearRegridder(lat_720, LON, np.array([-90.0]), np.array([0.0]))

    def test_regional_longitude_does_not_wrap(self):
        with pytest.raises(ValueError, match="outside"):
            LinearRegridder(LAT, np.arange(0.0, 100.0, 5.0), LAT[:2], np.array([150.0]))

    def test_to_resolution_subsamples_a_quarter_degree_grid(self):
        regridder = LinearRegridder.to_resolution(
            np.linspace(90.0, -90.0, 721), np.arange(0.0, 360.0, 0.25), 1.0
        )
        assert regridder is not None
        coords = regridder.target_coords()
        np.testing.assert_array_equal(coords["lat"], np.arange(90.0, -91.0, -1.0))
        np.testing.assert_array_equal(coords["lon"], np.arange(0.0, 360.0, 1.0))
        x = torch.arange(721 * 1440, dtype=torch.float32).view(721, 1440)
        torch.testing.assert_close(
            regridder.apply(x, spatial_dims=("lat", "lon")), x[::4, ::4]
        )

    def test_to_resolution_keeps_only_covered_rows(self):
        regridder = LinearRegridder.to_resolution(
            np.linspace(90.0, -89.75, 720), np.arange(0.0, 360.0, 0.25), 1.0
        )
        assert regridder is not None
        np.testing.assert_array_equal(
            regridder.target_coords()["lat"], np.arange(90.0, -90.0, -1.0)
        )

    def test_to_resolution_follows_the_source_extent(self):
        western = LinearRegridder.to_resolution(
            np.linspace(90.0, -90.0, 721), np.arange(-180.0, 180.0, 0.25), 1.0
        )
        assert western is not None
        np.testing.assert_array_equal(
            western.target_coords()["lon"], np.arange(-180.0, 180.0, 1.0)
        )
        regional = LinearRegridder.to_resolution(
            np.arange(60.0, 19.9, -0.5), np.arange(100.0, 160.1, 0.5), 1.0
        )
        assert regional is not None
        coords = regional.target_coords()
        np.testing.assert_array_equal(coords["lat"], np.arange(60.0, 19.0, -1.0))
        np.testing.assert_array_equal(coords["lon"], np.arange(100.0, 161.0, 1.0))

    def test_to_resolution_is_none_for_a_grid_that_is_already_coarse(self):
        assert (
            LinearRegridder.to_resolution(
                np.linspace(90.0, -90.0, 121), np.arange(0.0, 360.0, 1.5), 1.0
            )
            is None
        )
        assert LinearRegridder.to_resolution(LAT, LON, 5.0) is None

    def test_apply_with_coords_replaces_spatial_dims(self):
        regridder = LinearRegridder(LAT, LON, LAT[::2], LON[::2])
        coords: CoordSystem = OrderedDict(
            {
                "time": np.array([EPOCH]),
                "lead_time": _hours(24),
                "variable": np.array(VARIABLES),
                "lat": LAT,
                "lon": LON,
            }
        )
        x = torch.zeros(1, 1, 2, len(LAT), len(LON))
        y, out = regridder.apply_with_coords(x, coords)
        assert list(out) == ["time", "lead_time", "variable", "lat", "lon"]
        assert y.shape == (1, 1, 2, 19, 36)
        np.testing.assert_array_equal(out["lat"], LAT[::2])

    def test_regridded_source_returns_target_grid(self):
        regridder = LinearRegridder(LAT, LON, LAT[::2], LON[::2])
        da = RegriddedSource(_HourSource(), regridder)(
            np.datetime64("2024-01-01T06"), VARIABLES
        )
        assert da.dims == ("time", "variable", "lat", "lon")
        np.testing.assert_array_equal(da.lat.values, LAT[::2])
        np.testing.assert_allclose(da.values, 6.0)

    def test_to_returns_self(self):
        regridder = LinearRegridder(LAT, LON, LAT[::2], LON[::2])
        assert regridder.to("cpu") is regridder


# ---------------------------------------------------------------------------
# WindowMeanForecastPipeline
# ---------------------------------------------------------------------------


class _CountingModel:
    """Stub prognostic with a 6-hour step whose state grows by one each step.

    The mean over steps ``a..b`` is the initial state plus ``(a + b) / 2``.
    Exposes the ``load_default_package`` / ``load_model`` hooks that
    :func:`src.models.load_prognostic` calls, so the pipeline sets up from
    a config exactly as with a real model.
    """

    @classmethod
    def load_default_package(cls) -> None:
        return None

    @classmethod
    def load_model(cls, package: Any = None) -> _CountingModel:
        return cls()

    def to(self, device: Any) -> _CountingModel:
        return self

    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": _hours(0),
                "variable": np.array(VARIABLES),
                "lat": LAT,
                "lon": LON,
            }
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        out = OrderedDict(input_coords)
        out["lead_time"] = _hours(6)
        return out

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        coords = OrderedDict(coords)
        yield x, coords
        step = 0
        while True:
            step += 1
            x = x + 1.0
            out = OrderedDict(coords)
            out["lead_time"] = _hours(6 * step)
            yield x, out


class _StoppingModel(_CountingModel):
    """Stops after six steps, two steps into the second 24-hour window."""

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        for step, (x_step, c) in enumerate(super().create_iterator(x, coords)):
            if step > 6:
                return
            yield x_step, c


class _PairModel(_CountingModel):
    """Yields two 6-hour leads per step, like a multi-lead model."""

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        coords = OrderedDict(coords)
        yield x, coords
        lt_axis = list(coords).index("lead_time")
        step = 0
        while True:
            step += 1
            pair = torch.cat([x + (2 * step - 1), x + 2 * step], dim=lt_axis)
            out = OrderedDict(coords)
            out["lead_time"] = _hours(12 * step - 6, 12 * step)
            yield pair, out


class _RecordingScorer:
    """Stands in for the online scorer: keeps every chunk ``Pipeline.run`` hands over."""

    def __init__(self) -> None:
        self.steps: list[tuple[torch.Tensor, CoordSystem]] = []

    def begin_item(self, item: WorkItem) -> None:
        pass

    def update(self, x: torch.Tensor, coords: CoordSystem) -> None:
        self.steps.append((x.clone(), OrderedDict(coords)))

    def finish_item(self, item: WorkItem) -> None:
        pass


def _cfg(
    tmp_path,
    *,
    nsteps: int = 8,
    window: str = "24h",
    target_resolution: float | None = 10.0,
    start_times: tuple[str, ...] = ("2024-01-01 00:00:00",),
    model: str = "_CountingModel",
):
    return OmegaConf.create(
        {
            "project": "test_eval",
            "run_id": "s2s",
            "start_times": list(start_times),
            "nsteps": nsteps,
            "ensemble_size": 1,
            "random_seed": 42,
            "pipeline": {
                "_target_": "src.pipelines.seasonal.WindowMeanForecastPipeline",
                "window": window,
                "target_resolution": target_resolution,
            },
            "model": {"architecture": f"{__name__}.{model}"},
            "data_source": {"_target_": f"{__name__}._HourSource"},
            "predownload": {"verification": {"enabled": False, "source": None}},
            "output": {
                "path": str(tmp_path / "outputs"),
                "variables": list(VARIABLES),
                "overwrite": True,
                "thread_writers": 0,
                "chunks": {"time": 1, "lead_time": 1},
            },
        }
    )


class TestWindowMeanForecastPipeline:
    @pytest.fixture(autouse=True)
    def _single_process(self):
        # load_prognostic wraps the package download in the distributed
        # rank-0 helper, which needs an initialized manager; run it inline.
        with patch(_RANK0, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
            yield

    def test_window_leads_end_each_window(self):
        pipeline = WindowMeanForecastPipeline(window="24h")
        leads = pipeline.window_leads(np.timedelta64(6, "h"), nsteps=8)
        np.testing.assert_array_equal(leads, _hours(24, 48))

    @pytest.mark.parametrize(
        "window, nsteps, match",
        [
            ("20h", 8, "whole number of model steps"),
            ("24h", 6, "whole number of 4-step windows"),
            ("24h", 2, "whole number of 4-step windows"),
        ],
    )
    def test_window_leads_reject_misaligned_windows(self, window, nsteps, match):
        pipeline = WindowMeanForecastPipeline(window=window)
        with pytest.raises(ValueError, match=match):
            pipeline.window_leads(np.timedelta64(6, "h"), nsteps)

    def test_rejects_non_positive_arguments(self):
        with pytest.raises(ValueError, match="window"):
            WindowMeanForecastPipeline(window="0h")
        with pytest.raises(ValueError, match="target_resolution"):
            WindowMeanForecastPipeline(window="24h", target_resolution=0)

    def test_setup_builds_lead_axis_and_target_grid(self, tmp_path):
        cfg = _cfg(tmp_path)
        pipeline = build_pipeline(cfg)
        pipeline.setup(cfg, torch.device("cpu"))
        total = pipeline.build_total_coords(np.array([EPOCH]), 1)
        assert list(total) == ["time", "lead_time", "lat", "lon"]
        np.testing.assert_array_equal(total["lead_time"], _hours(24, 48))
        np.testing.assert_array_equal(total["lat"], np.arange(90.0, -91.0, -10.0))
        np.testing.assert_array_equal(total["lon"], np.arange(0.0, 360.0, 10.0))
        assert pipeline.supports_member_batching()

    def test_coarser_model_grid_is_kept(self, tmp_path):
        cfg = _cfg(tmp_path, target_resolution=1.0)
        pipeline = build_pipeline(cfg)
        pipeline.setup(cfg, torch.device("cpu"))
        assert pipeline._output_regridder is None
        np.testing.assert_array_equal(pipeline.effective_spatial_ref()["lat"], LAT)

    def test_run_item_yields_one_mean_per_window(self, tmp_path):
        cfg = _cfg(tmp_path)
        pipeline = build_pipeline(cfg)
        pipeline.setup(cfg, torch.device("cpu"))
        item = WorkItem(time=EPOCH, ensemble_id=0, seed=0)
        steps = list(pipeline.run_item(item, _HourSource(), torch.device("cpu")))
        assert len(steps) == 2
        assert [c["lead_time"][0] for _, c in steps] == list(_hours(24, 48))
        # The IC field is 0 (hours since the epoch) and step k adds k, so the
        # windows average steps 1..4 and 5..8 on the model's native grid.
        assert steps[0][0].shape == (1, 1, 2, len(LAT), len(LON))
        torch.testing.assert_close(steps[0][0], torch.full_like(steps[0][0], 2.5))
        torch.testing.assert_close(steps[1][0], torch.full_like(steps[1][0], 6.5))

    def test_run_item_batched_averages_each_member(self, tmp_path):
        cfg = _cfg(tmp_path)
        pipeline = build_pipeline(cfg)
        pipeline.setup(cfg, torch.device("cpu"))
        items = [WorkItem(time=EPOCH, ensemble_id=m, seed=m) for m in range(2)]
        steps = list(
            pipeline.run_item_batched(items, _HourSource(), torch.device("cpu"))
        )
        assert len(steps) == 2
        x, coords = steps[0]
        assert list(coords)[0] == "ensemble"
        assert list(coords["ensemble"]) == [0, 1]
        assert x.shape == (2, 1, 1, 2, len(LAT), len(LON))
        torch.testing.assert_close(x, torch.full_like(x, 2.5))

    def test_rollout_that_stops_early_raises(self, tmp_path):
        cfg = _cfg(tmp_path, model="_StoppingModel")
        pipeline = build_pipeline(cfg)
        pipeline.setup(cfg, torch.device("cpu"))
        item = WorkItem(time=EPOCH, ensemble_id=0, seed=0)
        with pytest.raises(RuntimeError, match="2 steps short"):
            list(pipeline.run_item(item, _HourSource(), torch.device("cpu")))

    def test_rollout_with_several_leads_per_step(self, tmp_path):
        cfg = _cfg(tmp_path, model="_PairModel")
        pipeline = build_pipeline(cfg)
        pipeline.setup(cfg, torch.device("cpu"))
        item = WorkItem(time=EPOCH, ensemble_id=0, seed=0)
        steps = pipeline.run_item(item, _HourSource(), torch.device("cpu"))
        # Leads 6h..24h and 30h..48h fill the two windows across step boundaries.
        first, second = next(steps), next(steps)
        torch.testing.assert_close(first[0], torch.full_like(first[0], 2.5))
        torch.testing.assert_close(second[0], torch.full_like(second[0], 6.5))
        assert second[1]["lead_time"][0] == _hours(48)[0]
        # nsteps counts iterator steps, so a third window has no lead on the axis.
        with pytest.raises(RuntimeError, match="next window end"):
            next(steps)

    def test_run_scores_regridded_means(self, tmp_path):
        cfg = _cfg(tmp_path, start_times=("2024-01-01 00:00:00", "2024-01-02 00:00:00"))
        pipeline = build_pipeline(cfg)
        pipeline.setup(cfg, torch.device("cpu"))
        times = [EPOCH, EPOCH + _hours(24)[0]]
        items = [WorkItem(time=t, ensemble_id=0, seed=i) for i, t in enumerate(times)]
        scorer = _RecordingScorer()
        pipeline.run(
            items,
            _HourSource(),
            None,
            VARIABLES,
            torch.device("cpu"),
            cfg,
            scorer=scorer,
        )
        assert len(scorer.steps) == 4
        x, coords = scorer.steps[3]
        assert x.shape == (1, 1, 2, 19, 36)
        assert coords["lead_time"][0] == _hours(48)[0]
        torch.testing.assert_close(x, torch.full_like(x, 24.0 + 6.5))

    def test_predownload_declares_only_the_ic_store(self, tmp_path):
        cfg = _cfg(tmp_path, start_times=("2024-01-01 00:00:00", "2024-01-02 00:00:00"))
        stores = build_pipeline(cfg).predownload_stores(cfg)
        assert [s.role for s in stores] == ["ic"]
        assert list(stores[0].times) == [EPOCH, EPOCH + _hours(24)[0]]
        assert list(stores[0].variables) == VARIABLES
        np.testing.assert_array_equal(stores[0].spatial_ref["lat"], LAT)

    def test_predownload_rejects_verification(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.predownload.verification.enabled = True
        with pytest.raises(ValueError, match="verification"):
            build_pipeline(cfg).predownload_stores(cfg)

    def test_predownload_follows_a_byo_ic_source(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.ic_source = {"_target_": f"{__name__}._HourSource"}
        assert build_pipeline(cfg).predownload_stores(cfg) == []

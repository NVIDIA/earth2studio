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

"""Window-mean runner for subseasonal-to-seasonal archives.

The prognostic is a persistence model whose state grows by one per 6-hour
step, so every window mean has a closed-form value relative to the initial
condition drawn from the ``Random`` source.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import xarray as xr
from main import run
from omegaconf import open_dict
from src.campaign import build_items, final_lead_time
from src.output import store_path
from src.runners import build_runner
from src.runners.ensemble import EnsembleRunner
from src.runners.seasonal import WindowMeanRunner, window_mean

from earth2studio.io import KVBackend
from earth2studio.models.px import Persistence
from earth2studio.run import Pipeline, work_items
from test.conftest import VARIABLES

T0 = np.datetime64("2024-01-01T00:00:00")
H6 = np.timedelta64(6, "h")
H12 = np.timedelta64(12, "h")


class _Counting(Persistence):
    """Persistence whose state grows by one every step."""

    def _forward(self, x: xr.DataArray) -> xr.DataArray:
        y = super()._forward(x)
        tensor, _ = y.e2s.to_torch()
        tensor.add_(1.0)
        return y

    def step(self, y: xr.DataArray, state: None) -> tuple[xr.DataArray, None]:
        out, state = super().step(y, state)
        tensor, _ = out.e2s.to_torch()
        tensor.add_(1.0)
        return out, state


@pytest.fixture()
def counting(small_grid) -> _Counting:
    return _Counting(variable=VARIABLES, domain_coords=small_grid)


@pytest.fixture()
def runner(counting, data_source, fake_diagnostic) -> WindowMeanRunner:
    inner = EnsembleRunner(counting, data_source, diagnostics={"diag": fake_diagnostic})
    return WindowMeanRunner(inner, window="12h")


def _values(x: xr.DataArray) -> np.ndarray:
    return x.e2s.to_torch()[0].cpu().numpy()


# ---------------------------------------------------------------------------
# Lead schema
# ---------------------------------------------------------------------------


def test_window_labels_replace_the_lead_axis(runner) -> None:
    schemas = runner.output_coords(4 * H6)
    assert set(schemas) == {"forecast", "diag"}
    inner = runner.inner.output_coords(4 * H6)
    for stream, schema in schemas.items():
        np.testing.assert_array_equal(
            schema["lead_time"].values,
            np.array([H12, 2 * H12], dtype="timedelta64[ns]"),
        )
        assert schema.dims == inner[stream].dims
        np.testing.assert_array_equal(
            schema["variable"].values, inner[stream]["variable"].values
        )
    assert runner.step == H6
    assert runner.window == np.timedelta64(H12, "ns")
    assert runner.supports_member_batching


def test_window_must_be_whole_steps(runner) -> None:
    runner.window = np.timedelta64(9, "h").astype("timedelta64[ns]")
    with pytest.raises(ValueError, match="whole number of model steps"):
        runner.output_coords(18 * np.timedelta64(1, "h"))


def test_final_lead_time_must_cover_whole_windows(runner) -> None:
    with pytest.raises(ValueError, match="whole number of .* windows"):
        runner.output_coords(3 * H6)
    with pytest.raises(ValueError, match="covers no window"):
        runner.output_coords(np.timedelta64(0, "h"))


def test_rejects_calendar_months_and_bad_windows(counting, data_source) -> None:
    inner = EnsembleRunner(counting, data_source)
    with pytest.raises(NotImplementedError, match="month"):
        WindowMeanRunner(inner, window="month")
    with pytest.raises(ValueError, match="positive"):
        WindowMeanRunner(inner, window="-6h")


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def test_run_item_averages_each_window(runner) -> None:
    item = work_items([T0], 4 * H6)[0]
    np.random.seed(0)
    ic = _values(next(iter(runner.inner.run_item(item)))["forecast"])
    np.random.seed(0)
    outputs = list(runner.run_item(item))
    assert len(outputs) == 2
    for k, step in enumerate(outputs, start=1):
        assert set(step) == {"forecast", "diag"}
        forecast = step["forecast"]
        assert forecast.dims == ("time", "lead_time", "variable", "lat", "lon")
        np.testing.assert_array_equal(forecast["lead_time"].values, [k * H12])
        np.testing.assert_array_equal(forecast["time"].values, [T0])
        # Window k holds steps 2k-1 and 2k: mean ic + 2k - 0.5.
        np.testing.assert_allclose(_values(forecast), ic + 2 * k - 0.5, rtol=1e-6)
        diag = step["diag"]
        np.testing.assert_array_equal(diag["lead_time"].values, [k * H12])
        np.testing.assert_array_equal(diag["variable"].values, ["diag_a", "t2m"])
        np.testing.assert_allclose(
            _values(diag.sel(variable="t2m")), _values(forecast.sel(variable="t2m"))
        )
        np.testing.assert_array_equal(_values(diag.sel(variable="diag_a")), 0.0)


def test_run_item_keeps_members_on_the_leading_axis(runner) -> None:
    item = work_items([T0], 2 * H6, members=[1, 4], member_batch=2)[0]
    outputs = list(runner.run_item(item))
    assert len(outputs) == 1
    forecast = outputs[0]["forecast"]
    assert forecast.dims[0] == "ensemble"
    np.testing.assert_array_equal(forecast["ensemble"].values, [1, 4])
    assert forecast.sizes["lead_time"] == 1


def test_pipeline_writes_one_field_per_window(runner) -> None:
    store = KVBackend()
    items = work_items(
        [T0, T0 + np.timedelta64(1, "D")], 4 * H6, members=4, member_batch=2
    )
    Pipeline(
        runner,
        output={"forecast": store},
        output_variables={"forecast": ["t2m"]},
        device="cpu",
    ).run(items)
    assert tuple(store.dims["t2m"]) == ("ensemble", "time", "lead_time", "lat", "lon")
    assert store["t2m"].shape[:3] == (4, 2, 2)
    np.testing.assert_array_equal(store["lead_time"].values, np.array([H12, 2 * H12]))
    assert not torch.isnan(store["t2m"]).any()
    assert "z500" not in store


# ---------------------------------------------------------------------------
# Builder and recipe entry point
# ---------------------------------------------------------------------------


def _window_cfg(base_cfg, nsteps: int = 4):
    cfg = base_cfg
    with open_dict(cfg):
        cfg.runner = {"name": "window_mean", "window": "12h"}
        cfg.nsteps = nsteps
    return cfg


def test_builder_wraps_the_forecast_runner(base_cfg) -> None:
    cfg = _window_cfg(base_cfg)
    runner = build_runner(cfg)
    assert isinstance(runner, WindowMeanRunner)
    assert isinstance(runner.inner, EnsembleRunner)
    assert runner.step == H6
    assert "data" in runner.source_roles
    # nsteps keeps meaning model steps.
    assert final_lead_time(cfg, runner) == 4 * H6
    items = build_items(cfg, runner)
    assert items[0].final_lead_time == 4 * H6
    direct = window_mean(cfg, window="1D")
    assert direct.window == np.timedelta64(1, "D").astype("timedelta64[ns]")


def test_builder_rejects_partial_windows_before_running(base_cfg) -> None:
    cfg = _window_cfg(base_cfg, nsteps=3)
    runner = build_runner(cfg)
    with pytest.raises(ValueError, match="whole number"):
        runner.output_coords(final_lead_time(cfg, runner))


def test_main_run_stores_window_means_on_the_output_grid(base_cfg) -> None:
    cfg = _window_cfg(base_cfg)
    with open_dict(cfg):
        cfg.output.regrid = {
            "grid": {
                "lat": {"start": 90.0, "stop": -90.0, "num": 2},
                "lon": {"start": 0.0, "stop": 180.0, "num": 2},
            },
            "method": "nearest",
        }
    run(cfg)
    ds = xr.open_zarr(store_path(cfg), consolidated=False)
    assert ds["t2m"].dims == ("time", "lead_time", "lat", "lon")
    np.testing.assert_array_equal(ds["lead_time"].values, np.array([H12, 2 * H12]))
    assert ds["t2m"].shape[-2:] == (2, 2)
    assert not np.isnan(ds["t2m"].values).any()

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

"""End-to-end: the recipe config drives the library Pipeline."""

from __future__ import annotations

import numpy as np
import xarray as xr
from main import run
from omegaconf import open_dict
from src.campaign import build_items
from src.output import store_path, stream_variables
from src.runners import build_runner
from src.runners.ensemble import EnsembleRunner

from test.conftest import SMALL_LAT, SMALL_LON, VARIABLES


def _open(cfg) -> xr.Dataset:
    return xr.open_zarr(store_path(cfg), consolidated=False)


def test_build_runner_defaults_to_forecast(base_cfg) -> None:
    runner = build_runner(base_cfg)
    assert isinstance(runner, EnsembleRunner)
    items = build_items(base_cfg, runner)
    assert len(items) == 1 and items[0].final_lead_time == np.timedelta64(12, "h")
    schemas = runner.output_coords(items[0].final_lead_time)
    assert list(schemas) == ["forecast"]
    assert stream_variables(schemas, base_cfg.output.variables) == {
        "forecast": VARIABLES
    }


def test_deterministic_run_writes_forecast_store(base_cfg) -> None:
    run(base_cfg)
    ds = _open(base_cfg)
    assert set(VARIABLES) <= set(ds.data_vars)
    assert ds["t2m"].dims == ("time", "lead_time", "lat", "lon")
    assert ds["t2m"].shape == (1, 3, len(SMALL_LAT), len(SMALL_LON))
    np.testing.assert_array_equal(
        ds["lead_time"].values, np.arange(3) * np.timedelta64(6, "h")
    )
    assert not np.isnan(ds["t2m"].values).any()


def test_ensemble_run_and_resume(base_cfg) -> None:
    cfg = base_cfg
    with open_dict(cfg):
        cfg.ensemble_size = 3
        cfg.perturbation = {
            "_target_": "earth2studio.perturbation.Gaussian",
            "noise_amplitude": 0.1,
        }
        cfg.start_times = ["2024-01-01 00:00:00", "2024-01-02 00:00:00"]
    run(cfg)
    ds = _open(cfg)
    assert ds["z500"].dims == ("ensemble", "time", "lead_time", "lat", "lon")
    assert ds["z500"].shape[:3] == (3, 2, 3)
    first = ds["z500"].values.copy()
    # Resuming skips every finished item and leaves the store untouched.
    with open_dict(cfg):
        cfg.resume = True
    run(cfg)
    np.testing.assert_array_equal(_open(cfg)["z500"].values, first)


def test_diagnostic_stream_shares_the_store(base_cfg, fake_diagnostic) -> None:
    cfg = base_cfg
    with open_dict(cfg):
        cfg.diagnostics = {
            "diag": {
                "_target_": "test.conftest.FakeDiagnostic",
                "input_variables": ["t2m", "z500"],
                "output_variables": ["diag_a"],
            }
        }
        cfg.output.variables = ["t2m", "diag_a"]
    run(cfg)
    ds = _open(cfg)
    assert set(ds.data_vars) == {"t2m", "diag_a"}
    assert ds["diag_a"].shape == ds["t2m"].shape
    np.testing.assert_array_equal(ds["diag_a"].values, 0.0)


def test_output_regrid_to_coarser_grid(base_cfg) -> None:
    cfg = base_cfg
    with open_dict(cfg):
        cfg.output.regrid = {
            "grid": {
                "lat": {"start": 90.0, "stop": -90.0, "num": 2},
                "lon": {"start": 0.0, "stop": 180.0, "num": 2},
            },
            "method": "nearest",
        }
    run(cfg)
    ds = _open(cfg)
    assert ds["t2m"].shape[-2:] == (2, 2)
    np.testing.assert_array_equal(ds["lat"].values, [90.0, -90.0])

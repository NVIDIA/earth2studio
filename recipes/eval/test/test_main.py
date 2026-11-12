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

import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from main import check_predownload, run
from omegaconf import open_dict
from src.campaign import build_items
from src.data import PredownloadedSource
from src.output import progress_path, store_path, stream_variables
from src.runners import build_runner

from earth2studio.run import DiagnosticRunner, PrognosticRunner
from test.conftest import SMALL_LAT, SMALL_LON, VARIABLES


def _open(cfg) -> xr.Dataset:
    return xr.open_zarr(store_path(cfg), consolidated=False)


def test_build_runner_defaults_to_forecast(base_cfg) -> None:
    runner = build_runner(base_cfg)
    assert isinstance(runner, PrognosticRunner)
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


def test_rerun_refuses_to_replace_an_existing_store(base_cfg) -> None:
    # The shipped default is output.overwrite=true; campaigns that keep their
    # forecasts set it to false.
    with open_dict(base_cfg):
        base_cfg.output.overwrite = False
    run(base_cfg)
    with pytest.raises(FileExistsError, match="already exists"):
        run(base_cfg)
    with open_dict(base_cfg):
        base_cfg.output.overwrite = True
    run(base_cfg)
    assert not np.isnan(_open(base_cfg)["t2m"].values).any()


def test_resume_refuses_to_switch_scoring_mode(base_cfg) -> None:
    run(base_cfg)
    with open_dict(base_cfg):
        base_cfg.scoring.mode = "online"
        base_cfg.resume = True
    with pytest.raises(ValueError, match="does not match"):
        run(base_cfg)


def test_unknown_scoring_mode_is_rejected(base_cfg) -> None:
    # A typo must not downgrade the campaign to the offline path.
    with open_dict(base_cfg):
        base_cfg.scoring.mode = "onlin"
    with pytest.raises(ValueError, match="scoring.mode"):
        run(base_cfg)


def test_resume_refuses_a_missing_store_with_recorded_progress(base_cfg) -> None:
    run(base_cfg)
    shutil.rmtree(store_path(base_cfg))
    with open_dict(base_cfg):
        base_cfg.resume = True
    with pytest.raises(FileNotFoundError, match="progress store records"):
        run(base_cfg)
    # Clearing the progress store lets the run start over.
    shutil.rmtree(progress_path(base_cfg))
    run(base_cfg)
    assert not np.isnan(_open(base_cfg)["t2m"].values).any()


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


class _NoFetchRunner:
    """Runner that declares no fetch at all."""

    step = np.timedelta64(6, "h")

    def data_requests(self, item):
        return ()


class _UnknownFetchRunner(_NoFetchRunner):
    """Runner that cannot describe its fetches."""

    def data_requests(self, item):
        return None


def test_check_predownload_skips_runners_that_fetch_nothing(base_cfg) -> None:
    with open_dict(base_cfg):
        base_cfg.require_predownload = True
    items = build_items(base_cfg, _NoFetchRunner())
    check_predownload(base_cfg, _NoFetchRunner(), items)
    with pytest.raises(RuntimeError, match="sentinel"):
        check_predownload(base_cfg, _UnknownFetchRunner(), items)
    with open_dict(base_cfg):
        base_cfg.require_predownload = False
    check_predownload(base_cfg, _UnknownFetchRunner(), items)


def test_diagnostic_runner_names_every_recommended_source(base_cfg) -> None:
    from predownload import run as predownload_run

    cfg = base_cfg
    source = {
        "_target_": "test.conftest.SeededRandom",
        "domain_coords": {"lat": SMALL_LAT.tolist(), "lon": SMALL_LON.tolist()},
    }
    with open_dict(cfg):
        cfg.runner = "diagnostic"
        cfg.nsteps = 0
        cfg.data_source = None
        cfg.diagnostics = {
            "a": {
                "_target_": "test.conftest.FakeDiagnostic",
                "input_variables": ["t2m"],
                "output_variables": ["diag_a"],
                "source": source,
            },
            "b": {
                "_target_": "test.conftest.FakeDiagnostic",
                "input_variables": ["z500"],
                "output_variables": ["diag_b"],
                "source": source,
            },
        }
        cfg.output.variables = ["diag_a", "diag_b"]
    runner = build_runner(cfg, prefer_predownloaded=False)
    assert isinstance(runner, DiagnosticRunner)
    assert set(runner.source_roles) == {"data_a", "data_b"}
    assert runner.source_roles["data_a"] is runner.sources["a"]
    assert runner.source_roles["data_b"] is runner.sources["b"]

    predownload_run(cfg)
    assert Path(store_path(cfg, "data_a.zarr")).is_dir()
    assert Path(store_path(cfg, "data_b.zarr")).is_dir()
    cached = build_runner(cfg)
    assert isinstance(cached.sources["a"], PredownloadedSource)
    assert isinstance(cached.sources["b"], PredownloadedSource)
    assert set(cached.source_roles) == {"data_a", "data_b"}
    run(cfg)
    ds = _open(cfg)
    assert set(ds.data_vars) == {"diag_a", "diag_b"}

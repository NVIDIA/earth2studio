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

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from src.output import OutputManager, build_forecast_coords
from src.pipelines import ForecastPipeline
from src.work import (
    WorkItem,
    clear_progress,
    filter_completed_items,
    progress_dir,
    write_marker,
)

VARIABLES = ["t2m", "z500"]

_DIST_PATH = "src.output.DistributedManager"
_RANK0_OUTPUT = "src.output.run_on_rank0_first"


def _passthrough(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _make_dist_mock(*, rank=0, world_size=1, distributed=False):
    class _FakeDist:
        def __init__(self):
            self.rank = rank
            self.world_size = world_size
            self.distributed = distributed

    return _FakeDist()


# ---------------------------------------------------------------------------
# Progress tracking (work.py)
# ---------------------------------------------------------------------------


class TestProgressTracking:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return OmegaConf.create({"output": {"path": str(tmp_path / "out")}})

    @pytest.fixture()
    def items(self):
        return [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
            WorkItem(time=np.datetime64("2024-02-01"), ensemble_id=0, seed=1),
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=1, seed=2),
        ]

    def test_progress_dir_path(self, cfg):
        d = progress_dir(cfg)
        assert d.name == ".progress"

    def test_write_marker_creates_file(self, cfg, items):
        write_marker(items[0], cfg)
        d = progress_dir(cfg)
        markers = list(d.glob("*.done"))
        assert len(markers) == 1
        assert markers[0].name == "20240101T000000_ens0.done"

    def test_write_marker_different_ensemble_ids(self, cfg, items):
        write_marker(items[0], cfg)  # 2024-01-01 ens0
        write_marker(items[2], cfg)  # 2024-01-01 ens1
        d = progress_dir(cfg)
        markers = sorted(f.name for f in d.glob("*.done"))
        assert len(markers) == 2
        assert "20240101T000000_ens0.done" in markers
        assert "20240101T000000_ens1.done" in markers

    def test_filter_no_progress_dir(self, cfg, items):
        # No markers exist — all items should be returned.
        result = filter_completed_items(items, cfg)
        assert result == items

    def test_filter_skips_completed(self, cfg, items):
        write_marker(items[0], cfg)
        result = filter_completed_items(items, cfg)
        assert len(result) == 2
        assert items[0] not in result
        assert items[1] in result
        assert items[2] in result

    def test_filter_all_completed(self, cfg, items):
        for item in items:
            write_marker(item, cfg)
        result = filter_completed_items(items, cfg)
        assert result == []

    def test_clear_progress_removes_dir(self, cfg, items):
        write_marker(items[0], cfg)
        d = progress_dir(cfg)
        assert d.exists()

        clear_progress(cfg)
        assert not d.exists()

    def test_clear_progress_noop_if_missing(self, cfg):
        # Should not raise when progress dir doesn't exist.
        clear_progress(cfg)


# ---------------------------------------------------------------------------
# OutputManager resume mode
# ---------------------------------------------------------------------------


class TestOutputManagerResume:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "resume": False,
                "output": {
                    "path": str(tmp_path / "out"),
                    "overwrite": True,
                    "thread_writers": 0,
                    "chunks": {"time": 1, "lead_time": 1},
                },
            }
        )

    @pytest.fixture()
    def resume_cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "resume": True,
                "output": {
                    "path": str(tmp_path / "out"),
                    "overwrite": True,
                    "thread_writers": 0,
                    "chunks": {"time": 1, "lead_time": 1},
                },
            }
        )

    @pytest.fixture()
    def total_coords(self, prognostic):
        times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-02-01")])
        return build_forecast_coords(prognostic, times, nsteps=2)

    def test_resume_opens_existing_store(
        self, cfg, resume_cfg, total_coords, prognostic
    ):
        # First run: create the store.
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    path = mgr._path

        # Resume run: should open without error (not overwrite).
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(resume_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    assert os.path.exists(path)
                    assert "t2m" in mgr.io
                    assert "z500" in mgr.io

    def test_resume_validates_schema(self, cfg, resume_cfg, total_coords, prognostic):
        # Create store with original coords.
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)

        # Resume with different variables — should fail validation.
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(resume_cfg) as mgr:
                    with pytest.raises(ValueError, match="missing from store"):
                        mgr.validate_output_store(total_coords, ["nonexistent_var"])

    def test_resume_creates_store_if_missing(self, resume_cfg, total_coords):
        # First resume run with no existing store — should create it.
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(resume_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    assert os.path.exists(mgr._path)
                    assert "t2m" in mgr.io

    def test_flush_drains_futures(self, prognostic, tmp_path):
        threaded_cfg = OmegaConf.create(
            {
                "resume": True,
                "output": {
                    "path": str(tmp_path / "threaded_out"),
                    "overwrite": True,
                    "thread_writers": 2,
                    "chunks": {"time": 1, "lead_time": 1},
                },
            }
        )
        times = np.array([np.datetime64("2024-01-01")])
        total_coords = build_forecast_coords(prognostic, times, nsteps=2)
        from collections import OrderedDict

        from earth2studio.utils.coords import CoordSystem

        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(threaded_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)

                    n_var = len(VARIABLES)
                    n_lat = len(total_coords["lat"])
                    n_lon = len(total_coords["lon"])

                    lead = np.array([total_coords["lead_time"][0]])
                    write_coords: CoordSystem = OrderedDict(
                        {
                            "time": times,
                            "lead_time": lead,
                            "variable": np.array(VARIABLES),
                            "lat": total_coords["lat"],
                            "lon": total_coords["lon"],
                        }
                    )
                    data = torch.randn(1, 1, n_var, n_lat, n_lon)
                    mgr.write(data, write_coords)
                    assert len(mgr._futures) == 1

                    mgr.flush()
                    assert len(mgr._futures) == 0


# ---------------------------------------------------------------------------
# Pipeline resume integration
# ---------------------------------------------------------------------------


def _make_forecast_pipeline(prognostic, nsteps=2):
    pipeline = ForecastPipeline()
    pipeline.prognostic = prognostic.to(torch.device("cpu"))
    pipeline.diagnostics = []
    pipeline.perturbation = None
    pipeline.nsteps = nsteps
    pipeline._prognostic_ic = prognostic.input_coords()
    pipeline._spatial_ref = prognostic.output_coords(pipeline._prognostic_ic)
    pipeline._dx_input_coords = {}
    return pipeline


class TestPipelineResume:
    @pytest.fixture()
    def resume_cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "project": "test_eval",
                "run_id": "resume_test",
                "start_times": [
                    "2024-01-01 00:00:00",
                    "2024-02-01 00:00:00",
                ],
                "nsteps": 2,
                "ensemble_size": 1,
                "random_seed": 42,
                "resume": True,
                "pipeline": "src.pipelines.forecast.ForecastPipeline",
                "model": {"architecture": "earth2studio.models.px.Persistence"},
                "data_source": {"_target_": "earth2studio.data.Random"},
                "output": {
                    "path": str(tmp_path / "outputs"),
                    "variables": list(VARIABLES),
                    "overwrite": True,
                    "thread_writers": 0,
                    "chunks": {"time": 1, "lead_time": 1},
                },
            }
        )

    def test_markers_written_during_resume_run(
        self, resume_cfg, prognostic, data_source
    ):
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
            WorkItem(time=np.datetime64("2024-02-01"), ensemble_id=0, seed=1),
        ]
        times = np.array([i.time for i in items])
        total_coords = build_forecast_coords(prognostic, times, nsteps=2)
        pipeline = _make_forecast_pipeline(prognostic)

        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(resume_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    pipeline.run(
                        work_items=items,
                        data_source=data_source,
                        output_mgr=mgr,
                        output_variables=VARIABLES,
                        device=torch.device("cpu"),
                        cfg=resume_cfg,
                    )

        # Both items should now have markers.
        d = progress_dir(resume_cfg)
        markers = sorted(f.name for f in d.glob("*.done"))
        assert len(markers) == 2

    def test_no_markers_without_resume(self, resume_cfg, prognostic, data_source):
        resume_cfg.resume = False
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
        ]
        times = np.array([i.time for i in items])
        total_coords = build_forecast_coords(prognostic, times, nsteps=2)
        pipeline = _make_forecast_pipeline(prognostic)

        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(resume_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    pipeline.run(
                        work_items=items,
                        data_source=data_source,
                        output_mgr=mgr,
                        output_variables=VARIABLES,
                        device=torch.device("cpu"),
                        cfg=resume_cfg,
                    )

        d = progress_dir(resume_cfg)
        assert not d.exists()

    def test_resumed_run_skips_completed_items(
        self, resume_cfg, prognostic, data_source
    ):
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
            WorkItem(time=np.datetime64("2024-02-01"), ensemble_id=0, seed=1),
        ]
        times = np.array([i.time for i in items])
        total_coords = build_forecast_coords(prognostic, times, nsteps=2)

        # Simulate first partial run: mark only the first item done.
        write_marker(items[0], resume_cfg)

        # Filter — should only return the second item.
        remaining = filter_completed_items(items, resume_cfg)
        assert len(remaining) == 1
        assert remaining[0].time == np.datetime64("2024-02-01")

        # Run pipeline on remaining items.
        pipeline = _make_forecast_pipeline(prognostic)
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(resume_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    pipeline.run(
                        work_items=remaining,
                        data_source=data_source,
                        output_mgr=mgr,
                        output_variables=VARIABLES,
                        device=torch.device("cpu"),
                        cfg=resume_cfg,
                    )

        # Now both items should be marked done.
        d = progress_dir(resume_cfg)
        markers = sorted(f.name for f in d.glob("*.done"))
        assert len(markers) == 2

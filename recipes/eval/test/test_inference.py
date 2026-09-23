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
from src.output import OutputManager, build_forecast_coords
from src.pipelines import ForecastPipeline
from src.work import WorkItem

_DIST_PATH = "src.output.DistributedManager"
_RANK0_OUTPUT = "src.output.run_on_rank0_first"

VARIABLES = ["t2m", "z500"]


def _passthrough(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _make_dist_mock(*, rank=0, world_size=1, distributed=False):
    class _FakeDist:
        def __init__(self):
            self.rank = rank
            self.world_size = world_size
            self.distributed = distributed

    return _FakeDist()


def _make_forecast_pipeline(prognostic, diagnostics=None, perturbation=None, nsteps=2):
    """Build a ForecastPipeline with pre-set attributes (bypassing setup)."""
    device = torch.device("cpu")
    pipeline = ForecastPipeline()
    pipeline.prognostic = prognostic.to(device)
    pipeline.diagnostics = [dx.to(device) for dx in (diagnostics or [])]
    pipeline.perturbation = perturbation
    pipeline.nsteps = nsteps
    pipeline._prognostic_ic = prognostic.input_coords()
    pipeline._spatial_ref = prognostic.output_coords(pipeline._prognostic_ic)
    pipeline._dx_input_coords = {
        id(dx): dx.input_coords() for dx in pipeline.diagnostics
    }
    return pipeline


class TestForecastPipeline:
    @pytest.fixture()
    def work_items(self):
        return [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
        ]

    @pytest.fixture()
    def pipeline(self, prognostic):
        return _make_forecast_pipeline(prognostic)

    @pytest.fixture()
    def output_mgr(self, base_cfg, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        total_coords = build_forecast_coords(prognostic, times, nsteps=2)
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                mgr = OutputManager(base_cfg)
                mgr.__enter__()
                mgr.validate_output_store(total_coords, VARIABLES)
                yield mgr
                mgr.__exit__(None, None, None)

    def test_deterministic_single_ic(
        self, work_items, pipeline, data_source, output_mgr
    ):
        pipeline.run(
            work_items=work_items,
            data_source=data_source,
            output_mgr=output_mgr,
            output_variables=VARIABLES,
            device=torch.device("cpu"),
        )

        assert os.path.exists(output_mgr._path)
        assert "t2m" in output_mgr.io
        assert "z500" in output_mgr.io

    def test_empty_work_items_skips(self, pipeline, data_source, output_mgr):
        pipeline.run(
            work_items=[],
            data_source=data_source,
            output_mgr=output_mgr,
            output_variables=VARIABLES,
            device=torch.device("cpu"),
        )

    def test_multiple_ics(self, pipeline, data_source, base_cfg):
        times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")])
        total_coords = build_forecast_coords(pipeline.prognostic, times, nsteps=2)
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
            WorkItem(time=np.datetime64("2024-01-02"), ensemble_id=0, seed=1),
        ]
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(base_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    pipeline.run(
                        work_items=items,
                        data_source=data_source,
                        output_mgr=mgr,
                        output_variables=VARIABLES,
                        device=torch.device("cpu"),
                    )

                    np.testing.assert_array_equal(mgr.io.coords["time"], times)
                    for var in ["t2m", "z500"]:
                        arr = mgr.io[var]
                        assert arr.shape[0] == 2, "expected two distinct IC time slices"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_single_gpu_cuda(self, prognostic, data_source, base_cfg):
        times = np.array([np.datetime64("2024-01-01")])
        total_coords = build_forecast_coords(prognostic, times, nsteps=2)
        items = [WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0)]
        pipeline = _make_forecast_pipeline(prognostic)
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(base_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    pipeline.run(
                        work_items=items,
                        data_source=data_source,
                        output_mgr=mgr,
                        output_variables=VARIABLES,
                        device=torch.device("cuda:0"),
                    )
                assert os.path.exists(mgr._path)

    def test_build_total_coords(self, pipeline):
        times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")])
        coords = pipeline.build_total_coords(times, ensemble_size=1)
        assert "ensemble" not in coords
        assert "time" in coords
        assert "lead_time" in coords
        assert len(coords["lead_time"]) == 3  # nsteps=2 → 3 lead times

    def test_build_total_coords_ensemble(self, pipeline):
        times = np.array([np.datetime64("2024-01-01")])
        coords = pipeline.build_total_coords(times, ensemble_size=3)
        assert "ensemble" in coords
        np.testing.assert_array_equal(coords["ensemble"], np.arange(3))

    def test_run_item_yields_correct_steps(self, pipeline, data_source):
        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0)
        steps = list(pipeline.run_item(item, data_source, torch.device("cpu")))
        # nsteps=2 → yields step 0 (analysis) + steps 1..2 = 3 total
        assert len(steps) == 3
        for x, coords in steps:
            assert isinstance(x, torch.Tensor)
            assert "variable" in coords


class TestForecastPipelineEnsemble:
    @pytest.fixture()
    def ensemble_cfg(self, base_cfg):
        cfg = base_cfg.copy()
        cfg.ensemble_size = 3
        return cfg

    @pytest.fixture()
    def ensemble_output_mgr(self, ensemble_cfg, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        total_coords = build_forecast_coords(
            prognostic, times, nsteps=2, ensemble_size=3
        )
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                mgr = OutputManager(ensemble_cfg)
                mgr.__enter__()
                mgr.validate_output_store(total_coords, VARIABLES)
                yield mgr
                mgr.__exit__(None, None, None)

    def test_ensemble_writes_to_correct_members(
        self, prognostic, data_source, ensemble_output_mgr
    ):
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=eid, seed=eid * 100)
            for eid in range(3)
        ]
        pipeline = _make_forecast_pipeline(prognostic)
        pipeline.run(
            work_items=items,
            data_source=data_source,
            output_mgr=ensemble_output_mgr,
            output_variables=VARIABLES,
            device=torch.device("cpu"),
        )

        assert "ensemble" in ensemble_output_mgr.io.coords
        np.testing.assert_array_equal(
            ensemble_output_mgr.io.coords["ensemble"], np.arange(3)
        )

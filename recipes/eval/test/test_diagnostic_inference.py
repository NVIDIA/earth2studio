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
from conftest import FakeDiagnostic
from src.diagnostic_inference import run_diagnostic_inference
from src.output import OutputManager, build_diagnostic_coords
from src.work import WorkItem

_DIST_PATH = "src.output.DistributedManager"
_RANK0_OUTPUT = "src.output.run_on_rank0_first"

DIAG_OUTPUT_VARS = ["diag_a"]


def _passthrough(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _make_dist_mock(*, rank=0, world_size=1, distributed=False):
    class _FakeDist:
        def __init__(self):
            self.rank = rank
            self.world_size = world_size
            self.distributed = distributed

    return _FakeDist()


class TestBuildDiagnosticCoords:
    def test_single_diagnostic(self, fake_diagnostic):
        times = np.array([np.datetime64("2024-01-01")])
        coords = build_diagnostic_coords([fake_diagnostic], times)
        assert "time" in coords
        assert "lead_time" in coords
        assert len(coords["lead_time"]) == 1
        assert coords["lead_time"][0] == np.timedelta64(0, "ns")
        assert "lat" in coords
        assert "lon" in coords
        assert "ensemble" not in coords

    def test_ensemble(self, fake_diagnostic):
        times = np.array([np.datetime64("2024-01-01")])
        coords = build_diagnostic_coords([fake_diagnostic], times, ensemble_size=5)
        assert "ensemble" in coords
        np.testing.assert_array_equal(coords["ensemble"], np.arange(5))

    def test_empty_diagnostics_raises(self):
        times = np.array([np.datetime64("2024-01-01")])
        with pytest.raises(ValueError, match="At least one"):
            build_diagnostic_coords([], times)


class TestRunDiagnosticInference:
    @pytest.fixture()
    def diag_cfg(self, tmp_path):
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "output": {
                    "path": str(tmp_path / "diag_out"),
                    "overwrite": True,
                    "thread_writers": 0,
                    "chunks": {"time": 1, "lead_time": 1},
                },
            }
        )

    @pytest.fixture()
    def diag_output_mgr(self, diag_cfg, fake_diagnostic):
        times = np.array([np.datetime64("2024-01-01")])
        total_coords = build_diagnostic_coords([fake_diagnostic], times)
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                mgr = OutputManager(diag_cfg)
                mgr.__enter__()
                mgr.validate_output_store(total_coords, DIAG_OUTPUT_VARS)
                yield mgr
                mgr.__exit__(None, None, None)

    def test_single_ic(self, fake_diagnostic, data_source, diag_output_mgr):
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
        ]
        run_diagnostic_inference(
            work_items=items,
            diagnostics=[fake_diagnostic],
            data_source=data_source,
            output_mgr=diag_output_mgr,
            output_variables=DIAG_OUTPUT_VARS,
            device=torch.device("cpu"),
        )

        assert os.path.exists(diag_output_mgr._path)
        assert "diag_a" in diag_output_mgr.io

    def test_empty_work_items_skips(self, fake_diagnostic, data_source, diag_output_mgr):
        run_diagnostic_inference(
            work_items=[],
            diagnostics=[fake_diagnostic],
            data_source=data_source,
            output_mgr=diag_output_mgr,
            output_variables=DIAG_OUTPUT_VARS,
            device=torch.device("cpu"),
        )

    def test_multiple_ics(self, fake_diagnostic, data_source, diag_cfg):
        times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")])
        total_coords = build_diagnostic_coords([fake_diagnostic], times)
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
            WorkItem(time=np.datetime64("2024-01-02"), ensemble_id=0, seed=1),
        ]
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(diag_cfg) as mgr:
                    mgr.validate_output_store(total_coords, DIAG_OUTPUT_VARS)
                    run_diagnostic_inference(
                        work_items=items,
                        diagnostics=[fake_diagnostic],
                        data_source=data_source,
                        output_mgr=mgr,
                        output_variables=DIAG_OUTPUT_VARS,
                        device=torch.device("cpu"),
                    )

                    np.testing.assert_array_equal(mgr.io.coords["time"], times)
                    arr = mgr.io["diag_a"]
                    assert arr.shape[0] == 2, "expected two distinct IC time slices"

    def test_ensemble_writes(self, fake_diagnostic, data_source):
        from omegaconf import OmegaConf

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            cfg = OmegaConf.create(
                {
                    "output": {
                        "path": tmp,
                        "overwrite": True,
                        "thread_writers": 0,
                        "chunks": {"time": 1, "lead_time": 1},
                    },
                }
            )
            times = np.array([np.datetime64("2024-01-01")])
            total_coords = build_diagnostic_coords(
                [fake_diagnostic], times, ensemble_size=3
            )
            items = [
                WorkItem(
                    time=np.datetime64("2024-01-01"), ensemble_id=eid, seed=eid * 100
                )
                for eid in range(3)
            ]
            with patch(_DIST_PATH, return_value=_make_dist_mock()):
                with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                    with OutputManager(cfg) as mgr:
                        mgr.validate_output_store(total_coords, DIAG_OUTPUT_VARS)
                        run_diagnostic_inference(
                            work_items=items,
                            diagnostics=[fake_diagnostic],
                            data_source=data_source,
                            output_mgr=mgr,
                            output_variables=DIAG_OUTPUT_VARS,
                            device=torch.device("cpu"),
                        )

                        assert "ensemble" in mgr.io.coords
                        np.testing.assert_array_equal(
                            mgr.io.coords["ensemble"], np.arange(3)
                        )

    def test_no_diagnostics_raises(self, data_source, diag_output_mgr):
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
        ]
        with pytest.raises(ValueError, match="At least one"):
            run_diagnostic_inference(
                work_items=items,
                diagnostics=[],
                data_source=data_source,
                output_mgr=diag_output_mgr,
                output_variables=DIAG_OUTPUT_VARS,
                device=torch.device("cpu"),
            )

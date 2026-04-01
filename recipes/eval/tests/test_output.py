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
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from earth2studio.utils.coords import CoordSystem

from src.output import OutputManager


def _make_dist_mock(*, rank: int = 0, world_size: int = 1, distributed: bool = False):
    class _FakeDist:
        def __init__(self):
            self.rank = rank
            self.world_size = world_size
            self.distributed = distributed

    return _FakeDist()


_DIST_PATH = "src.output.DistributedManager"
_RANK0_PATH = "src.output.run_on_rank0_first"


class TestOutputManager:
    @pytest.fixture()
    def cfg(self, tmp_path, prognostic):
        return OmegaConf.create(
            {
                "output": {
                    "path": str(tmp_path / "out"),
                    "variables": ["t2m", "z500"],
                    "overwrite": True,
                    "thread_writers": 0,
                    "chunks": {"time": 1, "lead_time": 1},
                },
            }
        )

    def test_creates_zarr_store(self, cfg, prognostic, tmp_path):
        times = np.array([np.datetime64("2024-01-01")])
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                mgr = OutputManager(cfg, prognostic=prognostic, times=times, nsteps=2)
                with mgr:
                    store_path = mgr._path
                    assert os.path.exists(store_path)
                    assert "t2m" in mgr.io
                    assert "z500" in mgr.io

    def test_output_coords_contain_expected_dims(self, cfg, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            mgr = OutputManager(cfg, prognostic=prognostic, times=times, nsteps=2)
            oc = mgr.output_coords
            assert "variable" in oc
            assert "lat" in oc
            assert "lon" in oc
            np.testing.assert_array_equal(oc["variable"], ["t2m", "z500"])

    def test_ensemble_coord_added_when_ensemble_gt1(self, cfg, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            mgr = OutputManager(
                cfg, prognostic=prognostic, times=times, nsteps=2, ensemble_size=3
            )
            assert "ensemble" in mgr._total_coords
            np.testing.assert_array_equal(
                mgr._total_coords["ensemble"], np.arange(3)
            )

    def test_no_ensemble_coord_for_deterministic(self, cfg, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            mgr = OutputManager(
                cfg, prognostic=prognostic, times=times, nsteps=2, ensemble_size=1
            )
            assert "ensemble" not in mgr._total_coords

    def test_overwrite_removes_existing_store(self, cfg, prognostic, tmp_path):
        times = np.array([np.datetime64("2024-01-01")])
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                mgr = OutputManager(
                    cfg, prognostic=prognostic, times=times, nsteps=2
                )
                with mgr:
                    pass

                # Opening again with overwrite=True should succeed
                mgr2 = OutputManager(
                    cfg, prognostic=prognostic, times=times, nsteps=2
                )
                with mgr2:
                    assert os.path.exists(mgr2._path)

    def test_error_without_context_manager(self, cfg, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            mgr = OutputManager(
                cfg, prognostic=prognostic, times=times, nsteps=2
            )
            with pytest.raises(RuntimeError, match="context manager"):
                _ = mgr.io

    def test_lead_time_coord_length(self, cfg, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        nsteps = 5
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            mgr = OutputManager(
                cfg, prognostic=prognostic, times=times, nsteps=nsteps
            )
            # nsteps+1 lead times (step 0 = analysis, steps 1..nsteps = forecast)
            assert len(mgr._total_coords["lead_time"]) == nsteps + 1

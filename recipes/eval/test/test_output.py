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
from src.output import OutputManager, build_forecast_coords

from earth2studio.utils.coords import CoordSystem


def _make_dist_mock(*, rank: int = 0, world_size: int = 1, distributed: bool = False):
    class _FakeDist:
        def __init__(self):
            self.rank = rank
            self.world_size = world_size
            self.distributed = distributed

    return _FakeDist()


_DIST_PATH = "src.output.DistributedManager"
_RANK0_PATH = "src.output.run_on_rank0_first"

VARIABLES = ["t2m", "z500"]


class TestBuildForecastCoords:
    def test_deterministic_no_ensemble(self, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        coords = build_forecast_coords(prognostic, times, nsteps=2, ensemble_size=1)
        assert "ensemble" not in coords
        assert "time" in coords
        assert "lead_time" in coords
        # nsteps+1 lead times (step 0 = analysis, steps 1..nsteps = forecast)
        assert len(coords["lead_time"]) == 3

    def test_ensemble_coord_added(self, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        coords = build_forecast_coords(prognostic, times, nsteps=2, ensemble_size=3)
        assert "ensemble" in coords
        np.testing.assert_array_equal(coords["ensemble"], np.arange(3))

    def test_lead_time_length(self, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        coords = build_forecast_coords(prognostic, times, nsteps=5)
        assert len(coords["lead_time"]) == 6

    def test_spatial_dims_from_model(self, prognostic):
        times = np.array([np.datetime64("2024-01-01")])
        coords = build_forecast_coords(prognostic, times, nsteps=1)
        assert "lat" in coords
        assert "lon" in coords


class TestOutputManager:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return OmegaConf.create(
            {
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
        times = np.array([np.datetime64("2024-01-01")])
        return build_forecast_coords(prognostic, times, nsteps=2)

    def test_creates_zarr_store(self, cfg, total_coords, tmp_path):
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                with OutputManager(cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    assert os.path.exists(mgr._path)
                    assert "t2m" in mgr.io
                    assert "z500" in mgr.io

    def test_error_without_validate(self, cfg):
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            mgr = OutputManager(cfg)
            with pytest.raises(RuntimeError, match="validate_output_store"):
                _ = mgr.io

    def test_overwrite_removes_existing_store(self, cfg, total_coords, tmp_path):
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                with OutputManager(cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)

                # Opening again with overwrite=True should succeed
                with OutputManager(cfg) as mgr2:
                    mgr2.validate_output_store(total_coords, VARIABLES)
                    assert os.path.exists(mgr2._path)

    def test_overwrite_false_raises_when_store_exists(
        self, prognostic, total_coords, tmp_path
    ):
        no_overwrite_cfg = OmegaConf.create(
            {
                "output": {
                    "path": str(tmp_path / "out"),
                    "overwrite": False,
                    "thread_writers": 0,
                    "chunks": {"time": 1, "lead_time": 1},
                },
            }
        )
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                with OutputManager(no_overwrite_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)

                with OutputManager(no_overwrite_cfg) as mgr2:
                    with pytest.raises(FileExistsError, match="already exists"):
                        mgr2.validate_output_store(total_coords, VARIABLES)

    def test_threaded_writes_produce_same_output(self, prognostic, tmp_path):
        threaded_cfg = OmegaConf.create(
            {
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
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                with OutputManager(threaded_cfg) as mgr:
                    mgr.validate_output_store(total_coords, VARIABLES)
                    assert mgr._executor is not None
                    assert mgr._executor._max_workers == 2

                    n_var = len(VARIABLES)
                    n_lat = len(total_coords["lat"])
                    n_lon = len(total_coords["lon"])

                    for step in range(3):
                        lead = np.array([total_coords["lead_time"][step]])
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

                assert os.path.exists(mgr._path)
                assert "t2m" in mgr.io
                assert "z500" in mgr.io

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

from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from src.pipelines import (
    DiagnosticPipeline,
    ForecastPipeline,
    Pipeline,
    build_pipeline,
)
from src.work import WorkItem

from earth2studio.utils.coords import CoordSystem


class _StubPipeline(Pipeline):
    """Minimal concrete pipeline for testing the ABC and registry."""

    def setup(self, cfg, device):
        self._spatial_ref = OrderedDict(
            {"lat": np.linspace(90, -90, 4), "lon": np.linspace(0, 360, 8)}
        )

    def build_total_coords(self, times, ensemble_size):
        total: CoordSystem = OrderedDict()
        if ensemble_size > 1:
            total["ensemble"] = np.arange(ensemble_size)
        total["time"] = times
        total["lead_time"] = np.array([np.timedelta64(0, "ns")])
        total["lat"] = self._spatial_ref["lat"]
        total["lon"] = self._spatial_ref["lon"]
        return total

    def run_item(self, item, data_source, device):
        x = torch.zeros(1, 1, 1, 4, 8)
        coords: CoordSystem = OrderedDict(
            {
                "time": np.array([item.time]),
                "lead_time": np.array([np.timedelta64(0, "ns")]),
                "variable": np.array(["dummy"]),
                "lat": self._spatial_ref["lat"],
                "lon": self._spatial_ref["lon"],
            }
        )
        yield x, coords


class TestBuildPipeline:
    def test_missing_pipeline_raises(self):
        cfg = OmegaConf.create({})
        with pytest.raises(ValueError, match="cfg.pipeline is required"):
            build_pipeline(cfg)

    def test_forecast_by_fqn(self):
        cfg = OmegaConf.create(
            {"pipeline": "src.pipelines.forecast.ForecastPipeline"}
        )
        pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, ForecastPipeline)

    def test_diagnostic_by_fqn(self):
        cfg = OmegaConf.create(
            {"pipeline": "src.pipelines.forecast.DiagnosticPipeline"}
        )
        pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, DiagnosticPipeline)

    def test_custom_pipeline_by_fqn(self):
        cfg = OmegaConf.create({"pipeline": "my_custom.module.MyPipeline"})
        with patch("src.pipelines.hydra.utils.get_class", return_value=_StubPipeline):
            pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, _StubPipeline)

    def test_non_pipeline_class_raises(self):
        cfg = OmegaConf.create({"pipeline": "builtins.dict"})
        with pytest.raises(TypeError, match="subclass of Pipeline"):
            build_pipeline(cfg)

    def test_target_block(self):
        cfg = OmegaConf.create(
            {
                "pipeline": {
                    "_target_": "src.pipelines.forecast.ForecastPipeline"
                }
            }
        )
        pipeline = build_pipeline(cfg)
        assert isinstance(pipeline, ForecastPipeline)

    def test_target_block_non_pipeline_raises(self):
        cfg = OmegaConf.create(
            {"pipeline": {"_target_": "builtins.dict"}}
        )
        with pytest.raises(TypeError, match="not a Pipeline subclass"):
            build_pipeline(cfg)


class TestPipelineABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Pipeline()

    def test_stub_pipeline_run_item(self):
        pipeline = _StubPipeline()
        pipeline.setup(OmegaConf.create({}), torch.device("cpu"))

        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0)
        results = list(pipeline.run_item(item, None, torch.device("cpu")))
        assert len(results) == 1
        x, coords = results[0]
        assert x.shape == (1, 1, 1, 4, 8)
        assert "variable" in coords

    def test_stub_pipeline_build_total_coords(self):
        pipeline = _StubPipeline()
        pipeline.setup(OmegaConf.create({}), torch.device("cpu"))

        times = np.array([np.datetime64("2024-01-01")])
        coords = pipeline.build_total_coords(times, ensemble_size=1)
        assert "ensemble" not in coords
        assert "time" in coords
        assert "lead_time" in coords

        coords_ens = pipeline.build_total_coords(times, ensemble_size=5)
        assert "ensemble" in coords_ens
        np.testing.assert_array_equal(coords_ens["ensemble"], np.arange(5))

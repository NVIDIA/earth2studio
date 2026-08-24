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
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch
from src.output import OutputManager, build_diagnostic_coords
from src.pipelines import DiagnosticPipeline
from src.pipelines.diagnostic import _spatial_ref_from_output_coords
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


def _make_diagnostic_pipeline(diagnostics):
    """Build a DiagnosticPipeline with pre-set attributes (bypassing setup)."""
    device = torch.device("cpu")
    pipeline = DiagnosticPipeline()
    pipeline.diagnostics = [dx.to(device) for dx in diagnostics]
    pipeline._dx_input_coords = {
        id(dx): dx.input_coords() for dx in pipeline.diagnostics
    }

    all_input_vars: list[str] = []
    seen: set[str] = set()
    for dx in pipeline.diagnostics:
        for v in pipeline._dx_input_coords[id(dx)]["variable"]:
            if v not in seen:
                all_input_vars.append(str(v))
                seen.add(str(v))
    pipeline._all_input_vars = all_input_vars

    dx0 = pipeline.diagnostics[0]
    pipeline._spatial_ref = _spatial_ref_from_output_coords(
        dx0.output_coords(pipeline._dx_input_coords[id(dx0)])
    )
    pipeline._zero_lead = np.array([np.timedelta64(0, "ns")])
    return pipeline


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


class TestDiagnosticPipeline:
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
    def pipeline(self, fake_diagnostic):
        return _make_diagnostic_pipeline([fake_diagnostic])

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

    def test_single_ic(self, pipeline, data_source, diag_output_mgr):
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
        ]
        pipeline.run(
            work_items=items,
            data_source=data_source,
            output_mgr=diag_output_mgr,
            output_variables=DIAG_OUTPUT_VARS,
            device=torch.device("cpu"),
        )

        assert os.path.exists(diag_output_mgr._path)
        assert "diag_a" in diag_output_mgr.io

    def test_empty_work_items_skips(self, pipeline, data_source, diag_output_mgr):
        pipeline.run(
            work_items=[],
            data_source=data_source,
            output_mgr=diag_output_mgr,
            output_variables=DIAG_OUTPUT_VARS,
            device=torch.device("cpu"),
        )

    def test_multiple_ics(self, pipeline, data_source, diag_cfg):
        times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")])
        total_coords = build_diagnostic_coords(pipeline.diagnostics, times)
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0),
            WorkItem(time=np.datetime64("2024-01-02"), ensemble_id=0, seed=1),
        ]
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(diag_cfg) as mgr:
                    mgr.validate_output_store(total_coords, DIAG_OUTPUT_VARS)
                    pipeline.run(
                        work_items=items,
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
            pipeline = _make_diagnostic_pipeline([fake_diagnostic])
            with patch(_DIST_PATH, return_value=_make_dist_mock()):
                with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                    with OutputManager(cfg) as mgr:
                        mgr.validate_output_store(total_coords, DIAG_OUTPUT_VARS)
                        pipeline.run(
                            work_items=items,
                            data_source=data_source,
                            output_mgr=mgr,
                            output_variables=DIAG_OUTPUT_VARS,
                            device=torch.device("cpu"),
                        )

                        assert "ensemble" in mgr.io.coords
                        np.testing.assert_array_equal(
                            mgr.io.coords["ensemble"], np.arange(3)
                        )

    def test_run_item_yields_single_output(self, pipeline, data_source):
        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0)
        steps = list(pipeline.run_item(item, data_source, torch.device("cpu")))
        assert len(steps) == 1
        x, coords = steps[0]
        assert isinstance(x, torch.Tensor)
        assert "variable" in coords

    def test_build_total_coords(self, pipeline):
        times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")])
        coords = pipeline.build_total_coords(times, ensemble_size=1)
        assert "ensemble" not in coords
        assert "time" in coords
        assert "lead_time" in coords
        assert len(coords["lead_time"]) == 1
        assert coords["lead_time"][0] == np.timedelta64(0, "ns")


# ---------------------------------------------------------------------------
# Generative diagnostics (phase 3): sample -> ensemble
# ---------------------------------------------------------------------------
#
# A generative diagnostic (e.g. CorrDiff) emits its own leading 'sample'
# axis instead of expressing ensemble draws through the pipeline's own
# machinery.  DiagnosticPipeline must: (1) strip 'sample' from the static
# spatial reference so it isn't classified as a bogus spatial dim, (2) set
# number_of_samples from the member block size (via seed_member), and (3)
# rename each call's 'sample' axis to 'ensemble', carrying this rank's
# *global* member ids.


class TestGenerativeDiagnosticSampleAxis:
    def test_spatial_ref_excludes_sample(self, fake_generative_diagnostic):
        pipeline = _make_diagnostic_pipeline([fake_generative_diagnostic])
        assert "sample" not in pipeline._spatial_ref
        assert "lat" in pipeline._spatial_ref
        assert "lon" in pipeline._spatial_ref

    def test_run_item_renames_sample_to_this_members_ensemble_id(
        self, fake_generative_diagnostic, data_source
    ):
        pipeline = _make_diagnostic_pipeline([fake_generative_diagnostic])
        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=3, seed=7)

        x, coords = next(
            iter(pipeline.run_item(item, data_source, torch.device("cpu")))
        )

        # seed_member set number_of_samples from _members_per_rank (default
        # 1, since run_item is unbatched -- the G=M, K=1 layout).
        assert fake_generative_diagnostic.number_of_samples == 1
        assert "sample" not in coords
        assert "ensemble" in coords
        np.testing.assert_array_equal(coords["ensemble"], np.array([3]))
        assert x.shape[list(coords).index("ensemble")] == 1

    def test_run_item_batched_renames_sample_to_the_batchs_member_ids(
        self, fake_generative_diagnostic, data_source
    ):
        pipeline = _make_diagnostic_pipeline([fake_generative_diagnostic])
        items = [
            WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=eid, seed=eid)
            for eid in (2, 3, 4)
        ]
        # Normally set by Pipeline.run() from its member_batch argument;
        # set directly here since the test drives run_item_batched itself.
        pipeline._members_per_rank = len(items)

        x, coords = next(
            iter(pipeline.run_item_batched(items, data_source, torch.device("cpu")))
        )

        assert fake_generative_diagnostic.number_of_samples == 3
        assert "sample" not in coords
        assert "ensemble" in coords
        np.testing.assert_array_equal(coords["ensemble"], np.array([2, 3, 4]))
        assert x.shape[list(coords).index("ensemble")] == 3

    def test_deterministic_and_generative_diagnostics_align(
        self, fake_generative_diagnostic, fake_diagnostic, data_source
    ):
        """Mixing a sample-bearing diagnostic with a deterministic one must
        broadcast the deterministic output (and the raw fetched input)
        onto the same ensemble axis before cat_coords -- which requires
        every operand to carry identical dim names in identical order --
        rather than erroring on the mismatch.
        """
        pipeline = _make_diagnostic_pipeline(
            [fake_generative_diagnostic, fake_diagnostic]
        )
        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=1, seed=1)

        x, coords = next(
            iter(pipeline.run_item(item, data_source, torch.device("cpu")))
        )

        assert "ensemble" in coords
        np.testing.assert_array_equal(coords["ensemble"], np.array([1]))
        variables = [str(v) for v in coords["variable"]]
        assert "sample_out" in variables, "generative diagnostic's own output"
        assert "diag_a" in variables, "deterministic diagnostic's output"
        assert "t2m" in variables, "raw fetched input passed through"

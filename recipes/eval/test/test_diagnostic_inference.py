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
        for v in pipeline._dx_input_coords[id(dx)].coords["variable"].values:
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
    def test_taiwan_example_writes_geographic_arrays(self):
        import ast
        from collections import OrderedDict
        from datetime import datetime
        from pathlib import Path

        import xarray as xr
        from loguru import logger

        from earth2studio.data import Constant, DataSource
        from earth2studio.grids import CurvilinearGrid
        from earth2studio.io import IOBackend, ZarrBackend
        from earth2studio.models.dx import CorrDiffTaiwan
        from earth2studio.utils.coords import coord_array, split_coords
        from earth2studio.utils.time import to_time_array

        lat, lon = np.meshgrid([20.0, 21.0], [120.0, 121.0, 122.0], indexing="ij")

        class RegionalDiagnostic:
            number_of_samples = 1

            def to(self, device):
                return self

            def input_coords(self):
                return coord_array(
                    ("batch", "variable", "lat", "lon"),
                    {
                        "variable": ["t2m"],
                        "lat": [20.0, 21.0],
                        "lon": [120.0, 121.0, 122.0],
                    },
                    dynamic=("batch",),
                )

            def output_coords(self, x):
                return coord_array(
                    ("time", "sample", "variable", "y", "x"),
                    {"time": x.time, "sample": [0], "variable": ["t2m"]},
                    grid=CurvilinearGrid(lat, lon),
                )

            def __call__(self, x):
                signature = self.output_coords(x)
                return xr.DataArray(
                    np.full(signature.shape, 7.0),
                    dims=signature.dims,
                    coords=signature.coords,
                )

        path = (
            Path(__file__).parents[3]
            / "examples/03_downscaling/01_corrdiff_inference.py"
        )
        tree = ast.parse(path.read_text())
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run"
        )
        namespace = dict(
            np=np,
            torch=torch,
            datetime=datetime,
            logger=logger,
            DataSource=DataSource,
            IOBackend=IOBackend,
            CorrDiffTaiwan=CorrDiffTaiwan,
            split_coords=split_coords,
            to_time_array=to_time_array,
        )
        exec(  # noqa: S102 - execute the repository example without downloading weights
            compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        io = namespace["run"](
            ["2024-01-01"],
            RegionalDiagnostic(),
            Constant(
                OrderedDict(
                    lat=np.array([20.0, 21.0]), lon=np.array([120.0, 121.0, 122.0])
                ),
                7,
            ),
            ZarrBackend(),
        )
        np.testing.assert_array_equal(io["lat"][:], lat)
        np.testing.assert_array_equal(io["lon"][:], lon)
        np.testing.assert_array_equal(io["t2m"][0, 0], 7)

    def test_cosmo_example_rollout_skips_initial_and_selects_labels(self):
        import ast
        from collections import OrderedDict
        from pathlib import Path

        from earth2studio.models.px import DiagnosticWrapper, Persistence
        from earth2studio.utils.coords import coord_array_like
        from earth2studio.utils.cupy import from_torch

        px = Persistence(
            ["t2m", "u10m"], OrderedDict(lat=np.arange(2), lon=np.arange(3))
        )
        signature = coord_array_like(px.input_coords(), {"batch": [0]}).isel(
            batch=0, drop=True
        )
        x = from_torch(torch.ones(signature.shape), signature).expand_dims(
            time=[np.datetime64("2024-01-01")]
        )

        class Diagnostic(torch.nn.Module):
            def input_coords(self):
                return px.input_coords()

            def __call__(self, field):
                return (
                    field.sel(variable=["t2m"])
                    .isel(lead_time=0, drop=True)
                    .assign_coords(time=field.time.values + field.lead_time.values[-1])
                    .rename(lat="y", lon="x")
                    .expand_dims(sample=[0])
                    .transpose("variable", "sample", "time", "y", "x")
                    + 280
                )

        wrapped = DiagnosticWrapper(
            px,
            Diagnostic(),
            prepare_dx_input_tensor=lambda field, coords: field,
            prepare_output_tensor=lambda field, outputs: outputs[0],
        )
        path = (
            Path(__file__).parents[3]
            / "examples/03_downscaling/04_cosmo_rea_downscaling.py"
        )
        nodes = ast.parse(path.read_text()).body
        start = next(
            i
            for i, n in enumerate(nodes)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "frames" for t in n.targets)
        )
        end = next(
            i
            for i, n in enumerate(nodes[start:], start)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "leads" for t in n.targets)
        )
        namespace = dict(
            wrapped=wrapped,
            x=x,
            np=np,
            lead_hours=12,
            dt_hours=6,
            ov=["T_2M"],
            init_time=np.datetime64("2024-01-01"),
        )
        exec(  # noqa: S102 - exercise the repository's rollout cell with real wrapper
            compile(
                ast.Module(body=nodes[start:end], type_ignores=[]), str(path), "exec"
            ),
            namespace,
        )
        assert list(namespace["frames"]) == [6, 12]
        for frame in namespace["frames"].values():
            assert frame.shape == (2, 3)
            np.testing.assert_allclose(frame, 7.85, atol=1e-4)

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
            for eid in (2, 3, 4, 5)
        ]
        # Normally set by Pipeline.run() from its member_batch argument;
        # set directly here since the test drives run_item_batched itself.
        pipeline._members_per_rank = len(items)

        x, coords = next(
            iter(pipeline.run_item_batched(items, data_source, torch.device("cpu")))
        )

        assert fake_generative_diagnostic.rng_calls == [(2, True)]
        assert fake_generative_diagnostic.number_of_samples == 4
        assert "sample" not in coords
        assert "ensemble" in coords
        np.testing.assert_array_equal(coords["ensemble"], np.array([2, 3, 4, 5]))
        assert x.shape[list(coords).index("ensemble")] == 4

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

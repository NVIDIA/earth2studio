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

"""Output-only variables at lead time 0, and resume tags."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from loguru import logger
from test_runner import _model, _Zeros
from test_runner_models import T0, _FusedDLESyMShaped, _GridSource

from earth2studio.io import KVBackend
from earth2studio.run import Pipeline, PrognosticRunner, ProgressStore, work_items
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

H12 = np.timedelta64(12, "h")


class _DoubleTp:
    """Diagnostic reading the fused model's output-only ``tp06``."""

    def __init__(self, grid: object) -> None:
        self.grid = grid

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "variable", "face", "height", "width"),
            {"variable": ["tp06"]},
            dynamic=("batch",),
            grid=self.grid,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(input_coords, {"variable": ["tp06x2"]})

    def to(self, device: torch.device) -> "_DoubleTp":
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        return (2 * x).assign_coords(variable=["tp06x2"])


class _Fused(_FusedDLESyMShaped):
    def to(self, device: torch.device) -> "_Fused":
        return self


def _fused_runner(**kwargs: object) -> PrognosticRunner:
    model = _Fused()
    source = _GridSource(
        model.input_coords(), model.atmos_variables + model.ocean_variables, fill=0.0
    )
    return PrognosticRunner(model, source, **kwargs)  # type: ignore[arg-type]


class _SeenVariables:
    """Scorer recording the variables of every field it receives."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []

    def begin_item(self, item: object) -> None:
        pass

    def update(self, item: object, stream: str, x: xr.DataArray) -> None:
        self.seen.append([str(v) for v in x["variable"].values])

    def finish_item(self, item: object) -> None:
        pass


@pytest.mark.parametrize("names", [["tp06", "z500"], ["tp06"]])
def test_selection_keeps_the_variables_each_step_holds(names: list[str]) -> None:
    """``output_variables`` may name variables the initial condition lacks.

    Each step writes and scores the selected variables it holds; a step holding
    none is skipped, and the store keeps its fill value at lead time 0.
    """
    store, scorer = KVBackend(), _SeenVariables()
    Pipeline(
        _fused_runner(),
        output={"forecast": store},
        scorer=scorer,
        output_variables={"forecast": names},
        device="cpu",
    ).run(work_items([T0], H12))
    assert "sst" not in store
    assert np.isnan(store["tp06"][0, 0]).all()
    assert not np.isnan(store["tp06"][0, 1:]).any()
    if "z500" in names:
        assert not np.isnan(store["z500"][:]).any()
        assert scorer.seen == [["z500"], names]
    else:
        assert scorer.seen == [["tp06"]]


def test_diagnostic_on_output_only_variables_starts_at_first_forecast() -> None:
    runner = _fused_runner(diagnostics={"tp": _DoubleTp(_FusedDLESyMShaped().grid)})
    coords = runner.output_coords(H12)
    forecast_leads = coords["forecast"]["lead_time"].values
    np.testing.assert_array_equal(coords["tp"]["lead_time"].values, forecast_leads[1:])
    store = KVBackend()
    Pipeline(runner, output={"tp": store}, device="cpu").run(work_items([T0], H12))
    assert not np.isnan(store["tp06x2"][:]).any()


def test_lead_zero_only_run_skips_unscheduled_diagnostic_store() -> None:
    """A diagnostic that starts at the first forecast has no store at lead 0 only."""
    runner = _fused_runner(diagnostics={"tp": _DoubleTp(_FusedDLESyMShaped().grid)})
    assert runner.output_coords(np.timedelta64(0, "h"))["tp"].sizes["lead_time"] == 0
    forecast, tp = KVBackend(), KVBackend()
    Pipeline(runner, output={"forecast": forecast, "tp": tp}, device="cpu").run(
        work_items([T0], np.timedelta64(0, "h"))
    )
    assert "z500" in forecast
    assert "tp06x2" not in tp


@pytest.fixture
def warnings_log() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(message), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


@pytest.mark.parametrize("tags", [None, {"model": "tiny", "checkpoint": "v1"}])
def test_resume_without_tags_warns(
    tmp_path: Path, warnings_log: list[str], tags: dict | None
) -> None:
    items = work_items([T0], H12)
    for _ in range(2):
        Pipeline(
            PrognosticRunner(_model(), _Zeros()),  # type: ignore[arg-type]
            output={"forecast": KVBackend()},
            progress=ProgressStore(tmp_path / "progress", tags=tags),
            device="cpu",
        ).run(items)
    warned = any("has no tags" in message for message in warnings_log)
    assert warned is (tags is None)

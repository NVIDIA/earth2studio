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

from earth2studio.io import KVBackend, ZarrBackend
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


def test_initial_condition_variables_lists_partial_streams() -> None:
    assert _fused_runner().initial_condition_variables() == {
        "forecast": ("z500", "sst")
    }
    # Outputs equal inputs: the initial condition is complete.
    assert PrognosticRunner(_model(), _Zeros()).initial_condition_variables() == {}  # type: ignore[arg-type]


def test_selecting_only_output_only_variables_fails_before_stores(
    tmp_path: Path,
) -> None:
    path = tmp_path / "forecast.zarr"
    pipeline = Pipeline(
        _fused_runner(),
        output={"forecast": ZarrBackend(str(path))},
        output_variables={"forecast": ["tp06"]},
        device="cpu",
    )
    with pytest.raises(ValueError, match="lead time 0"):
        pipeline.run(work_items([T0], H12))
    store = ZarrBackend(str(path))
    assert "tp06" not in store


def test_partial_selection_keeps_nan_at_lead_zero() -> None:
    store = KVBackend()
    Pipeline(
        _fused_runner(),
        output={"forecast": store},
        output_variables={"forecast": ["tp06", "z500"]},
        device="cpu",
    ).run(work_items([T0], H12))
    assert np.isnan(store["tp06"][0, 0]).all()
    assert not np.isnan(store["tp06"][0, 1:]).any()
    assert not np.isnan(store["z500"][:]).any()


def test_diagnostic_on_output_only_variables_starts_at_first_forecast() -> None:
    runner = _fused_runner(diagnostics={"tp": _DoubleTp(_FusedDLESyMShaped().grid)})
    coords = runner.output_coords(H12)
    forecast_leads = coords["forecast"]["lead_time"].values
    np.testing.assert_array_equal(coords["tp"]["lead_time"].values, forecast_leads[1:])
    store = KVBackend()
    Pipeline(runner, output={"tp": store}, device="cpu").run(work_items([T0], H12))
    assert not np.isnan(store["tp06x2"][:]).any()


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

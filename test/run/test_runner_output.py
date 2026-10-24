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

"""Runner stream signatures drive the output manager.

Covers the boundary between the runner's planning signatures and
``OutputManager``, which neither side can exercise alone: a run's schemas plan
every store, each stream lands in its own backend, and a stream without a
backend is simply not written. The first test drives these layers through
``Pipeline``; the others drive the loop by hand, so a failure points at one of
the two layers.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_runner import _model, _Zeros

from earth2studio.io import KVBackend, ZarrBackend
from earth2studio.models.dx import Identity
from earth2studio.run.output import OutputManager
from earth2studio.run.pipeline import Pipeline
from earth2studio.run.runner import PrognosticRunner
from earth2studio.run.work import work_items

T0 = np.datetime64("2026-01-01T00")
T1 = np.datetime64("2026-01-02T00")


def _runner() -> PrognosticRunner:
    return PrognosticRunner(_model(), _Zeros(), diagnostics={"copy": Identity()})


def test_signatures_plan_one_store_per_stream(tmp_path) -> None:
    """Schemas from every item plan both stores; each stream writes its own.

    Driven through ``Pipeline``, which plans the stores from these schemas.
    """
    runner = _runner()
    items = work_items([T0, T1], horizon=np.timedelta64(12, "h"))
    schemas = runner.output_coords(max(item.horizon for item in items))
    assert set(schemas) == {"forecast", "copy"}
    assert schemas["forecast"].nbytes == 0  # allocation-free signature

    forecast, copy = KVBackend(), ZarrBackend(str(tmp_path / "copy.zarr"))
    Pipeline(runner, output={"forecast": forecast, "copy": copy}, device="cpu").run(
        items
    )
    assert copy._closed  # closed by the pipeline, and still readable

    # Stores span all items: both times and every lead the horizon allows.
    assert forecast["u10m"].shape == (2, 3, 2, 3)  # time, lead_time, lat, lon
    assert copy["u10m"].shape == (2, 3, 2, 3)
    # Step k holds the value k, written at the item's own time.
    np.testing.assert_array_equal(forecast["u10m"][0, :, 0, 0], np.arange(3))
    np.testing.assert_array_equal(forecast["u10m"][1, :, 0, 0], np.arange(3))
    np.testing.assert_array_equal(copy["u10m"][0, :, 0, 0], np.arange(3))


def test_stream_without_a_backend_is_not_written(tmp_path) -> None:
    """An unmapped stream is skipped, so Pipeline._emit needs no check."""
    runner = _runner()
    items = work_items([T0], horizon=np.timedelta64(6, "h"))
    schemas = runner.output_coords(items[0].horizon)

    copy = ZarrBackend(str(tmp_path / "copy_only.zarr"))
    outputs = OutputManager({"copy": copy})
    outputs.prepare(schemas, items)
    for step in runner.run_item(items[0]):
        for stream, x in step.items():
            outputs.write(stream, x)
    outputs.close()

    assert "u10m" in copy  # the mapped stream landed
    assert copy["u10m"].shape == (1, 2, 2, 3)


def test_backend_naming_an_unknown_stream_is_rejected() -> None:
    """A typo in the output mapping fails before any store is created."""
    runner = _runner()
    items = work_items([T0], horizon=np.timedelta64(6, "h"))
    schemas = runner.output_coords(items[0].horizon)
    outputs = OutputManager({"forcast": KVBackend()})
    with pytest.raises(ValueError, match="unknown streams"):
        outputs.prepare(schemas, items)

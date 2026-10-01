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

"""Graph declarations stay metadata-only; graph state stays graph-private."""

from collections.abc import Iterator

import numpy as np
import xarray as xr

from earth2studio.coupling import Binding
from earth2studio.coupling.contracts import ConnectorSnapshot, GraphSnapshot
from earth2studio.models.px.base import StepResult
from earth2studio.run.component import (
    ComponentSnapshot,
    ComponentSpec,
    FieldRequirement,
)
from earth2studio.run.schedules import Schedule
from earth2studio.run.session import (
    CheckpointCapability,
    OutputPort,
    PortRef,
    RunSnapshot,
    SnapshotCompatibility,
)
from earth2studio.utils.coords import coord_array


class _OneActivation:
    def iter_between(
        self,
        reference_time: np.datetime64,
        start: np.datetime64,
        stop: np.datetime64,
    ) -> Iterator[np.datetime64]:
        if start <= reference_time < stop:
            yield reference_time

    def fingerprint(self) -> str:
        return "at-reference-time"


def test_static_declarations_allocate_nothing() -> None:
    signature = coord_array(("variable",), {"variable": ["t2m"]})
    schedule: Schedule = _OneActivation()
    requirement = FieldRequirement("temperature", ("t2m",), signature, "step", schedule)
    port = OutputPort("forecast", ("t2m",), signature, schedule)
    spec = ComponentSpec(
        "atmos", (requirement,), (port,), schedule, CheckpointCapability.SNAPSHOTTABLE
    )
    binding = Binding(
        PortRef("source", "temperature"),
        PortRef("atmos", "temperature"),
        ("t2m",),
        "current",
    )

    assert signature.data.nbytes == 0
    assert spec.inputs[0].signature is signature
    assert spec.outputs[0].name == "forecast"
    assert binding.target == PortRef("atmos", "temperature")
    assert list(
        schedule.iter_between(
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
        )
    ) == [np.datetime64("2024-01-01")]


def test_step_result_publishes_one_entry_per_port() -> None:
    """A model on two grids returns two ports, not one fused array."""
    atmos = xr.DataArray(np.array([1.0]), dims=("variable",))
    ocean = xr.DataArray(np.array([2.0, 3.0]), dims=("variable",))
    result = StepResult(
        state={"history": atmos},
        outputs={"atmos": atmos, "ocean": ocean},
        rng=None,
    )

    assert set(result.outputs) == {"atmos", "ocean"}
    assert result.outputs["ocean"].sizes["variable"] == 2
    assert result.rng is None


def test_scientific_state_travels_as_an_opaque_run_snapshot_payload() -> None:
    """Component and connector state are graph-private, not shared-boundary types."""
    array = xr.DataArray(np.array([1.0]), dims=("variable",))
    payload = GraphSnapshot(
        component_states={
            "atmos": ComponentSnapshot(
                model_state={"history": array},
                rng_state={"generator": array},
                step_index=3,
                adapter_state={"normalization": array},
            )
        },
        connector_states={"coupling": ConnectorSnapshot({"window": array})},
        event_cursor=4,
    )
    snapshot = RunSnapshot(
        compatibility=SnapshotCompatibility(
            schema_version=1,
            earth2studio_version="1.0.0a0",
            plan_identity="test",
            component_versions={"atmos": "1"},
        ),
        output_cursor={PortRef("atmos", "forecast"): 2},
        payload=payload,
    )

    assert snapshot.output_cursor[PortRef("atmos", "forecast")] == 2
    assert isinstance(snapshot.payload, GraphSnapshot)
    assert snapshot.payload.component_states["atmos"].step_index == 3
    assert snapshot.payload.connector_states["coupling"].state["window"] is array
    assert snapshot.payload.event_cursor == 4
    assert not hasattr(snapshot, "component_states")

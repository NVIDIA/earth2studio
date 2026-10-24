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

"""Coupled runner with mock models: a StormScope-like pair and an atmosphere/ocean pair.

Every model adds ``increment`` to its state plus the mean of its forcing, so the
values at each step expose exactly what was delivered to which step.
"""

from collections.abc import Iterator

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.coupler import CoupledRunner, Link
from earth2studio.io import KVBackend
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import Pipeline, Runner, WorkItem, work_items
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

T0 = np.datetime64("2026-01-01T00")


def _h(*hours: float, unit: str = "h") -> np.ndarray:
    return np.array(hours).astype(f"timedelta64[{unit}]")


class _Fill:
    """Source returning a constant per variable."""

    def __init__(self, **values: float) -> None:
        self.values = values

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        data = np.stack(
            [np.full((len(time), 2, 3), self.values.get(v, 0.0)) for v in variable], 1
        )
        return xr.DataArray(
            data.astype(np.float32),
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
        )


class _Mock(PrognosticMixin):
    """State plus ``gain`` times the forcing mean, plus one; records its forcing."""

    def __init__(
        self,
        variables: list[str],
        step: np.timedelta64,
        forcing: list[str] | None = None,
        forcing_leads: np.ndarray | None = None,
        gain: float = 1.0,
        static: list[str] | None = None,
    ) -> None:
        self.variables, self.step_size = variables, step
        self.forcing_variables = forcing
        self.forcing_leads = forcing_leads
        self.static = static
        self.gain = gain
        self.seen: list[tuple[list[float], list[float]]] = []  # (leads_h, means)
        self.created_with: tuple[xr.DataArray, ...] = ()

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": _h(0),
                "variable": self.variables,
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
            dynamic=("batch",),
        )

    def forcing_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...] | None:
        slots = []
        if self.forcing_variables is not None:
            leads = self.forcing_leads if self.forcing_leads is not None else _h(0)
            slots.append(
                coord_array_like(
                    self.input_coords(),
                    {"variable": self.forcing_variables, "lead_time": leads},
                )
            )
        if self.static is not None:
            slots.append(
                coord_array(
                    ("variable", "lat", "lon"),
                    {"variable": self.static, "lat": [10, 0], "lon": [0, 10, 20]},
                )
            )
        if not slots:
            return None
        return slots[0] if len(slots) == 1 else tuple(slots)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(
            input_coords,
            {"lead_time": input_coords["lead_time"].values[-1:] + self.step_size},
        )

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray | None) -> xr.DataArray:
        extra = 0.0
        if forcing is not None:
            self.seen.append(
                (
                    list(forcing["lead_time"].values / np.timedelta64(1, "h")),
                    [float(forcing.values.mean())],
                )
            )
            extra = self.gain * float(forcing.values.mean())
        y = x + 1 + extra
        return y.assign_coords(lead_time=x["lead_time"].values[-1:] + self.step_size)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, *forcing: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        self.created_with = forcing
        return self._advance(x, forcing[0] if self.forcing_variables else None), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, *forcing: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y, forcing[0] if self.forcing_variables else None), None

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, *forcing: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x, *forcing)

    def to(self, device: torch.device) -> "_Mock":
        return self


def _storm() -> tuple[_Mock, _Mock]:
    goes = _Mock(["abi01", "abi02"], np.timedelta64(10, "m"))
    mrms = _Mock(["refc"], np.timedelta64(10, "m"), forcing=["abi01", "abi02"])
    return goes, mrms


def _storm_runner(**kwargs: object) -> tuple[CoupledRunner, _Mock, _Mock]:
    goes, mrms = _storm()
    runner = CoupledRunner(
        {"goes": goes, "mrms": mrms},
        {"goes": _Fill(abi01=5.0, abi02=7.0), "mrms": _Fill(refc=0.0)},
        **kwargs,  # type: ignore[arg-type]
    )
    return runner, goes, mrms


def _atmos_ocean() -> tuple[_Mock, _Mock]:
    atmos = _Mock(["u10m"], np.timedelta64(6, "h"), forcing=["sst"])
    ocean = _Mock(["sst"], np.timedelta64(48, "h"), forcing=["u10m"], gain=1.0)
    return atmos, ocean


def _atmos_ocean_runner(**kwargs: object) -> tuple[CoupledRunner, _Mock, _Mock]:
    atmos, ocean = _atmos_ocean()
    runner = CoupledRunner(
        {"atmos": atmos, "ocean": ocean},
        {"atmos": _Fill(u10m=0.0), "ocean": _Fill(sst=100.0)},
        **kwargs,  # type: ignore[arg-type]
    )
    return runner, atmos, ocean


# ------------------------------------------------------------------ StormScope


def test_storm_wiring_and_protocol() -> None:
    runner, _, _ = _storm_runner()
    protocol: Runner = runner  # Structural conformance, checked by mypy.
    assert protocol.supports_member_batching is False
    assert runner.window == np.timedelta64(10, "m")
    assert [(k.src, k.dst, k.slot, k.reduce) for k in runner.links.values()] == [
        ("goes", "mrms", 0, "last")
    ]
    assert runner.order == ("goes", "mrms")
    assert "goes -> mrms[0]" in runner.describe()


def test_storm_conditioning_matches_goes_lead() -> None:
    runner, goes, mrms = _storm_runner()
    steps = list(runner.run_item(WorkItem(T0, np.timedelta64(30, "m"))))
    assert len(steps) == 4  # Initial condition and three windows.
    assert all(set(s) == {"goes", "mrms"} for s in steps)
    for k, step in enumerate(steps):  # GOES: +1 per step from the source's 5.
        np.testing.assert_array_equal(step["goes"].sel(variable="abi01").values, 5 + k)
        assert step["mrms"]["lead_time"].values[-1] == np.timedelta64(10 * k, "m")
    # MRMS first step sees GOES's initial condition (5, 7 -> mean 6); each later step
    # sees the GOES frame at the lead MRMS has just reached: 6 + k.
    assert [m for _, (m,) in mrms.seen] == [6.0, 7.0, 8.0]
    assert [lead[-1] for lead, _ in mrms.seen] == [0.0, 10 / 60, 20 / 60]


def test_storm_initial_conditions_come_from_goes() -> None:
    runner, _, mrms = _storm_runner(
        links=[Link("goes", "mrms")],
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"))))
    np.testing.assert_array_equal(
        mrms.created_with[0].sel(variable="abi02").values, 7.0
    )


def test_storm_pipeline(tmp_path: object) -> None:
    runner, _, _ = _storm_runner()
    out = {"goes": KVBackend(), "mrms": KVBackend()}
    Pipeline(runner, output=out, device="cpu").run(
        work_items([T0], np.timedelta64(30, "m"))
    )
    assert out["goes"]["abi01"].shape == (1, 4, 2, 3)
    assert out["mrms"]["refc"].shape == (1, 4, 2, 3)
    np.testing.assert_array_equal(out["goes"]["abi01"][0, :, 0, 0], [5, 6, 7, 8])
    # MRMS: 0 -> +1 +6 -> +1 +7 -> +1 +8 (goes frames at the leads MRMS reached)
    np.testing.assert_array_equal(out["mrms"]["refc"][0, :, 0, 0], [0, 7, 15, 24])


def test_forcing_from_source_when_no_producer() -> None:
    goes = _Mock(["abi01"], np.timedelta64(10, "m"), forcing=["glm"])
    runner = CoupledRunner({"goes": goes}, {"goes": (_Fill(abi01=1.0), _Fill(glm=3.0))})
    steps = list(runner.run_item(WorkItem(T0, np.timedelta64(20, "m"))))
    np.testing.assert_array_equal([s["goes"].values.mean() for s in steps], [1, 5, 9])


def test_static_slot_comes_from_source() -> None:
    model = _Mock(["a"], np.timedelta64(1, "h"), static=["lsm"])
    runner = CoupledRunner({"m": model}, {"m": (_Fill(a=0.0), _Fill(lsm=2.0))})
    list(runner.run_item(WorkItem(T0, np.timedelta64(2, "h"))))
    assert "lead_time" not in model.created_with[0].dims
    np.testing.assert_array_equal(model.created_with[0].values, 2.0)


# --------------------------------------------------------------- Atmos / ocean


def test_atmos_ocean_schedule_and_streams() -> None:
    runner, _, _ = _atmos_ocean_runner()
    assert runner.window == np.timedelta64(48, "h")
    schedule = runner.describe()
    assert "model atmos:" in schedule and "8 per window" in schedule
    assert "link ocean -> atmos[0]" in schedule
    coords = runner.output_coords(np.timedelta64(96, "h"))
    assert set(coords) == {"atmos", "ocean"}
    assert list(coords["atmos"]["lead_time"].values / np.timedelta64(1, "h")) == [
        *range(0, 97, 6)
    ]
    assert list(coords["ocean"]["lead_time"].values / np.timedelta64(1, "h")) == [
        0,
        48,
        96,
    ]


def _mean_link() -> list[Link]:
    return [Link("atmos", "ocean", reduce="mean"), Link("ocean", "atmos")]


def test_atmos_ocean_window_mean_and_held_sst() -> None:
    runner, atmos, ocean = _atmos_ocean_runner(links=_mean_link())
    steps = list(runner.run_item(WorkItem(T0, np.timedelta64(96, "h"))))
    assert len(steps) == 3
    assert steps[0]["atmos"].sizes["lead_time"] == 1
    assert steps[1]["atmos"].sizes["lead_time"] == 8
    assert list(steps[1]["atmos"]["lead_time"].values / np.timedelta64(1, "h")) == [
        *range(6, 49, 6)
    ]
    assert steps[1]["ocean"].sizes["lead_time"] == 1
    # Atmos values: 1 + increments of sst/100 gain 1.0*sst -> state grows by 1 + sst.
    atm_y = [float(steps[1]["atmos"].isel(lead_time=i).values.mean()) for i in range(8)]
    # Ocean after the first window: 100 + mean(atmos over 6..48h).
    mean_first = np.mean(atm_y)
    np.testing.assert_allclose(
        steps[1]["ocean"].values.mean(), 100 + 1 + mean_first, rtol=1e-5
    )
    # The atmosphere sees the initial SST for the whole first window...
    first_window = [m for _, (m,) in atmos.seen[:8]]
    assert first_window == [100.0] * 8
    # ...and the new SST, held constant, for the whole second.
    second_window = [m for _, (m,) in atmos.seen[8:]]
    assert second_window == [pytest.approx(float(steps[1]["ocean"].values.mean()))] * 8
    assert len(ocean.seen) == 2


def test_ocean_mean_uses_atmos_outputs_of_its_own_window() -> None:
    runner, _, ocean = _atmos_ocean_runner(links=_mean_link())
    list(runner.run_item(WorkItem(T0, np.timedelta64(96, "h"))))
    first, second = (m for _, (m,) in ocean.seen)
    assert second > first  # Atmosphere keeps growing, so does its second mean.


def test_atmos_ocean_pipeline() -> None:
    runner, _, _ = _atmos_ocean_runner(links=_mean_link())
    out = {"atmos": KVBackend(), "ocean": KVBackend()}
    Pipeline(runner, output=out, device="cpu").run(
        work_items([T0], np.timedelta64(96, "h"))
    )
    assert out["atmos"]["u10m"].shape == (1, 17, 2, 3)
    assert out["ocean"]["sst"].shape == (1, 3, 2, 3)


def test_default_wiring_is_last_both_ways() -> None:
    # Without links both directions are inferred as held, lagged exchange.
    runner, atmos, _ = _atmos_ocean_runner()
    assert {(k.src, k.dst) for k in runner.links.values()} == {
        ("ocean", "atmos"),
        ("atmos", "ocean"),
    }
    list(runner.run_item(WorkItem(T0, np.timedelta64(96, "h"))))
    # Ocean sees the atmosphere's state at its own start (0 -> then 48h state).
    assert atmos.seen[0][1] == [100.0]


def test_transform_applied_to_delivered_field() -> None:
    atmos, ocean = _atmos_ocean()
    runner = CoupledRunner(
        {"atmos": atmos, "ocean": ocean},
        {"atmos": _Fill(u10m=0.0), "ocean": _Fill(sst=100.0)},
        links=[
            Link("ocean", "atmos", transform=lambda x: x - 273.0),
            Link("atmos", "ocean", reduce="mean"),
        ],
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(48, "h"))))
    assert atmos.seen[0][1] == [-173.0]


# ------------------------------------------------------------------- Failures


def test_window_must_divide_steps() -> None:
    atmos = _Mock(["u10m"], np.timedelta64(6, "h"))
    ocean = _Mock(["sst"], np.timedelta64(7, "h"))
    with pytest.raises(ValueError, match="not a multiple"):
        CoupledRunner({"a": atmos, "o": ocean}, {"a": _Fill(), "o": _Fill()})


def test_final_lead_time_must_be_window_multiple() -> None:
    runner, _, _ = _atmos_ocean_runner()
    with pytest.raises(ValueError, match="not a multiple"):
        runner.output_coords(np.timedelta64(60, "h"))


def test_unknown_model_in_link() -> None:
    goes, mrms = _storm()
    with pytest.raises(ValueError, match="unknown model"):
        CoupledRunner(
            {"goes": goes, "mrms": mrms},
            {"goes": _Fill(), "mrms": _Fill()},
            links=[Link("nope", "mrms")],
        )


def test_ambiguous_producers_need_a_link() -> None:
    a = _Mock(["abi01", "abi02"], np.timedelta64(10, "m"))
    b = _Mock(["abi01", "abi02"], np.timedelta64(10, "m"))
    c = _Mock(["refc"], np.timedelta64(10, "m"), forcing=["abi01", "abi02"])
    sources = {"a": _Fill(), "b": _Fill(), "c": _Fill()}
    with pytest.raises(ValueError, match="can be fed by"):
        CoupledRunner({"a": a, "b": b, "c": c}, sources)
    runner = CoupledRunner({"a": a, "b": b, "c": c}, sources, links=[Link("b", "c")])
    assert [(k.src, k.dst) for k in runner.links.values()] == [("b", "c")]


def test_unfed_forcing_without_source_fails() -> None:
    model = _Mock(["a"], np.timedelta64(1, "h"), forcing=["missing"])
    with pytest.raises(ValueError, match="neither a producing model nor a source"):
        CoupledRunner({"m": model}, {"m": _Fill()})


def test_missing_initial_source_fails() -> None:
    goes, _ = _storm()
    with pytest.raises(ValueError, match="no source"):
        CoupledRunner({"goes": goes})


def test_mismatched_grids_require_transform() -> None:
    class _Coarse(_Mock):
        def input_coords(self) -> CoordinateSystem:
            return coord_array_like(super().input_coords(), {"lon": [0, 20]})

    goes = _Mock(["abi01"], np.timedelta64(10, "m"))
    coarse = _Coarse(["refc"], np.timedelta64(10, "m"), forcing=["abi01"])
    with pytest.raises(ValueError, match="grids differ"):
        CoupledRunner({"goes": goes, "c": coarse}, {"goes": _Fill(), "c": _Fill()})


def test_cyclic_mean_links_rejected() -> None:
    a = _Mock(["x"], np.timedelta64(1, "h"), forcing=["y"])
    b = _Mock(["y"], np.timedelta64(1, "h"), forcing=["x"])
    with pytest.raises(ValueError, match="Cyclic"):
        CoupledRunner(
            {"a": a, "b": b},
            {"a": _Fill(), "b": _Fill()},
            links=[Link("a", "b", reduce="mean"), Link("b", "a", reduce="mean")],
        )


def test_members_are_rejected() -> None:
    runner, _, _ = _storm_runner()
    with pytest.raises(ValueError, match="batch"):
        list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"), (0, 1))))

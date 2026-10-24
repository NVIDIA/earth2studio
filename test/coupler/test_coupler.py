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

from earth2studio.coupler import (
    CoupledRunner,
    EnsembleCoupledRunner,
    hold,
    window_mean,
)
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


STORM = {"mrms": hold("goes")}


def _storm_runner(**kwargs: object) -> tuple[CoupledRunner, _Mock, _Mock]:
    goes, mrms = _storm()
    kwargs.setdefault("couplings", STORM)
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
    kwargs.setdefault(
        "couplings",
        {"atmos": hold("ocean"), "ocean": window_mean("atmos", _h(48)[0])},
    )
    runner = CoupledRunner(
        {"atmos": atmos, "ocean": ocean},
        {"atmos": _Fill(u10m=0.0), "ocean": _Fill(sst=100.0)},
        **kwargs,  # type: ignore[arg-type]
    )
    return runner, atmos, ocean


# ------------------------------------------------------------------ StormScope


def test_storm_protocol_and_window() -> None:
    runner, _, _ = _storm_runner()
    protocol: Runner = runner  # Structural conformance, checked by mypy.
    assert protocol.supports_member_batching is False
    assert runner.window == np.timedelta64(10, "m")


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
    runner, _, mrms = _storm_runner()
    list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"))))
    np.testing.assert_array_equal(
        mrms.created_with[0].sel(variable="abi02").values, 7.0
    )


def test_storm_pipeline() -> None:
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


def test_forcing_from_source_without_a_coupling() -> None:
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


def test_coupling_may_regrid_and_convert_units() -> None:
    goes, mrms = _storm()
    runner = CoupledRunner(
        {"goes": goes, "mrms": mrms},
        {"goes": _Fill(abi01=300.0, abi02=300.0), "mrms": _Fill()},
        {"mrms": lambda out, leads: hold("goes")(out, leads) * 2 - 273},
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"))))
    assert mrms.seen[0][1] == [327.0]


# --------------------------------------------------------------- Atmos / ocean


def test_atmos_ocean_window_and_streams() -> None:
    runner, _, _ = _atmos_ocean_runner()
    assert runner.window == np.timedelta64(48, "h")
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


def test_atmos_ocean_window_mean_and_held_sst() -> None:
    runner, atmos, ocean = _atmos_ocean_runner()
    steps = list(runner.run_item(WorkItem(T0, np.timedelta64(96, "h"))))
    assert len(steps) == 3
    assert steps[0]["atmos"].sizes["lead_time"] == 1
    assert steps[1]["atmos"].sizes["lead_time"] == 8
    assert list(steps[1]["atmos"]["lead_time"].values / np.timedelta64(1, "h")) == [
        *range(6, 49, 6)
    ]
    assert steps[1]["ocean"].sizes["lead_time"] == 1
    # Ocean after the first window: 100 + 1 + mean of the atmosphere over 6..48 h.
    atm_y = [float(steps[1]["atmos"].isel(lead_time=i).values.mean()) for i in range(8)]
    np.testing.assert_allclose(
        steps[1]["ocean"].values.mean(), 100 + 1 + np.mean(atm_y), rtol=1e-5
    )
    # The atmosphere sees the initial SST for the whole first window...
    assert [m for _, (m,) in atmos.seen[:8]] == [100.0] * 8
    # ...and the ocean's new SST, held, for the whole second.
    assert [m for _, (m,) in atmos.seen[8:]] == [
        pytest.approx(float(steps[1]["ocean"].values.mean()))
    ] * 8
    first, second = (m for _, (m,) in ocean.seen)
    assert second > first  # The atmosphere keeps growing, so does its mean.


def test_atmos_ocean_pipeline() -> None:
    runner, _, _ = _atmos_ocean_runner()
    out = {"atmos": KVBackend(), "ocean": KVBackend()}
    Pipeline(runner, output=out, device="cpu").run(
        work_items([T0], np.timedelta64(96, "h"))
    )
    assert out["atmos"]["u10m"].shape == (1, 17, 2, 3)
    assert out["ocean"]["sst"].shape == (1, 3, 2, 3)


def test_mutual_coupling_is_independent_of_model_order() -> None:
    def seen(order: tuple[str, str]) -> tuple[list[float], list[float]]:
        models = {
            "a": _Mock(["x"], np.timedelta64(1, "h"), forcing=["y"]),
            "b": _Mock(["y"], np.timedelta64(1, "h"), forcing=["x"]),
        }
        runner = CoupledRunner(
            {k: models[k] for k in order},
            {"a": _Fill(x=0.0), "b": _Fill(y=10.0)},
            {"a": hold("b"), "b": hold("a")},
        )
        list(runner.run_item(WorkItem(T0, np.timedelta64(3, "h"))))
        return (
            [m for _, (m,) in models["a"].seen],
            [m for _, (m,) in models["b"].seen],
        )

    # Each sees the other at its own lead, which exists either way.
    assert seen(("a", "b")) == seen(("b", "a"))
    assert seen(("a", "b"))[0] == [10.0, 11.0, 23.0]


def test_data_requests_list_source_fetches_only() -> None:
    runner, _, _ = _storm_runner()
    # Two initial conditions; the conditioning window is coupled to GOES.
    assert len(runner.data_requests(WorkItem(T0, np.timedelta64(30, "m")))) == 2
    goes = _Mock(["abi01"], np.timedelta64(10, "m"), forcing=["glm"])
    plain = CoupledRunner({"goes": goes}, {"goes": (_Fill(abi01=1.0), _Fill(glm=3.0))})
    # IC, initial glm window, then one fetch for each later step.
    assert len(plain.data_requests(WorkItem(T0, np.timedelta64(30, "m")))) == 4


def test_window_mean_needs_the_producer_to_run_first() -> None:
    atmos, ocean = _atmos_ocean()
    runner = CoupledRunner(
        {"ocean": ocean, "atmos": atmos},  # Wrong order: ocean needs atmos first.
        {"atmos": _Fill(u10m=0.0), "ocean": _Fill(sst=100.0)},
        {"atmos": hold("ocean"), "ocean": window_mean("atmos", _h(48)[0])},
    )
    with pytest.raises(ValueError, match="list it before"):
        list(runner.run_item(WorkItem(T0, np.timedelta64(48, "h"))))


def test_multi_frame_forcing_window() -> None:
    goes = _Mock(["abi01"], np.timedelta64(10, "m"))
    mrms = _Mock(
        ["refc"],
        np.timedelta64(10, "m"),
        forcing=["abi01"],
        forcing_leads=_h(-10, 0, unit="m"),
    )

    def coupling(out, leads):  # type: ignore[no-untyped-def]
        # The producer starts at lead 0, so earlier forcing leads repeat it.
        return hold("goes")(out, np.maximum(leads, np.timedelta64(0, "m")))

    runner = CoupledRunner(
        {"goes": goes, "mrms": mrms},
        {"goes": _Fill(abi01=5.0), "mrms": _Fill()},
        {"mrms": coupling},
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(30, "m"))))
    assert [m for _, (m,) in mrms.seen] == [5.0, 6.0, 7.0]
    assert [leads for leads, _ in mrms.seen][0] == pytest.approx([-1 / 6, 0.0])


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


def test_unknown_or_unforced_model_in_couplings() -> None:
    goes, mrms = _storm()
    sources = {"goes": _Fill(), "mrms": _Fill()}
    with pytest.raises(ValueError, match="unknown model"):
        CoupledRunner(
            {"goes": goes, "mrms": mrms}, sources, {**STORM, "nope": hold("goes")}
        )
    with pytest.raises(ValueError, match="absent or static"):
        CoupledRunner(
            {"goes": goes, "mrms": mrms}, sources, {**STORM, "goes": hold("mrms")}
        )


def test_uncoupled_forcing_without_source_fails() -> None:
    model = _Mock(["a"], np.timedelta64(1, "h"), forcing=["missing"])
    with pytest.raises(ValueError, match="no source|no coupling"):
        CoupledRunner({"m": model}, {"m": _Fill()})
    CoupledRunner({"m": model}, {"m": _Fill()}, {"m": lambda out, leads: out["m"]})


def test_missing_initial_source_fails() -> None:
    goes, _ = _storm()
    with pytest.raises(ValueError, match="no source"):
        CoupledRunner({"goes": goes})


def test_multi_lead_output_rejected() -> None:
    class _Burst(_Mock):
        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            return coord_array_like(input_coords, {"lead_time": _h(6, 12)})

    with pytest.raises(ValueError, match="one output lead time"):
        CoupledRunner({"b": _Burst(["a"], np.timedelta64(6, "h"))}, {"b": _Fill()})


class _Shift:
    """Initial-condition perturbation adding the member's seed."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        return x + float(self.seed % 7)


def test_ensemble_one_member_per_item_through_pipeline() -> None:
    goes, mrms = _storm()
    runner = EnsembleCoupledRunner(
        {"goes": goes, "mrms": mrms},
        {"goes": _Fill(abi01=5.0, abi02=7.0), "mrms": _Fill()},
        STORM,
        perturbations={"goes": _Shift},
    )
    out = {"goes": KVBackend(), "mrms": KVBackend()}
    items = work_items([T0], np.timedelta64(20, "m"), members=2, member_batch=1)
    Pipeline(runner, output=out, device="cpu").run(items)
    assert out["goes"]["abi01"].shape == (2, 1, 3, 2, 3)  # (ensemble, time, lead, ...)
    assert out["mrms"]["refc"].shape == (2, 1, 3, 2, 3)
    # Members differ through the perturbation, and each is reproducible.
    first = out["goes"]["abi01"][:, 0, 0, 0, 0]
    assert first[0] != first[1]
    again = list(runner.run_item(items[1]))[0]["goes"].sel(variable="abi01").values
    assert float(again.flat[0]) == first[1]


def test_ensemble_without_perturbation_is_identical_per_member() -> None:
    runner, _, _ = _storm_runner()
    out = {"goes": KVBackend(), "mrms": KVBackend()}
    items = work_items([T0], np.timedelta64(10, "m"), members=2)
    Pipeline(runner, output=out, device="cpu").run(items)
    np.testing.assert_array_equal(out["mrms"]["refc"][0], out["mrms"]["refc"][1])


def test_unknown_perturbation_model_rejected() -> None:
    goes, mrms = _storm()
    with pytest.raises(ValueError, match="unknown model"):
        EnsembleCoupledRunner(
            {"goes": goes, "mrms": mrms},
            {"goes": _Fill(abi01=5.0, abi02=7.0), "mrms": _Fill()},
            STORM,
            perturbations={"nope": _Shift},
        )


def test_members_are_rejected() -> None:
    runner, _, _ = _storm_runner()
    with pytest.raises(ValueError, match="member_batch=1"):
        list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"), (0, 1))))


class _Coarse(_Mock):
    def input_coords(self) -> CoordinateSystem:
        return coord_array_like(super().input_coords(), {"lon": [0, 20]})


def test_coupling_grid_is_mapped_or_rejected() -> None:
    goes = _Mock(["abi01"], np.timedelta64(10, "m"))
    coarse = _Coarse(["refc"], np.timedelta64(10, "m"), forcing=["abi01"])
    sources = {"goes": _Fill(abi01=2.0), "c": _Fill()}
    item = WorkItem(T0, np.timedelta64(10, "m"))

    def run(coupling) -> list:  # type: ignore[no-untyped-def]
        runner = CoupledRunner({"goes": goes, "c": coarse}, sources, {"c": coupling})
        return list(runner.run_item(item))

    # An exact crop of the source grid is selected, with no coupling code.
    steps = run(hold("goes"))
    assert steps[1]["c"].sizes["lon"] == 2
    np.testing.assert_array_equal(steps[1]["c"]["lon"].values, [0, 20])
    with pytest.raises(ValueError, match="expected 1"):
        run(lambda out, leads: xr.concat([out["goes"]] * 2, "lead_time"))


def test_coupling_off_the_source_grid_is_refused() -> None:
    pytest.importorskip("scipy")  # Automatic regridding is the `utils` extra.
    goes = _Mock(["abi01"], np.timedelta64(10, "m"))
    coarse = _Coarse(["refc"], np.timedelta64(10, "m"), forcing=["abi01"])
    runner = CoupledRunner(
        {"goes": goes, "c": coarse},
        {"goes": _Fill(abi01=2.0), "c": _Fill()},
        # Nearest within the source resolution only: never extrapolated.
        {
            "c": lambda out, leads: hold("goes")(out, leads).assign_coords(
                lon=[100, 110, 120]
            )
        },
    )
    with pytest.raises(ValueError, match="regrid inside the coupling"):
        list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"))))


def test_stochastic_models_get_distinct_stable_seeds() -> None:
    class _Noisy(_Mock):
        stochastic = True

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.seeds: list[int] = []

        def set_rng(self, seed: int, reset: bool = True) -> None:
            self.seeds.append(seed)

    a = _Noisy(["x"], np.timedelta64(1, "h"))
    b = _Noisy(["y"], np.timedelta64(1, "h"))
    runner = CoupledRunner({"a": a, "b": b}, {"a": _Fill(), "b": _Fill()}, seed=3)
    for _ in range(2):
        list(runner.run_item(WorkItem(T0, np.timedelta64(1, "h"))))
    assert a.seeds[0] != b.seeds[0]
    assert a.seeds[0] == a.seeds[1] and b.seeds[0] == b.seeds[1]

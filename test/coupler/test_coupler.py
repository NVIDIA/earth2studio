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
    Link,
    window_last,
    window_max,
    window_mean,
    window_min,
    window_sum,
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
        ("goes", "mrms", 0, None)
    ]
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
    return [Link("atmos", "ocean", reduce=window_mean), Link("ocean", "atmos")]


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


def test_custom_reduction_is_injected() -> None:
    atmos, ocean = _atmos_ocean()
    runner = CoupledRunner(
        {"atmos": atmos, "ocean": ocean},
        {"atmos": _Fill(u10m=0.0), "ocean": _Fill(sst=100.0)},
        links=[
            Link("atmos", "ocean", reduce=lambda x: x.max("lead_time", keepdims=True)),
        ],
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(48, "h"))))
    assert ocean.seen[0][1] == [808.0]  # Eight steps of +1 +sst(100): the last.


def test_regridder_bridges_differing_grids() -> None:
    class _Coarse(_Mock):
        def input_coords(self) -> CoordinateSystem:
            return coord_array_like(super().input_coords(), {"lon": [0, 20]})

    class _Pick:
        """Regridder stand-in: picks lon 0 and 20."""

        def __init__(self, source: object, target: object) -> None:
            self.source_grid, self.target_grid = source, target

        def __call__(self, x: xr.DataArray) -> xr.DataArray:
            return x.sel(lon=[0, 20])

    goes = _Mock(["abi01"], np.timedelta64(10, "m"))
    coarse = _Coarse(["refc"], np.timedelta64(10, "m"), forcing=["abi01"])
    models = {"goes": goes, "c": coarse}
    sources = {"goes": _Fill(abi01=2.0), "c": _Fill()}
    from earth2studio.grids import infer_grid

    pick = _Pick(
        infer_grid(goes.output_coords(goes.input_coords())),
        infer_grid(coarse.forcing_coords()),  # type: ignore[arg-type]
    )
    runner = CoupledRunner(models, sources, links=[Link("goes", "c", regridder=pick)])  # type: ignore[arg-type]
    steps = list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"))))
    assert steps[1]["c"].sizes["lon"] == 2


def test_describe_names_the_reduction() -> None:
    runner, _, _ = _atmos_ocean_runner(links=_mean_link())
    assert "atmos -> ocean[0] ['u10m']: window_mean" in runner.describe()
    assert "ocean -> atmos[0] ['sst']: hold" in runner.describe()


def test_data_requests_lists_source_fetches_only() -> None:
    runner, _, _ = _storm_runner()
    requests = runner.data_requests(WorkItem(T0, np.timedelta64(30, "m")))
    # Two initial conditions; the conditioning window comes from GOES itself.
    assert len(requests) == 2
    runner, _, _ = _atmos_ocean_runner(links=_mean_link())
    # Atmosphere IC, ocean IC; atmosphere's SST window comes from the ocean's IC.
    assert len(runner.data_requests(WorkItem(T0, np.timedelta64(96, "h")))) == 2


def test_data_requests_includes_unwired_forcing() -> None:
    goes = _Mock(["abi01"], np.timedelta64(10, "m"), forcing=["glm"])
    runner = CoupledRunner({"goes": goes}, {"goes": (_Fill(abi01=1.0), _Fill(glm=3.0))})
    # IC, initial glm window, then one fetch for each later step.
    assert len(runner.data_requests(WorkItem(T0, np.timedelta64(30, "m")))) == 4


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


def test_mismatched_grids_require_regridder() -> None:
    class _Coarse(_Mock):
        def input_coords(self) -> CoordinateSystem:
            return coord_array_like(super().input_coords(), {"lon": [0, 20]})

    goes = _Mock(["abi01"], np.timedelta64(10, "m"))
    coarse = _Coarse(["refc"], np.timedelta64(10, "m"), forcing=["abi01"])
    with pytest.raises(ValueError, match="grids differ"):
        CoupledRunner({"goes": goes, "c": coarse}, {"goes": _Fill(), "c": _Fill()})


def test_multi_lead_output_rejected() -> None:
    class _Burst(_Mock):
        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            return coord_array_like(input_coords, {"lead_time": _h(6, 12)})

    with pytest.raises(ValueError, match="one output lead time"):
        CoupledRunner({"b": _Burst(["a"], np.timedelta64(6, "h"))}, {"b": _Fill()})


def test_cyclic_reducing_links_rejected() -> None:
    a = _Mock(["x"], np.timedelta64(1, "h"), forcing=["y"])
    b = _Mock(["y"], np.timedelta64(1, "h"), forcing=["x"])
    with pytest.raises(ValueError, match="Cyclic"):
        CoupledRunner(
            {"a": a, "b": b},
            {"a": _Fill(), "b": _Fill()},
            links=[
                Link("a", "b", reduce=window_mean),
                Link("b", "a", reduce=window_mean),
            ],
        )


def test_members_are_rejected() -> None:
    runner, _, _ = _storm_runner()
    with pytest.raises(ValueError, match="batch"):
        list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"), (0, 1))))


# ------------------------------------------------------------- Review follow-ups


def _pick_regridder(coarse: _Mock, goes: _Mock, picker: object) -> object:
    from earth2studio.grids import infer_grid

    class _Bound:
        def __init__(self) -> None:
            self.source_grid = infer_grid(goes.output_coords(goes.input_coords()))
            self.target_grid = infer_grid(coarse.forcing_coords())  # type: ignore[arg-type]

        def __call__(self, x: xr.DataArray) -> xr.DataArray:
            return picker(x)  # type: ignore[operator]

    return _Bound()


class _Coarse(_Mock):
    def input_coords(self) -> CoordinateSystem:
        return coord_array_like(super().input_coords(), {"lon": [0, 20]})


def test_regridder_output_is_validated() -> None:
    goes = _Mock(["abi01"], np.timedelta64(10, "m"))
    coarse = _Coarse(["refc"], np.timedelta64(10, "m"), forcing=["abi01"])

    def wrong(x: xr.DataArray) -> xr.DataArray:  # Right size, wrong places.
        return x.isel(lon=[0, 1]).assign_coords(lon=[5, 15])

    runner = CoupledRunner(
        {"goes": goes, "c": coarse},
        {"goes": _Fill(abi01=2.0), "c": _Fill()},
        links=[Link("goes", "c", regridder=_pick_regridder(coarse, goes, wrong))],  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="differs from the slot grid"):
        list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"))))


def test_explicit_source_wins_over_auto_wiring() -> None:
    goes, mrms = _storm()
    runner = CoupledRunner(
        {"goes": goes, "mrms": mrms},
        {
            "goes": _Fill(abi01=5.0, abi02=7.0),
            "mrms": (_Fill(refc=0.0), _Fill(abi01=99.0, abi02=99.0)),
        },
    )
    assert runner.links == {}
    list(runner.run_item(WorkItem(T0, np.timedelta64(20, "m"))))
    assert [m for _, (m,) in mrms.seen] == [99.0, 99.0]


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


@pytest.mark.parametrize(
    "reduce", [window_last, window_max, window_min, window_sum, window_mean]
)
def test_builtin_reductions(reduce: object) -> None:
    atmos, ocean = _atmos_ocean()
    # The atmosphere is forced by 100 and gains 1 + 100 per step.
    runner = CoupledRunner(
        {"atmos": atmos, "ocean": ocean},
        {"atmos": (_Fill(u10m=0.0), _Fill(sst=100.0)), "ocean": _Fill(sst=0.0)},
        links=[Link("atmos", "ocean", reduce=reduce)],  # type: ignore[arg-type]
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(48, "h"))))
    got = ocean.seen[0][1][0]
    if reduce in (window_last, window_max):
        assert got == pytest.approx(8 * 101.0)
    elif reduce is window_min:
        assert got == pytest.approx(101.0)
    elif reduce is window_sum:
        assert got == pytest.approx(sum(101.0 * k for k in range(1, 9)))
    else:
        assert got == pytest.approx(np.mean([101.0 * k for k in range(1, 9)]))


def test_check_finite_catches_nan() -> None:
    goes, mrms = _storm()
    sources = {"goes": _Fill(abi01=float("nan"), abi02=1.0), "mrms": _Fill()}
    runner = CoupledRunner({"goes": goes, "mrms": mrms}, sources, check_finite=True)
    with pytest.raises(ValueError, match="not finite"):
        list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"))))
    off = CoupledRunner({"goes": goes, "mrms": mrms}, sources)
    list(off.run_item(WorkItem(T0, np.timedelta64(10, "m"))))  # Propagates silently.


def test_link_scale_and_offset_convert_units() -> None:
    goes, mrms = _storm()
    runner = CoupledRunner(
        {"goes": goes, "mrms": mrms},
        {"goes": _Fill(abi01=300.0, abi02=300.0), "mrms": _Fill()},
        links=[Link("goes", "mrms", scale=2.0, offset=-273.0)],
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(10, "m"))))
    assert mrms.seen[0][1] == [327.0]
    assert "*2-273" in runner.describe()


def test_units_mismatch_requires_conversion() -> None:
    class _Kelvin(_Mock):
        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            out = super().output_coords(input_coords)
            out.attrs["units"] = "K"
            return out

    class _Celsius(_Mock):
        def forcing_coords(self) -> CoordinateSystem:  # type: ignore[override]
            out = super().forcing_coords()
            out.attrs["units"] = "degC"  # type: ignore[union-attr]
            return out  # type: ignore[return-value]

    kelvin = _Kelvin(["t"], np.timedelta64(1, "h"))
    celsius = _Celsius(["o"], np.timedelta64(1, "h"), forcing=["t"])
    models, sources = {"k": kelvin, "c": celsius}, {"k": _Fill(), "c": _Fill()}
    with pytest.raises(ValueError, match="units"):
        CoupledRunner(models, sources)
    CoupledRunner(models, sources, links=[Link("k", "c", offset=-273.15)])


def test_window_start_holds_the_producer_for_the_window() -> None:
    def build(window_start: bool) -> _Mock:
        goes, mrms = _storm()
        slow = _Mock(["z"], np.timedelta64(30, "m"))
        runner = CoupledRunner(
            {"goes": goes, "mrms": mrms, "slow": slow},
            {
                "goes": _Fill(abi01=5.0, abi02=7.0),
                "mrms": _Fill(),
                "slow": _Fill(),
            },
            links=[Link("goes", "mrms", window_start=window_start)],
        )
        list(runner.run_item(WorkItem(T0, np.timedelta64(60, "m"))))
        return mrms

    assert [m for _, (m,) in build(False).seen] == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
    # Each 30 min window sees GOES as it was when the window started.
    assert [m for _, (m,) in build(True).seen] == [6.0, 6.0, 6.0, 9.0, 9.0, 9.0]


def test_window_start_rejected_with_reduction() -> None:
    atmos, ocean = _atmos_ocean()
    with pytest.raises(ValueError, match="held links"):
        CoupledRunner(
            {"atmos": atmos, "ocean": ocean},
            {"atmos": _Fill(), "ocean": _Fill()},
            links=[Link("atmos", "ocean", reduce=window_mean, window_start=True)],
        )


def test_diagnostics_publish_their_own_stream() -> None:
    class _Double:
        def input_coords(self) -> CoordinateSystem:
            return coord_array(
                ("batch", "variable", "lat", "lon"),
                {"variable": ["refc"], "lat": [10, 0], "lon": [0, 10, 20]},
                dynamic=("batch",),
            )

        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            return coord_array_like(input_coords, {"variable": ["refc2"]})

        def to(self, device: torch.device) -> "_Double":
            return self

        def __call__(self, x: xr.DataArray) -> xr.DataArray:
            return (2 * x).assign_coords(variable=["refc2"])

    runner, _, _ = _storm_runner(diagnostics={"twice": ("mrms", _Double())})
    assert set(runner.output_coords(np.timedelta64(20, "m"))) == {
        "goes",
        "mrms",
        "twice",
    }
    steps = list(runner.run_item(WorkItem(T0, np.timedelta64(20, "m"))))
    assert all("twice" in s for s in steps)
    np.testing.assert_array_equal(steps[1]["twice"].values, 2 * steps[1]["mrms"].values)
    out = {k: KVBackend() for k in ("goes", "mrms", "twice")}
    Pipeline(runner, output=out, device="cpu").run(
        work_items([T0], np.timedelta64(20, "m"))
    )
    assert out["twice"]["refc2"].shape == (1, 3, 2, 3)
    with pytest.raises(ValueError, match="clashes"):
        _storm_runner(diagnostics={"goes": ("mrms", _Double())})


def test_mutual_hold_is_independent_of_model_order() -> None:
    def seen(order: tuple[str, str]) -> tuple[list[float], list[float]]:
        models = {
            "a": _Mock(["x"], np.timedelta64(1, "h"), forcing=["y"]),
            "b": _Mock(["y"], np.timedelta64(1, "h"), forcing=["x"]),
        }
        runner = CoupledRunner(
            {k: models[k] for k in order}, {"a": _Fill(x=0.0), "b": _Fill(y=10.0)}
        )
        list(runner.run_item(WorkItem(T0, np.timedelta64(3, "h"))))
        return (
            [m for _, (m,) in models["a"].seen],
            [m for _, (m,) in models["b"].seen],
        )

    assert seen(("a", "b")) == seen(("b", "a"))


def test_held_link_with_multi_frame_forcing_window() -> None:
    goes = _Mock(["abi01"], np.timedelta64(10, "m"))
    mrms = _Mock(
        ["refc"],
        np.timedelta64(10, "m"),
        forcing=["abi01"],
        forcing_leads=_h(-10, 0, unit="m"),
    )
    sources = {"goes": _Fill(abi01=5.0), "mrms": (_Fill(), _Fill(abi01=1.0))}
    # The producer's one-frame initial condition cannot cover a two-frame window,
    # so an explicit source supplies it and the link takes over for later steps.
    runner = CoupledRunner(
        {"goes": goes, "mrms": mrms}, sources, links=[Link("goes", "mrms")]
    )
    list(runner.run_item(WorkItem(T0, np.timedelta64(30, "m"))))
    assert [m for _, (m,) in mrms.seen] == [1.0, 6.0, 7.0]
    assert [leads for leads, _ in mrms.seen][0] == pytest.approx([-1 / 6, 0.0])

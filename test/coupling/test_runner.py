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

"""CoupledRunner scheduling, execution and validation on tiny CPU components."""

from collections.abc import Mapping

import numpy as np
import pytest
import torch
import xarray as xr
from conftest import (
    ATMOS_GRID,
    OCEAN_GRID,
    T0,
    Bucket,
    Double,
    Fill,
    H,
    Observations,
    Rain,
    _Toy,
    atmosphere,
    ocean,
    signature,
)

from earth2studio.coupling import Connector, CoupledRunner, CouplingError
from earth2studio.grids import NearestRegridder
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import PrognosticRunner, Runner, WorkItem
from earth2studio.utils import coord_array_like
from earth2studio.utils.type import CoordinateSystem


def _toy(ic_sources: Mapping[str, Fill]) -> CoupledRunner:
    return CoupledRunner(
        {"atmos": atmosphere(), "ocean": ocean()},
        [
            Connector(
                "ocean",
                "atmos",
                time="hold",
                regridder=NearestRegridder(OCEAN_GRID, ATMOS_GRID),
            ),
            Connector(
                "atmos", "ocean", regridder=NearestRegridder(ATMOS_GRID, OCEAN_GRID)
            ),
        ],
        sources=dict(ic_sources),
    )


def _run(runner: CoupledRunner, hours: int) -> dict[str, list[xr.DataArray]]:
    streams: dict[str, list[xr.DataArray]] = {}
    for step in runner.run_item(WorkItem(T0, np.timedelta64(hours, "h"))):
        for name, x in step.items():
            streams.setdefault(name, []).append(x)
    return streams


def _series(frames: list[xr.DataArray]) -> dict[int, float]:
    return {int(x["lead_time"].values[0] // H): float(x.mean()) for x in frames}


# -- the #1229 toy -------------------------------------------------------------


def test_reproduces_the_hand_computed_toy(ic_sources: Mapping[str, Fill]) -> None:
    runner: Runner = _toy(ic_sources)
    streams = _run(runner, 96)  # type: ignore[arg-type]
    z, sst = _series(streams["atmos"]), _series(streams["ocean"])
    assert list(z) == list(range(0, 97, 6))
    assert list(sst) == [0, 48, 96]
    np.testing.assert_allclose(z[48], 9.6, atol=1e-5)
    np.testing.assert_allclose(z[96], 19.2336, atol=1e-4)
    np.testing.assert_allclose(sst[48], 2.042, atol=1e-6)
    np.testing.assert_allclose(sst[96], 2.180147, atol=1e-6)


def test_streams_keep_each_component_grid_and_time(
    ic_sources: Mapping[str, Fill],
) -> None:
    streams = _run(_toy(ic_sources), 48)
    for name, grid in (("atmos", ATMOS_GRID), ("ocean", OCEAN_GRID)):
        for x in streams[name]:
            assert x["time"].values.tolist() == [T0]
            np.testing.assert_array_equal(x["lat"], grid.latitude)


def test_output_coords_match_what_runs(ic_sources: Mapping[str, Fill]) -> None:
    runner = _toy(ic_sources)
    coords = runner.output_coords(np.timedelta64(96, "h"))
    streams = _run(runner, 96)
    for name, signature_ in coords.items():
        leads = np.concatenate([x["lead_time"].values for x in streams[name]])
        np.testing.assert_array_equal(signature_["lead_time"].values, leads)
        assert "time" in signature_.attrs["earth2studio_dynamic_dims"]


def test_short_items_skip_slow_components(ic_sources: Mapping[str, Fill]) -> None:
    runner = _toy(ic_sources)
    streams = _run(runner, 24)
    assert list(_series(streams["ocean"])) == [0]
    assert list(_series(streams["atmos"])) == [0, 6, 12, 18, 24]
    assert list(runner.output_coords(np.timedelta64(0, "h"))) == ["atmos", "ocean"]


def test_lagging_and_ordering_are_derived(ic_sources: Mapping[str, Fill]) -> None:
    text = _toy(ic_sources).describe(np.timedelta64(96, "h"))
    lines = text.splitlines()
    # The held edge reads the ocean's last published frame; the exact edge waits
    # for the atmosphere's chunk at +48h and reduces +0h..+42h.
    assert any(
        "atmos  step" in line and "ocean at +42h from +0h" in line for line in lines
    )
    assert any(
        "atmos  step" in line and "ocean at +48h" in line and "from" not in line
        for line in lines
    )
    ocean_line = next(line for line in lines if "ocean  initialize" in line)
    assert "atmos at +48h from +0h..+42h" in ocean_line
    order = [line.split()[1] for line in lines if line.strip().startswith("+48h")]
    assert order == ["atmos", "ocean"]
    assert "ocean -> atmos forcing[0] (hold, regridded): sst" in text


def test_data_requests_list_every_fetch(ic_sources: Mapping[str, Fill]) -> None:
    runner = _toy(ic_sources)
    requests = runner.data_requests(WorkItem(T0, np.timedelta64(96, "h")))
    assert [r.source for r in requests] == [ic_sources["atmos"], ic_sources["ocean"]]
    assert [list(r.variable) for r in requests] == [["z1000"], ["sst"]]
    _run(runner, 96)
    assert len(ic_sources["atmos"].calls) == len(ic_sources["ocean"].calls) == 1


# -- feedback --------------------------------------------------------------------


def _pair(time: str) -> dict:
    """Two 6 h models each forced by the other's frame at the output valid time."""
    left = _Toy("a", "b", ATMOS_GRID, 6, 1.0, 1.0, forcing_lead=6)
    right = _Toy("b", "a", ATMOS_GRID, 6, 0.0, 0.5, forcing_lead=6)
    sources = {
        "left": Fill(ATMOS_GRID, {"a": 0.0}),
        "right": Fill(ATMOS_GRID, {"b": 1.0}),
    }
    connectors = [Connector("right", "left", time=time), Connector("left", "right")]
    return {
        "components": {"left": left, "right": right},
        "connectors": connectors,
        "sources": sources,
    }


def test_feedback_loop_names_the_cycle() -> None:
    with pytest.raises(CouplingError, match=r"cycle left -> right -> left.*hold"):
        CoupledRunner(**_pair("exact"))


def test_hold_breaks_the_cycle() -> None:
    streams = _run(CoupledRunner(**_pair("hold")), 12)
    a, b = _series(streams["left"]), _series(streams["right"])
    # left holds right's frame from the previous step; right reads left's
    # fresh frame at the same valid time: a6 = 0 + 1 + 1, b6 = 1 + 0.5 * a6,
    # a12 = a6 + 1 + b6, b12 = b6 + 0.5 * a12.
    assert a == pytest.approx({0: 0.0, 6: 2.0, 12: 5.0})
    assert b == pytest.approx({0: 1.0, 6: 2.0, 12: 4.5})


def test_exact_edge_from_a_slower_producer_is_explained() -> None:
    slow = _Toy("sst", "x", OCEAN_GRID, 48, 0.0, 0.0)
    with pytest.raises(CouplingError, match=r"'ocean' publishes no frames for 'sst'"):
        CoupledRunner(
            {"atmos": atmosphere(), "ocean": slow},
            [
                Connector(
                    "ocean", "atmos", regridder=NearestRegridder(OCEAN_GRID, ATMOS_GRID)
                )
            ],
            sources={
                "atmos": Fill(ATMOS_GRID, {"z1000": 0.0}),
                "ocean": (Fill(OCEAN_GRID, {"sst": 2.0}), Fill(OCEAN_GRID, {"x": 0.0})),
            },
        )


def test_exact_edges_both_ways_deadlock(ic_sources: Mapping[str, Fill]) -> None:
    with pytest.raises(CouplingError, match="cycle atmos -> ocean -> atmos"):
        CoupledRunner(
            {"atmos": atmosphere(), "ocean": ocean()},
            [
                Connector(
                    "ocean", "atmos", regridder=NearestRegridder(OCEAN_GRID, ATMOS_GRID)
                ),
                Connector(
                    "atmos", "ocean", regridder=NearestRegridder(ATMOS_GRID, OCEAN_GRID)
                ),
            ],
            sources=dict(ic_sources),
        )


# -- diagnostics -------------------------------------------------------------------


def test_diagnostics_run_when_inputs_arrive(ic_sources: Mapping[str, Fill]) -> None:
    double = Double()
    runner = CoupledRunner(
        {"atmos": atmosphere(), "double": double},
        [Connector("atmos", "double")],
        sources={"atmos": (ic_sources["atmos"], Fill(ATMOS_GRID, {"sst": 2.0}))},
    )
    streams = _run(runner, 24)
    z, z2 = _series(streams["atmos"]), _series(streams["double"])
    assert list(z2) == list(z) == [0, 6, 12, 18, 24]
    assert z2 == pytest.approx({k: 2 * v for k, v in z.items()})
    assert double.calls == 5
    leads = runner.output_coords(np.timedelta64(24, "h"))["double"]["lead_time"]
    assert [int(v // H) for v in leads.values] == [0, 6, 12, 18, 24]


def test_diagnostic_output_feeds_a_consumer(ic_sources: Mapping[str, Fill]) -> None:
    consumer = _Toy("sst", "z2:mean:48h", OCEAN_GRID, 48, 0.0, 0.01, forcing_lead=48)
    runner = CoupledRunner(
        {"atmos": atmosphere(), "double": Double(), "ocean": consumer},
        [
            Connector("atmos", "double"),
            Connector(
                "double", "ocean", regridder=NearestRegridder(ATMOS_GRID, OCEAN_GRID)
            ),
        ],
        sources={
            "atmos": (ic_sources["atmos"], Fill(ATMOS_GRID, {"sst": 2.0})),
            "ocean": ic_sources["ocean"],
        },
    )
    sst = _series(_run(runner, 48)["ocean"])
    np.testing.assert_allclose(sst[48], 2.0 + 0.01 * 8.4, atol=1e-6)


def test_diagnostic_needs_an_exact_connector() -> None:
    with pytest.raises(CouplingError, match="Diagnostic 'double' needs"):
        CoupledRunner(
            {"atmos": atmosphere(), "double": Double()},
            [Connector("atmos", "double", time="hold")],
        )


# -- stateful assimilation with sparse observations --------------------------------


def test_observations_nudge_only_when_present() -> None:
    bucket, observations = Bucket(), Observations()
    runner = CoupledRunner(
        {"rain": Rain(), "soil": bucket},
        [Connector("rain", "soil")],
        sources={
            "rain": Fill(ATMOS_GRID, {"tp": 2.0}),
            "soil": (Fill(ATMOS_GRID, {"swvl1": 0.1}), None, observations),
        },
    )
    soil = _series(_run(runner, 96)["soil"])
    assert bucket.observed == [False, False, True, False]
    expected = {0: 0.1, 24: 0.108, 48: 0.116, 72: 0.262, 96: 0.27}
    assert soil == pytest.approx(expected, abs=1e-6)
    assert [int((t - T0) // H) for t in observations.requests] == [24, 48, 72, 96]
    # data_requests describes fetch_data calls only; DataFrame fetches are not.
    requests = runner.data_requests(WorkItem(T0, np.timedelta64(96, "h")))
    assert all(r.source is not observations for r in requests)
    assert len(requests) == 2


def test_a_consumer_initializes_once_its_window_is_complete() -> None:
    runner = CoupledRunner(
        {"rain": Rain(), "soil": Bucket()},
        [Connector("rain", "soil")],
        sources={
            "rain": Fill(ATMOS_GRID, {"tp": 2.0}),
            "soil": (Fill(ATMOS_GRID, {"swvl1": 0.1}), None, Observations()),
        },
    )
    lines = runner.describe(np.timedelta64(24, "h")).splitlines()
    soil = next(line for line in lines if "soil" in line and "initialize" in line)
    assert soil.split()[0] == "+24h"
    assert "rain at +24h from +0h..+18h" in soil


# -- history before the item time ------------------------------------------------


class _Windowed(_Toy):
    """Toy whose forcing window starts 6 h before the item time."""

    def forcing_coords(self) -> CoordinateSystem:
        return signature([self.forcing], self.grid, [-6, 0])


def test_frames_before_the_item_time_come_from_the_slot_source() -> None:
    history = Fill(ATMOS_GRID, {"z1000": 10.0})
    consumer = _Windowed("q", "z1000", ATMOS_GRID, 6, 0.0, 1.0)
    runner = CoupledRunner(
        {"atmos": atmosphere(), "consumer": consumer},
        [Connector("atmos", "consumer")],
        sources={
            "atmos": (Fill(ATMOS_GRID, {"z1000": 0.0}), Fill(ATMOS_GRID, {"sst": 2.0})),
            "consumer": (Fill(ATMOS_GRID, {"q": 0.0}), history),
        },
    )
    q = _series(_run(runner, 6)["consumer"])
    assert q[6] == pytest.approx(5.0)  # mean of history 10 at -6h and atmos 0 at +0h
    assert history.calls == [([T0 - 6 * H], ["z1000"])]
    requests = runner.data_requests(WorkItem(T0, np.timedelta64(6, "h")))
    assert [r.lead_time.tolist() for r in requests if r.source is history] == [
        [np.timedelta64(-6 * 3600 * 10**9, "ns")]
    ]


# -- chunks and delays ---------------------------------------------------------------


class _Chunked(Rain):
    """Model publishing two 3 h frames per 6 h step: tp = 2, then 4."""

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        leads = input_coords["lead_time"].values[-1:] + np.array([3, 6]) * H
        return coord_array_like(input_coords, {"lead_time": leads})

    def _advance(self, x: xr.DataArray) -> xr.DataArray:
        last = x.isel(lead_time=[-1])
        frames = [last * 0 + 2.0, last * 0 + 4.0]
        y = xr.concat(frames, dim="lead_time")
        return y.assign_coords(
            lead_time=last["lead_time"].values[-1] + np.array([3, 6]) * H
        )


def test_consumers_wait_for_multi_lead_chunks() -> None:
    consumer = _Toy("q", "tp", ATMOS_GRID, 3, 0.0, 1.0)
    runner = CoupledRunner(
        {"rain": _Chunked(), "consumer": consumer},
        [Connector("rain", "consumer")],
        sources={
            "rain": Fill(ATMOS_GRID, {"tp": 1.0}),
            "consumer": Fill(ATMOS_GRID, {"q": 0.0}),
        },
    )
    lines = runner.describe(np.timedelta64(12, "h")).splitlines()
    # The +6h chunk carries the +3h frame the consumer's second step needs.
    assert next(line for line in lines if "consumer  step" in line).split()[0] == "+6h"
    q = _series(_run(runner, 12)["consumer"])
    assert q == pytest.approx({0: 0.0, 3: 1.0, 6: 3.0, 9: 7.0, 12: 9.0})


# -- equivalence with the single-model runner ----------------------------------------


def test_a_lone_forced_model_matches_prognostic_runner() -> None:
    sources = (Fill(ATMOS_GRID, {"z1000": 0.0}), Fill(ATMOS_GRID, {"sst": 3.0}))
    coupled = _series(
        _run(CoupledRunner({"atmos": atmosphere()}, sources={"atmos": sources}), 18)[
            "atmos"
        ]
    )
    single = PrognosticRunner(atmosphere(), sources[0], forcing=sources[1:])
    direct = {
        int(s["forecast"]["lead_time"].values[-1] // H): float(s["forecast"].mean())
        for s in single.run_item(WorkItem(T0, np.timedelta64(18, "h")))
    }
    assert coupled == pytest.approx(direct)


# -- seeding and devices ---------------------------------------------------------------


class _Stochastic(_Toy):
    stochastic = True

    def __init__(self) -> None:
        super().__init__("z1000", "sst", ATMOS_GRID, 6, 1.0, 0.1)
        self.seeds: list[int] = []

    def set_rng(self, seed: int, reset: bool = True) -> None:
        self.seeds.append(seed)


def test_stochastic_components_are_seeded_per_member() -> None:
    from earth2studio.run.work import member_seed

    model = _Stochastic()
    sources = (Fill(ATMOS_GRID, {"z1000": 0.0}), Fill(ATMOS_GRID, {"sst": 2.0}))
    runner = CoupledRunner({"atmos": model}, sources={"atmos": sources}, seed=3)
    item = WorkItem(T0, np.timedelta64(6, "h"), member_ids=(4,))
    list(runner.run_item(item))
    assert model.seeds == [member_seed(3, item.time, 4)]
    with pytest.raises(ValueError, match="batch"):
        list(runner.run_item(WorkItem(T0, np.timedelta64(6, "h"), member_ids=(0, 1))))


def test_to_moves_models_and_regridders(ic_sources: Mapping[str, Fill]) -> None:
    runner = _toy(ic_sources)
    assert runner.to(torch.device("cpu")) is runner
    assert runner.device == torch.device("cpu")
    assert len(runner.regridders) == 2


# -- validation ---------------------------------------------------------------------


def _atmos_ocean(*connectors: Connector, **kwargs: object) -> CoupledRunner:
    return CoupledRunner(
        {"atmos": atmosphere(), "ocean": ocean()}, list(connectors), **kwargs  # type: ignore[arg-type]
    )


TO_OCEAN = NearestRegridder(ATMOS_GRID, OCEAN_GRID)
TO_ATMOS = NearestRegridder(OCEAN_GRID, ATMOS_GRID)


@pytest.mark.parametrize(
    "connectors, match",
    [
        (
            [Connector("atmos", "oceans")],
            "unknown component 'oceans'; did you mean 'ocean'",
        ),
        ([Connector("atmos", "ocean", slot=1, regridder=TO_OCEAN)], "no time-varying"),
        (
            [
                Connector("atmos", "ocean", regridder=TO_OCEAN),
                Connector("atmos", "ocean", regridder=TO_OCEAN),
            ],
            "feed the same slot",
        ),
        ([Connector("atmos", "ocean")], "grids differ; pass a regridder"),
        ([Connector("atmos", "ocean", regridder=TO_ATMOS)], r"source grid.*\(C5\)"),
        (
            [
                Connector(
                    "atmos", "ocean", regridder=NearestRegridder(ATMOS_GRID, ATMOS_GRID)
                )
            ],
            r"target grid.*\(C5\)",
        ),
        (
            [Connector("atmos", "ocean", time="hold", regridder=TO_OCEAN)],
            "applies to plain labels",
        ),
        (
            [
                Connector(
                    "atmos", "ocean", variables={"z1000": "z100"}, regridder=TO_OCEAN
                )
            ],
            "does not publish 'z100'.*did you mean 'z1000'",
        ),
        (
            [
                Connector(
                    "atmos", "ocean", variables={"t2m": "z1000"}, regridder=TO_OCEAN
                )
            ],
            "renames 't2m'",
        ),
    ],
)
def test_declarations_are_validated_eagerly(
    connectors: list[Connector], match: str
) -> None:
    with pytest.raises(CouplingError, match=match):
        _atmos_ocean(*connectors)


def test_windows_must_divide_into_producer_frames() -> None:
    consumer = _Toy("sst", "z1000:mean:45h", OCEAN_GRID, 48, 0.0, 0.01, forcing_lead=48)
    with pytest.raises(CouplingError, match="cannot be reduced from frames every 6h"):
        CoupledRunner(
            {"atmos": atmosphere(), "ocean": consumer},
            [Connector("atmos", "ocean", regridder=TO_OCEAN)],
        )


def test_unconnected_slots_need_a_source() -> None:
    with pytest.raises(CouplingError, match=r"'atmos' forcing\[0\] needs a source"):
        CoupledRunner({"atmos": atmosphere()}, sources={"atmos": Fill(ATMOS_GRID, {})})


def test_recommended_sources_are_built_only_when_a_fetch_needs_one() -> None:
    calls = []

    def recommend() -> tuple[Fill, Fill]:
        calls.append(1)
        return Fill(ATMOS_GRID, {"z1000": 1.0}), Fill(ATMOS_GRID, {"sst": 2.0})

    atmos = atmosphere()
    atmos.default_sources = recommend  # type: ignore[attr-defined]
    given = (Fill(ATMOS_GRID, {"z1000": 1.0}), Fill(ATMOS_GRID, {"sst": 2.0}))
    CoupledRunner({"atmos": atmos}, sources={"atmos": given})
    assert not calls
    runner = CoupledRunner({"atmos": atmos}, sources={"atmos": given[0]})
    assert len(calls) == 1
    list(runner.run_item(WorkItem(T0, 12 * H)))
    assert len(calls) == 1


def test_components_must_be_models() -> None:
    with pytest.raises(CouplingError, match="neither"):
        CoupledRunner({"x": object()})
    with pytest.raises(CouplingError, match="at least one stateful"):
        CoupledRunner({"double": Double()})
    with pytest.raises(CouplingError, match="at least one component"):
        CoupledRunner({})
    with pytest.raises(CouplingError, match="unknown component 'atmo'"):
        CoupledRunner({"atmos": atmosphere()}, sources={"atmo": Fill(ATMOS_GRID, {})})


# -- components are ordinary models ------------------------------------------------


@pytest.mark.parametrize(
    "model", [atmosphere(), ocean(), Rain()], ids=["atmos", "ocean", "rain"]
)
def test_components_are_conformant_prognostic_models(model: _Toy) -> None:
    from earth2studio.models.conformance import check_prognostic_contract

    skipped = check_prognostic_contract(model)  # type: ignore[arg-type]
    assert all(rule.split(":")[0] in {"P14", "P21"} for rule in skipped)


def test_diagnostic_component_is_conformant() -> None:
    from earth2studio.models.conformance import check_diagnostic_contract

    skipped = check_diagnostic_contract(Double())  # type: ignore[arg-type]
    assert all(rule.split(":")[0] == "D10" for rule in skipped)


# -- several output slots -------------------------------------------------------------


class _TwoSlots(PrognosticMixin):
    """6 h model with output slots on two grids.

    With ``derived=False`` both slots are state: ``z1000`` on the atmosphere grid
    gains 1 and ``sst`` on the ocean grid gains 0.5 per step. With
    ``derived=True`` the second slot is output-only: ``zbar``, the mean of the
    new ``z1000``, on the ocean grid.
    """

    def __init__(self, derived: bool = False, ocean_step: int = 6) -> None:
        self.derived, self.ocean_step = derived, ocean_step

    def input_coords(self) -> CoordinateSystem | tuple[CoordinateSystem, ...]:
        atmos = signature(["z1000"], ATMOS_GRID)
        return atmos if self.derived else (atmos, signature(["sst"], OCEAN_GRID))

    def output_coords(self, *inputs: CoordinateSystem) -> tuple[CoordinateSystem, ...]:
        lead = inputs[0]["lead_time"].values[-1:]
        atmos = coord_array_like(inputs[0], {"lead_time": lead + 6 * H})
        second = signature(["zbar"] if self.derived else ["sst"], OCEAN_GRID)
        second = coord_array_like(second, {"lead_time": lead + self.ocean_step * H})
        return atmos, second

    def _advance(self, *x: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
        z = x[0].isel(lead_time=[-1]) + 1.0
        z = z.assign_coords(lead_time=z["lead_time"] + 6 * H)
        if self.derived:
            template = Fill(OCEAN_GRID, {"zbar": 0.0})(
                z["time"].values, np.array(["zbar"])
            ).expand_dims(lead_time=z["lead_time"].values, axis=1)
            return z, template + float(z.mean())
        sst = x[1].isel(lead_time=[-1]) + 0.5
        return z, sst.assign_coords(lead_time=z["lead_time"].values)

    def initialize(self, *x: xr.DataArray) -> tuple[tuple, None]:  # type: ignore[override]
        return self._advance(*x), None

    def step(self, *y: xr.DataArray, state: None = None) -> tuple[tuple, None]:  # type: ignore[override]
        return self._advance(*y), None

    def to(self, device: torch.device) -> "_TwoSlots":
        return self


def _two_slot_sources() -> tuple[Fill, Fill]:
    return Fill(ATMOS_GRID, {"z1000": 0.0}), Fill(OCEAN_GRID, {"sst": 2.0})


def test_each_output_slot_is_its_own_stream() -> None:
    sst_reader = _Toy("q", "sst", OCEAN_GRID, 6, 0.0, 1.0)
    runner = CoupledRunner(
        {"model": _TwoSlots(), "reader": sst_reader},
        [Connector("model", "reader")],  # sst lives in slot 1, on the ocean grid
        sources={"model": _two_slot_sources(), "reader": Fill(OCEAN_GRID, {"q": 0.0})},
    )
    streams = _run(runner, 12)
    assert set(streams) == {"model.0", "model.1", "reader"}
    assert _series(streams["model.0"]) == pytest.approx({0: 0.0, 6: 1.0, 12: 2.0})
    assert _series(streams["model.1"]) == pytest.approx({0: 2.0, 6: 2.5, 12: 3.0})
    # reader adds the sst valid at its state's time: 2.0, then 2.5.
    assert _series(streams["reader"]) == pytest.approx({0: 0.0, 6: 2.0, 12: 4.5})
    coords = runner.output_coords(np.timedelta64(12, "h"))
    np.testing.assert_array_equal(coords["model.1"]["lat"], OCEAN_GRID.latitude)
    assert "streams model.0, model.1" in runner.describe(np.timedelta64(6, "h"))


def test_output_only_slots_start_at_the_first_chunk() -> None:
    runner = CoupledRunner(
        {"model": _TwoSlots(derived=True)},
        sources={"model": Fill(ATMOS_GRID, {"z1000": 0.0})},
    )
    streams = _run(runner, 12)
    assert list(_series(streams["model.0"])) == [0, 6, 12]
    assert _series(streams["model.1"]) == pytest.approx({6: 1.0, 12: 2.0})
    leads = runner.output_coords(np.timedelta64(12, "h"))["model.1"]["lead_time"]
    assert [int(v // H) for v in leads.values] == [6, 12]


def test_output_slots_must_share_lead_times() -> None:
    with pytest.raises(CouplingError, match="different lead times.*split"):
        CoupledRunner({"model": _TwoSlots(ocean_step=48)})


def test_a_variable_in_two_output_slots_is_ambiguous() -> None:
    class _Duplicate(_TwoSlots):
        def output_coords(self, *inputs: CoordinateSystem) -> tuple:
            atmos, _ = super().output_coords(*inputs)
            return atmos, atmos

    with pytest.raises(CouplingError, match=r"'z1000' in output slots \[0, 1\]"):
        CoupledRunner(
            {"model": _Duplicate(), "double": Double()},
            [Connector("model", "double")],
        )


def test_stream_names_must_be_unique() -> None:
    with pytest.raises(CouplingError, match=r"\['model.0'\] are used twice"):
        CoupledRunner({"model": _TwoSlots(), "model.0": atmosphere()})


class _Both(Double):
    """Diagnostic with two output slots: ``z2`` and ``z3``."""

    def output_coords(self, input_coords: CoordinateSystem) -> tuple:  # type: ignore[override]
        return super().output_coords(input_coords), coord_array_like(
            input_coords, {"variable": ["z3"]}
        )

    def __call__(self, x: xr.DataArray) -> tuple:  # type: ignore[override]
        return super().__call__(x), (3 * x).assign_coords(variable=["z3"])


def test_diagnostics_may_have_several_output_slots(
    ic_sources: Mapping[str, Fill],
) -> None:
    runner = CoupledRunner(
        {"atmos": atmosphere(), "both": _Both()},
        [Connector("atmos", "both")],
        sources={"atmos": (ic_sources["atmos"], Fill(ATMOS_GRID, {"sst": 2.0}))},
    )
    streams = _run(runner, 6)
    assert _series(streams["both.0"]) == pytest.approx({0: 0.0, 6: 2.4})
    assert _series(streams["both.1"]) == pytest.approx({0: 0.0, 6: 3.6})

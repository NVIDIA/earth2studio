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

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import BilinearRegridder, NearestRegridder
from earth2studio.io import KVBackend
from earth2studio.nvcoupler import (
    Connector,
    Coupler,
    CouplingError,
    RunSequence,
    Slot,
)
from earth2studio.nvcoupler.testing import (
    ATMOS_SHAPE,
    OCEAN_SHAPE,
    ToyAtmosphere,
    ToyImpact,
    ToyOcean,
    ToyPrognostic,
    toy_grid,
    toy_source,
)
from earth2studio.run import Pipeline, Runner, WorkItem, work_items
from earth2studio.run.work import member_seed

T0 = np.datetime64("2026-01-01T00")
ATMOS_GRID = toy_grid(*ATMOS_SHAPE)
OCEAN_GRID = toy_grid(*OCEAN_SHAPE)


def _hours(*values: int) -> np.ndarray:
    return np.array(values, dtype="timedelta64[h]")


def _value(step: dict, stream: str) -> float:
    return float(step[stream].isel(lat=1, lon=1).values.ravel()[0])


def _item(hours: int) -> WorkItem:
    return WorkItem(T0, np.timedelta64(hours, "h"))


def _atmos_ocean(**kwargs) -> Coupler:
    return Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
        sources={
            "atmos": (toy_source(ATMOS_GRID, 0.0), None),
            "ocean": (toy_source(OCEAN_GRID, 2.0), None),
        },
        **kwargs,
    )


# -- construction -----------------------------------------------------------


def test_coupler_reads_cadences_and_derives_the_sequence():
    coupler = _atmos_ocean()
    runner: Runner = coupler  # structurally a Runner
    assert runner.supports_member_batching is False
    assert coupler.cadences == {
        "atmos": np.timedelta64(6, "h"),
        "ocean": np.timedelta64(48, "h"),
    }
    assert coupler.dt == np.timedelta64(6, "h")
    assert coupler.sequence_derived
    assert [slot.models for slot in coupler.sequence.slots] == [("atmos",), ("ocean",)]
    assert "Coupler(models=['atmos', 'ocean']" in repr(coupler)


def test_describe_lists_models_exchanges_and_sequence():
    coupler = _atmos_ocean()
    text = coupler.describe()
    for needle in (
        "atmos",
        "z1000:mean:48h",
        "BilinearRegridder",
        "lagged",
        "same step",
        "@2d",
    ):
        assert needle in text
    html = coupler._repr_html_()
    assert "<table>" in html and "ocean-&gt;atmos" in html


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"models": {}}, "at least one model"),
        ({"cadence": {"zz": "6h"}}, "cadence names unknown"),
        ({"sources": {"zz": None}}, "sources names unknown"),
        ({"dt": "5h"}, "not a multiple of the clock step"),
        ({"connectors": [Connector("ocean", "nope")]}, "unknown model 'nope'"),
        (
            {"connectors": [Connector("ocean", "atmos", ["nope"])]},
            "does not export 'nope'",
        ),
        (
            {"connectors": [Connector("ocean", "atmos", ["sst:mean:7h"])]},
            "not a multiple of 'ocean'",
        ),
        (
            {"connectors": [Connector("ocean", "atmos", ["sst:median:48h"])]},
            "not a valid qualified",
        ),
        (
            {
                "connectors": [
                    Connector("ocean", "atmos"),
                    Connector("ocean", "atmos", ["sst"]),
                ]
            },
            "from both",
        ),
    ],
)
def test_construction_errors(kwargs, match):
    base = dict(
        models={"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
        sources={
            "atmos": toy_source(ATMOS_GRID, 0.0),
            "ocean": toy_source(OCEAN_GRID, 2.0),
        },
    )
    with pytest.raises(CouplingError, match=match):
        Coupler(**{**base, **kwargs})


def test_connector_without_matching_labels_is_rejected():
    with pytest.raises(CouplingError, match="declares no forcing or state variable"):
        Coupler(
            {"a": ToyPrognostic(["z1000"]), "b": ToyPrognostic(["sst"])},
            connectors=[Connector("a", "b")],
            sources={"a": toy_source(ATMOS_GRID, 0), "b": toy_source(ATMOS_GRID, 0)},
        )
    with pytest.raises(
        CouplingError, match="declares no forcing or state variable 'tp'"
    ):
        Coupler(
            {"a": ToyPrognostic(["tp"]), "b": ToyPrognostic(["sst"])},
            connectors=[Connector("a", "b", ["tp"])],
        )


def test_unserved_import_slot_needs_a_source_and_a_diagnostic_needs_a_cadence():
    with pytest.raises(CouplingError, match="no cadence"):
        Coupler({"impact": ToyImpact()})
    with pytest.raises(CouplingError, match="needs \\['z1000'\\] from a data source"):
        Coupler({"impact": ToyImpact()}, cadence={"impact": "6h"})
    with pytest.raises(CouplingError, match="forcing slot 0 needs \\['sst'\\]"):
        Coupler(
            {"atmos": ToyAtmosphere()}, sources={"atmos": toy_source(ATMOS_GRID, 0)}
        )
    with pytest.raises(CouplingError, match="2 input and forcing slots but 3"):
        Coupler({"atmos": ToyAtmosphere()}, sources={"atmos": (None, None, None)})


def test_explicit_sequence_is_validated_against_dependencies():
    models = {"atmos": ToyAtmosphere(), "impact": ToyImpact()}
    sources = {"atmos": (toy_source(ATMOS_GRID, 0.0), toy_source(ATMOS_GRID, 2.0))}
    connectors = [Connector("atmos", "impact")]
    with pytest.raises(CouplingError, match="must advance before"):
        Coupler(
            models,
            connectors,
            sources,
            sequence=RunSequence((Slot("6h", ("impact", "atmos")),)),
        )
    coupler = Coupler(
        models,
        connectors,
        sources,
        sequence=RunSequence((Slot("6h", ("atmos", "impact")),)),
    )
    assert not coupler.sequence_derived


# -- execution --------------------------------------------------------------


def test_atmos_ocean_values_are_hand_computable():
    coupler = _atmos_ocean()
    steps = list(coupler.run_item(_item(96)))
    assert len(steps) == 17
    assert set(steps[0]) == {"atmos", "ocean"}
    assert steps[0]["atmos"].dims == ("time", "lead_time", "variable", "lat", "lon")
    np.testing.assert_array_equal(steps[0]["atmos"]["time"], [T0])
    # Atmosphere: z += 1 + 0.1 * sst, with sst = 2 until the ocean steps at 48 h.
    np.testing.assert_allclose(
        [_value(s, "atmos") for s in steps[:9]], 1.2 * np.arange(9)
    )
    # Ocean at 48 h: sst = 2 + 0.01 * mean(z over [0, 48h)) = 2 + 0.01 * 4.2.
    assert "ocean" in steps[8] and "ocean" not in steps[7]
    np.testing.assert_allclose(_value(steps[8], "ocean"), 2.042)
    np.testing.assert_array_equal(steps[8]["ocean"]["lead_time"], _hours(48))
    # The atmosphere then sees the new SST from its next step on.
    np.testing.assert_allclose(_value(steps[9], "atmos"), 9.6 + 1 + 0.2042)
    # Ocean at 96 h: mean of z over [48h, 96h) = frames 9.6 + 1.2042 * k.
    expected = np.mean([9.6 + 1.2042 * k for k in range(8)])
    np.testing.assert_allclose(_value(steps[16], "ocean"), 2.042 + 0.01 * expected)
    ocean = coupler.models["ocean"]
    assert ocean.calls == [("z1000:mean:48h", [48]), ("z1000:mean:48h", [96])]


def test_outputs_match_declared_stream_schemas():
    coupler = _atmos_ocean()
    item = _item(96)
    schemas = coupler.output_coords(item.final_lead_time)
    assert set(schemas) == {"atmos", "ocean"}
    np.testing.assert_array_equal(
        schemas["atmos"]["lead_time"], _hours(*range(0, 97, 6))
    )
    np.testing.assert_array_equal(schemas["ocean"]["lead_time"], _hours(0, 48, 96))
    assert schemas["atmos"].dims[-3:] == ("variable", "lat", "lon")
    assert schemas["atmos"].data.nbytes == 0
    for step in coupler.run_item(item):
        for stream, out in step.items():
            schema = schemas[stream]
            assert out.dims == tuple(d for d in schema.dims if d != "batch")
            assert set(out["lead_time"].values) <= set(schema["lead_time"].values)
    with pytest.raises(ValueError, match="not a multiple"):
        coupler.output_coords(np.timedelta64(7, "h"))


def test_diagnostic_runs_when_its_inputs_are_delivered():
    impact = ToyImpact()
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "impact": impact},
        connectors=[Connector("atmos", "impact")],
        sources={"atmos": (toy_source(ATMOS_GRID, 0.0), toy_source(ATMOS_GRID, 2.0))},
    )
    assert coupler.cadences["impact"] == np.timedelta64(6, "h")
    steps = list(coupler.run_item(_item(12)))
    assert [set(s) for s in steps] == [{"atmos", "impact"}] * 3
    assert impact.calls == 3
    np.testing.assert_allclose([_value(s, "impact") for s in steps], [0, 0.012, 0.024])
    assert steps[1]["impact"].dims == ("time", "lead_time", "variable", "lat", "lon")
    np.testing.assert_array_equal(steps[2]["impact"]["lead_time"], _hours(12))
    schema = coupler.output_coords(np.timedelta64(12, "h"))["impact"]
    np.testing.assert_array_equal(schema["lead_time"], _hours(0, 6, 12))
    np.testing.assert_array_equal(schema["variable"], ["ws10m"])


def test_diagnostic_skips_the_initial_condition_it_cannot_compute():
    class _Output(ToyPrognostic):
        def output_coords(self, input_coords):
            out = super().output_coords(input_coords)
            from earth2studio.utils import coord_array_like

            return coord_array_like(out, {"variable": np.array(["z1000", "ws10m"])})

    def advance(x, forcing):
        return xr.concat(
            [x + 1, 0.01 * x.assign_coords(variable=["ws10m"])], "variable"
        )

    class _Needs(ToyImpact):
        def input_coords(self):
            from earth2studio.utils import coord_array_like

            return coord_array_like(
                super().input_coords(), {"variable": np.array(["ws10m"])}
            )

        def output_coords(self, input_coords):
            from earth2studio.utils import coord_array_like

            return coord_array_like(input_coords, {"variable": np.array(["gust"])})

    coupler = Coupler(
        {"atmos": _Output(["z1000"], advance=advance), "gust": _Needs()},
        connectors=[Connector("atmos", "gust")],
        sources={"atmos": toy_source(ATMOS_GRID, 0.0)},
    )
    steps = list(coupler.run_item(_item(6)))
    assert set(steps[0]) == {"atmos"}
    assert set(steps[1]) == {"atmos", "gust"}


def test_state_overwrite_couples_models_without_forcing_slots():
    seen = []

    def advance(x, forcing):
        sst = x.sel(variable=["sst"])
        seen.append(float(sst.isel(lat=1, lon=1).values.ravel()[0]))
        z = (
            x.sel(variable=["z1000"])
            + 1.0
            + 0.1 * sst.assign_coords(variable=["z1000"])
        )
        return xr.concat([z, sst], "variable")

    atmos = ToyPrognostic(["z1000", "sst"], advance=advance)
    coupler = Coupler(
        {"atmos": atmos, "ocean": ToyOcean()},
        connectors=[Connector("ocean", "atmos"), Connector("atmos", "ocean")],
        sources={
            "atmos": toy_source(ATMOS_GRID, 0.0),
            "ocean": toy_source(OCEAN_GRID, 2.0),
        },
    )
    assert "state" in coupler.describe()
    steps = list(coupler.run_item(_item(60)))
    # Fetched sst of 0 is overwritten by the ocean's 2 from the first step on,
    # and by its 48 h update from the step after the ocean advanced.
    assert seen[0] == 2.0 and seen[7] == 2.0
    np.testing.assert_allclose(seen[8], 2.042)
    # The ocean's state variable travels with the atmosphere's output stream.
    np.testing.assert_allclose(
        float(
            steps[9]["atmos"].sel(variable="sst").isel(lat=1, lon=1).values.ravel()[0]
        ),
        2.042,
    )


def test_same_step_forcing_runs_the_source_first():
    a = ToyPrognostic(["z1000"])
    lagged = ToyPrognostic(["sst"], forcing=["z1000"], forcing_lead="0h")
    same = ToyPrognostic(["sst"], forcing=["z1000"], forcing_lead="6h")
    sources = {"a": toy_source(ATMOS_GRID, 0.0), "b": toy_source(ATMOS_GRID, 0.0)}
    coupler = Coupler({"b": lagged, "a": a}, [Connector("a", "b")], sources)
    assert coupler.sequence.slots[0].models == ("b", "a")
    list(coupler.run_item(_item(12)))
    assert lagged.calls == [("z1000", [0]), ("z1000", [6])]
    coupler = Coupler({"b": same, "a": a}, [Connector("a", "b")], sources)
    assert coupler.sequence.slots[0].models == ("a", "b")
    list(coupler.run_item(_item(12)))
    assert same.calls == [("z1000", [6]), ("z1000", [12])]


def test_explicit_sequence_lagging_a_same_step_dependency_fails_at_runtime():
    a = ToyPrognostic(["z1000"])
    b = ToyPrognostic(["sst"], forcing=["z1000"], forcing_lead="6h")
    coupler = Coupler(
        {"a": a, "b": b},
        [Connector("a", "b")],
        {"a": toy_source(ATMOS_GRID, 0.0), "b": toy_source(ATMOS_GRID, 0.0)},
    )
    # Bypass validation to show the runtime check names the fix.
    coupler.sequence = RunSequence((Slot("6h", ("b", "a")),))
    with pytest.raises(CouplingError, match="advance 'a' first"):
        list(coupler.run_item(_item(6)))


def test_forcing_falls_back_to_a_source_before_the_coupled_model_has_frames():
    a = ToyPrognostic(["z1000"])
    b = ToyPrognostic(["sst"], forcing=["z1000"], forcing_lead="-6h")
    forcing_source = toy_source(ATMOS_GRID, 5.0)
    coupler = Coupler(
        {"a": a, "b": b},
        [Connector("a", "b")],
        {
            "a": toy_source(ATMOS_GRID, 0.0),
            "b": (toy_source(ATMOS_GRID, 0.0), forcing_source),
        },
    )
    requests = coupler.data_requests(_item(12))
    assert [r.lead_time.astype("timedelta64[h]").tolist() for r in requests[2:]] == [
        [np.timedelta64(-6, "h")]
    ]
    list(coupler.run_item(_item(12)))
    assert b.calls == [("z1000", [-6]), ("z1000", [0])]
    with pytest.raises(CouplingError, match="Bind a source"):
        list(
            Coupler(
                {"a": a, "b": b},
                [Connector("a", "b")],
                {"a": toy_source(ATMOS_GRID, 0.0), "b": toy_source(ATMOS_GRID, 0.0)},
            ).run_item(_item(6))
        )


def test_ics_replace_fetched_initial_conditions():
    coupler = _atmos_ocean()
    ic = xr.DataArray(
        np.full((1, 1, *ATMOS_SHAPE), 10.0, dtype=np.float32),
        dims=("lead_time", "variable", "lat", "lon"),
        coords={
            "lead_time": _hours(0),
            "variable": ["z1000"],
            **ATMOS_GRID.coords(only_index=True),
        },
    )
    steps = list(coupler.run_item(_item(6), ics={"atmos": ic}))
    np.testing.assert_allclose(_value(steps[0], "atmos"), 10.0)
    np.testing.assert_allclose(_value(steps[1], "atmos"), 11.2)
    assert coupler.data_requests(_item(6)) is not None
    no_source = Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        [Connector("ocean", "atmos"), Connector("atmos", "ocean")],
        {"ocean": toy_source(OCEAN_GRID, 2.0)},
    )
    assert no_source.data_requests(_item(6)) is None
    with pytest.raises(CouplingError, match="no bound source"):
        list(no_source.run_item(_item(6)))
    for bad, match in (
        ({"zz": ic}, "unknown models"),
        ({"atmos": (ic, ic)}, "2 arrays but"),
        ({"atmos": ic.isel(lead_time=0, drop=True)}, "lead_time"),
        ({"atmos": 3}, "DataArrays"),
    ):
        with pytest.raises(CouplingError, match=match):
            list(coupler.run_item(_item(6), ics=bad))


def test_data_requests_cover_initial_conditions_and_source_fed_forcing():
    forcing = toy_source(ATMOS_GRID, 2.0)
    coupler = Coupler(
        {"atmos": ToyAtmosphere()},
        sources={"atmos": (toy_source(ATMOS_GRID, 0.0), forcing)},
    )
    requests = coupler.data_requests(_item(18))
    assert [sorted(map(str, r.variable)) for r in requests] == [
        ["z1000"],
        ["sst"],
        ["sst"],
        ["sst"],
    ]
    assert [
        r.lead_time.astype("timedelta64[h]").astype(int).tolist() for r in requests
    ] == [
        [0],
        [0],
        [6],
        [12],
    ]
    steps = list(coupler.run_item(_item(18)))
    np.testing.assert_allclose([_value(s, "atmos") for s in steps], 1.2 * np.arange(4))
    diagnostic = Coupler(
        {"impact": ToyImpact()},
        sources={"impact": toy_source(ATMOS_GRID, 3.0)},
        cadence={"impact": "6h"},
    )
    requests = diagnostic.data_requests(_item(12))
    assert [
        r.lead_time.astype("timedelta64[h]").astype(int).tolist() for r in requests
    ] == [
        [0],
        [6],
        [12],
    ]
    steps = list(diagnostic.run_item(_item(12)))
    np.testing.assert_allclose([_value(s, "impact") for s in steps], 0.03)


# -- regridding and transforms ----------------------------------------------


def test_fields_are_regridded_between_grids():
    coupler = _atmos_ocean()
    regridders = {
        c.name: type(p.regridder).__name__
        for c in coupler.connectors
        for providers in coupler._providers[c.dst].values()
        for p in providers.values()
    }
    assert regridders == {
        "ocean->atmos": "BilinearRegridder",
        "atmos->ocean": "BilinearRegridder",
    }
    steps = list(coupler.run_item(_item(48)))
    assert steps[8]["ocean"].shape[-2:] == OCEAN_SHAPE
    assert steps[8]["atmos"].shape[-2:] == ATMOS_SHAPE


def test_custom_regridder_must_match_both_grids():
    nearest = NearestRegridder(OCEAN_GRID, ATMOS_GRID)
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        [Connector("ocean", "atmos", regridder=nearest), Connector("atmos", "ocean")],
        {"atmos": toy_source(ATMOS_GRID, 0.0), "ocean": toy_source(OCEAN_GRID, 2.0)},
    )
    steps = list(coupler.run_item(_item(6)))
    np.testing.assert_allclose(_value(steps[1], "atmos"), 1.2)
    with pytest.raises(CouplingError, match="source_grid does not match"):
        Coupler(
            {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
            [
                Connector(
                    "ocean",
                    "atmos",
                    regridder=BilinearRegridder(ATMOS_GRID, ATMOS_GRID),
                )
            ],
            {
                "atmos": toy_source(ATMOS_GRID, 0.0),
                "ocean": toy_source(OCEAN_GRID, 2.0),
            },
        )


def test_transform_applies_after_regridding_and_is_checked():
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        [
            Connector("ocean", "atmos", transform=lambda x: x - 1.0),
            Connector("atmos", "ocean"),
        ],
        {"atmos": toy_source(ATMOS_GRID, 0.0), "ocean": toy_source(OCEAN_GRID, 2.0)},
    )
    steps = list(coupler.run_item(_item(6)))
    np.testing.assert_allclose(_value(steps[1], "atmos"), 1.1)
    for bad, match in (
        (lambda x: x.isel(lat=0), "dimensions"),
        (lambda x: x.assign_coords(lon=x["lon"] + 1), "changed the 'lon'"),
        (lambda x: x.values, "dimensions"),
    ):
        broken = Coupler(
            {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
            [Connector("ocean", "atmos", transform=bad), Connector("atmos", "ocean")],
            {
                "atmos": toy_source(ATMOS_GRID, 0.0),
                "ocean": toy_source(OCEAN_GRID, 2.0),
            },
        )
        with pytest.raises(CouplingError, match=match):
            list(broken.run_item(_item(6)))


def test_state_overwrite_requires_an_exact_grid_match():
    atmos = ToyPrognostic(["z1000", "sst"])
    ocean = ToyPrognostic(["sst"], step="48h", grid=OCEAN_GRID)
    # A regridder onto a grid other than the destination's is rejected up front.
    with pytest.raises(CouplingError, match="target_grid does not match"):
        Coupler(
            {"atmos": atmos, "ocean": ocean},
            [
                Connector(
                    "ocean",
                    "atmos",
                    regridder=NearestRegridder(OCEAN_GRID, toy_grid(8, 16)),
                )
            ],
            {
                "atmos": toy_source(ATMOS_GRID, 0.0),
                "ocean": toy_source(OCEAN_GRID, 2.0),
            },
        )


# -- runner behaviour -------------------------------------------------------


def test_member_batching_is_rejected_and_models_are_seeded():
    class _Stochastic(ToyPrognostic):
        stochastic = True

        def __init__(self):
            super().__init__(["z1000"])
            self.seeds = []

        def set_rng(self, seed, reset=True):
            self.seeds.append((seed, reset))

    model = _Stochastic()
    coupler = Coupler({"a": model}, sources={"a": toy_source(ATMOS_GRID, 0.0)}, seed=7)
    with pytest.raises(ValueError, match="batch"):
        list(coupler.run_item(WorkItem(T0, np.timedelta64(6, "h"), member_ids=(0, 1))))
    list(coupler.run_item(WorkItem(T0, np.timedelta64(6, "h"), member_ids=(3,))))
    assert model.seeds == [(member_seed(7, T0, 3), True)]


def test_to_moves_models_and_regridders():
    coupler = _atmos_ocean()
    assert coupler.to("cpu") is coupler
    assert str(coupler.device) == "cpu"
    list(coupler.run_item(_item(6)))


def test_pipeline_drives_the_coupler_like_any_runner():
    coupler = _atmos_ocean()
    atmos, ocean = KVBackend(), KVBackend()
    Pipeline(coupler, output={"atmos": atmos, "ocean": ocean}, device="cpu").run(
        work_items([T0], np.timedelta64(48, "h"))
    )
    assert atmos["z1000"].shape == (1, 9, *ATMOS_SHAPE)
    np.testing.assert_allclose(atmos["z1000"][0, :, 1, 1], 1.2 * np.arange(9))
    assert ocean["sst"].shape == (1, 2, *OCEAN_SHAPE)
    np.testing.assert_allclose(ocean["sst"][0, :, 1, 1], [2.0, 2.042])


def test_zero_final_lead_time_publishes_only_initial_conditions():
    coupler = _atmos_ocean()
    steps = list(coupler.run_item(_item(0)))
    assert len(steps) == 1 and set(steps[0]) == {"atmos", "ocean"}
    assert coupler.data_requests(_item(0)) is not None


def test_model_returning_wrong_number_of_outputs_is_rejected():
    class _Two(ToyPrognostic):
        def initialize(self, x, forcing=None):
            y, state = super().initialize(x, forcing)
            return (y, y), state

    with pytest.raises(CouplingError, match="returned 2 outputs"):
        list(
            Coupler(
                {"a": _Two(["z1000"])}, sources={"a": toy_source(ATMOS_GRID, 0.0)}
            ).run_item(_item(6))
        )

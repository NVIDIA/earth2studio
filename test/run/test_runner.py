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

"""The built-in single-model runner.

The model is FCN's real DataArray execution path with an add-one core, and the
source returns zeros, so step ``k`` holds the value ``k`` everywhere.
"""

from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.run as run
from earth2studio.data import CompositeSource, RegriddedSource
from earth2studio.grids import BilinearRegridder, LatLonGrid, infer_grid
from earth2studio.io import KVBackend
from earth2studio.io.utils import output_template
from earth2studio.models.dx import Identity
from earth2studio.models.px.fcn import FCN
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.perturbation import (
    BredVector,
    Brown,
    Gaussian,
    HemisphericCentredBredVector,
    LaggedEnsemble,
)
from earth2studio.run import DiagnosticRunner, PrognosticRunner, Runner, WorkItem
from earth2studio.run.runner import _signature_for_stream
from earth2studio.run.work import member_seed
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS
from earth2studio.utils.type import CoordinateSystem

T0 = np.datetime64("2026-01-01T00")


class _AddOne(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _TinyFCN(FCN):
    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": ["u10m", "v10m"],
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
            dynamic=("batch",),
        )


class _StochasticFCN(_TinyFCN):
    """Stochastic model recording the seeds the runner sets on it."""

    stochastic = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.seeds: list[tuple[int, bool]] = []

    def set_rng(self, seed: int, reset: bool = True) -> None:
        self.seeds.append((seed, reset))


class _UnseededFCN(_StochasticFCN):
    """Declares no randomness, so the runner must leave its stream alone."""

    stochastic = False


class _Zeros:
    fill = 0.0

    def __init__(self) -> None:
        self.times: list[np.datetime64] = []

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        self.times.extend(time)
        return xr.DataArray(
            np.full((len(time), len(variable), 2, 3), self.fill, dtype=np.float32),
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
        )


class _Ones(_Zeros):
    fill = 1.0


class _WindSpeed:
    """Diagnostic computing 10 m wind speed from its components."""

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": ["u10m", "v10m"], "lat": [10, 0], "lon": [0, 10, 20]},
            dynamic=("batch",),
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(input_coords, {"variable": ["ws10m"]})

    def to(self, device: torch.device) -> "_WindSpeed":
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        speed = np.hypot(x.sel(variable="u10m"), x.sel(variable="v10m"))
        return speed.expand_dims(variable=["ws10m"], axis=x.get_axis_num("variable"))


class _StochasticWindSpeed(_WindSpeed):
    """Stochastic diagnostic recording the seeds the runner sets on it."""

    stochastic = True

    def __init__(self) -> None:
        self.seeds: list[tuple[int, bool]] = []

    def set_rng(self, seed: int, reset: bool = True) -> None:
        self.seeds.append((seed, reset))


class _Forced(PrognosticMixin):
    """Migrated model adding its forcing; records the forcing leads it receives.

    With ``background`` hours, the input is a background that many hours before
    the analysis time, which ``initialize`` brings up to it before stepping 6 h.
    """

    def __init__(self, background: int = 0) -> None:
        self.background = np.timedelta64(background, "h")
        self.seen: list[list[np.timedelta64]] = []

    def input_coords(self) -> CoordinateSystem:
        return coord_array_like(
            _model().input_coords(), {"lead_time": np.array([-self.background])}
        )

    def forcing_coords(self) -> CoordinateSystem:
        return coord_array_like(self.input_coords(), {"variable": ["sst"]})

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        lead = input_coords["lead_time"].values[-1:] + self.background
        return coord_array_like(
            input_coords, {"lead_time": lead + np.timedelta64(6, "h")}
        )

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        self.seen.append(list(forcing["lead_time"].values))
        y = x + 1 + float(forcing.values.mean())
        return y.assign_coords(lead_time=x["lead_time"] + np.timedelta64(6, "h"))

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        x = x.assign_coords(lead_time=x["lead_time"] + self.background)
        return self._advance(x, forcing), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, forcing: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y, forcing), None

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x, forcing)

    def to(self, device: torch.device) -> "_Forced":
        return self


def _model() -> _TinyFCN:
    return _TinyFCN(_AddOne(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))


def _stochastic_model() -> _StochasticFCN:
    return _StochasticFCN(_AddOne(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))


def _item(hours: int = 18) -> WorkItem:
    return WorkItem(T0, np.timedelta64(hours, "h"))


def test_steps_through_final_lead_time() -> None:
    runner: Runner = PrognosticRunner(_model(), _Zeros())
    steps = list(runner.run_item(_item()))
    assert [set(s) for s in steps] == [{"forecast"}] * 4
    for k, step in enumerate(steps):
        np.testing.assert_array_equal(step["forecast"].values, k)
    leads = [s["forecast"]["lead_time"].values[-1] for s in steps]
    np.testing.assert_array_equal(
        leads, runner.output_coords(_item().final_lead_time)["forecast"]["lead_time"]
    )


def test_output_coords_are_allocation_free_stream_signatures() -> None:
    coords = PrognosticRunner(_model(), _Zeros()).output_coords(_item().final_lead_time)
    forecast = coords["forecast"]
    # One DataArray signature per stream, with lead_time leading the fixed dims
    # and time and ensemble left to the dynamic prefix.
    assert forecast.dims == ("batch", "lead_time", "variable", "lat", "lon")
    assert forecast.attrs[E2S_DYNAMIC_DIMS] == ("batch",)
    assert forecast.sizes["batch"] == 0
    assert forecast.data.nbytes == 0
    assert forecast.dtype == np.float32
    np.testing.assert_array_equal(
        forecast["lead_time"], np.arange(4) * np.timedelta64(6, "h")
    )
    np.testing.assert_array_equal(forecast["variable"], ["u10m", "v10m"])


def test_stream_signature_fixes_a_dynamic_lead_time() -> None:
    # Diagnostics such as WindGust declare lead_time dynamic; the stream fixes it.
    signature = coord_array(
        ("batch", "time", "lead_time", "variable", "lat", "lon"),
        {"variable": np.array(["u10m"]), "lat": np.arange(2.0), "lon": np.arange(3.0)},
        dynamic=("batch", "time", "lead_time"),
    )
    leads = np.arange(3) * np.timedelta64(6, "h")
    stream = _signature_for_stream(signature, leads)
    assert stream.dims == ("batch", "time", "lead_time", "variable", "lat", "lon")
    assert stream.attrs[E2S_DYNAMIC_DIMS] == ("batch", "time")
    np.testing.assert_array_equal(stream["lead_time"], leads)


def test_output_coords_round_trip_to_io_templates() -> None:
    runner = PrognosticRunner(_model(), _Zeros())
    signature = runner.output_coords(_item().final_lead_time)["forecast"]
    template = output_template(
        signature,
        leading={"ensemble": np.array([0, 1]), "time": np.array([T0])},
    )
    assert template.dims == ("ensemble", "time", "lead_time", "variable", "lat", "lon")
    io = KVBackend()
    io.add_array(template)
    for step in PrognosticRunner(_model(), _Zeros()).run_item(_item()):
        io.write(step["forecast"].expand_dims(ensemble=[0]))
    assert io["u10m"].shape == (2, 1, 4, 2, 3)
    np.testing.assert_array_equal(io["u10m"][0, 0, :, 0, 0], np.arange(4))


def test_matches_deterministic_workflow() -> None:
    model, source = _model(), _Zeros()
    # Pin CPU: deterministic moves the shared model to CUDA when available, and the
    # runner then follows the model's device.
    io = run.deterministic(
        [T0], 3, model, source, KVBackend(), device=torch.device("cpu"), verbose=False
    )
    assert PrognosticRunner(model, source).device == torch.device("cpu")
    steps = list(PrognosticRunner(model, source).run_item(_item()))
    written = io["u10m"][0]  # time, lead_time, lat, lon
    for k, step in enumerate(steps):
        np.testing.assert_array_equal(
            written[k], step["forecast"].sel(variable="u10m").values.squeeze()
        )


def test_diagnostics_publish_their_own_streams() -> None:
    runner = PrognosticRunner(_model(), _Zeros(), diagnostics={"copy": Identity()})
    coords = runner.output_coords(_item().final_lead_time)
    assert set(coords) == {"forecast", "copy"}
    np.testing.assert_array_equal(
        coords["copy"]["lead_time"], coords["forecast"]["lead_time"]
    )
    last = list(runner.run_item(_item()))[-1]
    xr.testing.assert_equal(last["copy"], last["forecast"])


def test_requests_are_per_item_and_match_fetch() -> None:
    later = WorkItem(np.datetime64("2026-02-01"), np.timedelta64(6, "h"))
    (request,) = PrognosticRunner(_model(), _Zeros()).data_requests(later)
    np.testing.assert_array_equal(request.time, [later.time])
    np.testing.assert_array_equal(request.variable, ["u10m", "v10m"])


def test_rejects_unreachable_final_lead_times() -> None:
    with pytest.raises(ValueError):
        list(PrognosticRunner(_model(), _Zeros()).run_item(_item(7)))


def test_forecast_stream_name_is_reserved() -> None:
    with pytest.raises(ValueError):
        PrognosticRunner(_model(), _Zeros(), diagnostics={"forecast": Identity()})


def test_initial_condition_only_for_zero_final_lead_time() -> None:
    steps = list(PrognosticRunner(_model(), _Zeros()).run_item(_item(0)))
    assert len(steps) == 1
    np.testing.assert_array_equal(steps[0]["forecast"].values, 0)


class _Decoded(PrognosticMixin):
    """Model whose outputs differ from its inputs: winds in, wind speed out."""

    def input_coords(self) -> CoordinateSystem:
        return _model().input_coords()

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(
            input_coords,
            {
                "lead_time": input_coords["lead_time"].values[-1:]
                + np.timedelta64(6, "h"),
                "variable": ["ws10m"],
            },
        )

    def _advance(self, x: xr.DataArray) -> xr.DataArray:
        y = x.isel(lead_time=[-1], variable=[0]) + 1
        return y.assign_coords(
            lead_time=y["lead_time"] + np.timedelta64(6, "h"), variable=["ws10m"]
        )

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:  # type: ignore[override]
        return self._advance(x), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        y = y.assign_coords(variable=["u10m"])
        return self._advance(y), None

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:  # type: ignore[override]
        return self._default_create_iterator(x)

    def to(self, device: torch.device) -> "_Decoded":
        return self


def test_outputs_unlike_inputs_start_at_the_first_forecast() -> None:
    runner = PrognosticRunner(_Decoded(), _Zeros())
    assert not runner.publishes_initial_condition
    assert PrognosticRunner(_model(), _Zeros()).publishes_initial_condition
    steps = list(runner.run_item(_item(12)))
    hours = [s["forecast"]["lead_time"].values[0] for s in steps]
    assert np.array(hours).astype("timedelta64[h]").astype(int).tolist() == [6, 12]
    planned = runner.output_coords(np.timedelta64(12, "h"))["forecast"]["lead_time"]
    assert planned.values.astype("timedelta64[h]").astype(int).tolist() == [6, 12]
    assert [str(v) for v in steps[0]["forecast"]["variable"].values] == ["ws10m"]
    assert list(runner.run_item(_item(0))) == []


def test_background_before_analysis_steps_at_forecast_interval() -> None:
    source = _Zeros()
    runner = PrognosticRunner(_Forced(background=9), source, forcing=(_Zeros(),))
    assert runner.step == np.timedelta64(6, "h")
    assert not runner.publishes_initial_condition
    assert runner.nsteps(np.timedelta64(18, "h")) == 3
    steps = list(runner.run_item(_item(18)))
    hours = [
        int(s["forecast"]["lead_time"].values[0] // np.timedelta64(1, "h"))
        for s in steps
    ]
    assert hours == [6, 12, 18]
    planned = runner.output_coords(np.timedelta64(18, "h"))["forecast"]["lead_time"]
    assert (planned.values // np.timedelta64(1, "h")).tolist() == hours
    assert source.times == [T0 - np.timedelta64(9, "h")]


@pytest.mark.parametrize(
    "workflow",
    [
        lambda m: run.deterministic([T0], 2, m, _Zeros(), KVBackend(), verbose=False),
        lambda m: run.diagnostic(
            [T0], 2, m, _WindSpeed(), _Zeros(), KVBackend(), verbose=False
        ),
        lambda m: run.ensemble(
            [T0], 2, 2, m, _Zeros(), KVBackend(), lambda x, c: (x, c), verbose=False
        ),
    ],
    ids=["deterministic", "diagnostic", "ensemble"],
)
def test_workflows_reject_a_background_before_the_analysis_time(
    workflow: Callable[[Any], object],
) -> None:
    with pytest.raises(ValueError, match="PrognosticRunner"):
        workflow(_Forced(background=9))


def test_forcing_is_fetched_requested_and_sent_each_step() -> None:
    model = _Forced()
    runner = PrognosticRunner(model, _Zeros(), forcing=(_Zeros(),))
    steps = list(runner.run_item(_item(18)))
    assert len(steps) == 4
    hours = [[int(v / np.timedelta64(1, "h")) for v in leads] for leads in model.seen]
    assert hours == [[0], [6], [12]]
    requests = runner.data_requests(_item(18))
    requested = [list(r.lead_time // np.timedelta64(1, "h")) for r in requests[1:]]
    assert requested == hours


class _CoarseOnes:
    """Ones on a grid coarser in longitude than the model's."""

    grid = LatLonGrid(np.array([10.0, 0.0]), np.array([0.0, 20.0]))

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        return xr.DataArray(
            np.ones((len(time), len(variable), *self.grid.shape), dtype=np.float32),
            dims=("time", "variable", *self.grid.dims),
            coords={"time": time, "variable": variable, **self.grid.coords()},
        )


def test_off_grid_forcing_runs_through_regridded_source() -> None:
    target = infer_grid(_Forced().forcing_coords())
    forcing = RegriddedSource(
        _CoarseOnes(), BilinearRegridder(_CoarseOnes.grid, target)
    )
    runner = PrognosticRunner(_Forced(), _Zeros(), forcing=(forcing,))
    assert runner.data_requests(_item(6))[1].source is forcing
    steps = list(runner.run_item(_item(12)))
    # Each step adds one plus the mean regridded forcing, which is one.
    np.testing.assert_allclose(steps[-1]["forecast"], 4.0)


def test_missing_forcing_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        PrognosticRunner(_Forced(), _Zeros())


def test_negative_final_lead_time_is_rejected_before_fetching() -> None:
    with pytest.raises(ValueError):
        PrognosticRunner(_model(), _Zeros()).nsteps(np.timedelta64(-6, "h"))


def test_device_follows_the_model() -> None:
    assert PrognosticRunner(_model(), _Zeros()).device == torch.device("cpu")


def test_diagnostic_runner_steps_through_source_times() -> None:
    source = _Ones()
    runner: Runner = DiagnosticRunner(
        {"wind": _WindSpeed()}, source, step=np.timedelta64(6, "h")
    )
    steps = list(runner.run_item(_item(12)))
    assert [set(s) for s in steps] == [{"wind"}] * 3
    np.testing.assert_allclose(steps[-1]["wind"].values, np.hypot(1, 1))
    np.testing.assert_array_equal(steps[-1]["wind"]["variable"], ["ws10m"])
    hours = [int((t - T0) / np.timedelta64(1, "h")) for t in source.times]
    assert hours == [0, 6, 12]
    coords = runner.output_coords(_item(12).final_lead_time)["wind"]
    np.testing.assert_array_equal(
        coords["lead_time"], np.array([0, 6, 12], dtype="timedelta64[h]")
    )
    assert list(coords["variable"]) == ["ws10m"]
    requests = runner.data_requests(_item(12))
    assert [r.lead_time.tolist() for r in requests] == [
        [lead] for lead in coords["lead_time"].values.tolist()
    ]


def test_diagnostic_runner_without_step_runs_one_step() -> None:
    runner = DiagnosticRunner({"wind": _WindSpeed()}, _Ones())
    assert len(list(runner.run_item(_item(0)))) == 1
    with pytest.raises(ValueError):
        list(runner.run_item(_item(6)))


def test_diagnostic_runner_needs_a_source() -> None:
    with pytest.raises(ValueError):
        DiagnosticRunner({"wind": _WindSpeed()})


def test_diagnostic_runner_needs_declared_input_variables() -> None:
    with pytest.raises(ValueError):
        DiagnosticRunner({"copy": Identity()}, _Ones())


def test_every_stochastic_model_is_seeded_per_member() -> None:
    model, diagnostic = _stochastic_model(), _StochasticWindSpeed()
    runner = PrognosticRunner(model, _Zeros(), diagnostics={"wind": diagnostic}, seed=7)
    for member in (0, 3):
        item = WorkItem(T0, np.timedelta64(6, "h"), member_ids=(member,))
        list(runner.run_item(item))
    expected = [(member_seed(7, T0, member), True) for member in (0, 3)]
    assert model.seeds == expected
    assert diagnostic.seeds == expected
    assert len({seed for seed, _ in expected}) == 2


def test_seeding_defaults_to_base_zero_and_member_zero() -> None:
    model = _stochastic_model()
    runner = PrognosticRunner(model, _Zeros())
    assert runner.seed == 0
    list(runner.run_item(_item(6)))  # No member_ids: the single member is member 0.
    assert model.seeds == [(member_seed(0, T0, 0), True)]


def test_seeding_leaves_models_without_randomness_and_global_rng_alone() -> None:
    model = _UnseededFCN(_AddOne(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))
    before = torch.random.get_rng_state()
    list(PrognosticRunner(model, _Zeros(), seed=11).run_item(_item(6)))
    assert model.seeds == []
    list(PrognosticRunner(_stochastic_model(), _Zeros(), seed=11).run_item(_item(6)))
    assert torch.equal(torch.random.get_rng_state(), before)


def test_diagnostic_runner_seeds_stochastic_diagnostics() -> None:
    diagnostic = _StochasticWindSpeed()
    runner = DiagnosticRunner({"wind": diagnostic}, _Ones(), seed=5)
    item = WorkItem(T0, np.timedelta64(0, "h"), member_ids=(2,))
    list(runner.run_item(item))
    assert diagnostic.seeds == [(member_seed(5, T0, 2), True)]


class _Members(_Zeros):
    """Ensemble-aware source: member ``k`` holds the value ``k`` everywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.ensembles: list[list[int] | None] = []

    def __call__(  # type: ignore[override]
        self, time: np.ndarray, variable: np.ndarray, ensemble: list | None = None
    ) -> xr.DataArray:
        x = super().__call__(time, variable)
        self.ensembles.append(None if ensemble is None else list(ensemble))
        if ensemble is None:
            return x
        members = np.asarray(ensemble)
        offset = xr.DataArray(members.astype(np.float32), dims="ensemble")
        return (x + offset).assign_coords(ensemble=members)


def _member_item(member: int, hours: int = 6) -> WorkItem:
    return WorkItem(T0, np.timedelta64(hours, "h"), member_ids=(member,))


def test_members_read_their_source_member() -> None:
    source = _Members()
    runner = PrognosticRunner(_model(), source, ensemble_member=lambda m: m // 2 + 1)
    for member in range(4):
        steps = list(runner.run_item(_member_item(member)))
        expected = member // 2 + 1
        np.testing.assert_array_equal(steps[0]["forecast"].values, expected)
        np.testing.assert_array_equal(steps[1]["forecast"].values, expected + 1)
        assert "ensemble" not in steps[0]["forecast"].dims
        (request,) = runner.data_requests(_member_item(member))
        np.testing.assert_array_equal(request.ensemble, [expected])
    assert source.ensembles == [[1], [1], [2], [2]]


def test_forcing_reads_the_member_while_other_slots_are_shared() -> None:
    model, initial, forcing = _Forced(), _Zeros(), _Members()
    runner = PrognosticRunner(model, initial, forcing=(forcing,), ensemble_member=int)
    steps = list(runner.run_item(_member_item(3, hours=12)))
    # Each step adds one plus the forcing, member 3's value.
    np.testing.assert_allclose(steps[-1]["forecast"], 8.0)
    assert forcing.ensembles == [[3], [3]]
    requests = runner.data_requests(_member_item(3, hours=12))
    assert requests[0].ensemble is None
    assert [r.ensemble.tolist() for r in requests[1:]] == [[3], [3]]


def test_members_share_sources_without_ensemble_member() -> None:
    source = _Members()
    steps = list(PrognosticRunner(_model(), source).run_item(_member_item(2)))
    np.testing.assert_array_equal(steps[0]["forecast"].values, 0)
    assert source.ensembles == [None]


def test_ensemble_member_needs_member_ids() -> None:
    runner = PrognosticRunner(_model(), _Members(), ensemble_member=int)
    with pytest.raises(ValueError, match="no member IDs"):
        list(runner.run_item(_item(6)))
    with pytest.raises(ValueError, match="no member IDs"):
        runner.data_requests(_item(6))


def test_diagnostic_runner_reads_the_member() -> None:
    source = _Members()
    runner = DiagnosticRunner(
        {"wind": _WindSpeed()}, source, ensemble_member=lambda m: m + 1
    )
    (step,) = list(runner.run_item(WorkItem(T0, np.timedelta64(0, "h"), (1,))))
    np.testing.assert_allclose(step["wind"].values, np.hypot(2, 2))
    assert source.ensembles == [[2]]
    (request,) = runner.data_requests(WorkItem(T0, np.timedelta64(0, "h"), (1,)))
    np.testing.assert_array_equal(request.ensemble, [2])


def test_composite_initial_conditions_read_the_member() -> None:
    members = _Members()
    source = CompositeSource(
        {"ens": members, "shared": _Zeros()}, {"ens": ["u10m"], "shared": ["v10m"]}
    )
    runner = PrognosticRunner(_model(), source, ensemble_member=lambda m: m + 1)
    step, _ = list(runner.run_item(_member_item(2)))
    np.testing.assert_array_equal(step["forecast"].sel(variable="u10m"), 3)
    np.testing.assert_array_equal(step["forecast"].sel(variable="v10m"), 0)
    assert members.ensembles == [[3]]


def test_batches_lay_members_along_a_leading_ensemble_dimension() -> None:
    model, initial, forcing = _Forced(), _Members(), _Zeros()
    runner = PrognosticRunner(
        model, initial, forcing=(forcing,), ensemble_member=lambda m: m // 2 + 1
    )
    item = WorkItem(T0, np.timedelta64(12, "h"), member_ids=(1, 2, 3))
    assert runner.supports_member_batching
    steps = list(runner.run_item(item))
    x = steps[-1]["forecast"]
    assert x.dims[0] == "ensemble"
    np.testing.assert_array_equal(x["ensemble"], [1, 2, 3])
    # Members 2 and 3 share source member 2, fetched once per call.
    np.testing.assert_array_equal(x.sel(ensemble=1), 1 + 2)
    np.testing.assert_array_equal(x.sel(ensemble=3), 2 + 2)
    assert initial.ensembles == [[1, 2]]
    request, *_ = runner.data_requests(item)
    np.testing.assert_array_equal(request.ensemble, [1, 2])
    # The shared forcing reaches the model with the batch's members.
    for step in steps:
        assert step["forecast"]["ensemble"].size == 3


def test_batched_members_match_single_member_runs() -> None:
    runner = PrognosticRunner(
        _model(),
        _Members(),
        diagnostics={"wind": _WindSpeed()},
        ensemble_member=lambda m: m,
    )
    batched = list(runner.run_item(WorkItem(T0, np.timedelta64(6, "h"), (4, 9))))
    for member in (4, 9):
        single = list(runner.run_item(_member_item(member)))
        for many, one in zip(batched, single):
            for stream in ("forecast", "wind"):
                xr.testing.assert_identical(
                    many[stream].sel(ensemble=member, drop=True), one[stream]
                )


def test_diagnostic_runner_batches_members() -> None:
    runner = DiagnosticRunner({"wind": _WindSpeed()}, _Ones())
    (step,) = list(runner.run_item(WorkItem(T0, np.timedelta64(0, "h"), (0, 1))))
    assert step["wind"].dims[0] == "ensemble"
    np.testing.assert_allclose(step["wind"].values, np.hypot(1, 1))


class _GlobalNoise:
    """Perturbation without ``set_rng``, drawing from the global Torch RNG."""

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        return x.copy(data=x.values + torch.randn(x.shape).numpy())


@pytest.mark.parametrize(
    "perturbation", [Gaussian, _GlobalNoise], ids=["set_rng", "global"]
)
def test_perturbation_draws_each_member_alone(perturbation: type) -> None:
    runner = PrognosticRunner(_model(), _Zeros(), perturbation=perturbation())
    before = torch.random.get_rng_state()
    batched = list(runner.run_item(WorkItem(T0, np.timedelta64(6, "h"), (4, 9))))
    assert torch.equal(torch.random.get_rng_state(), before)
    initial = batched[0]["forecast"]
    assert not np.allclose(initial.sel(ensemble=4), initial.sel(ensemble=9))
    assert not np.allclose(initial, 0)
    for member in (4, 9):
        single = list(runner.run_item(_member_item(member)))
        for many, one in zip(batched, single):
            xr.testing.assert_identical(
                many["forecast"].sel(ensemble=member, drop=True), one["forecast"]
            )
    # Deterministic items perturb as member 0.
    xr.testing.assert_identical(
        list(runner.run_item(_item(0)))[0]["forecast"],
        list(runner.run_item(_member_item(0, hours=0)))[0]["forecast"],
    )


@pytest.mark.parametrize(
    "perturbation",
    [
        lambda: BredVector(lambda x: x),
        lambda: HemisphericCentredBredVector(_model(), _Zeros(), Brown()),
        lambda: LaggedEnsemble(_Zeros(), np.array([0], dtype="timedelta64[h]")),
    ],
    ids=["bred", "hcbv", "lagged"],
)
def test_runner_rejects_perturbations_needing_whole_ensembles(
    perturbation: Callable[[], Any],
) -> None:
    with pytest.raises(ValueError, match="not supported by the runners"):
        PrognosticRunner(_model(), _Zeros(), perturbation=perturbation())

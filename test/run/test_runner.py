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

from collections.abc import Iterator

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.run as run
from earth2studio.data import MemberSource, RegriddedSource
from earth2studio.grids import BilinearRegridder, LatLonGrid, infer_grid
from earth2studio.io import KVBackend
from earth2studio.io.utils import output_template
from earth2studio.models.dx import Identity
from earth2studio.models.px.fcn import FCN
from earth2studio.models.px.utils import PrognosticMixin
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
    """Migrated model adding its forcing; records the forcing leads it receives."""

    def __init__(self) -> None:
        self.seen: list[list[np.timedelta64]] = []

    def input_coords(self) -> CoordinateSystem:
        return _model().input_coords()

    def forcing_coords(self) -> CoordinateSystem:
        return coord_array_like(self.input_coords(), {"variable": ["sst"]})

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(
            input_coords, {"lead_time": np.array([6], dtype="timedelta64[h]")}
        )

    def _advance(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        self.seen.append(list(forcing["lead_time"].values))
        y = x + 1 + float(forcing.values.mean())
        return y.assign_coords(lead_time=x["lead_time"] + np.timedelta64(6, "h"))

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        return self._advance(x, forcing), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, forcing: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y, forcing), None

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x, forcing)


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


@pytest.mark.parametrize(
    "item",
    [
        WorkItem(T0, np.timedelta64(7, "h")),
        WorkItem(T0, np.timedelta64(6, "h"), member_ids=(0, 1)),
    ],
)
def test_rejects_unsupported_items(item: WorkItem) -> None:
    with pytest.raises(ValueError):
        list(PrognosticRunner(_model(), _Zeros()).run_item(item))


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


class _FromBackground(PrognosticMixin):
    """Model initialized from a background 9 h before the analysis time.

    ``initialize`` brings the background up to the analysis time and forecasts
    3 h; each ``step`` then advances 3 h.
    """

    def input_coords(self) -> CoordinateSystem:
        return coord_array_like(
            _model().input_coords(),
            {"lead_time": np.array([-9], dtype="timedelta64[h]")},
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(
            input_coords,
            {
                "lead_time": input_coords["lead_time"].values[-1:]
                + np.timedelta64(12, "h")
            },
        )

    def _advance(self, x: xr.DataArray, hours: int) -> xr.DataArray:
        return (x + 1).assign_coords(
            lead_time=x["lead_time"] + np.timedelta64(hours, "h")
        )

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:  # type: ignore[override]
        return self._advance(x, 12), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(y, 3), None

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:  # type: ignore[override]
        return self._default_create_iterator(x)

    def to(self, device: torch.device) -> "_FromBackground":
        return self


def test_background_before_analysis_steps_at_forecast_interval() -> None:
    source = _Zeros()
    runner = PrognosticRunner(_FromBackground(), source)
    assert runner.step == np.timedelta64(3, "h")
    assert not runner.publishes_initial_condition
    assert runner.nsteps(np.timedelta64(9, "h")) == 3
    steps = list(runner.run_item(_item(9)))
    hours = [
        int(s["forecast"]["lead_time"].values[0] // np.timedelta64(1, "h"))
        for s in steps
    ]
    assert hours == [3, 6, 9]
    planned = runner.output_coords(np.timedelta64(9, "h"))["forecast"]["lead_time"]
    assert (planned.values // np.timedelta64(1, "h")).tolist() == hours
    assert source.times == [T0 - np.timedelta64(9, "h")]


def test_workflows_reject_a_background_before_the_analysis_time() -> None:
    model = _FromBackground()
    with pytest.raises(ValueError, match="PrognosticRunner"):
        run.deterministic([T0], 2, model, _Zeros(), KVBackend(), verbose=False)
    with pytest.raises(ValueError, match="PrognosticRunner"):
        run.diagnostic(
            [T0], 2, model, _WindSpeed(), _Zeros(), KVBackend(), verbose=False  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="PrognosticRunner"):
        run.ensemble(
            [T0], 2, 2, model, _Zeros(), KVBackend(), lambda x, c: (x, c), verbose=False  # type: ignore[arg-type]
        )


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


class _Fill(_Zeros):
    def __init__(self, fill: float) -> None:
        super().__init__()
        self.fill = fill


def test_member_sources_are_resolved_per_member() -> None:
    initial = MemberSource(lambda m: _Fill(float(m)))
    forcing = MemberSource(lambda m: _Fill(100.0 * m))
    runner = PrognosticRunner(_Forced(), initial, forcing=(forcing,))
    for member in (0, 2):
        item = WorkItem(T0, np.timedelta64(12, "h"), member_ids=(member,))
        requests = runner.data_requests(item)
        assert requests[0].source is initial.select(member)
        assert all(r.source is forcing.select(member) for r in requests[1:])
        last = list(runner.run_item(item))[-1]["forecast"]
        # Two steps of +1 + forcing, from the member's initial condition
        np.testing.assert_allclose(last.values, member + 2 * (1 + 100.0 * member))
    deterministic = list(runner.run_item(_item(0)))
    np.testing.assert_array_equal(deterministic[0]["forecast"].values, 0)

    diagnostic = DiagnosticRunner({"ws10m": _WindSpeed()}, initial)
    (step,) = diagnostic.run_item(WorkItem(T0, np.timedelta64(0, "h"), (3,)))
    np.testing.assert_allclose(step["ws10m"].values, np.hypot(3, 3))


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

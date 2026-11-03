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

"""StormScopeRunner: coupled GOES to MRMS rollout on fake models.

The fake GOES model adds one per step and the fake MRMS model adds ten plus
the mean of the GOES conditioning it receives, so the values a step holds
show which frames each model saw.  Sources return constants on a 3 x 4
curvilinear grid, so every test runs on CPU without downloads.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import open_dict
from src.campaign import build_items, final_lead_time
from src.runners import build_runner
from src.runners.stormscope import StormScopeRunner, stormscope

from earth2studio.grids import CurvilinearGrid, Region
from earth2studio.io import KVBackend, ZarrBackend
from earth2studio.run import Pipeline, work_items
from earth2studio.run.work import member_seed
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem
from test.conftest import compose_cfg

T0 = np.datetime64("2023-12-05T12:00:00")
STEP = np.timedelta64(10, "m")
GOES_LEADS = np.array([-10, 0], dtype="timedelta64[m]")
MRMS_LEADS = np.array([-10, 0], dtype="timedelta64[m]")
GOES_VARS = ["abi07c", "abi13c", "abi01c"]
MRMS_VARS = ["refc", "glm_density"]
CONDITIONING_VARS = ["abi13c", "abi07c"]
LAT = np.array(
    [[40.0, 40.1, 40.2, 40.3], [39.0, 39.1, 39.2, 39.3], [38.0, 38.1, 38.2, 38.3]]
)
LON = np.array([[260.0, 261.0, 262.0, 263.0]] * 3)
GRID = CurvilinearGrid(
    LAT,
    LON,
    y=np.array([0.0, 3000.0, 6000.0]),
    x=np.array([0.0, 3000.0, 6000.0, 9000.0]),
)


class ConstantSource:
    """Data source returning ``fill`` everywhere on the model grid."""

    def __init__(self, fill: float, variables: list[str] | None = None) -> None:
        self.fill = fill
        self.variables = variables
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        if self.variables is not None:
            missing = set(variable.tolist()) - set(self.variables)
            if missing:
                raise KeyError(f"Source lacks {sorted(missing)}")
        self.calls.append((time, variable))
        signature = coord_array(
            ("time", "variable", "y", "x"),
            {"time": time, "variable": variable},
            grid=GRID,
        )
        return from_torch(
            torch.full(signature.shape, self.fill, dtype=torch.float32), signature
        )


class FakeGOES:
    """Unforced GOES model adding ``add`` per ten-minute step."""

    stochastic = True

    def __init__(self, add: float = 1.0, variables: list[str] | None = None) -> None:
        self.add = add
        self.variables = list(variables or GOES_VARS)
        self.seeds: list[int] = []
        self.regions: list[Region] = []
        self.fill = 100.0

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "time", "lead_time", "variable", "y", "x"),
            {"lead_time": GOES_LEADS, "variable": np.array(self.variables)},
            dynamic=("batch", "time"),
            grid=GRID,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        lead = input_coords["lead_time"].values[-1]
        return coord_array_like(input_coords, {"lead_time": np.array([lead + STEP])})

    def forcing_coords(self) -> None:
        return None

    def default_sources(self) -> ConstantSource:
        return ConstantSource(self.fill)

    def to(self, device: torch.device) -> FakeGOES:
        return self

    def set_rng(self, seed: int, reset: bool = True) -> None:
        self.seeds.append(seed)

    def with_region(self, region: Region) -> FakeGOES:
        self.regions.append(region)
        return self

    def _advance(self, x: xr.DataArray, increment: float) -> xr.DataArray:
        last = x.isel(lead_time=[-1])
        y = last + increment
        y = y.assign_coords(lead_time=last["lead_time"].values + STEP)
        y.attrs = dict(x.attrs)
        return y

    def create_iterator(self, x: xr.DataArray) -> Generator[xr.DataArray, Any, None]:
        y = self._advance(x, self.add)
        while True:
            yield y
            y = self._advance(y, self.add)


class FakeMRMS(FakeGOES):
    """Radar model adding ten plus the mean of the GOES conditioning it receives."""

    def __init__(self) -> None:
        super().__init__(add=10.0, variables=MRMS_VARS)
        self.conditioning: list[xr.DataArray] = []
        self.fill = 5.0

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "time", "lead_time", "variable", "y", "x"),
            {"lead_time": MRMS_LEADS, "variable": np.array(self.variables)},
            dynamic=("batch", "time"),
            grid=GRID,
        )

    def forcing_coords(self) -> CoordinateSystem:
        return coord_array_like(
            self.input_coords(), {"variable": np.array(CONDITIONING_VARS)}
        )

    def default_sources(self) -> tuple[ConstantSource, ConstantSource]:  # type: ignore[override]
        return ConstantSource(self.fill), ConstantSource(-1.0)

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> Generator[xr.DataArray, xr.DataArray, None]:
        self.conditioning.append(conditioning)
        y = self._advance(x, self.add + float(conditioning.mean()))
        while True:
            sent = yield y
            self.conditioning.append(sent)
            y = self._advance(y, self.add + float(sent.mean()))


def _runner(**kwargs: Any) -> StormScopeRunner:
    return StormScopeRunner(FakeGOES(), FakeMRMS(), seed=7, **kwargs)


def _steps(runner: StormScopeRunner, item: Any) -> list[dict[str, xr.DataArray]]:
    return [dict(step) for step in runner.run_item(item)]


def test_step_and_output_coords_cover_every_lead_time() -> None:
    runner = _runner()
    assert runner.step == STEP
    schemas = runner.output_coords(3 * STEP)
    assert set(schemas) == {"goes", "mrms"}
    for name, variables in (("goes", GOES_VARS), ("mrms", MRMS_VARS)):
        signature = schemas[name]
        assert signature.dims == ("batch", "time", "lead_time", "variable", "y", "x")
        np.testing.assert_array_equal(
            signature["lead_time"].values, np.arange(4) * STEP
        )
        assert signature["variable"].values.tolist() == variables
        assert signature.sizes["y"] == 3 and signature.sizes["x"] == 4
    with pytest.raises(ValueError):
        runner.output_coords(np.timedelta64(15, "m"))


def test_run_item_publishes_initial_conditions_then_coupled_steps() -> None:
    runner = _runner()
    steps = _steps(runner, work_items([T0], 2 * STEP)[0])
    assert len(steps) == 3
    for k, step in enumerate(steps):
        assert set(step) == {"goes", "mrms"}
        for x in step.values():
            assert x.dims == ("time", "lead_time", "variable", "y", "x")
            np.testing.assert_array_equal(x["lead_time"].values, [k * STEP])
            assert x["time"].values[0] == T0
    # GOES: observations 100, then +1 per step.
    for k, step in enumerate(steps):
        np.testing.assert_allclose(step["goes"].values, 100.0 + k)
    # MRMS: observations 5, first step conditioned on GOES observations (100),
    # the second on the first GOES forecast (101).
    np.testing.assert_allclose(steps[0]["mrms"].values, 5.0)
    np.testing.assert_allclose(steps[1]["mrms"].values, 5.0 + 10.0 + 100.0)
    np.testing.assert_allclose(steps[2]["mrms"].values, 115.0 + 10.0 + 101.0)


def test_mrms_is_conditioned_on_goes_forecast_frames_at_its_lead_times() -> None:
    runner = _runner()
    mrms = runner.mrms
    _steps(runner, work_items([T0], 3 * STEP)[0])
    initial, *frames = mrms.conditioning
    # Initialization: the GOES observation history, conditioning variables only.
    assert initial["variable"].values.tolist() == CONDITIONING_VARS
    np.testing.assert_array_equal(initial["lead_time"].values, MRMS_LEADS)
    np.testing.assert_allclose(initial.values, 100.0)
    # Each later step: the GOES forecast at the previous radar forecast's lead time.
    assert len(frames) == 2
    for k, frame in enumerate(frames, start=1):
        assert frame["variable"].values.tolist() == CONDITIONING_VARS
        np.testing.assert_array_equal(frame["lead_time"].values, [k * STEP])
        np.testing.assert_allclose(frame.values, 100.0 + k)
        assert frame.dims == ("time", "lead_time", "variable", "y", "x")


def test_zero_steps_publishes_only_initial_conditions() -> None:
    runner = _runner()
    steps = _steps(runner, work_items([T0], np.timedelta64(0, "h"))[0])
    assert len(steps) == 1
    assert runner.mrms.conditioning == []


def test_data_requests_name_the_fetches_and_reuse_goes_for_conditioning() -> None:
    runner = _runner()
    item = work_items([T0], 2 * STEP)[0]
    requests = runner.data_requests(item)
    assert [r.source for r in requests] == [runner.goes_source, runner.mrms_source]
    assert runner.conditioning_source is None
    np.testing.assert_array_equal(requests[0].variable, GOES_VARS)
    np.testing.assert_array_equal(requests[0].lead_time, GOES_LEADS)
    np.testing.assert_array_equal(requests[1].variable, MRMS_VARS)
    _steps(runner, item)
    # ``fetch_data`` calls a source once per lead time; the GOES observations
    # arrive once and double as the radar conditioning, never fetched twice.
    assert len(runner.goes_source.calls) == len(GOES_LEADS)
    assert len(runner.mrms_source.calls) == len(MRMS_LEADS)


def test_explicit_conditioning_source_is_fetched_separately() -> None:
    conditioning = ConstantSource(50.0)
    runner = _runner(conditioning_source=conditioning)
    item = work_items([T0], STEP)[0]
    requests = runner.data_requests(item)
    assert [r.source for r in requests] == [
        runner.goes_source,
        runner.mrms_source,
        conditioning,
    ]
    np.testing.assert_array_equal(requests[2].variable, CONDITIONING_VARS)
    steps = _steps(runner, item)
    np.testing.assert_allclose(steps[1]["mrms"].values, 5.0 + 10.0 + 50.0)
    assert len(conditioning.calls) == len(requests[2].lead_time)


def test_conditioning_falls_back_to_mrms_default_when_goes_lacks_variables() -> None:
    goes = FakeGOES(variables=["abi01c"])
    runner = StormScopeRunner(goes, FakeMRMS())
    assert isinstance(runner.conditioning_source, ConstantSource)
    assert runner.conditioning_source.fill == -1.0


def test_models_are_seeded_per_member_and_batching_is_rejected() -> None:
    runner = _runner()
    items = work_items([T0], STEP, members=[2, 5], member_batch=1)
    for item in items:
        _steps(runner, item)
    expected = [member_seed(7, T0, 2), member_seed(7, T0, 5)]
    assert runner.goes.seeds == expected
    assert runner.mrms.seeds == expected
    assert runner.supports_member_batching is False
    with pytest.raises(ValueError, match="batch"):
        _steps(runner, work_items([T0], STEP, members=[0, 1], member_batch=2)[0])


def test_mismatched_grids_or_strides_are_rejected() -> None:
    class OtherGrid(FakeMRMS):
        def input_coords(self) -> CoordinateSystem:
            return coord_array(
                ("batch", "time", "lead_time", "variable", "y", "x"),
                {"lead_time": MRMS_LEADS, "variable": np.array(MRMS_VARS)},
                dynamic=("batch", "time"),
                grid=CurvilinearGrid(LAT, LON),
            )

    with pytest.raises(ValueError, match="grid"):
        StormScopeRunner(FakeGOES(), OtherGrid())

    class Hourly(FakeMRMS):
        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            lead = input_coords["lead_time"].values[-1]
            return coord_array_like(
                input_coords, {"lead_time": np.array([lead + np.timedelta64(1, "h")])}
            )

    with pytest.raises(ValueError, match="stride"):
        StormScopeRunner(FakeGOES(), Hourly())


def test_pipeline_writes_both_streams(tmp_path) -> None:
    runner = _runner()
    goes_store, mrms_store = KVBackend(), KVBackend()
    items = work_items([T0, T0 + np.timedelta64(1, "h")], 2 * STEP)
    Pipeline(runner, output={"goes": goes_store, "mrms": mrms_store}, device="cpu").run(
        items
    )
    assert tuple(goes_store.dims["abi13c"]) == ("time", "lead_time", "y", "x")
    assert goes_store["abi13c"].shape == (2, 3, 3, 4)
    assert mrms_store["refc"].shape == (2, 3, 3, 4)
    np.testing.assert_allclose(mrms_store["refc"][:, 1], 115.0)
    np.testing.assert_allclose(goes_store["abi01c"][:, 2], 102.0)

    zarr_store = ZarrBackend(str(tmp_path / "forecast.zarr"))
    members = work_items([T0], STEP, members=2, member_batch=1)
    Pipeline(
        runner,
        output={"goes": zarr_store, "mrms": zarr_store},
        output_variables={"goes": ["abi13c"], "mrms": ["refc"]},
        device="cpu",
    ).run(members)
    ds = xr.open_zarr(tmp_path / "forecast.zarr", consolidated=False)
    assert set(ds.data_vars) == {"abi13c", "refc"}
    assert ds["refc"].dims == ("ensemble", "time", "lead_time", "y", "x")
    assert ds["refc"].shape == (2, 1, 2, 3, 4)
    np.testing.assert_array_equal(ds["ensemble"].values, [0, 1])


def _stormscope_cfg(tmp_path, region: dict | None = None):
    cfg = compose_cfg(tmp_path, ["nsteps=3", "runner=stormscope", "random_seed=11"])
    with open_dict(cfg):
        cfg.model = {
            "goes": {"_target_": "test.test_stormscope.FakeGOES"},
            "mrms": {"_target_": "test.test_stormscope.FakeMRMS"},
            "region": region,
        }
        cfg.output.variables = GOES_VARS + MRMS_VARS
    return cfg


def test_builder_binds_default_sources_and_records_roles(tmp_path) -> None:
    cfg = _stormscope_cfg(tmp_path)
    runner = build_runner(cfg)
    assert isinstance(runner, StormScopeRunner)
    assert runner.seed == 11
    assert set(runner.source_roles) == {"goes", "mrms"}
    assert runner.source_roles["goes"] is runner.goes_source
    assert runner.source_roles["mrms"] is runner.mrms_source
    assert runner.goes.regions == [] and runner.mrms.regions == []
    assert final_lead_time(cfg, runner) == 3 * STEP
    items = build_items(cfg, runner)
    assert len(items) == 1 and items[0].final_lead_time == 3 * STEP


def test_builder_applies_region_and_configured_sources(tmp_path) -> None:
    cfg = _stormscope_cfg(
        tmp_path, region={"kind": "box", "params": [-100.0, 30.0, -90.0, 40.0]}
    )
    with open_dict(cfg):
        cfg.model.goes.ic_source = {
            "_target_": "test.test_stormscope.ConstantSource",
            "fill": 3.0,
        }
        cfg.model.mrms.conditioning_source = {
            "_target_": "test.test_stormscope.ConstantSource",
            "fill": 4.0,
        }
    runner = stormscope(cfg, prefer_predownloaded=False)
    assert runner.goes.regions == [Region.box(-100.0, 30.0, -90.0, 40.0)]
    assert runner.mrms.regions == runner.goes.regions
    assert runner.goes_source.fill == 3.0
    assert runner.conditioning_source.fill == 4.0
    assert set(runner.source_roles) == {"goes", "mrms", "mrms_conditioning"}


def test_builder_requires_both_model_nodes(tmp_path) -> None:
    cfg = compose_cfg(tmp_path, ["runner=stormscope"])
    with pytest.raises(ValueError, match="cfg.model.goes"):
        stormscope(cfg)

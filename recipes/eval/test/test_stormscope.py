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

"""The ``stormscope`` builder and the coupled runner it returns, on fake models.

The fake GOES model adds one per step and the fake MRMS model adds ten plus
the mean of the GOES conditioning it receives, so the values a step holds
show which frames each model saw.  Sources return constants on a 3 x 4
curvilinear grid, so every test runs on CPU without downloads.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import open_dict
from src.campaign import build_items, final_lead_time
from src.runners import build_runner
from src.runners.stormscope import COMPONENTS, couple, stormscope

from earth2studio.coupling import CoupledRunner, CouplingError
from earth2studio.data import fetch_data
from earth2studio.grids import CurvilinearGrid, Region
from earth2studio.io import KVBackend, ZarrBackend
from earth2studio.perturbation import Gaussian
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


def _seed(base: int, name: str, member: int) -> int:
    """The seed the coupled runner hands component ``name`` for ``member``."""
    return member_seed(base + zlib.crc32(name.encode()), T0, member)


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

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:
        return self._advance(x, self.add), None

    def step(self, y: xr.DataArray, state: None = None) -> tuple[xr.DataArray, None]:
        return self._advance(y, self.add), None


class FakeMRMS(FakeGOES):
    """Radar model adding ten plus the mean of the GOES conditioning it receives."""

    leads = MRMS_LEADS

    def __init__(self) -> None:
        super().__init__(add=10.0, variables=MRMS_VARS)
        self.conditioning: list[xr.DataArray] = []
        self.fill = 5.0

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "time", "lead_time", "variable", "y", "x"),
            {"lead_time": self.leads, "variable": np.array(self.variables)},
            dynamic=("batch", "time"),
            grid=GRID,
        )

    def forcing_coords(self) -> CoordinateSystem:
        return coord_array_like(
            self.input_coords(), {"variable": np.array(CONDITIONING_VARS)}
        )

    def default_sources(self) -> tuple[ConstantSource, ConstantSource]:  # type: ignore[override]
        return ConstantSource(self.fill), ConstantSource(-1.0)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        self.conditioning.append(conditioning)
        return self._advance(x, self.add + float(conditioning.mean())), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, conditioning: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        self.conditioning.append(conditioning)
        return self._advance(y, self.add + float(conditioning.mean())), None


def _runner(
    goes: FakeGOES | None = None, mrms: FakeMRMS | None = None, **kwargs: Any
) -> CoupledRunner:
    return couple(goes or FakeGOES(), mrms or FakeMRMS(), seed=7, **kwargs)


def _streams(runner: CoupledRunner, item: Any) -> dict[str, list[xr.DataArray]]:
    """Every frame ``run_item`` publishes, per stream, in order."""
    frames: dict[str, list[xr.DataArray]] = {name: [] for name in COMPONENTS}
    for published in runner.run_item(item):
        for name, x in published.items():
            frames[name].append(x)
    return frames


def test_step_and_output_coords_cover_every_lead_time() -> None:
    runner = _runner()
    assert runner.step == STEP
    assert runner.supports_member_batching
    schemas = runner.output_coords(3 * STEP)
    assert set(schemas) == set(COMPONENTS)
    for name, variables in (("goes", GOES_VARS), ("mrms", MRMS_VARS)):
        signature = schemas[name]
        assert signature.dims == ("batch", "time", "lead_time", "variable", "y", "x")
        np.testing.assert_array_equal(
            signature["lead_time"].values, np.arange(4) * STEP
        )
        assert signature["variable"].values.tolist() == variables
        assert signature.sizes["y"] == 3 and signature.sizes["x"] == 4


def test_run_item_publishes_initial_conditions_then_coupled_steps() -> None:
    runner = _runner()
    streams = _streams(runner, work_items([T0], 2 * STEP)[0])
    for frames in streams.values():
        assert len(frames) == 3
        for k, x in enumerate(frames):
            assert x.dims == ("time", "lead_time", "variable", "y", "x")
            np.testing.assert_array_equal(x["lead_time"].values, [k * STEP])
            assert x["time"].values[0] == T0
    # GOES: observations 100, then +1 per step.
    for k, x in enumerate(streams["goes"]):
        np.testing.assert_allclose(x.values, 100.0 + k)
    # MRMS: observations 5, first step conditioned on the GOES history (100),
    # the second on the first GOES forecast (101).
    np.testing.assert_allclose(streams["mrms"][0].values, 5.0)
    np.testing.assert_allclose(streams["mrms"][1].values, 5.0 + 10.0 + 100.0)
    np.testing.assert_allclose(streams["mrms"][2].values, 115.0 + 10.0 + 101.0)


def test_mrms_is_conditioned_on_goes_forecast_frames_at_its_lead_times() -> None:
    mrms = FakeMRMS()
    runner = _runner(mrms=mrms)
    _streams(runner, work_items([T0], 3 * STEP)[0])
    initial, *frames = mrms.conditioning
    # Initialization: the GOES history the GOES model holds, conditioning
    # variables only.
    assert initial["variable"].values.tolist() == CONDITIONING_VARS
    np.testing.assert_array_equal(initial["lead_time"].values, MRMS_LEADS)
    np.testing.assert_allclose(initial.values, 100.0)
    # Each later step: the GOES forecast at the previous radar forecast's lead time.
    assert len(frames) == 2
    for k, frame in enumerate(frames, start=1):
        assert frame["variable"].values.tolist() == CONDITIONING_VARS
        np.testing.assert_array_equal(frame["lead_time"].values, [k * STEP])
        np.testing.assert_allclose(frame.values, 100.0 + k)


def test_zero_steps_publishes_only_initial_conditions() -> None:
    mrms = FakeMRMS()
    runner = _runner(mrms=mrms)
    streams = _streams(runner, work_items([T0], np.timedelta64(0, "h"))[0])
    assert [len(frames) for frames in streams.values()] == [1, 1]
    assert mrms.conditioning == []


def test_data_requests_name_the_fetches_and_reuse_goes_for_conditioning() -> None:
    goes_source, mrms_source = ConstantSource(100.0), ConstantSource(5.0)
    runner = _runner(goes_source=goes_source, mrms_source=mrms_source)
    assert runner.sources["goes"] == (goes_source,)
    assert runner.sources["mrms"] == (mrms_source, goes_source)
    item = work_items([T0], 2 * STEP)[0]
    requests = runner.data_requests(item)
    assert [r.source for r in requests] == [goes_source, mrms_source]
    np.testing.assert_array_equal(requests[0].variable, GOES_VARS)
    np.testing.assert_array_equal(requests[0].lead_time, GOES_LEADS)
    np.testing.assert_array_equal(requests[1].variable, MRMS_VARS)
    _streams(runner, item)
    # ``fetch_data`` calls a source once per lead time; the GOES history the
    # GOES model fetched doubles as the radar conditioning, never fetched twice.
    assert len(goes_source.calls) == len(GOES_LEADS)
    assert len(mrms_source.calls) == len(MRMS_LEADS)


class LongHistoryMRMS(FakeMRMS):
    """Radar model whose history starts one frame before the GOES one."""

    leads = np.array([-20, -10, 0], dtype="timedelta64[m]")


def test_history_the_goes_model_lacks_comes_from_the_conditioning_source() -> None:
    goes_source, conditioning = ConstantSource(100.0), ConstantSource(50.0)
    mrms = LongHistoryMRMS()
    runner = _runner(
        mrms=mrms, goes_source=goes_source, conditioning_source=conditioning
    )
    item = work_items([T0], STEP)[0]
    backfill = [r for r in runner.data_requests(item) if r.source is conditioning]
    assert len(backfill) == 1
    np.testing.assert_array_equal(backfill[0].variable, CONDITIONING_VARS)
    np.testing.assert_array_equal(backfill[0].lead_time, [np.timedelta64(-20, "m")])
    _streams(runner, item)
    initial = mrms.conditioning[0]
    np.testing.assert_array_equal(initial["lead_time"].values, LongHistoryMRMS.leads)
    np.testing.assert_allclose(initial.isel(lead_time=0).values, 50.0)
    np.testing.assert_allclose(initial.isel(lead_time=[1, 2]).values, 100.0)
    assert len(conditioning.calls) == 1
    # Without a conditioning source, the GOES source serves those frames.
    runner = _runner(mrms=LongHistoryMRMS(), goes_source=goes_source)
    assert runner.sources["mrms"][1] is goes_source


def test_models_are_seeded_per_model_and_per_member() -> None:
    goes, mrms = FakeGOES(), FakeMRMS()
    runner = _runner(goes=goes, mrms=mrms)
    for item in work_items([T0], STEP, members=[2, 5], member_batch=1):
        _streams(runner, item)
    assert goes.seeds == [_seed(7, "goes", 2), _seed(7, "goes", 5)]
    assert mrms.seeds == [_seed(7, "mrms", 2), _seed(7, "mrms", 5)]
    assert goes.seeds[0] != mrms.seeds[0]


def test_batched_members_match_single_member_runs() -> None:
    goes = FakeGOES()
    runner = _runner(goes=goes)
    batched = _streams(
        runner, work_items([T0], 2 * STEP, members=[3, 4], member_batch=2)[0]
    )
    # The runner seeds a batch once, from its first member (EXECUTION_CONTRACT X5).
    assert goes.seeds == [_seed(7, "goes", 3)]
    for frames in batched.values():
        for x in frames:
            assert x.dims[0] == "ensemble"
            np.testing.assert_array_equal(x["ensemble"].values, [3, 4])
    for member in (3, 4):
        single = _streams(
            runner, work_items([T0], 2 * STEP, members=[member], member_batch=1)[0]
        )
        for name in COMPONENTS:
            assert len(batched[name]) == len(single[name])
            for many, one in zip(batched[name], single[name]):
                xr.testing.assert_allclose(many.sel(ensemble=member, drop=True), one)


def test_mismatched_grids_are_rejected() -> None:
    class OtherGrid(FakeMRMS):
        def input_coords(self) -> CoordinateSystem:
            return coord_array(
                ("batch", "time", "lead_time", "variable", "y", "x"),
                {"lead_time": MRMS_LEADS, "variable": np.array(MRMS_VARS)},
                dynamic=("batch", "time"),
                grid=CurvilinearGrid(LAT + 0.5, LON, y=GRID.y, x=GRID.x),
            )

    with pytest.raises(CouplingError, match="grid"):
        _runner(mrms=OtherGrid())


def test_conditioning_variables_the_goes_model_lacks_are_rejected() -> None:
    with pytest.raises(CouplingError, match="does not publish"):
        _runner(goes=FakeGOES(variables=["abi01c"]))


def test_models_of_different_cadences_couple_on_their_common_step() -> None:
    class Hourly(FakeMRMS):
        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            lead = input_coords["lead_time"].values[-1]
            return coord_array_like(
                input_coords, {"lead_time": np.array([lead + np.timedelta64(1, "h")])}
            )

    runner = _runner(mrms=Hourly())
    assert runner.step == np.timedelta64(1, "h")
    schemas = runner.output_coords(np.timedelta64(2, "h"))
    assert schemas["goes"].sizes["lead_time"] == 13
    np.testing.assert_array_equal(
        schemas["mrms"]["lead_time"].values, np.array([0, 60, 120], "timedelta64[m]")
    )


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
    members = work_items([T0], STEP, members=2, member_batch=2)
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
    np.testing.assert_allclose(ds["refc"].isel(lead_time=1).values, 115.0)


def _stormscope_cfg(tmp_path, region: dict | None = None, *overrides: str):
    cfg = compose_cfg(
        tmp_path, ["nsteps=3", "runner=stormscope", "random_seed=11", *overrides]
    )
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
    assert isinstance(runner, CoupledRunner)
    assert runner.seed == 11
    assert runner.perturbation == {}
    assert set(runner.source_roles) == {"goes", "mrms"}
    goes_source, mrms_source = runner.source_roles["goes"], runner.source_roles["mrms"]
    assert isinstance(goes_source, ConstantSource) and goes_source.fill == 100.0
    assert isinstance(mrms_source, ConstantSource) and mrms_source.fill == 5.0
    # The GOES model feeds the radar conditioning; its source backs that slot.
    assert runner.sources["goes"] == (goes_source,)
    assert runner.sources["mrms"] == (mrms_source, goes_source)
    goes, mrms = (runner.nodes[name].model for name in COMPONENTS)
    assert goes.regions == [] and mrms.regions == []
    # Verification composes the live GOES and MRMS observations over the
    # output variables of both models.
    verification = runner.verification_source
    assert set(verification.sources) == {"goes", "mrms"}
    field = fetch_data(
        verification, time=np.array([T0]), variable=np.array(["abi13c", "refc"])
    )
    assert list(field["variable"].values) == ["abi13c", "refc"]
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
    goes, mrms = (runner.nodes[name].model for name in COMPONENTS)
    assert goes.regions == [Region.box(-100.0, 30.0, -90.0, 40.0)]
    assert mrms.regions == goes.regions
    assert set(runner.source_roles) == {"goes", "mrms", "mrms_conditioning"}
    assert runner.source_roles["goes"].fill == 3.0
    assert runner.sources["goes"] == (runner.source_roles["goes"],)
    assert runner.sources["mrms"][1] is runner.source_roles["mrms_conditioning"]
    assert runner.sources["mrms"][1].fill == 4.0


def test_builder_perturbs_each_model_for_ensembles_only(tmp_path) -> None:
    cfg = _stormscope_cfg(tmp_path, None, "ensemble_size=2")
    with open_dict(cfg):
        cfg.model.goes.perturbation = {
            "_target_": "earth2studio.perturbation.Gaussian",
            "noise_amplitude": 0.1,
        }
    runner = stormscope(cfg, prefer_predownloaded=False)
    assert set(runner.perturbation) == {"goes"}
    assert isinstance(runner.perturbation["goes"], Gaussian)
    # The perturbation node never reaches the model loader.
    assert runner.nodes["goes"].model.variables == GOES_VARS
    with open_dict(cfg):
        cfg.ensemble_size = 1
    assert stormscope(cfg, prefer_predownloaded=False).perturbation == {}


def test_builder_requires_both_model_nodes(tmp_path) -> None:
    cfg = compose_cfg(tmp_path, ["runner=stormscope"])
    with pytest.raises(ValueError, match="cfg.model.goes"):
        stormscope(cfg)

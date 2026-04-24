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

"""Tests for StormScopePipeline.

Uses stub GOES and MRMS models that mimic the minimal StormScope API
surface — ``input_coords`` / ``output_coords`` / ``__call__`` /
``call_with_conditioning`` / ``next_input`` / ``build_*_interpolator``
and the ``y`` / ``x`` grid attributes.  This lets the tests exercise
the full coupling loop without loading either real checkpoint.

Covers:

* Helper functions — ``_infer_step_delta``, ``_concat_var_lists``.
* ``StormScopePipeline.build_total_coords`` — shape of the zarr schema.
* ``StormScopePipeline.run_item`` — coupling-loop call order (GOES
  forward, then MRMS with GOES-state as conditioning, then both
  ``next_input``), yield count, and combined variable axis.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf
from src.data import CompositeSource, PredownloadedSource, resolve_ic_source
from src.pipelines import StormScopePipeline
from src.pipelines.stormscope import (
    _concat_var_lists,
    _infer_step_delta,
)
from src.work import WorkItem

from earth2studio.utils.coords import CoordSystem

# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------


class TestConcatVarLists:
    def test_no_overlap(self):
        assert _concat_var_lists(["a", "b"], ["c", "d"]) == ["a", "b", "c", "d"]

    def test_full_overlap(self):
        assert _concat_var_lists(["a", "b"], ["a", "b"]) == ["a", "b"]

    def test_partial_overlap_preserves_first_list_order(self):
        assert _concat_var_lists(
            ["abi01c", "abi02c", "abi07c"], ["abi07c", "refc"]
        ) == [
            "abi01c",
            "abi02c",
            "abi07c",
            "refc",
        ]


# ---------------------------------------------------------------------------
# Stub StormScope models
# ---------------------------------------------------------------------------


def _hours_td(m: int) -> np.timedelta64:
    return np.timedelta64(m, "m").astype("timedelta64[ns]")


class _StubStormScope:
    """Shared scaffolding for GOES and MRMS stubs.

    Implements the StormScope API surface that :class:`StormScopePipeline`
    calls during setup, build_total_coords, and run_item.  Subclasses
    override ``variables`` and whether ``call_with_conditioning`` is
    provided.  Forward passes return a zero tensor reshaped to the
    model's output shape — sufficient for exercising coord bookkeeping
    and yield-count logic without any actual math.

    * ``input_coords``: ``{batch, time, lead_time=[0min], variable, y, x}``
    * ``output_coords``: ``{batch, time, lead_time=[input_last+60min], variable, y, x}``
    * ``next_input``: non-sliding — returns pred unchanged.
    """

    variables: list[str]

    def __init__(self) -> None:
        self.y = torch.arange(4)
        self.x = torch.arange(5)
        self.input_interp_built = False
        self.conditioning_interp_built = False
        self.call_count = 0
        self.next_input_calls = 0

    # --- coords ------------------------------------------------------------

    def input_coords(self) -> CoordSystem:
        return OrderedDict(
            [
                ("batch", np.empty(0)),
                ("time", np.empty(0)),
                ("lead_time", np.array([_hours_td(0)])),
                ("variable", np.array(self.variables)),
                ("y", self.y.numpy()),
                ("x", self.x.numpy()),
            ]
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        out_lt = np.array([_hours_td(60)]) + input_coords["lead_time"][-1]
        return OrderedDict(
            [
                ("batch", input_coords["batch"]),
                ("time", input_coords["time"]),
                ("lead_time", out_lt),
                ("variable", np.array(self.variables)),
                ("y", self.y.numpy()),
                ("x", self.x.numpy()),
            ]
        )

    # --- interpolators -----------------------------------------------------

    def build_input_interpolator(
        self, lats: Any, lons: Any, max_dist_km: Any = None
    ) -> None:
        self.input_interp_built = True

    def build_conditioning_interpolator(
        self, lats: Any, lons: Any, max_dist_km: Any = None
    ) -> None:
        self.conditioning_interp_built = True

    # --- inference ---------------------------------------------------------

    def _forward_shape(
        self, coords: CoordSystem, out_coords: CoordSystem
    ) -> tuple[int, ...]:
        # (batch=1, time=1, lead=1, var=N, y=4, x=5) — matches StormScope's 6D layout.
        return (
            1,
            1,
            len(out_coords["lead_time"]),
            len(self.variables),
            len(self.y),
            len(self.x),
        )

    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        self.call_count += 1
        out_coords = self.output_coords(coords)
        return torch.zeros(self._forward_shape(coords, out_coords)), out_coords

    def next_input(
        self,
        pred: torch.Tensor,
        pred_coords: CoordSystem,
        x: torch.Tensor,
        x_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        # Non-sliding (60-min model): pred becomes the next input directly.
        self.next_input_calls += 1
        return pred, pred_coords.copy()

    # --- misc --------------------------------------------------------------

    def to(self, device: Any) -> _StubStormScope:
        return self

    def eval(self) -> _StubStormScope:
        return self


class _StubStormScopeGOES(_StubStormScope):
    variables = ["abi01c", "abi02c", "abi07c"]


class _StubStormScopeMRMS(_StubStormScope):
    variables = ["refc"]

    def __init__(self) -> None:
        super().__init__()
        self.cond_call_count = 0
        self.last_conditioning: torch.Tensor | None = None

    def call_with_conditioning(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor,
        conditioning_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        self.cond_call_count += 1
        self.last_conditioning = conditioning
        return self.__call__(x, coords)


# ---------------------------------------------------------------------------
# _infer_step_delta
# ---------------------------------------------------------------------------


class TestInferStepDelta:
    def test_60min_stride(self):
        model = _StubStormScopeGOES()
        delta = _infer_step_delta(model)
        assert delta == _hours_td(60)


# ---------------------------------------------------------------------------
# Pipeline construction (bypassing Hydra setup) and build_total_coords
# ---------------------------------------------------------------------------


def _make_stub_pipeline() -> StormScopePipeline:
    """Build a StormScopePipeline with both models pre-wired.

    Mirrors what ``setup`` would do, minus the hydra.utils.call /
    instantiate calls that require a full config tree.
    """
    pipeline = StormScopePipeline()
    pipeline.model_goes = _StubStormScopeGOES()
    pipeline.model_mrms = _StubStormScopeMRMS()
    pipeline.nsteps = 2

    # Cached IC coords (matches what setup reads from model.input_coords()).
    pipeline._goes_ic_coords = pipeline.model_goes.input_coords()
    pipeline._mrms_ic_coords = pipeline.model_mrms.input_coords()

    # Minimal IC sources — just need to expose a callable that returns an
    # xr.DataArray with the right shape.  See _stub_source() below.
    pipeline._goes_ic_source = _stub_source(
        pipeline.model_goes.variables, pipeline.model_goes
    )
    pipeline._mrms_ic_source = _stub_source(
        pipeline.model_mrms.variables, pipeline.model_mrms
    )

    # Spatial reference — union of GOES + MRMS variables on HRRR y/x.
    all_vars = _concat_var_lists(
        pipeline.model_goes.variables, pipeline.model_mrms.variables
    )
    pipeline._spatial_ref = OrderedDict(
        [
            ("variable", np.array(all_vars)),
            ("y", pipeline.model_goes.y.numpy()),
            ("x", pipeline.model_goes.x.numpy()),
        ]
    )
    return pipeline


def _stub_source(variables: list[str], model: _StubStormScope):
    """DataSource returning zero-filled DataArrays on the model's native grid."""

    class _Src:
        def __call__(self, time, variable):
            t = np.atleast_1d(time)
            v = np.atleast_1d(variable)
            data = np.zeros(
                (len(t), len(v), len(model.y), len(model.x)),
                dtype="float32",
            )
            return xr.DataArray(
                data,
                dims=("time", "variable", "y", "x"),
                coords={
                    "time": t,
                    "variable": v,
                    "y": model.y.numpy(),
                    "x": model.x.numpy(),
                },
            )

        async def fetch(self, time, variable):
            return self(time, variable)

    return _Src()


class TestBuildTotalCoords:
    def test_lead_time_steps_from_model_stride(self):
        pipeline = _make_stub_pipeline()
        times = np.array([np.datetime64("2023-12-05T12:00:00")])
        coords = pipeline.build_total_coords(times, ensemble_size=1)

        assert "ensemble" not in coords
        assert "time" in coords
        # nsteps=2, stride=60min → lead_time = [60min, 120min]
        assert len(coords["lead_time"]) == 2
        assert coords["lead_time"][0] == _hours_td(60)
        assert coords["lead_time"][1] == _hours_td(120)
        assert "y" in coords
        assert "x" in coords

    def test_ensemble_coord_added(self):
        pipeline = _make_stub_pipeline()
        times = np.array([np.datetime64("2023-12-05T12:00:00")])
        coords = pipeline.build_total_coords(times, ensemble_size=3)
        assert "ensemble" in coords
        np.testing.assert_array_equal(coords["ensemble"], np.arange(3))

    def test_mismatched_strides_raises(self):
        pipeline = _make_stub_pipeline()

        # Monkey-patch MRMS stub to advertise a 10-minute stride.
        def _mrms_output_coords(input_coords):
            out_lt = np.array([_hours_td(10)]) + input_coords["lead_time"][-1]
            oc = input_coords.copy()
            oc["lead_time"] = out_lt
            return oc

        pipeline.model_mrms.output_coords = _mrms_output_coords  # type: ignore[method-assign]
        times = np.array([np.datetime64("2023-12-05T12:00:00")])
        with pytest.raises(ValueError, match="share a forecast stride"):
            pipeline.build_total_coords(times, ensemble_size=1)


# ---------------------------------------------------------------------------
# run_item — coupling-loop behavior
# ---------------------------------------------------------------------------


class TestStormScopeRunItem:
    @pytest.fixture()
    def pipeline(self):
        return _make_stub_pipeline()

    def test_yields_nsteps_outputs(self, pipeline):
        item = WorkItem(
            time=np.datetime64("2023-12-05T12:00:00"), ensemble_id=0, seed=0
        )
        outputs = list(pipeline.run_item(item, None, torch.device("cpu")))
        assert len(outputs) == pipeline.nsteps

    def test_calls_models_nsteps_times(self, pipeline):
        item = WorkItem(
            time=np.datetime64("2023-12-05T12:00:00"), ensemble_id=0, seed=0
        )
        list(pipeline.run_item(item, None, torch.device("cpu")))

        assert (
            pipeline.model_goes.call_count == pipeline.nsteps
        ), "GOES.__call__ should be invoked once per step"
        assert (
            pipeline.model_mrms.cond_call_count == pipeline.nsteps
        ), "MRMS.call_with_conditioning should be invoked once per step"
        assert pipeline.model_goes.next_input_calls == pipeline.nsteps
        assert pipeline.model_mrms.next_input_calls == pipeline.nsteps

    def test_yielded_variable_axis_is_combined(self, pipeline):
        item = WorkItem(
            time=np.datetime64("2023-12-05T12:00:00"), ensemble_id=0, seed=0
        )
        outputs = list(pipeline.run_item(item, None, torch.device("cpu")))

        expected_vars = pipeline.model_goes.variables + pipeline.model_mrms.variables
        for _, coords in outputs:
            assert list(coords["variable"]) == expected_vars

    def test_yielded_tensor_includes_batch_dim(self, pipeline):
        """run_item yields the raw model output (with a size-1 ``batch`` dim).

        :class:`Pipeline.run` squeezes the ``batch`` axis before output
        filtering because :attr:`StormScopePipeline._run_item_includes_batch_dim`
        is ``True``; this test pins the contract at the subclass boundary.
        """
        assert pipeline._run_item_includes_batch_dim is True

        item = WorkItem(
            time=np.datetime64("2023-12-05T12:00:00"), ensemble_id=0, seed=0
        )
        outputs = list(pipeline.run_item(item, None, torch.device("cpu")))

        for x, coords in outputs:
            assert "batch" in coords
            # Expect (batch=1, time=1, lead=1, var=N, y=4, x=5) = 6 dims
            assert x.ndim == 6
            assert x.shape[0] == 1
            assert x.shape[-2:] == (4, 5)

    def test_mrms_conditioning_is_goes_state(self, pipeline):
        """MRMS must receive the GOES state tensor (pre-next_input) as conditioning."""
        item = WorkItem(
            time=np.datetime64("2023-12-05T12:00:00"), ensemble_id=0, seed=0
        )
        list(pipeline.run_item(item, None, torch.device("cpu")))

        # The stub records the last conditioning tensor passed in; its variable
        # axis must have len(GOES vars), not len(MRMS vars).
        assert pipeline.model_mrms.last_conditioning is not None
        # Shape (batch=1, time=1, lead=1, var=3, y=4, x=5)
        assert pipeline.model_mrms.last_conditioning.shape[3] == len(
            pipeline.model_goes.variables
        )


# ---------------------------------------------------------------------------
# resolve_ic_source — predownloaded-cache preference (StormScope call path)
# ---------------------------------------------------------------------------


class TestResolveIcSource:
    """Verify :func:`src.data.resolve_ic_source` honours StormScope's call path.

    The helper decides between:
    * a :class:`PredownloadedSource` wrapping ``<output.path>/<store_name>``
      when the directory exists on disk, and
    * a hydra-instantiated live source otherwise.
    """

    def test_returns_predownloaded_source_when_cache_exists(self, tmp_path):
        cache = tmp_path / "data_goes.zarr"

        # Write a minimal zarr that PredownloadedSource can open.
        data = np.zeros((1, 4, 5), dtype="float32")
        ds = xr.Dataset(
            {"abi01c": (("time", "y", "x"), data)},
            coords={
                "time": np.array(["2023-12-05T12:00:00"], dtype="datetime64[ns]"),
                "y": np.arange(4),
                "x": np.arange(5),
            },
        )
        ds.to_zarr(str(cache), mode="w")

        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path)},
                "model": {
                    "goes": {
                        "ic_source": {
                            "_target_": "unreachable.module.Source",
                        }
                    }
                },
            }
        )
        src = resolve_ic_source(
            cfg,
            store_name="data_goes.zarr",
            live_source=cfg.model.goes.ic_source,
        )
        assert isinstance(src, PredownloadedSource)

    def test_falls_back_to_live_source_when_cache_absent(self, tmp_path):
        # No zarr on disk → should instantiate the live source.  Point _target_
        # at a trivial built-in so we can assert the fallback without network I/O.
        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path)},
                "model": {
                    "goes": {
                        "ic_source": {
                            "_target_": "collections.OrderedDict",
                        }
                    }
                },
            }
        )
        src = resolve_ic_source(
            cfg,
            store_name="data_goes.zarr",
            live_source=cfg.model.goes.ic_source,
        )
        assert isinstance(src, OrderedDict)


# ---------------------------------------------------------------------------
# verification_source — CompositeSource over per-model predownload stores
# ---------------------------------------------------------------------------


def _write_yx_zarr(path, variables):
    """Minimal (time, y, x) zarr so PredownloadedSource can open it."""
    import xarray as xr

    t = np.array(["2023-12-05T12:00:00"], dtype="datetime64[ns]")
    y = np.arange(4)
    x = np.arange(5)
    ds = xr.Dataset()
    for v in variables:
        ds[v] = xr.DataArray(
            np.zeros((1, 4, 5), dtype="float32"),
            dims=["time", "y", "x"],
            coords={"time": t, "y": y, "x": x},
        )
    ds.to_zarr(str(path), mode="w")


class TestVerificationSource:
    """``Pipeline.verification_source`` honours the per-model ``data_*.zarr``
    layout used by StormScope via its default ``verification_zarr_paths``
    discovery (the StormScope subclass no longer needs to override)."""

    def test_returns_composite_when_both_stores_exist(self, tmp_path):
        _write_yx_zarr(tmp_path / "data_goes.zarr", ["abi01c", "abi02c"])
        _write_yx_zarr(tmp_path / "data_mrms.zarr", ["refc"])
        cfg = OmegaConf.create({"output": {"path": str(tmp_path)}})

        pipeline = StormScopePipeline()
        src = pipeline.verification_source(cfg)
        assert isinstance(src, CompositeSource)

        # Dispatch must work for variables from both stores.
        t = np.array(["2023-12-05T12:00:00"], dtype="datetime64[ns]")
        result = src(t, ["abi01c", "refc"])
        assert result.sizes["variable"] == 2

    def test_raises_when_no_stores_exist(self, tmp_path):
        cfg = OmegaConf.create({"output": {"path": str(tmp_path)}})
        pipeline = StormScopePipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.verification_source(cfg)

    def test_returns_predownloaded_source_when_only_one_store_exists(self, tmp_path):
        _write_yx_zarr(tmp_path / "data_goes.zarr", ["abi01c"])
        cfg = OmegaConf.create({"output": {"path": str(tmp_path)}})
        pipeline = StormScopePipeline()
        src = pipeline.verification_source(cfg)
        assert isinstance(src, PredownloadedSource)


# ---------------------------------------------------------------------------
# Conditioning source swap on setup
# ---------------------------------------------------------------------------


class _ModelStub:
    """Just enough surface for ``_maybe_swap_conditioning`` to mutate."""

    def __init__(self) -> None:
        self.conditioning_data_source = object()  # sentinel "live source"


class TestMaybeSwapConditioning:
    """``_maybe_swap_conditioning`` replaces a model's conditioning source
    with a :class:`PredownloadedSource` when ``cond_<side>.zarr`` is
    present on disk; otherwise it leaves the attribute untouched so the
    model keeps whatever source the campaign config installed at load
    time."""

    def test_swaps_when_cache_exists(self, tmp_path):
        # Write a minimal cond_goes.zarr PredownloadedSource can open.
        cond_path = tmp_path / "cond_goes.zarr"
        xr.Dataset(
            {
                "z500": xr.DataArray(
                    np.zeros((1, 4, 5), dtype="float32"),
                    dims=["time", "y", "x"],
                    coords={
                        "time": np.array(
                            ["2023-12-05T12:00:00"], dtype="datetime64[ns]"
                        ),
                        "y": np.arange(4),
                        "x": np.arange(5),
                    },
                )
            }
        ).to_zarr(str(cond_path), mode="w")

        # `_maybe_swap_conditioning` reads cfg.model.<side>.conditioning_cadence,
        # so the model block must exist (even if empty).
        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path)},
                "model": {"goes": {}},
            }
        )
        pipeline = StormScopePipeline()
        model = _ModelStub()
        sentinel = model.conditioning_data_source

        pipeline._maybe_swap_conditioning(cfg, "goes", model)

        assert isinstance(model.conditioning_data_source, PredownloadedSource)
        assert model.conditioning_data_source is not sentinel

    def test_no_op_when_cache_absent(self, tmp_path):
        cfg = OmegaConf.create({"output": {"path": str(tmp_path)}})
        pipeline = StormScopePipeline()
        model = _ModelStub()
        sentinel = model.conditioning_data_source

        pipeline._maybe_swap_conditioning(cfg, "goes", model)

        # Live source retained.
        assert model.conditioning_data_source is sentinel

    def test_wraps_with_cadence_when_configured(self, tmp_path):
        """With ``conditioning_cadence`` set, the predownloaded source is
        wrapped in ``CadenceRoundedSource`` so the GOES model's sub-hour
        queries route to the nearest stored hourly value."""
        from src.data import CadenceRoundedSource

        cond_path = tmp_path / "cond_goes.zarr"
        xr.Dataset(
            {
                "z500": xr.DataArray(
                    np.zeros((1, 4, 5), dtype="float32"),
                    dims=["time", "y", "x"],
                    coords={
                        "time": np.array(
                            ["2023-12-05T12:00:00"], dtype="datetime64[ns]"
                        ),
                        "y": np.arange(4),
                        "x": np.arange(5),
                    },
                )
            }
        ).to_zarr(str(cond_path), mode="w")

        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path)},
                "model": {
                    "goes": {"conditioning_cadence": "1h"},
                },
            }
        )
        pipeline = StormScopePipeline()
        model = _ModelStub()
        pipeline._maybe_swap_conditioning(cfg, "goes", model)

        assert isinstance(model.conditioning_data_source, CadenceRoundedSource)


# ---------------------------------------------------------------------------
# _build_conditioning_store — DataSource vs ForecastSource, cadence dedup
# ---------------------------------------------------------------------------


class _FakeConditioningModel:
    """StormScope-model-shaped stub sufficient for ``_build_conditioning_store``.

    Advertises conditioning variables, a cropped HRRR sub-region, and
    numpy-array ``y`` / ``x``.  No actual forward pass.
    """

    def __init__(self, variables=("z500",)):
        self.conditioning_variables = list(variables)
        self.latitudes = torch.zeros(4, 5, dtype=torch.float32)
        self.longitudes = torch.zeros(4, 5, dtype=torch.float32)
        self.y = np.arange(4)
        self.x = np.arange(5)


def _grid_resolver_registered():
    """Register a Hydra-instantiable stub grid resolver once per test session."""
    import sys
    import types

    mod_name = "test._stormscope_cond_stubs"
    if mod_name in sys.modules:
        return mod_name
    mod = types.ModuleType(mod_name)

    def stub_grid():
        lats = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        lons = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        return lats, lons

    class _StubDataSource:
        def __call__(self, time, variable):
            times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
            variables = np.atleast_1d(variable)
            data = np.zeros((len(times), len(variables), 2, 2), dtype="float32")
            return xr.DataArray(
                data,
                dims=["time", "variable", "lat", "lon"],
                coords={
                    "time": times,
                    "variable": variables,
                    "lat": [0.0, 1.0],
                    "lon": [0.0, 1.0],
                },
            )

        async def fetch(self, time, variable):
            return self(time, variable)

    class _StubForecastSource:
        def __call__(self, time, lead_time, variable):
            times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
            leads = np.atleast_1d(np.asarray(lead_time, dtype="timedelta64[ns]"))
            variables = np.atleast_1d(variable)
            data = np.zeros(
                (len(times), len(leads), len(variables), 2, 2), dtype="float32"
            )
            return xr.DataArray(
                data,
                dims=["time", "lead_time", "variable", "lat", "lon"],
                coords={
                    "time": times,
                    "lead_time": leads,
                    "variable": variables,
                    "lat": [0.0, 1.0],
                    "lon": [0.0, 1.0],
                },
            )

    mod.stub_grid = stub_grid
    mod._StubDataSource = _StubDataSource
    mod._StubForecastSource = _StubForecastSource
    sys.modules[mod_name] = mod
    return mod_name


class TestBuildConditioningStore:
    """End-to-end tests for ``StormScopePipeline._build_conditioning_store``.

    Covers source-type autodetection (ForecastSource → wrapped in
    ``ValidTimeForecastAdapter``; DataSource → forwarded directly) and
    cadence-based deduplication of requested valid times.
    """

    def _make_model_cfg(self, *, source_target: str, cadence: str | None = None):
        mod = _grid_resolver_registered()
        block = {
            "load_args": {
                "conditioning_data_source": {
                    "_target_": f"{mod}.{source_target}",
                },
            },
            "conditioning_grid": {
                "_target_": f"{mod}.stub_grid",
            },
        }
        if cadence is not None:
            block["conditioning_cadence"] = cadence
        return OmegaConf.create(block)

    def _run(
        self,
        *,
        source_target: str,
        cadence: str | None,
        input_offsets: list,
        forecast_offsets: list,
    ):
        from src.data import ValidTimeForecastAdapter

        pipeline = StormScopePipeline()
        model = _FakeConditioningModel()
        model_cfg = self._make_model_cfg(source_target=source_target, cadence=cadence)
        spatial_ref = OrderedDict([("y", model.y), ("x", model.x)])
        ic_times = [np.datetime64("2023-12-05T12:00:00", "ns")]

        store = pipeline._build_conditioning_store(
            model=model,
            model_cfg=model_cfg,
            side="goes",
            unique_ic_times=ic_times,
            input_offsets=input_offsets,
            forecast_offsets=forecast_offsets,
            spatial_ref=spatial_ref,
            max_dist_km=30.0,
        )
        return store, ValidTimeForecastAdapter

    def test_forecast_source_is_wrapped_in_adapter(self):
        store, ValidTimeForecastAdapter = self._run(
            source_target="_StubForecastSource",
            cadence=None,
            input_offsets=[np.timedelta64(0, "m")],
            forecast_offsets=[np.timedelta64(60, "m")],
        )
        assert store is not None
        # RegriddedSource wraps the adapter; peek at _source.
        from src.regrid import RegriddedSource

        assert isinstance(store.source, RegriddedSource)
        assert isinstance(store.source._source, ValidTimeForecastAdapter)

    def test_data_source_is_forwarded_directly(self):
        """ARCO-style ``DataSource`` conditioning — no valid-time adapter needed."""
        store, ValidTimeForecastAdapter = self._run(
            source_target="_StubDataSource",
            cadence=None,
            input_offsets=[np.timedelta64(0, "m")],
            forecast_offsets=[np.timedelta64(60, "m")],
        )
        assert store is not None
        from src.regrid import RegriddedSource

        assert isinstance(store.source, RegriddedSource)
        # Underlying is the raw DataSource, NOT the adapter.
        assert not isinstance(store.source._source, ValidTimeForecastAdapter)

    def test_cadence_dedupes_subhour_offsets(self):
        """10-min offsets inside a single hour dedupe to one stored time
        when ``conditioning_cadence='1h'``."""
        store, _ = self._run(
            source_target="_StubDataSource",
            cadence="1h",
            input_offsets=[np.timedelta64(0, "m")],
            forecast_offsets=[
                np.timedelta64(10, "m"),
                np.timedelta64(20, "m"),
                np.timedelta64(30, "m"),
                np.timedelta64(40, "m"),
                np.timedelta64(50, "m"),
                np.timedelta64(60, "m"),
            ],
        )
        assert store is not None
        # 0, 10, 20min → 12:00; 30, 40, 50min → 13:00; 60min → 13:00.
        # Unique stored valid times: {12:00, 13:00}.
        assert len(store.times) == 2
        assert store.times[0] == np.datetime64("2023-12-05T12:00:00", "ns")
        assert store.times[1] == np.datetime64("2023-12-05T13:00:00", "ns")

    def test_no_cadence_keeps_every_offset(self):
        store, _ = self._run(
            source_target="_StubDataSource",
            cadence=None,
            input_offsets=[np.timedelta64(0, "m")],
            forecast_offsets=[
                np.timedelta64(10, "m"),
                np.timedelta64(20, "m"),
                np.timedelta64(30, "m"),
            ],
        )
        assert store is not None
        # 4 distinct valid times when no rounding applies.
        assert len(store.times) == 4

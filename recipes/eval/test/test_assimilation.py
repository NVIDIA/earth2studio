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

"""Tests for the data-assimilation plumbing and pipelines.

Uses a lightweight ``FakeDA`` model (satisfying the ``AssimilationModel``
protocol) and an in-memory ``FakeObsSource`` so no network access, GPUs,
or model checkpoints are required — mirroring the ``Persistence`` /
``Random`` pattern used by the forecast pipeline tests.
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf
from src.assimilation import (
    ObsSourceSet,
    StatelessAssimilationRunner,
    analysis_spatial_ref,
    analysis_to_tensor,
    analysis_variables,
    build_runner,
    insert_zero_lead_time,
)
from src.data import PredownloadedFrameSource, frame_filename, frame_store_path
from src.output import OutputManager
from src.pipelines import AssimilationForecastPipeline, AssimilationPipeline
from src.predownload_utils import declare_verification_only_store
from src.work import WorkItem

from earth2studio.data import Random
from earth2studio.models.px import Persistence
from earth2studio.utils.type import CoordSystem

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
DA_VARIABLES = ["t2m", "z500"]

_DIST_PATH = "src.output.DistributedManager"
_RANK0_OUTPUT = "src.output.run_on_rank0_first"
_RANK0_ASSIM = "src.assimilation.run_on_rank0_first"


def _passthrough(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _make_dist_mock(*, rank=0, world_size=1, distributed=False):
    class _FakeDist:
        def __init__(self):
            self.rank = rank
            self.world_size = world_size
            self.distributed = distributed

    return _FakeDist()


# ======================================================================
# Fakes (hydra-instantiable via test.test_assimilation.<Name>)
# ======================================================================


class FakeObsSource:
    """Minimal DataFrameSource returning a fixed-value observation frame."""

    def __init__(
        self,
        value: float = 1.0,
        time_tolerance: tuple | None = None,
        cache: bool = True,
    ) -> None:
        self.value = value
        self.time_tolerance = time_tolerance
        self.cache = cache
        self.call_count = 0
        # When True, return a schema-shaped 0-row frame (simulates a total
        # archive gap: all diag files missing for the request).
        self._force_empty = False

    def __call__(self, time, variable, fields=None) -> pd.DataFrame:
        self.call_count += 1
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = [str(v) for v in np.atleast_1d(variable)]
        if self._force_empty:
            times = np.empty(0, dtype="datetime64[ns]")
            variables = []
        n = len(times) * len(variables)
        data = {
            "time": np.repeat(times, len(variables)),
            "lat": np.zeros(n, dtype=np.float32),
            "lon": np.zeros(n, dtype=np.float32),
            "observation": np.full(n, self.value, dtype=np.float32),
            "variable": np.tile(np.array(variables), len(times)),
        }
        if fields is not None:
            data = {k: v for k, v in data.items() if k in set(map(str, fields))}
        return pd.DataFrame(data)


class FakeDA:
    """Stateless AssimilationModel-protocol fake on a small lat/lon grid."""

    def __init__(
        self,
        variables: list[str] | None = None,
        lat: np.ndarray | None = None,
        lon: np.ndarray | None = None,
    ) -> None:
        self.variables = list(variables or DA_VARIABLES)
        self.lat = SMALL_LAT if lat is None else np.asarray(lat)
        self.lon = SMALL_LON if lon is None else np.asarray(lon)
        self.call_count = 0

    # -- AutoModelMixin-style loading ---------------------------------
    @classmethod
    def load_default_package(cls):
        return None

    @classmethod
    def load_model(cls, package=None, **kwargs):
        return cls(**kwargs)

    # -- AssimilationModel protocol ------------------------------------
    def init_coords(self):
        return None

    def input_coords(self):
        conv_schema = OrderedDict(
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "observation": np.empty(0, dtype=np.float32),
                "variable": np.array(["u", "v", "t"], dtype=str),
            }
        )
        sat_schema = OrderedDict(
            {
                "time": np.empty(0, dtype="datetime64[ns]"),
                "lat": np.empty(0, dtype=np.float32),
                "lon": np.empty(0, dtype=np.float32),
                "observation": np.empty(0, dtype=np.float32),
                "variable": np.array(["atms"], dtype=str),
            }
        )
        return conv_schema, sat_schema

    def output_coords(self, input_coords, request_time=None, **kwargs):
        if request_time is None:
            request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")
        return (
            OrderedDict(
                {
                    "time": request_time,
                    "variable": np.array(self.variables, dtype=str),
                    "lat": self.lat,
                    "lon": self.lon,
                }
            ),
        )

    def __call__(self, conv_obs=None, sat_obs=None) -> xr.DataArray:
        if conv_obs is None and sat_obs is None:
            raise ValueError("At least one of conv_obs or sat_obs must be provided.")
        request_time = None
        for df in (conv_obs, sat_obs):
            if df is not None:
                request_time = df.attrs.get("request_time")
                if request_time is not None:
                    break
        if request_time is None:
            raise ValueError("Observation DataFrame must have 'request_time' attrs.")
        request_time = np.asarray(request_time, dtype="datetime64[ns]")

        self.call_count += 1
        (output_coords,) = self.output_coords(
            self.input_coords(), request_time=request_time
        )
        shape = (
            len(request_time),
            len(self.variables),
            len(self.lat),
            len(self.lon),
        )
        rng = np.random.default_rng(self.call_count)
        return xr.DataArray(
            rng.standard_normal(shape).astype(np.float32),
            dims=["time", "variable", "lat", "lon"],
            coords=output_coords,
        )

    def create_generator(self):
        inputs = yield None
        while True:
            conv_obs, sat_obs = inputs if inputs is not None else (None, None)
            inputs = yield self(conv_obs, sat_obs)

    def to(self, device):
        return self


class FakeStatefulDA(FakeDA):
    """A DA fake that requires a background state (init_coords not None)."""

    def init_coords(self):
        return (
            OrderedDict(
                {
                    "variable": np.array(self.variables, dtype=str),
                    "lat": self.lat,
                    "lon": self.lon,
                }
            ),
        )


def _make_obs_set(model, conv_source=None, sat_source=None) -> ObsSourceSet:
    return ObsSourceSet([conv_source, sat_source], model.input_coords())


# ======================================================================
# ObsSourceSet
# ======================================================================


class TestObsSourceSet:
    def test_from_config_positional_order_and_flags(self):
        model = FakeDA()
        obs_cfg = OmegaConf.create(
            {
                "conv": {
                    "_target_": "test.test_assimilation.FakeObsSource",
                    "value": 2.0,
                    "time_tolerance_hours": [-6, 3],
                },
                "sat": {
                    "_target_": "test.test_assimilation.FakeObsSource",
                    "enabled": False,
                },
            }
        )
        obs_set = ObsSourceSet.from_config(obs_cfg, model.input_coords())

        conv, sat = obs_set._sources
        assert isinstance(conv, FakeObsSource)
        assert conv.value == 2.0
        assert conv.time_tolerance == (
            np.timedelta64(-6, "h"),
            np.timedelta64(3, "h"),
        )
        assert sat is None

    def test_from_config_missing_target_raises(self):
        model = FakeDA()
        obs_cfg = OmegaConf.create({"conv": {"value": 1.0}, "sat": {"value": 1.0}})
        with pytest.raises(ValueError, match="_target_"):
            ObsSourceSet.from_config(obs_cfg, model.input_coords())

    def test_source_count_mismatch_raises(self):
        model = FakeDA()
        with pytest.raises(ValueError, match="one config entry is required"):
            ObsSourceSet([FakeObsSource()], model.input_coords())

    def test_fetch_sets_request_time_and_none_slots(self):
        model = FakeDA()
        obs_set = _make_obs_set(model, conv_source=FakeObsSource())
        time = np.datetime64("2024-01-01T00:00:00")

        conv, sat = obs_set.fetch(time)
        assert sat is None
        assert isinstance(conv, pd.DataFrame)
        np.testing.assert_array_equal(
            conv.attrs["request_time"],
            np.array([time], dtype="datetime64[ns]"),
        )
        # variables/fields derived from the model's conv schema
        assert set(conv["variable"]) == {"u", "v", "t"}


# ======================================================================
# Runners
# ======================================================================


class TestStatelessRunner:
    def test_analysis_time_coord(self):
        model = FakeDA()
        runner = StatelessAssimilationRunner(
            model, _make_obs_set(model, FakeObsSource())
        )
        time = np.datetime64("2024-01-01T00:00:00")

        analysis = runner.analysis(time)
        assert analysis.dims == ("time", "variable", "lat", "lon")
        np.testing.assert_array_equal(
            analysis.coords["time"].values,
            np.array([time], dtype="datetime64[ns]"),
        )

    def test_all_sources_disabled_raises(self):
        model = FakeDA()
        runner = StatelessAssimilationRunner(model, _make_obs_set(model))
        with pytest.raises(ValueError, match="disabled"):
            runner.analysis(np.datetime64("2024-01-01"))

    def test_stateful_model_rejected(self):
        model = FakeStatefulDA()
        with pytest.raises(NotImplementedError, match="stateful"):
            StatelessAssimilationRunner(model, _make_obs_set(model, FakeObsSource()))

    def test_build_runner_default_is_stateless(self):
        model = FakeDA()
        da_cfg = OmegaConf.create({})
        runner = build_runner(da_cfg, model, _make_obs_set(model, FakeObsSource()))
        assert isinstance(runner, StatelessAssimilationRunner)


# ======================================================================
# Conversion helpers
# ======================================================================


class TestConversionHelpers:
    def test_analysis_to_tensor(self):
        model = FakeDA()
        runner = StatelessAssimilationRunner(
            model, _make_obs_set(model, FakeObsSource())
        )
        analysis = runner.analysis(np.datetime64("2024-01-01"))

        x, coords = analysis_to_tensor(analysis, torch.device("cpu"))
        assert x.dtype == torch.float32
        assert x.shape == (1, len(DA_VARIABLES), 4, 8)
        assert list(coords.keys()) == ["time", "variable", "lat", "lon"]

    def test_insert_zero_lead_time(self):
        x = torch.zeros(1, 2, 4, 8)
        coords: CoordSystem = OrderedDict(
            {
                "time": np.array([np.datetime64("2024-01-01")]),
                "variable": np.array(DA_VARIABLES),
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            }
        )
        x2, coords2 = insert_zero_lead_time(x, coords)
        assert x2.shape == (1, 1, 2, 4, 8)
        assert list(coords2.keys()) == ["time", "lead_time", "variable", "lat", "lon"]
        assert coords2["lead_time"][0] == np.timedelta64(0, "ns")

    def test_spatial_ref_and_variables(self):
        model = FakeDA()
        ref = analysis_spatial_ref(model)
        assert list(ref.keys()) == ["lat", "lon"]
        assert analysis_variables(model) == DA_VARIABLES


# ======================================================================
# AssimilationPipeline (analysis mode)
# ======================================================================


def _analysis_cfg(tmp_path, variables=None) -> OmegaConf:
    return OmegaConf.create(
        {
            "project": "test_eval",
            "run_id": "da_unit",
            "start_times": ["2024-01-01 00:00:00"],
            "nsteps": 0,
            "ensemble_size": 1,
            "random_seed": 42,
            "pipeline": "src.pipelines.assimilation.AssimilationPipeline",
            "model": {
                "da": {
                    "architecture": "test.test_assimilation.FakeDA",
                    "load_args": {"variables": list(variables or DA_VARIABLES)},
                    "obs_sources": {
                        "conv": {"_target_": "test.test_assimilation.FakeObsSource"},
                        "sat": {
                            "_target_": "test.test_assimilation.FakeObsSource",
                            "enabled": False,
                        },
                    },
                }
            },
            "data_source": {"_target_": "earth2studio.data.Random"},
            "predownload": {"verification": {"enabled": False, "source": None}},
            "output": {
                "path": str(tmp_path / "outputs"),
                "variables": list(variables or DA_VARIABLES),
                "overwrite": True,
                "thread_writers": 0,
                "chunks": {"time": 1, "lead_time": 1},
            },
        }
    )


def _setup_pipeline(pipeline, cfg):
    """Run pipeline.setup with rank-0 loading patched out (no dist init)."""
    with patch(_RANK0_ASSIM, side_effect=_passthrough):
        pipeline.setup(cfg, torch.device("cpu"))
    return pipeline


class TestAssimilationPipeline:
    def test_setup_and_total_coords(self, tmp_path):
        cfg = _analysis_cfg(tmp_path)
        pipeline = _setup_pipeline(AssimilationPipeline(), cfg)

        times = np.array([np.datetime64("2024-01-01")])
        total = pipeline.build_total_coords(times, ensemble_size=1)
        assert list(total.keys()) == ["time", "lead_time", "lat", "lon"]
        assert len(total["lead_time"]) == 1
        assert total["lead_time"][0] == np.timedelta64(0, "ns")

        total_ens = pipeline.build_total_coords(times, ensemble_size=3)
        assert "ensemble" in total_ens

    def test_end_to_end_run(self, tmp_path):
        cfg = _analysis_cfg(tmp_path)
        pipeline = _setup_pipeline(AssimilationPipeline(), cfg)

        times = np.array([np.datetime64("2024-01-01")])
        total_coords = pipeline.build_total_coords(times, ensemble_size=1)
        items = [WorkItem(time=times[0], ensemble_id=0, seed=0)]

        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_OUTPUT, side_effect=_passthrough):
                with OutputManager(cfg) as mgr:
                    mgr.validate_output_store(total_coords, DA_VARIABLES)
                    pipeline.run(
                        work_items=items,
                        data_source=None,
                        output_mgr=mgr,
                        output_variables=DA_VARIABLES,
                        device=torch.device("cpu"),
                    )
                    for v in DA_VARIABLES:
                        assert v in mgr.io
                        assert mgr.io[v].shape == (1, 1, 4, 8)

    def test_predownload_stores_disabled(self, tmp_path):
        cfg = _analysis_cfg(tmp_path)
        pipeline = AssimilationPipeline()
        assert pipeline.predownload_stores(cfg) == []

    def test_predownload_stores_verification_only(self, tmp_path):
        cfg = _analysis_cfg(tmp_path)
        cfg.predownload.verification.enabled = True
        # Random needs domain_coords, which isn't expressible in plain YAML —
        # patch the source instantiation and check the declaration itself.
        source = Random(OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}))
        with patch(_RANK0_ASSIM, side_effect=_passthrough):
            with patch(
                "src.predownload_utils.hydra.utils.instantiate", return_value=source
            ):
                stores = AssimilationPipeline().predownload_stores(cfg)
        assert len(stores) == 1
        store = stores[0]
        assert store.name == "verification"
        assert store.role == "verification"
        assert store.variables == DA_VARIABLES
        assert store.times == [np.datetime64("2024-01-01T00:00:00")]
        assert list(store.spatial_ref.keys()) == ["lat", "lon"]


# ======================================================================
# declare_verification_only_store gates
# ======================================================================


class TestVerificationOnlyStore:
    def _cfg(self, enabled=True, byo=False):
        cfg = {
            "data_source": {"_target_": "earth2studio.data.Random"},
            "predownload": {"verification": {"enabled": enabled, "source": None}},
        }
        if byo:
            cfg["verification_source"] = {"_target_": "earth2studio.data.Random"}
        return OmegaConf.create(cfg)

    def test_disabled_returns_empty(self):
        stores = declare_verification_only_store(
            self._cfg(enabled=False),
            verif_variables=["t2m"],
            verif_times=[np.datetime64("2024-01-01")],
            spatial_ref=OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}),
        )
        assert stores == []

    def test_byo_returns_empty(self):
        stores = declare_verification_only_store(
            self._cfg(enabled=True, byo=True),
            verif_variables=["t2m"],
            verif_times=[np.datetime64("2024-01-01")],
            spatial_ref=OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}),
        )
        assert stores == []


# ======================================================================
# AssimilationForecastPipeline (DA-initialized forecast)
# ======================================================================


def _make_da_forecast_pipeline(
    da_variables: list[str],
    fill_source=None,
    nsteps: int = 2,
) -> AssimilationForecastPipeline:
    """Assemble a pipeline with pre-set attributes (bypassing setup)."""
    device = torch.device("cpu")
    domain = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
    prognostic = Persistence(variable=DA_VARIABLES, domain_coords=domain).to(device)

    pipeline = AssimilationForecastPipeline()
    pipeline.prognostic = prognostic
    pipeline.diagnostics = []
    pipeline.perturbation = None
    pipeline.nsteps = nsteps
    pipeline._prognostic_ic = prognostic.input_coords()
    pipeline._spatial_ref = prognostic.output_coords(pipeline._prognostic_ic)
    pipeline._dx_input_coords = {}

    da_model = FakeDA(variables=da_variables)
    pipeline.da_model = da_model
    pipeline.obs_set = _make_obs_set(da_model, conv_source=FakeObsSource())
    pipeline.runner = StatelessAssimilationRunner(da_model, pipeline.obs_set)
    pipeline._missing_vars = [v for v in DA_VARIABLES if v not in set(da_variables)]
    pipeline._fill_source = fill_source
    return pipeline


class TestAssimilationForecastPipeline:
    def test_run_item_full_da_coverage(self):
        pipeline = _make_da_forecast_pipeline(da_variables=DA_VARIABLES)
        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0)

        outputs = list(pipeline.run_item(item, None, torch.device("cpu")))
        assert len(outputs) == pipeline.nsteps + 1
        x0, coords0 = outputs[0]
        assert list(coords0["variable"]) == DA_VARIABLES
        # Step 0 is the (unperturbed) DA initial condition itself.
        assert pipeline.da_model.call_count >= 1

    def test_run_item_fills_missing_variables(self):
        domain = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        pipeline = _make_da_forecast_pipeline(
            da_variables=["t2m"],  # z500 must come from the fill source
            fill_source=Random(domain),
        )
        assert pipeline._missing_vars == ["z500"]

        item = WorkItem(time=np.datetime64("2024-01-01"), ensemble_id=0, seed=0)
        outputs = list(pipeline.run_item(item, None, torch.device("cpu")))
        assert len(outputs) == pipeline.nsteps + 1
        _, coords0 = outputs[0]
        assert list(coords0["variable"]) == DA_VARIABLES

    def test_setup_computes_missing_and_resolves_fill(self, tmp_path):
        cfg = _analysis_cfg(tmp_path, variables=["t2m"])
        cfg.pipeline = "src.pipelines.assimilation.AssimilationForecastPipeline"
        cfg.model.forecast = {"architecture": "unused.Fake"}
        cfg.nsteps = 2

        domain = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        prognostic = Persistence(variable=DA_VARIABLES, domain_coords=domain)
        fill = Random(domain)

        with patch("src.pipelines.forecast.load_prognostic", return_value=prognostic):
            with patch("src.data.resolve_ic_source", return_value=fill):
                pipeline = _setup_pipeline(AssimilationForecastPipeline(), cfg)

        assert pipeline._missing_vars == ["z500"]
        assert pipeline._fill_source is fill

    def test_setup_fill_disabled_raises(self, tmp_path):
        cfg = _analysis_cfg(tmp_path, variables=["t2m"])
        cfg.model.forecast = {"architecture": "unused.Fake"}
        cfg.model.fill_missing_variables = False
        cfg.nsteps = 2

        domain = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        prognostic = Persistence(variable=DA_VARIABLES, domain_coords=domain)

        with patch("src.pipelines.forecast.load_prognostic", return_value=prognostic):
            with pytest.raises(ValueError, match="does not provide"):
                _setup_pipeline(AssimilationForecastPipeline(), cfg)


# ======================================================================
# Observation predownload — frame stores
# ======================================================================


def _write_frame_store(output_path, name, times, source=None):
    """Write parquet frames for *times* the way predownload.py does."""
    from pathlib import Path

    source = source or FakeObsSource()
    store_dir = Path(frame_store_path(str(output_path), name))
    store_dir.mkdir(parents=True, exist_ok=True)
    for t in times:
        df = source(
            np.array([t], dtype="datetime64[ns]"),
            np.array(["u", "v", "t"]),
        )
        df.to_parquet(store_dir / frame_filename(t))
    return store_dir


class TestPredownloadedFrameSource:
    def test_round_trip_single_time(self, tmp_path):
        t = np.datetime64("2024-01-01T00:00:00")
        store_dir = _write_frame_store(tmp_path, "obs_conv", [t])

        src = PredownloadedFrameSource(str(store_dir))
        df = src(t, np.array(["u", "v", "t"]))
        assert isinstance(df, pd.DataFrame)
        assert set(df["variable"]) == {"u", "v", "t"}

    def test_multiple_times_concatenated(self, tmp_path):
        times = [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-02-01T00:00:00"),
        ]
        store_dir = _write_frame_store(tmp_path, "obs_conv", times)

        src = PredownloadedFrameSource(str(store_dir))
        df = src(np.array(times, dtype="datetime64[ns]"), np.array(["u"]))
        # Row filter narrows to the requested variable subset.
        assert set(df["variable"]) == {"u"}
        assert len(df) == 2  # one 'u' row per stored time

    def test_missing_time_raises(self, tmp_path):
        store_dir = _write_frame_store(
            tmp_path, "obs_conv", [np.datetime64("2024-01-01T00:00:00")]
        )
        src = PredownloadedFrameSource(str(store_dir))
        with pytest.raises(FileNotFoundError, match="Re-run predownload"):
            src(np.datetime64("2024-06-01T00:00:00"), np.array(["u"]))

    def test_missing_store_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no frame store"):
            PredownloadedFrameSource(str(tmp_path / "obs_missing.parquet"))

    def test_fields_filter_and_missing_field_raises(self, tmp_path):
        t = np.datetime64("2024-01-01T00:00:00")
        store_dir = _write_frame_store(tmp_path, "obs_conv", [t])
        src = PredownloadedFrameSource(str(store_dir))

        df = src(t, np.array(["u"]), fields=np.array(["time", "observation"]))
        assert list(df.columns) == ["time", "observation"]

        with pytest.raises(KeyError, match="missing requested fields"):
            src(t, np.array(["u"]), fields=np.array(["not_a_column"]))

    def test_fetch_dataframe_attaches_request_time(self, tmp_path):
        from earth2studio.data import fetch_dataframe

        t = np.datetime64("2024-01-01T00:00:00")
        store_dir = _write_frame_store(tmp_path, "obs_conv", [t])
        src = PredownloadedFrameSource(str(store_dir))

        times = np.array([t], dtype="datetime64[ns]")
        df = fetch_dataframe(src, time=times, variable=np.array(["u", "v", "t"]))
        np.testing.assert_array_equal(df.attrs["request_time"], times)


class TestObsFrameStoreDeclaration:
    def test_obs_set_declares_enabled_slots_only(self):
        model = FakeDA()
        obs_set = ObsSourceSet(
            [FakeObsSource(), None], model.input_coords(), names=["conv", "sat"]
        )
        times = [np.datetime64("2024-01-01T00:00:00")]

        stores = obs_set.predownload_frame_stores(times)
        assert len(stores) == 1
        store = stores[0]
        assert store.name == "obs_conv"
        assert store.role == "observation"
        assert store.times == times
        assert store.variables == ["u", "v", "t"]
        assert store.fields == ["time", "lat", "lon", "observation", "variable"]

    def test_analysis_pipeline_frame_stores(self, tmp_path):
        cfg = _analysis_cfg(tmp_path)
        with patch(_RANK0_ASSIM, side_effect=_passthrough):
            stores = AssimilationPipeline().predownload_frame_stores(cfg)
        assert [s.name for s in stores] == ["obs_conv"]  # sat is disabled
        assert stores[0].times == [np.datetime64("2024-01-01T00:00:00")]
        assert isinstance(stores[0].source, FakeObsSource)

    def test_analysis_pipeline_frame_stores_gate_disabled(self, tmp_path):
        cfg = _analysis_cfg(tmp_path)
        cfg.predownload.observations = {"enabled": False}
        assert AssimilationPipeline().predownload_frame_stores(cfg) == []

    def test_forecast_pipeline_frame_stores(self, tmp_path):
        cfg = _analysis_cfg(tmp_path)
        cfg.pipeline = "src.pipelines.assimilation.AssimilationForecastPipeline"
        cfg.model.forecast = {"architecture": "unused.Fake"}

        domain = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        prognostic = Persistence(variable=DA_VARIABLES, domain_coords=domain)
        ic_leads = prognostic.input_coords()["lead_time"]
        ic_time = np.datetime64("2024-01-01T00:00:00")
        expected_times = sorted({ic_time + lt for lt in ic_leads})

        with patch("src.models.load_prognostic", return_value=prognostic):
            with patch(_RANK0_ASSIM, side_effect=_passthrough):
                pipeline = AssimilationForecastPipeline()
                stores = pipeline.predownload_frame_stores(cfg)
        assert [s.name for s in stores] == ["obs_conv"]
        assert stores[0].times == expected_times

    def test_from_config_prefers_predownloaded_store(self, tmp_path):
        model = FakeDA()
        output_path = tmp_path / "outputs"
        _write_frame_store(
            output_path, "obs_conv", [np.datetime64("2024-01-01T00:00:00")]
        )
        obs_cfg = OmegaConf.create(
            {
                "conv": {"_target_": "test.test_assimilation.FakeObsSource"},
                "sat": {
                    "_target_": "test.test_assimilation.FakeObsSource",
                    "enabled": False,
                },
            }
        )

        obs_set = ObsSourceSet.from_config(
            obs_cfg, model.input_coords(), output_path=str(output_path)
        )
        conv, sat = obs_set._sources
        assert isinstance(conv, PredownloadedFrameSource)
        assert sat is None

        # Without output_path (the predownload declaration path) the live
        # source is constructed even when a frame store exists on disk.
        obs_set_live = ObsSourceSet.from_config(obs_cfg, model.input_coords())
        assert isinstance(obs_set_live._sources[0], FakeObsSource)

    def test_end_to_end_run_from_predownloaded_obs(self, tmp_path):
        cfg = _analysis_cfg(tmp_path)
        t = np.datetime64("2024-01-01T00:00:00")
        _write_frame_store(cfg.output.path, "obs_conv", [t])

        pipeline = _setup_pipeline(AssimilationPipeline(), cfg)
        assert isinstance(pipeline.obs_set._sources[0], PredownloadedFrameSource)

        analysis = pipeline.runner.analysis(t)
        np.testing.assert_array_equal(
            analysis.coords["time"].values,
            np.array([t], dtype="datetime64[ns]"),
        )


class TestDownloadFrameStore:
    def _run(self, cfg, store, overwrite=False):
        from predownload import _download_frame_store

        dist = _make_dist_mock()
        _download_frame_store(cfg, dist, store, overwrite)

    def test_writes_parquet_and_markers_with_resume(self, tmp_path):
        from pathlib import Path

        from src.pipelines import PredownloadFrameStore

        cfg = _analysis_cfg(tmp_path)
        source = FakeObsSource()
        times = [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-02-01T00:00:00"),
        ]
        store = PredownloadFrameStore(
            name="obs_conv",
            source=source,
            times=times,
            variables=["u", "v", "t"],
            fields=["time", "lat", "lon", "observation", "variable"],
        )

        self._run(cfg, store)
        store_dir = Path(frame_store_path(cfg.output.path, "obs_conv"))
        for t in times:
            assert (store_dir / frame_filename(t)).exists()
        assert source.call_count == 2

        # Second run resumes off the markers — no further fetches.
        self._run(cfg, store)
        assert source.call_count == 2

        # The written store round-trips through the runtime reader.
        reader = PredownloadedFrameSource(str(store_dir))
        df = reader(times[0], np.array(["u", "v", "t"]))
        assert set(df["variable"]) == {"u", "v", "t"}

    def test_all_empty_frames_logs_error_and_still_writes(self, tmp_path):
        """Every-frame-empty (dates outside archive coverage) is surfaced loudly.

        The obs sources no longer raise on missing files, so predownload
        must escalate the fully-empty case to a prominent ERROR while
        still writing the (empty) frames so the run can proceed.
        """
        from pathlib import Path

        from loguru import logger
        from src.pipelines import PredownloadFrameStore

        cfg = _analysis_cfg(tmp_path)
        source = FakeObsSource(value=0.0)
        # Make every fetch come back empty (simulates a total archive gap).
        source._force_empty = True

        times = [
            np.datetime64("2024-01-01T00:00:00"),
            np.datetime64("2024-02-01T00:00:00"),
        ]
        store = PredownloadFrameStore(
            name="obs_conv",
            source=source,
            times=times,
            variables=["u", "v", "t"],
            fields=["time", "lat", "lon", "observation", "variable"],
        )

        errors: list[str] = []
        sink_id = logger.add(errors.append, level="ERROR")
        try:
            self._run(cfg, store)
        finally:
            logger.remove(sink_id)

        # Loud ERROR naming the coverage/date-range cause.
        assert any("EVERY fetched frame" in m for m in errors)

        # Frames are still written (empty) so a resumed run doesn't refetch.
        store_dir = Path(frame_store_path(cfg.output.path, "obs_conv"))
        for t in times:
            assert (store_dir / frame_filename(t)).exists()
        reader = PredownloadedFrameSource(str(store_dir))
        assert len(reader(times[0], np.array(["u", "v", "t"]))) == 0

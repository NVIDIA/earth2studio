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

"""Data-assimilation runners: observation sources, analyses and DA-initialized forecasts.

A ``FakeDA`` independent DA model with two ``FrameSchema`` input slots and a
4 x 8 lat/lon analysis, plus an in-memory ``FakeObsSource``, so everything
runs on CPU without downloads.
"""

from __future__ import annotations

import os
import threading
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf, open_dict
from src.data import PredownloadedFrameSource, frame_filename, frame_store_path
from src.runners import build_runner
from src.runners.assimilation import (
    AnalysisSource,
    AssimilationRunner,
    GridAlignedSource,
    ObsSources,
    analysis_coords,
    assimilation,
    assimilation_forecast,
    shared_window_guard,
)

from earth2studio.data import CompositeSource
from earth2studio.io import KVBackend
from earth2studio.run import Pipeline, PrognosticRunner, work_items
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, coord_array
from test.conftest import SMALL_LAT, SMALL_LON, compose_cfg

DA_VARIABLES = ["t2m", "z500"]
T0 = np.datetime64("2024-01-01T00:00:00")
T1 = np.datetime64("2024-01-02T00:00:00")
H0 = np.timedelta64(0, "h")
H6 = np.timedelta64(6, "h")


class FakeObsSource:
    """DataFrameSource returning a fixed-value long-layout observation frame."""

    SCHEMA = pa.schema(
        [
            ("time", pa.timestamp("ns")),
            ("lat", pa.float32()),
            ("lon", pa.float32()),
            ("observation", pa.float32()),
            ("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        value: float = 1.0,
        time_tolerance: tuple | None = None,
        cache: bool = True,
    ) -> None:
        self.value = value
        self.time_tolerance = time_tolerance
        self.cache = cache
        self.calls: list[tuple[np.ndarray, list[str], list[str] | None]] = []

    def __call__(self, time, variable, fields=None) -> pd.DataFrame:
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = [str(v) for v in np.atleast_1d(variable)]
        wanted = None if fields is None else [str(f) for f in np.atleast_1d(fields)]
        self.calls.append((times, variables, wanted))
        n = len(times) * len(variables)
        data = {
            "time": np.repeat(times, len(variables)),
            "lat": np.zeros(n, dtype=np.float32),
            "lon": np.zeros(n, dtype=np.float32),
            "observation": np.full(n, self.value, dtype=np.float32),
            "variable": np.tile(np.array(variables), len(times)),
        }
        if wanted is not None:
            data = {k: v for k, v in data.items() if k in wanted}
        return pd.DataFrame(data)


def _schema(variables: list[str]) -> OrderedDict:
    return OrderedDict(
        {
            "time": np.empty(0, dtype="datetime64[ns]"),
            "lat": np.empty(0, dtype=np.float32),
            "lon": np.empty(0, dtype=np.float32),
            "observation": np.empty(0, dtype=np.float32),
            "variable": np.array(variables, dtype=str),
        }
    )


def analysis_value(time: np.datetime64, variable_index: int) -> float:
    """Deterministic analysis value: hours since the epoch plus the variable index."""
    hours = np.datetime64(time, "ns").astype("datetime64[h]").astype("int64")
    return float(hours + variable_index)


class FakeDA:
    """Independent AssimilationModel with (conv, sat) FrameSchema inputs."""

    stochastic = False

    def __init__(self, variables: list[str] | None = None) -> None:
        self.variables = list(variables or DA_VARIABLES)
        self.lat = SMALL_LAT
        self.lon = SMALL_LON
        self.calls: list[tuple] = []
        self.seeds: list[int] = []

    @classmethod
    def load_default_package(cls):
        return None

    @classmethod
    def load_model(cls, package=None, **kwargs):
        return cls(**kwargs)

    def set_rng(self, seed: int, reset: bool = True) -> None:
        self.seeds.append(seed)

    def input_coords(self):
        return _schema(["u", "v", "t"]), _schema(["atms"])

    def output_coords(self, conv_obs=None, sat_obs=None, request_time=None):
        if request_time is None:
            request_time = np.array([np.datetime64("NaT")], dtype="datetime64[ns]")
        return coord_array(
            ("time", "variable", "lat", "lon"),
            {
                "time": request_time,
                "variable": np.array(self.variables, dtype=str),
                "lat": self.lat,
                "lon": self.lon,
            },
        )

    def default_sources(self):
        return FakeObsSource(value=10.0), FakeObsSource(value=20.0)

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, conv_obs=None, sat_obs=None) -> xr.DataArray:
        if conv_obs is None and sat_obs is None:
            raise ValueError("At least one observation frame is required.")
        frame = conv_obs if conv_obs is not None else sat_obs
        request_time = np.asarray(frame.attrs["request_time"], dtype="datetime64[ns]")
        self.calls.append((conv_obs, sat_obs))
        coords = self.output_coords(request_time=request_time)
        data = np.empty(
            (len(request_time), len(self.variables), len(self.lat), len(self.lon)),
            np.float32,
        )
        for i, t in enumerate(request_time):
            for j in range(len(self.variables)):
                data[i, j] = analysis_value(t, j)
        return xr.DataArray(data, dims=coords.dims, coords=coords.coords)


class StochasticDA(FakeDA):
    stochastic = True


def _obs(model, conv=None, sat=None, names=("conv", "sat")) -> ObsSources:
    return ObsSources([conv, sat], model.input_coords(), names=names)


# -- ObsSources ---------------------------------------------------------------


def test_from_config_matches_slots_positionally_and_disables() -> None:
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
    obs = ObsSources.from_config(model, obs_cfg)
    conv, sat = obs.sources
    assert obs.names == ("conv", "sat")
    assert isinstance(conv, FakeObsSource) and conv.value == 2.0
    assert conv.time_tolerance == (np.timedelta64(-6, "h"), np.timedelta64(3, "h"))
    assert sat is None
    assert obs.enabled
    assert obs.roles == {"conv": conv, "sat": None}


def test_from_config_falls_back_to_model_defaults() -> None:
    model = FakeDA()
    obs = ObsSources.from_config(model, None)
    assert obs.names == ("obs0", "obs1")
    assert [s.value for s in obs.sources] == [10.0, 20.0]
    # A null entry picks the default for that slot only.
    obs_cfg = OmegaConf.create(
        {
            "conv": None,
            "sat": {"_target_": "test.test_assimilation.FakeObsSource", "value": 3.0},
        }
    )
    obs = ObsSources.from_config(model, obs_cfg)
    assert [s.value for s in obs.sources] == [10.0, 3.0]


def test_from_config_rejects_bad_entries() -> None:
    model = FakeDA()
    with pytest.raises(ValueError, match="one per input slot"):
        ObsSources.from_config(model, OmegaConf.create({"conv": {"_target_": "x"}}))
    with pytest.raises(ValueError, match="_target_"):
        ObsSources.from_config(
            model, OmegaConf.create({"conv": {"value": 1}, "sat": {"value": 1}})
        )
    with pytest.raises(ValueError, match="one entry is required"):
        ObsSources([FakeObsSource()], model.input_coords())


def test_from_config_prefers_predownloaded_frame_store(tmp_path) -> None:
    model = FakeDA()
    store = frame_store_path(str(tmp_path), "conv")
    os.makedirs(store)
    pd.DataFrame({"observation": [1.0], "variable": ["u"]}).to_parquet(
        os.path.join(store, frame_filename(T0))
    )
    obs_cfg = OmegaConf.create(
        {
            "conv": {"_target_": "test.test_assimilation.FakeObsSource"},
            "sat": {"_target_": "test.test_assimilation.FakeObsSource"},
        }
    )
    obs = ObsSources.from_config(model, obs_cfg, output_path=str(tmp_path))
    assert isinstance(obs.sources[0], PredownloadedFrameSource)
    assert isinstance(obs.sources[1], FakeObsSource)
    live = ObsSources.from_config(
        model, obs_cfg, output_path=str(tmp_path), prefer_predownloaded=False
    )
    assert isinstance(live.sources[0], FakeObsSource)


def test_fetch_sets_request_time_fields_and_none_slots() -> None:
    model = FakeDA()
    conv_source = FakeObsSource()
    conv, sat = _obs(model, conv=conv_source).fetch(T0)
    assert sat is None
    np.testing.assert_array_equal(
        conv.attrs["request_time"], np.array([T0], dtype="datetime64[ns]")
    )
    assert set(conv["variable"]) == {"u", "v", "t"}
    # The long-layout source lists 'variable' in its SCHEMA, so the fetch requests it.
    _, _, fields = conv_source.calls[0]
    assert fields == ["time", "lat", "lon", "observation", "variable"]


def test_requests_cover_enabled_slots_only() -> None:
    model = FakeDA()
    obs = _obs(model, conv=FakeObsSource())
    requests = obs.requests(np.array([T0, T1]))
    assert [r.name for r in requests] == ["conv"]
    np.testing.assert_array_equal(
        requests[0].time, np.array([T0, T1], dtype="datetime64[ns]")
    )
    assert list(requests[0].variable) == ["u", "v", "t"]


def test_variable_is_a_requested_column_only_of_long_frames() -> None:
    # A wide frame names each variable in its columns; a wide-layout source
    # rejects a "variable" field even though its long SCHEMA lists one.
    wide = OrderedDict(
        (k, v) for k, v in _schema(["u", "v"]).items() if k != "observation"
    )
    wide["observation_u"] = np.empty(0, dtype=np.float32)
    obs = ObsSources([FakeObsSource(), FakeObsSource()], [_schema(["t"]), wide])
    long_request, wide_request = obs.requests(np.array([T0]))
    assert "variable" in list(long_request.fields)
    assert "variable" not in list(wide_request.fields)
    assert "observation_u" in list(wide_request.fields)


def test_window_guard_marks_only_a_successful_fetch(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    with shared_window_guard(T0):
        pass
    assert (tmp_path / "obs_windows" / "2024-01-01T000000.done").exists()
    # A failed fetch leaves no marker, so the next process fetches itself.
    with pytest.raises(RuntimeError, match="fetch failed"):
        with shared_window_guard(T1):
            raise RuntimeError("fetch failed")
    assert not (tmp_path / "obs_windows" / "2024-01-02T000000.done").exists()


def test_window_guard_lets_one_process_fetch_first(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    events: list[str] = []
    first_in, release = threading.Event(), threading.Event()

    def first() -> None:
        with shared_window_guard(T0):
            events.append("first")
            first_in.set()
            release.wait(10)

    def second() -> None:
        first_in.wait(10)
        with shared_window_guard(T0):
            events.append("second")

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    first_in.wait(10)
    time.sleep(0.3)  # long enough for the second fetch to reach the lock
    assert events == ["first"]
    release.set()
    for thread in threads:
        thread.join(10)
    assert events == ["first", "second"]


def test_single_process_fetch_skips_the_window_guard(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    model = FakeDA()
    _obs(model, conv=FakeObsSource()).fetch(T0)
    assert not (tmp_path / "obs_windows").exists()


# -- AssimilationRunner -------------------------------------------------------


def test_analysis_coords_makes_time_dynamic() -> None:
    signature = analysis_coords(FakeDA())
    assert signature.dims == ("time", "variable", "lat", "lon")
    assert tuple(signature.attrs[E2S_DYNAMIC_DIMS]) == ("time",)
    assert signature.sizes["time"] == 0


def test_runner_output_layout_and_requests() -> None:
    model = FakeDA()
    runner = AssimilationRunner(model, _obs(model, conv=FakeObsSource()), seed=7)
    schemas = runner.output_coords(H0)
    assert list(schemas) == ["analysis"]
    assert schemas["analysis"].dims == ("time", "lead_time", "variable", "lat", "lon")
    np.testing.assert_array_equal(schemas["analysis"]["lead_time"].values, [H0])
    assert runner.step is None and runner.supports_member_batching is False
    assert runner.source_roles == {} and list(runner.frame_roles) == ["conv", "sat"]

    item = work_items([T0], H0)[0]
    assert runner.data_requests(item) == ()
    (request,) = runner.frame_requests(item)
    assert request.name == "conv"
    np.testing.assert_array_equal(request.time, np.array([T0], dtype="datetime64[ns]"))

    steps = list(runner.run_item(item))
    assert len(steps) == 1 and list(steps[0]) == ["analysis"]
    x = steps[0]["analysis"]
    assert x.dims == ("time", "lead_time", "variable", "lat", "lon")
    np.testing.assert_array_equal(
        x["time"].values, np.array([T0], dtype="datetime64[ns]")
    )
    np.testing.assert_array_equal(x["lead_time"].values, [H0])
    assert float(
        x.sel(variable="z500").isel(time=0, lead_time=0, lat=0, lon=0)
    ) == analysis_value(T0, 1)
    assert model.seeds == []  # deterministic model is not seeded


def test_runner_seeds_stochastic_model_per_member() -> None:
    model = StochasticDA()
    runner = AssimilationRunner(model, _obs(model, conv=FakeObsSource()), seed=7)
    items = work_items([T0], H0, members=[0, 1])
    for item in items:
        list(runner.run_item(item))
    assert len(model.seeds) == 2 and model.seeds[0] != model.seeds[1]


def test_runner_rejects_disabled_slots_and_lead_times() -> None:
    model = FakeDA()
    with pytest.raises(ValueError, match="disabled"):
        AssimilationRunner(model, _obs(model))
    runner = AssimilationRunner(model, _obs(model, conv=FakeObsSource()))
    with pytest.raises(ValueError, match="nsteps=0"):
        runner.output_coords(H6)


def test_pipeline_writes_analysis_store() -> None:
    model = FakeDA()
    runner = AssimilationRunner(model, _obs(model, sat=FakeObsSource()))
    store = KVBackend()
    Pipeline(runner, output={"analysis": store}, device="cpu").run(
        work_items([T0, T1], H0)
    )
    assert tuple(store.dims["t2m"]) == ("time", "lead_time", "lat", "lon")
    assert store["t2m"].shape == (2, 1, len(SMALL_LAT), len(SMALL_LON))
    assert float(store["t2m"][1, 0, 0, 0]) == analysis_value(T1, 0)
    assert len(model.calls) == 2 and all(conv is None for conv, _ in model.calls)


def test_pipeline_ensemble_items_get_an_ensemble_axis() -> None:
    model = FakeDA()
    runner = AssimilationRunner(model, _obs(model, conv=FakeObsSource()))
    store = KVBackend()
    Pipeline(runner, output={"analysis": store}, device="cpu").run(
        work_items([T0], H0, members=2)
    )
    assert tuple(store.dims["t2m"]) == ("ensemble", "time", "lead_time", "lat", "lon")
    assert store["t2m"].shape[:3] == (2, 1, 1)


# -- AnalysisSource and the DA-initialized forecast ---------------------------


def test_analysis_source_runs_one_analysis_per_time() -> None:
    model = FakeDA()
    source = AnalysisSource(model, _obs(model, conv=FakeObsSource()))
    field = source(np.array([T0, T1]), ["z500"])
    assert field.dims == ("time", "variable", "lat", "lon")
    assert field.shape == (2, 1, len(SMALL_LAT), len(SMALL_LON))
    assert len(model.calls) == 2
    assert float(field.isel(time=1, variable=0, lat=0, lon=0)) == analysis_value(T1, 1)
    assert isinstance(field.data, np.ndarray)
    with pytest.raises(KeyError, match="sst"):
        source(T0, ["sst"])
    (request,) = source.frame_requests(np.array([T0]))
    assert request.name == "conv"


def _da_forecast_cfg(tmp_path, variables, fill=True, source_grid=None):
    cfg = compose_cfg(tmp_path, ["nsteps=1", "runner=assimilation_forecast"])
    lat = (SMALL_LAT if source_grid is None else source_grid[0]).tolist()
    lon = (SMALL_LON if source_grid is None else source_grid[1]).tolist()
    with open_dict(cfg):
        cfg.model = {
            "da": {
                "architecture": "test.test_assimilation.FakeDA",
                "load_args": {"variables": list(DA_VARIABLES)},
                "obs_sources": {
                    "conv": {"_target_": "test.test_assimilation.FakeObsSource"},
                    "sat": {
                        "_target_": "test.test_assimilation.FakeObsSource",
                        "enabled": False,
                    },
                },
            },
            "forecast": {
                "_target_": "earth2studio.models.px.Persistence",
                "variable": list(variables),
                "domain_coords": "latlon-small",
            },
            "fill_missing_variables": fill,
        }
        cfg.data_source = {
            "_target_": "earth2studio.data.Random",
            "domain_coords": {"lat": lat, "lon": lon},
        }
        cfg.output.variables = list(variables)
    return cfg


def test_assimilation_builder_from_config(tmp_path) -> None:
    cfg = _da_forecast_cfg(tmp_path, DA_VARIABLES)
    with open_dict(cfg):
        cfg.runner = "assimilation"
        cfg.nsteps = 0
    runner = build_runner(cfg)
    assert isinstance(runner, AssimilationRunner)
    assert runner.obs.names == ("conv", "sat") and runner.obs.sources[1] is None
    assert isinstance(assimilation(cfg), AssimilationRunner)
    with open_dict(cfg):
        cfg.model.da.runner = {"_target_": "x"}
    with pytest.raises(ValueError, match="model.da.runner"):
        assimilation(cfg)


def test_forecast_without_fill_uses_the_analysis_directly(tmp_path) -> None:
    cfg = _da_forecast_cfg(tmp_path, ["t2m"])
    runner = build_runner(cfg)
    assert isinstance(runner, PrognosticRunner)
    assert isinstance(runner.source, AnalysisSource)
    assert "data" not in runner.source_roles
    item = work_items([T0], H6)[0]
    assert runner.data_requests(item) == ()
    (request,) = runner.frame_requests(item)
    assert request.name == "conv"
    np.testing.assert_array_equal(request.time, np.array([T0], dtype="datetime64[ns]"))
    steps = [s["forecast"] for s in runner.run_item(item)]
    assert len(steps) == 2
    assert float(
        steps[0].sel(variable="t2m").isel(time=0, lead_time=0, lat=0, lon=0)
    ) == analysis_value(T0, 0)


def test_forecast_fills_missing_variables_from_data_source(tmp_path) -> None:
    cfg = _da_forecast_cfg(tmp_path, ["t2m", "sst"])
    runner = assimilation_forecast(cfg)
    assert isinstance(runner.source, CompositeSource)
    assert runner.source.variables == {"analysis": ("t2m", "z500"), "fill": ("sst",)}
    assert "data" in runner.source_roles
    item = work_items([T0], H6)[0]
    (request,) = runner.data_requests(item)
    assert request.source is runner.source_roles["data"]
    assert list(request.variable) == ["sst"]

    store = KVBackend()
    Pipeline(runner, output={"forecast": store}, device="cpu").run([item])
    assert tuple(store.dims["t2m"]) == ("time", "lead_time", "lat", "lon")
    assert store["t2m"].shape == (1, 2, len(SMALL_LAT), len(SMALL_LON))
    assert float(store["t2m"][0, 0, 0, 0]) == analysis_value(T0, 0)
    assert np.isfinite(store["sst"]).all()


def test_forecast_fill_disabled_raises(tmp_path) -> None:
    cfg = _da_forecast_cfg(tmp_path, ["t2m", "sst"], fill=False)
    with pytest.raises(ValueError, match="fill_missing_variables"):
        assimilation_forecast(cfg)


def test_fill_variables_come_from_the_data_source(tmp_path) -> None:
    # The analysis provides z500, but the campaign takes it from the data
    # source, even with filling of missing variables turned off.
    cfg = _da_forecast_cfg(tmp_path, ["t2m", "z500"], fill=False)
    with open_dict(cfg):
        cfg.model.fill_variables = ["z500"]
    runner = assimilation_forecast(cfg)
    assert isinstance(runner.source, CompositeSource)
    assert runner.source.variables == {"analysis": ("t2m",), "fill": ("z500",)}
    item = work_items([T0], H6)[0]
    (request,) = runner.data_requests(item)  # predownload caches it too
    assert list(request.variable) == ["z500"]
    field = runner.source(np.array([T0]), ["t2m", "z500"])
    assert float(field.isel(time=0, variable=0, lat=0, lon=0)) == analysis_value(T0, 0)
    assert float(field.isel(time=0, variable=1, lat=0, lon=0)) != analysis_value(T0, 1)


def test_fill_variables_must_be_prognostic_inputs(tmp_path) -> None:
    cfg = _da_forecast_cfg(tmp_path, ["t2m"])
    with open_dict(cfg):
        cfg.model.fill_variables = ["sst"]
    with pytest.raises(ValueError, match="fill_variables"):
        assimilation_forecast(cfg)


def test_fill_source_is_regridded_onto_the_analysis_grid(tmp_path) -> None:
    fine = (np.linspace(90, -90, 8), np.linspace(0, 360, 16, endpoint=False))
    cfg = _da_forecast_cfg(tmp_path, ["t2m", "sst"], source_grid=fine)
    runner = assimilation_forecast(cfg)
    fill = runner.source.sources["fill"]
    assert isinstance(fill, GridAlignedSource)
    field = fill(np.array([T0]), ["sst"])
    assert field.shape[-2:] == (len(SMALL_LAT), len(SMALL_LON))
    assert fill._aligned is not fill.source
    item = work_items([T0], H6)[0]
    steps = list(runner.run_item(item))
    assert steps[0]["forecast"].shape[-2:] == (len(SMALL_LAT), len(SMALL_LON))


def test_forecast_runner_moves_the_da_model_too(tmp_path) -> None:
    cfg = _da_forecast_cfg(tmp_path, DA_VARIABLES)
    runner = assimilation_forecast(cfg)
    analysis = runner.source
    assert isinstance(analysis, AnalysisSource)
    assert not hasattr(analysis.model, "device")
    assert runner.to(torch.device("cpu")) is runner
    assert analysis.model.device == torch.device("cpu")

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

"""Tests for offline scoring (src/scoring.py and score.py).

The tests write the forecast and verification stores through the recipe's own
IO layer (``src.output.build_backend`` and the library ``ZarrBackend``) with
closed-form fields, so every score has an analytic expectation.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf, open_dict
from score import run as run_score
from src.data import PredownloadedSource
from src.metrics import ensemble_variance, mse
from src.output import build_backend, parallel_coords, store_path
from src.scoring import (
    RegionalMetric,
    _apply_valid_ranges,
    _concat_chunks,
    _is_statistic,
    add_score_arrays,
    all_score_variable_names,
    apply_valid_ranges,
    build_input_coords_template,
    build_lead_time_chunks,
    group_score_arrays_by_dims,
    instantiate_metrics,
    load_prediction_chunk,
    load_verification_chunk,
    open_score_store,
    open_verification_source,
    score_variable_names,
    scoring_progress_path,
    spatial_coords_from_dataset,
    validate_lead_time_chunking,
)

from earth2studio.io import ZarrBackend
from earth2studio.run import ProgressStore, work_items
from earth2studio.statistics import rmse
from earth2studio.utils.coords import coord_array

from .conftest import SMALL_LAT, SMALL_LON, VARIABLES, compose_cfg

TIMES = np.array(["2024-01-01T00", "2024-01-01T12"], dtype="datetime64[ns]")
LEAD_TIMES = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
# Every valid time the two forecasts reach, on the 6-hourly grid.
VALID_TIMES = np.arange(
    TIMES[0],
    TIMES[-1] + LEAD_TIMES[-1] + np.timedelta64(6, "h"),
    np.timedelta64(6, "h"),
).astype("datetime64[ns]")

# z500 forecast: a latitude profile (lat + 90) / 90 = [2, 4/3, 2/3, 0] on
# the small grid, verified against zero.
Z500_PROFILE = (SMALL_LAT + 90.0) / 90.0
Z500_RMSE_GLOBAL = np.sqrt(np.mean(Z500_PROFILE**2))  # sqrt(14/9)
Z500_RMSE_NORTH = np.sqrt(np.mean(Z500_PROFILE[:2] ** 2))  # sqrt(26/9)
Z500_RMSE_LATW = np.sqrt(
    np.sum(np.cos(np.deg2rad(SMALL_LAT)) * Z500_PROFILE**2)
    / np.sum(np.cos(np.deg2rad(SMALL_LAT)))
)  # sqrt(10/9)
# t2m forecast: a constant; verification counts 6-hour steps since TIMES[0].
T2M_FORECAST = 100.0


def _t2m_truth(valid: np.ndarray) -> np.ndarray:
    return (np.asarray(valid).astype("datetime64[ns]") - TIMES[0]).astype(
        "timedelta64[h]"
    ).astype(float) / 6.0


# ---------------------------------------------------------------------------
# Store writers (the recipe's own IO paths)
# ---------------------------------------------------------------------------


def write_forecast_store(cfg, fields: dict[str, np.ndarray], ensemble=None) -> str:
    """Write ``forecast.zarr`` with ``fields[var]`` of shape
    ``(ensemble?, time, lead_time, lat, lon)`` through ``src.output``."""
    members = None if ensemble is None else list(ensemble)
    items = work_items(list(TIMES), LEAD_TIMES[-1], members=members)
    parallel = parallel_coords(items, LEAD_TIMES)
    path = store_path(cfg)
    io = build_backend(cfg, path, parallel, overwrite=True, rank=0)
    dims = (["ensemble"] if ensemble is not None else []) + [
        "time",
        "lead_time",
        "variable",
        "lat",
        "lon",
    ]
    coords = {
        "time": TIMES,
        "lead_time": LEAD_TIMES,
        "variable": np.array(list(fields)),
        "lat": SMALL_LAT,
        "lon": SMALL_LON,
    }
    if ensemble is not None:
        coords["ensemble"] = np.asarray(ensemble, dtype=np.int64)
    io.add_array(coord_array(dims, coords))
    data = np.stack([fields[v] for v in fields], axis=-3).astype("float32")
    io.write(xr.DataArray(data, dims=dims, coords=coords))
    io.close()
    return path


def write_verification_store(
    cfg, fields: dict[str, np.ndarray], times=VALID_TIMES
) -> str:
    """Write ``verification.zarr`` (one array per variable on ``(time, lat,
    lon)``) as predownload does, through the library ``ZarrBackend``."""
    path = store_path(cfg, "verification.zarr")
    os.makedirs(cfg.output.path, exist_ok=True)
    io = ZarrBackend(path)
    dims = ["time", "variable", "lat", "lon"]
    coords = {
        "time": np.asarray(times),
        "variable": np.array(list(fields)),
        "lat": SMALL_LAT,
        "lon": SMALL_LON,
    }
    io.add_array(coord_array(dims, coords))
    data = np.stack([fields[v] for v in fields], axis=1).astype("float32")
    io.write(xr.DataArray(data, dims=dims, coords=coords))
    io.close()
    return path


def analytic_forecast(ensemble=None) -> dict[str, np.ndarray]:
    shape = (len(TIMES), len(LEAD_TIMES), len(SMALL_LAT), len(SMALL_LON))
    t2m = np.full(shape, T2M_FORECAST)
    z500 = np.broadcast_to(Z500_PROFILE[:, None], shape[-2:]) * np.ones(shape)
    if ensemble is None:
        return {"t2m": t2m, "z500": z500}
    # Member m sits at truth + 3 (m - 1): ensemble mean equals the truth and
    # the (N-1) ensemble variance is 9 everywhere.
    truth_t2m = analytic_verification_on_forecast_grid()
    offsets = 3.0 * (np.asarray(ensemble, dtype=float) - 1.0)
    t2m_ens = truth_t2m[None] + offsets[:, None, None, None, None]
    z500_ens = np.zeros_like(t2m_ens) + offsets[:, None, None, None, None]
    return {"t2m": t2m_ens, "z500": z500_ens}


def analytic_verification() -> dict[str, np.ndarray]:
    shape = (len(VALID_TIMES), len(SMALL_LAT), len(SMALL_LON))
    t2m = _t2m_truth(VALID_TIMES)[:, None, None] * np.ones(shape)
    return {"t2m": t2m, "z500": np.zeros(shape)}


def analytic_verification_on_forecast_grid() -> np.ndarray:
    """t2m truth laid out as ``(time, lead_time, lat, lon)``."""
    valid = TIMES[:, None] + LEAD_TIMES[None, :]
    return _t2m_truth(valid)[:, :, None, None] * np.ones(
        (len(TIMES), len(LEAD_TIMES), len(SMALL_LAT), len(SMALL_LON))
    )


def scoring_cfg(tmp_path, metrics: dict | None = None, **scoring_overrides):
    cfg = compose_cfg(tmp_path, ["nsteps=2", "ensemble_size=1"])
    with open_dict(cfg):
        cfg.start_times = [
            str(t).replace("T", " ") for t in TIMES.astype("datetime64[s]")
        ]
        cfg.output.variables = list(VARIABLES)
        cfg.scoring.lat_weights = False
        cfg.scoring.lead_time_chunk_size = None
        cfg.scoring.metrics = metrics or {
            "rmse": {
                "_target_": "earth2studio.statistics.rmse",
                "reduction_dimensions": ["lat", "lon"],
            }
        }
        for key, value in scoring_overrides.items():
            OmegaConf.update(cfg, f"scoring.{key}", value, merge=False)
    return cfg


@pytest.fixture()
def det_cfg(tmp_path):
    """Deterministic forecast and verification stores plus a scoring config."""
    cfg = scoring_cfg(tmp_path)
    write_forecast_store(cfg, analytic_forecast())
    write_verification_store(cfg, analytic_verification())
    return cfg


def open_scores(cfg) -> xr.Dataset:
    path = os.path.join(cfg.output.path, cfg.scoring.output.store_name)
    return xr.open_zarr(path, consolidated=False)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestBuildLeadTimeChunks:
    @pytest.mark.parametrize("size", [None, 0, -1, 3, 10])
    def test_no_chunking(self, size):
        chunks = build_lead_time_chunks(LEAD_TIMES, size)
        assert len(chunks) == 1
        np.testing.assert_array_equal(chunks[0], LEAD_TIMES)

    def test_chunks_cover_all_lead_times(self):
        chunks = build_lead_time_chunks(LEAD_TIMES, 2)
        assert [len(c) for c in chunks] == [2, 1]
        np.testing.assert_array_equal(np.concatenate(chunks), LEAD_TIMES)


class TestValidateLeadTimeChunking:
    def test_no_chunking_always_passes(self):
        validate_lead_time_chunking({"m": mse(["lead_time", "lat", "lon"])}, None, 3)
        validate_lead_time_chunking({"m": mse(["lead_time", "lat", "lon"])}, 5, 3)

    def test_lead_time_reduction_raises(self):
        with pytest.raises(ValueError, match="lead_time"):
            validate_lead_time_chunking({"m": mse(["lead_time", "lat", "lon"])}, 2, 3)

    def test_spatial_reduction_passes(self):
        validate_lead_time_chunking({"m": mse(["lat", "lon"])}, 2, 3)


def _template(ensemble: bool = False):
    dims = (["ensemble"] if ensemble else []) + ["lead_time", "variable", "lat", "lon"]
    coords = {
        "lead_time": LEAD_TIMES,
        "variable": np.array(VARIABLES),
        "lat": SMALL_LAT,
        "lon": SMALL_LON,
    }
    if ensemble:
        coords["ensemble"] = np.arange(3)
    return coord_array(dims, coords)


class TestScoreNames:
    def test_with_variable_dim(self):
        out = rmse(["lat", "lon"]).output_coords(_template())
        assert score_variable_names("rmse", out) == ["rmse__t2m", "rmse__z500"]

    def test_without_variable_dim(self):
        out = rmse(["variable", "lat", "lon"]).output_coords(_template())
        assert score_variable_names("rmse", out) == ["rmse"]

    def test_all_names(self):
        metrics = OrderedDict(rmse=rmse(["lat", "lon"]), mse=mse(["lat", "lon"]))
        assert all_score_variable_names(metrics, _template()) == [
            "rmse__t2m",
            "rmse__z500",
            "mse__t2m",
            "mse__z500",
        ]


class TestGroupScoreArraysByDims:
    def test_single_group(self):
        metrics = OrderedDict(rmse=rmse(["lat", "lon"]), mse=mse(["lat", "lon"]))
        groups = group_score_arrays_by_dims(metrics, _template(), TIMES)
        assert len(groups) == 1
        coords, names = groups[0]
        assert list(coords) == ["time", "lead_time"]
        np.testing.assert_array_equal(coords["time"], TIMES)
        assert names == ["rmse__t2m", "rmse__z500", "mse__t2m", "mse__z500"]

    def test_mixed_dims_make_separate_groups(self):
        metrics = OrderedDict(
            member_mse=mse(["lat", "lon"]),
            ensemble_mean_mse=mse(["lat", "lon"], ensemble_dimension="ensemble"),
            ensemble_variance=ensemble_variance("ensemble", ["lat", "lon"]),
        )
        groups = group_score_arrays_by_dims(metrics, _template(ensemble=True), TIMES)
        keys = [tuple(coords) for coords, _ in groups]
        assert keys == [("time", "ensemble", "lead_time"), ("time", "lead_time")]
        assert groups[1][1] == [
            "ensemble_mean_mse__t2m",
            "ensemble_mean_mse__z500",
            "ensemble_variance__t2m",
            "ensemble_variance__z500",
        ]


class TestConcatChunks:
    def test_single_chunk_passthrough(self):
        da = xr.DataArray(
            np.zeros(3), dims=["lead_time"], coords={"lead_time": LEAD_TIMES}
        )
        assert _concat_chunks([da], LEAD_TIMES) is da

    def test_two_chunks(self):
        a = xr.DataArray(
            np.array([[1.0, 2.0]]).T,
            dims=["lead_time", "variable"],
            coords={"lead_time": LEAD_TIMES[:2], "variable": ["t2m"]},
        )
        b = xr.DataArray(
            np.array([[3.0]]),
            dims=["lead_time", "variable"],
            coords={"lead_time": LEAD_TIMES[2:], "variable": ["t2m"]},
        )
        out = _concat_chunks([a, b], LEAD_TIMES)
        np.testing.assert_array_equal(out.values[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(out["lead_time"].values, LEAD_TIMES)


# ---------------------------------------------------------------------------
# Store introspection and data loading
# ---------------------------------------------------------------------------


class TestStoreIntrospection:
    def test_spatial_coords(self, det_cfg):
        ds = xr.open_zarr(store_path(det_cfg), consolidated=False)
        coords = spatial_coords_from_dataset(ds)
        assert list(coords) == ["lat", "lon"]
        np.testing.assert_array_equal(coords["lat"], SMALL_LAT)

    def test_input_template_deterministic(self, det_cfg):
        ds = xr.open_zarr(store_path(det_cfg), consolidated=False)
        template = build_input_coords_template(ds, LEAD_TIMES, VARIABLES)
        assert tuple(template.dims) == ("lead_time", "variable", "lat", "lon")
        assert template.shape == (3, 2, 4, 8)

    def test_input_template_ensemble(self, tmp_path):
        cfg = scoring_cfg(tmp_path)
        write_forecast_store(
            cfg, analytic_forecast(ensemble=range(3)), ensemble=range(3)
        )
        ds = xr.open_zarr(store_path(cfg), consolidated=False)
        template = build_input_coords_template(ds, LEAD_TIMES, VARIABLES)
        assert tuple(template.dims) == (
            "ensemble",
            "lead_time",
            "variable",
            "lat",
            "lon",
        )
        np.testing.assert_array_equal(template["ensemble"].values, [0, 1, 2])


class TestLoadChunks:
    def test_prediction_chunk(self, det_cfg):
        ds = xr.open_zarr(store_path(det_cfg), consolidated=False)
        x = load_prediction_chunk(ds, TIMES[1], LEAD_TIMES[:2], VARIABLES, "cpu")
        assert tuple(x.dims) == ("lead_time", "variable", "lat", "lon")
        assert x.shape == (2, 2, 4, 8)
        assert x.dtype == np.float32
        np.testing.assert_allclose(x.sel(variable="t2m").values, T2M_FORECAST)
        np.testing.assert_allclose(
            x.sel(variable="z500").isel(lead_time=0, lon=0).values, Z500_PROFILE
        )

    def test_verification_chunk_is_at_valid_times(self, det_cfg):
        source = PredownloadedSource(store_path(det_cfg, "verification.zarr"))
        coords = OrderedDict(lat=SMALL_LAT, lon=SMALL_LON)
        y = load_verification_chunk(
            source, TIMES[1], LEAD_TIMES, VARIABLES, coords, "cpu"
        )
        assert tuple(y.dims) == ("lead_time", "variable", "lat", "lon")
        np.testing.assert_array_equal(y["lead_time"].values, LEAD_TIMES)
        # TIMES[1] = 12 h after TIMES[0]: valid times 12, 18, 24 h -> 2, 3, 4.
        np.testing.assert_allclose(
            y.sel(variable="t2m").mean(["lat", "lon"]).values, [2.0, 3.0, 4.0]
        )

    def test_verification_grid_mismatch_raises(self, det_cfg):
        source = PredownloadedSource(store_path(det_cfg, "verification.zarr"))
        coords = OrderedDict(lat=SMALL_LAT[:2], lon=SMALL_LON)
        with pytest.raises(ValueError, match="grid"):
            load_verification_chunk(
                source, TIMES[0], LEAD_TIMES, VARIABLES, coords, "cpu"
            )

    def test_missing_variable_is_explained(self, det_cfg):
        source = PredownloadedSource(store_path(det_cfg, "verification.zarr"))
        coords = OrderedDict(lat=SMALL_LAT, lon=SMALL_LON)
        with pytest.raises(KeyError, match="verification source"):
            load_verification_chunk(
                source, TIMES[0], LEAD_TIMES, ["nope"], coords, "cpu"
            )


# ---------------------------------------------------------------------------
# Metric instantiation
# ---------------------------------------------------------------------------


class TestInstantiateMetrics:
    def _cfg(self, metrics, lat_weights=False, regions=None, events=None):
        return OmegaConf.create(
            {
                "scoring": {
                    "metrics": metrics,
                    "lat_weights": lat_weights,
                    "regions": regions,
                    "events": events,
                }
            }
        )

    def test_basic(self):
        cfg = self._cfg(
            {
                "rmse": {
                    "_target_": "earth2studio.statistics.rmse",
                    "reduction_dimensions": ["lat", "lon"],
                },
                "mse": {
                    "_target_": "src.metrics.mse",
                    "reduction_dimensions": ["lat", "lon"],
                },
            }
        )
        metrics = instantiate_metrics(cfg, OrderedDict(lat=SMALL_LAT, lon=SMALL_LON))
        assert list(metrics) == ["rmse", "mse"]
        assert isinstance(metrics["rmse"], rmse)
        assert isinstance(metrics["mse"], mse)
        assert metrics["rmse"].weights is None

    def test_lat_weights_injected(self):
        cfg = self._cfg(
            {
                "rmse": {
                    "_target_": "earth2studio.statistics.rmse",
                    "reduction_dimensions": ["lat", "lon"],
                }
            },
            lat_weights=True,
        )
        metrics = instantiate_metrics(cfg, OrderedDict(lat=SMALL_LAT, lon=SMALL_LON))
        weights = metrics["rmse"].weights
        assert tuple(weights.shape) == (len(SMALL_LAT), len(SMALL_LON))
        # Equator-ward rows weigh more than the poles.
        assert weights[1, 0] > weights[0, 0]

    def test_lat_weights_skipped_without_lat_reduction(self):
        cfg = self._cfg(
            {
                "rmse": {
                    "_target_": "earth2studio.statistics.rmse",
                    "reduction_dimensions": ["lon"],
                }
            },
            lat_weights=True,
        )
        metrics = instantiate_metrics(cfg, OrderedDict(lat=SMALL_LAT, lon=SMALL_LON))
        assert metrics["rmse"].weights is None

    def test_accepts_signature(self):
        cfg = self._cfg(
            {
                "rmse": {
                    "_target_": "earth2studio.statistics.rmse",
                    "reduction_dimensions": ["lat", "lon"],
                }
            },
            lat_weights=True,
        )
        metrics = instantiate_metrics(cfg, _template())
        assert tuple(metrics["rmse"].weights.shape) == (len(SMALL_LAT), len(SMALL_LON))

    def test_regions_wrap_metric(self):
        cfg = self._cfg(
            {
                "rmse": {
                    "_target_": "earth2studio.statistics.rmse",
                    "reduction_dimensions": ["lat", "lon"],
                }
            },
            regions={"global": None, "north": {"lat": [0, 90]}},
        )
        metrics = instantiate_metrics(cfg, OrderedDict(lat=SMALL_LAT, lon=SMALL_LON))
        regional = metrics["rmse"]
        assert isinstance(regional, RegionalMetric)
        assert regional.regions == ["global", "north"]
        assert not _is_statistic(regional)
        out = regional.output_coords(_template())
        assert tuple(out.dims) == ("region", "lead_time", "variable")
        assert list(out["region"].values) == ["global", "north"]

    def test_regional_values(self):
        cfg = self._cfg(
            {
                "rmse": {
                    "_target_": "earth2studio.statistics.rmse",
                    "reduction_dimensions": ["lat", "lon"],
                }
            },
            regions={"global": None, "north": {"lat": [0, 90]}},
        )
        metrics = instantiate_metrics(cfg, OrderedDict(lat=SMALL_LAT, lon=SMALL_LON))
        x = xr.DataArray(
            np.broadcast_to(Z500_PROFILE[:, None], (3, 1, 4, 8)).copy(),
            dims=["lead_time", "variable", "lat", "lon"],
            coords={
                "lead_time": LEAD_TIMES,
                "variable": ["z500"],
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            },
        )
        out = metrics["rmse"](x, xr.zeros_like(x))
        assert tuple(out.dims) == ("region", "lead_time", "variable")
        np.testing.assert_allclose(
            out.sel(region="global").values, Z500_RMSE_GLOBAL, rtol=1e-5
        )
        np.testing.assert_allclose(
            out.sel(region="north").values, Z500_RMSE_NORTH, rtol=1e-5
        )

    def test_event_box_becomes_region(self):
        cfg = self._cfg(
            {
                "rmse": {
                    "_target_": "earth2studio.statistics.rmse",
                    "reduction_dimensions": ["lat", "lon"],
                }
            },
            events={
                "storm": {
                    "start": "2024-01-01 00:00:00",
                    "end": "2024-01-02 00:00:00",
                    "region": {"lat": [0, 90]},
                }
            },
        )
        metrics = instantiate_metrics(cfg, OrderedDict(lat=SMALL_LAT, lon=SMALL_LON))
        assert metrics["rmse"].regions == ["global", "storm"]

    def test_partial_spatial_reduction_stays_global(self):
        cfg = self._cfg(
            {
                "rmse": {
                    "_target_": "earth2studio.statistics.rmse",
                    "reduction_dimensions": ["lon"],
                }
            },
            regions={"north": {"lat": [0, 90]}},
        )
        metrics = instantiate_metrics(cfg, OrderedDict(lat=SMALL_LAT, lon=SMALL_LON))
        assert isinstance(metrics["rmse"], rmse)

    def test_regional_statistic(self):
        cfg = self._cfg(
            {
                "ensemble_variance": {
                    "_target_": "src.metrics.ensemble_variance",
                    "ensemble_dimension": "ensemble",
                    "reduction_dimensions": ["lat", "lon"],
                }
            },
            regions={"global": None, "north": {"lat": [0, 90]}},
        )
        metrics = instantiate_metrics(cfg, OrderedDict(lat=SMALL_LAT, lon=SMALL_LON))
        regional = metrics["ensemble_variance"]
        assert _is_statistic(regional)
        out = regional.output_coords(_template(ensemble=True))
        assert tuple(out.dims) == ("region", "lead_time", "variable")


# ---------------------------------------------------------------------------
# Verification source resolution
# ---------------------------------------------------------------------------


class TestOpenVerificationSource:
    def test_predownloaded_store(self, det_cfg):
        source = open_verification_source(det_cfg)
        assert isinstance(source, PredownloadedSource)
        assert source.store_path == store_path(det_cfg, "verification.zarr")

    def test_byo_wins(self, det_cfg):
        other = store_path(det_cfg, "other.zarr")
        with open_dict(det_cfg):
            det_cfg.verification_source = {
                "_target_": "src.data.PredownloadedSource",
                "store_path": store_path(det_cfg, "verification.zarr"),
            }
        source = open_verification_source(det_cfg)
        assert isinstance(source, PredownloadedSource)
        assert source.store_path != other

    def test_live_fallback(self, tmp_path):
        cfg = scoring_cfg(tmp_path)
        with open_dict(cfg):
            cfg.data_source = {
                "_target_": "earth2studio.data.Random",
                "domain_coords": {"lat": SMALL_LAT.tolist(), "lon": SMALL_LON.tolist()},
            }
        from earth2studio.data import Random

        assert isinstance(open_verification_source(cfg), Random)

    def test_nothing_configured_raises(self, tmp_path):
        cfg = scoring_cfg(tmp_path)
        with open_dict(cfg):
            cfg.data_source = None
        with pytest.raises(ValueError, match="verification"):
            open_verification_source(cfg)


# ---------------------------------------------------------------------------
# Score store
# ---------------------------------------------------------------------------


class TestScoreStore:
    def test_add_score_arrays_is_idempotent(self, tmp_path):
        io = ZarrBackend(str(tmp_path / "scores.zarr"))
        groups = group_score_arrays_by_dims(
            OrderedDict(rmse=rmse(["lat", "lon"])), _template(), TIMES
        )
        add_score_arrays(io, groups)
        add_score_arrays(io, groups)
        assert "rmse__t2m" in io and "rmse__z500" in io
        assert io["rmse__t2m"].shape == (len(TIMES), len(LEAD_TIMES))

    def test_open_score_store_stamps_events_and_chunks(self, det_cfg):
        groups = group_score_arrays_by_dims(
            OrderedDict(rmse=rmse(["lat", "lon"])), _template(), TIMES
        )
        io = open_score_store(
            det_cfg, groups, rank=0, events={"storm": {"label": "Storm"}}
        )
        assert io.root.attrs["events"] == {"storm": {"label": "Storm"}}
        assert io["rmse__t2m"].chunks[0] == 1  # scoring.output.chunks: {time: 1}


# ---------------------------------------------------------------------------
# End to end through score.run
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_deterministic(self, det_cfg):
        run_score(det_cfg, "cpu")
        ds = open_scores(det_cfg)
        assert set(ds.data_vars) == {"rmse__t2m", "rmse__z500"}
        assert ds["rmse__t2m"].dims == ("time", "lead_time")
        np.testing.assert_array_equal(ds.time.values, TIMES)
        np.testing.assert_array_equal(ds.lead_time.values, LEAD_TIMES)
        truth = _t2m_truth(TIMES[:, None] + LEAD_TIMES[None, :])
        np.testing.assert_allclose(
            ds["rmse__t2m"].values, np.abs(T2M_FORECAST - truth), rtol=1e-5
        )
        np.testing.assert_allclose(ds["rmse__z500"].values, Z500_RMSE_GLOBAL, rtol=1e-5)
        assert not ds["rmse__t2m"].isnull().any()

    def test_lat_weights(self, det_cfg):
        det_cfg.scoring.lat_weights = True
        run_score(det_cfg, "cpu")
        ds = open_scores(det_cfg)
        np.testing.assert_allclose(ds["rmse__z500"].values, Z500_RMSE_LATW, rtol=1e-4)

    def test_chunking_matches_unchunked(self, det_cfg):
        run_score(det_cfg, "cpu")
        unchunked = open_scores(det_cfg).load()
        det_cfg.scoring.lead_time_chunk_size = 2
        det_cfg.scoring.output.store_name = "scores_chunked.zarr"
        run_score(det_cfg, "cpu")
        chunked = open_scores(det_cfg).load()
        xr.testing.assert_allclose(unchunked, chunked)

    def test_regions_and_events(self, tmp_path):
        cfg = scoring_cfg(
            tmp_path,
            regions={"global": None, "north": {"lat": [0, 90]}},
            events={
                "storm": {
                    "label": "Storm",
                    "start": "2024-01-01 06:00:00",
                    "end": "2024-01-01 12:00:00",
                    "region": {"lat": [0, 90]},
                }
            },
        )
        write_forecast_store(cfg, analytic_forecast())
        write_verification_store(cfg, analytic_verification())
        run_score(cfg, "cpu")
        ds = open_scores(cfg)
        assert ds["rmse__z500"].dims == ("time", "region", "lead_time")
        assert list(ds.region.values) == ["global", "north", "storm"]
        np.testing.assert_allclose(
            ds["rmse__z500"].sel(region="global").values, Z500_RMSE_GLOBAL, rtol=1e-5
        )
        np.testing.assert_allclose(
            ds["rmse__z500"].sel(region="north").values, Z500_RMSE_NORTH, rtol=1e-5
        )
        np.testing.assert_allclose(
            ds["rmse__z500"].sel(region="storm").values, Z500_RMSE_NORTH, rtol=1e-5
        )
        assert "storm" in ds.attrs["events"]
        assert ds.attrs["events"]["storm"]["label"] == "Storm"

    def test_ensemble(self, tmp_path):
        members = range(3)
        cfg = scoring_cfg(
            tmp_path,
            metrics={
                "mse": {
                    "_target_": "src.metrics.mse",
                    "reduction_dimensions": ["lat", "lon"],
                },
                "ensemble_mean_mse": {
                    "_target_": "src.metrics.mse",
                    "ensemble_dimension": "ensemble",
                    "reduction_dimensions": ["lat", "lon"],
                },
                "ensemble_variance": {
                    "_target_": "src.metrics.ensemble_variance",
                    "ensemble_dimension": "ensemble",
                    "reduction_dimensions": ["lat", "lon"],
                },
            },
        )
        write_forecast_store(cfg, analytic_forecast(ensemble=members), ensemble=members)
        write_verification_store(cfg, analytic_verification())
        run_score(cfg, "cpu")
        ds = open_scores(cfg)
        assert ds["mse__t2m"].dims == ("time", "ensemble", "lead_time")
        assert ds["ensemble_mean_mse__t2m"].dims == ("time", "lead_time")
        assert ds["ensemble_variance__t2m"].dims == ("time", "lead_time")
        np.testing.assert_allclose(ds["ensemble_mean_mse__t2m"].values, 0.0, atol=1e-6)
        np.testing.assert_allclose(ds["ensemble_variance__t2m"].values, 9.0, rtol=1e-5)
        np.testing.assert_allclose(ds["ensemble_variance__z500"].values, 9.0, rtol=1e-5)
        per_member = ds["mse__t2m"].transpose("ensemble", "time", "lead_time").values
        np.testing.assert_allclose(per_member[0], 9.0, rtol=1e-5)
        np.testing.assert_allclose(per_member[1], 0.0, atol=1e-6)
        np.testing.assert_allclose(per_member[2], 9.0, rtol=1e-5)

    def test_resume_skips_scored_times(self, det_cfg):
        run_score(det_cfg, "cpu")
        progress = ProgressStore(
            scoring_progress_path(det_cfg),
            tags={"project": str(det_cfg.project), "run_id": str(det_cfg.run_id)},
        )
        progress.open(
            {"store": "scores.zarr", "metrics": ["rmse"], "variables": list(VARIABLES)}
        )
        items = work_items(list(TIMES), np.timedelta64(LEAD_TIMES[-1], "s"))
        assert progress.remaining(items) == []
        progress.close()

        # Change the truth; a resumed run must leave the recorded scores alone.
        shifted = analytic_verification()
        shifted["t2m"] = shifted["t2m"] + 50.0
        write_verification_store(det_cfg, shifted)
        before = open_scores(det_cfg)["rmse__t2m"].load()
        det_cfg.scoring.resume = True
        run_score(det_cfg, "cpu")
        after = open_scores(det_cfg)["rmse__t2m"].load()
        xr.testing.assert_allclose(before, after)

        # Without resume the store is rebuilt from the new truth.
        det_cfg.scoring.resume = False
        run_score(det_cfg, "cpu")
        rebuilt = open_scores(det_cfg)["rmse__t2m"].load()
        assert not np.allclose(before.values, rebuilt.values)

    def test_missing_initial_time_raises(self, det_cfg):
        with open_dict(det_cfg):
            det_cfg.start_times = det_cfg.start_times + ["2024-02-01 00:00:00"]
        with pytest.raises(ValueError, match="not in the prediction store"):
            run_score(det_cfg, "cpu")

    def test_missing_forecast_store_raises(self, tmp_path):
        cfg = scoring_cfg(tmp_path)
        with pytest.raises(FileNotFoundError, match="main.py"):
            run_score(cfg, "cpu")


# ---------------------------------------------------------------------------
# NaN policy and valid ranges
# ---------------------------------------------------------------------------


class TestNanPolicy:
    def _stores(self, cfg):
        truth = analytic_verification()
        truth["z500"] = truth["z500"].copy()
        truth["z500"][:, 0, :] = np.nan  # the polar row is invalid
        write_forecast_store(cfg, analytic_forecast())
        write_verification_store(cfg, truth)

    def test_propagate_default_yields_nan(self, tmp_path):
        cfg = scoring_cfg(tmp_path)
        self._stores(cfg)
        run_score(cfg, "cpu")
        ds = open_scores(cfg)
        assert ds["rmse__z500"].isnull().all()
        assert not ds["rmse__t2m"].isnull().any()

    def test_zero_fill_is_finite(self, tmp_path):
        cfg = scoring_cfg(tmp_path, nan_policy="zero_fill")
        self._stores(cfg)
        run_score(cfg, "cpu")
        ds = open_scores(cfg)
        # Forecast minus zero-filled truth is just the forecast profile.
        np.testing.assert_allclose(ds["rmse__z500"].values, Z500_RMSE_GLOBAL, rtol=1e-5)

    def test_invalid_policy_raises(self, tmp_path):
        cfg = scoring_cfg(tmp_path, nan_policy="drop")
        self._stores(cfg)
        with pytest.raises(ValueError, match="nan_policy"):
            run_score(cfg, "cpu")


def _two_var_field():
    return xr.DataArray(
        np.array([[-5.0, 0.0, 5.0], [-5.0, 0.0, 5.0]]),
        dims=["variable", "lon"],
        coords={"variable": ["refc", "t2m"], "lon": [0.0, 1.0, 2.0]},
    )


class TestApplyValidRanges:
    def test_clamps_named_variable_only(self):
        out = apply_valid_ranges(_two_var_field(), {"refc": {"min": 0.0, "max": 3.0}})
        np.testing.assert_array_equal(out.sel(variable="refc").values, [0.0, 0.0, 3.0])
        np.testing.assert_array_equal(out.sel(variable="t2m").values, [-5.0, 0.0, 5.0])

    def test_input_is_not_modified(self):
        da = _two_var_field()
        apply_valid_ranges(da, {"refc": {"min": 0.0}})
        np.testing.assert_array_equal(da.sel(variable="refc").values, [-5.0, 0.0, 5.0])

    def test_one_sided_and_noop(self):
        out = apply_valid_ranges(_two_var_field(), {"refc": {"max": None, "min": 0.0}})
        np.testing.assert_array_equal(out.sel(variable="refc").values, [0.0, 0.0, 5.0])
        out = apply_valid_ranges(_two_var_field(), {"refc": {"min": None, "max": None}})
        np.testing.assert_array_equal(out.sel(variable="refc").values, [-5.0, 0.0, 5.0])

    def test_unknown_variable_skipped(self):
        out = apply_valid_ranges(_two_var_field(), {"other": {"min": 0.0}})
        xr.testing.assert_equal(out, _two_var_field())

    def test_tensor_helper(self):
        x = torch.tensor([[-5.0, 0.0, 5.0], [-5.0, 0.0, 5.0]])
        coords = OrderedDict(variable=np.array(["refc", "t2m"]), lon=np.arange(3))
        out = _apply_valid_ranges(x, coords, {"refc": {"min": 0.0, "max": 3.0}})
        assert out.tolist() == [[0.0, 0.0, 3.0], [-5.0, 0.0, 5.0]]


class TestValidRangesEndToEnd:
    def _stores(self, cfg, fill):
        truth = analytic_verification()
        truth["z500"] = truth["z500"].copy()
        truth["z500"][:, -1, :] = fill  # a no-data fill in the last row
        write_forecast_store(cfg, analytic_forecast())
        write_verification_store(cfg, truth)

    def test_fill_inflates_rmse_without_ranges(self, tmp_path):
        cfg = scoring_cfg(tmp_path)
        self._stores(cfg, -9999.0)
        run_score(cfg, "cpu")
        assert (open_scores(cfg)["rmse__z500"].values > 100).all()

    def test_fill_is_clamped_with_ranges(self, tmp_path):
        cfg = scoring_cfg(tmp_path, valid_ranges={"z500": {"min": 0.0, "max": 10.0}})
        self._stores(cfg, -9999.0)
        run_score(cfg, "cpu")
        # The forecast profile is 0 in the last row, so the clamped fill
        # (0) matches it and the score is the untouched global value.
        np.testing.assert_allclose(
            open_scores(cfg)["rmse__z500"].values, Z500_RMSE_GLOBAL, rtol=1e-5
        )

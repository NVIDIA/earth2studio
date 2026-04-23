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

"""Tests for the scoring module (src/scoring.py)."""

from __future__ import annotations

import os
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf
from src.data import PredownloadedSource
from src.output import OutputManager, build_score_coords
from src.scoring import (
    _concat_chunks,
    add_score_arrays,
    all_score_variable_names,
    build_input_coords_template,
    build_lead_time_chunks,
    build_superset_score_coords,
    group_score_arrays_by_dims,
    instantiate_metrics,
    load_prediction_chunk,
    load_verification_chunk,
    open_verification_source,
    run_scoring,
    score_variable_names,
    spatial_coords_from_dataset,
    validate_lead_time_chunking,
)
from src.work import (
    clear_scoring_progress,
    filter_scoring_completed,
    scoring_progress_dir,
    write_scoring_marker,
)

# ---------------------------------------------------------------------------
# Shared test constants and helpers
# ---------------------------------------------------------------------------

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
VARIABLES = ["t2m", "z500"]
TIMES = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
LEAD_TIMES = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")


def _create_prediction_zarr(
    path, times, lead_times, variables, lat, lon, ensemble_size=1
):
    """Write a minimal prediction zarr store."""
    rng = np.random.default_rng(42)
    ds = xr.Dataset()
    dims = []
    coords = {}

    if ensemble_size > 1:
        dims.append("ensemble")
        coords["ensemble"] = np.arange(ensemble_size)

    dims.extend(["time", "lead_time"])
    coords["time"] = times
    coords["lead_time"] = lead_times

    spatial_dims = ["lat", "lon"]
    dims.extend(spatial_dims)
    coords["lat"] = lat
    coords["lon"] = lon

    shape = [len(coords[d]) for d in dims]

    for var in variables:
        ds[var] = xr.DataArray(
            rng.standard_normal(shape).astype("float32"),
            dims=dims,
            coords=coords,
        )
    ds.to_zarr(str(path))
    return ds


def _create_verification_zarr(path, times, variables, lat, lon):
    """Write a minimal verification zarr store (flat time dimension)."""
    rng = np.random.default_rng(123)
    ds = xr.Dataset()
    for var in variables:
        ds[var] = xr.DataArray(
            rng.standard_normal((len(times), len(lat), len(lon))).astype("float32"),
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": lat, "lon": lon},
        )
    ds.to_zarr(str(path))
    return ds


def _make_dist_mock(*, rank=0, world_size=1, distributed=False):
    class _FakeDist:
        def __init__(self):
            self.rank = rank
            self.world_size = world_size
            self.distributed = distributed
            self.device = torch.device("cpu")

    return _FakeDist()


# Paths to mock
_DIST_PATH = "src.output.DistributedManager"
_RANK0_PATH = "src.output.run_on_rank0_first"
_SCORING_DIST_PATH = "src.scoring.DistributedManager"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def prediction_path(tmp_path):
    return tmp_path / "out" / "forecast.zarr"


@pytest.fixture()
def verification_path(tmp_path):
    return tmp_path / "out" / "data.zarr"


@pytest.fixture()
def verification_times():
    """All valid times needed for verification (IC times + lead offsets)."""
    all_valid = set()
    for t in TIMES:
        for lt in LEAD_TIMES:
            all_valid.add(t + lt)
    return np.array(sorted(all_valid), dtype="datetime64[ns]")


@pytest.fixture()
def prediction_ds(prediction_path):
    _create_prediction_zarr(
        prediction_path, TIMES, LEAD_TIMES, VARIABLES, SMALL_LAT, SMALL_LON
    )
    return xr.open_zarr(str(prediction_path))


@pytest.fixture()
def prediction_ds_ensemble(prediction_path):
    _create_prediction_zarr(
        prediction_path,
        TIMES,
        LEAD_TIMES,
        VARIABLES,
        SMALL_LAT,
        SMALL_LON,
        ensemble_size=3,
    )
    return xr.open_zarr(str(prediction_path))


@pytest.fixture()
def verif_source(verification_path, verification_times):
    _create_verification_zarr(
        verification_path, verification_times, VARIABLES, SMALL_LAT, SMALL_LON
    )
    return PredownloadedSource(str(verification_path))


@pytest.fixture()
def spatial_coords():
    return OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})


@pytest.fixture()
def input_template():
    """Input coords template for a deterministic prediction."""
    return OrderedDict(
        {
            "lead_time": LEAD_TIMES,
            "variable": np.array(VARIABLES),
            "lat": SMALL_LAT,
            "lon": SMALL_LON,
        }
    )


@pytest.fixture()
def input_template_ensemble():
    """Input coords template with ensemble dimension."""
    return OrderedDict(
        {
            "ensemble": np.arange(3),
            "lead_time": LEAD_TIMES,
            "variable": np.array(VARIABLES),
            "lat": SMALL_LAT,
            "lon": SMALL_LON,
        }
    )


# ---------------------------------------------------------------------------
# build_lead_time_chunks
# ---------------------------------------------------------------------------


class TestBuildLeadTimeChunks:
    def test_no_chunking_none(self):
        result = build_lead_time_chunks(LEAD_TIMES, None)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], LEAD_TIMES)

    def test_no_chunking_zero(self):
        result = build_lead_time_chunks(LEAD_TIMES, 0)
        assert len(result) == 1

    def test_no_chunking_large(self):
        result = build_lead_time_chunks(LEAD_TIMES, 100)
        assert len(result) == 1

    def test_chunk_size_1(self):
        result = build_lead_time_chunks(LEAD_TIMES, 1)
        assert len(result) == 3
        for chunk in result:
            assert len(chunk) == 1

    def test_chunk_size_2(self):
        result = build_lead_time_chunks(LEAD_TIMES, 2)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1

    def test_chunks_cover_all_lead_times(self):
        result = build_lead_time_chunks(LEAD_TIMES, 2)
        concatenated = np.concatenate(result)
        np.testing.assert_array_equal(concatenated, LEAD_TIMES)


# ---------------------------------------------------------------------------
# score_variable_names
# ---------------------------------------------------------------------------


class TestScoreVariableNames:
    def test_with_variable_dim(self):
        coords = OrderedDict(
            {
                "lead_time": LEAD_TIMES,
                "variable": np.array(["t2m", "z500"]),
            }
        )
        names = score_variable_names("rmse", coords)
        assert names == ["rmse__t2m", "rmse__z500"]

    def test_without_variable_dim(self):
        coords = OrderedDict({"lead_time": LEAD_TIMES})
        names = score_variable_names("energy_score", coords)
        assert names == ["energy_score"]


class TestAllScoreVariableNames:
    def test_multiple_metrics(self, input_template):
        from earth2studio.statistics import mae, rmse

        metrics = OrderedDict(
            {
                "rmse": rmse(reduction_dimensions=["lat", "lon"]),
                "mae": mae(reduction_dimensions=["lat", "lon"]),
            }
        )
        names = all_score_variable_names(metrics, input_template)
        assert names == ["rmse__t2m", "rmse__z500", "mae__t2m", "mae__z500"]


# ---------------------------------------------------------------------------
# validate_metric_coords
# ---------------------------------------------------------------------------


class TestBuildSupersetScoreCoords:
    def test_single_metric(self, input_template):
        from earth2studio.statistics import rmse

        metrics = OrderedDict(
            {
                "rmse": rmse(reduction_dimensions=["lat", "lon"]),
            }
        )
        coords = build_superset_score_coords(metrics, input_template, TIMES)
        assert "time" in coords
        assert "lead_time" in coords
        assert "lat" not in coords
        assert "lon" not in coords

    def test_mixed_metrics_superset(self, input_template_ensemble):
        """Metrics with different output dims produce a union of dims."""
        from earth2studio.statistics import crps, rmse

        metrics = OrderedDict(
            {
                "rmse": rmse(reduction_dimensions=["lat", "lon"]),
                "crps": crps(
                    reduction_dimensions=["lat", "lon"],
                    ensemble_dimension="ensemble",
                ),
            }
        )
        coords = build_superset_score_coords(metrics, input_template_ensemble, TIMES)
        # RMSE preserves ensemble; CRPS does not. Superset has both.
        assert "time" in coords
        assert "ensemble" in coords
        assert "lead_time" in coords


class TestGroupScoreArraysByDims:
    def test_same_dims_single_group(self, input_template):
        from earth2studio.statistics import mae, rmse

        metrics = OrderedDict(
            {
                "rmse": rmse(reduction_dimensions=["lat", "lon"]),
                "mae": mae(reduction_dimensions=["lat", "lon"]),
            }
        )
        groups = group_score_arrays_by_dims(metrics, input_template, TIMES)
        assert len(groups) == 1
        _, var_names = groups[0]
        assert "rmse__t2m" in var_names
        assert "mae__t2m" in var_names

    def test_different_dims_multiple_groups(self, input_template_ensemble):
        """RMSE (keeps ensemble) and CRPS (drops ensemble) form two groups."""
        from earth2studio.statistics import crps, rmse

        metrics = OrderedDict(
            {
                "rmse": rmse(reduction_dimensions=["lat", "lon"]),
                "crps": crps(
                    reduction_dimensions=["lat", "lon"],
                    ensemble_dimension="ensemble",
                ),
            }
        )
        groups = group_score_arrays_by_dims(metrics, input_template_ensemble, TIMES)
        assert len(groups) == 2
        all_vars = [v for _, vs in groups for v in vs]
        assert "rmse__t2m" in all_vars
        assert "crps__t2m" in all_vars


# ---------------------------------------------------------------------------
# validate_lead_time_chunking
# ---------------------------------------------------------------------------


class TestValidateLeadTimeChunking:
    def test_no_chunking_always_passes(self):
        from earth2studio.statistics import rmse

        metrics = OrderedDict(
            {
                "temporal": rmse(reduction_dimensions=["lead_time"]),
            }
        )
        # No chunking — should not raise.
        validate_lead_time_chunking(metrics, None, 10)
        validate_lead_time_chunking(metrics, 0, 10)
        validate_lead_time_chunking(metrics, 10, 10)

    def test_chunking_with_lead_time_reduction_raises(self):
        from earth2studio.statistics import rmse

        metrics = OrderedDict(
            {
                "temporal": rmse(reduction_dimensions=["lead_time"]),
            }
        )
        with pytest.raises(ValueError, match="reduces over 'lead_time'"):
            validate_lead_time_chunking(metrics, 2, 10)

    def test_chunking_without_lead_time_reduction_passes(self):
        from earth2studio.statistics import rmse

        metrics = OrderedDict(
            {
                "spatial": rmse(reduction_dimensions=["lat", "lon"]),
            }
        )
        validate_lead_time_chunking(metrics, 2, 10)


# ---------------------------------------------------------------------------
# spatial_coords_from_dataset / build_input_coords_template
# ---------------------------------------------------------------------------


class TestSpatialCoordsFromDataset:
    def test_extracts_lat_lon(self, prediction_ds):
        coords = spatial_coords_from_dataset(prediction_ds)
        assert "lat" in coords
        assert "lon" in coords
        assert "time" not in coords
        assert "lead_time" not in coords

    def test_ensemble_excluded(self, prediction_ds_ensemble):
        coords = spatial_coords_from_dataset(prediction_ds_ensemble)
        assert "ensemble" not in coords
        assert "lat" in coords


class TestBuildInputCoordsTemplate:
    def test_deterministic(self, prediction_ds):
        template = build_input_coords_template(prediction_ds, LEAD_TIMES, VARIABLES)
        assert "ensemble" not in template
        assert list(template.keys()) == ["lead_time", "variable", "lat", "lon"]
        np.testing.assert_array_equal(template["variable"], VARIABLES)

    def test_ensemble(self, prediction_ds_ensemble):
        template = build_input_coords_template(
            prediction_ds_ensemble, LEAD_TIMES, VARIABLES
        )
        assert "ensemble" in template
        assert list(template.keys()) == [
            "ensemble",
            "lead_time",
            "variable",
            "lat",
            "lon",
        ]


# ---------------------------------------------------------------------------
# load_prediction_chunk / load_verification_chunk
# ---------------------------------------------------------------------------


class TestLoadPredictionChunk:
    def test_deterministic_shape(self, prediction_ds):
        x, coords = load_prediction_chunk(
            prediction_ds, TIMES[0], LEAD_TIMES, VARIABLES, torch.device("cpu")
        )
        assert x.shape == (3, 2, 4, 8)  # lead_time, variable, lat, lon
        assert list(coords.keys()) == ["lead_time", "variable", "lat", "lon"]

    def test_ensemble_shape(self, prediction_ds_ensemble):
        x, coords = load_prediction_chunk(
            prediction_ds_ensemble,
            TIMES[0],
            LEAD_TIMES,
            VARIABLES,
            torch.device("cpu"),
        )
        assert x.shape == (3, 3, 2, 4, 8)  # ensemble, lt, var, lat, lon
        assert list(coords.keys()) == [
            "ensemble",
            "lead_time",
            "variable",
            "lat",
            "lon",
        ]

    def test_single_lead_time(self, prediction_ds):
        lt_chunk = LEAD_TIMES[:1]
        x, coords = load_prediction_chunk(
            prediction_ds, TIMES[0], lt_chunk, VARIABLES, torch.device("cpu")
        )
        assert x.shape[0] == 1  # single lead time
        assert len(coords["lead_time"]) == 1


class TestLoadVerificationChunk:
    def test_shape_and_coords(self, verif_source, spatial_coords):
        x, coords = load_verification_chunk(
            verif_source,
            TIMES[0],
            LEAD_TIMES,
            VARIABLES,
            spatial_coords,
            torch.device("cpu"),
        )
        assert x.shape == (3, 2, 4, 8)  # lead_time, variable, lat, lon
        assert list(coords.keys()) == ["lead_time", "variable", "lat", "lon"]
        # lead_time coords should be lead times, not valid times.
        np.testing.assert_array_equal(coords["lead_time"], LEAD_TIMES)

    def test_single_lead_time(self, verif_source, spatial_coords):
        lt_chunk = LEAD_TIMES[:1]
        x, coords = load_verification_chunk(
            verif_source,
            TIMES[0],
            lt_chunk,
            VARIABLES,
            spatial_coords,
            torch.device("cpu"),
        )
        assert x.shape[0] == 1


# ---------------------------------------------------------------------------
# _concat_chunks
# ---------------------------------------------------------------------------


class TestConcatChunks:
    def test_single_chunk_passthrough(self):
        score = torch.randn(3, 2)  # lead_time, variable
        coords = OrderedDict(
            {
                "lead_time": LEAD_TIMES,
                "variable": np.array(VARIABLES),
            }
        )
        result_score, result_coords = _concat_chunks([(score, coords)], LEAD_TIMES)
        assert torch.equal(result_score, score)

    def test_two_chunks_concatenated(self):
        score1 = torch.randn(2, 2)
        coords1 = OrderedDict(
            {
                "lead_time": LEAD_TIMES[:2],
                "variable": np.array(VARIABLES),
            }
        )
        score2 = torch.randn(1, 2)
        coords2 = OrderedDict(
            {
                "lead_time": LEAD_TIMES[2:],
                "variable": np.array(VARIABLES),
            }
        )
        result_score, result_coords = _concat_chunks(
            [(score1, coords1), (score2, coords2)], LEAD_TIMES
        )
        assert result_score.shape == (3, 2)
        np.testing.assert_array_equal(result_coords["lead_time"], LEAD_TIMES)


# ---------------------------------------------------------------------------
# build_score_coords (output.py)
# ---------------------------------------------------------------------------


class TestBuildScoreCoords:
    def test_spatial_reduction(self, input_template):
        from earth2studio.statistics import rmse

        metric = rmse(reduction_dimensions=["lat", "lon"])
        coords = build_score_coords(metric, TIMES, input_template)
        assert list(coords.keys()) == ["time", "lead_time", "variable"]
        np.testing.assert_array_equal(coords["time"], TIMES)

    def test_no_double_time(self, input_template):
        """time should appear once even if metric output somehow has it."""
        from earth2studio.statistics import rmse

        metric = rmse(reduction_dimensions=["lat", "lon"])
        coords = build_score_coords(metric, TIMES, input_template)
        assert list(coords.keys()).count("time") == 1


# ---------------------------------------------------------------------------
# instantiate_metrics
# ---------------------------------------------------------------------------


class TestInstantiateMetrics:
    def test_basic_instantiation(self, spatial_coords):
        cfg = OmegaConf.create(
            {
                "scoring": {
                    "lat_weights": False,
                    "metrics": {
                        "rmse": {
                            "_target_": "earth2studio.statistics.rmse",
                            "reduction_dimensions": ["lat", "lon"],
                        }
                    },
                }
            }
        )
        metrics = instantiate_metrics(cfg, spatial_coords)
        assert "rmse" in metrics
        assert hasattr(metrics["rmse"], "reduction_dimensions")

    def test_lat_weights_injected(self, spatial_coords):
        cfg = OmegaConf.create(
            {
                "scoring": {
                    "lat_weights": True,
                    "metrics": {
                        "rmse": {
                            "_target_": "earth2studio.statistics.rmse",
                            "reduction_dimensions": ["lat", "lon"],
                        }
                    },
                }
            }
        )
        metrics = instantiate_metrics(cfg, spatial_coords)
        # The metric should have been created with weights.
        assert metrics["rmse"].weights is not None

    def test_lat_weights_skipped_when_no_lat_in_reduction(self, spatial_coords):
        cfg = OmegaConf.create(
            {
                "scoring": {
                    "lat_weights": True,
                    "metrics": {
                        "rmse": {
                            "_target_": "earth2studio.statistics.rmse",
                            "reduction_dimensions": ["lead_time"],
                        }
                    },
                }
            }
        )
        metrics = instantiate_metrics(cfg, spatial_coords)
        assert metrics["rmse"].weights is None


# ---------------------------------------------------------------------------
# Scoring progress tracking (work.py)
# ---------------------------------------------------------------------------


class TestScoringProgress:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return OmegaConf.create({"output": {"path": str(tmp_path / "outputs")}})

    def test_progress_dir_path(self, cfg, tmp_path):
        d = scoring_progress_dir(cfg)
        assert d == tmp_path / "outputs" / ".scoring_progress"

    def test_write_and_filter_marker(self, cfg):
        t1 = np.datetime64("2024-01-01T00:00:00")
        t2 = np.datetime64("2024-01-02T00:00:00")
        times = [t1, t2]

        # No markers yet — all times remain.
        remaining = filter_scoring_completed(times, cfg)
        assert len(remaining) == 2

        # Write marker for t1.
        write_scoring_marker(t1, cfg)

        # Now only t2 should remain.
        remaining = filter_scoring_completed(times, cfg)
        assert len(remaining) == 1
        assert remaining[0] == t2

    def test_clear_scoring_progress(self, cfg):
        t1 = np.datetime64("2024-01-01T00:00:00")
        write_scoring_marker(t1, cfg)

        d = scoring_progress_dir(cfg)
        assert d.exists()

        clear_scoring_progress(cfg)
        assert not d.exists()


# ---------------------------------------------------------------------------
# End-to-end: run_scoring
# ---------------------------------------------------------------------------


class TestRunScoring:
    @pytest.fixture()
    def cfg(self, tmp_path):
        return OmegaConf.create(
            {
                "output": {
                    "path": str(tmp_path / "out"),
                    "overwrite": True,
                    "thread_writers": 0,
                    "chunks": {"time": 1},
                },
                "scoring": {
                    "resume": False,
                    "lat_weights": False,
                    "lead_time_chunk_size": None,
                    "variables": list(VARIABLES),
                    "metrics": {
                        "rmse": {
                            "_target_": "earth2studio.statistics.rmse",
                            "reduction_dimensions": ["lat", "lon"],
                        },
                    },
                    "output": {
                        "store_name": "scores.zarr",
                        "overwrite": True,
                        "chunks": {"time": 1},
                    },
                },
            }
        )

    def _setup_score_store(self, cfg, prediction_ds, metrics):
        """Helper: build coords and array groups for test store setup."""
        input_template = build_input_coords_template(
            prediction_ds, LEAD_TIMES, VARIABLES
        )
        superset_coords = build_superset_score_coords(metrics, input_template, TIMES)
        array_groups = group_score_arrays_by_dims(metrics, input_template, TIMES)
        return input_template, superset_coords, array_groups

    def test_end_to_end_deterministic(
        self, cfg, prediction_ds, verif_source, spatial_coords, tmp_path
    ):
        """Run scoring on deterministic predictions and verify output."""
        from earth2studio.statistics import rmse

        metrics = OrderedDict(
            {
                "rmse": rmse(reduction_dimensions=["lat", "lon"]),
            }
        )
        input_template, superset_coords, array_groups = self._setup_score_store(
            cfg, prediction_ds, metrics
        )

        lt_chunks = build_lead_time_chunks(LEAD_TIMES, None)
        device = torch.device("cpu")

        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                with patch(_SCORING_DIST_PATH, return_value=_make_dist_mock()):
                    with OutputManager(
                        cfg, store_name="scores.zarr", overwrite=True
                    ) as mgr:
                        mgr.validate_output_store(superset_coords, [])
                        add_score_arrays(mgr.io, array_groups)

                        run_scoring(
                            list(TIMES),
                            prediction_ds,
                            verif_source,
                            metrics,
                            mgr,
                            VARIABLES,
                            LEAD_TIMES,
                            lt_chunks,
                            spatial_coords,
                            device,
                            cfg,
                        )

        # Verify the output store exists and has the right arrays.
        score_path = os.path.join(cfg.output.path, "scores.zarr")
        assert os.path.exists(score_path)

        score_ds = xr.open_zarr(score_path)
        assert "rmse__t2m" in score_ds
        assert "rmse__z500" in score_ds

        # Scores should be non-negative (RMSE >= 0).
        rmse_t2m = score_ds["rmse__t2m"].values
        assert np.all(np.isfinite(rmse_t2m))
        assert np.all(rmse_t2m >= 0)

        # Check dimensions: (time, lead_time).
        assert score_ds["rmse__t2m"].dims == ("time", "lead_time")
        assert len(score_ds.time) == 2
        assert len(score_ds.lead_time) == 3

    def test_end_to_end_with_chunking(
        self, cfg, prediction_ds, verif_source, spatial_coords
    ):
        """Chunked and non-chunked scoring should produce identical results."""
        from earth2studio.statistics import rmse

        metrics = OrderedDict(
            {
                "rmse": rmse(reduction_dimensions=["lat", "lon"]),
            }
        )
        _, superset_coords, array_groups = self._setup_score_store(
            cfg, prediction_ds, metrics
        )

        device = torch.device("cpu")
        results = {}

        for label, chunk_size in [("full", None), ("chunked", 1)]:
            lt_chunks = build_lead_time_chunks(LEAD_TIMES, chunk_size)
            store_name = f"scores_{label}.zarr"

            with patch(_DIST_PATH, return_value=_make_dist_mock()):
                with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                    with patch(_SCORING_DIST_PATH, return_value=_make_dist_mock()):
                        with OutputManager(
                            cfg, store_name=store_name, overwrite=True
                        ) as mgr:
                            mgr.validate_output_store(superset_coords, [])
                            add_score_arrays(mgr.io, array_groups)
                            run_scoring(
                                list(TIMES),
                                prediction_ds,
                                verif_source,
                                metrics,
                                mgr,
                                VARIABLES,
                                LEAD_TIMES,
                                lt_chunks,
                                spatial_coords,
                                device,
                                cfg,
                            )

            ds = xr.open_zarr(os.path.join(cfg.output.path, store_name))
            results[label] = ds["rmse__t2m"].values

        np.testing.assert_allclose(results["full"], results["chunked"], rtol=1e-5)

    def test_end_to_end_ensemble(
        self, cfg, prediction_ds_ensemble, verif_source, spatial_coords
    ):
        """Scoring ensemble predictions with RMSE (ensemble mean first)."""
        from earth2studio.statistics import rmse

        metrics = OrderedDict(
            {
                "rmse": rmse(
                    reduction_dimensions=["lat", "lon"],
                    ensemble_dimension="ensemble",
                ),
            }
        )
        _, superset_coords, array_groups = self._setup_score_store(
            cfg, prediction_ds_ensemble, metrics
        )

        lt_chunks = build_lead_time_chunks(LEAD_TIMES, None)
        device = torch.device("cpu")

        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                with patch(_SCORING_DIST_PATH, return_value=_make_dist_mock()):
                    with OutputManager(
                        cfg, store_name="scores.zarr", overwrite=True
                    ) as mgr:
                        mgr.validate_output_store(superset_coords, [])
                        add_score_arrays(mgr.io, array_groups)
                        run_scoring(
                            list(TIMES),
                            prediction_ds_ensemble,
                            verif_source,
                            metrics,
                            mgr,
                            VARIABLES,
                            LEAD_TIMES,
                            lt_chunks,
                            spatial_coords,
                            device,
                            cfg,
                        )

        score_ds = xr.open_zarr(os.path.join(cfg.output.path, "scores.zarr"))
        assert "rmse__t2m" in score_ds
        assert score_ds["rmse__t2m"].dims == ("time", "lead_time")
        assert np.all(score_ds["rmse__t2m"].values >= 0)


# ---------------------------------------------------------------------------
# open_verification_source — pipeline hook resolution
# ---------------------------------------------------------------------------


class TestOpenVerificationSourceHook:
    """``open_verification_source`` delegates to
    :meth:`Pipeline.verification_source`.  The base implementation picks
    up BYO overrides, ``verification.zarr``, ``data.zarr``, or
    per-model ``data_*.zarr`` (→ CompositeSource) in that order, so the
    pipeline class selects the correct verification layout."""

    def _make_cfg(self, tmp_path, pipeline: str):
        return OmegaConf.create(
            {
                "output": {"path": str(tmp_path)},
                "pipeline": pipeline,
            }
        )

    def test_stormscope_picks_up_per_model_stores(self, tmp_path):
        # Write per-model IC zarrs so the default glob picks them up and
        # wraps them in a CompositeSource.  Leave verification.zarr /
        # data.zarr absent so the fallback path is exercised.
        from src.data import CompositeSource
        from test.test_data import _create_yx_zarr_store

        t = np.array(["2024-01-01"], dtype="datetime64[ns]")
        _create_yx_zarr_store(
            tmp_path / "data_goes.zarr", t, ["abi01c"]
        )
        _create_yx_zarr_store(
            tmp_path / "data_mrms.zarr", t, ["refc"]
        )

        cfg = self._make_cfg(
            tmp_path,
            pipeline="src.pipelines.stormscope.StormScopePipeline",
        )
        src = open_verification_source(cfg)
        assert isinstance(src, CompositeSource)

    def test_raises_when_no_verification_data_available(self, tmp_path):
        # With no override and no predownloaded stores, the hook raises.
        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path)},
                "pipeline": "src.pipelines.forecast.ForecastPipeline",
            }
        )
        with pytest.raises(FileNotFoundError, match="No verification data"):
            open_verification_source(cfg)


# ---------------------------------------------------------------------------
# nan_policy — scoring tolerant of NaN grid points
# ---------------------------------------------------------------------------


class TestNanPolicy:
    """``scoring.nan_policy`` controls whether NaNs in prediction or
    verification tensors propagate through metric reductions.

    * ``propagate`` (default) preserves earth2studio's stock behavior
      so DLESyM-style invalid-lead NaNs surface as NaN in the scores.
    * ``zero_fill`` replaces NaNs with 0 before the metric runs, so
      spatial-reduction metrics (e.g. rmse over ``[y, x]``) return a
      real number rather than collapsing to NaN — matters for
      StormScope's HRRR-sub-domain invalid-pixel mask.
    """

    def _build_inputs(self, tmp_path):
        """Construct minimal objects for ``run_scoring``.

        Uses the same store setup as :class:`TestRunScoring` — superset
        coords + ``add_score_arrays`` — so the zarr write path matches
        production.
        """
        # One IC, one lead time, single variable.  Prediction is 0 everywhere;
        # verification is 0 everywhere except one NaN pixel that simulates an
        # invalid HRRR-subdomain grid point.
        times = np.array(["2024-01-01"], dtype="datetime64[ns]")
        lead_times = np.array([60], dtype="timedelta64[m]").astype(
            "timedelta64[ns]"
        )
        y = np.arange(4)
        x = np.arange(5)

        pred = np.zeros((1, 1, 4, 5), dtype="float32")
        truth = np.zeros((1, 4, 5), dtype="float32")
        truth[0, 0, 0] = np.nan

        pred_path = tmp_path / "forecast.zarr"
        xr.Dataset(
            {
                "v": xr.DataArray(
                    pred,
                    dims=["time", "lead_time", "y", "x"],
                    coords={
                        "time": times,
                        "lead_time": lead_times,
                        "y": y,
                        "x": x,
                    },
                )
            }
        ).to_zarr(str(pred_path), mode="w")
        prediction_ds = xr.open_zarr(str(pred_path))

        valid_times = times + lead_times[0]
        verif_path = tmp_path / "verification.zarr"
        xr.Dataset(
            {
                "v": xr.DataArray(
                    truth,
                    dims=["time", "y", "x"],
                    coords={"time": valid_times, "y": y, "x": x},
                )
            }
        ).to_zarr(str(verif_path), mode="w")
        verif_source = PredownloadedSource(str(verif_path))

        import earth2studio.statistics

        metrics = OrderedDict(
            {
                "rmse": earth2studio.statistics.rmse(
                    reduction_dimensions=["y", "x"]
                )
            }
        )
        variables = ["v"]
        input_template = build_input_coords_template(
            prediction_ds, lead_times, variables
        )
        superset_coords = build_superset_score_coords(
            metrics, input_template, times
        )
        array_groups = group_score_arrays_by_dims(
            metrics, input_template, times
        )
        spatial = OrderedDict([("y", y), ("x", x)])

        return {
            "times": list(times),
            "prediction_ds": prediction_ds,
            "verif_source": verif_source,
            "metrics": metrics,
            "variables": variables,
            "lead_times": lead_times,
            "lt_chunks": [lead_times],
            "spatial": spatial,
            "superset_coords": superset_coords,
            "array_groups": array_groups,
        }

    def _cfg(self, tmp_path, nan_policy: str | None):
        scoring_block: dict = {
            "output": {"store_name": "scores.zarr", "overwrite": True},
        }
        if nan_policy is not None:
            scoring_block["nan_policy"] = nan_policy
        return OmegaConf.create(
            {
                "output": {
                    "path": str(tmp_path),
                    "chunks": {"time": 1, "lead_time": 1},
                },
                "scoring": scoring_block,
            }
        )

    def _run(self, cfg, inputs):
        """Execute the standard score-store-setup + run_scoring flow."""
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                with patch(_SCORING_DIST_PATH, return_value=_make_dist_mock()):
                    with OutputManager(
                        cfg, store_name="scores.zarr", overwrite=True
                    ) as mgr:
                        mgr.validate_output_store(inputs["superset_coords"], [])
                        add_score_arrays(mgr.io, inputs["array_groups"])
                        run_scoring(
                            inputs["times"],
                            inputs["prediction_ds"],
                            inputs["verif_source"],
                            inputs["metrics"],
                            mgr,
                            inputs["variables"],
                            inputs["lead_times"],
                            inputs["lt_chunks"],
                            inputs["spatial"],
                            torch.device("cpu"),
                            cfg,
                        )

    def test_propagate_default_yields_nan_score(self, tmp_path):
        """Default policy lets NaNs propagate — a single invalid pixel
        poisons the entire RMSE over ``[y, x]``."""
        inputs = self._build_inputs(tmp_path)
        cfg = self._cfg(tmp_path, nan_policy=None)
        self._run(cfg, inputs)

        score_ds = xr.open_zarr(str(tmp_path / "scores.zarr"))
        assert np.isnan(score_ds["rmse__v"].values).all()

    def test_zero_fill_produces_finite_score(self, tmp_path):
        """With ``nan_policy=zero_fill`` the NaN pixel becomes 0 and the
        RMSE is a real (non-NaN) number."""
        inputs = self._build_inputs(tmp_path)
        cfg = self._cfg(tmp_path, nan_policy="zero_fill")
        self._run(cfg, inputs)

        score_ds = xr.open_zarr(str(tmp_path / "scores.zarr"))
        vals = score_ds["rmse__v"].values
        assert not np.isnan(vals).any()
        # pred=0 everywhere, truth=0 everywhere except one NaN → 0 after
        # zero_fill.  So sum-of-squares = 0, RMSE = 0.
        np.testing.assert_allclose(vals, 0.0)

    def test_invalid_nan_policy_raises(self, tmp_path):
        inputs = self._build_inputs(tmp_path)
        cfg = self._cfg(tmp_path, nan_policy="bogus")
        with pytest.raises(ValueError, match="Invalid scoring.nan_policy"):
            self._run(cfg, inputs)


# ---------------------------------------------------------------------------
# valid_ranges — per-variable clamp to suppress fill-value outliers
# ---------------------------------------------------------------------------


class TestApplyValidRanges:
    """Unit tests for the private ``_apply_valid_ranges`` helper.

    The helper clamps per-variable tensor slices along the ``variable``
    axis so that prediction/verification values outside a configured
    physical range are truncated — the mechanism used to prevent
    large-magnitude no-data fill values (MRMS ``refc``'s ``-9999``)
    from dominating RMSE-style reductions.
    """

    def test_clamps_named_variable_only(self):
        from src.scoring import _apply_valid_ranges

        coords = OrderedDict(
            [
                ("lead_time", np.array([0, 60], dtype="timedelta64[m]")),
                ("variable", np.array(["refc", "abi01c"])),
                ("y", np.arange(3)),
                ("x", np.arange(4)),
            ]
        )
        x = torch.tensor(
            [
                [  # lead_time=0
                    [[-9999.0] * 4] * 3,  # refc: fill values
                    [[0.5] * 4] * 3,  # abi01c: in-range
                ],
                [  # lead_time=60
                    [[50.0] * 4] * 3,  # refc: in-range
                    [[2.0] * 4] * 3,  # abi01c: over-range
                ],
            ],
            dtype=torch.float32,
        )
        valid_ranges = {"refc": {"min": 0, "max": 75}}

        out = _apply_valid_ranges(x.clone(), coords, valid_ranges)
        # refc: -9999 → 0 (clamped to min), 50 unchanged
        assert (out[0, 0] == 0.0).all()
        assert (out[1, 0] == 50.0).all()
        # abi01c: untouched (no entry in valid_ranges)
        assert (out[0, 1] == 0.5).all()
        assert (out[1, 1] == 2.0).all()

    def test_null_bound_is_one_sided(self):
        from src.scoring import _apply_valid_ranges

        coords = OrderedDict(
            [
                ("variable", np.array(["v"])),
                ("y", np.arange(2)),
                ("x", np.arange(2)),
            ]
        )
        x = torch.tensor(
            [[[-100.0, 50.0], [100.0, -50.0]]],
            dtype=torch.float32,
        )
        # Only a lower bound: negatives clip to 0, large positives pass.
        out = _apply_valid_ranges(
            x.clone(), coords, {"v": {"min": 0, "max": None}}
        )
        assert out[0, 0, 0].item() == 0.0
        assert out[0, 0, 1].item() == 50.0
        assert out[0, 1, 0].item() == 100.0  # unclamped from above
        assert out[0, 1, 1].item() == 0.0

    def test_both_bounds_none_is_noop(self):
        from src.scoring import _apply_valid_ranges

        coords = OrderedDict(
            [
                ("variable", np.array(["v"])),
                ("y", np.arange(1)),
                ("x", np.arange(1)),
            ]
        )
        x = torch.tensor([[[-1e6]]], dtype=torch.float32)
        out = _apply_valid_ranges(
            x.clone(), coords, {"v": {"min": None, "max": None}}
        )
        assert out[0, 0, 0].item() == -1e6

    def test_unknown_variable_is_skipped(self):
        from src.scoring import _apply_valid_ranges

        coords = OrderedDict(
            [
                ("variable", np.array(["refc"])),
                ("y", np.arange(1)),
                ("x", np.arange(1)),
            ]
        )
        x = torch.tensor([[[-9999.0]]], dtype=torch.float32)
        # No entry for "refc" — nothing should be clamped.
        out = _apply_valid_ranges(x.clone(), coords, {"other_var": {"min": 0}})
        assert out[0, 0, 0].item() == -9999.0


class TestValidRangesEndToEnd:
    """``scoring.valid_ranges`` integrated with the full scoring flow.

    Builds a tiny store whose verification tensor has a ``-9999`` fill
    value at one pixel; asserts the resulting RMSE is ~0 when the
    valid range is configured, and huge otherwise.
    """

    def _build_inputs(self, tmp_path, truth_fill_value: float):
        times = np.array(["2024-01-01"], dtype="datetime64[ns]")
        lead_times = np.array([60], dtype="timedelta64[m]").astype(
            "timedelta64[ns]"
        )
        y = np.arange(4)
        x = np.arange(5)

        # Prediction: 0 everywhere (matches the StormScope fill-invalid behavior).
        pred = np.zeros((1, 1, 4, 5), dtype="float32")
        # Truth: 0 everywhere except one "invalid" pixel with the fill value.
        truth = np.zeros((1, 4, 5), dtype="float32")
        truth[0, 0, 0] = truth_fill_value

        pred_path = tmp_path / "forecast.zarr"
        xr.Dataset(
            {
                "refc": xr.DataArray(
                    pred,
                    dims=["time", "lead_time", "y", "x"],
                    coords={
                        "time": times,
                        "lead_time": lead_times,
                        "y": y,
                        "x": x,
                    },
                )
            }
        ).to_zarr(str(pred_path), mode="w")
        prediction_ds = xr.open_zarr(str(pred_path))

        valid_times = times + lead_times[0]
        verif_path = tmp_path / "verification.zarr"
        xr.Dataset(
            {
                "refc": xr.DataArray(
                    truth,
                    dims=["time", "y", "x"],
                    coords={"time": valid_times, "y": y, "x": x},
                )
            }
        ).to_zarr(str(verif_path), mode="w")
        verif_source = PredownloadedSource(str(verif_path))

        import earth2studio.statistics

        metrics = OrderedDict(
            {
                "rmse": earth2studio.statistics.rmse(
                    reduction_dimensions=["y", "x"]
                )
            }
        )
        variables = ["refc"]
        input_template = build_input_coords_template(
            prediction_ds, lead_times, variables
        )
        superset_coords = build_superset_score_coords(
            metrics, input_template, times
        )
        array_groups = group_score_arrays_by_dims(
            metrics, input_template, times
        )
        spatial = OrderedDict([("y", y), ("x", x)])

        return {
            "times": list(times),
            "prediction_ds": prediction_ds,
            "verif_source": verif_source,
            "metrics": metrics,
            "variables": variables,
            "lead_times": lead_times,
            "lt_chunks": [lead_times],
            "spatial": spatial,
            "superset_coords": superset_coords,
            "array_groups": array_groups,
        }

    def _run(self, cfg, inputs):
        with patch(_DIST_PATH, return_value=_make_dist_mock()):
            with patch(_RANK0_PATH, side_effect=lambda fn, *a, **kw: fn(*a, **kw)):
                with patch(_SCORING_DIST_PATH, return_value=_make_dist_mock()):
                    with OutputManager(
                        cfg, store_name="scores.zarr", overwrite=True
                    ) as mgr:
                        mgr.validate_output_store(inputs["superset_coords"], [])
                        add_score_arrays(mgr.io, inputs["array_groups"])
                        run_scoring(
                            inputs["times"],
                            inputs["prediction_ds"],
                            inputs["verif_source"],
                            inputs["metrics"],
                            mgr,
                            inputs["variables"],
                            inputs["lead_times"],
                            inputs["lt_chunks"],
                            inputs["spatial"],
                            torch.device("cpu"),
                            cfg,
                        )

    def _cfg(self, tmp_path, valid_ranges: dict | None):
        scoring_block: dict = {
            "output": {"store_name": "scores.zarr", "overwrite": True},
            "nan_policy": "zero_fill",
        }
        if valid_ranges is not None:
            scoring_block["valid_ranges"] = valid_ranges
        return OmegaConf.create(
            {
                "output": {
                    "path": str(tmp_path),
                    "chunks": {"time": 1, "lead_time": 1},
                },
                "scoring": scoring_block,
            }
        )

    def test_without_valid_ranges_fill_inflates_rmse(self, tmp_path):
        """Baseline: without valid_ranges, the -9999 fill value drives the
        RMSE to ~999 (sqrt of 9999^2 averaged over 20 pixels)."""
        inputs = self._build_inputs(tmp_path, truth_fill_value=-9999.0)
        cfg = self._cfg(tmp_path, valid_ranges=None)
        self._run(cfg, inputs)

        score_ds = xr.open_zarr(str(tmp_path / "scores.zarr"))
        vals = score_ds["rmse__refc"].values
        assert float(vals[0, 0]) > 1000.0  # heavily inflated

    def test_with_valid_ranges_fill_is_clamped(self, tmp_path):
        """With valid_ranges clamping refc to [0, 75], the -9999 pixel
        becomes 0, pred=0, diff=0, RMSE=0."""
        inputs = self._build_inputs(tmp_path, truth_fill_value=-9999.0)
        cfg = self._cfg(
            tmp_path, valid_ranges={"refc": {"min": 0, "max": 75}}
        )
        self._run(cfg, inputs)

        score_ds = xr.open_zarr(str(tmp_path / "scores.zarr"))
        vals = score_ds["rmse__refc"].values
        np.testing.assert_allclose(vals, 0.0)

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

"""Tests for the report generation module (src/report.py)."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import OmegaConf
from src.report import (
    SectionOutput,
    _grid_shape,
    aggregate_over_ensemble,
    aggregate_over_time,
    build_summary_csv,
    generate_report,
    parse_score_arrays,
    plot_ic_heatmap,
    plot_metric_grid,
    plot_metric_vs_leadtime,
    plot_prediction_vs_truth,
    render_header_visualization,
    render_lead_time_curves,
    render_summary_table,
    render_visualization,
    resolve_time_groups,
    snapshot_at_lead_times,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

TIMES = np.array(
    ["2024-01-01", "2024-03-01", "2024-06-01", "2024-09-01"],
    dtype="datetime64[ns]",
)
LEAD_TIMES = np.array([0, 1, 3, 5, 7], dtype="timedelta64[D]").astype("timedelta64[ns]")
ENSEMBLE = np.arange(4)
VARIABLES = ["t2m", "z500"]
SMALL_LAT = np.linspace(90, -90, 8)
SMALL_LON = np.linspace(0, 360, 16, endpoint=False)


def _create_score_dataset(
    *, include_ensemble_rmse: bool = False, include_crps: bool = False
) -> xr.Dataset:
    """Build a synthetic score dataset for testing."""
    rng = np.random.default_rng(42)
    ds = xr.Dataset()

    for var in VARIABLES:
        if include_ensemble_rmse:
            # RMSE with ensemble dimension: (time, ensemble, lead_time)
            ds[f"rmse__{var}"] = xr.DataArray(
                rng.uniform(
                    0.5, 5.0, (len(TIMES), len(ENSEMBLE), len(LEAD_TIMES))
                ).astype("float32"),
                dims=["time", "ensemble", "lead_time"],
                coords={
                    "time": TIMES,
                    "ensemble": ENSEMBLE,
                    "lead_time": LEAD_TIMES,
                },
            )
        else:
            # RMSE deterministic: (time, lead_time)
            ds[f"rmse__{var}"] = xr.DataArray(
                rng.uniform(0.5, 5.0, (len(TIMES), len(LEAD_TIMES))).astype("float32"),
                dims=["time", "lead_time"],
                coords={"time": TIMES, "lead_time": LEAD_TIMES},
            )

        if include_crps:
            # CRPS: (time, lead_time) — ensemble already reduced
            ds[f"crps__{var}"] = xr.DataArray(
                rng.uniform(0.1, 3.0, (len(TIMES), len(LEAD_TIMES))).astype("float32"),
                dims=["time", "lead_time"],
                coords={"time": TIMES, "lead_time": LEAD_TIMES},
            )

    return ds


def _write_score_zarr(path, ds):
    """Write a score dataset to zarr."""
    ds.to_zarr(str(path))


# ---------------------------------------------------------------------------
# parse_score_arrays
# ---------------------------------------------------------------------------


class TestParseScoreArrays:
    def test_basic(self):
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        assert "rmse" in groups
        assert set(groups["rmse"]) == {"t2m", "z500"}

    def test_multiple_metrics(self):
        ds = _create_score_dataset(include_crps=True)
        groups = parse_score_arrays(ds)
        assert "rmse" in groups
        assert "crps" in groups
        assert set(groups["crps"]) == {"t2m", "z500"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregateOverTime:
    def test_rmse_quadratic(self):
        """RMSE aggregation should use sqrt(mean(x²)), not mean(x)."""
        # Create known values: RMSE = [1, 3] across two times.
        # Correct: sqrt((1+9)/2) = sqrt(5) ≈ 2.236
        # Wrong (naive mean): (1+3)/2 = 2.0
        da = xr.DataArray(
            np.array([[1.0], [3.0]]),
            dims=["time", "lead_time"],
            coords={
                "time": TIMES[:2],
                "lead_time": LEAD_TIMES[:1],
            },
        )
        result = aggregate_over_time(da, "rmse")
        expected = np.sqrt((1.0 + 9.0) / 2.0)
        np.testing.assert_allclose(result.values, [expected], rtol=1e-5)

    def test_non_quadratic_uses_mean(self):
        """Non-RMSE metrics should use simple mean."""
        da = xr.DataArray(
            np.array([[1.0], [3.0]]),
            dims=["time", "lead_time"],
            coords={
                "time": TIMES[:2],
                "lead_time": LEAD_TIMES[:1],
            },
        )
        result = aggregate_over_time(da, "crps")
        np.testing.assert_allclose(result.values, [2.0], rtol=1e-5)

    def test_removes_time_dim(self):
        ds = _create_score_dataset()
        da = ds["rmse__t2m"]
        result = aggregate_over_time(da, "rmse")
        assert "time" not in result.dims
        assert "lead_time" in result.dims


class TestAggregateOverEnsemble:
    def test_with_ensemble(self):
        ds = _create_score_dataset(include_ensemble_rmse=True)
        da = ds["rmse__t2m"]
        result = aggregate_over_ensemble(da)
        assert "ensemble" not in result.dims
        assert "time" in result.dims
        assert "lead_time" in result.dims

    def test_without_ensemble_noop(self):
        ds = _create_score_dataset()
        da = ds["rmse__t2m"]
        result = aggregate_over_ensemble(da)
        assert result is da  # should be the same object


# ---------------------------------------------------------------------------
# resolve_time_groups
# ---------------------------------------------------------------------------


class TestResolveTimeGroups:
    def test_basic_ranges(self):
        cfg = {
            "winter": [{"start": "2024-01-01", "end": "2024-02-29"}],
            "summer": [{"start": "2024-06-01", "end": "2024-08-31"}],
        }
        groups = resolve_time_groups(cfg, TIMES)
        assert "winter" in groups
        assert "summer" in groups
        assert len(groups["winter"]) == 1  # 2024-01-01
        assert len(groups["summer"]) == 1  # 2024-06-01

    def test_multiple_ranges_per_group(self):
        """DJF wraps around — needs two ranges."""
        cfg = {
            "DJF": [
                {"start": "2024-01-01", "end": "2024-02-29"},
                {"start": "2024-12-01", "end": "2024-12-31"},
            ],
        }
        times = np.array(
            ["2024-01-15", "2024-06-15", "2024-12-15"],
            dtype="datetime64[ns]",
        )
        groups = resolve_time_groups(cfg, times)
        assert len(groups["DJF"]) == 2  # Jan and Dec

    def test_empty_group_skipped(self):
        cfg = {
            "future": [{"start": "2025-01-01", "end": "2025-12-31"}],
        }
        groups = resolve_time_groups(cfg, TIMES)
        assert "future" not in groups

    def test_inclusive_boundaries(self):
        """Boundaries should be inclusive."""
        cfg = {
            "exact": [{"start": "2024-01-01", "end": "2024-01-01"}],
        }
        groups = resolve_time_groups(cfg, TIMES)
        assert "exact" in groups
        assert len(groups["exact"]) == 1


# ---------------------------------------------------------------------------
# snapshot_at_lead_times
# ---------------------------------------------------------------------------


class TestSnapshotAtLeadTimes:
    def test_basic(self):
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        df = snapshot_at_lead_times(ds, groups, ["1 days", "5 days"])
        assert not df.empty
        assert set(df.columns) == {"metric", "variable", "lead_time", "value"}
        assert set(df["lead_time"]) == {"1 days", "5 days"}

    def test_missing_lead_time_excluded(self):
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        df = snapshot_at_lead_times(ds, groups, ["1 days", "999 days"])
        assert "999 days" not in df["lead_time"].values


# ---------------------------------------------------------------------------
# build_summary_csv
# ---------------------------------------------------------------------------


class TestBuildSummaryCSV:
    def test_columns_and_model(self):
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        df = build_summary_csv(ds, groups, "test_model")
        assert set(df.columns) == {"model", "metric", "variable", "lead_time", "value"}
        assert (df["model"] == "test_model").all()

    def test_all_lead_times_present(self):
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        df = build_summary_csv(ds, groups, "test")
        # Should have entries for all 5 lead times × 2 variables × 1 metric
        assert len(df) == 5 * 2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlotMetricVsLeadtime:
    def test_basic(self):
        ds = _create_score_dataset()
        fig = plot_metric_vs_leadtime(ds, "rmse", VARIABLES)
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_variable_group_name(self):
        ds = _create_score_dataset()
        fig = plot_metric_vs_leadtime(
            ds, "rmse", VARIABLES, variable_group_name="surface"
        )
        assert "surface" in fig.axes[0].get_title()
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_with_time_groups(self):
        ds = _create_score_dataset()
        time_groups = {
            "winter": TIMES[:1],
            "summer": TIMES[2:3],
        }
        fig = plot_metric_vs_leadtime(ds, "rmse", VARIABLES, time_groups=time_groups)
        # Should have more lines than just variables
        n_lines = len(fig.axes[0].get_lines())
        assert n_lines > len(VARIABLES)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_time_group_lines_have_distinct_styles(self):
        """Each time group should get a different line style."""
        ds = _create_score_dataset()
        time_groups = {
            "winter": TIMES[:1],
            "summer": TIMES[2:3],
        }
        fig = plot_metric_vs_leadtime(ds, "rmse", ["t2m"], time_groups=time_groups)
        lines = fig.axes[0].get_lines()
        # lines[0] = "all" solid, lines[1] = winter, lines[2] = summer
        assert len(lines) == 3
        # The two group lines should have different linestyles
        ls1 = lines[1].get_linestyle()
        ls2 = lines[2].get_linestyle()
        assert ls1 != ls2
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestPlotIcHeatmap:
    def test_basic(self):
        ds = _create_score_dataset()
        fig = plot_ic_heatmap(ds, "rmse", "t2m")
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_uses_plasma_colormap(self):
        ds = _create_score_dataset()
        fig = plot_ic_heatmap(ds, "rmse", "t2m")
        # pcolormesh is in ax.collections
        mesh = fig.axes[0].collections[0]
        assert mesh.cmap.name == "plasma"
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_single_ic_falls_back_to_bar_chart(self):
        """With one IC time a heatmap is degenerate and pcolormesh's
        autoscale often clips the single row.  The renderer falls back
        to a bar chart of metric vs lead time."""
        import matplotlib.pyplot as plt

        # Build a 1-IC, multi-lead score dataset.
        lead_times = np.array([60, 120], dtype="timedelta64[m]").astype(
            "timedelta64[ns]"
        )
        one_time = np.array(["2023-12-05T12:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "rmse__refc": xr.DataArray(
                    np.array([[0.5, 1.0]], dtype="float32"),
                    dims=["time", "lead_time"],
                    coords={"time": one_time, "lead_time": lead_times},
                )
            }
        )
        fig = plot_ic_heatmap(ds, "rmse", "refc", time_unit="hours")
        ax = fig.axes[0]
        # No colorbar axis in the fallback.
        assert len(fig.axes) == 1
        # Bar patches (one per lead time) rather than a QuadMesh collection.
        bars = [p for p in ax.patches if p.get_height() > 0]
        assert len(bars) == 2
        # IC label should appear in the title.
        assert "2023-12-05" in ax.get_title()
        plt.close(fig)

    def test_multi_ic_still_renders_quadmesh(self):
        """With multiple ICs the heatmap renders a QuadMesh (not the
        single-IC bar fallback).  We don't probe the internal ``_shading``
        attribute because matplotlib normalizes ``"nearest"`` into a
        padded-coords ``"flat"`` internally, and the attribute isn't a
        stable cross-version invariant."""
        import matplotlib.pyplot as plt
        from matplotlib.collections import QuadMesh

        ds = _create_score_dataset()  # 4 ICs, 5 lead times
        fig = plot_ic_heatmap(ds, "rmse", "t2m")
        ax = fig.axes[0]
        meshes = [c for c in ax.collections if isinstance(c, QuadMesh)]
        assert len(meshes) == 1
        # Colorbar axis is present (bar-chart fallback has only 1 axis).
        assert len(fig.axes) >= 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

# Dummy cfg for renderers that don't need data stores.
_EMPTY_CFG = OmegaConf.create({"output": {"path": "/nonexistent"}})


class TestRenderSummaryTable:
    def test_basic(self):
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create(
            {
                "lead_times": ["1 days", "5 days"],
                "collapsed": False,
            }
        )
        report_cfg = OmegaConf.create({})
        result = render_summary_table(
            ds,
            section_cfg,
            report_cfg,
            groups,
            _EMPTY_CFG,
        )
        assert isinstance(result, SectionOutput)
        assert "## Summary" in result.markdown
        assert "t2m" in result.markdown
        assert "z500" in result.markdown

    def test_collapsed(self):
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create(
            {
                "lead_times": ["1 days"],
                "collapsed": True,
            }
        )
        report_cfg = OmegaConf.create({})
        result = render_summary_table(
            ds,
            section_cfg,
            report_cfg,
            groups,
            _EMPTY_CFG,
        )
        assert "<details>" in result.markdown

    def test_variables_subset(self):
        """Only specified variables appear in the summary table."""
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create(
            {
                "lead_times": ["1 days", "5 days"],
                "variables": ["t2m"],
                "collapsed": False,
            }
        )
        report_cfg = OmegaConf.create({})
        result = render_summary_table(
            ds,
            section_cfg,
            report_cfg,
            groups,
            _EMPTY_CFG,
        )
        assert "t2m" in result.markdown
        assert "z500" not in result.markdown


class TestGridShape:
    def test_single(self):
        assert _grid_shape(1) == (1, 1)

    def test_two(self):
        assert _grid_shape(2) == (1, 2)

    def test_three(self):
        assert _grid_shape(3) == (1, 3)

    def test_four(self):
        assert _grid_shape(4) == (1, 4)

    def test_five(self):
        nrows, ncols = _grid_shape(5)
        assert nrows * ncols >= 5
        assert ncols <= 4

    def test_eleven(self):
        """FCN3 has 11 variable groups."""
        nrows, ncols = _grid_shape(11)
        assert nrows * ncols >= 11
        assert ncols <= 4
        assert nrows <= 4


class TestPlotMetricGrid:
    def test_basic(self):
        ds = _create_score_dataset()
        groups = OrderedDict({"t2m": ["t2m"], "z500": ["z500"]})
        fig = plot_metric_grid(ds, "rmse", groups)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_hides_empty_cells(self):
        ds = _create_score_dataset()
        # 3 groups → 1×3 grid, all cells filled, none hidden
        groups = OrderedDict({"a": ["t2m"], "b": ["z500"], "c": ["t2m", "z500"]})
        fig = plot_metric_grid(ds, "rmse", groups)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 3
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_suptitle_uses_display_name(self):
        ds = _create_score_dataset()
        groups = OrderedDict({"t2m": ["t2m"]})
        fig = plot_metric_grid(ds, "rmse", groups)
        assert fig._suptitle.get_text() == "RMSE"
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRenderLeadTimeCurves:
    def test_one_grid_per_metric(self):
        """Each metric produces one composite grid figure."""
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create({"collapsed": True})
        report_cfg = OmegaConf.create({})
        result = render_lead_time_curves(
            ds,
            section_cfg,
            report_cfg,
            groups,
            _EMPTY_CFG,
        )
        assert isinstance(result, SectionOutput)
        # 1 metric (rmse) → 1 composite grid figure
        assert len(result.figures) == 1
        import matplotlib.pyplot as plt

        for fig in result.figures.values():
            plt.close(fig)

    def test_multiple_metrics(self):
        """Each metric gets its own grid figure."""
        ds = _create_score_dataset(include_crps=True)
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create({"collapsed": True})
        report_cfg = OmegaConf.create({})
        result = render_lead_time_curves(
            ds,
            section_cfg,
            report_cfg,
            groups,
            _EMPTY_CFG,
        )
        # rmse + crps → 2 grid figures
        assert len(result.figures) == 2
        import matplotlib.pyplot as plt

        for fig in result.figures.values():
            plt.close(fig)

    def test_grid_has_correct_subplots(self):
        """Grid figure should have one subplot per variable group."""
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create({"collapsed": True})
        report_cfg = OmegaConf.create(
            {
                "variable_groups": {
                    "paired": ["t2m", "z500"],
                }
            }
        )
        result = render_lead_time_curves(
            ds,
            section_cfg,
            report_cfg,
            groups,
            _EMPTY_CFG,
        )
        assert len(result.figures) == 1
        fig = list(result.figures.values())[0]
        # 1 group → 1×1 grid → 1 visible axes
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 1
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_ungrouped_vars_become_subplots(self):
        """Variables not in any explicit group each become a subplot."""
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create({"collapsed": True})
        report_cfg = OmegaConf.create(
            {
                "variable_groups": {
                    "paired": ["t2m"],
                    # z500 is NOT listed → gets its own subplot
                }
            }
        )
        result = render_lead_time_curves(
            ds,
            section_cfg,
            report_cfg,
            groups,
            _EMPTY_CFG,
        )
        assert len(result.figures) == 1  # still 1 figure (grid)
        fig = list(result.figures.values())[0]
        # 2 subplots: "paired" + "z500"
        visible_axes = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible_axes) == 2
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# End-to-end: generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    def test_end_to_end(self, tmp_path):
        """Generate a full report from a synthetic score store."""
        ds = _create_score_dataset(include_crps=True)
        score_path = tmp_path / "out" / "scores.zarr"
        score_path.parent.mkdir(parents=True)
        _write_score_zarr(score_path, ds)

        cfg = OmegaConf.create(
            {
                "run_id": "test_model",
                "output": {"path": str(tmp_path / "out")},
                "scoring": {
                    "output": {"store_name": "scores.zarr"},
                },
                "report": {
                    "sections": [
                        {
                            "type": "summary_table",
                            "lead_times": ["1 days", "5 days"],
                            "collapsed": False,
                        },
                        {"type": "lead_time_curves", "collapsed": True},
                        {
                            "type": "ic_heatmap",
                            "metrics": ["rmse"],
                            "variables": ["t2m"],
                            "collapsed": True,
                        },
                    ],
                    "figure_format": "png",
                },
            }
        )

        report_path = generate_report(cfg)
        assert report_path.exists()
        content = report_path.read_text()
        assert "test_model" in content
        assert "Summary" in content

        # Figures directory should exist with PNGs
        fig_dir = tmp_path / "out" / "report" / "figures"
        assert fig_dir.exists()
        pngs = list(fig_dir.glob("*.png"))
        assert len(pngs) > 0

        # CSV should exist
        csv_path = tmp_path / "out" / "report" / "tables" / "scores_summary.csv"
        assert csv_path.exists()
        df = pd.read_csv(csv_path)
        assert "model" in df.columns
        assert (df["model"] == "test_model").all()

    def test_with_time_groups(self, tmp_path):
        """Report with time groups produces per-group CSVs."""
        ds = _create_score_dataset()
        score_path = tmp_path / "out" / "scores.zarr"
        score_path.parent.mkdir(parents=True)
        _write_score_zarr(score_path, ds)

        cfg = OmegaConf.create(
            {
                "run_id": "test_model",
                "output": {"path": str(tmp_path / "out")},
                "scoring": {
                    "output": {"store_name": "scores.zarr"},
                },
                "report": {
                    "sections": [
                        {
                            "type": "lead_time_curves",
                            "time_groups": True,
                            "collapsed": True,
                        },
                    ],
                    "figure_format": "png",
                    "time_groups": {
                        "winter": [
                            {"start": "2024-01-01", "end": "2024-02-29"},
                        ],
                        "summer": [
                            {"start": "2024-06-01", "end": "2024-08-31"},
                        ],
                    },
                },
            }
        )

        report_path = generate_report(cfg)
        assert report_path.exists()

        # Per-group CSVs
        table_dir = tmp_path / "out" / "report" / "tables"
        assert (table_dir / "scores_summary_winter.csv").exists()
        assert (table_dir / "scores_summary_summer.csv").exists()

    def test_missing_score_store_raises(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "run_id": "missing",
                "output": {"path": str(tmp_path / "nonexistent")},
                "scoring": {"output": {"store_name": "scores.zarr"}},
                "report": {"sections": []},
            }
        )
        with pytest.raises(FileNotFoundError):
            generate_report(cfg)


# ---------------------------------------------------------------------------
# Visualization — prediction vs truth
# ---------------------------------------------------------------------------

# Helpers for creating minimal forecast + verification zarr stores.

VIS_TIMES = np.array(["2024-01-01", "2024-03-01"], dtype="datetime64[ns]")
VIS_LEAD_TIMES = np.array([0, 1, 5], dtype="timedelta64[D]").astype("timedelta64[ns]")
VIS_VARS = ["z500", "t2m"]


def _create_forecast_zarr(path):
    """Write a minimal forecast.zarr."""
    rng = np.random.default_rng(99)
    ds = xr.Dataset()
    for var in VIS_VARS:
        ds[var] = xr.DataArray(
            rng.standard_normal(
                (len(VIS_TIMES), len(VIS_LEAD_TIMES), len(SMALL_LAT), len(SMALL_LON))
            ).astype("float32"),
            dims=["time", "lead_time", "lat", "lon"],
            coords={
                "time": VIS_TIMES,
                "lead_time": VIS_LEAD_TIMES,
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            },
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(path))
    return ds


def _create_verif_zarr(path):
    """Write a minimal data.zarr with all needed valid times."""
    rng = np.random.default_rng(77)
    all_valid = set()
    for t in VIS_TIMES:
        for lt in VIS_LEAD_TIMES:
            all_valid.add(t + lt)
    vtimes = np.array(sorted(all_valid), dtype="datetime64[ns]")

    ds = xr.Dataset()
    for var in VIS_VARS:
        ds[var] = xr.DataArray(
            rng.standard_normal((len(vtimes), len(SMALL_LAT), len(SMALL_LON))).astype(
                "float32"
            ),
            dims=["time", "lat", "lon"],
            coords={"time": vtimes, "lat": SMALL_LAT, "lon": SMALL_LON},
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_zarr(str(path))
    return ds


class TestPlotPredictionVsTruth:
    def _latlon_coords(self):
        return OrderedDict([("lat", SMALL_LAT), ("lon", SMALL_LON)])

    def test_basic(self):
        rng = np.random.default_rng(0)
        pred = rng.standard_normal((8, 16)).astype("float32")
        truth = rng.standard_normal((8, 16)).astype("float32")
        fig = plot_prediction_vs_truth(
            pred, truth, self._latlon_coords(), "z500", "2024-01-01", "5 days"
        )
        # 3 plot axes + 3 colorbar axes = 6
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_title_contains_info(self):
        rng = np.random.default_rng(0)
        pred = rng.standard_normal((8, 16)).astype("float32")
        truth = rng.standard_normal((8, 16)).astype("float32")
        fig = plot_prediction_vs_truth(
            pred, truth, self._latlon_coords(), "z500", "2024-01-01", "5 days"
        )
        suptitle = fig._suptitle.get_text()
        assert "z500" in suptitle
        assert "2024-01-01" in suptitle
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_explicit_color_range_overrides_auto(self):
        """vmin/vmax/diff_abs kwargs override the nanmin/nanmax defaults.

        This is the mechanism that suppresses MRMS refc's large-negative
        fill values from crushing the color scale.
        """
        # Truth contains an outlier fill-value that would normally pull
        # nanmin down to -9999.  With an explicit vmin=0, the colormap is
        # clamped to the physical range.
        truth = np.zeros((8, 16), dtype="float32")
        truth[0, 0] = -9999.0  # fill-value outlier
        pred = np.zeros((8, 16), dtype="float32")
        fig = plot_prediction_vs_truth(
            pred,
            truth,
            self._latlon_coords(),
            "refc",
            "2024-01-01",
            "60 min",
            vmin=0.0,
            vmax=75.0,
            diff_abs=40.0,
        )
        # The prediction pcolormesh should have vmin=0 (not nanmin=-9999).
        pred_ax = fig.axes[0]
        # Find the QuadMesh (pcolormesh output) in the axes' collections.
        mesh = next(c for c in pred_ax.collections if hasattr(c, "get_clim"))
        vmin_actual, vmax_actual = mesh.get_clim()
        assert vmin_actual == pytest.approx(0.0)
        assert vmax_actual == pytest.approx(75.0)
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_show_diff_false_drops_third_panel(self):
        """``show_diff=False`` produces a 2-panel figure (pred, truth)
        instead of the default 3-panel (pred, truth, diff)."""
        pred = np.zeros((8, 16), dtype="float32")
        truth = np.zeros((8, 16), dtype="float32")
        fig = plot_prediction_vs_truth(
            pred,
            truth,
            self._latlon_coords(),
            "z500",
            "2024-01-01",
            "5 days",
            show_diff=False,
        )
        # 2 panels + 2 colorbars = 4 axes; default is 6.
        assert len(fig.axes) == 4
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_yx_coords_with_lambert_projection(self):
        """HRRR-style ``(y, x)`` coords in meters plot without reprojection
        when the data's CRS matches the map projection."""
        pytest.importorskip("cartopy")
        from src.report import _resolve_projection

        projection = _resolve_projection("hrrr_lambert")
        if projection is None:
            pytest.skip("cartopy/LambertConformal unavailable")

        rng = np.random.default_rng(0)
        # HRRR-meter scale y/x values; sizes match pred/truth.
        y = np.linspace(-1.5e6, 1.5e6, 6, dtype="float32")
        x = np.linspace(-2.5e6, 2.5e6, 8, dtype="float32")
        pred = rng.standard_normal((6, 8)).astype("float32")
        truth = rng.standard_normal((6, 8)).astype("float32")
        spatial = OrderedDict([("y", y), ("x", x)])

        fig = plot_prediction_vs_truth(
            pred,
            truth,
            spatial,
            "refc",
            "2023-12-05",
            "60 min",
            projection=projection,
        )
        # 3 pcolormesh axes + 3 colorbars = 6
        assert len(fig.axes) == 6
        import matplotlib.pyplot as plt

        plt.close(fig)


class TestRenderHeaderVisualization:
    def test_basic(self, tmp_path):
        _create_forecast_zarr(tmp_path / "out" / "forecast.zarr")
        _create_verif_zarr(tmp_path / "out" / "data.zarr")

        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create(
            {
                "variable": "z500",
                "lead_time": "1 days",
            }
        )
        report_cfg = OmegaConf.create({})
        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path / "out")},
                "predownload": {"verification": {"enabled": True, "source": None}},
            }
        )

        result = render_header_visualization(
            ds,
            section_cfg,
            report_cfg,
            groups,
            cfg,
        )
        assert len(result.figures) == 1
        assert "Sample Visualization" in result.markdown
        import matplotlib.pyplot as plt

        for fig in result.figures.values():
            plt.close(fig)

    def test_missing_data_omits_section(self, tmp_path):
        """Should produce empty markdown, not crash, when data is missing."""
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create({"variable": "z500"})
        report_cfg = OmegaConf.create({})
        cfg = OmegaConf.create({"output": {"path": str(tmp_path / "empty")}})

        result = render_header_visualization(
            ds,
            section_cfg,
            report_cfg,
            groups,
            cfg,
        )
        assert len(result.figures) == 0
        assert result.markdown == ""


class TestRenderVisualization:
    def test_basic(self, tmp_path):
        _create_forecast_zarr(tmp_path / "out" / "forecast.zarr")
        _create_verif_zarr(tmp_path / "out" / "data.zarr")

        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create(
            {
                "variables": ["z500", "t2m"],
                "lead_times": ["1 days"],
                "collapsed": True,
            }
        )
        report_cfg = OmegaConf.create({})
        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path / "out")},
                "predownload": {"verification": {"enabled": True, "source": None}},
            }
        )

        result = render_visualization(
            ds,
            section_cfg,
            report_cfg,
            groups,
            cfg,
        )
        assert len(result.figures) == 2  # 2 vars × 1 lead time
        assert "<details>" in result.markdown
        import matplotlib.pyplot as plt

        for fig in result.figures.values():
            plt.close(fig)

    def test_missing_data_degrades(self, tmp_path):
        ds = _create_score_dataset()
        groups = parse_score_arrays(ds)
        section_cfg = OmegaConf.create(
            {
                "variables": ["z500"],
                "lead_times": ["1 days"],
                "collapsed": True,
            }
        )
        report_cfg = OmegaConf.create({})
        cfg = OmegaConf.create({"output": {"path": str(tmp_path / "empty")}})

        result = render_visualization(
            ds,
            section_cfg,
            report_cfg,
            groups,
            cfg,
        )
        assert len(result.figures) == 0
        assert "not available" in result.markdown


class TestGenerateReportWithVisualization:
    def test_end_to_end_with_header(self, tmp_path):
        """Full report with header visualization."""
        ds = _create_score_dataset()
        score_path = tmp_path / "out" / "scores.zarr"
        _write_score_zarr(score_path, ds)
        _create_forecast_zarr(tmp_path / "out" / "forecast.zarr")
        _create_verif_zarr(tmp_path / "out" / "data.zarr")

        cfg = OmegaConf.create(
            {
                "run_id": "vis_test",
                "output": {"path": str(tmp_path / "out")},
                "scoring": {"output": {"store_name": "scores.zarr"}},
                "predownload": {"verification": {"enabled": True, "source": None}},
                "report": {
                    "sections": [
                        {
                            "type": "header_visualization",
                            "variable": "z500",
                            "lead_time": "1 days",
                        },
                        {
                            "type": "summary_table",
                            "lead_times": ["1 days", "5 days"],
                            "collapsed": False,
                        },
                        {
                            "type": "visualization",
                            "variables": ["z500", "t2m"],
                            "lead_times": ["1 days", "5 days"],
                            "collapsed": True,
                        },
                    ],
                    "figure_format": "png",
                },
            }
        )

        report_path = generate_report(cfg)
        assert report_path.exists()
        content = report_path.read_text()
        assert "Sample Visualization" in content
        assert "Visualization" in content

        fig_dir = tmp_path / "out" / "report" / "figures"
        header_figs = list(fig_dir.glob("header_*"))
        assert len(header_figs) == 1
        vis_figs = list(fig_dir.glob("vis_*"))
        assert len(vis_figs) == 4  # 2 vars × 2 lead times


# ---------------------------------------------------------------------------
# HRRR / Lambert Conformal & per-model store support
# ---------------------------------------------------------------------------


class TestLeadTimeAxis:
    """``_lead_time_axis`` converts timedelta64 lead times to fractional
    values in the chosen unit and returns a matching axis label.  Used
    to render short-horizon campaigns (StormScope: 60 min, 120 min) in
    hours or minutes instead of the days-compressed default."""

    def test_days_default_unchanged(self):
        from src.report import _lead_time_axis

        lts = np.array(
            [np.timedelta64(1, "D"), np.timedelta64(3, "D")],
            dtype="timedelta64[ns]",
        )
        vals, label = _lead_time_axis(lts, "days")
        np.testing.assert_allclose(vals, [1.0, 3.0])
        assert "days" in label

    def test_hours(self):
        from src.report import _lead_time_axis

        lts = np.array(
            [np.timedelta64(60, "m"), np.timedelta64(120, "m")],
            dtype="timedelta64[ns]",
        )
        vals, label = _lead_time_axis(lts, "hours")
        np.testing.assert_allclose(vals, [1.0, 2.0])
        assert "hours" in label

    def test_minutes(self):
        from src.report import _lead_time_axis

        lts = np.array(
            [np.timedelta64(30, "m"), np.timedelta64(60, "m")],
            dtype="timedelta64[ns]",
        )
        vals, label = _lead_time_axis(lts, "minutes")
        np.testing.assert_allclose(vals, [30.0, 60.0])
        assert "minutes" in label

    def test_aliases(self):
        from src.report import _lead_time_axis

        lts = np.array([np.timedelta64(60, "m")], dtype="timedelta64[ns]")
        for alias in ("h", "hour", "HOURS"):
            vals, _ = _lead_time_axis(lts, alias)
            np.testing.assert_allclose(vals, [1.0])

    def test_unknown_unit_falls_back_to_days(self):
        from src.report import _lead_time_axis

        lts = np.array([np.timedelta64(1, "D")], dtype="timedelta64[ns]")
        vals, label = _lead_time_axis(lts, "fortnights")
        np.testing.assert_allclose(vals, [1.0])
        assert "days" in label


class TestSummaryShortLeadTimes:
    """``snapshot_at_lead_times`` must match the configured lead-time
    strings against the score store's ``timedelta64[ns]`` values even
    when the caller uses short units (``"60 min"``, ``"2 h"``).

    Regression: previously ``np.timedelta64(pd.Timedelta("60 min"))``
    could return a pandas subclass that the downstream membership
    check failed to equate with the store's ns-unit values, leaving
    the summary table empty for StormScope-style campaigns.
    """

    def test_60min_lead_time_lookup(self):
        from src.report import snapshot_at_lead_times

        # Score store with two lead times (60 and 120 min) in ns units.
        lead_times = np.array([60, 120], dtype="timedelta64[m]").astype(
            "timedelta64[ns]"
        )
        times = np.array(["2023-12-05T12:00:00"], dtype="datetime64[ns]")
        ds = xr.Dataset(
            {
                "rmse__refc": xr.DataArray(
                    np.array([[0.5, 1.0]], dtype="float32"),
                    dims=["time", "lead_time"],
                    coords={"time": times, "lead_time": lead_times},
                )
            }
        )
        df = snapshot_at_lead_times(ds, {"rmse": ["refc"]}, ["60 min", "120 min"])
        assert not df.empty
        assert set(df["lead_time"]) == {"60 min", "120 min"}
        # Values should round-trip after the rmse quadratic time aggregation
        # (single time in the store so the time axis is a no-op).
        vals = dict(zip(df["lead_time"], df["value"]))
        np.testing.assert_allclose(vals["60 min"], 0.5)
        np.testing.assert_allclose(vals["120 min"], 1.0)


class TestColorRangeFor:
    """``_color_range_for`` pulls per-variable color range overrides from
    the report config.  Unset or missing entries fall back to an empty
    dict so the plotting call auto-scales from the data."""

    def test_returns_empty_when_no_color_ranges(self):
        from src.report import _color_range_for

        report_cfg = OmegaConf.create({})
        assert _color_range_for(report_cfg, "refc") == {}

    def test_returns_empty_when_variable_absent(self):
        from src.report import _color_range_for

        report_cfg = OmegaConf.create(
            {"color_ranges": {"other_var": {"vmin": 0, "vmax": 1}}}
        )
        assert _color_range_for(report_cfg, "refc") == {}

    def test_returns_defined_keys(self):
        from src.report import _color_range_for

        report_cfg = OmegaConf.create(
            {
                "color_ranges": {
                    "refc": {"vmin": 0, "vmax": 75, "diff_abs": 40},
                }
            }
        )
        out = _color_range_for(report_cfg, "refc")
        assert out == {"vmin": 0.0, "vmax": 75.0, "diff_abs": 40.0}

    def test_omits_null_keys(self):
        """A null/None value in config means 'fall back to auto' — the
        key should be dropped from the kwargs so the plotting default
        (nanmin/nanmax) wins."""
        from src.report import _color_range_for

        report_cfg = OmegaConf.create(
            {
                "color_ranges": {
                    "refc": {"vmin": 0, "vmax": None, "diff_abs": 40},
                }
            }
        )
        out = _color_range_for(report_cfg, "refc")
        assert out == {"vmin": 0.0, "diff_abs": 40.0}


class TestResolveProjection:
    """The report's projection resolver accepts a custom ``hrrr_lambert``
    alias for HRRR's Lambert-conformal CRS alongside the standard cartopy
    names.  Falls back gracefully when cartopy is missing or the name
    is unrecognised."""

    def test_known_name_returns_crs(self):
        pytest.importorskip("cartopy")
        from src.report import _resolve_projection

        proj = _resolve_projection("robinson")
        assert proj is not None

    def test_hrrr_lambert_returns_lambert_conformal(self):
        pytest.importorskip("cartopy")
        from src.report import _resolve_projection

        proj = _resolve_projection("hrrr_lambert")
        assert proj is not None
        # Identify by class name to avoid importing cartopy at module scope.
        assert type(proj).__name__ == "LambertConformal"

    def test_none_returns_none(self):
        from src.report import _resolve_projection

        assert _resolve_projection(None) is None

    def test_unknown_name_returns_none(self):
        pytest.importorskip("cartopy")
        from src.report import _resolve_projection

        assert _resolve_projection("nonexistent_projection") is None


class TestOpenDataStoresMultiStore:
    """``_open_data_stores`` falls back to merging ``data_*.zarr`` stores
    when neither ``verification.zarr`` nor ``data.zarr`` is present —
    the layout StormScope's predownload writes."""

    def _write_yx_zarr(self, path, variables):
        t = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
        y = np.arange(4)
        x = np.arange(5)
        ds = xr.Dataset()
        for v in variables:
            ds[v] = xr.DataArray(
                np.zeros((2, 4, 5), dtype="float32"),
                dims=["time", "y", "x"],
                coords={"time": t, "y": y, "x": x},
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(str(path), mode="w")

    def test_merges_per_model_stores(self, tmp_path):
        from src.report import _open_data_stores

        out = tmp_path / "out"
        out.mkdir()
        # No verification.zarr / data.zarr — just per-model stores.
        self._write_yx_zarr(out / "data_goes.zarr", ["abi01c", "abi02c"])
        self._write_yx_zarr(out / "data_mrms.zarr", ["refc"])
        # Prediction store (forecast.zarr) is optional; omit here to
        # focus the test on verification resolution.
        cfg = OmegaConf.create({"output": {"path": str(out)}})

        _, verif_ds = _open_data_stores(cfg)
        assert verif_ds is not None
        # Merged dataset has variables from both stores.
        assert "abi01c" in verif_ds
        assert "abi02c" in verif_ds
        assert "refc" in verif_ds

    def test_prefers_verification_zarr_when_present(self, tmp_path):
        """When ``verification.zarr`` exists, the per-model fallback is
        not consulted — the legacy path wins."""
        from src.report import _open_data_stores

        out = tmp_path / "out"
        out.mkdir()
        self._write_yx_zarr(out / "verification.zarr", ["legacy_var"])
        self._write_yx_zarr(out / "data_goes.zarr", ["abi01c"])
        cfg = OmegaConf.create({"output": {"path": str(out)}})

        _, verif_ds = _open_data_stores(cfg)
        assert verif_ds is not None
        assert "legacy_var" in verif_ds
        assert "abi01c" not in verif_ds

    def test_returns_none_when_no_stores(self, tmp_path):
        from src.report import _open_data_stores

        out = tmp_path / "out"
        out.mkdir()
        cfg = OmegaConf.create({"output": {"path": str(out)}})

        _, verif_ds = _open_data_stores(cfg)
        assert verif_ds is None


class TestGenerateReportStormScope:
    """Smoke-level end-to-end generator run with a StormScope-shaped
    layout: prediction zarr on ``(y, x)`` HRRR meters, per-model
    verification zarrs, ``hrrr_lambert`` projection, and multi-group
    variable layout."""

    def test_end_to_end_hrrr_layout(self, tmp_path):
        pytest.importorskip("cartopy")

        out = tmp_path / "out"
        out.mkdir()

        # --- Prediction zarr: (time, lead_time, variable, y, x) ---
        ic_times = np.array(["2023-12-05T12:00:00"], dtype="datetime64[ns]")
        lead_times = np.array([60, 120], dtype="timedelta64[m]").astype(
            "timedelta64[ns]"
        )
        y = np.linspace(-1.5e6, 1.5e6, 6, dtype="float32")
        x = np.linspace(-2.5e6, 2.5e6, 8, dtype="float32")

        rng = np.random.default_rng(0)
        pred_ds = xr.Dataset(
            {
                v: xr.DataArray(
                    rng.standard_normal(
                        (len(ic_times), len(lead_times), len(y), len(x))
                    ).astype("float32"),
                    dims=["time", "lead_time", "y", "x"],
                    coords={
                        "time": ic_times,
                        "lead_time": lead_times,
                        "y": y,
                        "x": x,
                    },
                )
                for v in ["abi01c", "refc"]
            }
        )
        pred_ds.to_zarr(str(out / "forecast.zarr"), mode="w")

        # --- Per-model verification zarrs keyed by valid time ---
        valid_times = ic_times + lead_times[-1]  # simple case
        valid_times = np.concatenate([ic_times + lt for lt in lead_times])
        for fname, vs in [
            ("data_goes.zarr", ["abi01c"]),
            ("data_mrms.zarr", ["refc"]),
        ]:
            xr.Dataset(
                {
                    v: xr.DataArray(
                        rng.standard_normal((len(valid_times), len(y), len(x))).astype(
                            "float32"
                        ),
                        dims=["time", "y", "x"],
                        coords={"time": valid_times, "y": y, "x": x},
                    )
                    for v in vs
                }
            ).to_zarr(str(out / fname), mode="w")

        # --- Minimal score zarr ---
        score_ds = xr.Dataset(
            {
                f"rmse__{v}": xr.DataArray(
                    rng.uniform(0.1, 1.0, (len(ic_times), len(lead_times))).astype(
                        "float32"
                    ),
                    dims=["time", "lead_time"],
                    coords={"time": ic_times, "lead_time": lead_times},
                )
                for v in ["abi01c", "refc"]
            }
        )
        score_ds.to_zarr(str(out / "scores.zarr"), mode="w")

        cfg = OmegaConf.create(
            {
                "run_id": "stormscope_test",
                "output": {"path": str(out)},
                "scoring": {"output": {"store_name": "scores.zarr"}},
                "report": {
                    "projection": "hrrr_lambert",
                    "variable_groups": {
                        "goes_abi": ["abi01c"],
                        "mrms": ["refc"],
                    },
                    "sections": [
                        {
                            "type": "header_visualization",
                            "variable": "refc",
                            "lead_time": "120 min",
                        },
                        {
                            "type": "visualization",
                            "variables": ["abi01c", "refc"],
                            "lead_times": ["60 min", "120 min"],
                            "collapsed": True,
                        },
                    ],
                    "figure_format": "png",
                },
            }
        )

        report_path = generate_report(cfg)
        assert report_path.exists()

        fig_dir = out / "report" / "figures"
        # Header figure for refc.
        assert any(p.name.startswith("header_refc") for p in fig_dir.iterdir())
        # 2 variables × 2 lead times = 4 vis_* figures.
        vis_figs = list(fig_dir.glob("vis_*"))
        assert len(vis_figs) == 4

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

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

import dask.array as da
import matplotlib
import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.io import ZarrBackend

matplotlib.use("Agg")

import plot as recipe_plot


def _config(
    tmp_path: Path,
    *,
    init_time: np.datetime64 | None = None,
    tp_semantics: str = "fuxi-hourly-mean",
) -> recipe_plot.PlotConfig:
    """Create a plotting configuration for synthetic data."""
    config = recipe_plot.PlotConfig(
        input=tmp_path / "forecast.zarr",
        output_dir=tmp_path / "plots",
        init_time=init_time,
        max_weeks=2,
        dpi=100,
        t2m_units="kelvin",
        tp_semantics=tp_semantics,
        precipitation_levels=recipe_plot.PRECIPITATION_LEVELS,
        overwrite=False,
    )
    return config


def _forecast(
    *,
    members: int = 5,
    days: int = 14,
    include_day_zero: bool = True,
    times: int = 1,
) -> xr.Dataset:
    """Build an exact global daily FuXi-like forecast fixture."""
    first_day = 0 if include_day_zero else 1
    lead_days = np.arange(first_day, days + 1)
    latitude = np.linspace(90.0, -90.0, 7)
    longitude = np.arange(0.0, 360.0, 30.0)
    member_labels = np.asarray([f"member-{index + 1}" for index in range(members)])
    shape = (members, times, lead_days.size, latitude.size, longitude.size)
    t2m: np.ndarray = np.empty(shape, dtype=np.float32)
    tp: np.ndarray = np.full(shape, 0.001, dtype=np.float32)
    for member in range(members):
        for lead_index, day in enumerate(lead_days):
            t2m[member, :, lead_index] = 273.15 + day + member
    return xr.Dataset(
        {
            "t2m": (("ensemble", "time", "lead_time", "lat", "lon"), t2m),
            "tp": (("ensemble", "time", "lead_time", "lat", "lon"), tp),
        },
        coords={
            "ensemble": member_labels,
            "time": np.datetime64("2020-06-02", "ns")
            + np.arange(times).astype("timedelta64[D]"),
            "lead_time": lead_days.astype("timedelta64[D]"),
            "lat": latitude,
            "lon": longitude,
        },
    )


@pytest.mark.parametrize("members", [1, 3])
def test_read_forecast_from_real_zarr_layout(
    tmp_path: Path,
    members: int,
) -> None:
    input_path = tmp_path / "forecast.zarr"
    dataset = _forecast(members=members, times=2)
    if members == 1:
        dataset = dataset.isel(ensemble=0, drop=True)
    else:
        dataset = dataset.assign_coords(ensemble=np.arange(members))

    coordinates = OrderedDict(
        (dimension, np.asarray(dataset.coords[dimension].values))
        for dimension in dataset["t2m"].dims
    )
    backend = ZarrBackend(str(input_path), backend_kwargs={"overwrite": True})
    backend.add_array(
        coordinates,
        ["t2m", "tp"],
        [torch.from_numpy(dataset[name].values) for name in ("t2m", "tp")],
    )
    backend.root.attrs["forecast_issue_time_utc"] = "2020-06-04T00:00:00Z"

    loaded, attributes = recipe_plot._read_forecast(
        input_path,
        np.datetime64("2020-06-03", "ns"),
        max_weeks=1,
    )

    expected_dims = (
        ("ensemble", "time", "lead_time", "lat", "lon")
        if members > 1
        else ("time", "lead_time", "lat", "lon")
    )
    assert loaded["t2m"].dims == expected_dims
    assert loaded.sizes["time"] == 1
    assert loaded.sizes["lead_time"] == 8
    assert isinstance(loaded["t2m"].data, da.Array)
    np.testing.assert_array_equal(loaded.time, [np.datetime64("2020-06-03")])
    np.testing.assert_array_equal(
        loaded.lead_time,
        np.arange(8).astype("timedelta64[D]"),
    )
    assert attributes["forecast_issue_time_utc"] == "2020-06-04T00:00:00Z"


@pytest.mark.parametrize(
    "include_day_zero,tp_semantics,expected_precipitation",
    [
        (True, "fuxi-hourly-mean", 168.0),
        (False, "daily-total", 7.0),
    ],
)
def test_weekly_products_use_coordinate_days_and_field_semantics(
    tmp_path: Path,
    include_day_zero: bool,
    tp_semantics: str,
    expected_precipitation: float,
) -> None:
    dataset = _forecast(include_day_zero=include_day_zero)
    dataset = dataset.transpose("lon", "lead_time", "ensemble", "time", "lat")
    products = recipe_plot.derive_weekly_products(
        dataset,
        {},
        _config(tmp_path, tp_semantics=tp_semantics),
    )

    np.testing.assert_allclose(
        products.temperature.isel(week=0, ensemble=0),
        4.0,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        products.temperature.isel(week=0, ensemble=4),
        8.0,
        atol=1.0e-5,
    )
    np.testing.assert_allclose(
        products.precipitation,
        expected_precipitation,
        rtol=1.0e-6,
    )
    assert products.lead_ranges == ((1, 7), (8, 14))
    assert products.valid_ranges[0] == (
        np.datetime64("2020-06-03"),
        np.datetime64("2020-06-09"),
    )
    assert products.issue_time == np.datetime64("2020-06-03")


def test_single_member_is_promoted_without_distribution_claims(tmp_path: Path) -> None:
    dataset = _forecast(members=1).isel(ensemble=0, drop=True)

    products = recipe_plot.derive_weekly_products(dataset, {}, _config(tmp_path))

    assert products.temperature.sizes["ensemble"] == 1
    assert products.member_labels == (0,)
    summaries = recipe_plot._member_statistics(
        xr.DataArray([[2.5]], dims=("week", "ensemble"))
    )
    assert summaries == [{"week": 1, "value": 2.5}]


def test_multiple_initializations_require_explicit_selection(tmp_path: Path) -> None:
    dataset = _forecast(times=2)

    with pytest.raises(ValueError, match="pass --init-time"):
        recipe_plot.derive_weekly_products(dataset, {}, _config(tmp_path))

    products = recipe_plot.derive_weekly_products(
        dataset,
        {},
        _config(tmp_path, init_time=np.datetime64("2020-06-03", "ns")),
    )
    assert products.initialization_time == np.datetime64("2020-06-03")
    assert products.issue_time == np.datetime64("2020-06-04")

    with pytest.raises(ValueError, match="selected daily-mean initialization"):
        recipe_plot.derive_weekly_products(
            dataset,
            {"forecast_issue_time_utc": "2020-06-03T00:00:00Z"},
            _config(tmp_path, init_time=np.datetime64("2020-06-03", "ns")),
        )


@pytest.mark.parametrize(
    "mutation,error",
    [
        (
            lambda dataset: dataset.drop_sel(lead_time=np.timedelta64(4, "D")),
            "week 1 is incomplete",
        ),
        (
            lambda dataset: dataset.assign_coords(
                lead_time=np.asarray(dataset.lead_time.values)[::-1]
            ),
            "strictly increasing",
        ),
        (
            lambda dataset: dataset.assign_coords(
                lead_time=np.asarray(dataset.lead_time.values).astype("timedelta64[h]")
                + np.timedelta64(1, "h")
            ),
            "whole days",
        ),
        (
            lambda dataset: dataset.assign_coords(
                lon=np.linspace(0.0, 360.0, dataset.sizes["lon"])
            ),
            "duplicated cyclic endpoint",
        ),
        (
            lambda dataset: dataset.assign_coords(
                lat=np.asarray([90.0, 60.0, 31.0, 0.0, -30.0, -60.0, -90.0])
            ),
            "latitude must be regularly spaced",
        ),
        (
            lambda dataset: dataset.assign_coords(
                lat=np.asarray([90.0, 60.0, 30.0, 0.0, -30.0, -90.0, -60.0])
            ),
            "latitude must be strictly monotonic",
        ),
    ],
)
def test_invalid_coordinates_are_rejected(
    tmp_path: Path,
    mutation: Callable[[xr.Dataset], xr.Dataset],
    error: str,
) -> None:
    invalid = mutation(_forecast())

    with pytest.raises(ValueError, match=error):
        recipe_plot.derive_weekly_products(invalid, {}, _config(tmp_path))


def test_partial_final_week_is_reported_not_averaged(tmp_path: Path) -> None:
    products = recipe_plot.derive_weekly_products(
        _forecast(days=10),
        {},
        _config(tmp_path),
    )

    assert products.temperature.sizes["week"] == 1
    assert products.omitted_lead_days == (8, 9, 10)


def test_generate_plots_writes_valid_pngs_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.input.mkdir()
    monkeypatch.setattr(
        recipe_plot,
        "_read_forecast",
        lambda path, init_time, max_weeks: (_forecast(), {}),
    )
    monkeypatch.setattr(recipe_plot, "_decorate_map", lambda axis: None)

    report = recipe_plot.generate_plots(config)

    assert report["status"] == "passed"
    assert report["ensemble_members"] == 5
    assert report["weeks_plotted"] == 2
    assert report["lead_day_ranges"] == [[1, 7], [8, 14]]
    assert report["input_zarr"] == str(config.input)
    scientific_contract = report["scientific_contract"]
    assert isinstance(scientific_contract, dict)
    assert scientific_contract["tp_hourly_mean_to_daily_factor"] == 24.0
    plots = report["plots"]
    assert isinstance(plots, list)
    assert len(plots) == 2
    assert [plot["path"] for plot in plots] == [
        recipe_plot.WEEKLY_MAP_NAME,
        recipe_plot.MEMBER_DISTRIBUTION_NAME,
    ]
    for name in (
        recipe_plot.WEEKLY_MAP_NAME,
        recipe_plot.MEMBER_DISTRIBUTION_NAME,
        recipe_plot.SUMMARY_NAME,
    ):
        assert (config.output_dir / name).is_file()


def test_existing_plot_requires_explicit_overwrite(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_dir.mkdir()
    (config.output_dir / recipe_plot.WEEKLY_MAP_NAME).touch()

    with pytest.raises(FileExistsError, match="--overwrite"):
        recipe_plot.generate_plots(config)


@pytest.mark.parametrize(
    "argv",
    [
        ["--max-weeks", "7"],
        ["--dpi", "99"],
        ["--precipitation-levels", "1", "1", "10"],
        ["--precipitation-levels", "-1", "5"],
        ["--precipitation-levels", "nan", "5"],
        ["--precipitation-levels", "1", "inf"],
    ],
)
def test_parse_args_rejects_invalid_plot_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        recipe_plot.parse_args(argv)

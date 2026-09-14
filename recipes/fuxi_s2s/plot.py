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

"""Create publication-quality weekly products from a FuXi-S2S Zarr forecast."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cartopy.crs as ccrs  # type: ignore[import-untyped]
import dask.array as da
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
from cartopy.util import add_cyclic_point  # type: ignore[import-untyped]
from loguru import logger
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PRECIPITATION_LEVELS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0)
WEEKLY_MAP_NAME = "fuxi_s2s_weekly_ensemble_mean.png"
MEMBER_DISTRIBUTION_NAME = "fuxi_s2s_ensemble_member_distributions.png"
SUMMARY_NAME = "fuxi_s2s_plot_summary.json"
SUPPORTED_DIMS = {"ensemble", "time", "lead_time", "lat", "lon"}


@dataclass(frozen=True)
class PlotConfig:
    """Validated plotting configuration."""

    input: Path
    output_dir: Path
    init_time: np.datetime64 | None
    max_weeks: int
    dpi: int
    t2m_units: str
    tp_semantics: str
    precipitation_levels: tuple[float, ...]
    overwrite: bool


@dataclass(frozen=True)
class WeeklyProducts:
    """Derived weekly member fields and provenance."""

    temperature: xr.DataArray
    precipitation: xr.DataArray
    initialization_time: np.datetime64
    issue_time: np.datetime64
    lead_ranges: tuple[tuple[int, int], ...]
    valid_ranges: tuple[tuple[np.datetime64, np.datetime64], ...]
    member_labels: tuple[object, ...]
    omitted_lead_days: tuple[int, ...]
    global_grid: bool


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _dpi(value: str) -> int:
    """Parse a practical raster resolution."""
    parsed = _positive_int(value)
    if not 100 <= parsed <= 600:
        raise argparse.ArgumentTypeError("must be between 100 and 600")
    return parsed


def _utc_datetime64(value: str) -> np.datetime64:
    """Parse an ISO-8601 timestamp as a naive UTC datetime64."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(parsed, "ns")


def _precipitation_levels(values: Sequence[str]) -> tuple[float, ...]:
    """Validate monotonically increasing precipitation display levels."""
    levels = tuple(float(value) for value in values)
    if len(levels) < 2 or not np.isfinite(levels).all() or levels[0] < 0.0:
        raise argparse.ArgumentTypeError(
            "--precipitation-levels needs at least two finite non-negative values"
        )
    if any(right <= left for left, right in zip(levels, levels[1:], strict=False)):
        raise argparse.ArgumentTypeError(
            "--precipitation-levels must be strictly increasing"
        )
    return levels


def parse_args(argv: Sequence[str] | None = None) -> PlotConfig:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/fuxi_s2s_forecast.zarr"),
        help="FuXi-S2S Zarr forecast produced by main.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fuxi_s2s_plots"),
    )
    parser.add_argument(
        "--init-time",
        type=_utc_datetime64,
        help="initialization label to select when the store contains multiple times",
    )
    parser.add_argument(
        "--max-weeks",
        type=_positive_int,
        default=6,
        help="maximum number of complete seven-day forecast weeks to plot",
    )
    parser.add_argument("--dpi", type=_dpi, default=200)
    parser.add_argument(
        "--t2m-units",
        choices=("kelvin", "celsius"),
        default="kelvin",
        help="units stored in the t2m field",
    )
    parser.add_argument(
        "--tp-semantics",
        choices=("fuxi-hourly-mean", "daily-total"),
        default="fuxi-hourly-mean",
        help=(
            "FuXi daily mean of hourly accumulations, or an already accumulated "
            "daily total; both are expected in metres"
        ),
    )
    parser.add_argument(
        "--precipitation-levels",
        nargs="+",
        default=[str(value) for value in PRECIPITATION_LEVELS],
        metavar="MM",
        help="fixed weekly precipitation color boundaries; first value is masked",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing plot products",
    )
    args = parser.parse_args(argv)
    try:
        levels = _precipitation_levels(args.precipitation_levels)
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if args.max_weeks > 6:
        parser.error("--max-weeks cannot exceed FuXi-S2S's six-week horizon")
    return PlotConfig(
        input=args.input,
        output_dir=args.output_dir,
        init_time=args.init_time,
        max_weeks=args.max_weeks,
        dpi=args.dpi,
        t2m_units=args.t2m_units,
        tp_semantics=args.tp_semantics,
        precipitation_levels=levels,
        overwrite=args.overwrite,
    )


def _array_dimensions(array: zarr.Array) -> tuple[str, ...]:
    """Read named dimensions from a Zarr v2 or v3 array."""
    dimension_names = getattr(array.metadata, "dimension_names", None)
    if dimension_names is None:
        dimension_names = array.attrs.get("_ARRAY_DIMENSIONS")
    if dimension_names is None or any(name is None for name in dimension_names):
        raise ValueError(f"Zarr array {array.path!r} does not record named dimensions")
    return tuple(str(name) for name in dimension_names)


def _read_forecast(
    path: Path,
    init_time: np.datetime64 | None,
    max_weeks: int,
) -> tuple[xr.Dataset, dict[str, object]]:
    """Open only the selected time and lead range as lazy arrays."""
    if not path.exists():
        raise FileNotFoundError(f"FuXi-S2S Zarr store does not exist: {path}")
    root = zarr.open_group(str(path), mode="r")
    required = {"time", "lead_time", "lat", "lon", "t2m", "tp"}
    missing = sorted(required - set(root.array_keys()))
    if missing:
        available = ", ".join(sorted(root.array_keys()))
        raise ValueError(
            f"Zarr store is missing {', '.join(missing)}; available arrays: {available}"
        )

    coordinate_names = ["time", "lead_time", "lat", "lon"]
    if "ensemble" in root:
        coordinate_names.insert(0, "ensemble")
    coordinate_values = {
        name: (_array_dimensions(root[name]), np.asarray(root[name][:]))
        for name in coordinate_names
    }
    coordinate_dataset = xr.Dataset(coords=coordinate_values)
    _, selected_time = _select_initialization(coordinate_dataset, init_time)
    times = np.asarray(coordinate_dataset["time"].values).astype("datetime64[ns]")
    time_index = int(np.flatnonzero(times == selected_time)[0])
    source_lead_days = _lead_days(coordinate_dataset)
    lead_stop = max(
        1,
        int(np.searchsorted(source_lead_days, max_weeks * 7, side="right")),
    )

    coordinates = dict(coordinate_values)
    time_dimensions, time_values = coordinates["time"]
    coordinates["time"] = (
        time_dimensions,
        time_values[time_index : time_index + 1],
    )
    lead_dimensions, lead_values = coordinates["lead_time"]
    coordinates["lead_time"] = (lead_dimensions, lead_values[:lead_stop])

    variables: dict[str, tuple[tuple[str, ...], da.Array]] = {}
    for name in ("t2m", "tp"):
        array = root[name]
        dimensions = _array_dimensions(array)
        unsupported = set(dimensions) - SUPPORTED_DIMS
        if unsupported:
            raise ValueError(
                f"{name} has unsupported dimensions: {', '.join(sorted(unsupported))}"
            )
        selection = tuple(
            (
                slice(time_index, time_index + 1)
                if dimension == "time"
                else slice(0, lead_stop) if dimension == "lead_time" else slice(None)
            )
            for dimension in dimensions
        )
        variables[name] = (dimensions, da.from_zarr(array)[selection])

    dataset = xr.Dataset(data_vars=variables, coords=coordinates)
    attributes = dict(root.attrs)
    attributes["_plot_source_lead_days"] = tuple(int(day) for day in source_lead_days)
    return dataset, attributes


def _select_initialization(
    dataset: xr.Dataset,
    requested: np.datetime64 | None,
) -> tuple[xr.Dataset, np.datetime64]:
    """Select exactly one initialization without pooling time and members."""
    if "time" not in dataset.coords or dataset["time"].ndim != 1:
        raise ValueError("time must be a one-dimensional coordinate")
    times = np.asarray(dataset["time"].values).astype("datetime64[ns]")
    if np.isnat(times).any() or len(np.unique(times)) != times.size:
        raise ValueError("time values must be finite and unique")
    if requested is None:
        if times.size != 1:
            raise ValueError(
                "Zarr store contains multiple initialization times; pass --init-time"
            )
        index = 0
    else:
        matches = np.flatnonzero(times == requested.astype("datetime64[ns]"))
        if matches.size != 1:
            available = ", ".join(np.datetime_as_string(times, unit="s"))
            raise ValueError(
                f"initialization {requested} was not found; available times: {available}"
            )
        index = int(matches[0])
    return dataset.isel(time=index, drop=True), times[index]


def _validate_grid(dataset: xr.Dataset) -> tuple[xr.Dataset, bool]:
    """Validate and normalize FuXi's global rectilinear grid."""
    for name in ("lat", "lon"):
        if name not in dataset.coords or dataset[name].ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional coordinate")
        values = np.asarray(dataset[name].values, dtype=float)
        if not np.isfinite(values).all() or len(np.unique(values)) != values.size:
            raise ValueError(f"{name} values must be finite and unique")

    latitude = np.asarray(dataset["lat"].values, dtype=float)
    longitude = np.asarray(dataset["lon"].values, dtype=float)
    if latitude.min() < -90.0 or latitude.max() > 90.0:
        raise ValueError("latitude values must lie within [-90, 90]")
    if latitude.size < 2 or longitude.size < 3:
        raise ValueError("at least two latitudes and three longitudes are required")

    latitude_differences = np.diff(latitude)
    if not (np.all(latitude_differences > 0.0) or np.all(latitude_differences < 0.0)):
        raise ValueError("latitude must be strictly monotonic")
    latitude_spacing = float(np.median(np.abs(latitude_differences)))
    if not np.allclose(
        np.abs(latitude_differences),
        latitude_spacing,
        rtol=1.0e-6,
        atol=1.0e-8,
    ):
        raise ValueError("latitude must be regularly spaced")

    longitude_differences = np.diff(longitude)
    if np.all(longitude_differences < 0.0):
        dataset = dataset.sortby("lon")
        longitude = np.asarray(dataset["lon"].values, dtype=float)
        longitude_differences = np.diff(longitude)
    if not np.all(longitude_differences > 0.0):
        raise ValueError("longitude must be strictly monotonic")
    spacing = float(np.median(longitude_differences))
    if not np.allclose(longitude_differences, spacing, rtol=1.0e-6, atol=1.0e-8):
        raise ValueError("longitude must be regularly spaced")
    if np.isclose((longitude[-1] - longitude[0]) % 360.0, 0.0, atol=1.0e-6):
        raise ValueError("longitude contains a duplicated cyclic endpoint")

    global_longitude = np.isclose(spacing * longitude.size, 360.0, atol=1.0e-3)
    global_latitude = latitude.min() <= -89.0 and latitude.max() >= 89.0
    if not global_longitude or not global_latitude:
        raise ValueError("weekly FuXi maps require a near-global rectilinear grid")
    return dataset, True


def _lead_days(dataset: xr.Dataset) -> np.ndarray:
    """Convert lead coordinates to strictly increasing whole days."""
    if "lead_time" not in dataset.coords or dataset["lead_time"].ndim != 1:
        raise ValueError("lead_time must be a one-dimensional coordinate")
    leads = np.asarray(dataset["lead_time"].values).astype("timedelta64[ns]")
    if np.isnat(leads).any():
        raise ValueError("lead_time contains missing values")
    day_ns: np.int64 = np.timedelta64(1, "D").astype("timedelta64[ns]").astype(np.int64)
    lead_ns = leads.astype(np.int64)
    if np.any(lead_ns % day_ns):
        raise ValueError("lead_time values must be whole days")
    days = lead_ns // day_ns
    if np.any(np.diff(days) <= 0):
        raise ValueError("lead_time values must be unique and strictly increasing")
    return days


def _issue_time(
    initialization_time: np.datetime64,
    attributes: dict[str, object],
) -> np.datetime64:
    """Resolve the strict issue time from recipe metadata or daily-mean label."""
    expected = initialization_time + np.timedelta64(1, "D")
    value = attributes.get("forecast_issue_time_utc")
    if isinstance(value, str):
        try:
            issue_time = _utc_datetime64(value)
        except argparse.ArgumentTypeError as error:
            raise ValueError("forecast_issue_time_utc is not valid ISO-8601") from error
        if issue_time != expected.astype("datetime64[ns]"):
            raise ValueError(
                "forecast_issue_time_utc must equal the selected daily-mean "
                "initialization plus one day"
            )
        return issue_time
    return expected


def derive_weekly_products(
    dataset: xr.Dataset,
    attributes: dict[str, object],
    config: PlotConfig,
) -> WeeklyProducts:
    """Derive complete weekly fields using FuXi's documented field semantics."""
    dataset, initialization_time = _select_initialization(dataset, config.init_time)
    dataset, global_grid = _validate_grid(dataset)
    days = _lead_days(dataset)
    day_to_index = {int(day): index for index, day in enumerate(days)}
    maximum_day = int(days.max())
    available_weeks = min(config.max_weeks, maximum_day // 7)
    if available_weeks < 1:
        raise ValueError("at least the complete D+1 through D+7 week is required")

    if "ensemble" not in dataset.dims:
        dataset = dataset.expand_dims(ensemble=np.asarray([0]))
    if dataset.sizes["ensemble"] < 1:
        raise ValueError("ensemble must contain at least one member")
    member_labels = tuple(np.asarray(dataset["ensemble"].values).tolist())

    temperature_weeks: list[xr.DataArray] = []
    precipitation_weeks: list[xr.DataArray] = []
    lead_ranges: list[tuple[int, int]] = []
    valid_ranges: list[tuple[np.datetime64, np.datetime64]] = []
    for week in range(1, available_weeks + 1):
        start = (week - 1) * 7 + 1
        end = week * 7
        required_days = list(range(start, end + 1))
        missing_days = [day for day in required_days if day not in day_to_index]
        if missing_days:
            raise ValueError(
                f"week {week} is incomplete; missing lead days {missing_days}"
            )
        indices = [day_to_index[day] for day in required_days]
        t2m = (
            dataset["t2m"]
            .isel(lead_time=indices)
            .transpose("ensemble", "lead_time", "lat", "lon")
        )
        tp = (
            dataset["tp"]
            .isel(lead_time=indices)
            .transpose("ensemble", "lead_time", "lat", "lon")
        )
        t2m_values = np.asarray(t2m.values)
        tp_values = np.asarray(tp.values)
        if not np.isfinite(t2m_values).all():
            raise ValueError(f"t2m contains non-finite values in week {week}")
        if not np.isfinite(tp_values).all() or np.any(tp_values < 0.0):
            raise ValueError(f"tp contains invalid values in week {week}")
        t2m = t2m.copy(data=t2m_values)
        tp = tp.copy(data=tp_values)

        temperature = t2m.mean("lead_time", skipna=False)
        if config.t2m_units == "kelvin":
            if float(t2m.min()) <= 100.0 or float(t2m.max()) >= 400.0:
                raise ValueError("t2m values are outside the expected Kelvin range")
            temperature = temperature - 273.15
        elif float(t2m.min()) <= -173.15 or float(t2m.max()) >= 126.85:
            raise ValueError("t2m values are outside the expected Celsius range")

        precipitation_factor = (
            24.0 if config.tp_semantics == "fuxi-hourly-mean" else 1.0
        )
        precipitation = (
            tp.sum("lead_time", skipna=False) * precipitation_factor * 1000.0
        )
        temperature_weeks.append(temperature)
        precipitation_weeks.append(precipitation)
        lead_ranges.append((start, end))
        valid_ranges.append(
            (
                initialization_time + np.timedelta64(start, "D"),
                initialization_time + np.timedelta64(end, "D"),
            )
        )

    weeks = np.arange(1, available_weeks + 1)
    weekly_temperature = xr.concat(
        temperature_weeks,
        dim=xr.IndexVariable("week", weeks),
    )
    weekly_precipitation = xr.concat(
        precipitation_weeks,
        dim=xr.IndexVariable("week", weeks),
    )
    expected_dims = ("week", "ensemble", "lat", "lon")
    if weekly_temperature.dims != expected_dims:
        weekly_temperature = weekly_temperature.transpose(*expected_dims)
    if weekly_precipitation.dims != expected_dims:
        weekly_precipitation = weekly_precipitation.transpose(*expected_dims)

    source_days_attribute = attributes.get("_plot_source_lead_days")
    source_days = (
        np.asarray(source_days_attribute, dtype=int)
        if isinstance(source_days_attribute, tuple)
        else days
    )
    plotted_days = set(range(1, available_weeks * 7 + 1))
    omitted_days = tuple(
        int(day) for day in source_days if day > 0 and day not in plotted_days
    )
    if omitted_days:
        logger.warning(
            "Lead days outside the requested complete-week products were omitted: {}",
            list(omitted_days),
        )
    return WeeklyProducts(
        temperature=weekly_temperature,
        precipitation=weekly_precipitation,
        initialization_time=initialization_time,
        issue_time=_issue_time(initialization_time, attributes),
        lead_ranges=tuple(lead_ranges),
        valid_ranges=tuple(valid_ranges),
        member_labels=member_labels,
        omitted_lead_days=omitted_days,
        global_grid=global_grid,
    )


def _date_label(start: np.datetime64, end: np.datetime64) -> str:
    """Return a compact date range for a weekly facet."""
    start_date = datetime.fromisoformat(np.datetime_as_string(start, unit="D"))
    end_date = datetime.fromisoformat(np.datetime_as_string(end, unit="D"))
    if start_date.month == end_date.month:
        return f"{start_date:%d}–{end_date:%d %b}"
    return f"{start_date:%d %b}–{end_date:%d %b}"


def _display_time(value: np.datetime64) -> str:
    """Format a datetime64 for plot annotations."""
    return datetime.fromisoformat(np.datetime_as_string(value, unit="D")).strftime(
        "%d %b %Y"
    )


def _temperature_limits(values: np.ndarray) -> tuple[float, float]:
    """Return rounded robust limits shared by all weekly temperature maps."""
    lower, upper = np.percentile(values, (0.5, 99.5))
    lower = float(np.floor(lower / 5.0) * 5.0)
    upper = float(np.ceil(upper / 5.0) * 5.0)
    if not upper > lower:
        lower -= 1.0
        upper += 1.0
    return lower, upper


def _decorate_map(axis: plt.Axes) -> None:
    """Add restrained geographic context to a map facet."""
    axis.set_global()
    axis.coastlines(resolution="110m", linewidth=0.5, color="#273444")
    axis.gridlines(
        linewidth=0.3,
        color="#66788a",
        alpha=0.28,
        linestyle="--",
    )


def plot_weekly_maps(
    products: WeeklyProducts,
    path: Path,
    dpi: int,
    precipitation_levels: tuple[float, ...],
) -> None:
    """Plot weekly ensemble-mean temperature and accumulated precipitation maps."""
    temperature_mean = products.temperature.mean("ensemble")
    precipitation_mean = products.precipitation.mean("ensemble")
    number_of_weeks = temperature_mean.sizes["week"]
    number_of_columns = min(3, number_of_weeks)
    number_of_blocks = int(np.ceil(number_of_weeks / number_of_columns))
    number_of_rows = 2 * number_of_blocks
    projection = ccrs.Robinson()
    longitude = np.asarray(temperature_mean["lon"].values)
    latitude = np.asarray(temperature_mean["lat"].values)
    temperature_limits = _temperature_limits(temperature_mean.values)

    precipitation_colormap = ListedColormap(
        plt.get_cmap("YlGnBu")(np.linspace(0.12, 1.0, 256)),
        name="fuxi_s2s_precipitation",
    ).with_extremes(bad=(1.0, 1.0, 1.0, 0.0))
    precipitation_norm = BoundaryNorm(
        precipitation_levels,
        precipitation_colormap.N,
        extend="max",
    )

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlecolor": "#16202c",
            "text.color": "#16202c",
            "figure.facecolor": "white",
        }
    ):
        fig = plt.figure(
            figsize=(5.0 * number_of_columns + 0.8, 3.15 * number_of_rows + 1.4),
            layout="constrained",
        )
        grid = fig.add_gridspec(
            number_of_rows,
            number_of_columns + 1,
            width_ratios=[1.0] * number_of_columns + [0.045],
        )
        temperature_images: list[object] = []
        precipitation_images: list[object] = []

        for week_index in range(number_of_weeks):
            block = week_index // number_of_columns
            column = week_index % number_of_columns
            temperature_row = 2 * block
            precipitation_row = temperature_row + 1
            start, end = products.lead_ranges[week_index]
            start_date, end_date = products.valid_ranges[week_index]
            header = (
                f"W{week_index + 1} · D+{start}–{end}\n"
                f"{_date_label(start_date, end_date)}"
            )

            temperature_axis = fig.add_subplot(
                grid[temperature_row, column], projection=projection
            )
            precipitation_axis = fig.add_subplot(
                grid[precipitation_row, column], projection=projection
            )

            temperature_field, cyclic_longitude = add_cyclic_point(
                temperature_mean.isel(week=week_index).values,
                coord=longitude,
            )
            temperature_image = temperature_axis.pcolormesh(
                cyclic_longitude,
                latitude,
                temperature_field,
                transform=ccrs.PlateCarree(),
                cmap="cividis",
                vmin=temperature_limits[0],
                vmax=temperature_limits[1],
                shading="auto",
                rasterized=True,
            )
            temperature_images.append(temperature_image)
            temperature_axis.set_title(header, fontsize=10, fontweight="bold", pad=5)
            temperature_axis.text(
                0.02,
                0.05,
                "T2m · 7-day mean",
                transform=temperature_axis.transAxes,
                fontsize=8,
                fontweight="bold",
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#16202c",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )
            _decorate_map(temperature_axis)

            precipitation = np.ma.masked_less(
                precipitation_mean.isel(week=week_index).values,
                precipitation_levels[0],
            )
            precipitation_field, cyclic_longitude = add_cyclic_point(
                precipitation,
                coord=longitude,
            )
            precipitation_image = precipitation_axis.pcolormesh(
                cyclic_longitude,
                latitude,
                precipitation_field,
                transform=ccrs.PlateCarree(),
                cmap=precipitation_colormap,
                norm=precipitation_norm,
                shading="auto",
                rasterized=True,
            )
            precipitation_images.append(precipitation_image)
            precipitation_axis.text(
                0.02,
                0.05,
                "TP · 7-day total",
                transform=precipitation_axis.transAxes,
                fontsize=8,
                fontweight="bold",
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#075985",
                    "edgecolor": "none",
                    "alpha": 0.86,
                },
            )
            _decorate_map(precipitation_axis)

        for block in range(number_of_blocks):
            temperature_row = 2 * block
            precipitation_row = temperature_row + 1
            temperature_colorbar_axis = fig.add_subplot(grid[temperature_row, -1])
            precipitation_colorbar_axis = fig.add_subplot(grid[precipitation_row, -1])
            fig.colorbar(
                temperature_images[0],
                cax=temperature_colorbar_axis,
                extend="both",
                label="T2m (°C)",
            )
            fig.colorbar(
                precipitation_images[0],
                cax=precipitation_colorbar_axis,
                extend="max",
                ticks=precipitation_levels,
                label="TP (mm / 7 days)",
            )

        member_count = len(products.member_labels)
        mean_label = (
            "member forecast" if member_count == 1 else f"mean of n={member_count}"
        )
        product_title = (
            "FuXi-S2S stochastic-member weekly forecast"
            if member_count == 1
            else "FuXi-S2S raw-ensemble weekly forecast"
        )
        fig.suptitle(
            f"{product_title}\n"
            f"Daily means through {_display_time(products.initialization_time)} · "
            f"issue {_display_time(products.issue_time)} · {mean_label}",
            fontsize=16,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.002,
            "Raw stochastic trajectories; not calibrated or verified against observations. "
            f"White precipitation areas are below {precipitation_levels[0]:g} mm/week.",
            ha="center",
            fontsize=8.5,
            color="#5c6773",
        )
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def _global_member_summaries(
    products: WeeklyProducts,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Calculate member-wise global area-weighted weekly summaries."""
    latitude_weights = xr.DataArray(
        np.cos(np.deg2rad(products.temperature["lat"].values)),
        dims=("lat",),
        coords={"lat": products.temperature["lat"]},
    )
    temperature = products.temperature.weighted(latitude_weights).mean(
        ("lat", "lon"), skipna=False
    )
    precipitation = products.precipitation.weighted(latitude_weights).mean(
        ("lat", "lon"), skipna=False
    )
    return temperature, precipitation


def plot_member_distributions(
    products: WeeklyProducts,
    path: Path,
    dpi: int,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Plot every member's global weekly summary with robust sample statistics."""
    temperature, precipitation = _global_member_summaries(products)
    weeks = np.arange(1, temperature.sizes["week"] + 1)
    member_count = temperature.sizes["ensemble"]
    values_and_styles = (
        (
            np.asarray(temperature.values),
            "Global area-weighted 7-day mean T2m (°C)",
            "#D55E00",
        ),
        (
            np.asarray(precipitation.values),
            "Global area-weighted 7-day accumulated TP (mm)",
            "#009E73",
        ),
    )

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.titlecolor": "#16202c",
            "axes.labelcolor": "#273444",
            "xtick.color": "#425466",
            "ytick.color": "#425466",
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
        }
    ):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(12.5, 8.6),
            sharex=True,
        )
        offsets = np.linspace(-0.16, 0.16, member_count) if member_count > 1 else [0.0]
        legend_handles: list[object] = []
        for axis, (values, ylabel, color) in zip(
            axes,
            values_and_styles,
            strict=True,
        ):
            for member in range(member_count):
                axis.plot(
                    weeks,
                    values[:, member],
                    color="#66788a",
                    alpha=0.25 if member_count > 1 else 0.55,
                    linewidth=0.9,
                    zorder=1,
                )
                axis.scatter(
                    weeks + offsets[member],
                    values[:, member],
                    s=30,
                    color=color,
                    alpha=0.78,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )

            if member_count > 1:
                median = np.median(values, axis=1)
                axis.plot(
                    weeks,
                    median,
                    color=color,
                    linewidth=2.6,
                    marker="o",
                    markersize=4.5,
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    zorder=4,
                )
                if member_count >= 5:
                    lower, upper = np.percentile(values, (25.0, 75.0), axis=1)
                    interval_label = "sample IQR"
                else:
                    lower, upper = values.min(axis=1), values.max(axis=1)
                    interval_label = "member range"
                axis.fill_between(
                    weeks,
                    lower,
                    upper,
                    color=color,
                    alpha=0.16,
                    linewidth=0.0,
                    zorder=2,
                )
            axis.set_ylabel(ylabel, fontsize=10)
            axis.grid(axis="y", color="#cad3dd", linewidth=0.75, alpha=0.7)
            axis.spines[["top", "right"]].set_visible(False)
            axis.margins(x=0.04, y=0.16)

            if not legend_handles:
                legend_handles = [
                    Line2D(
                        [0],
                        [0],
                        color="#66788a",
                        marker="o",
                        markerfacecolor=color,
                        markeredgecolor="white",
                        linewidth=0.9,
                        label=(
                            "stochastic members"
                            if member_count > 1
                            else "stochastic trajectory"
                        ),
                    )
                ]
                if member_count > 1:
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            linewidth=2.6,
                            label="sample median",
                        )
                    )
                    legend_handles.append(
                        Patch(facecolor=color, alpha=0.16, label=interval_label)
                    )

        tick_labels = [
            f"W{week}\n{_date_label(*products.valid_ranges[week - 1])}"
            for week in weeks
        ]
        axes[-1].set_xticks(weeks, tick_labels)
        axes[0].legend(
            handles=legend_handles,
            loc="upper left",
            frameon=False,
            ncols=len(legend_handles),
        )
        if member_count == 1:
            title = (
                "FuXi-S2S stochastic-trajectory weekly summaries\n"
                "One raw generated trajectory; each point is a weekly global summary"
            )
        else:
            title = (
                "FuXi-S2S ensemble-member weekly summaries\n"
                f"Raw stochastic-member sample (n={member_count}); every point is "
                "one generated member"
            )
        fig.suptitle(title, fontsize=15, fontweight="bold")
        fig.subplots_adjust(
            left=0.10,
            right=0.98,
            top=0.86,
            bottom=0.17,
            hspace=0.24,
        )
        fig.text(
            0.5,
            0.025,
            "Global summaries can hide regional differences; these are not calibrated "
            "probabilities, spatial uncertainty, or verification.",
            ha="center",
            fontsize=8.5,
            color="#5c6773",
        )
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return temperature, precipitation


def _plot_metadata(path: Path, relative_to: Path) -> dict[str, object]:
    """Validate a PNG and return lightweight rendering evidence."""
    if not path.is_file() or path.stat().st_size < 20_000:
        raise RuntimeError(f"plot is missing or unexpectedly small: {path}")
    image = np.asarray(mpimg.imread(path))
    height, width = image.shape[:2]
    pixel_standard_deviation = float(image[..., :3].std())
    if width < 1000 or height < 600 or pixel_standard_deviation < 0.05:
        raise RuntimeError(
            f"plot failed visual checks: {path}, size={(width, height)}, "
            f"pixel_std={pixel_standard_deviation:.3f}"
        )
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "size_pixels": [width, height],
        "pixel_standard_deviation": pixel_standard_deviation,
    }


def _member_statistics(values: xr.DataArray) -> list[dict[str, object]]:
    """Summarize each forecast week's raw member values."""
    result: list[dict[str, object]] = []
    for week in range(values.sizes["week"]):
        members = np.asarray(values.isel(week=week).values, dtype=float)
        if members.size == 1:
            result.append({"week": week + 1, "value": float(members[0])})
            continue
        result.append(
            {
                "week": week + 1,
                "minimum": float(members.min()),
                "sample_q25": float(np.percentile(members, 25.0)),
                "sample_median": float(np.median(members)),
                "sample_q75": float(np.percentile(members, 75.0)),
                "maximum": float(members.max()),
            }
        )
    return result


def _json_value(value: object) -> object:
    """Convert NumPy coordinate labels to JSON-compatible values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def generate_plots(config: PlotConfig) -> dict[str, object]:
    """Generate and validate all weekly FuXi-S2S plot products."""
    targets = (
        config.output_dir / WEEKLY_MAP_NAME,
        config.output_dir / MEMBER_DISTRIBUTION_NAME,
        config.output_dir / SUMMARY_NAME,
    )
    existing = [path for path in targets if path.exists()]
    if existing and not config.overwrite:
        raise FileExistsError(
            "plot products already exist; pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading FuXi-S2S forecast from {}", config.input)
    dataset, attributes = _read_forecast(
        config.input,
        config.init_time,
        config.max_weeks,
    )
    products = derive_weekly_products(dataset, attributes, config)
    map_path, distribution_path, summary_path = targets
    plot_weekly_maps(
        products,
        map_path,
        config.dpi,
        config.precipitation_levels,
    )
    global_temperature, global_precipitation = plot_member_distributions(
        products,
        distribution_path,
        config.dpi,
    )

    precipitation_factor = 24.0 if config.tp_semantics == "fuxi-hourly-mean" else 1.0
    report: dict[str, object] = {
        "status": "passed",
        "input_zarr": str(config.input),
        "initialization_daily_mean_utc": (
            np.datetime_as_string(products.initialization_time, unit="s") + "Z"
        ),
        "forecast_issue_time_utc": (
            np.datetime_as_string(products.issue_time, unit="s") + "Z"
        ),
        "ensemble_members": len(products.member_labels),
        "member_labels": [_json_value(value) for value in products.member_labels],
        "global_grid_validated": products.global_grid,
        "weeks_plotted": len(products.lead_ranges),
        "lead_day_ranges": [list(values) for values in products.lead_ranges],
        "valid_date_ranges": [
            [
                np.datetime_as_string(start, unit="D"),
                np.datetime_as_string(end, unit="D"),
            ]
            for start, end in products.valid_ranges
        ],
        "unplotted_lead_days": list(products.omitted_lead_days),
        "scientific_contract": {
            "t2m_input_units": config.t2m_units,
            "t2m_output_units": "degree Celsius",
            "t2m_aggregation": (
                "arithmetic mean of seven daily means per member; member mean for maps"
            ),
            "tp_input_units": "metres",
            "tp_input_semantics": config.tp_semantics,
            "tp_hourly_mean_to_daily_factor": precipitation_factor,
            "tp_metres_to_millimetres_factor": 1000.0,
            "tp_aggregation": (
                "sum of seven daily fields per member; member mean for maps"
            ),
            "missing_data_policy": "reject non-finite values; plot complete weeks only",
            "area_weighting": "cos(latitude) on validated global regular lon-lat grid",
            "spatial_domain": "near-global regular longitude-latitude grid",
            "member_weighting": "equal",
            "precipitation_display_levels_mm": list(config.precipitation_levels),
            "precipitation_display_mask_below_mm": config.precipitation_levels[0],
            "display_mask_applied_to_statistics": False,
            "interpretation": (
                "raw stochastic trajectories; not calibrated probabilities and not "
                "verified against observations"
            ),
        },
        "global_area_weighted_member_statistics": {
            "weekly_t2m_degree_celsius": _member_statistics(global_temperature),
            "weekly_tp_millimetres": _member_statistics(global_precipitation),
        },
        "plots": [
            _plot_metadata(map_path, config.output_dir),
            _plot_metadata(distribution_path, config.output_dir),
        ],
    }
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.success("Weekly maps written to {}", map_path)
    logger.success("Member distributions written to {}", distribution_path)
    logger.success("Plot report written to {}", summary_path)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run the weekly plotting command."""
    generate_plots(parse_args(argv))


if __name__ == "__main__":
    main()

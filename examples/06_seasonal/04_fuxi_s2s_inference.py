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

# %%
"""
Running FuXi-S2S Inference
==========================

Run and validate a six-week FuXi-S2S ensemble from the official prepared sample.

FuXi-S2S predicts global daily means at 1.5-degree resolution. Its initial
condition is two consecutive UTC calendar-day means, which differs from the
instantaneous analysis used by most medium-range models. This example uses the
prepared sample from the official `FuXi-S2S Zenodo record
<https://zenodo.org/records/15718402>`_ so that the input aggregation and
checkpoint can both be tested end to end.

In this example you will learn:

- How to load the FuXi-S2S ONNX checkpoint
- How to reconstruct the official sample in Earth2Studio units
- How to run and validate a stochastic subseasonal ensemble
- How to build six-week temperature and precipitation ensemble-mean maps
- How to visualize the distribution of every ensemble member
- How to write a machine-readable validation report

Run the default six-week, 10-member example with
``uv run examples/06_seasonal/04_fuxi_s2s_inference.py``. The forecast length and
ensemble size can be reduced for a smoke test with ``FUXI_S2S_FORECAST_DAYS`` and
``FUXI_S2S_ENSEMBLE_MEMBERS``.

.. warning::
   The FuXi-S2S checkpoint is licensed under CC BY-NC-ND 4.0. The Zenodo
   record restricts it to research use and prohibits commercial or competition
   use without prior author permission.
"""

# /// script
# dependencies = [
#   "earth2studio[fuxi-s2s] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "cartopy",
#   "matplotlib",
#   "pillow",
# ]
# ///

# %%
# Set Up
# ------
# This example uses:
#
# - Prognostic model: [`earth2studio.models.px.FuXiS2S`][earth2studio.models.px.FuXiS2S].
# - Data source: an in-memory wrapper around the official prepared two-day sample.
# - IO backend: [`earth2studio.io.KVBackend`][earth2studio.io.KVBackend].
#
# A CUDA-capable GPU is required. The model checkpoint is about 2.1 GB.

# %% tags=["e2sg-profile:setup"]
import io
import json
import os
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import xarray as xr
from dotenv import load_dotenv

from earth2studio.io import KVBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import FuXiS2S
from earth2studio.models.px.fuxi_s2s import VARIABLES
from earth2studio.utils.type import TimeArray, VariableArray

load_dotenv()
output_directory = Path(os.environ.get("FUXI_S2S_OUTPUT_DIR", "outputs"))
os.makedirs(output_directory, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA-capable GPU is required for this example")

device = torch.device("cuda:0")
forecast_days = int(os.environ.get("FUXI_S2S_FORECAST_DAYS", "42"))
if forecast_days < 14 or forecast_days > 42 or forecast_days % 7:
    raise ValueError("FUXI_S2S_FORECAST_DAYS must be a multiple of 7 from 14 to 42")
ensemble_members = int(os.environ.get("FUXI_S2S_ENSEMBLE_MEMBERS", "10"))
if ensemble_members < 2:
    raise ValueError("FUXI_S2S_ENSEMBLE_MEMBERS must be at least 2")
random_seed = int(os.environ.get("FUXI_S2S_RANDOM_SEED", "42"))
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
ort.set_seed(random_seed)
total_start = time.perf_counter()


class _PreparedSample:
    def __init__(self, data: xr.DataArray) -> None:
        self.data = data

    def __call__(
        self,
        time: TimeArray,
        variable: VariableArray,
    ) -> xr.DataArray:
        return self.data.sel(time=time, variable=variable)


# %%
# Load the Model
# --------------
# Resolving the package downloads both the ONNX graph and its external weight file.
# Hugging Face's native Xet downloader is used when the files are not already cached.
# The ONNX Runtime provider check prevents an unnoticed CPU fallback.
# ``FUXI_S2S_FORECAST_DAYS`` selects 14--42 days in weekly increments (default: 42),
# ``FUXI_S2S_ENSEMBLE_MEMBERS`` selects at least 2 stochastic members (default: 10),
# ``FUXI_S2S_RANDOM_SEED`` sets the stochastic runtime seed (default: 42), and
# ``FUXI_S2S_OUTPUT_DIR`` changes where the plots and validation report are written.

# %% tags=["e2sg-profile:setup"]
asset_start = time.perf_counter()
model = FuXiS2S.load_model(FuXiS2S.load_default_package()).to(device)
asset_seconds = time.perf_counter() - asset_start

session_start = time.perf_counter()
providers = model._get_ort_session().get_providers()
session_seconds = time.perf_counter() - session_start
if "CUDAExecutionProvider" not in providers:
    raise RuntimeError(f"CUDAExecutionProvider is not active: {providers}")

print(f"Checkpoint resolution: {asset_seconds:.2f} seconds")
print(f"ONNX session creation: {session_seconds:.2f} seconds")
print(f"ONNX Runtime providers: {providers}")
print(f"Stochastic runtime seed: {random_seed}")
print(f"Ensemble members: {ensemble_members}")

# %%
# Prepare the Official Initial Condition
# --------------------------------------
# The archived input is normalized in the checkpoint's native units. We undo the
# normalization, restore Earth2Studio's accumulated-field units, and expose it as a
# small data source. In particular, ``tp`` is converted from ``log1p(mm)`` to metres
# and ``ttr`` from W m\ :sup:`-2` to J m\ :sup:`-2`. Both remain daily means of
# 24 one-hour accumulations rather than 24-hour totals.

# %% tags=["e2sg-profile:setup"]
sample_start = time.perf_counter()
sample_package = Package(
    "https://zenodo.org/records/15718402/files",
    cache_options={
        "cache_storage": Package.default_cache("fuxi_s2s"),
        "same_names": True,
    },
)
with zipfile.ZipFile(sample_package.resolve("data.zip?download=1")) as archive:
    sample_datasets = {}
    for name in ("input", "mean", "std"):
        with archive.open(f"data/{name}.nc") as stream:
            sample_datasets[name] = xr.open_dataset(io.BytesIO(stream.read())).load()
    official_samples = {}
    for variable, file_name in {
        "sst": "sea_surface_temperature.nc",
        "t2m": "2m_temperature.nc",
        "tp": "total_precipitation.nc",
        "ttr": "top_net_thermal_radiation.nc",
        "z500": "geopotential.nc",
    }.items():
        with archive.open(f"data/sample/{file_name}") as stream:
            official_samples[variable] = xr.open_dataarray(
                io.BytesIO(stream.read())
            ).load()

official_names = {
    "u10m": "10u",
    "v10m": "10v",
    "u100m": "100u",
    "v100m": "100v",
}
official_variables = [official_names.get(variable, variable) for variable in VARIABLES]
normalized = sample_datasets["input"]["data"].sel(level=official_variables).values
center = sample_datasets["mean"]["data"].sel(level=official_variables).values
scale = sample_datasets["std"]["data"].sel(level=official_variables).values
initial_values = normalized * scale[None, :, None, None]
initial_values += center[None, :, None, None]

ttr_index = VARIABLES.index("ttr")
tp_index = VARIABLES.index("tp")
sst_index = VARIABLES.index("sst")
initial_values[:, ttr_index] *= 3600.0
initial_values[:, tp_index] = (
    np.expm1(initial_values[:, tp_index]).clip(min=0.0) / 1000.0
)
# The normalized archive stores zero over land, which reconstructs to the SST
# climatological mean. Restore the official NaN land mask required by FuXi-S2S.
official_sst = official_samples["sst"].values[:, 0]
if not np.isnan(official_sst).any() or not np.isfinite(official_sst).any():
    raise RuntimeError("Official SST sample must contain both ocean data and land NaNs")
initial_values[:, sst_index] = official_sst

np.testing.assert_allclose(
    initial_values[:, tp_index],
    official_samples["tp"].values[:, 0],
    rtol=1.0e-5,
    atol=1.0e-7,
)
np.testing.assert_allclose(
    initial_values[:, ttr_index],
    official_samples["ttr"].values[:, 0],
    rtol=1.0e-6,
    atol=0.1,
)
np.testing.assert_allclose(
    initial_values[:, VARIABLES.index("t2m")],
    official_samples["t2m"].values[:, 0],
)
np.testing.assert_allclose(
    initial_values[:, VARIABLES.index("z500")],
    official_samples["z500"].sel(level=500).values,
)
np.testing.assert_array_equal(
    np.isnan(initial_values[:, sst_index]),
    np.isnan(official_sst),
)
np.testing.assert_allclose(
    initial_values[:, sst_index],
    official_sst,
    equal_nan=True,
)

sample_data = xr.DataArray(
    initial_values,
    dims=("time", "variable", "lat", "lon"),
    coords={
        "time": sample_datasets["input"]["time"].values,
        "variable": np.array(VARIABLES),
        "lat": sample_datasets["input"]["lat"].values,
        "lon": sample_datasets["input"]["lon"].values,
    },
)
data = _PreparedSample(sample_data)
forecast_time = np.datetime_as_string(sample_data["time"].values[-1], unit="D")
sample_seconds = time.perf_counter() - sample_start
print(f"Forecast initialization: {forecast_time}")
print(f"Initial-condition preparation: {sample_seconds:.2f} seconds")

# %%
# Run and Validate the Forecast
# -----------------------------
# The official FuXi-S2S graph samples a stochastic perturbation internally. The
# ensemble workflow therefore uses a zero initial-condition perturbation: each member
# starts from the same analysis, while independent ONNX calls produce distinct
# stochastic trajectories. We retain only ``t2m`` and ``tp`` in memory while the model
# keeps its complete rolling state.

# %% tags=["e2sg-profile:inference"]
import earth2studio.run as run
from earth2studio.perturbation import Zero

output_coords = OrderedDict({"variable": np.array(["t2m", "tp"])})
io_backend = KVBackend()

torch.cuda.synchronize(device)
inference_start = time.perf_counter()
io_backend = run.ensemble(
    [forecast_time],
    forecast_days,
    ensemble_members,
    model,
    data,
    io_backend,
    Zero(),
    batch_size=1,
    output_coords=output_coords,
    device=device,
)
torch.cuda.synchronize(device)
inference_seconds = time.perf_counter() - inference_start

forecast = io_backend.to_xarray()
expected_variables = ("t2m", "tp")
expected_dims = ("ensemble", "time", "lead_time", "lat", "lon")
expected_shape = (ensemble_members, 1, forecast_days + 1, 121, 240)
if tuple(forecast.data_vars) != expected_variables:
    raise RuntimeError(
        f"Unexpected output variables: {tuple(forecast.data_vars)}, "
        f"expected {expected_variables}"
    )
for variable in expected_variables:
    if forecast[variable].dims != expected_dims:
        raise RuntimeError(
            f"Unexpected {variable} dimensions: {forecast[variable].dims}, "
            f"expected {expected_dims}"
        )
    if forecast[variable].shape != expected_shape:
        raise RuntimeError(
            f"Unexpected {variable} shape: {forecast[variable].shape}, "
            f"expected {expected_shape}"
        )
    if forecast[variable].dtype != np.float32:
        raise RuntimeError(
            f"Unexpected {variable} dtype: {forecast[variable].dtype}, expected float32"
        )

model_coords = model.input_coords()
np.testing.assert_array_equal(forecast["lat"].values, model_coords["lat"])
np.testing.assert_array_equal(forecast["lon"].values, model_coords["lon"])
np.testing.assert_array_equal(
    forecast["time"].values,
    np.array([np.datetime64(forecast_time)]),
)
expected_leads: np.ndarray = np.arange(forecast_days + 1).astype("timedelta64[D]")
np.testing.assert_array_equal(forecast["lead_time"].values, expected_leads)

if not np.isfinite(forecast[list(expected_variables)].to_array()).all():
    raise RuntimeError("FuXi-S2S produced non-finite values")
temperature = forecast["t2m"].values
precipitation = forecast["tp"].values
if not ((temperature > 100.0) & (temperature < 400.0)).all():
    raise RuntimeError("FuXi-S2S produced implausible 2-m temperatures")
if not ((precipitation >= 0.0) & (precipitation < 1.1)).all():
    raise RuntimeError("FuXi-S2S produced implausible daily precipitation")

latest_sample = sample_data.sel(time=np.datetime64(forecast_time))
for variable in expected_variables:
    np.testing.assert_allclose(
        forecast[variable].isel(time=0, lead_time=0).values,
        np.broadcast_to(
            latest_sample.sel(variable=variable).values,
            (ensemble_members, 121, 240),
        ),
        rtol=1.0e-6,
        atol=1.0e-7,
    )

minimum_spatial_std = {"t2m": 1.0, "tp": 1.0e-8}
for variable, minimum in minimum_spatial_std.items():
    final_std = float(
        forecast[variable].isel(time=0, lead_time=-1).std(("lat", "lon")).mean()
    )
    if final_std <= minimum:
        raise RuntimeError(
            f"{variable} has insufficient spatial variation: {final_std} <= {minimum}"
        )

ensemble_spread = {
    variable: float(
        forecast[variable]
        .isel(time=0, lead_time=-1)
        .std("ensemble")
        .mean(("lat", "lon"))
    )
    for variable in expected_variables
}
if ensemble_spread["t2m"] <= 1.0e-5 or ensemble_spread["tp"] <= 1.0e-10:
    raise RuntimeError(f"FuXi-S2S ensemble members are not distinct: {ensemble_spread}")

temperature_change = forecast["t2m"].isel(time=0, lead_time=-1) - forecast["t2m"].isel(
    time=0, lead_time=0
)
if float(np.abs(temperature_change).mean().values) <= 0.01:
    raise RuntimeError("FuXi-S2S temperature does not evolve across the forecast")

variable_metadata = {
    "t2m": ("K", "daily mean"),
    "tp": ("m", "daily mean of 24 one-hour accumulations"),
}
validation_statistics = {
    variable: {
        "units": variable_metadata[variable][0],
        "temporal_semantics": variable_metadata[variable][1],
        "all_leads_grid_min": float(forecast[variable].min().values),
        "all_leads_grid_mean_unweighted": float(forecast[variable].mean().values),
        "all_leads_grid_max": float(forecast[variable].max().values),
        "final_lead_grid_std_unweighted": float(
            forecast[variable]
            .isel(time=0, lead_time=-1)
            .std(("lat", "lon"))
            .mean()
            .values
        ),
        "final_lead_ensemble_spread_grid_mean_unweighted": ensemble_spread[variable],
    }
    for variable in expected_variables
}
print(
    f"{ensemble_members}-member, {forecast_days}-day inference: "
    f"{inference_seconds:.2f} seconds"
)
print(forecast)
print("Validation statistics (grid points are unweighted):")
print(json.dumps(validation_statistics, indent=2))

# %%
# Plot Weekly Ensemble Products
# -----------------------------
# Subseasonal forecasts are usually interpreted as weekly ensemble products. The
# first figure maps the ensemble-mean 2-m temperature and accumulated precipitation
# for every available week. The second figure shows the distribution of global,
# area-weighted weekly summaries across every generated member. These are raw
# stochastic samples, not calibrated probabilities or verification against
# observations.

# %% tags=["e2sg-profile:plotting"]
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image

plot_start = time.perf_counter()
plt.close("all")
projection = ccrs.Robinson()
number_of_weeks = forecast_days // 7
week_numbers = np.arange(1, number_of_weeks + 1)
initialization_date = np.datetime64(forecast_time)
latitude_weights = xr.DataArray(
    np.cos(np.deg2rad(forecast["lat"].values)),
    dims=("lat",),
    coords={"lat": forecast["lat"]},
)

weekly_temperature_members = []
weekly_precipitation_members = []
week_date_ranges = []
for week_number in week_numbers:
    week_start = int((week_number - 1) * 7 + 1)
    week_end = int(week_number * 7)
    week = slice(week_start, week_end + 1)
    weekly_temperature_members.append(
        forecast["t2m"].isel(time=0, lead_time=week).mean("lead_time") - 273.15
    )
    # FuXi-S2S ``tp`` is the daily mean of 24 one-hour accumulations. Recover each
    # day's total with the factor of 24 before summing the seven daily predictions.
    weekly_precipitation_members.append(
        forecast["tp"].isel(time=0, lead_time=week).sum("lead_time") * 24.0 * 1000.0
    )
    week_date_ranges.append(
        (
            np.datetime_as_string(
                initialization_date + np.timedelta64(week_start, "D"), unit="D"
            ),
            np.datetime_as_string(
                initialization_date + np.timedelta64(week_end, "D"), unit="D"
            ),
        )
    )

weekly_temperature = xr.concat(
    weekly_temperature_members,
    dim=xr.IndexVariable("week", week_numbers),
)
weekly_precipitation = xr.concat(
    weekly_precipitation_members,
    dim=xr.IndexVariable("week", week_numbers),
)
if weekly_temperature.dims != ("week", "ensemble", "lat", "lon"):
    raise RuntimeError(
        f"Unexpected weekly temperature dimensions: {weekly_temperature.dims}"
    )
if weekly_precipitation.dims != ("week", "ensemble", "lat", "lon"):
    raise RuntimeError(
        f"Unexpected weekly precipitation dimensions: {weekly_precipitation.dims}"
    )
if not np.isfinite(weekly_temperature.values).all():
    raise RuntimeError("Weekly temperature products contain non-finite values")
if (
    not np.isfinite(weekly_precipitation.values).all()
    or not (weekly_precipitation.values >= 0.0).all()
):
    raise RuntimeError("Weekly precipitation products contain invalid values")

weekly_temperature_ensemble_mean = weekly_temperature.mean("ensemble")
weekly_precipitation_ensemble_mean = weekly_precipitation.mean("ensemble")

fig, axes = plt.subplots(
    2,
    number_of_weeks,
    figsize=(3.5 * number_of_weeks, 7.2),
    layout="constrained",
    subplot_kw={"projection": projection},
    squeeze=False,
)

temperature_limits = np.nanpercentile(
    weekly_temperature_ensemble_mean.values,
    (0.5, 99.5),
)
temperature_limits = np.array(
    [
        np.floor(temperature_limits[0] / 5.0) * 5.0,
        np.ceil(temperature_limits[1] / 5.0) * 5.0,
    ]
)

temperature_image = None
for week_index, ax in enumerate(axes[0]):
    field = weekly_temperature_ensemble_mean.isel(week=week_index).values
    cyclic_field, cyclic_lon = add_cyclic_point(
        field,
        coord=forecast["lon"].values,
    )
    temperature_image = ax.pcolormesh(
        cyclic_lon,
        forecast["lat"].values,
        cyclic_field,
        transform=ccrs.PlateCarree(),
        cmap="cividis",
        vmin=temperature_limits[0],
        vmax=temperature_limits[1],
        shading="auto",
        rasterized=True,
    )
    week_start = week_index * 7 + 1
    week_end = (week_index + 1) * 7
    start_date, end_date = week_date_ranges[week_index]
    ax.set_title(
        f"Week {week_index + 1} (D+{week_start}–{week_end})\n"
        f"{start_date} to {end_date}",
        fontsize=9,
    )
    ax.coastlines(linewidth=0.45)
    ax.gridlines(linewidth=0.25, alpha=0.4)

if temperature_image is None:
    raise RuntimeError("No temperature product was plotted")
temperature_colorbar = fig.colorbar(
    temperature_image,
    ax=list(axes[0]),
    orientation="horizontal",
    shrink=0.7,
    pad=0.02,
    extend="both",
)
temperature_colorbar.set_label("Ensemble-mean 2-m temperature (°C)")

precipitation_levels = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0])
precipitation_display_threshold = precipitation_levels[0]
displayed_precipitation_count = int(
    np.count_nonzero(
        weekly_precipitation_ensemble_mean.values >= precipitation_display_threshold
    )
)
if displayed_precipitation_count == 0:
    raise RuntimeError("Weekly precipitation has no values at or above 1 mm")
# Skip the nearly white end of YlGnBu so the first visible 1--5 mm bin remains
# distinguishable from masked trace precipitation.
precipitation_colormap = ListedColormap(
    plt.get_cmap("YlGnBu")(np.linspace(0.12, 1.0, 256)),
    name="fuxi_s2s_precipitation",
)
precipitation_colormap.set_bad((1.0, 1.0, 1.0, 0.0))
precipitation_norm = BoundaryNorm(
    precipitation_levels,
    precipitation_colormap.N,
    extend="max",
)
precipitation_image = None
for week_index, ax in enumerate(axes[1]):
    masked_precipitation = np.ma.masked_less(
        weekly_precipitation_ensemble_mean.isel(week=week_index).values,
        precipitation_display_threshold,
    )
    cyclic_precipitation, cyclic_lon = add_cyclic_point(
        masked_precipitation,
        coord=forecast["lon"].values,
    )
    precipitation_image = ax.pcolormesh(
        cyclic_lon,
        forecast["lat"].values,
        cyclic_precipitation,
        transform=ccrs.PlateCarree(),
        cmap=precipitation_colormap,
        norm=precipitation_norm,
        shading="auto",
        rasterized=True,
    )
    week_start = week_index * 7 + 1
    week_end = (week_index + 1) * 7
    ax.set_title(
        f"Week {week_index + 1} precipitation (D+{week_start}–{week_end})",
        fontsize=9,
    )
    ax.coastlines(linewidth=0.45)
    ax.gridlines(linewidth=0.25, alpha=0.4)

if precipitation_image is None:
    raise RuntimeError("No precipitation product was plotted")
precipitation_colorbar = fig.colorbar(
    precipitation_image,
    ax=list(axes[1]),
    orientation="horizontal",
    shrink=0.7,
    pad=0.02,
    extend="max",
    ticks=precipitation_levels,
)
precipitation_colorbar.set_label(
    "Ensemble-mean 7-day accumulation (mm); values below 1 mm masked"
)

fig.suptitle(
    "FuXi-S2S weekly ensemble-mean forecast\n"
    f"{ensemble_members} stochastic members (seed {random_seed}); "
    "not verified against observations"
)
weekly_maps_path = output_directory / "04_fuxi_s2s_weekly_ensemble_mean.png"
fig.savefig(weekly_maps_path, dpi=180, bbox_inches="tight")

# Global area-weighted summaries make the ensemble distribution legible in a compact
# figure. Every dot is one stochastic member; boxes show the median and interquartile
# range. These distributions describe the generated ensemble only.
global_weekly_temperature = weekly_temperature.weighted(latitude_weights).mean(
    ("lat", "lon")
)
global_weekly_precipitation = weekly_precipitation.weighted(latitude_weights).mean(
    ("lat", "lon")
)
distribution_values = (
    (global_weekly_temperature.values, "2-m temperature (°C)", "#d95f02"),
    (global_weekly_precipitation.values, "7-day precipitation (mm)", "#1b9e77"),
)

distribution_figure, distribution_axes = plt.subplots(
    2,
    1,
    figsize=(10, 8),
    layout="constrained",
    sharex=True,
)
member_offsets = np.linspace(-0.18, 0.18, ensemble_members)
for ax, (values, ylabel, color) in zip(distribution_axes, distribution_values):
    if values.shape != (number_of_weeks, ensemble_members):
        raise RuntimeError(f"Unexpected member-distribution shape: {values.shape}")
    ax.boxplot(
        [values[index] for index in range(number_of_weeks)],
        positions=week_numbers,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": color, "alpha": 0.22, "edgecolor": color},
        medianprops={"color": "#202020", "linewidth": 1.8},
        whiskerprops={"color": color},
        capprops={"color": color},
    )
    for member in range(ensemble_members):
        ax.plot(
            week_numbers + member_offsets[member],
            values[:, member],
            color="#555555",
            alpha=0.2,
            linewidth=0.65,
            zorder=1,
        )
        ax.scatter(
            week_numbers + member_offsets[member],
            values[:, member],
            s=20,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
            zorder=2,
        )
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.7, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

distribution_axes[-1].set_xlabel("Forecast week")
distribution_axes[-1].set_xticks(
    week_numbers,
    [f"Week {week}" for week in week_numbers],
)
distribution_figure.suptitle(
    "FuXi-S2S ensemble-member distributions\n"
    "Global area-weighted weekly summaries; boxes show median and IQR\n"
    f"All {ensemble_members} stochastic members shown (seed {random_seed}); "
    "not calibrated or verified against observations",
    fontsize=12,
)
ensemble_distribution_path = (
    output_directory / "04_fuxi_s2s_ensemble_member_distributions.png"
)
distribution_figure.savefig(ensemble_distribution_path, dpi=200, bbox_inches="tight")

product_statistics = {
    "weekly_t2m": {
        "units": "degC",
        "aggregation": "7-day mean, then ensemble mean for maps",
        "global_area_weighted_member_summary": [
            {
                "week": int(week),
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
            }
            for week, values in zip(week_numbers, global_weekly_temperature.values)
        ],
    },
    "weekly_tp": {
        "units": "mm",
        "aggregation": (
            "24 hourly accumulations per day summed across 7 days, then ensemble "
            "mean for maps"
        ),
        "global_area_weighted_member_summary": [
            {
                "week": int(week),
                "minimum": float(values.min()),
                "median": float(np.median(values)),
                "maximum": float(values.max()),
            }
            for week, values in zip(
                week_numbers,
                global_weekly_precipitation.values,
            )
        ],
        "ensemble_mean_grid_fraction_at_or_above_1_mm": float(
            displayed_precipitation_count
            / weekly_precipitation_ensemble_mean.values.size
        ),
    },
}


def _validate_plot(path: Path) -> dict[str, int | float | list[int]]:
    if not path.is_file() or path.stat().st_size < 20_000:
        raise RuntimeError(f"Plot is missing or unexpectedly small: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
        width, height = image.size
    pixel_std = float(rgb.std())
    if width < 1000 or height < 500 or pixel_std < 10.0:
        raise RuntimeError(
            f"Plot failed visual smoke checks: {path}, "
            f"size={(width, height)}, pixel_std={pixel_std:.2f}"
        )
    return {
        "bytes": path.stat().st_size,
        "size_pixels": [width, height],
        "pixel_std": pixel_std,
    }


plot_metadata = [
    {
        "relative_path": path.name,
        **_validate_plot(path),
    }
    for path in (weekly_maps_path, ensemble_distribution_path)
]
plot_seconds = time.perf_counter() - plot_start
total_seconds = time.perf_counter() - total_start

report = {
    "status": "passed",
    "forecast_time": forecast_time,
    "forecast_days": forecast_days,
    "ensemble_members": ensemble_members,
    "stochastic_runtime_seed": random_seed,
    "providers": providers,
    "timings_seconds": {
        "checkpoint_resolution": asset_seconds,
        "session_creation": session_seconds,
        "initial_condition_preparation": sample_seconds,
        "inference": inference_seconds,
        "plotting_and_plot_validation": plot_seconds,
        "total": total_seconds,
    },
    "variables": validation_statistics,
    "products": product_statistics,
    "plots": plot_metadata,
}
report_path = output_directory / "04_fuxi_s2s_validation.json"
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print(f"Plotting: {plot_seconds:.2f} seconds")
print(f"Total runtime: {total_seconds:.2f} seconds")
print(f"Saved weekly ensemble-mean maps to {weekly_maps_path}")
print(f"Saved ensemble-member distributions to {ensemble_distribution_path}")
print(f"Saved validation report to {report_path}")

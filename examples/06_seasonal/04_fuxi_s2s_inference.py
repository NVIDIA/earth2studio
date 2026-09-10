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

Run and validate a 14-day FuXi-S2S forecast from the official prepared sample.

FuXi-S2S predicts global daily means at 1.5-degree resolution. Its initial
condition is two consecutive UTC calendar-day means, which differs from the
instantaneous analysis used by most medium-range models. This example uses the
prepared sample from the official `FuXi-S2S Zenodo record
<https://zenodo.org/records/15718402>`_ so that the input aggregation and
checkpoint can both be tested end to end.

In this example you will learn:

- How to load the FuXi-S2S ONNX checkpoint
- How to reconstruct the official sample in Earth2Studio units
- How to run and validate a subseasonal forecast
- How to build weekly temperature, circulation, and precipitation products
- How to write a machine-readable validation report

Run the default two-week smoke test with
``uv run examples/06_seasonal/04_fuxi_s2s_inference.py``. Set
``FUXI_S2S_FORECAST_DAYS=42`` for the full six-week horizon; the map then shows
week 6 and its change from week 5.

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
forecast_days = int(os.environ.get("FUXI_S2S_FORECAST_DAYS", "14"))
if forecast_days < 14 or forecast_days > 42 or forecast_days % 7:
    raise ValueError("FUXI_S2S_FORECAST_DAYS must be a multiple of 7 from 14 to 42")
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
# ``FUXI_S2S_FORECAST_DAYS`` selects 14--42 days in weekly increments (default: 14),
# ``FUXI_S2S_RANDOM_SEED`` sets the stochastic runtime seed (default: 42), and
# ``FUXI_S2S_OUTPUT_DIR`` changes where the plot and validation report are written.

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
# The deterministic workflow refers to a single forecast trajectory; the official
# FuXi-S2S graph samples a stochastic perturbation internally. We retain only ``t2m``,
# ``tp``, and ``z500`` in memory while the model keeps its complete rolling state.

# %% tags=["e2sg-profile:inference"]
import earth2studio.run as run

output_coords = OrderedDict({"variable": np.array(["t2m", "tp", "z500"])})
io_backend = KVBackend()

torch.cuda.synchronize(device)
inference_start = time.perf_counter()
io_backend = run.deterministic(
    [forecast_time],
    forecast_days,
    model,
    data,
    io_backend,
    output_coords=output_coords,
    device=device,
)
torch.cuda.synchronize(device)
inference_seconds = time.perf_counter() - inference_start

forecast = io_backend.to_xarray()
expected_variables = ("t2m", "tp", "z500")
expected_dims = ("time", "lead_time", "lat", "lon")
expected_shape = (1, forecast_days + 1, 121, 240)
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
expected_leads = np.arange(forecast_days + 1).astype("timedelta64[D]")
np.testing.assert_array_equal(forecast["lead_time"].values, expected_leads)

if not np.isfinite(forecast[list(expected_variables)].to_array()).all():
    raise RuntimeError("FuXi-S2S produced non-finite values")
temperature = forecast["t2m"].values
precipitation = forecast["tp"].values
height_500 = forecast["z500"].values / 9.80665
if not ((temperature > 100.0) & (temperature < 400.0)).all():
    raise RuntimeError("FuXi-S2S produced implausible 2-m temperatures")
if not ((precipitation >= 0.0) & (precipitation < 1.1)).all():
    raise RuntimeError("FuXi-S2S produced implausible daily precipitation")
if not ((height_500 > 3000.0) & (height_500 < 7000.0)).all():
    raise RuntimeError("FuXi-S2S produced implausible 500-hPa heights")

latest_sample = sample_data.sel(time=np.datetime64(forecast_time))
for variable in expected_variables:
    np.testing.assert_allclose(
        forecast[variable].isel(time=0, lead_time=0).values,
        latest_sample.sel(variable=variable).values,
        rtol=1.0e-6,
        atol=1.0e-7,
    )

minimum_spatial_std = {"t2m": 1.0, "tp": 1.0e-8, "z500": 100.0}
for variable, minimum in minimum_spatial_std.items():
    final_std = float(forecast[variable].isel(time=0, lead_time=-1).std().values)
    if final_std <= minimum:
        raise RuntimeError(
            f"{variable} has insufficient spatial variation: {final_std} <= {minimum}"
        )

temperature_change = forecast["t2m"].isel(time=0, lead_time=-1) - forecast["t2m"].isel(
    time=0, lead_time=0
)
if float(np.abs(temperature_change).mean().values) <= 0.01:
    raise RuntimeError("FuXi-S2S temperature does not evolve across the forecast")

variable_metadata = {
    "t2m": ("K", "daily mean"),
    "tp": ("m", "daily mean of 24 one-hour accumulations"),
    "z500": ("m^2 s^-2", "daily mean"),
}
validation_statistics = {
    variable: {
        "units": variable_metadata[variable][0],
        "temporal_semantics": variable_metadata[variable][1],
        "all_leads_grid_min": float(forecast[variable].min().values),
        "all_leads_grid_mean_unweighted": float(forecast[variable].mean().values),
        "all_leads_grid_max": float(forecast[variable].max().values),
        "final_lead_grid_std_unweighted": float(
            forecast[variable].isel(time=0, lead_time=-1).std().values
        ),
    }
    for variable in expected_variables
}
print(f"{forecast_days}-day inference: {inference_seconds:.2f} seconds")
print(forecast)
print("Validation statistics (grid points are unweighted):")
print(json.dumps(validation_statistics, indent=2))

# %%
# Plot the Forecast
# -----------------
# Subseasonal forecasts are usually interpreted as weekly products rather than as a
# sequence of individual weather maps. The first figure therefore compares the
# initialized daily mean with the final forecast week's mean temperature, overlays
# the corresponding 500-hPa circulation, shows the week-to-week temperature change,
# and maps the final week's accumulated precipitation.
# This is one stochastic trajectory and is not verification against observations.

# %% tags=["e2sg-profile:plotting"]
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from PIL import Image

plot_start = time.perf_counter()
plt.close("all")
projection = ccrs.Robinson()
fig, axes = plt.subplots(
    2,
    2,
    figsize=(15, 9),
    layout="constrained",
    subplot_kw={"projection": projection},
)

target_week_number = forecast_days // 7
previous_week_number = target_week_number - 1
previous_week_start = forecast_days - 13
previous_week_end = forecast_days - 7
target_week_start = forecast_days - 6
previous_week = slice(previous_week_start, previous_week_end + 1)
target_week = slice(target_week_start, forecast_days + 1)
initialization_date = np.datetime64(forecast_time)
previous_week_temperature = (
    forecast["t2m"].isel(time=0, lead_time=previous_week).mean("lead_time").values
    - 273.15
)
target_week_temperature = (
    forecast["t2m"].isel(time=0, lead_time=target_week).mean("lead_time").values
    - 273.15
)
temperature_products = (
    (
        f"Initialization daily mean\n{forecast_time}",
        forecast["t2m"].isel(time=0, lead_time=0).values - 273.15,
        forecast["z500"].isel(time=0, lead_time=0).values / 9.80665,
    ),
    (
        f"Week {target_week_number} mean (D+{target_week_start}–{forecast_days})\n"
        f"{np.datetime_as_string(initialization_date + np.timedelta64(target_week_start, 'D'), unit='D')}"
        " to "
        f"{np.datetime_as_string(initialization_date + np.timedelta64(forecast_days, 'D'), unit='D')}",
        target_week_temperature,
        forecast["z500"].isel(time=0, lead_time=target_week).mean("lead_time").values
        / 9.80665,
    ),
)
temperature_limits = np.nanpercentile(
    np.stack([product[1] for product in temperature_products]),
    (0.5, 99.5),
)
temperature_limits = np.array(
    [
        np.floor(temperature_limits[0] / 5.0) * 5.0,
        np.ceil(temperature_limits[1] / 5.0) * 5.0,
    ]
)

temperature_image = None
height_levels = np.arange(4800.0, 6121.0, 120.0)
for ax, (title, field, geopotential_height) in zip(axes[0], temperature_products):
    cyclic_field, cyclic_lon = add_cyclic_point(
        field,
        coord=forecast["lon"].values,
    )
    cyclic_height, _ = add_cyclic_point(
        geopotential_height,
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
    if not np.any(
        (height_levels >= np.nanmin(geopotential_height))
        & (height_levels <= np.nanmax(geopotential_height))
    ):
        raise RuntimeError(f"No 500-hPa contour levels intersect {title!r}")
    height_contours = ax.contour(
        cyclic_lon,
        forecast["lat"].values,
        cyclic_height,
        levels=height_levels,
        colors="#303030",
        linewidths=0.45,
        transform=ccrs.PlateCarree(),
    )
    ax.clabel(height_contours, height_contours.levels[::2], fmt="%.0f", fontsize=6)
    ax.set_title(title)
    ax.coastlines(linewidth=0.6)
    ax.gridlines(linewidth=0.3, alpha=0.5)

if temperature_image is None:
    raise RuntimeError("No temperature product was plotted")
temperature_colorbar = fig.colorbar(
    temperature_image,
    ax=list(axes[0]),
    orientation="horizontal",
    shrink=0.82,
    pad=0.04,
    extend="both",
)
temperature_colorbar.set_label("2-m temperature (°C); contours: 500-hPa height (m)")

week_to_week_temperature_change = target_week_temperature - previous_week_temperature
mean_absolute_weekly_change = float(np.mean(np.abs(week_to_week_temperature_change)))
if mean_absolute_weekly_change <= 0.01:
    raise RuntimeError("Week-to-week temperature product has insufficient variation")
temperature_change_limit = max(
    float(np.ceil(np.nanpercentile(np.abs(week_to_week_temperature_change), 99.5))),
    1.0,
)
cyclic_temperature_change, cyclic_lon = add_cyclic_point(
    week_to_week_temperature_change,
    coord=forecast["lon"].values,
)
temperature_change_image = axes[1, 0].pcolormesh(
    cyclic_lon,
    forecast["lat"].values,
    cyclic_temperature_change,
    transform=ccrs.PlateCarree(),
    cmap="RdBu_r",
    norm=TwoSlopeNorm(
        vmin=-temperature_change_limit,
        vcenter=0.0,
        vmax=temperature_change_limit,
    ),
    shading="auto",
    rasterized=True,
)
axes[1, 0].set_title(
    f"Temperature change: week {target_week_number} minus "
    f"week {previous_week_number}"
)
axes[1, 0].coastlines(linewidth=0.6)
axes[1, 0].gridlines(linewidth=0.3, alpha=0.5)
temperature_change_colorbar = fig.colorbar(
    temperature_change_image,
    ax=axes[1, 0],
    orientation="horizontal",
    pad=0.04,
    extend="both",
)
temperature_change_colorbar.set_label("2-m temperature change (°C)")

# FuXi-S2S ``tp`` is the daily mean of 24 one-hour accumulations. Recover each
# day's total with the factor of 24 before summing the seven daily predictions.
target_week_precipitation = (
    forecast["tp"].isel(time=0, lead_time=target_week).sum("lead_time").values
    * 24.0
    * 1000.0
)
if (
    not np.isfinite(target_week_precipitation).all()
    or not (target_week_precipitation >= 0.0).all()
):
    raise RuntimeError("Invalid target-week precipitation product")
precipitation_levels = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 400.0])
precipitation_display_threshold = precipitation_levels[0]
displayed_precipitation_count = int(
    np.count_nonzero(target_week_precipitation >= precipitation_display_threshold)
)
if displayed_precipitation_count == 0:
    raise RuntimeError("Target-week precipitation has no values at or above 1 mm")
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
masked_precipitation = np.ma.masked_less(
    target_week_precipitation,
    precipitation_display_threshold,
)
cyclic_precipitation, cyclic_lon = add_cyclic_point(
    masked_precipitation,
    coord=forecast["lon"].values,
)
precipitation_image = axes[1, 1].pcolormesh(
    cyclic_lon,
    forecast["lat"].values,
    cyclic_precipitation,
    transform=ccrs.PlateCarree(),
    cmap=precipitation_colormap,
    norm=precipitation_norm,
    shading="auto",
    rasterized=True,
)
axes[1, 1].set_title(
    f"Week {target_week_number} accumulated precipitation "
    f"(D+{target_week_start}–{forecast_days})\n"
    f"{np.datetime_as_string(initialization_date + np.timedelta64(target_week_start, 'D'), unit='D')}"
    " to "
    f"{np.datetime_as_string(initialization_date + np.timedelta64(forecast_days, 'D'), unit='D')}"
)
axes[1, 1].coastlines(linewidth=0.6)
axes[1, 1].gridlines(linewidth=0.3, alpha=0.5)
precipitation_colorbar = fig.colorbar(
    precipitation_image,
    ax=axes[1, 1],
    orientation="horizontal",
    pad=0.04,
    extend="max",
    ticks=precipitation_levels,
)
precipitation_colorbar.set_label("7-day accumulation (mm); values below 1 mm masked")

product_statistics = {
    "week_to_week_t2m_change": {
        "units": "degC",
        "from_week": previous_week_number,
        "to_week": target_week_number,
        "grid_mean_unweighted": float(np.mean(week_to_week_temperature_change)),
        "grid_mean_absolute_unweighted": mean_absolute_weekly_change,
        "grid_p01": float(np.percentile(week_to_week_temperature_change, 1.0)),
        "grid_p99": float(np.percentile(week_to_week_temperature_change, 99.0)),
    },
    "target_week_precipitation": {
        "units": "mm",
        "week": target_week_number,
        "aggregation": "24 hourly accumulations per day summed across 7 days",
        "grid_mean_unweighted": float(np.mean(target_week_precipitation)),
        "grid_p50": float(np.percentile(target_week_precipitation, 50.0)),
        "grid_p90": float(np.percentile(target_week_precipitation, 90.0)),
        "grid_p99": float(np.percentile(target_week_precipitation, 99.0)),
        "grid_max": float(np.max(target_week_precipitation)),
        "grid_fraction_at_or_above_1_mm": float(
            displayed_precipitation_count / target_week_precipitation.size
        ),
    },
}

fig.suptitle(
    "FuXi-S2S weekly forecast products\n"
    f"Single stochastic trajectory (seed {random_seed}); "
    "not verified against observations"
)
forecast_plot_path = output_directory / "04_fuxi_s2s_weekly_forecast.png"
fig.savefig(forecast_plot_path, dpi=200, bbox_inches="tight")


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


plot_metadata = {
    "relative_path": forecast_plot_path.name,
    **_validate_plot(forecast_plot_path),
}
plot_seconds = time.perf_counter() - plot_start
total_seconds = time.perf_counter() - total_start

report = {
    "status": "passed",
    "forecast_time": forecast_time,
    "forecast_days": forecast_days,
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
    "plots": [plot_metadata],
}
report_path = output_directory / "04_fuxi_s2s_validation.json"
report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print(f"Plotting: {plot_seconds:.2f} seconds")
print(f"Total runtime: {total_seconds:.2f} seconds")
print(f"Saved weekly forecast plot to {forecast_plot_path}")
print(f"Saved validation report to {report_path}")

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
- How to plot temperature and daily precipitation predictions

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
import os
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from dotenv import load_dotenv

from earth2studio.io import KVBackend
from earth2studio.models.auto import Package
from earth2studio.models.px import FuXiS2S
from earth2studio.models.px.fuxi_s2s import VARIABLES
from earth2studio.utils.type import TimeArray, VariableArray

os.makedirs("outputs", exist_ok=True)
load_dotenv()

if not torch.cuda.is_available():
    raise RuntimeError("A CUDA-capable GPU is required for this example")

device = torch.device("cuda:0")
forecast_days = 14
forecast_time = "2020-06-02"
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

# %%
# Prepare the Official Initial Condition
# --------------------------------------
# The archived input is normalized in the checkpoint's native units. We undo the
# normalization, restore Earth2Studio's accumulated-field units, and expose it as a
# small data source. In particular, ``tp`` is converted from ``log1p(mm)`` to metres
# and ``ttr`` from W m\ :sup:`-2` to J m\ :sup:`-2`.

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
    with archive.open("data/sample/total_precipitation.nc") as stream:
        official_tp = xr.open_dataarray(io.BytesIO(stream.read())).load()

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
initial_values[:, ttr_index] *= 3600.0
initial_values[:, tp_index] = (
    np.expm1(initial_values[:, tp_index]).clip(min=0.0) / 1000.0
)
np.testing.assert_allclose(
    initial_values[:, tp_index],
    official_tp.values[:, 0],
    rtol=1.0e-5,
    atol=1.0e-7,
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
sample_seconds = time.perf_counter() - sample_start
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
expected_shape = (1, forecast_days + 1, 121, 240)
for variable in ("t2m", "tp", "z500"):
    if forecast[variable].shape != expected_shape:
        raise RuntimeError(
            f"Unexpected {variable} shape: {forecast[variable].shape}, "
            f"expected {expected_shape}"
        )
if not np.isfinite(forecast[["t2m", "tp", "z500"]].to_array()).all():
    raise RuntimeError("FuXi-S2S produced non-finite values")
if not (forecast["tp"].values >= 0.0).all():
    raise RuntimeError("FuXi-S2S produced negative daily precipitation")
if not ((forecast["t2m"].values > 100.0) & (forecast["t2m"].values < 400.0)).all():
    raise RuntimeError("FuXi-S2S produced implausible 2-m temperatures")

expected_leads = np.arange(forecast_days + 1).astype("timedelta64[D]")
np.testing.assert_array_equal(
    forecast["lead_time"].values.astype("timedelta64[D]"),
    expected_leads,
)
print(f"{forecast_days}-day inference: {inference_seconds:.2f} seconds")
print(forecast)

# %%
# Plot the Forecast
# -----------------
# Compare the initialized and predicted 2-m temperature fields, then show the daily
# precipitation on the final forecast day. All precipitation values are displayed in
# millimetres.

# %% tags=["e2sg-profile:plotting"]
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

plot_start = time.perf_counter()
plt.close("all")
projection = ccrs.Robinson()
fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 8),
    subplot_kw={"projection": projection},
)

temperature_steps = (0, forecast_days // 2, forecast_days)
temperature_fields = [
    forecast["t2m"].isel(time=0, lead_time=step).values for step in temperature_steps
]
temperature_limits = np.nanpercentile(
    np.stack(temperature_fields),
    (1.0, 99.0),
)

for ax, step, field in zip(axes.flat[:3], temperature_steps, temperature_fields):
    image = ax.pcolormesh(
        forecast["lon"].values,
        forecast["lat"].values,
        field,
        transform=ccrs.PlateCarree(),
        cmap="Spectral_r",
        vmin=temperature_limits[0],
        vmax=temperature_limits[1],
    )
    ax.set_title(f"2-m temperature: day {step}")
    ax.coastlines(linewidth=0.6)
    ax.gridlines(linewidth=0.3)
    fig.colorbar(image, ax=ax, orientation="horizontal", pad=0.04, label="K")

precipitation = forecast["tp"].isel(time=0, lead_time=forecast_days).values * 1000.0
precipitation_max = max(float(np.nanpercentile(precipitation, 99.0)), 1.0)
image = axes[1, 1].pcolormesh(
    forecast["lon"].values,
    forecast["lat"].values,
    precipitation,
    transform=ccrs.PlateCarree(),
    cmap="Blues",
    vmin=0.0,
    vmax=precipitation_max,
)
axes[1, 1].set_title(f"Daily precipitation: day {forecast_days}")
axes[1, 1].coastlines(linewidth=0.6)
axes[1, 1].gridlines(linewidth=0.3)
fig.colorbar(image, ax=axes[1, 1], orientation="horizontal", pad=0.04, label="mm")

fig.suptitle(f"FuXi-S2S forecast initialized {forecast_time}")
fig.tight_layout()
output_path = Path("outputs/04_fuxi_s2s_forecast.jpg")
fig.savefig(output_path, dpi=150, bbox_inches="tight")
plot_seconds = time.perf_counter() - plot_start

if output_path.stat().st_size == 0:
    raise RuntimeError(f"Plot was not written to {output_path}")
print(f"Plotting: {plot_seconds:.2f} seconds")
print(f"Total runtime: {time.perf_counter() - total_start:.2f} seconds")
print(f"Saved forecast plot to {output_path}")

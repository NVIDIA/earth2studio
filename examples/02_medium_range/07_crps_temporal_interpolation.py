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
Probabilistic Temporal Interpolation (CRPS DiT)
===============================================

Ensemble temporal interpolation with the ``InterpCRPSDiT`` model.

Where :py:class:`~earth2studio.models.px.InterpModAFNO` produces a single deterministic sub-hourly
trajectory, ``InterpCRPSDiT`` is a probabilistic (ensemble) interpolator. For each gap between two
coarse frames it reproduces the two endpoints exactly and fills in the frames between them with a
learned correction scaled by a ``sin(pi tau)`` envelope that vanishes at the endpoints; every ensemble
member draws its own random latent, so members coincide at the shared endpoints and diverge through the
interior; the ``sin(pi tau)`` envelope weights the learned correction toward the middle of the gap,
where the pinned endpoints constrain the field least.
It was trained with CRPS, a score that rewards ensemble spread that reflects forecast uncertainty.

Because that spread is produced by the model itself, we drive it with
:py:func:`earth2studio.run.ensemble` and a :py:class:`~earth2studio.perturbation.Zero` (no-op)
perturbation -- all members share one initial condition and the diversity comes entirely from the
model. ``run.ensemble``'s ``batch_size`` then caps GPU memory by running members in chunks, so peak GPU
memory is set by the chunk size rather than the total ensemble size (output storage and runtime still
scale with the member count).

In this example you will learn:

- Replay an ERA5 trajectory as the coarse input via ``DataReplay``
- Produce a many-member ensemble from a single initial condition (the spread comes from the model)
- Use ``run.ensemble(batch_size=...)`` to keep peak GPU memory bounded as the ensemble grows
- Compare the mid-gap ensemble against the ERA5 reference field
- Zoom into a region with ``set_domain`` and go to 15-min output -- hurricane central-pressure tracks
  for three storms vs hourly ERA5
- Wrap a forecast model (SFNO) instead of a reanalysis replay

.. note::

   ``InterpCRPSDiT`` weights are not yet hosted
   (:meth:`~earth2studio.models.px.InterpCRPSDiT.load_default_package` returns a placeholder
   URL). To run this before they are hosted, load a local bundle instead of the default package --
   replace the ``load_default_package()`` call below with ``Package("/path/to/bundle")``
   (``from earth2studio.models.auto import Package``).
"""
# /// script
# dependencies = [
#   "torch==2.11.0", # Pinned for torch-harmonics compatibility
#   "earth2studio[sfno,interp-crps-dit] @ git+https://github.com/NVIDIA/earth2studio.git",
#   "matplotlib",
# ]
# ///

# %%
# Set Up
# ------
# We use ARCO ERA5 as the data source so that (a) the replayed bracket endpoints are ERA5 reference states
# and (b) we have the ERA5 reference mid-bracket (3 h) field to validate against. ARCO is on the 721-lat native
# grid and the model uses 720 lat (it drops the -90 pole row), but the interpolator reconciles any
# 0.25 deg base onto its own grid internally, so we point ``DataReplay`` straight at ARCO on its native
# grid -- no manual regridding.
#
# This example needs the following:
#
# - Interpolation Model: :py:class:`earth2studio.models.px.InterpCRPSDiT`.
# - Base trajectory: :py:class:`earth2studio.models.px.DataReplay` over ARCO ERA5.
# - Datasource: :py:class:`earth2studio.data.ARCO`.
# - Perturbation: :py:class:`earth2studio.perturbation.Zero` (spread comes from the model, not the IC).

# %%
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch

from earth2studio.data import ARCO
from earth2studio.io import ZarrBackend
from earth2studio.models.px import SFNO, DataReplay, InterpCRPSDiT
from earth2studio.models.px.interpcrpsdit import VARIABLES
from earth2studio.perturbation import Zero
from earth2studio.run import ensemble

os.makedirs("outputs", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the interpolator and wrap a DataReplay over ARCO ERA5 (6 h coarse step -> hourly output), on
# ARCO's native grid. The interpolator crops the 721-lat base to its own 720-lat grid internally.
# We run the DiT in bf16 (``amp_dtype``): the forward runs under bf16 autocast (weights stay fp32;
# autocast downcasts eligible operations), which can reduce activation memory. The linear base + endpoint
# pin still run in fp32, so the emitted coarse frames are unchanged. Drop ``amp_dtype`` for the fp32 default if you have the memory.
data = ARCO()
replay = DataReplay(
    data, VARIABLES, OrderedDict(lat=data.ARCO_LAT, lon=data.ARCO_LON), step=6
)
package = InterpCRPSDiT.load_default_package()
model = (
    InterpCRPSDiT.load_model(
        package, px_model=replay, num_interp_steps=6, amp_dtype=torch.bfloat16
    )
    .to(device)
    .eval()
)

t0 = np.datetime64(
    "2005-08-28T12:00:00"
)  # global/SFNO snapshot time (also Katrina's Gulf bracket below)

# %%
# Global Ensemble Over One 6 h Bracket
# ------------------------------------
# Interpolate the 12Z -> 18Z bracket to hourly for a 4-member ensemble. The ``Zero`` perturbation
# leaves every member's initial condition identical, so the spread is produced entirely by the
# model's per-member latent. ``batch_size=1`` runs the members one at a time, so peak VRAM is
# independent of the ensemble size. We then compare the 3 h ensemble against the ERA5 reference field.

# %%
nmembers = 4
io = ensemble(
    [t0],
    6,
    nmembers,
    model,
    data,
    ZarrBackend(),
    Zero(),
    batch_size=1,
    output_coords={"variable": np.array(["tcwv"])},
    device=device,
)
# io dims: (ensemble, time, lead_time, lat, lon); hourly leads 0..6 h, so index 3 is the 3 h midpoint.
# tcwv (total column water vapour) has fine filamentary structure that
# shows the interpolation quality far better than a smooth field such as t2m.
members_mid = np.asarray(io["tcwv"])[
    :, 0, 3
]  # (nmembers, 720, 1440) on the model's 720-lat grid
# ERA5 reference on the same 720 grid: ARCO's 721 lat with the -90 pole row dropped (== the model grid).
truth_mid = np.asarray(data(np.array([t0 + np.timedelta64(3, "h")]), ["tcwv"]).values)[
    0, 0, :720
]

fig, axs = plt.subplots(2, 3, figsize=(15, 6))
vmin, vmax = float(truth_mid.min()), float(truth_mid.max())
panels = [("ERA5 reference (3 h)", truth_mid), ("Ensemble mean", members_mid.mean(0))]
panels += [(f"Member {m}", members_mid[m]) for m in range(nmembers)]
for ax, (title, fld) in zip(axs.ravel(), panels):
    # interpolation="none": show pixels 1:1 (no antialias/blur); the PNG below is lossless.
    im = ax.imshow(
        fld, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto", interpolation="none"
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.025, label="tcwv [kg m$^{-2}$]")
plt.tight_layout()
plt.savefig("outputs/07_global_midpoint_ensemble.png", dpi=200)

# %%
# Regional Sub-Domain: Sub-Hourly Central-Pressure Ensembles for Three Hurricanes
# -------------------------------------------------------------------------------
# ``set_domain`` slices the invariant stack + output grid to a bounding box and returns a new model that
# shares the network; the DiT runs on the cropped grid in a single forward pass. We go sub-hourly
# (15-min output) and sweep three US-Gulf-Coast landfalling hurricanes through
# the same box, tracking each storm's central pressure (min MSLP over the patch). We reuse the global
# ``replay`` unchanged -- the interpolator crops each global frame to the bounding box for us. The ensemble is
# pinned to the 6-hourly ERA5 endpoints; the hourly ERA5 points in between are the
# reference for whether the 15-min curve tracks the reference deepening.

# %%
# halo=32: a wide (8 deg) border of extra cells that set_domain trims off the output, which pushes the
# non-periodic sub-domain edge farther from the returned storm area.
sub = model.set_domain(
    lat_min=18.0, lat_max=34.0, lon_min=262.0, lon_max=284.0, halo=32
)
sub.num_interp_steps = 24  # small sub-domain -> sub-hourly: 6 h / 24 = 15 min output
# set_domain already carries over the global replay as sub.px_model, so no reassignment is needed.

# Each storm: a 6 h bracket over its Gulf rapid-intensification / landfall.
storms = {
    "Ida (2021)": np.datetime64("2021-08-29T12:00:00"),
    "Michael (2018)": np.datetime64("2018-10-10T12:00:00"),
    "Katrina (2005)": np.datetime64("2005-08-28T12:00:00"),
}
nmembers_tc = 16
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
for ax, (name, storm_t0) in zip(axs, storms.items()):
    io_tc = ensemble(
        [storm_t0],
        sub.num_interp_steps,
        nmembers_tc,
        sub,
        data,
        ZarrBackend(),
        Zero(),
        batch_size=4,
        output_coords={"variable": np.array(["msl"])},
        device=device,
    )
    central = (np.asarray(io_tc["msl"])[:, 0] / 100.0).min(
        axis=(-2, -1)
    )  # (nmembers, lead)
    lead_h = np.asarray(io_tc.coords["lead_time"]) / np.timedelta64(
        1, "h"
    )  # 0, 1/4, ..., 6 h
    out_lat, out_lon = np.asarray(io_tc.coords["lat"]), np.asarray(
        io_tc.coords["lon"]
    )  # bounding box
    # ERA5 reference central pressure over the same patch. ERA5 is hourly, so the reference exists only at whole hours
    # -- 0 h/6 h are pinned exactly, and the hourly points test the 15-min curve in between.
    truth_h = np.arange(7)
    truth_times = np.array([storm_t0 + np.timedelta64(int(h), "h") for h in truth_h])
    truth_da = data(truth_times, ["msl"]).sel(
        lat=out_lat, lon=out_lon, method="nearest"
    )
    truth_central = (np.asarray(truth_da.values)[:, 0] / 100.0).min(axis=(-2, -1))

    for m in range(
        nmembers_tc
    ):  # each member: line + a fine x-marker at every 15-min sample
        ax.plot(
            lead_h, central[m], color="C0", lw=0.8, alpha=0.4, marker="x", ms=3, mew=0.6
        )
    ax.plot(
        [],
        [],
        color="C0",
        lw=0.8,
        marker="x",
        ms=3,
        label=f"members (n={nmembers_tc}, 15-min)",
    )
    ax.plot(lead_h, central.mean(0), color="C1", lw=2.5, label="ensemble mean")
    ax.plot(
        truth_h, truth_central, "ko", ms=6, zorder=5, label="ERA5 reference (hourly)"
    )
    ax.set_title(name)
    ax.set_xlabel("lead time [h]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axs[0].set_ylabel("min MSLP over patch [hPa]")
plt.tight_layout()
plt.savefig("outputs/07_tc_central_pressure.png", dpi=150)

# %%
# Wrapping a Forecast Model (SFNO) Instead of a Replay
# ----------------------------------------------------
# The base model need not be a reanalysis replay -- any 0.25 deg prognostic model that supplies the required input variables and a coarse step divisible by ``num_interp_steps`` works.
# Here we reuse the same interpolator and just swap in SFNO as the base to interpolate its 6 h forecast
# to hourly. SFNO runs on its native 721-lat grid; the interpolator reconciles each SFNO frame onto its
# own 720-lat grid (it drops the -90 pole row), so no manual regridding is needed.
#
# .. warning::
#
#    ``InterpCRPSDiT`` was trained on ERA5. The bracket endpoints are the verbatim base-model states
#    (correct for any base); the learned interior is applied here on a forecast trajectory, so the
#    inputs differ from the ERA5 training data.

# %%
sfno = SFNO.load_model(SFNO.load_default_package()).to(device).eval()
model.px_model = (
    sfno  # reuse the Part A interpolator; just swap the base model (no second DiT load)
)

io_sfno = ensemble(
    [t0],
    6,
    nmembers,
    model,
    data,
    ZarrBackend(),
    Zero(),
    batch_size=1,
    output_coords={"variable": np.array(["tcwv"])},
    device=device,
)
sfno_mid = np.asarray(io_sfno["tcwv"])[
    :, 0, 3
]  # 3 h midpoint tcwv, (nmembers, 720, 1440)
vmin, vmax = float(sfno_mid.min()), float(sfno_mid.max())  # shared scale across panels
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, (title, fld) in zip(
    axs,
    [
        ("SFNO interp mean (3 h)", sfno_mid.mean(0)),
        ("Member 0", sfno_mid[0]),
        ("Member 1", sfno_mid[1]),
    ],
):
    im = ax.imshow(
        fld, cmap="turbo", vmin=vmin, vmax=vmax, aspect="auto", interpolation="none"
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.025, label="tcwv [kg m$^{-2}$]")
plt.tight_layout()
plt.savefig("outputs/07_sfno_interp_ensemble.png", dpi=200)

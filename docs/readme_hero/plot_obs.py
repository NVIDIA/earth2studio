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

import cartopy.crs as ccrs
from common import *
from scipy.ndimage import gaussian_filter

o = np.load(P("obs.npz"), allow_pickle=True)
H, W = 1440, 2880  # 0.125 deg


def dens(lat, lon, sig):
    lon = ((lon + 180) % 360) - 180
    h, _, _ = np.histogram2d(lat, lon, bins=[H, W], range=[[-90, 90], [-180, 180]])
    h = gaussian_filter(h, sig, mode="wrap")
    return h


def norm(h, p=99.7, gamma=0.55):
    h = h / np.percentile(h[h > 0], p) if (h > 0).any() else h
    return np.clip(h, 0, 1) ** gamma


layers = {
    "sat": (
        dens(o["sw_lat"], o["sw_lon"], 1.2),
        np.array([0.10, 0.55, 1.0]),
    ),  # satellite winds - blue
    "gps": (
        dens(o["gps_lat"], o["gps_lon"], 1.0),
        np.array([1.0, 0.30, 0.75]),
    ),  # GPS-RO - magenta
    "uv": (
        dens(o["uv_lat"], o["uv_lon"], 1.0),
        np.array([1.0, 0.72, 0.15]),
    ),  # conventional winds - amber
    "t": (
        dens(o["t_lat"], o["t_lon"], 1.0),
        np.array([0.15, 1.0, 0.65]),
    ),  # temperature - green
}
rgb = np.zeros((H, W, 3))
for k, (h, c) in layers.items():
    n = norm(h)
    rgb += n[..., None] * c[None, None, :] * (0.9 if k != "sat" else 0.75)
rgb = 1 - np.exp(-2.1 * rgb)  # soft tonemap
rgb = np.clip(rgb, 0, 1)
N = 1960
fig = plt.figure(figsize=(N / 100, N / 100), dpi=100, facecolor="black")
proj = ccrs.Orthographic(central_longitude=115, central_latitude=15)
ax = fig.add_axes([0.03, 0.03, 0.94, 0.94], projection=proj)
ax.set_facecolor((0.015, 0.02, 0.04))
ax.spines["geo"].set_visible(False)
ax.imshow(
    rgb,
    origin="upper",
    extent=[-180, 180, -90, 90],
    transform=ccrs.PlateCarree(),
    regrid_shape=2400,
    interpolation="bilinear",
)
ax.coastlines(resolution="50m", linewidth=0.35, color=(1, 1, 1, 0.28))
ax.gridlines(
    linewidth=0.3,
    color=(1, 1, 1, 0.10),
    xlocs=range(-180, 181, 30),
    ylocs=range(-90, 91, 30),
)
fig.savefig(P("panels/obs_raw.png"), facecolor="black", dpi=100)
im = np.asarray(Image.open(P("panels/obs_raw.png")).convert("RGB")).astype(float) / 255
stats("obs", im)

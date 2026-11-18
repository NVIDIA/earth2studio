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
import matplotlib
from common import *

g = np.load(P("gfs.npz"), allow_pickle=True)
t2 = g["d"][1]
tc = np.load(P("era5.npz"))["d"]
lat, lon = g["lat"], g["lon"]
print(lat[:3], lon[:3], tc.shape)
cm = LinearSegmentedColormap.from_list(
    "wv",
    [
        "#000000",
        "#03121f",
        "#05304a",
        "#0a6a86",
        "#16b6b0",
        "#6ff0b8",
        "#e9ffd0",
        "#ffffff",
    ],
)
N = 1400
cm2 = plt.cm.turbo
for name, arr, cmap_, lo, hi, clon, clat in (
    ("era5_tcwv", tc, cm, 0, 65, 85, 10),
    ("gfs_t2m", t2, cm2, 228, 312, -75, 15),
):
    fig = plt.figure(figsize=(N / 100, N / 100), dpi=100, facecolor="black")
    ax = fig.add_axes(
        [0.02, 0.02, 0.96, 0.96],
        projection=ccrs.Orthographic(central_longitude=clon, central_latitude=clat),
    )
    ax.set_facecolor("black")
    ax.spines["geo"].set_visible(False)
    a = np.roll(arr, 720, axis=1)
    ax.imshow(
        a,
        origin="upper",
        extent=[-180, 180, -90, 90],
        transform=ccrs.PlateCarree(),
        cmap=cmap_,
        vmin=lo,
        vmax=hi,
        regrid_shape=1800,
        interpolation="bilinear",
    )
    ax.coastlines(resolution="50m", linewidth=0.35, color=(1, 1, 1, 0.4))
    ax.gridlines(
        linewidth=0.3,
        color=(1, 1, 1, 0.1),
        xlocs=range(-180, 181, 30),
        ylocs=range(-90, 91, 30),
    )
    fig.savefig(P(f"panels/{name}.png"), facecolor="black", dpi=100)
    stats(
        name,
        np.asarray(Image.open(P(f"panels/{name}.png")).convert("RGB")).astype(float)
        / 255,
    )

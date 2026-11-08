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
import cartopy.feature as cfeature
import matplotlib
from common import *

f = np.load(P("fcst.npz"), allow_pickle=True)
a = f["l144"]
u, v = a[0], a[1]
ws = np.hypot(u, v)
lat = np.linspace(90, -90, 721)
lon = np.linspace(0, 359.75, 1440)
N = 1960
tw = plt.cm.twilight
stops = (
    ["#000000", "#07041a"]
    + [matplotlib.colors.to_hex(tw(x)) for x in (0.47, 0.40, 0.32, 0.24, 0.16, 0.08)]
    + ["#ffd9f2", "#ff7ad0", "#ff9a3c", "#fff3b0"]
)
cmap = LinearSegmentedColormap.from_list("twilight_wind", stops)
fig = plt.figure(figsize=(N / 100, N / 100), dpi=100, facecolor="black")
proj = ccrs.Orthographic(central_longitude=-35, central_latitude=22)
ax = fig.add_axes([0.03, 0.03, 0.94, 0.94], projection=proj)
ax.set_facecolor("black")
ax.spines["geo"].set_visible(False)
# shaded wind speed (data lon 0..360: wrap by roll to -180..180)
wsr = np.roll(ws, 720, axis=1)
lonr = np.linspace(-180, 179.75, 1440)
# background t2m subtle? keep: wind speed
ax.imshow(
    wsr,
    origin="upper",
    extent=[-180, 180, -90, 90],
    transform=ccrs.PlateCarree(),
    cmap=cmap,
    vmin=0,
    vmax=22,
    regrid_shape=2400,
    interpolation="bilinear",
)
# MSL pressure contours
msl = np.roll(a[4], 720, axis=1) / 100.0
lat_a = lat[::-1]
ax.contour(
    lonr,
    lat_a,
    msl[::-1],
    levels=np.arange(940, 1060, 4),
    colors="white",
    linewidths=0.5,
    alpha=0.35,
    transform=ccrs.PlateCarree(),
)
# streamlines
s = 6
uu = np.roll(u, 720, axis=1)[::-1][::s, ::s]
vv = np.roll(v, 720, axis=1)[::-1][::s, ::s]
ax.streamplot(
    lonr[::s],
    lat_a[::s],
    uu,
    vv,
    transform=ccrs.PlateCarree(),
    color=(1, 1, 1, 0.55),
    linewidth=0.55,
    density=3.2,
    arrowsize=0,
)
ax.coastlines(resolution="50m", linewidth=0.45, color=(1, 1, 1, 0.55))
ax.add_feature(
    cfeature.BORDERS.with_scale("50m"), linewidth=0.2, edgecolor=(1, 1, 1, 0.22)
)
gl = ax.gridlines(
    linewidth=0.3,
    color=(1, 1, 1, 0.12),
    xlocs=range(-180, 181, 30),
    ylocs=range(-90, 91, 30),
)
fig.savefig(P("panels/globe_raw.png"), facecolor="black", dpi=100)
im = (
    np.asarray(Image.open(P("panels/globe_raw.png")).convert("RGB")).astype(float) / 255
)
stats("globe", im)

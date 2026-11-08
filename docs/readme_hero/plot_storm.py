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
from common import *

s = np.load(P("storm.npz"), allow_pickle=True)
g = s["g"][0, 0, 0]
r = s["r"][0, 0, 0]
lat = s["lat"]
lon = s["lon"] - 360
gv = list(s["gv"])
rv = list(s["rv"])
ir = g[gv.index("abi13c")]
vis = np.stack(
    [g[gv.index(k)] for k in ("abi03c", "abi02c", "abi01c")], -1
)  # R,G,B approx
refc = r[rv.index("refc")]
glm = r[rv.index("glm_density")]
# enhanced IR: white/grey -> cold cloud colours
ir_cols = [
    (0.0, "#000000"),
    (0.35, "#2a2f3a"),
    (0.55, "#8a93a6"),
    (0.68, "#e6ecf5"),
    (0.74, "#1ec8ff"),
    (0.82, "#3a4bff"),
    (0.88, "#b020ff"),
    (0.94, "#ff3d7f"),
    (1.0, "#fff36a"),
]
irmap = LinearSegmentedColormap.from_list("ir", ir_cols)
# map 300K->0 ... 200K ->1
irn = np.clip((300 - ir) / 100, 0, 1)
irn = np.where(np.isfinite(ir), irn, 0)
# radar NWS-ish colours
nws = ListedColormap(
    [
        "#04e9e7",
        "#019ff4",
        "#0300f4",
        "#02fd02",
        "#01c501",
        "#008e00",
        "#fdf802",
        "#e5bc00",
        "#fd9500",
        "#fd0000",
        "#d40000",
        "#bc0000",
        "#f800fd",
        "#9854c6",
        "#ffffff",
    ]
)
bounds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
from matplotlib.colors import BoundaryNorm

norm = BoundaryNorm(bounds[: len(nws.colors) + 1], nws.N)
rm = np.ma.masked_less(refc, 8)
N_W, N_H = 2400, 1500
fig = plt.figure(figsize=(N_W / 100, N_H / 100), dpi=100, facecolor="black")
proj = ccrs.LambertConformal(
    central_longitude=-97, central_latitude=38, standard_parallels=(33, 45)
)
ax = fig.add_axes([0, 0, 1, 1], projection=proj)
ax.set_facecolor("black")
ax.spines["geo"].set_visible(False)
ax.set_extent([-125.5, -69, 22.5, 50.5], ccrs.PlateCarree())
ds = 1
ax.pcolormesh(
    lon[::ds, ::ds],
    lat[::ds, ::ds],
    irn[::ds, ::ds],
    cmap=irmap,
    vmin=0,
    vmax=1,
    transform=ccrs.PlateCarree(),
    shading="auto",
    rasterized=True,
)
ax.pcolormesh(
    lon,
    lat,
    rm,
    cmap=nws,
    norm=norm,
    transform=ccrs.PlateCarree(),
    shading="auto",
    alpha=0.92,
    rasterized=True,
)
gm = np.ma.masked_less(glm, 6)
ax.pcolormesh(
    lon,
    lat,
    gm,
    cmap=ListedColormap(["#ffffff"]),
    transform=ccrs.PlateCarree(),
    shading="auto",
    alpha=0.95,
    rasterized=True,
)
ax.coastlines(resolution="50m", linewidth=0.5, color=(1, 1, 1, 0.55))
ax.add_feature(
    cfeature.STATES.with_scale("50m"),
    linewidth=0.25,
    edgecolor=(1, 1, 1, 0.3),
    facecolor="none",
)
ax.add_feature(
    cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor=(1, 1, 1, 0.45)
)
fig.savefig(P("panels/storm_raw.png"), facecolor="black", dpi=100)
im = (
    np.asarray(Image.open(P("panels/storm_raw.png")).convert("RGB")).astype(float) / 255
)
stats("storm", im)
print(
    "ir<250K frac",
    (ir < 250).mean(),
    "refc>20 frac",
    (refc > 20).mean(),
    "glm>6 frac",
    (glm > 6).mean(),
)

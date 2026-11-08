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
si = np.load(P("storm_in.npz"), allow_pickle=True)
g = s["g"][0, 0, 0]
lat = s["lat"]
lon = s["lon"] - 360
gv = [str(v) for v in s["gv"]]
b, r, nir, ir = (g[gv.index(k)] for k in ("abi01c", "abi02c", "abi03c", "abi13c"))


def st(x, lo, hi, g=1.0):
    """Scale values to the unit interval and apply an inverse-gamma stretch."""
    return np.clip((x - lo) / (hi - lo), 0, 1) ** (1.0 / g)


c = lambda k: np.nan_to_num(g[gv.index(k)])
valid = (
    np.isfinite(g[gv.index("abi13c")])
    & (c("abi13c") > 150)
    & (c("abi08c") > 150)
    & (c("abi10c") > 150)
    & (c("abi07c") > 150)
)
print("invalid frac", 1 - valid.mean())
b62, b73, b39, b103 = c("abi08c"), c("abi10c"), c("abi07c"), c("abi13c")
print(
    "BT p1/50/99",
    [np.percentile(x, [1, 50, 99]).round(1) for x in (b62, b73, b39, b103)],
)
R = st(b62 - b73, -35, 5) ** 1.1  # WV 6.2 - 7.3 : upper-level moisture / overshooting
G = st(b39 - b103, -5, 60, 2.0)  # 3.9 - 10.3 : small-particle, hot/cold cloud tops
B = st(300 - b103, 0, 100, 1.2)  # cold cloud top (inverted 10.3 um)
rgb = np.clip(np.stack([R, G, B], -1) * 1.2, 0, 1)
rgb[~valid] = 0
print("chan means", rgb.reshape(-1, 3).mean(0).round(3))
N_W, N_H = 2400, 1500
fig = plt.figure(figsize=(N_W / 100, N_H / 100), dpi=100, facecolor="black")
proj = ccrs.LambertConformal(
    central_longitude=-97, central_latitude=38, standard_parallels=(33, 45)
)
ax = fig.add_axes([0, 0, 1, 1], projection=proj)
ax.set_facecolor("black")
ax.spines["geo"].set_visible(False)
ax.set_extent([-125.5, -69, 22.5, 50.5], ccrs.PlateCarree())
ax.pcolormesh(
    lon,
    lat,
    rgb[:-1, :-1],
    transform=ccrs.PlateCarree(),
    shading="flat",
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
fig.savefig(P("panels/storm_rgb.png"), facecolor="black", dpi=100)
stats(
    "storm_rgb",
    np.asarray(Image.open(P("panels/storm_rgb.png")).convert("RGB")).astype(float)
    / 255,
)

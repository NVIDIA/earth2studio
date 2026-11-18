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

j = np.load(P("jet.npz"))
u, v = j["d"]
ws = np.hypot(u, v)
cm = LinearSegmentedColormap.from_list(
    "jet",
    [
        "#000000",
        "#0a0420",
        "#2b0a6b",
        "#6a14b8",
        "#d01fa0",
        "#ff4f5e",
        "#ff9a2e",
        "#ffe35a",
        "#ffffff",
    ],
)
lonr = np.linspace(-180, 179.75, 1440)
latr = np.linspace(90, -90, 721)[::-1]
N = 1400
fig = plt.figure(figsize=(N / 100, N / 100), dpi=100, facecolor="black")
ax = fig.add_axes(
    [0.02, 0.02, 0.96, 0.96],
    projection=ccrs.Orthographic(central_longitude=-135, central_latitude=38),
)
ax.set_facecolor("black")
ax.spines["geo"].set_visible(False)
ax.imshow(
    np.roll(ws, 720, axis=1),
    origin="upper",
    extent=[-180, 180, -90, 90],
    transform=ccrs.PlateCarree(),
    cmap=cm,
    vmin=0,
    vmax=85,
    regrid_shape=1800,
    interpolation="bilinear",
)
s = 6
ax.streamplot(
    lonr[::s],
    latr[::s],
    np.roll(u, 720, axis=1)[::-1][::s, ::s],
    np.roll(v, 720, axis=1)[::-1][::s, ::s],
    transform=ccrs.PlateCarree(),
    color=(1, 1, 1, 0.35),
    linewidth=0.45,
    density=2.6,
    arrowsize=0,
)
ax.coastlines(resolution="50m", linewidth=0.4, color=(1, 1, 1, 0.5))
ax.gridlines(
    linewidth=0.3,
    color=(1, 1, 1, 0.1),
    xlocs=range(-180, 181, 30),
    ylocs=range(-90, 91, 30),
)
fig.savefig(P("panels/gfs_jet.png"), facecolor="black", dpi=100)
stats(
    "gfs_jet",
    np.asarray(Image.open(P("panels/gfs_jet.png")).convert("RGB")).astype(float) / 255,
)

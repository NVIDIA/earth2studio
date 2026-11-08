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

from earth2studio.data.meteosat_fci import (
    PERSPECTIVE_POINT_HEIGHT,
    SEMI_MAJOR_AXIS,
    SEMI_MINOR_AXIS,
)

e = np.load(P("eu.npz"), allow_pickle=True)
var = [str(v) for v in e["var"]]


def ch(a, n):
    return a[var.index(n)]


C1, C2 = 1.191042e-5, 1.4387752
NU = {
    "fci38ir": 1e4 / 3.8,
    "fci63wv": 1e4 / 6.3,
    "fci73wv": 1e4 / 7.35,
    "fci105ir": 1e4 / 10.5,
}


def bt(L, n):
    nu = NU[n]
    return C2 * nu / np.log1p(C1 * nu**3 / np.maximum(L, 1e-3))


# radiance (mW m-2 sr-1 (cm-1)-1) -> approx reflectance using nominal band-centre solar irradiance
ESUN = {
    "fci04vis": (0.444, 1920),
    "fci05vis": (0.510, 1830),
    "fci06vis": (0.640, 1590),
    "fci13nir": (1.380, 360),
    "fci16nir": (1.610, 247),
    "fci22nir": (2.250, 84),
}


def refl(L, n):
    lam, es = ESUN[n]
    nm_per_cm1 = (lam * 1e-4) ** 2 * 1e7
    return L / (es / np.pi * nm_per_cm1)


def stretch(x, lo, hi, g=1.0):
    return np.clip((x - lo) / (hi - lo), 0, 1) ** (1.0 / g)


def render(a, which):
    L = lambda n: np.nan_to_num(ch(a, n))
    if which == "conv":
        R = stretch(bt(L("fci63wv"), "fci63wv") - bt(L("fci73wv"), "fci73wv"), -35, 5)
        G = stretch(
            bt(L("fci38ir"), "fci38ir") - bt(L("fci105ir"), "fci105ir"),
            -5,
            60,
            0.5 * 1.0,
        )
        B = stretch(
            (refl(L("fci16nir"), "fci16nir") - refl(L("fci06vis"), "fci06vis")) * 100,
            -75,
            25,
        )
        B = B**2.8  # darken the clear-sky blue pedestal so storms pop on black
        R = R**1.25
        rgb = np.stack([R, G, B], -1)
        rgb = np.clip(rgb * 1.25, 0, 1)
    elif which == "cloud":
        R = stretch(refl(L("fci06vis"), "fci06vis"), 0.0, 0.85)
        G = stretch(refl(L("fci16nir"), "fci16nir"), 0.0, 0.55)
        B = stretch(refl(L("fci22nir"), "fci22nir"), 0.0, 0.30)
        rgb = np.clip(np.stack([R, G, B], -1) ** 0.75 * 1.18, 0, 1)
    elif which == "true":
        rgb = (
            np.stack([ch(a, n) for n in ("fci06vis", "fci05vis", "fci04vis")], -1)
            / 24.0
        )
    return rgb


xa, ya = e["xa"], e["ya"]
dx, dy = abs(xa[1] - xa[0]), abs(ya[1] - ya[0])
H_ = PERSPECTIVE_POINT_HEIGHT
extent = [
    -(xa.max() + dx / 2) * H_,
    -(xa.min() - dx / 2) * H_,
    (ya.min() - dy / 2) * H_,
    (ya.max() + dy / 2) * H_,
]
proj = ccrs.Geostationary(
    central_longitude=0.0,
    satellite_height=H_,
    globe=ccrs.Globe(semimajor_axis=SEMI_MAJOR_AXIS, semiminor_axis=SEMI_MINOR_AXIS),
    sweep_axis="y",
)
ASPECT = 1.6


def panel(a, which, name, bg_nan=True):
    rgb = render(a, which)
    valid = np.isfinite(ch(a, "fci06vis"))
    rgb = np.where(valid[..., None], rgb, 0)
    W = 2400
    Hh = int(W / ASPECT)
    # crop columns to target aspect around centre, keep rows
    full_w = extent[1] - extent[0]
    full_h = extent[3] - extent[2]
    want_w = full_h * ASPECT
    cx = 0.5 * (extent[0] + extent[1]) + 0.04 * full_w
    xl = (cx - want_w / 2, cx + want_w / 2)
    fig = plt.figure(figsize=(W / 100, Hh / 100), dpi=100, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], projection=proj)
    ax.set_facecolor("black")
    ax.spines["geo"].set_visible(False)
    ax.imshow(
        rgb, transform=proj, extent=extent, origin="lower", interpolation="bilinear"
    )
    ax.set_xlim(*xl)
    ax.set_ylim(extent[2], extent[3])
    ax.coastlines(resolution="50m", color=(1, 1, 1, 0.75), linewidth=0.7)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"), edgecolor=(1, 1, 1, 0.35), linewidth=0.4
    )
    fig.savefig(P(f"panels/{name}.png"), facecolor="black", dpi=100)
    im = (
        np.asarray(Image.open(P(f"panels/{name}.png")).convert("RGB")).astype(float)
        / 255
    )
    stats(name, im)
    r = rgb[valid]
    print(
        "  channel means",
        r.mean(0).round(3),
        "p99",
        np.percentile(r, 99, axis=0).round(3),
    )


panel(e["pred"], "conv", "eu_conv")
panel(e["pred"], "cloud", "eu_cloud")
panel(e["inp"], "conv", "eu_conv_in")

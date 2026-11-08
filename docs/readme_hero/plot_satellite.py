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

from common import *
from scipy.ndimage import zoom


def down(a, k):
    h, w = a.shape[-2:]
    h2, w2 = h // k * k, w // k * k
    a = a[..., :h2, :w2]
    return np.nanmean(a.reshape(*a.shape[:-2], h2 // k, k, w2 // k, k), axis=(-3, -1))


ir_cols = [
    (0.0, "#000000"),
    (0.30, "#1b2030"),
    (0.52, "#6c7488"),
    (0.66, "#dfe6f2"),
    (0.72, "#18c6ff"),
    (0.80, "#3a4bff"),
    (0.87, "#b020ff"),
    (0.94, "#ff3d7f"),
    (1.0, "#fff36a"),
]
irmap = LinearSegmentedColormap.from_list("ir", ir_cols)


def truecolor(r, g, b, gain, gamma=0.62):
    rgb = np.stack([r, g, b], -1) * gain
    rgb = np.clip(np.nan_to_num(rgb), 0, 1) ** gamma
    return rgb


def ir_rgb(ir):
    n = np.clip((300 - ir) / 100, 0, 1)
    n = np.nan_to_num(n)
    return irmap(n)[..., :3]


def blend(vis, ir, valid):
    lum = vis.mean(-1, keepdims=True)
    w = np.clip((lum - 0.05) / 0.25, 0, 1)  # day weight
    out = vis * w + (ir * 0.8) * (1 - w)
    out[~valid] = 0
    return out


# GOES-16 : abi01 blue, abi02 red, abi03 veggie
d = np.load(P("goes_fd.npz"), allow_pickle=True)["d"]
d = down(d, 2)
b, r, nir, ir = d
gr = 0.45 * r + 0.10 * nir + 0.45 * b
valid = np.isfinite(ir)
vis = truecolor(r, gr, b, 1.45)
goes = blend(vis, ir_rgb(ir), valid)
stats("goes", goes)
save_rgb(goes, "goes")
# Himawari-9 : ahi01 blue, ahi02 green, ahi03 red, ahi13 IR  -> IR-enhanced full colour
h = np.load(P("him.npz"), allow_pickle=True)["d"]
h = down(h, 2)
b, g, r, ir = h
from scipy.ndimage import binary_erosion

valid = np.isfinite(ir) & (ir > 190)
valid = binary_erosion(valid, iterations=30)
ir = np.where(valid, ir, np.nan)
vis = truecolor(r, g, b, 1.35)
him = blend(vis, ir_rgb(ir), valid)
stats("himawari", him)
save_rgb(him, "himawari")
# IR-only enhanced version of himawari for contrast
him_ir = ir_rgb(ir)
him_ir[~valid] = 0
stats("himawari_ir", him_ir)
save_rgb(him_ir, "himawari_ir")

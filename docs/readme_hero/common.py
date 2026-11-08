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

import matplotlib
import numpy as np

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from paths import WORK, P
from PIL import Image

MPL = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")


def font(bold=False, mono=False, size=20):
    """Load a bundled DejaVu font with the requested style and size."""
    from PIL import ImageFont

    n = (
        ("DejaVuSansMono" if mono else "DejaVuSans")
        + ("-Bold" if bold else "")
        + ".ttf"
    )
    return ImageFont.truetype(os.path.join(MPL, n), size)


def save_rgb(arr, name):
    """Save a normalized RGB array as an eight-bit panel PNG."""
    a = np.clip(arr, 0, 1)
    Image.fromarray((a * 255).astype(np.uint8)).save(P(f"panels/{name}.png"))


def stats(name, arr):
    """Print image dimensions, mean brightness, and nonblack pixel coverage."""
    a = np.asarray(arr)[..., :3]
    lum = a.mean(-1)
    print(
        name,
        a.shape,
        "mean lum %.3f" % lum.mean(),
        "nonblack %.1f%%" % (100 * (lum > 0.03).mean()),
    )

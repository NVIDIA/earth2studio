# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import os

import xarray as xr
from matplotlib import pyplot as plt


def check_diagnostic_outputs(fn: str = "../diagnostic_distributed.zarr") -> None:
    """Check the diagnostic outputs."""

    with xr.open_dataset(fn, engine="zarr") as ds:
        tp = ds["tp"].values

    min_tp = tp.min()
    max_tp = tp.max()
    mean_tp = tp.mean()

    print(f"Minimum tp: {min_tp}")
    print(f"Maximum tp: {max_tp}")
    print(f"Mean tp: {mean_tp}")

    os.makedirs("test_figures", exist_ok=True)
    for i in range(tp.shape[1]):
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(tp[0, i, :, :])
        fig.colorbar(im, ax=ax, orientation="vertical")
        fig.savefig(f"test_figures/tp_{i:02d}.png", bbox_inches="tight")


if __name__ == "__main__":
    check_diagnostic_outputs()

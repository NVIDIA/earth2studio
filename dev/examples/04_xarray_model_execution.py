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

"""DataArray model execution
=========================

FCN and PrecipitationAFNO accept one NumPy- or CuPy-backed DataArray. This example
uses FCN's execution path with an identity core and a tiny synthetic signature so
it requires no model weights, downloads, or GPU. Production FCN uses its declared
26 variables and registered fcn1 grid, including the grid ID and CRS metadata.
"""

from collections.abc import Generator

import numpy as np
import torch
import xarray as xr

from earth2studio.models.px.fcn import FCN
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


class TinyFCN(FCN):
    """Small synthetic signature for demonstrating the real FCN execution path."""

    def input_coords(self) -> xr.DataArray:
        """Return a two-variable, 2 by 3 input signature."""
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": ["u10m", "v10m"],
                "lat": [10, 0],
                "lon": [0, 10, 20],
            },
            dynamic=("batch",),
        )


def add_bias(x: xr.DataArray) -> xr.DataArray:
    """Add a unit bias while retaining coordinates and attributes."""
    return x.copy(data=x.data + 1)


# %%
# Build data from the allocation-free signature. The bridge strips signature-only
# attributes and shares the CPU tensor memory with NumPy.
model = TinyFCN(torch.nn.Identity(), torch.zeros(2, 1, 1), torch.ones(2, 1, 1))
signature = coord_array_like(model.input_coords(), {"batch": [0, 1]})
x = from_torch(torch.zeros(signature.shape), signature, name="weather")
x = x.rename(batch="member").expand_dims(time=[np.datetime64("2026-01-01")])
x.attrs["experiment"] = "synthetic"

# %%
# Model batching flattens time/member for the Torch core and restores them on output.
forecast = model(x)
np.testing.assert_equal(forecast.dims, x.dims)
np.testing.assert_equal(forecast.lead_time.values[0], np.timedelta64(6, "h"))
np.testing.assert_equal(forecast.attrs["experiment"], "synthetic")
np.testing.assert_equal(model.output_coords(forecast).data.nbytes, 0)

# %%
# Hooks receive the unbatched DataArray and run only inside the iterator.
model.rear_hook = add_bias
iterator = model.create_iterator(x)
xr.testing.assert_identical(next(iterator), x)
step = next(iterator)
np.testing.assert_array_equal(step.data, 1)
if isinstance(iterator, Generator):
    iterator.close()
model.clear_hooks()

# %%
# For CUDA execution (when CuPy is installed), use model.to("cuda:0") together
# with x.e2s.as_cupy(device=0). Outputs remain on CUDA through the DLPack bridge.
# PrecipitationAFNO follows the same one-array API: select its required variables
# from an FCN forecast and call diagnostic(forecast.sel(variable=variables)).
# Its output retains leading time/lead_time dimensions and reports tp:sum:6h.

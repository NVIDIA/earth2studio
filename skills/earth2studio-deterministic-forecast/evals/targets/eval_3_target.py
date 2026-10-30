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

import numpy as np
import torch
from tqdm import tqdm

from earth2studio.data import GFS, fetch_data
from earth2studio.io import ZarrBackend
from earth2studio.io.utils import output_template
from earth2studio.models.px import Pangu
from earth2studio.utils.coords import handshake_dataarray
from earth2studio.utils.time import to_time_array

# 1. Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
time = to_time_array(["2024-03-01T00:00:00"])
nsteps = 20  # 5 days / 6h time step = 20 steps
output_variables = np.array(["z500", "t2m"])

# 2. Initialize model and move to device
model = Pangu.load_model(Pangu.load_default_package())
model = model.to(device)

# 3. Initialize data source
data = GFS()

# 4. Fetch initial conditions; GFS shares Pangu's native 0.25 degree grid
prognostic_ic = model.input_coords()
x = fetch_data(
    source=data,
    time=time,
    variable=prognostic_ic.coords["variable"].values,
    lead_time=prognostic_ic.coords["lead_time"].values,
    device=device,
)
handshake_dataarray(x, prognostic_ic)

# 5. Plan the IO template over the whole rollout, keeping only z500 and t2m
signature = model.output_coords(prognostic_ic)
step = signature.coords["lead_time"].values[-1]
template = output_template(
    signature,
    leading={"time": time},
    coords={"lead_time": np.arange(nsteps + 1) * step, "variable": output_variables},
)

# 6. Initialize IO
io = ZarrBackend("output_eval7.zarr")
io.add_array(template)

# 7. Create prognostic iterator
model_iterator = model.create_iterator(x)

# 8. Step through the model and write output
for step_index, field in enumerate(
    tqdm(model_iterator, total=nsteps + 1, desc="Running inference")
):
    # Writes are located by coordinate label, so each step writes its own slice
    io.write(field.sel(variable=output_variables))
    if step_index == nsteps:
        break
io.close()

print("Forecast complete. Output at: output_eval7.zarr")

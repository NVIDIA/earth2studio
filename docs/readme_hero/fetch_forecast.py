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
from paths import P

from earth2studio.data import GFS, fetch_data
from earth2studio.models.px import SFNO
from earth2studio.utils.coords import map_coords

m = SFNO.load_model(SFNO.load_default_package()).to("cuda")
src = GFS()
t = np.array([np.datetime64("2026-01-15T00:00")])
ic = m.input_coords()
x = fetch_data(src, t, ic["variable"], lead_time=ic["lead_time"], device="cuda").astype(
    "float32"
)
print(x.shape)
outs = {}
it = m.create_iterator(x)
vars_keep = ["u10m", "v10m", "t2m", "tcwv", "msl", "z500"]
for i, y in enumerate(it):
    lead = int(y.coords["lead_time"].values[0] / np.timedelta64(1, "h"))
    if lead in (96, 144):
        yy = y.e2s.as_numpy() if hasattr(y, "e2s") else y
        outs[lead] = yy.sel(variable=vars_keep).squeeze().values
        print("lead", lead, outs[lead].shape)
    if lead >= 144:
        break
np.savez(
    P("fcst.npz"), **{f"l{k}": v for k, v in outs.items()}, vars=np.array(vars_keep)
)

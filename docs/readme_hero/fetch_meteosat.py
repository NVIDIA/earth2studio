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

from datetime import datetime, timezone

import numpy as np
import torch
from paths import P

from earth2studio.data import fetch_data
from earth2studio.models.px.stormscope_meteosat import StormScopeMeteosatEU

device = torch.device("cuda")
pkg = StormScopeMeteosatEU.load_default_package()
model = StormScopeMeteosatEU.load_model(package=pkg, compile=False).to(device).eval()
ic = model.input_coords()
print("vars", list(ic["variable"].values), "leads", ic["lead_time"].values)
t0 = np.datetime64(datetime(2026, 6, 30, 12, 0, 0, tzinfo=timezone.utc))
x = fetch_data(
    model.default_sources(),
    time=np.array([t0]),
    variable=ic["variable"].values,
    lead_time=ic["lead_time"].values,
    device=device,
)
x = x.expand_dims(ensemble=np.arange(1)).copy(deep=True)
frames = {}
for step, xp in enumerate(model.create_iterator(x)):
    if step in (5, 11):
        frames[step] = xp.e2s.as_numpy().isel(ensemble=0, time=0, lead_time=0)
    if step == 11:
        break
xin = x.e2s.as_numpy().isel(ensemble=0, time=0, lead_time=-1)
f = frames[11]
print(f.dims, f.shape)
np.savez(
    P("eu.npz"),
    pred=f.values.astype(np.float32),
    pred6=frames[5].values.astype(np.float32),
    inp=xin.values.astype(np.float32),
    var=np.array(f["variable"].values),
    xa=f["x"].values,
    ya=f["y"].values,
)
print("saved")

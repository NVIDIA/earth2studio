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

from earth2studio.data import fetch_data
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)

pkg = StormScopeBase.load_default_package()
g = (
    StormScopeGOES.load_model(
        package=pkg, model_name="3km_10min_flash", amp=True, compile=False
    )
    .to("cuda")
    .eval()
)
r = (
    StormScopeMRMS.load_model(
        package=pkg, model_name="3km_10min_flash", amp=True, compile=False
    )
    .to("cuda")
    .eval()
)
print(g.input_coords()["variable"], g.input_coords()["lead_time"])
print(r.input_coords()["variable"], r.input_coords()["lead_time"])
t = np.array([np.datetime64("2024-04-26T22:00")])
ic = g.input_coords()
x = fetch_data(
    g.default_sources(),
    t,
    ic["variable"].values,
    lead_time=ic["lead_time"].values,
    device="cuda",
)
rc = r.input_coords()
rs, _ = r.default_sources()
xr_ = fetch_data(
    rs, t, rc["variable"].values, lead_time=rc["lead_time"].values, device="cuda"
)
x = x.expand_dims(batch=[0]).astype(np.float32)
xr_ = xr_.expand_dims(batch=[0]).astype(np.float32)
print(x.shape, xr_.shape)
yg, sg = g.initialize(x)
yr, sr = r.initialize(xr_, x)
frames = {}
for k in range(12):
    yr, sr = r.step(yr, yg, state=sr)
    yg, sg = g.step(yg, state=sg)
    frames[k] = (
        yg.e2s.as_numpy() if hasattr(yg, "e2s") else yg,
        yr.e2s.as_numpy() if hasattr(yr, "e2s") else yr,
    )
yg_, yr_ = frames[11]
print(yg_.dims, yg_.shape, yr_.dims, yr_.shape)
np.savez(
    P("storm.npz"),
    g=yg_.values,
    r=yr_.values,
    gv=np.array(yg_["variable"].values),
    rv=np.array(yr_["variable"].values),
    lat=yg_["lat"].values,
    lon=yg_["lon"].values,
)
xo = x.e2s.as_numpy() if hasattr(x, "e2s") else x
np.savez(
    P("storm_in.npz"),
    g=xo.values[0, 0, -1],
    rin=(xr_.e2s.as_numpy() if hasattr(xr_, "e2s") else xr_).values[0, 0, -1],
)
print("done")

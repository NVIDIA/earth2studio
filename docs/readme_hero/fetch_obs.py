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

from datetime import timedelta

import numpy as np
from paths import P

from earth2studio.data import NNJAObsConv, NNJAObsSatwnd, fetch_dataframe

t = np.array([np.datetime64("2026-01-15T00:00")])
tol = (timedelta(hours=-3), timedelta(hours=3))
conv = NNJAObsConv(time_tolerance=tol, layout="wide", async_timeout=600)
sw = NNJAObsSatwnd(time_tolerance=tol, layout="wide", async_timeout=600)


def get(src, var, cols):
    df = fetch_dataframe(src, t, np.array(var))
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    return {c: np.asarray(df[c]) for c in cols}


d = {}
for k, v in get(conv, ["t"], ["lat", "lon", "pres", "type", "observation_t"]).items():
    d["t_" + k] = v
for k, v in get(
    conv, ["u", "v"], ["lat", "lon", "pres", "observation_u", "observation_v"]
).items():
    d["uv_" + k] = v
for k, v in get(conv, ["gps"], ["lat", "lon", "elev", "observation_gps"]).items():
    d["gps_" + k] = v
for k, v in get(
    sw, ["u", "v"], ["lat", "lon", "pres", "observation_u", "observation_v"]
).items():
    d["sw_" + k] = v
np.savez(P("obs.npz"), **d)
for k, v in d.items():
    print(
        k,
        v.shape,
        v.dtype,
        np.nanmin(v) if v.dtype.kind == "f" else "",
        np.nanmax(v) if v.dtype.kind == "f" else "",
    )

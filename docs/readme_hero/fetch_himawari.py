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
from paths import P

from earth2studio.data import HimawariAHI

t = np.array([np.datetime64("2024-04-26T22:00")])
h = HimawariAHI(satellite="himawari9", verbose=False)
da = h(t, ["ahi01", "ahi02", "ahi03", "ahi13"])
print(da.dims, da.shape, list(da.coords))
da = da.e2s.as_numpy() if hasattr(da, "e2s") else da
d = da.values[0].astype(np.float32)
print(
    d.shape,
    [
        (float(np.nanmin(x)), float(np.nanpercentile(x, 99)), float(np.isnan(x).mean()))
        for x in d
    ],
)
lat = [c for c in da.coords if "lat" in c]
lon = [c for c in da.coords if "lon" in c]
print(lat, lon)
np.savez(
    P("him.npz"),
    d=d,
    lat=np.asarray(da.coords[lat[0]].values),
    lon=np.asarray(da.coords[lon[0]].values),
)

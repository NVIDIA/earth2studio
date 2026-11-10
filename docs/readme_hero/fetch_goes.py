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

from earth2studio.data import GOES

t = np.array([np.datetime64("2024-04-26T22:00")])
g = GOES(satellite="goes16", scan_mode="F")
da = g(t, ["abi01c", "abi02c", "abi03c", "abi13c"])
print(da.dims, da.shape, list(da.coords))
da = da.e2s.as_numpy()
np.savez(
    P("goes_fd.npz"),
    d=da.values[0],
    lat=np.asarray(da.coords["_lat"].values),
    lon=np.asarray(da.coords["_lon"].values),
    v=da["variable"].values,
)

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

from earth2studio.data import ARCO_ERA5

t = np.array([np.datetime64("2024-04-27T12:00")])
da = ARCO_ERA5(verbose=False)(t, ["tcwv"])
da = da.e2s.as_numpy() if hasattr(da, "e2s") else da
print(da.dims, da.shape, da["lat"].values[:3], da["lon"].values[:3])
v = da.values[0, 0]
print(float(v.min()), float(v.max()))
np.savez(
    P("era5.npz"), d=v.astype(np.float32), lat=da["lat"].values, lon=da["lon"].values
)

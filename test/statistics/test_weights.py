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
import pytest
import torch

from earth2studio.statistics import lat_weight


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_lat_weight_non_negative(backend: str, dtype: str) -> None:
    """Weights must be non-negative even at the poles.

    In float32, cos(+-90 deg) rounds to a small negative number (-4.4e-8);
    a negative weight can flip a weighted mean of non-negative values
    negative (e.g. sqrt of a weighted MSE becoming NaN when the only error
    sits on a pole row). Regression test for the clamp in lat_weight.
    """
    lat = np.linspace(90.0, -90.0, 721).astype(dtype)
    if backend == "torch":
        weights = lat_weight(torch.as_tensor(lat))
        assert bool((weights >= 0).all())
        weights = weights.numpy()
    else:
        weights = lat_weight(lat)
        assert (weights >= 0).all()

    # Error concentrated entirely on the pole row must never produce a
    # negative weighted mean square.
    err2 = np.zeros_like(weights)
    err2[-1] = 1.0e3
    assert (weights * err2).sum() / weights.sum() >= 0.0

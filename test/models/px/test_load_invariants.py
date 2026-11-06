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

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.data import Random
from earth2studio.models.px.utils import load_invariants

TIME = np.datetime64("2026-01-01T00:00")
VARIABLES = ["lsm", "sdor", "slor", "z"]


class Float64Source(Random):
    """Tiny float64 source that counts its calls."""

    def __init__(self):
        super().__init__(
            OrderedDict(lat=np.array([10.0, 0.0]), lon=np.array([0.0, 1.0, 2.0]))
        )
        self.calls = 0

    def __call__(self, time, variable):
        self.calls += 1
        return super().__call__(time, variable).astype(np.float64)


def test_load_invariants_fetch(tmp_path):
    source = Float64Source()
    invariants = load_invariants(str(tmp_path), TIME, VARIABLES, source=source)
    assert invariants.dtype == torch.float32
    assert invariants.shape == (len(VARIABLES), 2, 3)
    cached = torch.load(tmp_path / "invariants.pt", weights_only=True)
    assert cached.dtype == torch.float32
    assert torch.equal(cached, invariants)


def test_load_invariants_float64_cache(tmp_path):
    # Caches written by earlier 1.0 versions hold float64
    torch.save(
        torch.ones(len(VARIABLES), 2, 3, dtype=torch.float64),
        tmp_path / "invariants.pt",
    )
    source = Float64Source()
    invariants = load_invariants(str(tmp_path), TIME, VARIABLES, source=source)
    assert invariants.dtype == torch.float32
    assert torch.equal(invariants, torch.ones(len(VARIABLES), 2, 3))
    assert source.calls == 0

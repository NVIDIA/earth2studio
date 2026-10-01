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

from earth2studio.models.conformance import ContractException, check_diagnostic_contract
from earth2studio.models.dx import CorrDiffEra5Hrrr


def test_corrdiff_era5_hrrr_conformance():
    p = CorrDiffEra5Hrrr.__new__(CorrDiffEra5Hrrr)
    torch.nn.Module.__init__(p)
    p.register_buffer("lat_input_grid", torch.linspace(30, 27, 4))
    p.lat_input_numpy = p.lat_input_grid.numpy()
    p.lon_input_numpy = np.arange(8) + 260.0
    p.register_buffer("lat_output_grid", torch.ones(2, 3) * 28)
    p._lat_out_cpu = p.lat_output_grid.numpy()
    p._lon_out_cpu = np.ones((2, 3)) * 262
    p.hrrr_y, p.hrrr_x = np.arange(2), np.arange(3)
    p.era5_variables, p.output_variables = np.array(["t2m"]), np.array(["t2m"])
    p.number_of_samples = 2
    p._forward = lambda x, time: torch.zeros(2, 1, 2, 3)
    check_diagnostic_contract(p)
    p._forward = lambda x, time: torch.randn(2, 1, 2, 3)
    with pytest.raises(ContractException) as exc_info:
        check_diagnostic_contract(p)
    assert {v.split(":")[0] for v in exc_info.value.violations} == {"D9"}

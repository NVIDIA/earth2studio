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


import pytest
import torch

from earth2studio.utils.vertical import (
    HybridLevels,
    PressureLevels,
    interpolate_pressure,
)


def test_vertical_coordinates_are_validated():
    with pytest.raises(ValueError, match="positive"):
        PressureLevels((500, 0))
    with pytest.raises(ValueError, match="repeat"):
        PressureLevels((500, 500))
    with pytest.raises(ValueError, match="equal length"):
        HybridLevels((0.0,), (0.0, 1.0))
    with pytest.raises(ValueError, match="increasing"):
        HybridLevels((0.0, 0.0), (1.0, 0.5))
    levels = HybridLevels.from_interfaces([0.0, 20000.0, 0.0], [0.0, 0.0, 1.0])
    assert levels == HybridLevels((10000.0, 10000.0), (0.0, 0.5))
    assert len(levels) == 2
    torch.testing.assert_close(
        levels.pressure(torch.tensor([100000.0])),
        torch.tensor([[10000.0, 60000.0]]),
    )


def test_interpolation_is_linear_in_log_pressure():
    pressure = torch.tensor([30000.0, 50000.0, 100000.0], dtype=torch.float64)
    x = torch.log(pressure).reshape(1, 3)
    target = torch.tensor([40000.0, 85000.0], dtype=torch.float64)
    torch.testing.assert_close(
        interpolate_pressure(x, pressure, target), torch.log(target).reshape(1, 2)
    )
    # Unordered one-dimensional pressures are sorted with the values.
    order = [2, 0, 1]
    torch.testing.assert_close(
        interpolate_pressure(x[:, order], pressure[order], target),
        torch.log(target).reshape(1, 2),
    )
    # Per-column pressure along any dimension.
    columns = torch.stack([pressure, pressure * 0.9])
    values = torch.log(columns).T.unsqueeze(-1)  # (level, column, 1)
    out = interpolate_pressure(values, columns.T.unsqueeze(-1), target, dim=0)
    assert out.shape == (2, 2, 1)
    torch.testing.assert_close(out[:, 0, 0], torch.log(target))
    torch.testing.assert_close(out[:, 1, 0], torch.log(target))


def test_half_precision_fields_interpolate_in_single_precision():
    # 850 and 1000 hPa in Pa overflow float16; the result keeps the field dtype.
    x = torch.tensor([1.0, 2.0], dtype=torch.float16)
    out = interpolate_pressure(
        x, torch.tensor([85000.0, 100000.0]), torch.tensor([92500.0])
    )
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out.float(), torch.tensor([1.5205]), atol=1e-3, rtol=0)


def test_targets_outside_the_column_clamp_or_fill():
    pressure = torch.tensor([30000.0, 100000.0])
    x = torch.tensor([1.0, 2.0])
    target = torch.tensor([10000.0, 105000.0])
    assert interpolate_pressure(x, pressure, target).tolist() == [1.0, 2.0]
    filled = interpolate_pressure(x, pressure, target, fill_value=-1.0)
    assert filled.tolist() == [-1.0, -1.0]
    with pytest.raises(ValueError, match="at least two"):
        interpolate_pressure(x[:1], pressure[:1], target)
    with pytest.raises(ValueError, match="3 pressures given for 2 levels"):
        interpolate_pressure(x, torch.ones(3), target)

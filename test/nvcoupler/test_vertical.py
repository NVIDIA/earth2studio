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
import pytest
import torch

from earth2studio.nvcoupler.errors import VerticalMismatchError
from earth2studio.nvcoupler.vertical import (
    HybridLevels,
    PressureLevels,
    interp_to_pressure,
)


def _column(values, nlat=4, nlon=8, levels=None):
    """(level, lat, lon) tensor with a horizontally-uniform column.

    `levels` sets the 'level' coordinate (hPa for pressure sources, model
    level index for hybrid sources); defaults to a 0..N-1 index.
    """
    lev = torch.tensor(values, dtype=torch.float64)
    x = lev.view(-1, 1, 1).expand(len(values), nlat, nlon).clone()
    if levels is None:
        levels = np.arange(len(values), dtype=np.float64)
    coords = OrderedDict(
        {
            "level": np.asarray(levels, dtype=np.float64),
            "lat": np.linspace(90, -90, nlat),
            "lon": np.linspace(0, 360, nlon, endpoint=False),
        }
    )
    return x, coords


def test_pressure_to_pressure_linear_in_logp():
    # Field varies linearly in log-p: f = log(p). Interpolation onto any
    # intermediate level must be exact.
    src_levels = (100.0, 300.0, 500.0, 850.0, 1000.0)
    f = np.log(np.array(src_levels) * 100.0)
    x, coords = _column(f, levels=src_levels)
    dst = PressureLevels((200.0, 400.0, 700.0))
    out, out_coords = interp_to_pressure(x, coords, PressureLevels(src_levels), dst)
    expected = np.log(np.array(dst.levels) * 100.0)
    assert np.allclose(out[:, 0, 0].numpy(), expected)
    assert list(out_coords["level"]) == [200.0, 400.0, 700.0]


def test_hybrid_to_pressure_with_surface_pressure():
    # Hybrid column: p_k = a_k + b_k * ps. With ps = 1000 hPa the levels land
    # at 300/700/1000 hPa; field = log(p) again for exactness.
    a = (30000.0, 20000.0, 0.0)
    b = (0.0, 0.5, 1.0)
    ps_value = 100000.0
    p_src = np.array(a) + np.array(b) * ps_value  # [30000, 70000, 100000] Pa
    x, coords = _column(np.log(p_src))
    ps = torch.full((4, 8), ps_value, dtype=torch.float64)
    dst = PressureLevels((500.0, 850.0))
    out, _ = interp_to_pressure(x, coords, HybridLevels(a, b), dst, ps=ps)
    expected = np.log(np.array([50000.0, 85000.0]))
    assert np.allclose(out[:, 0, 0].numpy(), expected)


def test_hybrid_requires_ps():
    x, coords = _column([1.0, 2.0, 3.0])
    with pytest.raises(VerticalMismatchError, match="surface pressure"):
        interp_to_pressure(
            x,
            coords,
            HybridLevels((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
            PressureLevels((500.0,)),
        )


def test_clamped_at_column_ends():
    x, coords = _column([10.0, 20.0], levels=(300.0, 700.0))
    src = PressureLevels((300.0, 700.0))
    out, _ = interp_to_pressure(x, coords, src, PressureLevels((100.0, 1000.0)))
    assert torch.all(out[0] == 10.0)  # above top -> top value
    assert torch.all(out[1] == 20.0)  # below bottom -> bottom value


def test_identity_shortcut_and_missing_level_dim():
    x, coords = _column([1.0, 2.0], levels=(500.0, 850.0))
    src = PressureLevels((500.0, 850.0))
    out, out_coords = interp_to_pressure(x, coords, src, PressureLevels((500.0, 850.0)))
    assert out is x  # no-op
    with pytest.raises(VerticalMismatchError, match="no 'level' dim"):
        bad = OrderedDict((k, v) for k, v in coords.items() if k != "level")
        interp_to_pressure(x[0], bad, src, PressureLevels((500.0,)))


def test_pressure_source_rejects_mismatched_level_coord():
    """Descending level data against an ascending PressureLevels source must
    raise instead of silently pairing slices with wrong pressures."""
    src = PressureLevels((500.0, 850.0, 1000.0))
    dst = PressureLevels((700.0,))
    # data ordered bottom-to-top, source declared top-to-bottom
    x, coords = _column([3.0, 2.0, 1.0], levels=(1000.0, 850.0, 500.0))
    with pytest.raises(VerticalMismatchError, match="1000.*850.*500"):
        interp_to_pressure(x, coords, src, dst)
    # arbitrary index coords (not the declared pressures) are rejected too
    x, coords = _column([1.0, 2.0, 3.0])  # level = [0, 1, 2]
    with pytest.raises(VerticalMismatchError, match="PressureLevels"):
        interp_to_pressure(x, coords, src, dst)
    # ... even on the identity (src == dst levels) shortcut path
    x, coords = _column([1.0, 2.0, 3.0])
    with pytest.raises(VerticalMismatchError, match="level"):
        interp_to_pressure(x, coords, src, PressureLevels(src.levels))


def test_hybrid_crossing_coefficients_rejected_at_construction():
    # crossing at low surface pressure: p = [50000, 30000] at ps = 50000 Pa
    with pytest.raises(ValueError, match="non-increasing"):
        HybridLevels((0.0, 30000.0), (1.0, 0.0))
    # increasing at ps = 50000 Pa but crossing at ps = 110000 Pa
    with pytest.raises(ValueError, match="non-increasing"):
        HybridLevels((0.0, 60000.0), (1.0, 0.0))
    # non-crossing across the whole plausible ps range is accepted
    HybridLevels((30000.0, 20000.0, 0.0), (0.0, 0.5, 1.0))


def test_hybrid_crossing_at_runtime_rejected():
    """Coefficients valid for plausible ps can still cross for extreme ps
    values; the interpolation-time monotonicity check must catch that."""
    hybrid = HybridLevels((0.0, 115000.0), (1.0, 0.0))  # ok in [50k, 110k] Pa
    x, coords = _column([1.0, 2.0])
    ps = torch.full((4, 8), 140000.0, dtype=torch.float64)  # crosses level 2
    with pytest.raises(VerticalMismatchError, match="strictly increasing"):
        interp_to_pressure(x, coords, hybrid, PressureLevels((1000.0,)), ps=ps)


def test_gradient_flows_through_interpolation():
    x, coords = _column([1.0, 2.0, 3.0], levels=(300.0, 500.0, 1000.0))
    x.requires_grad_(True)
    out, _ = interp_to_pressure(
        x, coords, PressureLevels((300.0, 500.0, 1000.0)), PressureLevels((400.0,))
    )
    out.sum().backward()
    assert x.grad is not None
    assert torch.all(x.grad[2] == 0)  # bottom level unused for 400 hPa target
    assert torch.all(x.grad[:2] > 0)

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

from earth2studio.data.utils_ir import (
    C1,
    C2,
    airs_wavenumber_table,
    brightness_temperature,
    cris_radiance_mw,
    iasi_radiance_mw,
    wavenumber_cm_inverse,
)


def test_iasi_wavenumber_grid():
    # Uniform 0.25 cm⁻¹ grid starting at 645.0 cm⁻¹
    out = wavenumber_cm_inverse("iasi", [1, 2, 8461])
    assert out.tolist() == [645.0, 645.25, 2760.0]
    with pytest.raises(ValueError):
        wavenumber_cm_inverse("iasi", [0])
    with pytest.raises(ValueError):
        wavenumber_cm_inverse("iasi", [8462])


def test_cris_band_wavenumbers():
    # Long-wave start, long-wave end, mid-wave start, short-wave start/end
    out = wavenumber_cm_inverse("cris", [1, 713, 714, 1579, 2211])
    assert out.tolist() == [650.0, 1095.0, 1210.0, 2155.0, 2550.0]
    with pytest.raises(ValueError):
        wavenumber_cm_inverse("cris", [2212])


def test_airs_wavenumber_table():
    table = airs_wavenumber_table()
    assert len(table) == 281
    assert wavenumber_cm_inverse("airs", [1]).tolist() == [649.62]
    # Channels outside the published aggregate subset are rejected
    with pytest.raises(ValueError):
        wavenumber_cm_inverse("airs", [2])


def test_unknown_sensor_rejected():
    with pytest.raises(ValueError):
        wavenumber_cm_inverse("modis", [1])


def test_brightness_temperature_planck_roundtrip():
    wavenumber = np.array([700.0, 1000.0, 2400.0])
    temperature = np.array([220.0, 280.0, 310.0])
    radiance = C1 * wavenumber**3 / (np.exp(C2 * wavenumber / temperature) - 1.0)
    out = brightness_temperature(radiance, wavenumber)
    np.testing.assert_allclose(out, temperature, rtol=1e-12)


def test_brightness_temperature_nonpositive_radiance_is_nan():
    out = brightness_temperature([0.0, -1.0], [700.0, 700.0])
    assert np.isnan(out).all()


def test_radiance_unit_conversions():
    # IASI: SCRA × 10^-CHSF W m⁻² sr⁻¹ m⁻¹ → mW m⁻² sr⁻¹ (cm⁻¹)⁻¹ is ×1e5
    assert iasi_radiance_mw([2.0], [7]).tolist() == [2.0 * 1e-7 * 1e5]
    # CrIS: W → mW
    assert cris_radiance_mw([0.5]).tolist() == [500.0]

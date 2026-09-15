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

from earth2studio.data.utils import PLANCK_C1, PLANCK_C2, radiance_to_bt
from earth2studio.data.utils_ir import (
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


def test_cris_grid_agrees_with_jpss_cris():
    # The CrIS FSR spectral axis is also defined by the JPSS granule source;
    # this asserts the two cannot drift apart silently.
    from earth2studio.data.jpss_cris import _CRIS_WAVENUMBER_APOD

    channels = np.arange(1, _CRIS_WAVENUMBER_APOD.size + 1)
    np.testing.assert_array_equal(
        wavenumber_cm_inverse("cris", channels), _CRIS_WAVENUMBER_APOD
    )


def test_iasi_grid_agrees_with_metop_iasi():
    # metop_iasi reads its axis from GIADR calibration data at runtime, so
    # only the channel count is a shared static constant; the grid formula
    # endpoints are pinned in test_iasi_wavenumber_grid above.
    from earth2studio.data.metop_iasi import _NUM_CHANNELS
    from earth2studio.data.utils_ir import IASI_CHANNELS

    assert IASI_CHANNELS == _NUM_CHANNELS


def test_sensors_without_formulaic_grid_rejected():
    # AIRS wavenumbers come from the per-channel LOGRCW field in the BUFR,
    # not from a formulaic grid
    with pytest.raises(ValueError):
        wavenumber_cm_inverse("airs", [1])
    with pytest.raises(ValueError):
        wavenumber_cm_inverse("modis", [1])


def test_planck_constants_pinned():
    # CODATA-2018 first and second radiation constants in the CRTM/GSI units
    # (mW m⁻² sr⁻¹ cm⁴ and cm·K). Pinned so a unit or value regression cannot
    # pass the roundtrip tests below unnoticed.
    assert PLANCK_C1 == 1.191042972e-5
    assert PLANCK_C2 == 1.438776877


def test_radiance_to_bt_planck_roundtrip():
    wavenumber = np.array([700.0, 1000.0, 2400.0])
    temperature = np.array([220.0, 280.0, 310.0])
    radiance = (
        PLANCK_C1 * wavenumber**3 / (np.exp(PLANCK_C2 * wavenumber / temperature) - 1.0)
    )
    out = radiance_to_bt(radiance, wavenumber)
    np.testing.assert_allclose(out, temperature, rtol=1e-12)


def test_radiance_to_bt_nonpositive_radiance_is_nan():
    out = radiance_to_bt(np.array([0.0, -1.0]), 700.0)
    assert np.isnan(out).all()


def test_radiance_unit_conversions():
    # IASI: SCRA × 10^-CHSF W m⁻² sr⁻¹ m⁻¹ → mW m⁻² sr⁻¹ (cm⁻¹)⁻¹ is ×1e5
    assert iasi_radiance_mw([2.0], [7]).tolist() == [2.0 * 1e-7 * 1e5]
    # CrIS: W → mW
    assert cris_radiance_mw([0.5]).tolist() == [500.0]

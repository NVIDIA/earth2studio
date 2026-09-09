# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for source-only GPS-RO vertical coordinates."""

import numpy as np
import pytest

from earth2studio.data import utils_gpsro

RADIUS = 6_371_000.0
SCALE_HEIGHT = 7_000.0
N_SURFACE = 300.0


def _exponential_profile(top_m: float = 60_000.0, step_m: float = 100.0):
    height = np.arange(0.0, top_m + step_m, step_m)
    return height, N_SURFACE * np.exp(-height / SCALE_HEIGHT)


def test_height_to_pressure_hpa_matches_ussa_layer_bases():
    heights = np.array([0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0])
    expected = np.array([1013.25, 226.321, 54.7489, 8.68019, 1.10906, 0.669387])
    assert utils_gpsro.height_to_pressure_hpa(heights) == pytest.approx(
        expected, rel=1e-3
    )
    # Clipped, monotone decreasing, finite everywhere.
    p = utils_gpsro.height_to_pressure_hpa(np.array([-500.0, 0.0, 30_000.0, 90_000.0]))
    assert np.all(np.isfinite(p))
    assert p[0] == p[1]
    assert np.all(np.diff(p[1:]) < 0)


def test_dry_pressure_profile_recovers_analytic_exponential_column():
    height, refractivity = _exponential_profile()
    h, p_hpa = utils_gpsro.dry_pressure_profile_hpa(height, refractivity)
    density = 100.0 * refractivity / (utils_gpsro.K1_REFRACTIVITY * utils_gpsro.R_DRY)
    expected_hpa = density * utils_gpsro.G0 * SCALE_HEIGHT / 100.0
    np.testing.assert_allclose(h, height)
    np.testing.assert_allclose(p_hpa, expected_hpa, rtol=2e-3)


def test_dry_pressure_profile_is_order_and_gap_robust():
    height, refractivity = _exponential_profile()
    shuffled = np.random.default_rng(0).permutation(height.size)
    h, p = utils_gpsro.dry_pressure_profile_hpa(
        height[shuffled], refractivity[shuffled]
    )
    assert np.all(np.diff(h) > 0)
    assert np.all(np.diff(p) < 0)

    with_gaps = refractivity.copy()
    with_gaps[::7] = np.nan
    h_gap, p_gap = utils_gpsro.dry_pressure_profile_hpa(height, with_gaps)
    assert h_gap.size == np.isfinite(with_gaps).sum()
    assert np.all(np.isfinite(p_gap))


def test_dry_pressure_profile_requires_two_levels():
    h, p = utils_gpsro.dry_pressure_profile_hpa(np.array([1000.0]), np.array([250.0]))
    assert h.size == 0 and p.size == 0
    out = utils_gpsro.dry_pressure_hpa(
        np.array([1000.0]), np.array([1000.0]), np.array([250.0])
    )
    assert np.isnan(out).all()


def test_dry_pressure_hpa_is_nan_outside_profile_span():
    height, refractivity = _exponential_profile(top_m=30_000.0)
    out = utils_gpsro.dry_pressure_hpa(
        np.array([-10.0, 5_000.0, 30_000.0, 30_001.0]), height, refractivity
    )
    assert np.isnan(out[0]) and np.isnan(out[3])
    assert np.isfinite(out[1]) and np.isfinite(out[2])


def test_refraction_corrected_height_below_geometric_and_converged():
    height, refractivity = _exponential_profile()
    geometric = np.array([3_000.0, 10_000.0, 40_000.0])
    impact = geometric + RADIUS
    corrected = utils_gpsro.refraction_corrected_height(
        impact, RADIUS, height, refractivity
    ).astype(np.float64)
    # Refraction lowers the tangent point by ~1e-6 N r: ~1.6 km in the lower
    # troposphere, a few meters at 40 km.
    assert np.all(corrected < geometric)
    assert corrected[0] == pytest.approx(geometric[0] - 1.6e3, abs=3e2)
    assert corrected[2] == pytest.approx(geometric[2], abs=50.0)
    # Fixed point: n(h) (R + h) == a, to within the float32 resolution of the
    # returned height (~4 mm at 40 km).
    n = 1.0 + 1e-6 * np.interp(corrected, height, refractivity)
    np.testing.assert_allclose(n * (RADIUS + corrected), impact, rtol=0, atol=1e-2)


def test_refraction_corrected_height_needs_profile():
    out = utils_gpsro.refraction_corrected_height(
        np.array([RADIUS + 5_000.0]), RADIUS, np.array([]), np.array([])
    )
    assert np.isnan(out).all()


def test_blended_pressure_weights_below_and_above_5km():
    height = np.array([1_000.0, 4_999.0, 5_000.0, 20_000.0])
    dry = np.array([900.0, 540.0, 540.0, 55.0])
    standard = np.array([898.0, 541.0, 541.0, 56.0])
    out = utils_gpsro.blended_pressure_hpa(height, dry, standard)
    assert out[0] == pytest.approx(0.8 * 898.0 + 0.2 * 900.0)
    assert out[1] == pytest.approx(0.8 * 541.0 + 0.2 * 540.0)
    assert out[2] == pytest.approx(540.0)
    assert out[3] == pytest.approx(55.0)
    assert np.isnan(
        utils_gpsro.blended_pressure_hpa(
            np.array([1_000.0]), np.array([np.nan]), np.array([900.0])
        )
    ).all()


def test_gpsro_level_coordinates_fallback_chain():
    height, refractivity = _exponential_profile(top_m=30_000.0)
    impact = RADIUS + np.array([2_000.0, 20_000.0, 45_000.0])
    pressure, level_height = utils_gpsro.gpsro_level_coordinates(
        impact, RADIUS, height, refractivity
    )
    assert np.all(np.isfinite(pressure)) and np.all(pressure > 0)
    assert np.all(np.diff(pressure) < 0)
    # Inside the profile: blended (below 5 km) then dry; outside: standard at the
    # corrected height, never the geometric height when the profile exists.
    corrected = utils_gpsro.refraction_corrected_height(
        impact, RADIUS, height, refractivity
    )
    np.testing.assert_allclose(level_height, corrected)
    dry = utils_gpsro.dry_pressure_hpa(corrected, height, refractivity)
    standard = utils_gpsro.height_to_pressure_hpa(corrected)
    assert pressure[0] == pytest.approx(0.8 * standard[0] + 0.2 * dry[0], rel=1e-6)
    assert pressure[1] == pytest.approx(dry[1], rel=1e-6)
    assert np.isnan(dry[2])
    assert pressure[2] == pytest.approx(standard[2], rel=1e-6)

    # No refractivity profile: geometric height and standard atmosphere on it.
    pressure, level_height = utils_gpsro.gpsro_level_coordinates(
        impact, RADIUS, np.array([]), np.array([])
    )
    np.testing.assert_allclose(level_height, impact - RADIUS)
    np.testing.assert_allclose(
        pressure, utils_gpsro.height_to_pressure_hpa(impact - RADIUS)
    )


def test_qfro_bit_set_uses_msb_numbering():
    # Bit 1 is the MSB (value 32768); bit 5 is 2048.
    assert utils_gpsro.qfro_bit_set(32768, 1).all()
    assert not utils_gpsro.qfro_bit_set(32768, 5).any()
    assert utils_gpsro.qfro_bit_set(2048, 5).all()
    assert utils_gpsro.qfro_bit_set(np.array([2048 + 1, 0, np.nan]), 5).tolist() == [
        True,
        False,
        False,
    ]

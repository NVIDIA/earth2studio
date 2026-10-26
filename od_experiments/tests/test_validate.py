import numpy as np
import pytest
from od_experiments.earth2_001.validate import compute_metrics, evaluate_forecast


def test_latitude_weighted_metrics():
    # Grid: 3 lats (-90, 0, 90), 2 lons
    # At poles (-90, 90), cos(lat) == 0; at equator (0), cos(lat) == 1
    lats = np.array([-90.0, 0.0, 90.0])
    
    # Forecast and reference differing only at the pole vs equator
    # Difference at equator is 10.0, at poles is 100.0
    fc = np.zeros((3, 2))
    ref = np.zeros((3, 2))
    
    ref[0, :] = 100.0  # Pole (-90): weight 0
    ref[1, :] = 10.0   # Equator (0): weight 1
    ref[2, :] = 100.0  # Pole (90): weight 0

    metrics = compute_metrics(fc, ref, lats=lats)
    # Since poles receive weight 0, the weighted MAE and RMSE must equal exactly 10.0
    assert np.isclose(metrics["mae"], 10.0)
    assert np.isclose(metrics["rmse"], 10.0)
    assert np.isclose(metrics["mean_bias"], -10.0)


def test_shape_mismatch_raises_value_error():
    fc = np.zeros((10, 10))
    ref = np.zeros((10, 12))
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_metrics(fc, ref)


def test_nan_handling_in_metrics():
    fc = np.array([[1.0, 2.0], [np.nan, 4.0]])
    ref = np.array([[1.0, 0.0], [3.0, 4.0]])
    # Valid differences: (1-1)=0, (2-0)=2, (4-4)=0. Mean abs = 2/3 = 0.6667
    metrics = compute_metrics(fc, ref)
    assert np.isclose(metrics["mae"], 0.6667, atol=1e-3)


def test_missing_reference_produces_not_validated():
    fc = {"t2m": np.zeros((5, 5))}
    report = evaluate_forecast(fc, reference_data=None)
    assert report["validation_status"] == "NOT_VALIDATED"
    assert "No reference verification data" in report["reason"]
    assert report["metrics"] == {}


def test_empty_reference_produces_not_validated():
    fc = {"t2m": np.zeros((5, 5))}
    report = evaluate_forecast(fc, reference_data={})
    assert report["validation_status"] == "NOT_VALIDATED"


def test_multidimensional_3d_4d_variable_validation():
    # 3D array: (time=2, lat=3, lon=4)
    lats = np.array([-45.0, 0.0, 45.0])
    fc_3d = np.ones((2, 3, 4)) * 250.0
    ref_3d = np.ones((2, 3, 4)) * 248.0  # constant difference of +2.0

    metrics_3d = compute_metrics(fc_3d, ref_3d, lats=lats)
    assert np.isclose(metrics_3d["mae"], 2.0)
    assert np.isclose(metrics_3d["rmse"], 2.0)
    assert np.isclose(metrics_3d["mean_bias"], 2.0)

    # 4D array: (batch=1, channel=2, lat=3, lon=4)
    fc_4d = np.full((1, 2, 3, 4), 300.0)
    ref_4d = np.full((1, 2, 3, 4), 305.0)  # constant difference of -5.0

    metrics_4d = compute_metrics(fc_4d, ref_4d, lats=lats)
    assert np.isclose(metrics_4d["mae"], 5.0)
    assert np.isclose(metrics_4d["rmse"], 5.0)
    assert np.isclose(metrics_4d["mean_bias"], -5.0)


def test_latitudinal_pole_vs_equator_weighting_ratio():
    # 5 latitudes: -90, -60, 0, 60, 90 deg
    # cos(0) = 1.0, cos(60) = 0.5, cos(90) = 0.0
    lats = np.array([-90.0, -60.0, 0.0, 60.0, 90.0])
    fc = np.zeros((5, 1))
    ref = np.zeros((5, 1))

    # Error at equator: +10.0 (weight = 1.0)
    # Error at 60 deg lat: +10.0 (weight = 0.5 each, total = 1.0)
    # Error at poles: +1000.0 (weight = 0.0)
    # Expected weighted mean bias: (10.0 * 1.0 + 10.0 * 0.5 + 10.0 * 0.5 + 0) / (1.0 + 0.5 + 0.5) = 20.0 / 2.0 = -10.0
    ref[0, 0] = 1000.0
    ref[1, 0] = 10.0
    ref[2, 0] = 10.0
    ref[3, 0] = 10.0
    ref[4, 0] = 1000.0

    metrics = compute_metrics(fc, ref, lats=lats)
    assert np.isclose(metrics["mae"], 10.0)
    assert np.isclose(metrics["rmse"], 10.0)
    assert np.isclose(metrics["mean_bias"], -10.0)


def test_uniform_offset_invariance():
    # Adding a uniform constant to both forecast and reference fields must preserve identical error metrics
    rng = np.random.default_rng(42)
    lats = np.linspace(-80.0, 80.0, 9)
    base_fc = rng.normal(loc=280.0, scale=10.0, size=(9, 12))
    base_ref = rng.normal(loc=280.0, scale=10.0, size=(9, 12))

    base_metrics = compute_metrics(base_fc, base_ref, lats=lats)

    # Shift temperature by 273.15 (e.g., Kelvin to Celsius invariant difference)
    shifted_fc = base_fc - 273.15
    shifted_ref = base_ref - 273.15

    shifted_metrics = compute_metrics(shifted_fc, shifted_ref, lats=lats)

    assert np.isclose(base_metrics["mae"], shifted_metrics["mae"], atol=1e-5)
    assert np.isclose(base_metrics["rmse"], shifted_metrics["rmse"], atol=1e-5)
    assert np.isclose(base_metrics["mean_bias"], shifted_metrics["mean_bias"], atol=1e-5)


def test_completely_missing_variable_handling():
    # Forecast contains multiple variables ('t2m', 'u10m', 'v10m')
    # Reference contains disjoint or partial variables ('z500' only, or non-overlapping)
    fc = {
        "t2m": np.full((4, 4), 280.0),
        "u10m": np.full((4, 4), 5.0),
        "v10m": np.full((4, 4), -2.0),
    }
    # Case 1: Reference has completely disjoint variables
    disjoint_ref = {"z500": np.full((4, 4), 5400.0)}
    report_disjoint = evaluate_forecast(fc, reference_data=disjoint_ref)

    assert report_disjoint["validation_status"] == "VALIDATED"
    assert report_disjoint["metrics"] == {}
    assert report_disjoint["evaluation_metadata"]["variables"] == []

    # Case 2: Reference has partial intersection ('t2m' only)
    partial_ref = {"t2m": np.full((4, 4), 282.0)}
    report_partial = evaluate_forecast(fc, reference_data=partial_ref)

    assert report_partial["validation_status"] == "VALIDATED"
    assert "t2m" in report_partial["metrics"]
    assert "u10m" not in report_partial["metrics"]
    assert np.isclose(report_partial["metrics"]["t2m"]["mean_bias"], -2.0)
    assert report_partial["evaluation_metadata"]["variables"] == ["t2m"]


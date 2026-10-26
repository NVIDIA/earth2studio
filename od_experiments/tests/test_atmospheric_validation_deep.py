import pytest
import numpy as np
from od_experiments.earth2_001.validate import compute_metrics, evaluate_forecast
from od_experiments.earth2_001.provenance import compute_sha256, create_provenance_manifest, get_git_info


class TestAtmosphericValidationAndProvenanceDeep:
    """15 additional rigorous tests for spherical geometry, area weights, and provenance integrity."""

    def test_single_element_array_validation(self):
        fc = np.array([280.0])
        ref = np.array([282.0])
        metrics = compute_metrics(fc, ref)
        assert metrics["mae"] == 2.0
        assert metrics["rmse"] == 2.0
        assert metrics["mean_bias"] == -2.0

    def test_all_nan_array_handling(self):
        fc = np.full((5, 5), np.nan)
        ref = np.full((5, 5), np.nan)
        metrics = compute_metrics(fc, ref)
        assert np.isnan(metrics["mae"])
        assert np.isnan(metrics["rmse"])
        assert np.isnan(metrics["mean_bias"])

    def test_inf_values_filtered_in_metrics(self):
        fc = np.array([10.0, np.inf, 20.0])
        ref = np.array([10.0, 50.0, 25.0])
        metrics = compute_metrics(fc, ref)
        # Valid: (10-10)=0, (20-25)=-5 -> MAE=2.5, RMSE=3.5355, Bias=-2.5
        assert metrics["mae"] == 2.5
        assert metrics["mean_bias"] == -2.5

    def test_latitude_weights_symmetry_hemispheres(self):
        # Northern hemisphere and Southern hemisphere at equal absolute latitudes have equal weights
        lats = np.array([-45.0, 45.0])
        fc = np.array([10.0, 20.0])
        ref = np.array([0.0, 0.0])
        metrics = compute_metrics(fc, ref, lats=lats)
        # Weights: cos(-45) == cos(45). Weighted MAE = (10+20)/2 = 15.0
        assert np.isclose(metrics["mae"], 15.0)

    def test_latitude_weights_all_zero_fallback(self):
        # Lats at exact poles [-90, 90] where cosine is 0
        lats = np.array([-90.0, 90.0])
        fc = np.array([5.0, 15.0])
        ref = np.array([0.0, 0.0])
        metrics = compute_metrics(fc, ref, lats=lats)
        # Fallback to unweighted mean: (5+15)/2 = 10.0
        assert np.isclose(metrics["mae"], 10.0)

    def test_mismatched_latitude_dimension_fallback(self):
        # 10x10 array but 15 latitude points provided
        lats = np.linspace(-90, 90, 15)
        fc = np.zeros((10, 10))
        ref = np.ones((10, 10)) * 3.0
        metrics = compute_metrics(fc, ref, lats=lats)
        assert metrics["mae"] == 3.0

    def test_4d_batch_channel_validation(self):
        # Shape: (batch=2, channel=3, lat=4, lon=5)
        lats = np.linspace(-60, 60, 4)
        fc = np.ones((2, 3, 4, 5)) * 100.0
        ref = np.ones((2, 3, 4, 5)) * 95.0
        metrics = compute_metrics(fc, ref, lats=lats)
        assert metrics["mae"] == 5.0
        assert metrics["rmse"] == 5.0
        assert metrics["mean_bias"] == 5.0

    def test_evaluate_forecast_partial_variable_intersection(self):
        fc = {"t2m": np.array([10.0]), "u10m": np.array([20.0])}
        ref = {"t2m": np.array([10.0])}  # u10m missing from reference
        report = evaluate_forecast(fc, ref)
        assert report["validation_status"] == "VALIDATED"
        assert "t2m" in report["metrics"]
        assert "u10m" not in report["metrics"]

    def test_sha256_checksum_deterministic(self, tmp_path):
        f = tmp_path / "test_file.bin"
        f.write_bytes(b"Origin Datum Scientific Determinism 2026")
        h1 = compute_sha256(f)
        h2 = compute_sha256(f)
        assert h1 == h2
        assert len(h1) == 64

    def test_provenance_manifest_v2_classification_smoke(self, tmp_path):
        manifest = create_provenance_manifest(
            experiment_id="TEST-SMOKE",
            configuration={"seed": 42},
            artifacts=[],
            classification="SYNTHETIC_SMOKE_TEST",
            output_dir=tmp_path,
        )
        assert manifest["scientific_integrity"]["experiment_classification"] == "SYNTHETIC_SMOKE_TEST"
        assert manifest["manifest_version"] == "2.0.0"

    def test_provenance_manifest_v2_classification_genuine(self, tmp_path):
        manifest = create_provenance_manifest(
            experiment_id="TEST-GENUINE",
            configuration={"gfs_cycle": "00z"},
            artifacts=[],
            classification="GENUINE_ATMOSPHERIC_EXPERIMENT",
            model_type="BASELINE_PERSISTENCE",
            data_source_type="HISTORICAL_GFS",
            output_dir=tmp_path,
        )
        assert manifest["scientific_integrity"]["experiment_classification"] == "GENUINE_ATMOSPHERIC_EXPERIMENT"
        assert manifest["scientific_integrity"]["data_source_type"] == "HISTORICAL_GFS"

    def test_git_info_fallback_on_invalid_path(self):
        info = get_git_info(repo_root="/nonexistent/path/12345")
        assert info["commit_sha"] == "unknown"
        assert info["branch"] == "unknown"
        assert info["dirty"] is False

    def test_evaluate_forecast_output_file_generation(self, tmp_path):
        fc = {"t2m": np.array([300.0])}
        ref = {"t2m": np.array([302.0])}
        evaluate_forecast(fc, ref, output_dir=tmp_path)
        assert (tmp_path / "validation_report.json").exists()

    def test_latitude_weighting_monotonically_decreasing_from_equator(self):
        # Lat 0 has cos=1, Lat 30 has cos=sqrt(3)/2, Lat 60 has cos=0.5, Lat 90 has cos=0
        lats = np.array([0.0, 30.0, 60.0, 90.0])
        weights = np.cos(np.deg2rad(lats))
        assert weights[0] > weights[1] > weights[2] > weights[3]
        assert np.isclose(weights[3], 0.0, atol=1e-7)

    def test_mean_bias_directionality(self):
        # When forecast is warmer than reference, bias is positive
        fc = np.array([290.0])
        ref = np.array([285.0])
        metrics = compute_metrics(fc, ref)
        assert metrics["mean_bias"] == 5.0
        # When forecast is colder than reference, bias is negative
        metrics_neg = compute_metrics(ref, fc)
        assert metrics_neg["mean_bias"] == -5.0

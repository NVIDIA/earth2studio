import numpy as np
import pytest
from od_experiments.earth2_001.validate import compute_metrics, evaluate_forecast


def test_validation_metrics_calculation():
    # Exact synthetic match
    data = np.ones((10, 10))
    res = compute_metrics(data, data)
    assert res["mae"] == 0.0
    assert res["rmse"] == 0.0
    assert res["mean_bias"] == 0.0

    # Known offset: offset by 2.0
    perturbed = data + 2.0
    res_offset = compute_metrics(perturbed, data)
    assert res_offset["mae"] == 2.0
    assert res_offset["rmse"] == 2.0
    assert res_offset["mean_bias"] == 2.0


def test_evaluate_forecast_dictionary(tmp_path):
    fc = {"t2m": np.full((10, 10), 285.0)}
    ref = {"t2m": np.full((10, 10), 280.0)}
    report = evaluate_forecast(fc, ref, output_dir=tmp_path)

    assert "t2m" in report["metrics"]
    assert report["metrics"]["t2m"]["mae"] == 5.0
    assert report["metrics"]["t2m"]["mean_bias"] == 5.0
    assert (tmp_path / "validation_report.json").exists()

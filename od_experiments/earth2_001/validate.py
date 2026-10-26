import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_metrics(
    forecast: np.ndarray,
    reference: np.ndarray,
    lats: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes MAE, RMSE, and Mean Bias between forecast and reference fields."""
    if forecast.shape != reference.shape:
        raise ValueError(f"Shape mismatch: forecast {forecast.shape} vs reference {reference.shape}")

    diff = forecast - reference

    if lats is not None:
        # Latitude weighting (cos of latitude in radians)
        weights = np.cos(np.deg2rad(lats))
        weights = np.clip(weights, 0, None)
        # Broadcast weights to match grid
        while weights.ndim < diff.ndim:
            weights = np.expand_dims(weights, axis=-1)
        weights_sum = np.sum(weights)
        if weights_sum > 0:
            mae = float(np.sum(np.abs(diff) * weights) / weights_sum)
            rmse = float(np.sqrt(np.sum((diff**2) * weights) / weights_sum))
            mean_bias = float(np.sum(diff * weights) / weights_sum)
        else:
            mae = float(np.mean(np.abs(diff)))
            rmse = float(np.sqrt(np.mean(diff**2)))
            mean_bias = float(np.mean(diff))
    else:
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        mean_bias = float(np.mean(diff))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_bias": round(mean_bias, 4),
    }


def evaluate_forecast(
    forecast_data: Dict[str, np.ndarray],
    reference_data: Dict[str, np.ndarray],
    lats: Optional[np.ndarray] = None,
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Validates multi-variable forecast against reference truth observations."""
    results: Dict[str, Any] = {
        "metrics": {},
        "evaluation_metadata": {
            "lat_weighted": lats is not None,
            "variables": list(forecast_data.keys()),
        },
    }

    for var_name, fcast_arr in forecast_data.items():
        if var_name not in reference_data:
            continue
        ref_arr = reference_data[var_name]
        results["metrics"][var_name] = compute_metrics(fcast_arr, ref_arr, lats=lats)

    if output_dir:
        out_file = Path(output_dir) / "validation_report.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results

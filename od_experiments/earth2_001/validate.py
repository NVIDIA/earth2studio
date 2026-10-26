import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_metrics(
    forecast: np.ndarray,
    reference: np.ndarray,
    lats: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Computes latitude-weighted MAE, RMSE, and Mean Bias between forecast and reference fields.
    Validates array shapes, handles NaNs gracefully, and normalizes cosine latitude weights correctly.
    """
    forecast_arr = np.asarray(forecast, dtype=np.float64)
    reference_arr = np.asarray(reference, dtype=np.float64)

    if forecast_arr.shape != reference_arr.shape:
        raise ValueError(
            f"Shape mismatch: forecast shape {forecast_arr.shape} does not match reference shape {reference_arr.shape}"
        )

    # Compute difference
    diff = forecast_arr - reference_arr

    # Mask valid (non-NaN, non-Inf) values
    valid_mask = np.isfinite(diff)
    if not np.any(valid_mask):
        return {"mae": float("nan"), "rmse": float("nan"), "mean_bias": float("nan")}

    if lats is not None:
        lats_arr = np.asarray(lats, dtype=np.float64)
        # Cosine of latitude in radians; latitude weighting clipped at 0
        weights_1d = np.cos(np.deg2rad(lats_arr))
        weights_1d = np.clip(weights_1d, 0.0, None)

        # Broadcast weights to match diff shape
        # Assume latitude dimension corresponds to matching dimension length
        lat_dim_idx = None
        for idx, dim_len in enumerate(diff.shape):
            if dim_len == len(lats_arr):
                lat_dim_idx = idx
                break

        if lat_dim_idx is not None:
            shape = [1] * diff.ndim
            shape[lat_dim_idx] = len(lats_arr)
            weights = np.reshape(weights_1d, shape)
            weights = np.broadcast_to(weights, diff.shape)
        else:
            # Fallback to unweighted if dimension does not match
            weights = np.ones_like(diff)

        # Combine with valid mask
        weights = np.where(valid_mask, weights, 0.0)
        total_weight = np.sum(weights)

        if total_weight > 0:
            mae = float(np.sum(np.abs(diff) * weights) / total_weight)
            rmse = float(np.sqrt(np.sum((diff**2) * weights) / total_weight))
            mean_bias = float(np.sum(diff * weights) / total_weight)
        else:
            mae = float(np.mean(np.abs(diff[valid_mask])))
            rmse = float(np.sqrt(np.mean(diff[valid_mask] ** 2)))
            mean_bias = float(np.mean(diff[valid_mask]))
    else:
        valid_diff = diff[valid_mask]
        mae = float(np.mean(np.abs(valid_diff)))
        rmse = float(np.sqrt(np.mean(valid_diff**2)))
        mean_bias = float(np.mean(valid_diff))

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_bias": round(mean_bias, 4),
    }


def evaluate_forecast(
    forecast_data: dict[str, np.ndarray],
    reference_data: dict[str, np.ndarray] | None = None,
    lats: np.ndarray | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Validates forecast against reference truth observations.
    If reference_data is None or empty, strictly reports NOT_VALIDATED without inventing data.
    """
    if not reference_data:
        results = {
            "validation_status": "NOT_VALIDATED",
            "reason": "No reference verification data was provided for the target verification time.",
            "metrics": {},
            "evaluation_metadata": {
                "lat_weighted": lats is not None,
                "variables": list(forecast_data.keys()),
            },
        }
    else:
        results = {
            "validation_status": "VALIDATED",
            "metrics": {},
            "evaluation_metadata": {
                "lat_weighted": lats is not None,
                "variables": [v for v in forecast_data.keys() if v in reference_data],
            },
        }
        for var_name, fcast_arr in forecast_data.items():
            if var_name not in reference_data:
                continue
            ref_arr = reference_data[var_name]
            results["metrics"][var_name] = compute_metrics(
                fcast_arr, ref_arr, lats=lats
            )

    if output_dir:
        out_file = Path(output_dir) / "validation_report.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results

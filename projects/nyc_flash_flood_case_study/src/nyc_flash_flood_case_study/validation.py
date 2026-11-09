from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio

from .config import EVENT_OUTPUTS, IMPACT_DATASETS, OUTPUTS_DIR, PROCESSED_DIR


def validate_against_observations() -> dict:
    if not IMPACT_DATASETS["nyc_311"].exists():
        return {"status": "no_observed_data_available", "valid_model_skill": False}

    observed = pd.read_csv(IMPACT_DATASETS["nyc_311"])
    required = {"lon", "lat", "impact_flag"}
    if not required.issubset(observed.columns) or not EVENT_OUTPUTS["susceptibility_map"].exists():
        return {"status": "required_observations_or_susceptibility_missing", "valid_model_skill": False}

    observed = observed.loc[observed["impact_flag"].eq(1)].copy()
    scores = []
    with rasterio.open(EVENT_OUTPUTS["susceptibility_map"]) as src:
        valid_grid = src.read(1, masked=True).compressed()
        sorted_grid = np.sort(valid_grid[np.isfinite(valid_grid)])
        for row in observed.itertuples(index=False):
            value = next(src.sample([(float(row.lon), float(row.lat))]))[0]
            if src.nodata is not None and np.isclose(value, src.nodata):
                continue
            if not np.isfinite(value):
                continue
            percentile = float(np.searchsorted(sorted_grid, value, side="right") / len(sorted_grid))
            scores.append({
                "unique_key": getattr(row, "unique_key", None),
                "created_date": getattr(row, "created_date", None),
                "lon": float(row.lon),
                "lat": float(row.lat),
                "complaint_type": getattr(row, "complaint_type", None),
                "descriptor": getattr(row, "descriptor", None),
                "susceptibility": float(value),
                "grid_percentile": percentile,
            })

    scored = pd.DataFrame(scores)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(PROCESSED_DIR / "flood_reports_scored.csv", index=False)
    if scored.empty:
        report = {
            "status": "no_observations_overlapping_valid_grid",
            "valid_model_skill": False,
            "n_observed_flood_reports": int(len(observed)),
            "n_reports_with_scores": 0,
        }
    else:
        report = {
            "status": "exploratory_spatial_overlap_only",
            "valid_model_skill": False,
            "n_observed_flood_reports": int(len(observed)),
            "n_reports_with_scores": int(len(scored)),
            "median_susceptibility_at_reports": float(scored["susceptibility"].median()),
            "mean_grid_percentile_at_reports": float(scored["grid_percentile"].mean()),
            "fraction_reports_in_top_decile": float(scored["grid_percentile"].ge(0.9).mean()),
            "skill_metrics": None,
            "limitations": [
                "NYC 311 reports are complaint locations, not independently surveyed flood depths or extents.",
                "No verified non-flood locations are available, so accuracy, ROC AUC, and calibration are not reported.",
                "Current MRMS file is a single BREF product whose decoded variable and units are unknown.",
                "Current DEM and imperviousness inputs are demo rasters, not acquired event-study products.",
                "The 40 by 18 output grid is too coarse for parcel- or street-scale flood interpretation.",
            ],
        }
    (OUTPUTS_DIR / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def summarize_validation() -> str:
    metrics = validate_against_observations()
    if metrics["status"] == "no_observed_data_available":
        return "No observed flood-impact dataset was available for comparison."
    if metrics.get("n_reports_with_scores", 0) == 0:
        return "No observed flood reports overlapped valid susceptibility cells; no skill metrics computed."
    return (
        f"Exploratory spatial overlap only: sampled {metrics['n_reports_with_scores']} of "
        f"{metrics['n_observed_flood_reports']} event flood reports. No predictive skill metrics computed."
    )


__all__ = ["validate_against_observations", "summarize_validation"]

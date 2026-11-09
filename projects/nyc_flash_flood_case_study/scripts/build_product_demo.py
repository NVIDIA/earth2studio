from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colormaps
from PIL import Image

from nyc_flash_flood_case_study.config import EVENT_OUTPUTS, FIGURES_DIR, IMPACT_DATASETS, STUDY_AREA, TERRAIN_DATASETS
from nyc_flash_flood_case_study.preprocessing import build_high_resolution_terrain_screening


SCENARIOS = {
    "moderate": {"label": "Moderate", "peak_mm_h": 35, "peak_depth_m": 0.12, "color": "#33a5a8"},
    "severe": {"label": "Severe", "peak_mm_h": 65, "peak_depth_m": 0.28, "color": "#ec9c3d"},
    "extreme": {"label": "Extreme", "peak_mm_h": 95, "peak_depth_m": 0.48, "color": "#d94b3d"},
}
TIMESTEPS_HOURS = [0, 1, 2, 3, 4, 5, 6]
RAIN_SHAPE = [0.0, 0.10, 0.24, 0.62, 1.0, 0.72, 0.36]


def _write_rgba(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)


def _scenario_frame(screening: np.ndarray, valid: np.ndarray, peak_depth: float, timestep: int) -> np.ndarray:
    percentile_cut = float(np.percentile(screening[valid], 75))
    support = np.clip((screening - percentile_cut) / max(float(np.percentile(screening[valid], 98)) - percentile_cut, 1e-6), 0, 1) ** 1.5
    accumulated = np.cumsum(RAIN_SHAPE[: timestep + 1])[-1] / sum(RAIN_SHAPE)
    depth = peak_depth * support * accumulated

    rgba = np.zeros((*screening.shape, 4), dtype=np.uint8)
    bins = [0.05, 0.15, 0.30]
    colors = [
        (55, 166, 183, 180),
        (232, 191, 70, 195),
        (235, 129, 58, 205),
        (207, 63, 58, 215),
    ]
    active = valid & (depth >= 0.02)
    class_index = np.digitize(depth, bins)
    for index, color in enumerate(colors):
        rgba[active & (class_index == index)] = color
    return rgba


def build_product_demo() -> Path:
    if not TERRAIN_DATASETS["dem_highres"].exists() or not EVENT_OUTPUTS["terrain_screening_highres"].exists():
        build_high_resolution_terrain_screening()

    with rasterio.open(EVENT_OUTPUTS["terrain_screening_highres"]) as source:
        screening = source.read(1)
        valid = source.read_masks(1) > 0
        transform = source.transform
        bounds = source.bounds
        height, width = source.height, source.width
        x_resolution, y_resolution = source.res

    output_dir = FIGURES_DIR / "product_demo_assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    terrain_rgba = colormaps["YlGnBu"](np.clip(screening, 0, 1), bytes=True)
    terrain_rgba[..., 3] = np.where(valid, 180, 0).astype(np.uint8)
    _write_rgba(output_dir / "terrain_screening.png", terrain_rgba)

    with rasterio.open(TERRAIN_DATASETS["dem_highres"]) as source:
        elevation = source.read(1, masked=True).filled(np.nan)
    elevation_valid = valid & np.isfinite(elevation)
    elevation_min, elevation_max = np.nanpercentile(elevation[elevation_valid], [2, 98])
    normalized_elevation = np.clip((np.nan_to_num(elevation, nan=elevation_min) - elevation_min) / max(elevation_max - elevation_min, 1e-6), 0, 1)
    elevation_rgba = colormaps["terrain"](normalized_elevation, bytes=True)
    elevation_rgba[..., 3] = np.where(elevation_valid, 185, 0).astype(np.uint8)
    _write_rgba(output_dir / "elevation.png", elevation_rgba)

    metrics = {"scenarios": {}, "timestep_hours": TIMESTEPS_HOURS, "rain_shape": RAIN_SHAPE}
    pixel_area_km2 = abs(x_resolution) * 111320 * np.cos(np.deg2rad((bounds.bottom + bounds.top) / 2)) * abs(y_resolution) * 111320 / 1e6
    report_points = pd.read_csv(IMPACT_DATASETS["nyc_311"]) if IMPACT_DATASETS["nyc_311"].exists() else pd.DataFrame(columns=["lon", "lat"])

    for key, scenario in SCENARIOS.items():
        frames = []
        timestep_metrics = []
        for timestep, hour in enumerate(TIMESTEPS_HOURS):
            rgba = _scenario_frame(screening, valid, scenario["peak_depth_m"], timestep)
            filename = f"{key}_hour_{hour:02d}.png"
            _write_rgba(output_dir / filename, rgba)
            frames.append(filename)

            accumulated = np.cumsum(RAIN_SHAPE[: timestep + 1])[-1] / sum(RAIN_SHAPE)
            peak_depth = scenario["peak_depth_m"] * accumulated
            screen_cut = float(np.percentile(screening[valid], 75))
            support = np.clip((screening - screen_cut) / max(float(np.percentile(screening[valid], 98)) - screen_cut, 1e-6), 0, 1) ** 1.5
            depth = scenario["peak_depth_m"] * support * accumulated
            exposed = valid & (depth >= 0.05)
            report_count = 0
            if not report_points.empty:
                rows, cols = rasterio.transform.rowcol(
                    transform,
                    report_points["lon"].to_numpy(),
                    report_points["lat"].to_numpy(),
                )
                in_grid = (np.asarray(rows) >= 0) & (np.asarray(rows) < height) & (np.asarray(cols) >= 0) & (np.asarray(cols) < width)
                report_count = int(sum(exposed[row, col] for row, col, inside in zip(rows, cols, in_grid) if inside))
            timestep_metrics.append({
                "hour": hour,
                "synthetic_rain_mm_h": round(scenario["peak_mm_h"] * RAIN_SHAPE[timestep], 1),
                "max_proxy_depth_cm": round(peak_depth * 100, 1),
                "land_area_over_5cm_km2": round(float(exposed.sum() * pixel_area_km2), 2),
                "observed_311_points_in_synthetic_footprint": report_count,
            })
        metrics["scenarios"][key] = {
            **scenario,
            "frames": frames,
            "timesteps": timestep_metrics,
        }

    if not report_points.empty:
        reports = gpd.GeoDataFrame(
            report_points,
            geometry=gpd.points_from_xy(report_points["lon"], report_points["lat"]),
            crs="EPSG:4326",
        )
        reports.to_file(output_dir / "observed_311.geojson", driver="GeoJSON")
        report_count = int(len(reports))
    else:
        report_count = 0

    manifest = {
        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "terrain_resolution_m_approx": 10,
        "land_cells": int(valid.sum()),
        "observed_311_reports": report_count,
        "scenarios_are_synthetic": True,
        "scenario_method": "Synthetic terrain-screening proxy for product-interface demonstration only; not calibrated hydrology.",
        **metrics,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    html_path = FIGURES_DIR / "product_demo.html"
    return html_path


if __name__ == "__main__":
    print(build_product_demo())
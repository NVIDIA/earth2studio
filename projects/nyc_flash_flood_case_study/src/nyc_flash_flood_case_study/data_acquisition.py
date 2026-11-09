from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from .config import DATA_DIR, EVENT, FIGURES_DIR, IMPACT_DATASETS, LANDCOVER_DATASETS, OUTPUTS_DIR, PROJECT_ROOT, RAINFALL_DATASETS, RAW_DIR, STUDY_AREA, TERRAIN_DATASETS


def ensure_directories() -> None:
    for directory in [
        DATA_DIR,
        RAW_DIR,
        RAW_DIR / "rainfall",
        RAW_DIR / "terrain",
        RAW_DIR / "landcover",
        RAW_DIR / "impacts",
        FIGURES_DIR,
        OUTPUTS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def make_study_polygon() -> gpd.GeoDataFrame:
    ensure_directories()
    xmin, ymin, xmax, ymax = (
        STUDY_AREA["bounds"]["xmin"],
        STUDY_AREA["bounds"]["ymin"],
        STUDY_AREA["bounds"]["xmax"],
        STUDY_AREA["bounds"]["ymax"],
    )
    polygon = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt([
            f"POLYGON (({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))"
        ]),
        crs=STUDY_AREA["crs"],
    )
    polygon["name"] = STUDY_AREA["name"]
    out_path = RAW_DIR / "study_area.geojson"
    polygon.to_file(out_path, driver="GeoJSON")
    return polygon


def write_demo_rainfall_and_terrain() -> None:
    ensure_directories()
    xmin, ymin, xmax, ymax = (
        STUDY_AREA["bounds"]["xmin"],
        STUDY_AREA["bounds"]["ymin"],
        STUDY_AREA["bounds"]["xmax"],
        STUDY_AREA["bounds"]["ymax"],
    )
    res = 0.0015
    nx = int((xmax - xmin) / res)
    ny = int((ymax - ymin) / res)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    xx, yy = np.meshgrid(xs, ys)

    lowland = 0.3 + 20.0 * np.exp(-((xx + 0.012) ** 2 + (yy - 40.645) ** 2) / 0.0004)
    rainfall = 15.0 * np.exp(-((xx + 0.14) ** 2 + (yy - 40.655) ** 2) / 0.0006)
    rainfall += 10.0 * np.exp(-((xx + 0.05) ** 2 + (yy - 40.620) ** 2) / 0.0005)
    rainfall = np.clip(rainfall, 0, 60)
    rain_total = rainfall * 0.75 + lowland * 0.2
    impervious = 0.55 + 0.35 * np.exp(-((xx + 0.08) ** 2 + (yy - 40.64) ** 2) / 0.0009)
    antecedent = 0.40 + 0.45 * np.exp(-((xx + 0.02) ** 2 + (yy - 40.63) ** 2) / 0.0008)
    dem = 11.0 + 2.5 * np.exp(-((xx + 0.1) ** 2 + (yy - 40.652) ** 2) / 0.0014)
    dem = dem + 5.0 * (np.maximum(0, 0.8 - yy + 40.6))
    dem = np.flipud(dem)
    rainfall = np.flipud(rainfall)
    impervious = np.flipud(impervious)
    antecedent = np.flipud(antecedent)

    transform = from_origin(xmin, ymax, res, res)
    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
    }

    for array, path in [
        (dem.astype(np.float32), TERRAIN_DATASETS["dem"]),
        (rain_total.astype(np.float32), RAW_DIR / "rainfall" / "demo_rainfall_total.tif"),
        (impervious.astype(np.float32), LANDCOVER_DATASETS["impervious"]),
        (antecedent.astype(np.float32), RAW_DIR / "antecedent_wetness_demo.tif"),
    ]:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array, 1)

    points = [
        (-73.83, 40.645, 1),
        (-73.86, 40.635, 1),
        (-73.79, 40.628, 1),
        (-73.91, 40.610, 0),
    ]
    df = pd.DataFrame(points, columns=["lon", "lat", "impact_flag"])
    df.to_csv(IMPACT_DATASETS["nyc_311"], index=False)

    flood_shape = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    flood_shape["impact_flag"] = df["impact_flag"]
    flood_shape.to_file(IMPACT_DATASETS["flood_reports"], driver="GeoJSON")

    if not (RAW_DIR / "study_area.geojson").exists():
        study_area = gpd.GeoDataFrame(
            geometry=gpd.GeoSeries.from_wkt([
                f"POLYGON (({xmin} {ymin}, {xmax} {ymin}, {xmax} {ymax}, {xmin} {ymax}, {xmin} {ymin}))"
            ]),
            crs="EPSG:4326",
        )
        study_area["name"] = STUDY_AREA["name"]
        study_area.to_file(RAW_DIR / "study_area.geojson", driver="GeoJSON")


def fetch_open_datasets() -> None:
    ensure_directories()
    if not (RAW_DIR / "study_area.geojson").exists():
        study_area = make_study_polygon()
        study_area.to_file(RAW_DIR / "study_area.geojson", driver="GeoJSON")

    if not any(path.exists() for path in [
        TERRAIN_DATASETS["dem"],
        LANDCOVER_DATASETS["impervious"],
        RAW_DIR / "rainfall" / "demo_rainfall_total.tif",
    ]):
        write_demo_rainfall_and_terrain()

    if not IMPACT_DATASETS["nyc_311"].exists():
        write_demo_rainfall_and_terrain()


__all__ = [
    "ensure_directories",
    "make_study_polygon",
    "write_demo_rainfall_and_terrain",
    "fetch_open_datasets",
]

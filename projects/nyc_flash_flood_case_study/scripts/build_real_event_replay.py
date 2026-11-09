from __future__ import annotations

import gzip
import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
from matplotlib import colormaps
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point

from nyc_flash_flood_case_study.config import FIGURES_DIR, IMPACT_DATASETS, RAW_DIR, STUDY_AREA, TERRAIN_DATASETS


MRMS_QPE_PRODUCT = "MultiSensor_QPE_01H_Pass1_00.00"
MRMS_FLASH_PRODUCT = "FLASH_CREST_MAXSTREAMFLOW_00.00"
FLASH_STEP_MINUTES = 30
EVENT_START = datetime(2021, 9, 1, tzinfo=timezone.utc)
EVENT_END = datetime(2021, 9, 2, tzinfo=timezone.utc)


def _download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _fetch_grib(product: str, timestamp: datetime, cache_dir: Path) -> Path:
    stamp = timestamp.strftime("%Y%m%d-%H%M%S")
    filename = f"MRMS_{product}_{stamp}.grib2.gz"
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{product}/{timestamp:%Y%m%d}/{filename}"
    return _download(url, cache_dir / filename)


def _decode_study_crop(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = STUDY_AREA["bounds"]
    with gzip.open(path, "rb") as compressed, tempfile.NamedTemporaryFile(suffix=".grib2") as raw:
        shutil.copyfileobj(compressed, raw)
        raw.flush()
        dataset = xr.open_dataset(raw.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
        try:
            variable = dataset[next(iter(dataset.data_vars))]
            values = variable.values.astype(np.float32)
            latitude = dataset["latitude"].values.astype(np.float64)
            longitude = dataset["longitude"].values.astype(np.float64)
        finally:
            dataset.close()

    longitude = np.where(longitude > 180, longitude - 360, longitude)
    latitude_indices = np.flatnonzero((latitude >= bounds["ymin"] - 0.02) & (latitude <= bounds["ymax"] + 0.02))
    longitude_indices = np.flatnonzero((longitude >= bounds["xmin"] - 0.02) & (longitude <= bounds["xmax"] + 0.02))
    if latitude_indices.size == 0 or longitude_indices.size == 0:
        raise ValueError(f"No MRMS grid cells intersect study area in {path.name}.")
    crop = values[np.ix_(latitude_indices, longitude_indices)]
    return crop, latitude[latitude_indices], longitude[longitude_indices]


def _rgba_image(values: np.ndarray, valid: np.ndarray, cmap_name: str, maximum: float) -> np.ndarray:
    normalized = np.clip(np.nan_to_num(values, nan=0.0) / max(maximum, 1e-6), 0, 1)
    rgba = colormaps[cmap_name](normalized, bytes=True)
    rgba[..., 3] = np.where(valid, np.minimum(220, 80 + normalized * 140).astype(np.uint8), 0)
    return rgba


def _write_image(path: Path, rgba: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)


def _get_catch_basins(output_dir: Path) -> tuple[Path, int, int]:
    bounds = STUDY_AREA["bounds"]
    raw_dir = RAW_DIR / "drainage"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "nycdep_catch_basins_south_queens.json"
    if not raw_path.exists():
        records = []
        offset = 0
        while True:
            response = requests.get(
                "https://data.cityofnewyork.us/resource/2w2g-fk3i.json",
                params={
                    "$select": "unitid,latitude,longitude,the_geom",
                    "$where": f"within_box(the_geom,{bounds['ymin']},{bounds['xmin']},{bounds['ymax']},{bounds['xmax']})",
                    "$limit": 50000,
                    "$offset": offset,
                },
                timeout=90,
            )
            response.raise_for_status()
            batch = response.json()
            records.extend(batch)
            if len(batch) < 50000:
                break
            offset += len(batch)
        raw_path.write_text(json.dumps(records))
    else:
        records = json.loads(raw_path.read_text())

    records = [record for record in records if record.get("longitude") and record.get("latitude")]
    basins = gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy(
            [float(record["longitude"]) for record in records],
            [float(record["latitude"]) for record in records],
        ),
        crs="EPSG:4326",
    )
    raw_count = len(basins)
    if basins.empty:
        return output_dir / "catch_basin_density.geojson", 0, 0

    projected = basins.to_crs("EPSG:26918")
    projected["grid_x"] = np.floor(projected.geometry.x / 500).astype(np.int64)
    projected["grid_y"] = np.floor(projected.geometry.y / 500).astype(np.int64)
    density = projected.groupby(["grid_x", "grid_y"], as_index=False).agg(catch_basins=("unitid", "count"))
    density["geometry"] = [
        Point((row.grid_x + 0.5) * 500, (row.grid_y + 0.5) * 500)
        for row in density.itertuples(index=False)
    ]
    density = gpd.GeoDataFrame(density, geometry="geometry", crs="EPSG:26918").to_crs("EPSG:4326")
    output = output_dir / "catch_basin_density.geojson"
    density.to_file(output, driver="GeoJSON")
    return output, raw_count, len(density)


def build_real_event_replay() -> Path:
    asset_dir = FIGURES_DIR / "product_demo_assets" / "observed_ida"
    qpe_cache = RAW_DIR / "rainfall" / "mrms_ida_qpe_1h"
    flash_cache = RAW_DIR / "rainfall" / "mrms_ida_flash_crest"

    qpe_data = {}
    qpe_coordinates = None
    for hour in range(24):
        timestamp = EVENT_START + timedelta(hours=hour)
        raw_path = _fetch_grib(MRMS_QPE_PRODUCT, timestamp, qpe_cache)
        values, latitude, longitude = _decode_study_crop(raw_path)
        qpe_data[hour] = np.where((values >= 0) & np.isfinite(values), values, np.nan)
        qpe_coordinates = (latitude, longitude)
        print(f"Loaded MRMS hourly QPE {timestamp:%H:%M} UTC")

    flash_data = {}
    for step in range(int((EVENT_END - EVENT_START).total_seconds() // (FLASH_STEP_MINUTES * 60))):
        timestamp = EVENT_START + timedelta(minutes=step * FLASH_STEP_MINUTES)
        raw_path = _fetch_grib(MRMS_FLASH_PRODUCT, timestamp, flash_cache)
        values, latitude, longitude = _decode_study_crop(raw_path)
        values = np.where((values >= 0) & np.isfinite(values), values, np.nan)
        flash_data[timestamp] = values
        if qpe_coordinates is not None and (not np.array_equal(latitude, qpe_coordinates[0]) or not np.array_equal(longitude, qpe_coordinates[1])):
            raise ValueError("MRMS QPE and FLASH grids do not align; refusing to generate a misregistered replay.")
        print(f"Loaded NOAA FLASH streamflow {timestamp:%H:%M} UTC")

    all_qpe = np.concatenate([array[np.isfinite(array)] for array in qpe_data.values()])
    all_flash = np.concatenate([array[np.isfinite(array)] for array in flash_data.values()])
    qpe_max = max(float(np.percentile(all_qpe, 99)), 1.0)
    flash_max = max(float(np.percentile(all_flash, 99.5)), 1.0)
    latitude, longitude = qpe_coordinates
    qpe_valid = np.isfinite(qpe_data[0])
    lat_step = float(np.median(np.abs(np.diff(latitude))))
    lon_step = float(np.median(np.abs(np.diff(longitude))))
    grid_transform = from_origin(longitude.min() - lon_step / 2, latitude.max() + lat_step / 2, lon_step, lat_step)

    with rasterio.open(TERRAIN_DATASETS["dem_highres"]) as dem:
        dem_transform, dem_crs = dem.transform, dem.crs
        land_mask = rasterize(
            [],
            out_shape=(len(latitude), len(longitude)),
            transform=grid_transform,
            fill=0,
            dtype="uint8",
        )
        if EVENT_OUTPUTS_LAND_MASK := (RAW_DIR / "landcover" / "ne_10m_land.zip").exists():
            land_geometry = gpd.read_file(f"zip://{RAW_DIR / 'landcover' / 'ne_10m_land.zip'}").to_crs(dem_crs)
            land_mask = rasterize(
                ((geometry, 1) for geometry in land_geometry.geometry if geometry is not None and not geometry.is_empty),
                out_shape=(len(latitude), len(longitude)),
                transform=grid_transform,
                fill=0,
                dtype="uint8",
            )

    timestamps = sorted(flash_data)
    frames = []
    for index, timestamp in enumerate(timestamps):
        qpe_hour = timestamp.hour
        qpe_values = qpe_data[qpe_hour]
        flash_values = flash_data[timestamp]
        qpe_image = _rgba_image(qpe_values, np.isfinite(qpe_values), "Blues", qpe_max)
        flash_valid = np.isfinite(flash_values) & (land_mask == 1)
        flash_image = _rgba_image(flash_values, flash_valid, "YlOrRd", flash_max)
        qpe_name = f"qpe_{timestamp:%H%M}.png"
        flash_name = f"flash_{timestamp:%H%M}.png"
        _write_image(asset_dir / qpe_name, qpe_image)
        _write_image(asset_dir / flash_name, flash_image)
        frames.append({
            "valid_time_utc": timestamp.isoformat(),
            "label": timestamp.strftime("%H:%M UTC"),
            "flash_frame": f"observed_ida/{flash_name}",
            "qpe_frame": f"observed_ida/{qpe_name}",
            "qpe_hour_mm": round(float(np.nanmax(qpe_values)), 1) if np.isfinite(qpe_values).any() else 0,
            "max_streamflow_m3_s": round(float(np.nanmax(flash_values)), 1) if np.isfinite(flash_values).any() else 0,
        })

    catchbasin_path, catchbasin_count, catchbasin_cells = _get_catch_basins(asset_dir)
    reports_path = asset_dir / "observed_311.geojson"
    if IMPACT_DATASETS["nyc_311"].exists():
        reports = pd.read_csv(IMPACT_DATASETS["nyc_311"])
        gpd.GeoDataFrame(
            reports,
            geometry=gpd.points_from_xy(reports["lon"], reports["lat"]),
            crs="EPSG:4326",
        ).to_file(reports_path, driver="GeoJSON")

    manifest = {
        "event": "Hurricane Ida rainfall and hydrologic replay",
        "event_start_utc": EVENT_START.isoformat(),
        "event_end_utc": EVENT_END.isoformat(),
        "bounds": [STUDY_AREA["bounds"]["xmin"], STUDY_AREA["bounds"]["ymin"], STUDY_AREA["bounds"]["xmax"], STUDY_AREA["bounds"]["ymax"]],
        "grid_shape": list(qpe_data[0].shape),
        "grid_spacing_degrees": {"latitude": lat_step, "longitude": lon_step},
        "rainfall_source": "NOAA MRMS MultiSensor_QPE_01H_Pass1; accumulation mm over 1-hour ending at timestamp.",
        "flood_model_source": "NOAA MRMS FLASH QPE-CREST operational modeled streamflow; 10-minute product sampled every 30 minutes; m3/s.",
        "catchbasin_source": "NYC DEP Citywide Catch Basins dataset 2w2g-fk3i; points aggregated to 500 m for display.",
        "catchbasin_count": catchbasin_count,
        "catchbasin_display_cells": catchbasin_cells,
        "qpe_display_max_mm": qpe_max,
        "flash_display_max_m3_s": flash_max,
        "frames": frames,
        "synthetic_fallback": "The synthetic scenario mode remains available when comparing illustrative alternatives; it is not shown as observed Ida data.",
        "limitations": [
            "FLASH is a regional hydrologic model, not an urban sewer or street-depth hydraulic model.",
            "The MRMS grid is about 1 km; high-resolution terrain is shown separately and does not increase rainfall-model resolution.",
            "NYC catch-basin points do not provide pipe connections, inlet capacities, or calibrated sewer hydraulics.",
            "311 reports are complaint locations, not verified flood extents or depths.",
        ],
    }
    output = FIGURES_DIR / "product_demo_assets" / "real_ida_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return output


if __name__ == "__main__":
    print(build_real_event_replay())
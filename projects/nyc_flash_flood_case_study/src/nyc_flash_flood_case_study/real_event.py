from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
from rasterio.transform import from_origin

from .config import DATA_DIR, EVENT_OUTPUTS, IMPACT_DATASETS, LANDCOVER_DATASETS, PROCESSED_DIR, RAW_DIR, STUDY_AREA, TERRAIN_DATASETS
from .data_acquisition import write_demo_rainfall_and_terrain
from .preprocessing import build_antecedent_wetness, build_terrain_layers, build_validation_points


IDA_EVENT = {
    "name": "Hurricane Ida remnants over NYC",
    "start": "2021-09-01T00:00:00Z",
    "end": "2021-09-02T23:59:59Z",
    "study_bounds": {
        "xmin": -74.12,
        "ymin": 40.55,
        "xmax": -73.72,
        "ymax": 40.73,
    },
}


@dataclass
class SourceManifest:
    name: str
    url: str
    destination: Path
    description: str


def _ensure_dirs() -> None:
    for target in [
        DATA_DIR,
        RAW_DIR,
        RAW_DIR / "rainfall",
        RAW_DIR / "terrain",
        RAW_DIR / "landcover",
        RAW_DIR / "impacts",
    ]:
        target.mkdir(parents=True, exist_ok=True)


def download_url(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not fetch {url}: {exc}") from exc

    with destination.open("wb") as handle:
        handle.write(response.content)
    return destination


def build_real_event_manifest() -> List[SourceManifest]:
    _ensure_dirs()

    rainfall = SourceManifest(
        name="noaa_mrms_bucket_index",
        url="https://noaa-mrms-pds.s3.amazonaws.com/",
        destination=RAW_DIR / "rainfall" / "mrms_bucket_index.xml",
        description="Public NOAA MRMS bucket index used to locate the precipitation archive for the event window.",
    )
    terrain = SourceManifest(
        name="usgs_3dep_index",
        url="https://prd-tnm.s3.amazonaws.com/",
        destination=RAW_DIR / "terrain" / "usgs_3dep_index.html",
        description="USGS 3DEP public S3 index used to locate the elevation product for the study area.",
    )
    impervious = SourceManifest(
        name="nlcd_source_page",
        url="https://www.mrlc.gov/data",
        destination=RAW_DIR / "landcover" / "nlcd_impervious_source.html",
        description="NLCD imperviousness source page for the event study area.",
    )
    nyc_311 = SourceManifest(
        name="nyc_311_service_requests",
        url="https://data.cityofnewyork.us/resource/erm2-nwe9.json?$where=created_date between '2021-09-01T00:00:00' and '2021-09-02T23:59:59'&$limit=50000",
        destination=RAW_DIR / "impacts" / "nyc_311_ida.json",
        description="NYC 311 service requests for the Ida event window, using the validated 311 resource ID.",
    )
    return [rainfall, terrain, impervious, nyc_311]


def download_mrms_event_file(date: str = "20210901", hour: str = "000000", product: str = "BREF_1HR_MAX_00.50") -> Path:
    """Download a real Ida-era MRMS product file from the public NOAA S3 archive."""
    target_dir = RAW_DIR / "rainfall" / "mrms_event"
    target_dir.mkdir(parents=True, exist_ok=True)
    filename = f"MRMS_{product}_{date}-{hour}.grib2.gz"
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{product}/{date}/{filename}"
    destination = target_dir / filename
    if destination.exists():
        return destination
    with requests.get(url, timeout=120) as response:
        response.raise_for_status()
        destination.write_bytes(response.content)
    return destination


def convert_mrms_grib_to_raster(grib_gz_path: Path, output_path: Path, lat_bounds=(20.0, 55.0), lon_bounds=(-130.0, -60.0)) -> Path:
    """Convert a downloaded MRMS GRIB2 file to a clipped GeoTIFF in EPSG:4326 using the NYC study extent."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decompressed_path = grib_gz_path.with_suffix("")
    if grib_gz_path.suffix == ".gz":
        decompressed_path = grib_gz_path.with_suffix("")
        if not decompressed_path.exists():
            with gzip.open(grib_gz_path, "rb") as f_in, open(decompressed_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

    ds = xr.open_dataset(str(decompressed_path), engine="cfgrib", backend_kwargs={"indexpath": ""})
    var_name = next(iter(ds.data_vars))
    arr = ds[var_name].values.astype(np.float32)
    lat = ds["latitude"].values.astype(np.float32)
    lon = ds["longitude"].values.astype(np.float32)

    if lon.max() > 180:
        lon = ((lon + 180.0) % 360.0) - 180.0
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        arr = arr[::-1, :]

    lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
    lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
    if not lat_mask.any() or not lon_mask.any():
        raise ValueError(f"The MRMS grid does not overlap the specified bounds {lat_bounds} / {lon_bounds}.")

    lat_sel = lat[lat_mask]
    lon_sel = lon[lon_mask]
    arr_sel = arr[np.ix_(lat_mask.nonzero()[0], lon_mask.nonzero()[0])]
    if arr_sel.size == 0:
        raise ValueError("No valid data remained after clipping the MRMS grid to the study area.")

    res_y = abs(float(np.median(np.diff(lat_sel)))) if len(lat_sel) > 1 else 0.01
    res_x = abs(float(np.median(np.diff(lon_sel)))) if len(lon_sel) > 1 else 0.01
    transform = from_origin(lon_sel.min(), lat_sel.max(), res_x, res_y)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=arr_sel.shape[0],
        width=arr_sel.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(arr_sel.astype(np.float32), 1)

    return output_path


def run_real_event_pipeline(download_only: bool = False) -> Dict[str, str]:
    manifest = build_real_event_manifest()
    result: Dict[str, str] = {}

    for source in manifest:
        try:
            local = download_url(source.url, source.destination)
            result[source.name] = str(local)
            print(f"Saved {source.name} to {local}")
        except RuntimeError as exc:
            result[source.name] = f"SKIPPED: {exc}"
            print(f"Skipping {source.name}: {exc}")

    if download_only:
        return result

    print(
        "Real-event metadata has been downloaded. Use the NOAA MRMS and USGS/NLCD product indices to locate the precise clipped rasters, "
        "then rerun the susceptibility workflow on the final event grid."
    )
    return result


def build_study_area_polygon() -> gpd.GeoDataFrame:
    bounds = STUDY_AREA["bounds"]
    polygon = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries.from_wkt([
            f"POLYGON (({bounds['xmin']} {bounds['ymin']}, {bounds['xmax']} {bounds['ymin']}, {bounds['xmax']} {bounds['ymax']}, {bounds['xmin']} {bounds['ymax']}, {bounds['xmin']} {bounds['ymin']}))"
        ]),
        crs=STUDY_AREA["crs"],
    )
    polygon["name"] = STUDY_AREA["name"]
    polygon.to_file(RAW_DIR / "study_area.geojson", driver="GeoJSON")
    return polygon


def _extract_311_validation_points() -> pd.DataFrame:
    json_path = RAW_DIR / "impacts" / "nyc_311_ida.json"
    columns = ["unique_key", "created_date", "lon", "lat", "impact_flag", "complaint_type", "descriptor"]
    if not json_path.exists():
        return pd.DataFrame(columns=columns)

    records = json.loads(json_path.read_text())
    clean = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        descriptor = str(rec.get("descriptor") or "")
        descriptor_lower = descriptor.lower()
        if not any(term in descriptor_lower for term in ("street flooding", "catch basin clogged/flooding", "rain garden flooding")):
            continue

        location = rec.get("location") or {}
        coordinates = location.get("coordinates") if isinstance(location, dict) else None
        lon = rec.get("longitude") or (coordinates[0] if coordinates else None)
        lat = rec.get("latitude") or (coordinates[1] if coordinates else None)
        if lon is None or lat is None:
            continue

        lon, lat = float(lon), float(lat)
        bounds = STUDY_AREA["bounds"]
        if not (bounds["xmin"] <= lon <= bounds["xmax"] and bounds["ymin"] <= lat <= bounds["ymax"]):
            continue

        clean.append({
            "unique_key": rec.get("unique_key"),
            "created_date": rec.get("created_date"),
            "lon": lon,
            "lat": lat,
            "impact_flag": 1,
            "complaint_type": rec.get("complaint_type"),
            "descriptor": descriptor,
        })

    return pd.DataFrame(clean, columns=columns)


def process_real_event_data() -> dict:
    """Prepare the case-study workflow for the actual Ida event if local public rasters exist.

    This stages a real MRMS Ida precipitation file when available, but still falls back to the
    validated synthetic demo dataset when exact clipped raster products are not yet in place.
    """
    study_area = RAW_DIR / "study_area.geojson"
    if not study_area.exists():
        build_study_area_polygon()

    rainfall_candidates = sorted(RAW_DIR.glob("**/*.tif"))
    has_real_rain = any("rain" in p.name.lower() or "precip" in p.name.lower() for p in rainfall_candidates)
    has_real_dem = TERRAIN_DATASETS["dem"].exists()
    has_real_imperv = LANDCOVER_DATASETS["impervious"].exists()

    real_mrms = RAW_DIR / "rainfall" / "mrms_event"
    real_mrms.mkdir(parents=True, exist_ok=True)
    real_mrms_download = None
    if not any(real_mrms.iterdir()):
        try:
            real_mrms_download = download_mrms_event_file()
            print(f"Downloaded real MRMS Ida event file: {real_mrms_download}")
        except Exception as exc:
            print(f"MRMS download for Ida event was unavailable: {exc}")
    else:
        real_mrms_download = next(real_mrms.iterdir())
        print(f"Using existing real MRMS Ida event file: {real_mrms_download}")

    clipped_rainfall = EVENT_OUTPUTS["rainfall_total"]
    if real_mrms_download is not None and real_mrms_download.exists():
        try:
            convert_mrms_grib_to_raster(
                real_mrms_download,
                clipped_rainfall,
                lat_bounds=(STUDY_AREA["bounds"]["ymin"], STUDY_AREA["bounds"]["ymax"]),
                lon_bounds=(STUDY_AREA["bounds"]["xmin"], STUDY_AREA["bounds"]["xmax"]),
            )
            print(f"Converted clipped MRMS raster to {clipped_rainfall}")
            has_real_rain = True
        except Exception as exc:
            print(f"MRMS conversion for the study area failed: {exc}")
            has_real_rain = False

    if not (has_real_rain and has_real_dem and has_real_imperv):
        print("Real-event rasters are incomplete or missing; using the validated synthetic NYC demo case for the susceptibility workflow.")
        write_demo_rainfall_and_terrain()
    else:
        print("Using real-event-style local rasters for the study area.")
        if not PROCESSED_DIR.exists():
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    validation_df = _extract_311_validation_points()
    if not validation_df.empty:
        validation_df.to_csv(IMPACT_DATASETS["nyc_311"], index=False)
    else:
        validation_df.to_csv(IMPACT_DATASETS["nyc_311"], index=False)

    build_terrain_layers()
    build_antecedent_wetness()
    build_validation_points()
    return {
        "study_area": str(RAW_DIR / "study_area.geojson"),
        "validation_points": str(IMPACT_DATASETS["nyc_311"]),
        "used_demo_fallback": not (has_real_rain and has_real_dem and has_real_imperv),
        "real_mrms_file": str(real_mrms_download) if real_mrms_download is not None else None,
    }

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from .config import EVENT_OUTPUTS, IMPACT_DATASETS, LANDCOVER_DATASETS, LAND_MASK_DATASETS, PROCESSED_DIR, RAW_DIR, STUDY_AREA, TERRAIN_DATASETS


def resolve_rainfall_source() -> Path | None:
    for candidate in [EVENT_OUTPUTS["rainfall_total"], RAW_DIR / "rainfall" / "demo_rainfall_total.tif"]:
        if candidate.exists():
            return candidate
    return None


def align_to_reference(source_path: str | Path, reference_path: str | Path, nodata_value: float = -9999.0):
    with rasterio.open(reference_path) as ref:
        ref_profile = ref.profile.copy()
        ref_arr = ref.read(1)
        ref_transform = ref.transform
        ref_crs = ref.crs

    with rasterio.open(source_path) as src:
        dest = np.full((ref_profile["height"], ref_profile["width"]), nodata_value, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_nodata=nodata_value,
            resampling=Resampling.bilinear,
        )
    return dest


def load_raster(path: str | Path):
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    return data, profile, transform, crs


def clip_to_study_area(raster_path: str | Path, study_area_path: str | Path):
    study = gpd.read_file(study_area_path)
    with rasterio.open(raster_path) as src:
        data, transform = rasterio.mask.mask(src, study.geometry, crop=True)
        clipped = data[0]
    return clipped, transform


def build_terrain_layers() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rainfall_source = resolve_rainfall_source()
    dem, profile, transform, crs = load_raster(TERRAIN_DATASETS["dem"])
    if rainfall_source is not None:
        with rasterio.open(rainfall_source) as ref:
            ref_profile = ref.profile.copy()
        if (dem.shape != (ref_profile["height"], ref_profile["width"])):
            dem = align_to_reference(TERRAIN_DATASETS["dem"], rainfall_source)
            profile = ref_profile.copy()
    slope = np.gradient(dem, axis=0)
    slope = np.abs(slope) + 0.01
    profile["dtype"] = "float32"
    profile["nodata"] = -9999.0
    for array, path in [
        (dem.astype(np.float32), PROCESSED_DIR / "dem_processed.tif"),
        (slope.astype(np.float32), PROCESSED_DIR / "slope.tif"),
    ]:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array.astype(np.float32), 1)


def build_antecedent_wetness() -> None:
    rainfall_source = resolve_rainfall_source()
    if rainfall_source is None:
        raise FileNotFoundError("No rainfall raster was available for the susceptibility analysis.")
    rainfall_total, _, _, _ = load_raster(rainfall_source)
    impervious = align_to_reference(LANDCOVER_DATASETS["impervious"], rainfall_source)
    impervious = np.clip(impervious, 0, 1)
    wetness = np.clip((rainfall_total / np.nanmax(rainfall_total)) * 0.7 + impervious * 0.3, 0, 1)
    profile = rasterio.open(rainfall_source).profile.copy()
    with rasterio.open(EVENT_OUTPUTS["antecedent_wetness"], "w", **profile) as dst:
        dst.write(wetness.astype(np.float32), 1)


def acquire_high_resolution_dem() -> Path:
    """Fetch only the study-area cutout from current USGS 1/3-arc-second DEM tiles."""
    output_path = TERRAIN_DATASETS["dem_highres"]
    if output_path.exists():
        return output_path

    bounds = STUDY_AREA["bounds"]
    response = requests.get(
        "https://tnmaccess.nationalmap.gov/api/v1/products",
        params={
            "bbox": f"{bounds['xmin']},{bounds['ymin']},{bounds['xmax']},{bounds['ymax']}",
            "datasets": "National Elevation Dataset (NED) 1/3 arc-second",
            "prodFormats": "GeoTIFF",
            "max": 100,
        },
        timeout=60,
    )
    response.raise_for_status()
    items = response.json().get("items", [])
    selected = {}
    for item in items:
        match = re.search(r"(n\d{2}w\d{3})\s+(\d{8})$", item.get("title", ""))
        if match is None:
            continue
        tile_name, tile_date = match.groups()
        if tile_name not in selected or tile_date > selected[tile_name][0]:
            selected[tile_name] = (tile_date, item["downloadURL"])
    if not selected:
        raise RuntimeError("USGS returned no 1/3-arc-second DEM tiles for the study area.")

    sources = []
    try:
        for tile_date, url in selected.values():
            sources.append(rasterio.open(f"/vsicurl/{url}"))
        data, transform = merge(
            sources,
            bounds=(bounds["xmin"], bounds["ymin"], bounds["xmax"], bounds["ymax"]),
            res=(1 / 10800, 1 / 10800),
            dtype="float32",
        )
        profile = sources[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=data.shape[1],
            width=data.shape[2],
            count=1,
            dtype="float32",
            transform=transform,
            compress="deflate",
            tiled=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data[0], 1)
            dst.update_tags(
                source="USGS 3DEP 1/3 arc-second DEM",
                resolution="approximately 10 m",
                source_tiles="; ".join(f"{name} {date}" for name, (date, _) in selected.items()),
                api="https://tnmaccess.nationalmap.gov/api/v1/products",
            )
    finally:
        for source in sources:
            source.close()
    return output_path


def build_land_mask(reference_path: str | Path | None = None, output_path: str | Path | None = None) -> np.ndarray:
    reference_path = Path(reference_path) if reference_path is not None else resolve_rainfall_source()
    if reference_path is None:
        raise FileNotFoundError("A reference raster is required to build the land mask.")
    output_path = Path(output_path) if output_path is not None else EVENT_OUTPUTS["land_mask"]

    archive = LAND_MASK_DATASETS["natural_earth_land"]
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(
            "https://naturalearth.s3.amazonaws.com/10m_physical/ne_10m_land.zip",
            timeout=120,
        )
        response.raise_for_status()
        archive.write_bytes(response.content)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Natural Earth land archive is invalid: {archive}")

    land = gpd.read_file(f"zip://{archive}")
    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        transform = reference.transform
        shape = (reference.height, reference.width)
        crs = reference.crs
    if land.crs != crs:
        land = land.to_crs(crs)

    mask = rasterize(
        ((geometry, 1) for geometry in land.geometry if geometry is not None and not geometry.is_empty),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    )
    profile.update(dtype="uint8", count=1, nodata=0, compress="deflate")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask, 1)
        dst.update_tags(source="Natural Earth 1:10m land polygons", url="https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-land/")
    return mask


def build_high_resolution_terrain_screening() -> Path:
    """Create a land-only terrain screening index without rainfall or drainage inputs."""
    dem_path = acquire_high_resolution_dem()
    dem, profile, transform, crs = load_raster(dem_path)
    land_mask = build_land_mask(dem_path, EVENT_OUTPUTS["land_mask_highres"])
    with rasterio.open(dem_path) as source:
        valid = source.read_masks(1) > 0
        nodata = source.nodata
        bounds = source.bounds
        resolution = source.res
    valid &= land_mask == 1
    if nodata is not None:
        valid &= dem != nodata
    valid &= np.isfinite(dem)
    if not valid.any():
        raise ValueError("The high-resolution DEM contains no valid land cells in the study area.")

    elevation = dem[valid]
    low, high = np.percentile(elevation, [5, 95])
    lowland = np.clip((high - dem) / max(float(high - low), 1e-6), 0, 1)
    latitude = (bounds.bottom + bounds.top) / 2
    pixel_y_m = abs(resolution[1]) * 111320.0
    pixel_x_m = abs(resolution[0]) * 111320.0 * np.cos(np.deg2rad(latitude))
    gradients = np.gradient(np.where(valid, dem, np.nanmedian(elevation)), pixel_y_m, pixel_x_m)
    slope = np.hypot(gradients[0], gradients[1])
    flatness = 1.0 / (1.0 + slope)
    screening = np.clip(0.65 * lowland + 0.35 * flatness, 0, 1).astype(np.float32)
    screening[~valid] = -9999.0
    profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate", tiled=True)
    output_path = EVENT_OUTPUTS["terrain_screening_highres"]
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(screening, 1)
        dst.update_tags(
            interpretation="Terrain-only screening index; excludes rainfall, imperviousness, soils, and drainage.",
            resolution="approximately 10 m",
        )
    return output_path


def build_validation_points() -> None:
    if IMPACT_DATASETS["nyc_311"].exists():
        impacts = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                [float(row["lon"]) for _, row in __import__("pandas").read_csv(IMPACT_DATASETS["nyc_311"]).iterrows()],
                [float(row["lat"]) for _, row in __import__("pandas").read_csv(IMPACT_DATASETS["nyc_311"]).iterrows()],
            ),
            crs="EPSG:4326",
        )
        impacts["impact_flag"] = __import__("pandas").read_csv(IMPACT_DATASETS["nyc_311"]) ["impact_flag"].astype(int)
        impacts.to_file(EVENT_OUTPUTS["validation_points"], driver="GeoJSON")


__all__ = [
    "load_raster",
    "clip_to_study_area",
    "build_terrain_layers",
    "build_antecedent_wetness",
    "build_land_mask",
    "acquire_high_resolution_dem",
    "build_high_resolution_terrain_screening",
    "build_validation_points",
]

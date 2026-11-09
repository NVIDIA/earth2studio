"""MRMS MultiSensor QPE (1 h) fetch + 6 h aggregation for Ida Earth2 QA.

Uses NOAA public S3 QPE products — not BREF/reflectivity. Earth2Studio's built-in
``data.MRMS`` currently exposes reflectivity only, so this module downloads QPE
directly from ``s3://noaa-mrms-pds``.
"""

from __future__ import annotations

import gzip
import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import rasterio
import requests
import xarray as xr
from matplotlib import colormaps
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

from .config import FIGURES_DIR, PROCESSED_DIR, RAW_DIR, STUDY_AREA

QPE_PRODUCT = "MultiSensor_QPE_01H_Pass1_00.00"
QPE_CACHE = RAW_DIR / "rainfall" / "mrms_qpe_01h"
BOUNDS = STUDY_AREA["bounds"]
ASSETS_DIR = FIGURES_DIR / "ida_v0_assets"
PROCESSED_QPE = PROCESSED_DIR / "mrms_qpe"
DISPLAY_SHAPE = (180, 400)


def _download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def fetch_hourly_qpe(timestamp: datetime) -> Path:
    """Download one MultiSensor 1 h QPE GRIB2.gz ending at ``timestamp`` (UTC)."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    filename = f"MRMS_{QPE_PRODUCT}_{timestamp:%Y%m%d-%H%M%S}.grib2.gz"
    url = f"https://noaa-mrms-pds.s3.amazonaws.com/CONUS/{QPE_PRODUCT}/{timestamp:%Y%m%d}/{filename}"
    return _download(url, QPE_CACHE / filename)


def decode_study_crop(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode QPE GRIB and crop to the study bbox. Returns mm values, lat, lon (−180…180)."""
    with gzip.open(path, "rb") as compressed, tempfile.NamedTemporaryFile(suffix=".grib2") as raw:
        shutil.copyfileobj(compressed, raw)
        raw.flush()
        dataset = xr.open_dataset(raw.name, engine="cfgrib", backend_kwargs={"indexpath": ""})
        try:
            variable = dataset[next(iter(dataset.data_vars))]
            values = variable.values.astype(np.float32)
            # Mask GRIB missing sentinel
            missing = float(variable.attrs.get("GRIB_missingValue", 3.4028235e38))
            values = np.where((values >= missing * 0.1) | ~np.isfinite(values), np.nan, values)
            latitude = dataset["latitude"].values.astype(np.float64)
            longitude = dataset["longitude"].values.astype(np.float64)
        finally:
            dataset.close()

    longitude = np.where(longitude > 180.0, longitude - 360.0, longitude)
    # Expand crop slightly so reproject has edge context
    lat_idx = np.flatnonzero((latitude >= BOUNDS["ymin"] - 0.05) & (latitude <= BOUNDS["ymax"] + 0.05))
    lon_idx = np.flatnonzero((longitude >= BOUNDS["xmin"] - 0.05) & (longitude <= BOUNDS["xmax"] + 0.05))
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError(f"No MRMS QPE cells intersect study area in {path.name}")
    crop = values[np.ix_(lat_idx, lon_idx)]
    return crop, latitude[lat_idx], longitude[lon_idx]


def _upsample(lat: np.ndarray, lon: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, object]:
    if lat[0] < lat[-1]:
        field = field[::-1, :]
        lat = lat[::-1]
    src_transform = from_bounds(
        float(lon.min()),
        float(lat.min()),
        float(lon.max()),
        float(lat.max()),
        field.shape[1],
        field.shape[0],
    )
    dst = np.full(DISPLAY_SHAPE, np.nan, dtype=np.float32)
    dst_transform = from_bounds(
        BOUNDS["xmin"], BOUNDS["ymin"], BOUNDS["xmax"], BOUNDS["ymax"], DISPLAY_SHAPE[1], DISPLAY_SHAPE[0]
    )
    reproject(
        source=field.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs="EPSG:4326",
        dst_transform=dst_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst, dst_transform


def _write_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)


def _rgba(values: np.ndarray, cmap_name: str, vmax: float, opacity: int = 200) -> np.ndarray:
    finite = np.isfinite(values)
    data = np.where(finite, values, 0.0).astype(np.float32)
    vmax = max(float(vmax), 1e-6)
    normalized = np.clip(data / vmax, 0.0, 1.0)
    rgba = colormaps[cmap_name](normalized, bytes=True)
    rgba[..., 3] = np.where(finite & (data > 0.05), np.minimum(opacity, 50 + (normalized * 190).astype(np.uint8)), 0)
    return rgba


def _write_tif(path: Path, data: np.ndarray, transform) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
    }
    filled = np.where(np.isfinite(data), data, -9999.0).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(filled, 1)


def interval_end_hours(init: datetime, lead_hours: list[int]) -> list[tuple[int, datetime, list[datetime]]]:
    """For each positive lead, return (lead, valid_time, list of hourly QPE end timestamps)."""
    if init.tzinfo is None:
        init = init.replace(tzinfo=timezone.utc)
    intervals = []
    previous = 0
    for lead in lead_hours:
        valid = init + timedelta(hours=lead)
        hours = [init + timedelta(hours=h) for h in range(previous + 1, lead + 1)]
        intervals.append((lead, valid, hours))
        previous = lead
    return intervals


def build_mrms_qpe_qa(
    earth2_frames: list[dict],
    init_time: str = "2021-09-01T12:00:00",
    shared_vmax: float | None = None,
) -> dict:
    """Fetch/aggregate MRMS QPE to match Earth2 6 h leads and write map assets."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_QPE.mkdir(parents=True, exist_ok=True)
    init = datetime.fromisoformat(init_time.replace("Z", "")).replace(tzinfo=timezone.utc)
    lead_hours = [int(frame["lead_hours"]) for frame in earth2_frames]
    intervals = interval_end_hours(init, lead_hours)

    hourly_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    frames_out = []
    storm_total = None
    transform = None

    for lead, valid, hours in intervals:
        print(f"  MRMS QPE lead +{lead} h ← {len(hours)} hourly files")
        stacked = []
        lat = lon = None
        for hour in hours:
            key = hour.strftime("%Y%m%d-%H%M%S")
            if key not in hourly_cache:
                path = fetch_hourly_qpe(hour)
                hourly_cache[key] = decode_study_crop(path)
            crop, lat, lon = hourly_cache[key]
            stacked.append(np.nan_to_num(crop, nan=0.0))
        accum = np.sum(np.stack(stacked, axis=0), axis=0).astype(np.float32)
        display, transform = _upsample(lat, lon, accum)
        storm_total = display if storm_total is None else storm_total + np.nan_to_num(display, nan=0.0)

        stem = f"mrms_qpe_lead_{lead:02d}h"
        tif = PROCESSED_QPE / f"{stem}.tif"
        png = ASSETS_DIR / f"{stem}.png"
        _write_tif(tif, display, transform)
        frames_out.append(
            {
                "lead_hours": lead,
                "valid_time": valid.strftime("%Y-%m-%dT%H:%M"),
                "png": png.name,
                "tif": str(tif),
                "mean_mm": float(np.nanmean(display)),
                "max_mm": float(np.nanmax(display)),
                "n_hourly_files": len(hours),
                "hourly_ends_utc": [h.strftime("%Y-%m-%dT%H:%M") for h in hours],
            }
        )

    # Shared color scale with Earth2 for fair visual compare
    e2_max = max(float(f["max_mm"]) for f in earth2_frames)
    mrms_max = max(float(f["max_mm"]) for f in frames_out)
    vmax = float(shared_vmax) if shared_vmax is not None else max(e2_max, mrms_max, 5.0)

    for frame in frames_out:
        with rasterio.open(frame["tif"]) as src:
            display = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        _write_png(ASSETS_DIR / frame["png"], _rgba(display, "YlGnBu", vmax=vmax))

    total_tif = PROCESSED_QPE / "mrms_qpe_storm_total.tif"
    total_png = ASSETS_DIR / "mrms_qpe_storm_total.png"
    _write_tif(total_tif, storm_total.astype(np.float32), transform)
    _write_png(total_png, _rgba(storm_total.astype(np.float32), "YlGnBu", vmax=max(float(np.nanmax(storm_total)), vmax)))

    # Difference maps (Earth2 − MRMS) on shared display grid
    diff_frames = []
    for e2, mrms in zip(earth2_frames, frames_out):
        with rasterio.open(e2["tif"]) as src_e2, rasterio.open(mrms["tif"]) as src_m:
            e2_data = src_e2.read(1, masked=True).filled(np.nan).astype(np.float32)
            m_data = src_m.read(1, masked=True).filled(np.nan).astype(np.float32)
            e2_transform = src_e2.transform
            m_transform = src_m.transform
        if e2_data.shape != m_data.shape:
            aligned = np.full_like(e2_data, np.nan)
            reproject(
                source=m_data,
                destination=aligned,
                src_transform=m_transform,
                src_crs="EPSG:4326",
                dst_transform=e2_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
            m_data = aligned
        diff = e2_data - m_data
        absmax = float(np.nanmax(np.abs(diff))) if np.isfinite(diff).any() else 1.0
        absmax = max(absmax, 1.0)
        # Diverging: blue=Earth2 dry vs MRMS, red=Earth2 wetter
        norm = np.clip((diff + absmax) / (2 * absmax), 0, 1)
        rgba = colormaps["RdBu_r"](norm, bytes=True)
        finite = np.isfinite(diff)
        rgba[..., 3] = np.where(finite, 180, 0).astype(np.uint8)
        png_name = f"diff_e2_minus_mrms_lead_{int(e2['lead_hours']):02d}h.png"
        _write_png(ASSETS_DIR / png_name, rgba)
        diff_frames.append(
            {
                "lead_hours": int(e2["lead_hours"]),
                "png": png_name,
                "mean_diff_mm": float(np.nanmean(diff)),
                "rmse_mm": float(np.sqrt(np.nanmean(diff**2))),
                "max_abs_diff_mm": absmax,
            }
        )

    meta = {
        "product": QPE_PRODUCT,
        "product_note": (
            "NOAA MRMS MultiSensor QPE 01H Pass1 (public S3). "
            "Hourly depths summed into 6 h intervals ending at each Earth2 valid time. "
            "Units treated as mm. Not BREF/reflectivity."
        ),
        "source": "https://noaa-mrms-pds.s3.amazonaws.com/",
        "init_time": init.strftime("%Y-%m-%dT%H:%M:%S"),
        "shared_display_vmax_mm": vmax,
        "frames": frames_out,
        "storm_total_png": total_png.name,
        "storm_total_tif": str(total_tif),
        "storm_total_max_mm": float(np.nanmax(storm_total)),
        "storm_total_mean_mm": float(np.nanmean(storm_total)),
        "diff_frames": diff_frames,
        "bounds": [[BOUNDS["ymin"], BOUNDS["xmin"]], [BOUNDS["ymax"], BOUNDS["xmax"]]],
    }
    (ASSETS_DIR / "mrms_qpe_qa.json").write_text(json.dumps(meta, indent=2))
    return meta

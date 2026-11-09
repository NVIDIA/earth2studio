#!/usr/bin/env python3
"""Build the Ida v0 Earth2 → map prototype (real precip, not synthetic scenarios)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from matplotlib import colormaps
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
from shapely.geometry import box, mapping

from nyc_flash_flood_case_study.config import (
    EVENT_OUTPUTS,
    FIGURES_DIR,
    IMPACT_DATASETS,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    STUDY_AREA,
    TERRAIN_DATASETS,
)
from nyc_flash_flood_case_study.forecast import normalize_earth2_forecast

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover
    raise SystemExit("xarray is required to build the Ida v0 map") from exc


BOUNDS = STUDY_AREA["bounds"]
ASSETS_DIR = FIGURES_DIR / "ida_v0_assets"
FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"
DEFAULT_GLOBAL = FORECAST_DIR / "earth2_precip_ida_20210901T12.nc"
DEFAULT_CLIPPED = FORECAST_DIR / "nyc_precip_ida_20210901T12.nc"
DISPLAY_SHAPE = (180, 400)  # lat, lon — bilinear display grid over study bbox
PRECIP_HOTSPOT_PERCENTILE = 75.0
TERRAIN_HOTSPOT_PERCENTILE = 75.0
MIN_PRECIP_HOTSPOT_MM = 20.0
MIN_TERRAIN_HOTSPOT = 0.55


def _ensure_dirs() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "earth2_precip").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "infrastructure").mkdir(parents=True, exist_ok=True)


def _rgba_from_field(values: np.ndarray, cmap_name: str, vmax: float | None = None, opacity: int = 200) -> np.ndarray:
    finite = np.isfinite(values)
    data = np.where(finite, values, 0.0).astype(np.float32)
    if vmax is None:
        vmax = float(np.nanpercentile(data[finite], 98)) if finite.any() else 1.0
    vmax = max(vmax, 1e-6)
    normalized = np.clip(data / vmax, 0.0, 1.0)
    rgba = colormaps[cmap_name](normalized, bytes=True)
    rgba[..., 3] = np.where(finite & (data > 0), np.minimum(opacity, 60 + (normalized * 180).astype(np.uint8)), 0)
    return rgba


def _write_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(path, optimize=True)


def _write_geotiff(path: Path, data: np.ndarray, transform: Affine, nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    filled = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(filled, 1)


def _upsample_to_display(lat: np.ndarray, lon: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, Affine]:
    """Bilinear-resample a coarse Earth2 cell field onto a denser display grid (visual only)."""
    src_transform = from_bounds(
        float(lon.min()) - 0.125,
        float(lat.min()) - 0.125,
        float(lon.max()) + 0.125,
        float(lat.max()) + 0.125,
        field.shape[1],
        field.shape[0],
    )
    # Ensure north-up: Earth2 lat is often descending
    if lat[0] < lat[-1]:
        field = field[::-1, :]
        lat = lat[::-1]
        src_transform = from_bounds(
            float(lon.min()) - 0.125,
            float(lat.min()) - 0.125,
            float(lon.max()) + 0.125,
            float(lat.max()) + 0.125,
            field.shape[1],
            field.shape[0],
        )

    dst = np.full(DISPLAY_SHAPE, np.nan, dtype=np.float32)
    dst_transform = from_bounds(BOUNDS["xmin"], BOUNDS["ymin"], BOUNDS["xmax"], BOUNDS["ymax"], DISPLAY_SHAPE[1], DISPLAY_SHAPE[0])
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


def export_precip_layers(clipped_nc: Path) -> dict:
    """Export lead-wise and storm-total precip GeoTIFF/PNG assets from clipped NetCDF."""
    with xr.open_dataset(clipped_nc) as ds:
        precip = ds["precipitation_mm"]
        if "ensemble" in precip.dims:
            precip = precip.isel(ensemble=0)
        lead_hours = [float(v) for v in precip["lead_time_hours"].values]
        valid_times = [str(np.datetime64(v, "m")) for v in precip["valid_time"].values]
        lat = precip["lat"].values.astype(np.float64)
        lon = precip["lon"].values.astype(np.float64)
        # Sort lat descending for image orientation consistency
        if lat[0] < lat[-1]:
            precip = precip.reindex(lat=lat[::-1])
            lat = precip["lat"].values.astype(np.float64)

        frames = []
        total = None
        vmax = float(np.nanmax(precip.values))
        vmax = max(vmax, 1.0)

        for index, (lead, valid) in enumerate(zip(lead_hours, valid_times)):
            field = precip.isel(lead_time_hours=index).values.astype(np.float32)
            display, transform = _upsample_to_display(lat, lon, field)
            total = display if total is None else np.nansum(np.stack([total, display], axis=0), axis=0)

            stem = f"precip_lead_{int(lead):02d}h"
            tif_path = PROCESSED_DIR / "earth2_precip" / f"{stem}.tif"
            png_path = ASSETS_DIR / f"{stem}.png"
            _write_geotiff(tif_path, display, transform)
            _write_png(png_path, _rgba_from_field(display, "Blues", vmax=max(vmax, 5.0)))

            # Native-resolution PNG for documentation
            fig, ax = plt.subplots(figsize=(6, 4))
            extent = (BOUNDS["xmin"], BOUNDS["xmax"], BOUNDS["ymin"], BOUNDS["ymax"])
            im = ax.imshow(display, origin="upper", extent=extent, cmap="Blues", vmin=0, vmax=max(vmax, 5.0))
            ax.set_title(f"Earth2 precip | lead +{int(lead)} h | valid {valid} UTC")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label="6 h precipitation (mm)")
            fig.tight_layout()
            fig.savefig(ASSETS_DIR / f"{stem}_panel.png", dpi=140)
            plt.close(fig)

            frames.append(
                {
                    "index": index,
                    "lead_hours": int(lead),
                    "valid_time": valid,
                    "png": png_path.name,
                    "tif": str(tif_path),
                    "mean_mm": float(np.nanmean(display)),
                    "max_mm": float(np.nanmax(display)),
                }
            )

        total_tif = PROCESSED_DIR / "earth2_precip" / "precip_storm_total.tif"
        total_png = ASSETS_DIR / "precip_storm_total.png"
        _, transform = _upsample_to_display(lat, lon, precip.isel(lead_time_hours=0).values.astype(np.float32))
        _write_geotiff(total_tif, total.astype(np.float32), transform)
        _write_png(total_png, _rgba_from_field(total.astype(np.float32), "BuPu", vmax=max(float(np.nanmax(total)), 10.0)))

        fig, ax = plt.subplots(figsize=(6, 4))
        extent = (BOUNDS["xmin"], BOUNDS["xmax"], BOUNDS["ymin"], BOUNDS["ymax"])
        im = ax.imshow(total, origin="upper", extent=extent, cmap="BuPu", vmin=0, vmax=max(float(np.nanmax(total)), 10.0))
        ax.set_title("Earth2 storm-total precipitation (sum of 6 h intervals)")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.colorbar(im, ax=ax, label="Precipitation (mm)")
        fig.tight_layout()
        fig.savefig(ASSETS_DIR / "precip_storm_total_panel.png", dpi=140)
        plt.close(fig)

        init_time = str(ds.attrs.get("forecast_initialization_time", "unknown"))
        model = str(ds.attrs.get("forecast_model", "FourCastNet + PrecipitationAFNO"))

    return {
        "init_time": init_time,
        "forecast_model": model,
        "frames": frames,
        "storm_total_png": total_png.name,
        "storm_total_tif": str(total_tif),
        "storm_total_max_mm": float(np.nanmax(total)),
        "storm_total_mean_mm": float(np.nanmean(total)),
        "display_note": (
            "Precip layers are bilinearly resampled from the native ~0.25° Earth2 grid "
            "onto a denser display grid for visualization only; this is not physical downscaling."
        ),
        "interval_hours": 6.0,
        "bounds": [[BOUNDS["ymin"], BOUNDS["xmin"]], [BOUNDS["ymax"], BOUNDS["xmax"]]],
    }


def _download_json(url: str, destination: Path, params: dict | None = None) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, params=params, timeout=180)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _overpass(query: str, cache_path: Path) -> dict:
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return json.loads(cache_path.read_text())
    headers = {
        "Accept": "application/json",
        "User-Agent": "earth2studio-ida-v0/0.1 (research prototype)",
    }
    cleaned = "\n".join(line.strip() for line in query.strip().splitlines() if line.strip())
    last_error: Exception | None = None
    for endpoint in (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ):
        try:
            response = requests.post(
                endpoint,
                data={"data": cleaned},
                headers=headers,
                timeout=180,
            )
            response.raise_for_status()
            payload = response.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(payload))
            return payload
        except Exception as exc:  # noqa: BLE001 - try next mirror
            last_error = exc
            continue
    raise RuntimeError(f"Overpass request failed for {cache_path.name}: {last_error}")


def fetch_infrastructure() -> dict:
    """Download OSM roads + subway stations, plus NYC DEP catch basins for the study bbox."""
    infra_dir = RAW_DIR / "infrastructure"
    study = box(BOUNDS["xmin"], BOUNDS["ymin"], BOUNDS["xmax"], BOUNDS["ymax"])
    bbox = f"{BOUNDS['ymin']},{BOUNDS['xmin']},{BOUNDS['ymax']},{BOUNDS['xmax']}"

    # Subway / rail stations via OSM (NYC Open Data subway SODA IDs are unstable)
    stations_payload = _overpass(
        f"""
        [out:json][timeout:120];
        (
          node["railway"="station"]({bbox});
          node["station"="subway"]({bbox});
          node["railway"="halt"]({bbox});
        );
        out body;
        """,
        infra_dir / "osm_rail_stations_south_queens.json",
    )
    station_features = []
    for element in stations_payload.get("elements", []):
        if element.get("type") != "node" or "lat" not in element:
            continue
        tags = element.get("tags", {})
        station_features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": tags.get("name", tags.get("railway", "station")),
                    "railway": tags.get("railway", ""),
                    "station": tags.get("station", ""),
                },
                "geometry": {"type": "Point", "coordinates": [float(element["lon"]), float(element["lat"])]},
            }
        )
    stations = gpd.GeoDataFrame.from_features(station_features, crs=4326) if station_features else gpd.GeoDataFrame(geometry=[], crs=4326)
    if not stations.empty:
        stations = stations[stations.geometry.within(study)].copy()

    # NYC DEP catch basins (public SODA) — useful infrastructure exposure points
    basins_raw = infra_dir / "nycdep_catch_basins_south_queens.json"
    # Reuse case-study drainage cache if present
    alt_basins = RAW_DIR / "drainage" / "nycdep_catch_basins_south_queens.json"
    if not basins_raw.exists() and alt_basins.exists():
        basins_raw.write_text(alt_basins.read_text())
    if not basins_raw.exists():
        records: list[dict] = []
        try:
            offset = 0
            while True:
                response = requests.get(
                    "https://data.cityofnewyork.us/resource/2w2g-fk3i.json",
                    params={
                        "$select": "unitid,latitude,longitude,the_geom",
                        "$where": (
                            f"within_box(the_geom,{BOUNDS['ymin']},{BOUNDS['xmin']},"
                            f"{BOUNDS['ymax']},{BOUNDS['xmax']})"
                        ),
                        "$limit": 50000,
                        "$offset": offset,
                    },
                    timeout=180,
                )
                response.raise_for_status()
                batch = response.json()
                if not batch or not isinstance(batch, list):
                    break
                records.extend(batch)
                if len(batch) < 50000:
                    break
                offset += len(batch)
            basins_raw.write_text(json.dumps(records))
        except Exception as exc:  # noqa: BLE001
            print(f"  catch-basin download skipped: {exc}")
            basins_raw.write_text("[]")
    basins = gpd.GeoDataFrame(geometry=[], crs=4326)
    if basins_raw.exists() and basins_raw.stat().st_size > 2:
        try:
            payload = json.loads(basins_raw.read_text())
            if isinstance(payload, list) and payload:
                basins = gpd.GeoDataFrame.from_features(
                    [
                        {
                            "type": "Feature",
                            "properties": {"unitid": row.get("unitid", "")},
                            "geometry": row.get("the_geom"),
                        }
                        for row in payload
                        if isinstance(row, dict) and row.get("the_geom")
                    ],
                    crs=4326,
                )
            elif isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
                basins = gpd.GeoDataFrame.from_features(payload.get("features", []), crs=4326)
        except Exception as exc:  # noqa: BLE001
            print(f"  catch-basin parse skipped: {exc}")
            basins = gpd.GeoDataFrame(geometry=[], crs=4326)
    if not basins.empty:
        if basins.crs is None:
            basins = basins.set_crs(4326)
        else:
            basins = basins.to_crs(4326)
        if len(basins) > 1500:
            basins = basins.sample(n=1500, random_state=0)

    stations_out = ASSETS_DIR / "subway_stations.geojson"
    basins_out = ASSETS_DIR / "catch_basins.geojson"
    stations.to_file(stations_out, driver="GeoJSON")
    if not basins.empty:
        basins.to_file(basins_out, driver="GeoJSON")
    else:
        basins_out.write_text(json.dumps({"type": "FeatureCollection", "features": []}))
        basins = gpd.GeoDataFrame(geometry=[], crs=4326)

    roads_payload = _overpass(
        f"""
        [out:json][timeout:120];
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential)$"]({bbox});
        );
        out geom;
        """,
        infra_dir / "osm_highways_south_queens.json",
    )
    features = []
    for element in roads_payload.get("elements", []):
        if element.get("type") != "way" or "geometry" not in element:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]
        if len(coords) < 2:
            continue
        tags = element.get("tags", {})
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "st_name": tags.get("name", tags.get("highway", "road")),
                    "physicalid": str(element.get("id")),
                    "highway": tags.get("highway", ""),
                },
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )
    roads = gpd.GeoDataFrame.from_features(features, crs=4326) if features else gpd.GeoDataFrame(geometry=[], crs=4326)
    if not roads.empty:
        roads = roads[roads.geometry.intersects(study)].copy()
        if len(roads) > 2500:
            roads = roads.sample(n=2500, random_state=0)
    roads_out = ASSETS_DIR / "roads.geojson"
    roads.to_file(roads_out, driver="GeoJSON")

    return {
        "stations_geojson": stations_out.name,
        "basins_geojson": basins_out.name,
        "roads_geojson": roads_out.name,
        "n_stations": int(len(stations)),
        "n_basins": int(len(basins)),
        "n_roads": int(len(roads)),
    }


def build_disruption_heuristic(precip_meta: dict) -> dict:
    """Flag road segments intersecting high precip × high terrain-screening cells (heuristic only)."""
    total_tif = Path(precip_meta["storm_total_tif"])
    screening_path = EVENT_OUTPUTS["terrain_screening_highres"]
    roads = gpd.read_file(ASSETS_DIR / "roads.geojson")

    with rasterio.open(total_tif) as precip_src:
        precip = precip_src.read(1, masked=True).astype(np.float32)
        precip_transform = precip_src.transform

    if screening_path.exists():
        with rasterio.open(screening_path) as screen_src:
            screening = np.empty(precip.shape, dtype=np.float32)
            reproject(
                source=rasterio.band(screen_src, 1),
                destination=screening,
                src_transform=screen_src.transform,
                src_crs=screen_src.crs,
                dst_transform=precip_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
    else:
        screening = np.ones(precip.shape, dtype=np.float32) * 0.7

    precip_data = precip.filled(0.0)
    precip_cut = max(
        MIN_PRECIP_HOTSPOT_MM,
        float(np.nanpercentile(precip_data[np.isfinite(precip_data)], PRECIP_HOTSPOT_PERCENTILE)),
    )
    screening_finite = screening[np.isfinite(screening)]
    terrain_cut = max(
        MIN_TERRAIN_HOTSPOT,
        float(np.nanpercentile(screening_finite, TERRAIN_HOTSPOT_PERCENTILE)) if screening_finite.size else MIN_TERRAIN_HOTSPOT,
    )
    hotspot = (precip_data >= precip_cut) & (screening >= terrain_cut)

    def sample_hotspot(geom) -> bool:
        if geom is None or geom.is_empty:
            return False
        # Sample along the line
        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
        else:
            lines = [geom]
        for line in lines:
            length = line.length
            n = max(3, int(length / 0.002))
            for fraction in np.linspace(0, 1, n):
                point = line.interpolate(fraction, normalized=True)
                row, col = rasterio.transform.rowcol(precip_transform, point.x, point.y)
                if 0 <= row < hotspot.shape[0] and 0 <= col < hotspot.shape[1] and hotspot[row, col]:
                    return True
        return False

    roads = roads.copy()
    roads["potential_disruption"] = roads.geometry.apply(sample_hotspot)
    # Prefer higher-class roads for the disruption overlay so the map stays readable.
    major = {"motorway", "trunk", "primary", "secondary", "tertiary"}
    disrupted = roads[roads["potential_disruption"]].copy()
    if "highway" in disrupted.columns:
        major_disrupted = disrupted[disrupted["highway"].isin(major)]
        if not major_disrupted.empty:
            disrupted = major_disrupted
    all_out = ASSETS_DIR / "roads_with_disruption.geojson"
    disrupted_out = ASSETS_DIR / "roads_potential_disruption.geojson"
    roads.to_file(all_out, driver="GeoJSON")
    disrupted.to_file(disrupted_out, driver="GeoJSON")

    # Hotspot raster PNG
    hotspot_rgba = np.zeros((*hotspot.shape, 4), dtype=np.uint8)
    hotspot_rgba[hotspot] = (210, 77, 52, 150)
    _write_png(ASSETS_DIR / "precip_terrain_hotspot.png", hotspot_rgba)

    return {
        "roads_geojson": all_out.name,
        "disrupted_geojson": disrupted_out.name,
        "hotspot_png": "precip_terrain_hotspot.png",
        "n_disrupted": int(disrupted.shape[0]),
        "n_roads": int(roads.shape[0]),
        "rule": (
            "Road segment marked if any sample point falls in cells with storm-total precip "
            f"≥ {precip_cut:.1f} mm (max of {MIN_PRECIP_HOTSPOT_MM} mm and domain "
            f"P{PRECIP_HOTSPOT_PERCENTILE:.0f}) AND terrain screening ≥ {terrain_cut:.2f} "
            f"(max of {MIN_TERRAIN_HOTSPOT} and domain P{TERRAIN_HOTSPOT_PERCENTILE:.0f}). "
            "Heuristic exposure only — not a traffic or inundation forecast."
        ),
        "precip_threshold_mm": precip_cut,
        "terrain_threshold": terrain_cut,
    }


def export_311() -> dict:
    reports_path = IMPACT_DATASETS["nyc_311"]
    if not reports_path.exists():
        return {"n_reports": 0, "geojson": None}
    df = pd.read_csv(reports_path)
    df = df.dropna(subset=["lon", "lat"])
    df = df[
        (df["lon"] >= BOUNDS["xmin"])
        & (df["lon"] <= BOUNDS["xmax"])
        & (df["lat"] >= BOUNDS["ymin"])
        & (df["lat"] <= BOUNDS["ymax"])
    ]
    features = []
    for row in df.itertuples(index=False):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "descriptor": str(getattr(row, "descriptor", "Flood-related report")),
                    "created_date": str(getattr(row, "created_date", "")),
                    "complaint_type": str(getattr(row, "complaint_type", "")),
                },
                "geometry": {"type": "Point", "coordinates": [float(row.lon), float(row.lat)]},
            }
        )
    geojson = {"type": "FeatureCollection", "features": features}
    out = ASSETS_DIR / "observed_311.geojson"
    out.write_text(json.dumps(geojson))
    return {"n_reports": len(features), "geojson": out.name}


def export_terrain_assets() -> dict:
    result = {}
    highres = EVENT_OUTPUTS.get("terrain_screening_highres")
    if highres and highres.exists():
        with rasterio.open(highres) as src:
            data = src.read(1, masked=True).astype(np.float32)
        rgba = colormaps["YlOrRd"](np.clip(data.filled(0), 0, 1), bytes=True)
        rgba[..., 3] = np.where(np.ma.getmaskarray(data), 0, 170).astype(np.uint8)
        # Downsample for web if huge
        if rgba.shape[0] > 900 or rgba.shape[1] > 1600:
            image = Image.fromarray(rgba, mode="RGBA")
            image = image.resize((1600, int(1600 * rgba.shape[0] / rgba.shape[1])), Image.Resampling.BILINEAR)
            image.save(ASSETS_DIR / "terrain_screening.png", optimize=True)
        else:
            _write_png(ASSETS_DIR / "terrain_screening.png", rgba)
        result["terrain_png"] = "terrain_screening.png"
        result["terrain_bounds"] = [[BOUNDS["ymin"], BOUNDS["xmin"]], [BOUNDS["ymax"], BOUNDS["xmax"]]]
    return result


def write_html(manifest: dict) -> Path:
    """Write a Leaflet map with Earth2 leads and optional MRMS QPE compare view."""
    html_path = FIGURES_DIR / "ida_v0_event_map.html"
    n_frames = len(manifest["precip"]["frames"])
    has_mrms = bool(manifest.get("mrms_qpe"))
    compare_disabled = "" if has_mrms else "disabled"
    mrms_checked = "checked" if has_mrms else "disabled"
    mrms_note = (
        "MRMS MultiSensor QPE 01H summed to 6 h intervals matching Earth2 leads. Not BREF."
        if has_mrms
        else "MRMS QPE assets not built — rerun with --with-mrms-qa."
    )
    mrms_product = "QPE 01H Pass1" if has_mrms else "n/a"
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ida v0 | Earth2 NYC flood-impact prototype</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css">
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css">
  <style>
    :root {{
      --ink: #1c2430; --muted: #5b6775; --panel: #f4f7f8; --line: #d5dde3;
      --accent: #1f6f8b; --warn: #9a3412;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; height: 100%; font-family: "IBM Plex Sans", "Segoe UI", sans-serif; color: var(--ink); background: #e8eef1; }}
    body {{ display: grid; grid-template-columns: 330px 1fr; }}
    aside {{ background: var(--panel); border-right: 1px solid var(--line); padding: 18px 16px; overflow: auto; }}
    h1 {{ font-size: 1.25rem; margin: 0 0 4px; letter-spacing: -0.02em; }}
    .eyebrow {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.68rem; color: var(--muted); margin: 0 0 8px; }}
    .sub {{ color: var(--muted); font-size: 0.86rem; margin: 0 0 16px; line-height: 1.4; }}
    .badge {{ display: inline-block; background: #fff7ed; color: var(--warn); border: 1px solid #fdba74; border-radius: 4px; padding: 2px 8px; font-size: 0.72rem; font-weight: 600; }}
    section {{ margin: 18px 0; padding-top: 12px; border-top: 1px solid var(--line); }}
    section h2 {{ font-size: 0.92rem; margin: 0 0 8px; }}
    label.row {{ display: flex; align-items: center; justify-content: space-between; gap: 8px; margin: 6px 0; font-size: 0.86rem; }}
    input[type="range"] {{ width: 100%; }}
    .readout {{ font-variant-numeric: tabular-nums; font-weight: 600; color: var(--accent); }}
    .map-stage {{ position: relative; display: grid; height: 100%; min-height: 0; width: 100%; }}
    .map-stage.single {{ grid-template-columns: 1fr; }}
    .map-stage.compare {{ grid-template-columns: 1fr 1fr; }}
    .map-pane {{ position: relative; min-height: 0; height: 100%; width: 100%; border-left: 1px solid var(--line); overflow: hidden; }}
    .map-pane:first-child {{ border-left: 0; }}
    .map-pane[hidden] {{ display: none !important; }}
    .map-el {{ position: absolute; inset: 0; }}
    .pane-label {{ position: absolute; z-index: 1000; top: 10px; left: 52px; background: rgba(255,255,255,0.92); border: 1px solid var(--line); padding: 4px 8px; font-size: 0.75rem; font-weight: 600; }}
    .disclaimer {{ position: absolute; z-index: 1000; left: 12px; bottom: 12px; max-width: 460px; background: rgba(255,255,255,0.94); border: 1px solid var(--line); padding: 8px 10px; font-size: 0.75rem; line-height: 1.35; color: var(--muted); pointer-events: none; }}
    details.limits {{ font-size: 0.8rem; color: var(--muted); }}
    details.limits summary {{ cursor: pointer; color: var(--ink); font-weight: 600; }}
    .metric {{ display: flex; justify-content: space-between; font-size: 0.82rem; margin: 4px 0; gap: 8px; }}
    .metric b {{ font-variant-numeric: tabular-nums; text-align: right; }}
    .legend-swatch {{ display: inline-block; width: 12px; height: 12px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
    .mode-btn {{ width: 100%; margin-top: 6px; padding: 8px 10px; border: 1px solid var(--line); background: #fff; border-radius: 6px; cursor: pointer; font-weight: 600; }}
    .mode-btn.active {{ border-color: var(--accent); background: #e8f4f8; color: var(--accent); }}
    .mode-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
    .qa-note {{ font-size: 0.78rem; color: var(--muted); line-height: 1.35; margin-top: 6px; }}
  </style>
</head>
<body>
  <aside>
    <p class="eyebrow">Earth2Studio · NYC flood impact · v0</p>
    <h1>Ida remnants replay</h1>
    <p class="sub">South Queens / Jamaica Bay · Earth2 FCN + PrecipitationAFNO with terrain, infrastructure, 311, and optional MRMS QPE QA.</p>
    <span class="badge">NOT inundation depth / NOT emergency guidance</span>

    <section>
      <h2>Forecast lead</h2>
      <input id="leadSlider" type="range" min="0" max="{n_frames - 1}" value="1" step="1" aria-label="Forecast lead">
      <div class="metric"><span>Lead</span><span class="readout" id="leadReadout">—</span></div>
      <div class="metric"><span>Valid time (UTC)</span><span class="readout" id="validReadout">—</span></div>
      <div class="metric"><span>Earth2 6 h max</span><span class="readout" id="precipReadout">—</span></div>
      <div class="metric"><span>MRMS QPE 6 h max</span><span class="readout" id="mrmsReadout">—</span></div>
      <div class="metric"><span>E2 − MRMS mean</span><span class="readout" id="diffReadout">—</span></div>
    </section>

    <section>
      <h2>View mode</h2>
      <button class="mode-btn active" id="modeSingle" type="button">Single map</button>
      <button class="mode-btn" id="modeCompare" type="button" {compare_disabled}>Side-by-side Earth2 | MRMS QPE</button>
      <p class="qa-note">{mrms_note}</p>
    </section>

    <section>
      <h2>Map layers</h2>
      <label class="row"><span><span class="legend-swatch" style="background:#3b82f6"></span>Earth2 6 h precip</span><input id="precipToggle" type="checkbox" checked></label>
      <label class="row"><span><span class="legend-swatch" style="background:#0f766e"></span>MRMS QPE 6 h</span><input id="mrmsToggle" type="checkbox" {mrms_checked}></label>
      <label class="row"><span><span class="legend-swatch" style="background:#7c3aed"></span>Earth2 storm-total</span><input id="totalToggle" type="checkbox"></label>
      <label class="row"><span><span class="legend-swatch" style="background:#115e59"></span>MRMS storm-total</span><input id="mrmsTotalToggle" type="checkbox" {compare_disabled}></label>
      <label class="row"><span><span class="legend-swatch" style="background:#b45309"></span>Diff (E2 − MRMS)</span><input id="diffToggle" type="checkbox" {compare_disabled}></label>
      <label class="row"><span><span class="legend-swatch" style="background:#ea580c"></span>Terrain screening</span><input id="terrainToggle" type="checkbox"></label>
      <label class="row"><span><span class="legend-swatch" style="background:#dc2626"></span>Precip×terrain hotspot</span><input id="hotspotToggle" type="checkbox" checked></label>
      <label class="row"><span><span class="legend-swatch" style="background:#64748b"></span>Roads</span><input id="roadsToggle" type="checkbox" checked></label>
      <label class="row"><span><span class="legend-swatch" style="background:#b91c1c"></span>Potential road disruption</span><input id="disruptToggle" type="checkbox" checked></label>
      <label class="row"><span><span class="legend-swatch" style="background:#0f766e"></span>Rail / subway stations</span><input id="stationsToggle" type="checkbox" checked></label>
      <label class="row"><span><span class="legend-swatch" style="background:#0369a1"></span>Catch basins (sampled)</span><input id="basinsToggle" type="checkbox"></label>
      <label class="row"><span><span class="legend-swatch" style="background:#e11d48"></span>311 flood-related reports</span><input id="reportsToggle" type="checkbox" checked></label>
    </section>

    <section>
      <h2>Run metadata</h2>
      <div class="metric"><span>Init</span><b id="initMeta">—</b></div>
      <div class="metric"><span>Model</span><b>FCN + PrecipAFNO</b></div>
      <div class="metric"><span>311 reports</span><b id="reportsMeta">—</b></div>
      <div class="metric"><span>Roads flagged</span><b id="disruptMeta">—</b></div>
      <div class="metric"><span>MRMS product</span><b style="max-width:160px">{mrms_product}</b></div>
    </section>

    <section>
      <details class="limits" open>
        <summary>What this is / is not</summary>
        <p><b>Earth2 precip:</b> six-hour accumulations (mm) from FCN + PrecipitationAFNO, GFS ICs. Display resampled from ~0.25° — not street-scale truth.</p>
        <p><b>MRMS QPE:</b> MultiSensor quantitative precipitation estimate (1 h), summed to the same 6 h windows. Observed rainfall reference — not BREF reflectivity.</p>
        <p><b>Diff:</b> Earth2 minus MRMS on the display grid (exploratory; grids differ in native resolution).</p>
        <p><b>Terrain / 311 / roads:</b> context and complaints only — not inundation depth or emergency guidance.</p>
      </details>
    </section>
  </aside>

  <div id="mapStage" class="map-stage single">
    <div class="map-pane">
      <div class="pane-label" id="leftLabel">Earth2 (+ context)</div>
      <div id="mapLeft" class="map-el"></div>
    </div>
    <div class="map-pane" id="rightPane" hidden>
      <div class="pane-label">MRMS QPE (observed)</div>
      <div id="mapRight" class="map-el"></div>
    </div>
    <div class="disclaimer">Prototype for Hurricane Ida remnants (2021-09-01). OpenStreetMap basemap. Layers are exploratory research products.</div>
  </div>

  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <script>
    const manifest = {json.dumps(manifest)};
    const assets = "ida_v0_assets/";
    const bounds = manifest.precip.bounds;
    const hasMrms = Boolean(manifest.mrms_qpe);

    const mapLeft = L.map("mapLeft", {{ zoomControl: true }}).fitBounds(bounds);
    L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
      maxZoom: 19, attribution: "&copy; OpenStreetMap"
    }}).addTo(mapLeft);

    let mapRight = null;
    let mapsLinked = false;
    let syncingMaps = false;
    function syncMaps(source, target) {{
      if (!target || syncingMaps) return;
      syncingMaps = true;
      try {{
        target.setView(source.getCenter(), source.getZoom(), {{ animate: false }});
      }} finally {{
        syncingMaps = false;
      }}
    }}
    function ensureRightMap() {{
      if (!hasMrms || mapRight) return mapRight;
      mapRight = L.map("mapRight", {{ zoomControl: true }});
      L.tileLayer("https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png", {{
        maxZoom: 19, attribution: "&copy; OpenStreetMap"
      }}).addTo(mapRight);
      if (!mapsLinked) {{
        mapLeft.on("moveend", () => syncMaps(mapLeft, mapRight));
        mapRight.on("moveend", () => syncMaps(mapRight, mapLeft));
        mapsLinked = true;
      }}
      return mapRight;
    }}

    const e2Overlays = manifest.precip.frames.map((frame) =>
      L.imageOverlay(assets + frame.png, bounds, {{ opacity: 0.72, interactive: false }})
    );
    let activeE2 = e2Overlays[1] || e2Overlays[0];
    activeE2.addTo(mapLeft);

    const mrmsOverlays = hasMrms
      ? manifest.mrms_qpe.frames.map((frame) => L.imageOverlay(assets + frame.png, bounds, {{ opacity: 0.72, interactive: false }}))
      : [];
    let activeMrms = mrmsOverlays[1] || mrmsOverlays[0] || null;

    const diffOverlays = hasMrms
      ? manifest.mrms_qpe.diff_frames.map((frame) => L.imageOverlay(assets + frame.png, bounds, {{ opacity: 0.7, interactive: false }}))
      : [];
    let activeDiff = null;

    const totalLayer = L.imageOverlay(assets + manifest.precip.storm_total_png, bounds, {{ opacity: 0.65 }});
    const mrmsTotalLayer = hasMrms
      ? L.imageOverlay(assets + manifest.mrms_qpe.storm_total_png, bounds, {{ opacity: 0.65 }})
      : null;
    const terrainLayer = manifest.terrain.terrain_png
      ? L.imageOverlay(assets + manifest.terrain.terrain_png, bounds, {{ opacity: 0.55 }})
      : null;
    const hotspotLayer = L.imageOverlay(assets + manifest.disruption.hotspot_png, bounds, {{ opacity: 0.55 }}).addTo(mapLeft);

    let compareMode = false;
    let roadsLayer = null, disruptLayer = null, stationsLayer = null, basinsLayer = null;
    const reportsLayer = L.markerClusterGroup({{ showCoverageOnHover: false }});

    function setLead(index) {{
      const frame = manifest.precip.frames[index];
      if (mapLeft.hasLayer(activeE2)) mapLeft.removeLayer(activeE2);
      activeE2 = e2Overlays[index];
      if (document.getElementById("precipToggle").checked) activeE2.addTo(mapLeft);

      if (hasMrms) {{
        const mrms = manifest.mrms_qpe.frames[index];
        const diff = manifest.mrms_qpe.diff_frames[index];
        if (activeMrms) {{
          if (mapLeft.hasLayer(activeMrms)) mapLeft.removeLayer(activeMrms);
          if (mapRight && mapRight.hasLayer(activeMrms)) mapRight.removeLayer(activeMrms);
        }}
        activeMrms = mrmsOverlays[index];
        if (compareMode && mapRight) activeMrms.addTo(mapRight);
        else if (document.getElementById("mrmsToggle").checked) activeMrms.addTo(mapLeft);

        if (activeDiff && mapLeft.hasLayer(activeDiff)) mapLeft.removeLayer(activeDiff);
        activeDiff = diffOverlays[index];
        if (document.getElementById("diffToggle").checked && !compareMode) activeDiff.addTo(mapLeft);

        document.getElementById("mrmsReadout").textContent = mrms.max_mm.toFixed(1) + " mm";
        document.getElementById("diffReadout").textContent = diff.mean_diff_mm.toFixed(1) + " mm";
      }} else {{
        document.getElementById("mrmsReadout").textContent = "n/a";
        document.getElementById("diffReadout").textContent = "n/a";
      }}

      document.getElementById("leadReadout").textContent = "+" + frame.lead_hours + " h";
      document.getElementById("validReadout").textContent = frame.valid_time.replace("T", " ");
      document.getElementById("precipReadout").textContent = frame.max_mm.toFixed(1) + " mm";
      document.getElementById("leftLabel").textContent = compareMode
        ? ("Earth2 · +" + frame.lead_hours + " h")
        : "Earth2 (+ context)";
    }}

    function setCompare(on) {{
      compareMode = on && hasMrms;
      const stage = document.getElementById("mapStage");
      const rightPane = document.getElementById("rightPane");
      document.getElementById("modeSingle").classList.toggle("active", !compareMode);
      document.getElementById("modeCompare").classList.toggle("active", compareMode);
      if (compareMode) {{
        stage.classList.add("compare"); stage.classList.remove("single");
        rightPane.hidden = false;
        [hotspotLayer, totalLayer, terrainLayer].forEach((layer) => {{ if (layer && mapLeft.hasLayer(layer)) mapLeft.removeLayer(layer); }});
        if (activeDiff && mapLeft.hasLayer(activeDiff)) mapLeft.removeLayer(activeDiff);
        if (activeMrms && mapLeft.hasLayer(activeMrms)) mapLeft.removeLayer(activeMrms);
        document.getElementById("mrmsToggle").checked = false;
        document.getElementById("diffToggle").checked = false;
        document.getElementById("precipToggle").checked = true;
        setTimeout(() => {{
          mapLeft.invalidateSize();
          const right = ensureRightMap();
          right.invalidateSize();
          right.setView(mapLeft.getCenter(), mapLeft.getZoom(), {{ animate: false }});
          setLead(Number(document.getElementById("leadSlider").value));
        }}, 150);
        return;
      }} else {{
        stage.classList.add("single"); stage.classList.remove("compare");
        rightPane.hidden = true;
        if (mapRight && activeMrms && mapRight.hasLayer(activeMrms)) mapRight.removeLayer(activeMrms);
        setTimeout(() => mapLeft.invalidateSize(), 50);
      }}
      setLead(Number(document.getElementById("leadSlider").value));
    }}

    document.getElementById("initMeta").textContent = (manifest.precip.init_time || "").slice(0, 16);
    document.getElementById("reportsMeta").textContent = String(manifest.reports.n_reports);
    document.getElementById("disruptMeta").textContent = manifest.disruption.n_disrupted + " / " + manifest.disruption.n_roads;
    document.getElementById("leadSlider").addEventListener("input", (event) => setLead(Number(event.target.value)));
    document.getElementById("modeSingle").addEventListener("click", () => setCompare(false));
    document.getElementById("modeCompare").addEventListener("click", () => setCompare(true));
    setLead(Number(document.getElementById("leadSlider").value));

    function bindToggle(id, layer) {{
      const input = document.getElementById(id);
      if (!input || input.disabled) return;
      input.addEventListener("change", () => {{
        if (!layer) return;
        if (input.checked) layer.addTo(mapLeft); else mapLeft.removeLayer(layer);
      }});
    }}
    bindToggle("totalToggle", totalLayer);
    bindToggle("mrmsTotalToggle", mrmsTotalLayer);
    bindToggle("terrainToggle", terrainLayer);
    bindToggle("hotspotToggle", hotspotLayer);
    document.getElementById("precipToggle").addEventListener("change", (event) => {{
      if (event.target.checked) activeE2.addTo(mapLeft); else mapLeft.removeLayer(activeE2);
    }});
    if (hasMrms) {{
      document.getElementById("mrmsToggle").addEventListener("change", (event) => {{
        if (compareMode) return;
        if (event.target.checked) activeMrms.addTo(mapLeft); else mapLeft.removeLayer(activeMrms);
      }});
      document.getElementById("diffToggle").addEventListener("change", (event) => {{
        if (compareMode) return;
        if (event.target.checked) activeDiff.addTo(mapLeft); else mapLeft.removeLayer(activeDiff);
      }});
    }}

    fetch(assets + manifest.infra.roads_geojson).then((r) => r.json()).then((geo) => {{
      roadsLayer = L.geoJSON(geo, {{
        style: {{ color: "#64748b", weight: 1, opacity: 0.55 }},
        onEachFeature: (feature, layer) => layer.bindTooltip(feature.properties.st_name || "road")
      }});
      if (document.getElementById("roadsToggle").checked) roadsLayer.addTo(mapLeft);
      bindToggle("roadsToggle", roadsLayer);
    }});
    fetch(assets + manifest.disruption.disrupted_geojson).then((r) => r.json()).then((geo) => {{
      disruptLayer = L.geoJSON(geo, {{
        style: {{ color: "#b91c1c", weight: 3, opacity: 0.9 }},
        onEachFeature: (feature, layer) => layer.bindTooltip((feature.properties.st_name || "road") + " · potential disruption (heuristic)")
      }}).addTo(mapLeft);
      bindToggle("disruptToggle", disruptLayer);
    }});
    fetch(assets + manifest.infra.stations_geojson).then((r) => r.json()).then((geo) => {{
      stationsLayer = L.geoJSON(geo, {{
        pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{
          radius: 5, color: "#0f766e", weight: 1, fillColor: "#14b8a6", fillOpacity: 0.9
        }}),
        onEachFeature: (feature, layer) => layer.bindTooltip(feature.properties.name || "Station")
      }}).addTo(mapLeft);
      bindToggle("stationsToggle", stationsLayer);
    }});
    if (manifest.infra.basins_geojson) {{
      fetch(assets + manifest.infra.basins_geojson).then((r) => r.json()).then((geo) => {{
        basinsLayer = L.geoJSON(geo, {{
          pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{
            radius: 2, color: "#0369a1", weight: 0, fillColor: "#0ea5e9", fillOpacity: 0.55
          }}),
          onEachFeature: (feature, layer) => layer.bindTooltip(feature.properties.unitid || "Catch basin")
        }});
        bindToggle("basinsToggle", basinsLayer);
      }});
    }}
    if (manifest.reports.geojson) {{
      fetch(assets + manifest.reports.geojson).then((r) => r.json()).then((geo) => {{
        L.geoJSON(geo, {{
          pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{
            radius: 4, color: "#9f1239", weight: 1, fillColor: "#e11d48", fillOpacity: 0.85
          }}),
          onEachFeature: (feature, layer) => {{
            const p = feature.properties;
            layer.bindTooltip((p.descriptor || "Flood report") + " · " + (p.created_date || ""));
          }}
        }}).eachLayer((layer) => reportsLayer.addLayer(layer));
        reportsLayer.addTo(mapLeft);
        bindToggle("reportsToggle", reportsLayer);
      }});
    }}
  </script>
</body>
</html>
"""
    html_path.write_text(html)
    return html_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Ida v0 Earth2 interactive map assets.")
    parser.add_argument("--global-forecast", type=Path, default=DEFAULT_GLOBAL)
    parser.add_argument("--clipped-forecast", type=Path, default=DEFAULT_CLIPPED)
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument(
        "--with-mrms-qa",
        action="store_true",
        help="Fetch NOAA MRMS MultiSensor QPE (not BREF) and add Earth2 vs observed compare layers.",
    )
    parser.add_argument(
        "--skip-infra",
        action="store_true",
        help="Reuse existing roads/stations geojson assets if present.",
    )
    args = parser.parse_args()

    _ensure_dirs()
    if not args.skip_normalize:
        if not args.global_forecast.exists():
            raise SystemExit(f"Missing global forecast NetCDF: {args.global_forecast}")
        print(f"Normalizing {args.global_forecast} → {args.clipped_forecast}")
        normalize_earth2_forecast(args.global_forecast, args.clipped_forecast)
    elif not args.clipped_forecast.exists():
        raise SystemExit(f"Missing clipped forecast: {args.clipped_forecast}")

    print("Exporting precip layers…")
    precip_meta = export_precip_layers(args.clipped_forecast)
    print(f"  {len(precip_meta['frames'])} leads; storm-total max {precip_meta['storm_total_max_mm']:.1f} mm")

    mrms_meta = None
    if args.with_mrms_qa:
        from nyc_flash_flood_case_study.mrms_qpe_qa import build_mrms_qpe_qa

        print("Fetching MRMS MultiSensor QPE and aggregating to 6 h leads…")
        init = precip_meta["init_time"].split(".")[0]
        mrms_meta = build_mrms_qpe_qa(precip_meta["frames"], init_time=init)
        print(
            f"  MRMS storm-total max {mrms_meta['storm_total_max_mm']:.1f} mm; "
            f"shared vmax {mrms_meta['shared_display_vmax_mm']:.1f} mm"
        )

    if args.skip_infra and (ASSETS_DIR / "roads.geojson").exists() and (ASSETS_DIR / "subway_stations.geojson").exists():
        print("Reusing cached infrastructure assets…")
        infra_meta = None
        if (ASSETS_DIR / "manifest.json").exists():
            infra_meta = json.loads((ASSETS_DIR / "manifest.json").read_text()).get("infra")
        if not infra_meta:
            infra_meta = {
                "stations_geojson": "subway_stations.geojson",
                "basins_geojson": "catch_basins.geojson",
                "roads_geojson": "roads.geojson",
                "n_stations": 0,
                "n_basins": 0,
                "n_roads": 0,
            }
    else:
        print("Fetching infrastructure…")
        infra_meta = fetch_infrastructure()
        print(f"  roads={infra_meta['n_roads']} stations={infra_meta['n_stations']}")

    print("Building disruption heuristic…")
    disruption_meta = build_disruption_heuristic(precip_meta)
    print(f"  flagged={disruption_meta['n_disrupted']}")

    print("Exporting 311 + terrain…")
    reports_meta = export_311()
    terrain_meta = export_terrain_assets()

    manifest = {
        "event": "Hurricane Ida remnants over NYC",
        "study_area": STUDY_AREA["name"],
        "precip": precip_meta,
        "infra": infra_meta,
        "disruption": disruption_meta,
        "reports": reports_meta,
        "terrain": terrain_meta,
        "disclaimer": (
            "Exploratory prototype. Not inundation depth, not emergency guidance, "
            "not a verified flood forecast. MRMS QPE is the observed precip reference; BREF is not used."
        ),
    }
    if mrms_meta is not None:
        manifest["mrms_qpe"] = mrms_meta

    manifest_path = ASSETS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    html_path = write_html(manifest)
    print(f"Wrote {manifest_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import folium
import rasterio
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster
from matplotlib import colormaps
from matplotlib.colors import LightSource

from .config import EVENT_OUTPUTS, FIGURES_DIR, IMPACT_DATASETS, STUDY_AREA, TERRAIN_DATASETS


def plot_susceptibility_map() -> Path:
    with rasterio.open(EVENT_OUTPUTS["susceptibility_map"]) as src:
        data = src.read(1, masked=True)
        bounds = src.bounds
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        data,
        cmap="YlOrRd",
        origin="upper",
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        interpolation="nearest",
    )
    ax.set_title("Flash-flood susceptibility prototype")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, label="Relative index (not probability)")
    ax.set_aspect("equal")
    out = FIGURES_DIR / "susceptibility_map.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_rainfall_and_susceptibility() -> Path:
    with rasterio.open(EVENT_OUTPUTS["runoff_index"]) as src:
        runoff = src.read(1, masked=True)
        bounds = src.bounds
    with rasterio.open(EVENT_OUTPUTS["susceptibility_map"]) as src:
        susceptibility = src.read(1, masked=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    runoff_image = axes[0].imshow(runoff, cmap="Blues", origin="upper", extent=extent, interpolation="nearest")
    susceptibility_image = axes[1].imshow(susceptibility, cmap="YlOrRd", origin="upper", extent=extent, interpolation="nearest")
    axes[0].set_title("Runoff index")
    axes[1].set_title("Event susceptibility prototype")
    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
    fig.colorbar(runoff_image, ax=axes[0], label="Relative index")
    fig.colorbar(susceptibility_image, ax=axes[1], label="Relative index (not probability)")
    out = FIGURES_DIR / "rainfall_and_susceptibility.png"
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_high_resolution_terrain_context() -> Path:
    with rasterio.open(TERRAIN_DATASETS["dem_highres"]) as dem_source:
        elevation = dem_source.read(1, masked=True).astype(np.float32)
        bounds = dem_source.bounds
        resolution = dem_source.res
    with rasterio.open(EVENT_OUTPUTS["terrain_screening_highres"]) as source:
        screening = source.read(1, masked=True)
    report_path = IMPACT_DATASETS["nyc_311"]
    observations = None
    if report_path.exists():
        import pandas as pd

        observations = pd.read_csv(report_path)

    extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
    valid_elevation = elevation.compressed()
    hillshade = LightSource(azdeg=315, altdeg=42).hillshade(
        elevation.filled(float(np.nanmedian(valid_elevation))),
        vert_exag=1.5,
        dx=abs(resolution[0]),
        dy=abs(resolution[1]),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), constrained_layout=True)
    dem_image = axes[0].imshow(elevation, cmap="terrain", origin="upper", extent=extent, interpolation="nearest")
    axes[0].imshow(hillshade, cmap="gray", origin="upper", extent=extent, alpha=0.25, interpolation="nearest")
    axes[0].set_title("USGS terrain | approx. 10 m")
    fig.colorbar(dem_image, ax=axes[0], label="Elevation (m)")

    index_image = axes[1].imshow(screening, cmap="YlOrRd", origin="upper", extent=extent, vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Terrain-only screening | not probability")
    fig.colorbar(index_image, ax=axes[1], label="Relative terrain index")

    if observations is not None and not observations.empty:
        counts, x_edges, y_edges = np.histogram2d(
            observations["lon"],
            observations["lat"],
            bins=(80, 48),
            range=((bounds.left, bounds.right), (bounds.bottom, bounds.top)),
        )
        density = axes[2].imshow(
            np.ma.masked_where(counts.T == 0, counts.T),
            cmap="magma",
            origin="lower",
            extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
            interpolation="nearest",
        )
        fig.colorbar(density, ax=axes[2], label="311 reports per ~500 m cell")
    axes[2].set_title("Event flood-related 311 reports")

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
    fig.suptitle("South Queens | terrain context, not a flood-depth forecast")
    output = FIGURES_DIR / "high_resolution_terrain_context.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _raster_overlay(path: Path, cmap_name: str, opacity: float = 0.78):
    with rasterio.open(path) as src:
        data = src.read(1, masked=True).astype(np.float32)
        bounds = src.bounds
        values = data.compressed()
    if values.size == 0:
        raise ValueError(f"Raster has no valid values: {path}")
    low, high = np.nanpercentile(values, [2, 98])
    if np.isclose(low, high):
        low, high = float(np.nanmin(values)), float(np.nanmax(values) + 1e-6)
    normalized = np.clip((data.filled(low) - low) / (high - low), 0, 1)
    rgba = colormaps[cmap_name](normalized, bytes=True)
    rgba[..., 3] = np.where(np.ma.getmaskarray(data), 0, int(opacity * 255)).astype(np.uint8)
    overlay_bounds = [[bounds.bottom, bounds.left], [bounds.top, bounds.right]]
    return rgba, overlay_bounds


def generate_interactive_event_map() -> Path:
    bounds = STUDY_AREA["bounds"]
    center = [(bounds["ymin"] + bounds["ymax"]) / 2, (bounds["xmin"] + bounds["xmax"]) / 2]
    event_map = folium.Map(location=center, zoom_start=11, tiles="OpenStreetMap", control_scale=True)

    rainfall_path = EVENT_OUTPUTS["rainfall_total"]
    susceptibility_path = EVENT_OUTPUTS["susceptibility_map"]
    if rainfall_path.exists():
        image, image_bounds = _raster_overlay(rainfall_path, "Blues")
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=image_bounds,
            opacity=0.78,
            name="MRMS source field (units unknown)",
            show=False,
        ).add_to(event_map)
    highres_screening = EVENT_OUTPUTS["terrain_screening_highres"]
    if highres_screening.exists():
        image, image_bounds = _raster_overlay(highres_screening, "YlOrRd")
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=image_bounds,
            opacity=0.82,
            name="Terrain screening (USGS approx. 10 m; no rainfall)",
            show=True,
        ).add_to(event_map)
    highres_dem = TERRAIN_DATASETS["dem_highres"]
    if highres_dem.exists():
        image, image_bounds = _raster_overlay(highres_dem, "terrain", opacity=0.78)
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=image_bounds,
            opacity=0.78,
            name="Elevation (USGS approx. 10 m)",
            show=False,
        ).add_to(event_map)
    if susceptibility_path.exists():
        image, image_bounds = _raster_overlay(susceptibility_path, "YlOrRd")
        folium.raster_layers.ImageOverlay(
            image=image,
            bounds=image_bounds,
            opacity=0.78,
            name="Coarse event index (unvalidated)",
            show=not highres_screening.exists(),
        ).add_to(event_map)

    reports = MarkerCluster(name="NYC 311 flood-related reports", show=True).add_to(event_map)
    if IMPACT_DATASETS["nyc_311"].exists():
        import pandas as pd

        observations = pd.read_csv(IMPACT_DATASETS["nyc_311"])
        for row in observations.itertuples(index=False):
            descriptor = escape(str(getattr(row, "descriptor", "Flood-related report")))
            created = escape(str(getattr(row, "created_date", "Date unavailable")))
            folium.CircleMarker(
                location=[float(row.lat), float(row.lon)],
                radius=3,
                color="#a52222",
                weight=1,
                fill=True,
                fill_color="#e34a33",
                fill_opacity=0.8,
                tooltip=f"{descriptor} | {created}",
            ).add_to(reports)
    event_map.fit_bounds([[bounds["ymin"], bounds["xmin"]], [bounds["ymax"], bounds["xmax"]]])
    folium.LayerControl(collapsed=False).add_to(event_map)
    if highres_screening.exists():
        LinearColormap(["#fff7bc", "#fec44f", "#fe9929", "#cc4c02"], vmin=0, vmax=1, caption="Terrain screening index (not probability)").add_to(event_map)
    warning = (
        "Terrain screening is derived from USGS 1/3-arc-second elevation (about 10 m) only; it excludes rainfall, "
        "soils, imperviousness, and drainage infrastructure. The event index is coarser and unvalidated. "
        "MRMS source units are unresolved. 311 reports are complaint locations, not confirmed flood depths. "
        "This is not an inundation or flood-depth forecast."
    )
    event_map.get_root().html.add_child(folium.Element(
        '<details style="position:fixed;bottom:28px;left:50px;z-index:9999;max-width:320px;'
        'padding:7px 10px;background:#fff;border:1px solid #666;font-size:12px;line-height:1.35">'
        '<summary style="cursor:pointer">Data limitations</summary>'
        f'<div style="padding-top:6px">{warning}</div></details>'
    ))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "interactive_event_map.html"
    event_map.save(output)
    return output


__all__ = [
    "plot_susceptibility_map",
    "plot_rainfall_and_susceptibility",
    "plot_high_resolution_terrain_context",
    "generate_interactive_event_map",
]

from __future__ import annotations

import numpy as np
import rasterio

from .config import EVENT_OUTPUTS, LANDCOVER_DATASETS, PROCESSED_DIR, RAW_DIR
from .preprocessing import align_to_reference, build_land_mask, load_raster, resolve_rainfall_source


def compute_runoff_index() -> None:
    rainfall_source = resolve_rainfall_source()
    if rainfall_source is None:
        raise FileNotFoundError("No rainfall raster was available for runoff estimation.")
    rainfall_total, _, _, _ = load_raster(rainfall_source)
    impervious = align_to_reference(LANDCOVER_DATASETS["impervious"], rainfall_source)
    impervious = np.clip(impervious, 0, 1)
    antecedent, _, _, _ = load_raster(EVENT_OUTPUTS["antecedent_wetness"])
    runoff = (rainfall_total / np.nanmax(rainfall_total)) * 0.7 + impervious * 0.2 + antecedent * 0.1
    runoff = np.clip(runoff, 0, 1)
    profile = rasterio.open(rainfall_source).profile.copy()
    with rasterio.open(EVENT_OUTPUTS["runoff_index"], "w", **profile) as dst:
        dst.write(runoff.astype(np.float32), 1)


def compute_susceptibility() -> None:
    runoff, _, _, _ = load_raster(EVENT_OUTPUTS["runoff_index"])
    slope, _, _, _ = load_raster(PROCESSED_DIR / "slope.tif")
    antecedent, _, _, _ = load_raster(EVENT_OUTPUTS["antecedent_wetness"])
    dem, _, _, _ = load_raster(PROCESSED_DIR / "dem_processed.tif")

    lowland = np.clip((np.max(dem) - dem) / (np.max(dem) - np.min(dem) + 1e-6), 0, 1)
    slope_factor = 1.0 / (1.0 + slope)
    susceptibility = 0.45 * runoff + 0.25 * antecedent + 0.20 * lowland + 0.10 * slope_factor
    susceptibility = np.clip(susceptibility, 0, 1)
    land_mask = build_land_mask()
    rainfall_source = resolve_rainfall_source()
    if rainfall_source is None:
        raise FileNotFoundError("No rainfall raster was available for susceptibility output creation.")
    profile = rasterio.open(rainfall_source).profile.copy()
    profile.update(dtype="float32", nodata=-9999.0)
    susceptibility[land_mask == 0] = profile["nodata"]
    with rasterio.open(EVENT_OUTPUTS["susceptibility_map"], "w", **profile) as dst:
        dst.write(susceptibility.astype(np.float32), 1)


__all__ = ["compute_runoff_index", "compute_susceptibility"]

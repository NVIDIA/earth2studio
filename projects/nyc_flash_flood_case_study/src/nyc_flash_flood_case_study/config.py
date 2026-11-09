from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

STUDY_AREA = {
    "name": "South Queens / Jamaica Bay",
    "bounds": {
        "xmin": -74.12,
        "ymin": 40.55,
        "xmax": -73.72,
        "ymax": 40.73,
    },
    "crs": "EPSG:4326",
}

EVENT = {
    "start": "2021-09-01 00:00:00",
    "end": "2021-09-02 23:00:00",
    "name": "Hurricane Ida remnants over NYC",
}

GEOM_DATASETS = {
    "nyc_boundary": RAW_DIR / "nyc_boundary.geojson",
    "study_area": RAW_DIR / "study_area.geojson",
}

RAINFALL_DATASETS = {
    "stage_iv": RAW_DIR / "rainfall" / "stage_iv",
    "gpm_imerg": RAW_DIR / "rainfall" / "gpm_imerg",
    "earth2_force": RAW_DIR / "rainfall" / "earth2_force",
}

TERRAIN_DATASETS = {
    "dem": RAW_DIR / "terrain" / "dem.tif",
    "dem_highres": RAW_DIR / "terrain" / "dem_usgs_1_3_arcsec.tif",
    "slope": RAW_DIR / "terrain" / "slope.tif",
    "flow_accumulation": RAW_DIR / "terrain" / "flow_accumulation.tif",
}

LANDCOVER_DATASETS = {
    "impervious": RAW_DIR / "landcover" / "impervious_surface.tif",
    "nlcd": RAW_DIR / "landcover" / "nlcd_landcover.tif",
}

LAND_MASK_DATASETS = {
    "natural_earth_land": RAW_DIR / "landcover" / "ne_10m_land.zip",
}

IMPACT_DATASETS = {
    "nyc_311": RAW_DIR / "impacts" / "nyc_311_flooding.csv",
    "flood_reports": RAW_DIR / "impacts" / "flood_reports.geojson",
    "sentinel_water": RAW_DIR / "impacts" / "water_extent.geojson",
}

EVENT_OUTPUTS = {
    "rainfall_total": PROCESSED_DIR / "rainfall_total.tif",
    "rainfall_peak_hour": PROCESSED_DIR / "rainfall_peak_hour.tif",
    "antecedent_wetness": PROCESSED_DIR / "antecedent_wetness.tif",
    "runoff_index": PROCESSED_DIR / "runoff_index.tif",
    "susceptibility_map": PROCESSED_DIR / "flash_flood_susceptibility.tif",
    "land_mask": PROCESSED_DIR / "land_mask.tif",
    "terrain_screening_highres": PROCESSED_DIR / "terrain_screening_10m.tif",
    "land_mask_highres": PROCESSED_DIR / "land_mask_10m.tif",
    "validation_points": PROCESSED_DIR / "validation_points.geojson",
}

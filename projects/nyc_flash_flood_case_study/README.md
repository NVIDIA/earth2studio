# NYC Flash-Flood Susceptibility Case Study

This project models flash-flood susceptibility for a compact New York City study area during a historical extreme-precipitation event, using Earth2Studio meteorological forcing as the atmospheric input and open urban geospatial data for terrain, imperviousness, and antecedent wetness.

## Study framing

This is a susceptibility study, not a direct flood-prediction model. The core logic is:

- Earth2Studio provides rainfall and near-surface meteorological forcing.
- Terrain, land cover, and antecedent conditions determine which urban areas are hydrologically vulnerable.
- Observed flood impacts are used only for validation and calibration context.

## Recommended historical event

- Hurricane Ida remnants over NYC, 2021-09-01 to 2021-09-02
- Use a compact sub-area, e.g. South Queens / Jamaica Bay / Howard Beach / Ozone Park corridor

## Data sources for the real event pipeline

This workflow is designed for the real Ida event and uses the following public sources:

- NOAA MRMS public S3 bucket index for precipitation products during the event window
- USGS 3DEP public S3 index for the elevation model and terrain derivatives
- NLCD imperviousness source page for urban runoff conditions
- NYC 311 service requests endpoint for flood- and drainage-related validation
- FEMA / USGS context data for flood observations and local validation

Validated public endpoints used by this workflow:

- NYC 311: `https://data.cityofnewyork.us/resource/erm2-nwe9.json`
- NOAA MRMS bucket index: `https://noaa-mrms-pds.s3.amazonaws.com/`
- USGS 3DEP index: `https://prd-tnm.s3.amazonaws.com/`
- NLCD: `https://www.mrlc.gov/data`

## Core workflow

1. Download open datasets for the selected event and study area.
2. Preprocess rainfall, elevation, imperviousness, wetness, and event observations.
3. Build the rainfall-runoff and terrain-routing inputs.
4. Estimate event-specific susceptibility using a transparent scoring or machine-learning model.
5. Validate against observed flood impacts, complaints, and water extent data.
6. Generate maps and figures for the final case study.

## Real-event execution

```bash
cd /home/jesscmiel/projects/earth2studio/projects/nyc_flash_flood_case_study
. .venv/bin/activate
python -m nyc_flash_flood_case_study --real-ida --download-only
python -m nyc_flash_flood_case_study --real-ida
```

The real-event workflow is intentionally framed as a susceptibility model rather than a direct flood-prediction system: atmospheric forcing identifies the meteorological driver, while terrain, imperviousness, and antecedent wetness determine where urban flooding is most likely.

## Real-event processing steps

1. Download the MRMS precipitation index and the NYC 311 request feed.
2. Clip the selected rainfall fields to the South Queens / Jamaica Bay study area.
3. Download or locate the USGS DEM and NLCD imperviousness cover for the same extent.
4. Derive slope, local elevation, imperviousness, and antecedent wetness.
5. Build a susceptibility field from rainfall, slope, imperviousness, and runoff accumulation.
6. Compare hotspots to NYC 311 flood complaints and any nearby flood extent observations.

This keeps the workflow faithful to an urban hydrologic susceptibility case study, without claiming that Earth2Studio or the workflow itself is a direct flood model.

## Project layout

- `data/` raw and processed datasets
- `figures/` output maps and validation plots
- `scripts/` workflow entry points
- `src/nyc_flash_flood_case_study/` case-study implementation

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m nyc_flash_flood_case_study --help
```

## Outputs

The workflow produces:

- event rainfall raster or gridded field
- terrain, slope, and runoff-index layers
- imperviousness and antecedent-wetness layers
- susceptibility hotspot map
- an interactive HTML map with toggleable raster layers and geocoded 311 flood-related reports
- a CSV of susceptibility scores sampled at observed report locations
- a JSON summary of exploratory spatial overlap

## Higher-Resolution Terrain Views

Generate a clipped USGS 3DEP 1/3-arc-second DEM cutout (about 10 m), land mask, terrain-only screening index, and updated map/figure:

```bash
python -m nyc_flash_flood_case_study --high-resolution-terrain
```

The cutout is read from the current USGS tiles using remote range requests; only the study-area subset is saved. Outputs include `data/raw/terrain/dem_usgs_1_3_arcsec.tif`, `data/processed/terrain_screening_10m.tif`, `figures/high_resolution_terrain_context.png`, and the updated interactive map. The screening index describes relative low elevation and flatness only. It is not a flood probability and does not include rainfall, soil, impervious cover, drainage, or hydraulic routing. The coarse event field remains a separate map layer. The MRMS source layer's units are still unresolved and are not used in the high-resolution terrain index.

## Ida v0 Earth2 map (real precip)

Deterministic FCN + PrecipitationAFNO forecast for **2021-09-01T12:00:00 UTC**, clipped to South Queens, with lead-wise precip layers, terrain screening, OSM roads/stations, heuristic road-disruption overlay, and clustered 311 reports:

```bash
# Forecast (Earth2Studio GPU env)
/home/jesscmiel/projects/earth2studio/.venv/bin/python \
  projects/nyc_flash_flood_case_study/scripts/run_earth2_forecast.py \
  --init-time 2021-09-01T12:00:00 --steps 4 --members 1 \
  --output projects/nyc_flash_flood_case_study/data/forecasts/earth2_precip_ida_20210901T12.nc

# Map assets (case-study env)
cd projects/nyc_flash_flood_case_study
.venv/bin/python scripts/build_ida_v0_map.py \
  --global-forecast data/forecasts/earth2_precip_ida_20210901T12.nc \
  --clipped-forecast data/forecasts/nyc_precip_ida_20210901T12.nc
```

Open `figures/ida_v0_event_map.html`. See `figures/IDA_V0_DEMO_README.md` for the layer guide and disclaimers. This is **not** inundation depth or emergency guidance.

Optional MRMS **QPE** QA (not BREF) for Earth2 vs observed precip, including a side-by-side compare mode:

```bash
.venv/bin/python scripts/build_ida_v0_map.py \
  --skip-normalize --skip-infra \
  --clipped-forecast data/forecasts/nyc_precip_ida_20210901T12.nc \
  --with-mrms-qa
```

## Product Concept Demo

Build the interactive scenario workbench with synthetic moderate, severe, and extreme rainfall/depth-proxy scenarios:

```bash
python scripts/build_product_demo.py
```

Open `figures/product_demo.html` in a browser. The demo includes scenario switching, a six-hour timeline, map layers, clustered observed 311 report points, terrain context, a synthetic impact snapshot, and a downloadable JSON brief. Only terrain and 311 locations are real event/context inputs; all scenario rainfall, proxy depths, and exposed-area metrics are illustrative and are explicitly marked synthetic. Do not use the generated scenario values as engineering, emergency-management, or forecast guidance.

## Interpretation and validation limits

The current interactive map is a prototype, not an inundation animation or a validated flood forecast. The staged MRMS file is a single BREF field whose decoded parameter and units are unknown, and the current DEM and imperviousness files are demo rasters. The raster grid is also too coarse for street-scale interpretation. The map labels these layers accordingly and does not animate water extent.

Susceptibility pixels are now masked to land using Natural Earth 1:10m land polygons rasterized onto the model grid; the meteorological source field remains visible over both land and water. This is a coastline mask, not a local hydrography or drainage-network product, and the effective resolution remains limited by the coarse model grid.

The event 311 feed currently yields geocoded Street Flooding and Catch Basin Flooding reports within the study bounds. The workflow samples susceptibility at those locations and reports score percentiles as an exploratory spatial-overlap diagnostic. It does not report accuracy, ROC AUC, or calibration: 311 complaints are not a complete flood inventory, and there are no independently verified non-flood labels. A defensible skill evaluation requires event-matched precipitation in known units, real terrain and imperviousness rasters, and independent flood observations or carefully justified control locations.

## Forecast-to-flood extension

The first coupled path is implemented as Earth2Studio FCN + PrecipitationAFNO ensemble forcing followed by PySWMM. FCN predicts the atmospheric state; PrecipitationAFNO diagnoses six-hour accumulated precipitation in meters. The case-study adapter clips to the study area, converts units, removes the initial-state output, and preserves ensemble members and lead times. PySWMM then runs each member through a supplied calibrated SWMM model, replacing the configured rain-gage time series and summarizing node flooding frequency.

Generate a forecast from GFS initial conditions in the separate Earth2Studio environment. The first run downloads model packages and may take time:

```bash
/home/jesscmiel/projects/earth2studio/.venv/bin/python \
	/home/jesscmiel/projects/earth2studio/projects/nyc_flash_flood_case_study/scripts/run_earth2_forecast.py \
	--init-time 2021-09-01T00:00:00 --steps 4 --members 4 \
	--output /home/jesscmiel/projects/earth2studio/projects/nyc_flash_flood_case_study/data/forecasts/earth2_precip_ensemble.nc
```

Normalize and run in the case-study environment:

```bash
cd /home/jesscmiel/projects/earth2studio/projects/nyc_flash_flood_case_study
. .venv/bin/activate
pip install -e '.[swmm]'
python -m nyc_flash_flood_case_study.forecast \
	--input data/forecasts/earth2_precip_ensemble.nc \
	--output data/forecasts/nyc_precipitation_forcing.nc
python -m nyc_flash_flood_case_study.swmm \
	--forcing data/forecasts/nyc_precipitation_forcing.nc \
	--model /path/to/calibrated_network.inp \
	--gages /path/to/rain_gages.csv \
	--nodes JUNCTION_A,JUNCTION_B \
	--output-dir outputs/swmm_forecast
```

The rain-gage CSV must have `gage_name,series_name,lat,lon` columns. Each gage must be configured in the SWMM `[RAINGAGES]` section as `VOLUME` with a `TIMESERIES` source, and its reporting interval must match the six-hour forecast interval. The chosen series must also be referenced by the model's subcatchments. The SWMM file, gage locations, monitored node IDs, drainage network, and calibration are site-specific inputs; this repository does not include a NYC sewer model. Results are written to `swmm_node_timeseries.csv`, `swmm_member_summary.csv`, `swmm_flood_probability.csv`, and `swmm_run_metadata.json`.

Six-hour accumulations do not resolve short-duration convective peaks. The adapter does not invent hourly rainfall from these accumulations; operational street-flood modeling needs higher-frequency quantitative precipitation forecasts or a validated disaggregation method. A later 2D hydraulic step (for example LISFLOOD-FP or HEC-RAS) can use calibrated runoff/boundary conditions and high-resolution terrain to produce depth and extent grids. Terrain, soils, impervious cover, drainage infrastructure, and antecedent wetness belong in those downstream models, not in the weather forecast model. Ensemble outputs should be interpreted as conditional probabilities given the selected forecast and SWMM configuration, and validated against independent observations before being treated as flood guidance.

After running the real-event workflow, open `figures/interactive_event_map.html` in a browser. Validation artifacts are written to `data/processed/flood_reports_scored.csv` and `outputs/validation_report.json`.

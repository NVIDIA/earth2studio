# Ida v0 Demo — Earth2 NYC Flood-Impact Map

**Open the map:** [`ida_v0_event_map.html`](./ida_v0_event_map.html) (static HTML; needs network only for OSM tiles + CDN Leaflet).

## What ran

| Item | Value |
| --- | --- |
| Event | Hurricane Ida remnants over NYC |
| Init time | **2021-09-01T12:00:00 UTC** (GFS IC) |
| Model | FourCastNet (FCN) + PrecipitationAFNO |
| Leads | **+6, +12, +18, +24 h** (four positive 6 h intervals) |
| Study area | South Queens / Jamaica Bay |
| Global forecast | `data/forecasts/earth2_precip_ida_20210901T12.nc` |
| Clipped precip | `data/forecasts/nyc_precip_ida_20210901T12.nc` |
| Map | `figures/ida_v0_event_map.html` |
| Assets | `figures/ida_v0_assets/` |

Peak interval precip in the clipped display domain is around the **+12 h** lead (~25 mm / 6 h); storm-total (sum of intervals) reaches ~39 mm on this coarse grid.

## Rebuild

From the Earth2Studio repo root (GPU env for forecast):

```bash
/home/jesscmiel/projects/earth2studio/.venv/bin/python \
  projects/nyc_flash_flood_case_study/scripts/run_earth2_forecast.py \
  --init-time 2021-09-01T12:00:00 --steps 4 --members 1 \
  --output projects/nyc_flash_flood_case_study/data/forecasts/earth2_precip_ida_20210901T12.nc
```

Then in the case-study environment:

```bash
cd projects/nyc_flash_flood_case_study
.venv/bin/python scripts/build_ida_v0_map.py \
  --global-forecast data/forecasts/earth2_precip_ida_20210901T12.nc \
  --clipped-forecast data/forecasts/nyc_precip_ida_20210901T12.nc
```

## Layer guide — what is / is not

| Layer | What it is | What it is **not** |
| --- | --- | --- |
| **Earth2 6 h precip** | AI weather forecast precipitation accumulation for each 6-hour lead (mm), from FCN + PrecipitationAFNO with GFS initial conditions. Time slider = forecast leads. | Street-scale QPE, radar, or gauge rainfall. Native model grid is ~0.25°; the map bilinearly resamples for display only (not physical downscaling). |
| **Storm-total precip** | Sum of the positive 6 h interval fields over the forecast window. | Verified storm rainfall total for Ida. |
| **Terrain screening** | Relative low-elevation / flatness index from USGS ~10 m DEM. | Flood probability; excludes rainfall, soils, imperviousness, and drainage. |
| **Precip × terrain hotspot** | Cells where storm-total precip and terrain screening are both high (percentile/threshold rule). | Inundation extent or water depth. |
| **Roads / potential disruption** | OSM highway segments; disruption flag = heuristic overlap with hotspot cells (higher-class roads preferred on the overlay). | Traffic closure prediction, hydraulic road flooding, or emergency routing. |
| **Rail / subway stations** | OSM railway/subway station points in the study bbox. | Flooded-station inventory or MTA outage status. |
| **311 flood-related reports** | Geocoded NYC 311 Street / Catch Basin flooding complaints for the event window (~599 in bbox). Clustered markers. | Surveyed flood depths, extents, or a complete flood inventory. |

## Disclaimers (read these)

- This is an **exploratory research prototype**, not operational flood guidance.
- It does **not** produce water depth, calibrated sewer hydraulics, or emergency alerts.
- Six-hour AI precip cannot resolve Ida’s extreme ~1-hour rainfall peaks.
- 311 complaints are biased by who calls; absence of a point is not “no flooding.”
- Do not use these layers for engineering design, insurance, or emergency management decisions.

## Related files

- Case-study overview: `../README.md`
- Earlier Folium exploratory map: `interactive_event_map.html`
- Synthetic product concept (not this demo): `product_demo.html`

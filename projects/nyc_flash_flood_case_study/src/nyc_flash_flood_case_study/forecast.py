from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from .config import STUDY_AREA, PROJECT_ROOT


def normalize_earth2_forecast(source: str | Path, destination: str | Path) -> Path:
    """Clip Earth2Studio tp ensemble output and convert interval depths to millimeters."""
    source = Path(source)
    destination = Path(destination)
    if not source.exists():
        raise FileNotFoundError(f"Earth2Studio forecast file not found: {source}")

    with xr.open_dataset(source) as dataset:
        if "tp" not in dataset:
            raise ValueError("Forecast NetCDF must contain the Earth2Studio 'tp' variable.")
        precipitation = dataset["tp"]
        units = str(precipitation.attrs.get("units", "")).strip().lower()
        if units not in {"m", "meter", "meters", "metre", "metres", "mm", "millimeter", "millimeters"}:
            raise ValueError(f"Unsupported or missing precipitation units {units!r}; expected meters or millimeters.")
        if "time" in precipitation.dims:
            if precipitation.sizes["time"] != 1:
                raise ValueError("Forecast must contain exactly one initialization time.")
            init_time = np.datetime64(dataset["time"].values[0], "ns")
            precipitation = precipitation.isel(time=0, drop=True)
        elif "time" in dataset.coords:
            init_time = np.datetime64(dataset["time"].values[0], "ns")
        else:
            raise ValueError("Forecast must include its initialization time coordinate.")

        if "ensemble" not in precipitation.dims:
            precipitation = precipitation.expand_dims(ensemble=[0])
        required_dims = {"ensemble", "lead_time", "lat", "lon"}
        if not required_dims.issubset(precipitation.dims):
            raise ValueError(f"Forecast dimensions must include {sorted(required_dims)}; found {precipitation.dims}.")
        precipitation = precipitation.transpose("ensemble", "lead_time", "lat", "lon")

        lead_values = np.asarray(precipitation["lead_time"].values)
        if not np.issubdtype(lead_values.dtype, np.timedelta64):
            raise ValueError("lead_time must be encoded as timedelta64 coordinates.")
        lead_hours = lead_values.astype("timedelta64[s]").astype(np.int64) / 3600.0
        keep = lead_hours > 0
        lead_hours = lead_hours[keep]
        if lead_hours.size == 0:
            raise ValueError("Forecast contains no positive lead times; the initial-state step is not precipitation forcing.")
        if not np.all(np.diff(lead_hours) > 0):
            raise ValueError("Forecast lead times must be strictly increasing.")
        interval_hours = np.diff(np.concatenate(([0.0], lead_hours)))
        if not np.allclose(interval_hours, interval_hours[0]):
            raise ValueError("Forecast precipitation intervals must have a constant duration for SWMM forcing.")

        precipitation = precipitation.isel(lead_time=np.flatnonzero(keep))
        precipitation = precipitation.assign_coords(lead_time_hours=("lead_time", lead_hours))
        precipitation = precipitation.swap_dims({"lead_time": "lead_time_hours"}).drop_vars("lead_time")

        longitudes = ((precipitation["lon"].values + 180.0) % 360.0) - 180.0
        precipitation = precipitation.assign_coords(lon=longitudes).sortby("lon").sortby("lat")
        bounds = STUDY_AREA["bounds"]
        latitude_spacing = float(np.median(np.abs(np.diff(precipitation.lat.values))))
        longitude_spacing = float(np.median(np.abs(np.diff(precipitation.lon.values))))
        latitude_cells = precipitation.lat.where(
            (precipitation.lat >= bounds["ymin"] - latitude_spacing / 2)
            & (precipitation.lat <= bounds["ymax"] + latitude_spacing / 2),
            drop=True,
        ).values
        longitude_cells = precipitation.lon.where(
            (precipitation.lon >= bounds["xmin"] - longitude_spacing / 2)
            & (precipitation.lon <= bounds["xmax"] + longitude_spacing / 2),
            drop=True,
        ).values
        precipitation = precipitation.sel(lat=latitude_cells, lon=longitude_cells)
        if precipitation.sizes.get("lat", 0) == 0 or precipitation.sizes.get("lon", 0) == 0:
            raise ValueError("Earth2Studio forecast grid does not intersect the configured NYC study area.")

        scale = 1000.0 if units in {"m", "meter", "meters", "metre", "metres"} else 1.0
        precipitation = (precipitation.clip(min=0) * scale).astype(np.float32)
        precipitation.name = "precipitation_mm"
        precipitation.attrs.update(
            units="mm",
            accumulation_duration_hours=float(interval_hours[0]),
            source_variable="tp",
            source_units=units,
            spatial_resolution_note="Native Earth2Studio grid cells selected by footprint intersection; no spatial downscaling applied.",
            grid_spacing_latitude_degrees=latitude_spacing,
            grid_spacing_longitude_degrees=longitude_spacing,
        )
        precipitation = precipitation.assign_coords(
            valid_time=("lead_time_hours", init_time + lead_hours.astype("timedelta64[h]").astype("timedelta64[ns]"))
        )
        output = precipitation.to_dataset()
        output.attrs.update(
            source_file=str(source.resolve()),
            forecast_initialization_time=str(init_time),
            forecast_model=str(dataset.attrs.get("forecast_model", "unspecified")),
            ensemble_method=str(dataset.attrs.get("ensemble_method", "unspecified")),
            interval_hours=float(interval_hours[0]),
            study_area=STUDY_AREA["name"],
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_netcdf(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize and clip Earth2Studio precipitation ensemble output.")
    parser.add_argument("--input", type=Path, required=True, help="Earth2Studio NetCDF containing 'tp'")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "forecasts" / "nyc_precipitation_forcing.nc",
    )
    args = parser.parse_args()
    print(normalize_earth2_forecast(args.input, args.output))


if __name__ == "__main__":
    main()
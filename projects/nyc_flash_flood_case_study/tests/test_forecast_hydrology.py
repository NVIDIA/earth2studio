from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nyc_flash_flood_case_study.forecast import normalize_earth2_forecast
from nyc_flash_flood_case_study.swmm import inject_forecast_rainfall, run_swmm_ensemble


def _earth2_fixture(path, units="m"):
    values = np.arange(72, dtype=np.float32).reshape(1, 2, 3, 3, 4) / 1000
    dataset = xr.Dataset(
        {"tp": (("time", "ensemble", "lead_time", "lat", "lon"), values)},
        coords={
            "time": [np.datetime64("2021-09-01T00")],
            "ensemble": [0, 1],
            "lead_time": np.array([0, 6, 12], dtype="timedelta64[h]"),
            "lat": [40.5, 40.75, 41.0],
            "lon": [285.75, 286.0, 286.25, 286.5],
        },
        attrs={"forecast_model": "test fixture"},
    )
    dataset["tp"].attrs["units"] = units
    dataset.to_netcdf(path)


def test_normalize_clips_converts_units_and_drops_initial_step(tmp_path):
    source = tmp_path / "earth2.nc"
    output = tmp_path / "nyc_forcing.nc"
    _earth2_fixture(source)

    normalize_earth2_forecast(source, output)

    with xr.open_dataset(output) as dataset:
        precipitation = dataset["precipitation_mm"]
        assert precipitation.dims == ("ensemble", "lead_time_hours", "lat", "lon")
        assert precipitation.shape == (2, 2, 2, 2)
        assert precipitation.attrs["units"] == "mm"
        assert np.array_equal(dataset["lead_time_hours"], [6, 12])
        assert np.allclose(dataset["lon"], [-74.0, -73.75])
        assert np.allclose(dataset["lat"], [40.5, 40.75])
        assert np.isclose(precipitation.isel(ensemble=0, lead_time_hours=0, lat=0, lon=0).item(), 13.0)


def test_normalize_rejects_unknown_units(tmp_path):
    source = tmp_path / "earth2.nc"
    _earth2_fixture(source, units="unknown")

    with pytest.raises(ValueError, match="Unsupported or missing precipitation units"):
        normalize_earth2_forecast(source, tmp_path / "nyc_forcing.nc")


def _forcing(amounts=(25.4, 50.8), interval_hours=6):
    return xr.DataArray(
        np.asarray(amounts, dtype=np.float32).reshape(-1, 1, 1),
        dims=("lead_time_hours", "lat", "lon"),
        coords={
            "lead_time_hours": np.arange(1, len(amounts) + 1) * interval_hours,
            "lat": [40.625],
            "lon": [-74.0],
            "valid_time": (
                "lead_time_hours",
                np.array(["2021-09-01T06", "2021-09-01T12"], dtype="datetime64[h]")[: len(amounts)],
            ),
        },
        attrs={"accumulation_duration_hours": interval_hours, "units": "mm"},
    )


def _swmm_template(interval="06:00"):
    return (
        "[OPTIONS]\nFLOW_UNITS CFS\n"
        "[RAINGAGES]\nRG1 VOLUME "
        f"{interval} 1.0 TIMESERIES TS1\n"
        "[TIMESERIES]\nTS1 09/01/2021 06:00 0\n"
        "[REPORT]\n"
    )


def _gage_locations():
    return pd.DataFrame([{"gage_name": "RG1", "series_name": "TS1", "lat": 40.625, "lon": -74.0}])


def test_swmm_timeseries_replaced_with_member_rainfall_and_converted_to_inches():
    result = inject_forecast_rainfall(_swmm_template(), _gage_locations(), _forcing(), "CFS")

    assert "TS1 09/01/2021 06:00 1.000000" in result
    assert "TS1 09/01/2021 12:00 2.000000" in result
    assert "TS1 09/01/2021 06:00 0\n" not in result


def test_swmm_requires_matching_rainfall_interval():
    with pytest.raises(ValueError, match="intervals must match"):
        inject_forecast_rainfall(_swmm_template("01:00"), _gage_locations(), _forcing(), "CFS")


def test_swmm_ensemble_runs_against_small_network(tmp_path):
    pyswmm = pytest.importorskip("pyswmm")
    package_dir = Path(pyswmm.__file__).parent
    model_path = package_dir / "tests" / "data" / "model_rain.inp"
    template = model_path.read_text().replace(
        "Gage1            INTENSITY 1:00     1.0      TIMESERIES test_rainfall",
        "Gage1            VOLUME    06:00  1.0      TIMESERIES test_rainfall",
    )
    model = tmp_path / "model.inp"
    model.write_text(template)
    forcing = tmp_path / "forcing.nc"
    precipitation = xr.DataArray(
        np.array([[[[25.4]], [[50.8]]], [[[50.8]], [[25.4]]]], dtype=np.float32),
        dims=("ensemble", "lead_time_hours", "lat", "lon"),
        coords={
            "ensemble": [0, 1],
            "lead_time_hours": [6, 12],
            "lat": [40.625],
            "lon": [-74.0],
            "valid_time": ("lead_time_hours", np.array(["2017-08-17T06", "2017-08-17T12"], dtype="datetime64[h]")),
        },
        name="precipitation_mm",
        attrs={"units": "mm", "accumulation_duration_hours": 6},
    )
    dataset = precipitation.to_dataset()
    dataset.attrs["forecast_initialization_time"] = "2017-08-17T00:00:00"
    dataset.to_netcdf(forcing)
    gages = tmp_path / "gages.csv"
    _gage_locations().assign(gage_name="Gage1", series_name="test_rainfall").to_csv(gages, index=False)

    probability = run_swmm_ensemble(forcing, model, gages, tmp_path / "results", ["J1"])

    assert probability["ensemble_members"].tolist() == [2]
    assert (tmp_path / "results" / "swmm_node_timeseries.csv").exists()
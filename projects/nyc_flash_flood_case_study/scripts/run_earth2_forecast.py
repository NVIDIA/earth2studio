from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from earth2studio.data import GFS
from earth2studio.io import NetCDF4Backend
from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.models.px import DiagnosticWrapper, FCN
from earth2studio.perturbation import SphericalGaussian
from earth2studio.run import ensemble


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Earth2Studio ensemble precipitation forcing for the NYC flood workflow.")
    parser.add_argument("--init-time", required=True, help="UTC forecast initialization time, e.g. 2021-09-01T00:00:00")
    parser.add_argument("--steps", type=int, default=4, help="Number of 6-hour forecast steps")
    parser.add_argument("--members", type=int, default=4, help="Number of perturbed initial-condition ensemble members")
    parser.add_argument("--batch-size", type=int, default=1, help="Ensemble members per GPU batch")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "forecasts" / "earth2_precip_ensemble.nc")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1 or args.members < 1 or args.batch_size < 1:
        raise ValueError("steps, members, and batch-size must all be positive.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prognostic_package = FCN.load_default_package()
    prognostic = FCN.load_model(prognostic_package)
    precipitation_package = PrecipitationAFNO.load_default_package()
    precipitation = PrecipitationAFNO.load_model(precipitation_package)
    coupled_model = DiagnosticWrapper(prognostic, precipitation)
    io = NetCDF4Backend(str(args.output), backend_kwargs={"mode": "w"})

    try:
        ensemble(
            [args.init_time],
            args.steps,
            args.members,
            coupled_model,
            GFS(),
            io,
            SphericalGaussian(noise_amplitude=0.15),
            batch_size=args.batch_size,
            output_coords={"variable": np.array(["tp"])},
        )
    finally:
        io.root.close()

    with Dataset(args.output, "r+") as forecast:
        forecast.setncattr("forecast_model", "FourCastNet + PrecipitationAFNO")
        forecast.setncattr("initial_condition_source", "NOAA GFS")
        forecast.setncattr("ensemble_method", "Spherical Gaussian initial-condition perturbations")
        forecast.setncattr("precipitation_interval_hours", 6)
        forecast.variables["tp"].setncattr("units", "m")
        forecast.variables["tp"].setncattr("cell_methods", "time: sum")

    print(f"Saved {args.members}-member 6-hour precipitation ensemble to {args.output}")


if __name__ == "__main__":
    main()
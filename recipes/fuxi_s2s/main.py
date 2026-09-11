# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run single-member or ensemble FuXi-S2S inference from public ARCO ERA5."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from loguru import logger

import earth2studio.run as run
from earth2studio.data import ARCO_ERA5, FuXiS2SERA5
from earth2studio.io import IOBackend, ZarrBackend
from earth2studio.models.px import FuXiS2S
from earth2studio.perturbation import Zero
from earth2studio.utils.type import CoordSystem

DEFAULT_VARIABLES = ("t2m", "z500", "msl", "tp")


@dataclass(frozen=True)
class ForecastConfig:
    """Validated command-line configuration for one forecast run."""

    issue_time: datetime
    nsteps: int
    members: int
    batch_size: int
    variables: tuple[str, ...] | None
    output: Path
    overwrite: bool
    cache: bool
    async_timeout: int
    device: str

    @property
    def workflow_time(self) -> datetime:
        """Return the latest complete UTC daily-mean label."""
        return self.issue_time - timedelta(days=1)


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _forecast_steps(value: str) -> int:
    """Parse a forecast length within FuXi-S2S's validated horizon."""
    parsed = _positive_int(value)
    if parsed > 42:
        raise argparse.ArgumentTypeError("must not exceed the 42-day horizon")
    return parsed


def _utc_midnight(value: str) -> datetime:
    """Parse an ISO timestamp and normalize it to naive UTC midnight."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    if parsed != parsed.replace(hour=0, minute=0, second=0, microsecond=0):
        raise argparse.ArgumentTypeError("must be aligned to 00:00 UTC")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> ForecastConfig:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--issue-time",
        type=_utc_midnight,
        default=_utc_midnight("2020-06-03T00:00:00Z"),
        help="strict forecast issue time at 00:00 UTC",
    )
    parser.add_argument(
        "--nsteps",
        type=_forecast_steps,
        default=14,
        help="number of daily forecast steps",
    )
    parser.add_argument(
        "--members",
        type=_positive_int,
        default=1,
        help="one trajectory or the requested ensemble size",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=1,
        help="ensemble members evaluated in one inference batch",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=list(DEFAULT_VARIABLES),
        help="variables to save, or the single value 'all'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/fuxi_s2s_forecast.zarr"),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output Zarr store",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable persistent caching of ARCO chunks",
    )
    parser.add_argument(
        "--async-timeout",
        type=_positive_int,
        default=1800,
        help="ARCO request timeout in seconds",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if args.batch_size > args.members:
        parser.error("--batch-size cannot exceed --members")
    if "all" in args.variables and args.variables != ["all"]:
        parser.error("'all' cannot be combined with named --variables")
    if len(args.variables) != len(set(args.variables)):
        parser.error("--variables must not contain duplicates")

    variables = None if args.variables == ["all"] else tuple(args.variables)
    return ForecastConfig(
        issue_time=args.issue_time,
        nsteps=args.nsteps,
        members=args.members,
        batch_size=args.batch_size,
        variables=variables,
        output=args.output,
        overwrite=args.overwrite,
        cache=not args.no_cache,
        async_timeout=args.async_timeout,
        device=args.device,
    )


def _output_coords(variables: tuple[str, ...] | None) -> CoordSystem:
    """Build the optional output-variable selection."""
    if variables is None:
        return OrderedDict()
    return OrderedDict({"variable": np.asarray(variables)})


def run_forecast(config: ForecastConfig) -> IOBackend:
    """Run FuXi-S2S with ARCO initial conditions using the selected mode."""
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FuXi-S2S inference requires an available CUDA GPU")
    if config.output.exists() and not config.overwrite:
        raise FileExistsError(
            f"Output {config.output} already exists; pass --overwrite to replace it"
        )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "FuXi-S2S issue time {}; latest daily mean {}; members {}; steps {}",
        config.issue_time,
        config.workflow_time,
        config.members,
        config.nsteps,
    )

    model = FuXiS2S.load_model(FuXiS2S.load_default_package())
    if config.variables is not None:
        model_variables = {
            str(variable) for variable in model.input_coords()["variable"]
        }
        unknown_variables = sorted(set(config.variables) - model_variables)
        if unknown_variables:
            raise ValueError(
                "Unknown FuXi-S2S output variables: " + ", ".join(unknown_variables)
            )

    data = FuXiS2SERA5(
        ARCO_ERA5(
            cache=config.cache,
            verbose=True,
            async_timeout=config.async_timeout,
        )
    )
    io = ZarrBackend(
        file_name=str(config.output),
        backend_kwargs={"overwrite": config.overwrite},
    )
    common = {
        "time": [config.workflow_time.isoformat()],
        "nsteps": config.nsteps,
        "prognostic": model,
        "data": data,
        "io": io,
        "output_coords": _output_coords(config.variables),
        "device": device,
    }

    if config.members == 1:
        logger.info(
            "Running one stochastic FuXi-S2S trajectory with the deterministic workflow"
        )
        result = run.deterministic(**common)
    else:
        logger.info("Running a {}-member stochastic ensemble", config.members)
        result = run.ensemble(
            nensemble=config.members,
            perturbation=Zero(),
            batch_size=config.batch_size,
            **common,
        )

    io.root.attrs.update(
        {
            "forecast_issue_time_utc": f"{config.issue_time.isoformat()}Z",
            "latest_complete_daily_mean_utc": (f"{config.workflow_time.isoformat()}Z"),
            "ensemble_members": config.members,
            "forecast_steps": config.nsteps,
        }
    )
    logger.success("Forecast written to {}", config.output)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line forecast recipe."""
    run_forecast(parse_args(argv))


if __name__ == "__main__":
    main()

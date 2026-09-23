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

"""Build hourly AI conditioning data for StormCast.

StormCast needs conditioning data for each hourly HRRR initialization time.
Direct GFS conditioning is limited to six-hour GFS cycles. This module instead
runs a global AI model from the enclosing GFS cycle, interpolates its output to
hourly steps, saves it to NetCDF, and exposes it through
``InferenceOutputSource``.

Two base models are currently implemented::

- ``sfno`` (default) provides all required variables, including surface
  pressure.
- ``fcn3`` derives missing surface pressure with
  ``DerivedSurfacePressure``.

``InterpModAFNO`` converts each model's six-hour steps into hourly output.
"""

from __future__ import annotations

import os
import uuid
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import xarray as xr  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from earth2studio.data import DataSource, InferenceOutputSource
    from earth2studio.io import IOBackend
    from earth2studio.models.px import DiagnosticWrapper, InterpModAFNO


class _ConditioningDataNotReady(Exception):
    """Signal that required GFS files are not yet available."""


# Pressure levels used to calculate surface pressure.
_SP_LEVELS: list[int] = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]


def build_conditioning_model(
    device: str | torch.device, model: str = "sfno"
) -> InterpModAFNO:
    """Build an hourly conditioning model from SFNO or FCN3.

    The returned ``InterpModAFNO`` wraps the selected global model and converts
    its six-hour forecast steps into hourly output.

    Parameters
    ----------
    device : str | torch.device
        Device for the conditioning model, such as ``"cuda:0"``.
    model : str, optional
        Base global model: ``"sfno"`` or ``"fcn3"``. The default is
        ``"sfno"``.

    Returns
    -------
    InterpModAFNO
        A prognostic model whose output covers StormCast's 26 conditioning variables at
        1 h resolution, ready to hand to ``earth2studio.run.deterministic``.

    Raises
    ------
    ValueError
        If ``model`` is not ``"sfno"`` or ``"fcn3"``.
    """
    from earth2studio.models.px import InterpModAFNO

    if model == "sfno":
        base_model = _build_sfno()
    elif model == "fcn3":
        base_model = _build_fcn3_with_sp()
    else:
        raise ValueError(
            f"unknown conditioning model {model!r}; expected 'sfno' or 'fcn3'"
        )

    interpolator = InterpModAFNO.from_pretrained()
    interpolator.px_model = base_model
    interpolator.to(device=device)
    return interpolator


def _build_sfno() -> Any:
    """SFNO base model: covers all 26 conditioning variables (incl ``sp``) natively."""
    from earth2studio.models.px import SFNO

    return SFNO.load_model(SFNO.load_default_package())


def _build_fcn3_with_sp() -> DiagnosticWrapper:
    """Load FCN3 and add its missing surface-pressure output."""
    from earth2studio.models.dx import DerivedSurfacePressure
    from earth2studio.models.px import FCN3, DiagnosticWrapper

    package = FCN3.load_default_package()
    fcn3 = FCN3.load_model(package)

    orography_path = package.resolve("orography.nc")
    with xr.open_dataset(orography_path) as dataset:
        surface_geopotential = torch.as_tensor(dataset["Z"][0].values)
    surface_geopotential_coords = OrderedDict(
        {dimension: fcn3.input_coords()[dimension] for dimension in ("lat", "lon")}
    )
    surface_pressure_model = DerivedSurfacePressure(
        p_levels=_SP_LEVELS,
        surface_geopotential=surface_geopotential,
        surface_geopotential_coords=surface_geopotential_coords,
    )
    return DiagnosticWrapper(px_model=fcn3, dx_model=surface_pressure_model)


def run_conditioning(
    model: InterpModAFNO,
    gfs: DataSource,
    start: datetime,
    num_hours: int,
    out_path: str,
    device: str | torch.device,
) -> InferenceOutputSource:
    """Run an hourly conditioning forecast and return it as a data source.

    The forecast is written to a temporary file and atomically moved into
    ``out_path`` on success, so a failed or not-ready cycle leaves the previous
    cycle's file intact.

    Parameters
    ----------
    model : InterpModAFNO
        Hourly conditioning model returned by
        :func:`build_conditioning_model`.
    gfs : DataSource
        GFS data used to initialize the model.
    start : datetime
        Six-hour-aligned GFS initialization time.
    num_hours : int
        Number of hourly forecast steps.
    out_path : str
        NetCDF output path.
    device : str | torch.device
        Device used to run the model.

    Returns
    -------
    InferenceOutputSource
        Data source backed by the completed NetCDF forecast.
    """
    import earth2studio.run as run
    from earth2studio.data import InferenceOutputSource
    from earth2studio.io import NetCDF4Backend

    # Write to a unique temporary file, then atomically replace the destination only on success.
    # If a newer GFS cycle turns out to be not-ready, run.deterministic raises before the replace,
    # so the in-use previous-cycle file is never destroyed. (NetCDF4Backend appends, so the temp
    # path must not pre-exist; a fresh uuid name guarantees that.)
    tmp_path = f"{out_path}.{uuid.uuid4().hex}.tmp"
    try:
        backend = NetCDF4Backend(tmp_path)
        try:
            try:
                run.deterministic(
                    [start.strftime("%Y-%m-%dT%H:%M:%S")],
                    num_hours,
                    model,
                    gfs,
                    # NetCDF4Backend.add_array's signature differs slightly from the IOBackend
                    # protocol in earth2studio, so mypy needs the cast; it is a valid backend.
                    cast("IOBackend", backend),
                    device=device,
                )
            except FileNotFoundError as exc:
                # A listed GFS cycle may still be uploading required files. Restrict
                # this signal to inference so unrelated file errors fail normally.
                raise _ConditioningDataNotReady(str(exc)) from exc
        finally:
            # Flush and close the file before replacing and reopening it.
            backend.close()
        # Atomic on POSIX: swaps the directory entry, so a previous source still reading the old
        # path keeps its open handle valid.
        os.replace(tmp_path, out_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return InferenceOutputSource(out_path)

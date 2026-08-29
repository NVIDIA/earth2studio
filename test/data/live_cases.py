# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Small real-provider cases for the live data-source contract pilot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

import earth2studio.data as e2data

LiveResult = xr.DataArray | pd.DataFrame


@dataclass(frozen=True)
class LiveCase:
    """One minimal call through a public data source to its real provider."""

    id: str
    source_type: type[Any]
    provider_group: str
    run: Callable[[], LiveResult]
    validate: Callable[[LiveResult], None]


def _run_gfs() -> xr.DataArray:
    source = e2data.GFS(cache=False)
    return source(datetime(2022, 12, 25), "t2m")


def _validate_gfs(result: LiveResult) -> None:
    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time", "variable", "lat", "lon")
    assert result.shape == (1, 1, 721, 1440)
    assert result.coords["variable"].item() == "t2m"
    assert np.isfinite(result.values).all()


def _run_ghcn_daily() -> pd.DataFrame:
    source = e2data.GHCNDaily(
        stations=["USW00013722"],
        time_tolerance=timedelta(0),
        cache=False,
        verbose=False,
    )
    return source(datetime(2023, 7, 4), ["t2m_max"])


def _validate_ghcn_daily(result: LiveResult) -> None:
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == e2data.GHCNDaily.SCHEMA.names
    assert set(result["station"]) == {"USW00013722"}
    assert set(result["variable"]) == {"t2m_max"}
    assert result["observation"].notna().all()


def _run_ufs_obs_conv() -> pd.DataFrame:
    source = e2data.UFSObsConv(
        time_tolerance=timedelta(hours=1),
        max_workers=2,
        cache=False,
        verbose=False,
        decode_workers=2,
    )
    return source(datetime(2024, 1, 1), ["t", "q"])


def _validate_ufs_obs_conv(result: LiveResult) -> None:
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert list(result.columns) == e2data.UFSObsConv.SCHEMA.names
    assert set(result["variable"]) == {"q", "t"}
    assert result["observation"].notna().any()


LIVE_CASES = (
    LiveCase(
        id="gfs",
        source_type=e2data.GFS,
        provider_group="noaa-object-store",
        run=_run_gfs,
        validate=_validate_gfs,
    ),
    LiveCase(
        id="ghcn_daily",
        source_type=e2data.GHCNDaily,
        provider_group="noaa-object-store",
        run=_run_ghcn_daily,
        validate=_validate_ghcn_daily,
    ),
    LiveCase(
        id="ufs_obs_conv",
        source_type=e2data.UFSObsConv,
        provider_group="noaa-ufs",
        run=_run_ufs_obs_conv,
        validate=_validate_ufs_obs_conv,
    ),
)

# Explicit classification of the public data-source surface. This inventory is
# intentionally broader than the three-case pilot. As families migrate,
# LIVE_CASES grows until it covers NETWORK_SOURCE_CLASSES completely.
NETWORK_SOURCE_CLASSES = frozenset(
    {
        e2data.ACE2ERA5Data,
        e2data.ARCO,
        e2data.CAMS_FX,
        e2data.CDS,
        e2data.CFS_FX,
        e2data.CFS_FX_Flux,
        e2data.CFS_Reforecast_FX,
        e2data.CFS_Reforecast_FX_Flux,
        e2data.CMIP6,
        e2data.CMIP6MultiRealm,
        e2data.DynamicalAIFS,
        e2data.DynamicalAIFS_ENS,
        e2data.DynamicalAIFS_FX,
        e2data.DynamicalAIFSENS_FX,
        e2data.DynamicalGEFS,
        e2data.DynamicalGEFS_FX,
        e2data.DynamicalGFS,
        e2data.DynamicalGFS_FX,
        e2data.DynamicalHRRR,
        e2data.DynamicalHRRR_FX,
        e2data.DynamicalICON_EU_FX,
        e2data.DynamicalIFS_ENS,
        e2data.DynamicalIFS_ENS_FX,
        e2data.DynamicalMRMS,
        e2data.EarthMoverBrightBandIFS,
        e2data.EarthMoverBrightBandIFS_FX,
        e2data.EarthMoverERA5,
        e2data.AIFS_ENS_FX,
        e2data.AIFS_FX,
        e2data.IFS,
        e2data.IFS_ENS,
        e2data.IFS_ENS_FX,
        e2data.IFS_FX,
        e2data.NomadsGDASObsConv,
        e2data.GEFS_FX,
        e2data.GEFS_FX_721x1440,
        e2data.GFS,
        e2data.GFS_FX,
        e2data.GHCNDaily,
        e2data.GHCNHourly,
        e2data.GOES,
        e2data.GOESGLM,
        e2data.GOESGLMGrid,
        e2data.HimawariAHI,
        e2data.HRRR,
        e2data.HRRR_FX,
        e2data.IBTrACS,
        e2data.IEM_ASOS,
        e2data.ISD,
        e2data.JPSS,
        e2data.JPSS_ATMS,
        e2data.JPSS_CRIS,
        e2data.LandSeaMask,
        e2data.MeteosatFCI,
        e2data.MetOpAMSUA,
        e2data.MetOpAVHRR,
        e2data.MetOpIASI,
        e2data.MetOpMHS,
        e2data.MRMS,
        e2data.NCAR_ERA5,
        e2data.NClimGridDaily,
        e2data.NNJAObsConv,
        e2data.NNJAObsSat,
        e2data.OPERA,
        e2data.PlanetaryComputerECMWFOpenDataIFS,
        e2data.PlanetaryComputerGOES,
        e2data.PlanetaryComputerMODISFire,
        e2data.PlanetaryComputerOISST,
        e2data.PlanetaryComputerSentinel3AOD,
        e2data.SamudrACEData,
        e2data.SamudrACEForcingData,
        e2data.SurfaceGeoPotential,
        e2data.UFSObsConv,
        e2data.UFSObsSat,
        e2data.WB2ERA5,
        e2data.WB2Climatology,
        e2data.WB2ERA5_32x64,
        e2data.WB2ERA5_121x240,
    }
)

NON_NETWORK_SOURCES = frozenset(
    {
        e2data.CBottle3D,
        e2data.Constant,
        e2data.Constant_FX,
        e2data.CosineSolarZenith,
        e2data.DataArrayDirectory,
        e2data.DataArrayFile,
        e2data.DataArrayPathList,
        e2data.DataSetFile,
        e2data.DataSource,
        e2data.ForecastSource,
        e2data.InferenceOutputSource,
        e2data.Random,
        e2data.RandomDataFrame,
        e2data.Random_FX,
        e2data.TimeWindow,
    }
)

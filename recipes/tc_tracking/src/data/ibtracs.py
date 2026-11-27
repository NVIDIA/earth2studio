# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download IBTrACS and read the best track of a storm.

IBTrACS (International Best Track Archive for Climate Stewardship) collects the
best tracks of all agencies. Agencies average the wind over different periods:
the US agencies (NHC, JTWC) report 1-minute sustained winds in every basin,
while the WMO columns hold 10-minute winds in most basins outside the Atlantic
and East Pacific. :func:`load_ibtracs_storm` therefore reads the ``USA_*``
columns, which are comparable across basins.

References
----------
- https://www.ncei.noaa.gov/products/international-best-track-archive
- Knapp et al. (2010), https://doi.org/10.1175/2009BAMS2755.1
- Gahtan et al. (2024), IBTrACS v04r01, https://doi.org/10.25921/82ty-9e16
"""

import os
import urllib.request

import pandas as pd
from loguru import logger

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)

KNOTS_TO_MS = 0.514444
HPA_TO_PA = 100.0
NAUTICAL_MILE_KM = 1.852

_QUADRANTS = ("NE", "SE", "SW", "NW")
_COLUMNS = [
    "SID",
    "SEASON",
    "NAME",
    "ISO_TIME",
    "NATURE",
    "TRACK_TYPE",
    "LAT",
    "LON",
    "USA_AGENCY",
    "USA_WIND",
    "USA_PRES",
    "USA_RMW",
    *(f"USA_R34_{quadrant}" for quadrant in _QUADRANTS),
]


def ensure_ibtracs(path: str) -> str:
    """Ensure the IBTrACS CSV exists at *path*, downloading it if necessary.

    The full IBTrACS v04r01 dataset (~300 MB) is downloaded from NCEI on
    first use and cached at the specified location for subsequent runs.

    Parameters
    ----------
    path : str
        Local file path where the IBTrACS CSV should reside.

    Returns
    -------
    str
        Absolute path to the IBTrACS CSV file.
    """
    path = os.path.abspath(path)
    if os.path.isfile(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info(
        f"IBTrACS file not found at {path} — downloading from NCEI (~300 MB)..."
    )
    # download to a temporary file, so that an interrupted download is not
    # mistaken for the dataset by the next run
    partial = f"{path}.part"
    urllib.request.urlretrieve(IBTRACS_URL, partial)  # noqa: S310
    os.replace(partial, path)
    logger.success(f"IBTrACS downloaded to {path}")
    return path


def load_ibtracs_storm(
    path: str, name: str, season: int, synoptic_only: bool = True
) -> pd.DataFrame:
    """Read the best track of one storm, in the units of the recipe's track files.

    Parameters
    ----------
    path : str
        IBTrACS CSV, downloaded if missing, see :func:`ensure_ibtracs`
    name : str
        Name of the storm, e.g. ``"melissa"``; case is ignored
    season : int
        Season of the storm, the year it formed in on the northern hemisphere
    synoptic_only : bool, optional
        Keep only 00, 06, 12 and 18 UTC, the times of the forecasts, dropping
        intermediate points such as landfalls, by default True

    Returns
    -------
    pd.DataFrame
        One row per time, with

        - ``time``, ``lat``, ``lon`` (degrees east, 0 to 360),
        - ``msl``: minimum central pressure in Pa,
        - ``wind_speed``: maximum 1-minute sustained wind in m/s,
        - ``rmw_km``: radius of maximum wind in km,
        - ``r17_km``: mean radius of 17.5 m/s wind over the quadrants having it,
        - ``nature``, ``agency`` and ``track_type``: storm type, the US agency
          and whether the track is final (``main``) or provisional

    Raises
    ------
    ValueError
        If no storm, or more than one, has this name in the season
    """
    # the second row holds the units
    table = pd.read_csv(
        ensure_ibtracs(path),
        usecols=_COLUMNS,
        skiprows=[1],
        dtype=str,
        keep_default_na=False,
    )
    storm = table[
        (table["NAME"].str.upper() == name.upper())
        & (table["SEASON"].astype(int) == season)
    ]
    sids = storm["SID"].unique()
    if len(sids) != 1:
        raise ValueError(
            f"expected one storm named {name} in season {season}, found "
            f"{len(sids)}: {sids.tolist()}"
        )

    def number(column: str) -> pd.Series:
        # missing values are blank
        return pd.to_numeric(storm[column].str.strip(), errors="coerce")

    radii = pd.concat([number(f"USA_R34_{q}") for q in _QUADRANTS], axis=1)
    track = pd.DataFrame(
        {
            "time": pd.to_datetime(storm["ISO_TIME"]),
            "lat": number("LAT"),
            "lon": number("LON") % 360.0,
            "msl": number("USA_PRES") * HPA_TO_PA,
            "wind_speed": number("USA_WIND") * KNOTS_TO_MS,
            "rmw_km": number("USA_RMW") * NAUTICAL_MILE_KM,
            "r17_km": radii.where(radii > 0).mean(axis=1) * NAUTICAL_MILE_KM,
            "nature": storm["NATURE"].str.strip(),
            "agency": storm["USA_AGENCY"].str.strip(),
            "track_type": storm["TRACK_TYPE"].str.strip(),
        }
    )
    if synoptic_only:
        on_the_hour = track["time"].dt.minute == 0
        track = track[on_the_hour & (track["time"].dt.hour % 6 == 0)]
    if set(track["track_type"]) != {"main"}:
        logger.warning(
            f"the best track of {name} ({season}) is still provisional "
            f"(track type {sorted(set(track['track_type']))})"
        )
    return track.reset_index(drop=True)

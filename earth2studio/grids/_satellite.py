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

"""Fixed grids of geostationary satellite imagers, labelled by scan angle."""

from __future__ import annotations

from typing import Literal

from earth2studio.grids.projected import ProjectedGrid


def goes_grid(satellite: str, scan_mode: str) -> ProjectedGrid:
    """GOES ABI fixed grid labelled by scan angle in radians, as the source returns.

    The ``geos`` projection gives scan angle times the perspective height, so a
    unit of that height makes the projected coordinates scan angles.
    """
    from earth2studio.data.goes import GOES

    height = GOES.PERSPECTIVE_POINT_HEIGHT
    if scan_mode == "F":
        y, x = GOES.FULL_DISK_YX
    else:
        y, x = GOES.CONTINENTAL_US_YX[satellite]
    return ProjectedGrid(
        y=y,
        x=x,
        coordinate_reference_system=(
            f"+proj=geos +h={height} "
            f"+lon_0={GOES.LONGITUDE_OF_PROJECTION_ORIGIN[satellite]} +sweep=x "
            f"+a={GOES.SEMI_MAJOR_AXIS} +b={GOES.SEMI_MINOR_AXIS} "
            f"+to_meter={height} +no_defs +type=crs"
        ),
    )


def fci_grid(resolution: Literal["2km", "1km", "500m"]) -> ProjectedGrid:
    """MTG FCI full-disk grid labelled by scan angle in radians, as the source returns.

    FCI scans with ``sweep=y``, and its x scan angle increases westward.
    """
    from earth2studio.data import meteosat_fci
    from earth2studio.data.meteosat_fci import MeteosatFCI

    height = meteosat_fci.PERSPECTIVE_POINT_HEIGHT
    return ProjectedGrid(
        y=MeteosatFCI.FCI_Y[resolution],
        x=MeteosatFCI.FCI_X[resolution],
        coordinate_reference_system=(
            f"+proj=geos +h={height} +lon_0={meteosat_fci.SUB_SATELLITE_LON} "
            f"+sweep=y +a={meteosat_fci.SEMI_MAJOR_AXIS} "
            f"+b={meteosat_fci.SEMI_MINOR_AXIS} +to_meter={height} +axis=wnu "
            "+no_defs +type=crs"
        ),
    )

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

from collections.abc import Callable

import numpy as np

from .base import LexiconType

_GLOBAL = "cams-global-atmospheric-composition-forecasts"


class CAMSLexicon(metaclass=LexiconType):
    """Copernicus Atmosphere Monitoring Service Lexicon

    CAMS specified ``<dataset>::<api_variable>::<netcdf_key>::<level>``

    The API variable name (used in the cdsapi request) differs from the NetCDF
    key (used to index the downloaded file). Both are stored in the VOCAB.

    Note
    ----
    Additional resources:
    https://ads.atmosphere.copernicus.eu/datasets/cams-global-atmospheric-composition-forecasts
    """

    VOCAB = {
        "aod550": f"{_GLOBAL}::total_aerosol_optical_depth_550nm::aod550::",
        "duaod550": f"{_GLOBAL}::dust_aerosol_optical_depth_550nm::duaod550::",
        "omaod550": f"{_GLOBAL}::organic_matter_aerosol_optical_depth_550nm::omaod550::",
        "bcaod550": f"{_GLOBAL}::black_carbon_aerosol_optical_depth_550nm::bcaod550::",
        "ssaod550": f"{_GLOBAL}::sea_salt_aerosol_optical_depth_550nm::ssaod550::",
        "suaod550": f"{_GLOBAL}::sulphate_aerosol_optical_depth_550nm::suaod550::",
        "tcco": f"{_GLOBAL}::total_column_carbon_monoxide::tcco::",
        "tcno2": f"{_GLOBAL}::total_column_nitrogen_dioxide::tcno2::",
        "tco3": f"{_GLOBAL}::total_column_ozone::tco3::",
        "tcso2": f"{_GLOBAL}::total_column_sulphur_dioxide::tcso2::",
        "gtco3": f"{_GLOBAL}::gems_total_column_ozone::gtco3::",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return name in CAMS vocabulary."""
        cams_key = cls.VOCAB[val]

        def mod(x: np.ndarray) -> np.ndarray:
            return x

        return cams_key, mod

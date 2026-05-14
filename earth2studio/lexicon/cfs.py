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

# Standard pressure levels shared between CFS pgbf and other AI weather model
# input archives.  Selected to match GFSLexicon / E2STUDIO_VOCAB pressure-level
# coverage.
_PGBF_PRESSURE_LEVELS = (
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
)

# Mapping from the leading character of an E2S pressure-level variable name
# (e.g. "u500") to the CFS grib parameter id used in the pgbf product.
_PGBF_PARAM = {
    "u": "UGRD",
    "v": "VGRD",
    "z": "HGT",
    "t": "TMP",
    "q": "SPFH",
    "r": "RH",
}


def _build_pgbf_vocab() -> dict[str, str]:
    """Return the full pgbf vocabulary mapping E2S names to CFS .idx keys.

    The CFS ``pgbf`` product is a pressure-level archive and does not publish
    most near-surface fields (``t2m``, ``u10m``, ``v10m``, ``sp``, ``q2m``,
    ``tcwv``); those live on the T126 Gaussian ``flxf`` grid -- use
    :class:`CFSFluxLexicon` and :class:`~earth2studio.data.CFS_FX_Flux`
    for them. Variables that *are* available at the surface in pgbf
    (dewpoint and relative humidity at 2 m, MSL pressure) are exposed here.
    """
    vocab: dict[str, str] = {
        # Surface / 2 m fields that pgbf does carry
        "d2m": "pgbf::DPT::2 m above ground",
        "r2m": "pgbf::RH::2 m above ground",
        "msl": "pgbf::PRMSL::mean sea level",
    }
    for letter, param in _PGBF_PARAM.items():
        for level in _PGBF_PRESSURE_LEVELS:
            vocab[f"{letter}{level}"] = f"pgbf::{param}::{level} mb"
    return vocab


class CFSLexicon(metaclass=LexiconType):
    """Climate Forecast System v2 (CFSv2) pressure-level forecast lexicon.

    Maps Earth2Studio variable identifiers to grib2 `.idx` records of the
    1 degree CFS `pgbf` product. Keys use the `<product>::<param>::<level>`
    convention: `<product>` is the file prefix (always ``pgbf`` here),
    `<param>` is the grib short name, and `<level>` is the grib level
    description.

    Note
    ----
    CFSv2 pgbf .idx files store the eastward/northward wind components
    ``UGRD`` and ``VGRD`` as adjacent records sharing the same byte offset
    (decimal-suffixed record numbers in the `.idx` file, e.g. ``7.1`` and
    ``7.2``). The CFS data source resolves this by reading the full vector
    grib message and selecting the requested component by short name; the
    lexicon entries treat ``UGRD`` and ``VGRD`` as independent variables.

    Note
    ----
    Additional resources:

    - https://www.nco.ncep.noaa.gov/pmb/products/cfs/
    - https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-3.shtml
    """

    VOCAB = _build_pgbf_vocab()

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return the CFS index key and modifier function for a variable.

        Parameters
        ----------
        val : str
            Earth2Studio variable identifier.

        Returns
        -------
        tuple[str, Callable]
            Tuple of `(grib_key, modifier)` where `grib_key` is the
            `<product>::<param>::<level>` substring used to look up the
            record in the grib `.idx` file, and `modifier` is applied to
            the raw grib values before they are returned (e.g. unit
            conversion from geopotential height to geopotential).
        """
        cfs_key = cls.VOCAB[val]
        if cfs_key.split("::")[1] == "HGT":

            def mod(x: np.ndarray) -> np.ndarray:
                """Convert geopotential height (m) to geopotential (m^2 s^-2)."""
                return x * 9.81

        else:

            def mod(x: np.ndarray) -> np.ndarray:
                """Identity modifier."""
                return x

        return cfs_key, mod


class CFSFluxLexicon(metaclass=LexiconType):
    """Climate Forecast System v2 (CFSv2) surface flux forecast lexicon.

    Maps Earth2Studio variable identifiers to grib2 `.idx` records of the
    CFS `flxf` product. The flxf product is published on a T126 Gaussian
    grid (190 x 384). Keys use the `<product>::<param>::<level>`
    convention; `<product>` is always ``flxf`` for entries in this
    lexicon.

    Note
    ----
    Additional resources:

    - https://www.nco.ncep.noaa.gov/pmb/products/cfs/
    - https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-2-0-3.shtml
    """

    VOCAB = {
        # Near-surface state variables
        "t2m": "flxf::TMP::2 m above ground",
        "q2m": "flxf::SPFH::2 m above ground",
        "u10m": "flxf::UGRD::10 m above ground",
        "v10m": "flxf::VGRD::10 m above ground",
        # Surface scalars
        "sp": "flxf::PRES::surface",
        "skt": "flxf::TMP::surface",
        "tpf": "flxf::PRATE::surface",
        "sde": "flxf::SNOD::surface",
        # Column / cloud-cover summaries
        "tcwv": "flxf::PWAT::entire atmosphere (considered as a single layer)",
        "tcc": "flxf::TCDC::entire atmosphere (considered as a single layer)",
        "hcc": "flxf::TCDC::high cloud layer",
        "mcc": "flxf::TCDC::middle cloud layer",
        "lcc": "flxf::TCDC::low cloud layer",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return the CFS index key and modifier function for a flxf variable.

        Parameters
        ----------
        val : str
            Earth2Studio variable identifier.

        Returns
        -------
        tuple[str, Callable]
            `(grib_key, modifier)` pair. `modifier` is the identity for
            every entry in this lexicon since all surface flux fields are
            already in physical units.
        """
        cfs_key = cls.VOCAB[val]

        def mod(x: np.ndarray) -> np.ndarray:
            """Identity modifier."""
            return x

        return cfs_key, mod

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Any

from earth2studio.lexicon.base import LexiconType


class JPSSLexicon(metaclass=LexiconType):
    """JPSS VIIRS lexicon for mapping standardized variable names to VIIRS product codes.

    This lexicon maps standardized variable names to VIIRS SDR (Level 1) and EDR (Level 2)
    product codes and HDF5 datasets used in AWS. It includes both raw radiance data and
    derived environmental products.

    Notes
    -----
    SDR Products (Level 1 - Calibrated Radiances):
    - I-bands (375m): viirs01i-viirs05i (high spatial resolution imagery)
    - M-bands (750m): viirs01m-viirs16m (moderate resolution multispectral)
    - Band numbering follows VIIRS instrument naming with zero-padding (I1→viirs01i, M15→viirs15m)
    - Applications: Ocean color, vegetation monitoring, cloud detection, fire detection, atmospheric profiling

    EDR Products (Level 2 - Derived Environmental Variables):
    - Surface/Land: lst (land surface temperature), salb (surface albedo), snc (snow cover)
    - Active Fire: afire (fire detection), fmask (fire confidence mask)
    - Cloud Suite: cmask (cloud mask), cphase (cloud phase), cth (cloud top height), etc.
    - Atmospheric: aod (aerosol optical depth), vash (volcanic ash detection)
    - Resolution: 375m-750m depending on input bands, global coverage every ~100 minutes

    References
    ----------
    - NOAA STAR. (n.d.). VIIRS instrument specifications and band characteristics.
      Retrieved September 29, 2025, from https://www.star.nesdis.noaa.gov/jpss/VIIRS.php
    - NOAA STAR. (n.d.). VIIRS Sensor Data Record (SDR) Algorithm Theoretical Basis
      Document. Retrieved September 29, 2025, from
      https://www.star.nesdis.noaa.gov/jpss/documents/ATBD/D0001-M01-S01-003_JPSS_ATBD_VIIRS-SDR_E.pdf
    """

    # Mapping of VIIRS band numbers to product codes and modifiers
    # Format: "viirs{band_number}{band_type}": ("product_type", "folder", "dataset", modifier_function)
    VOCAB: dict[str, tuple[str, str, str, Callable[[Any], Any]]] = {
        # === SDR Level 1 Products: Calibrated Radiances ===
        # I-bands (375m resolution) - High spatial resolution imagery bands
        # Used for: Land/water discrimination, cloud detection, fire detection
        "viirs01i": (
            "I",
            "VIIRS-I1-SDR",
            "Radiance",
            lambda x: x,
        ),  # I1: 0.64 μm (Red) - Land/vegetation, cloud optical depth
        "viirs02i": (
            "I",
            "VIIRS-I2-SDR",
            "Radiance",
            lambda x: x,
        ),  # I2: 0.864 μm (Near-IR) - Vegetation index, cloud particle size
        "viirs03i": (
            "I",
            "VIIRS-I3-SDR",
            "Radiance",
            lambda x: x,
        ),  # I3: 1.58 μm (SWIR) - Snow/ice discrimination, fire detection
        "viirs04i": (
            "I",
            "VIIRS-I4-SDR",
            "Radiance",
            lambda x: x,
        ),  # I4: 3.74 μm (MWIR) - Fire detection, cloud phase discrimination
        "viirs05i": (
            "I",
            "VIIRS-I5-SDR",
            "Radiance",
            lambda x: x,
        ),  # I5: 11.45 μm (LWIR) - Sea surface temp, land surface temp, cloud top temp
        # M-bands (750m resolution) - Moderate resolution multispectral bands
        # Reflective Solar Bands (RSB): Solar illumination dependent, daytime only
        "viirs01m": (
            "M",
            "VIIRS-M1-SDR",
            "Radiance",
            lambda x: x,
        ),  # M1: 0.412 μm (Violet) - Ocean color, aerosol over ocean
        "viirs02m": (
            "M",
            "VIIRS-M2-SDR",
            "Radiance",
            lambda x: x,
        ),  # M2: 0.445 μm (Blue) - Ocean color, atmospheric correction
        "viirs03m": (
            "M",
            "VIIRS-M3-SDR",
            "Radiance",
            lambda x: x,
        ),  # M3: 0.488 μm (Blue-Green) - Ocean color, true color imagery
        "viirs04m": (
            "M",
            "VIIRS-M4-SDR",
            "Radiance",
            lambda x: x,
        ),  # M4: 0.555 μm (Green) - Ocean color, vegetation monitoring
        "viirs05m": (
            "M",
            "VIIRS-M5-SDR",
            "Radiance",
            lambda x: x,
        ),  # M5: 0.672 μm (Red) - Ocean color, vegetation health (chlorophyll absorption)
        "viirs06m": (
            "M",
            "VIIRS-M6-SDR",
            "Radiance",
            lambda x: x,
        ),  # M6: 0.746 μm (Red Edge) - Vegetation stress, atmospheric correction
        "viirs07m": (
            "M",
            "VIIRS-M7-SDR",
            "Radiance",
            lambda x: x,
        ),  # M7: 0.865 μm (Near-IR) - Vegetation index, land/water boundaries
        "viirs08m": (
            "M",
            "VIIRS-M8-SDR",
            "Radiance",
            lambda x: x,
        ),  # M8: 1.24 μm (Near-IR) - Cloud particle size, cirrus detection
        "viirs09m": (
            "M",
            "VIIRS-M9-SDR",
            "Radiance",
            lambda x: x,
        ),  # M9: 1.378 μm (Cirrus) - Thin cirrus detection over land/ocean
        "viirs10m": (
            "M",
            "VIIRS-M10-SDR",
            "Radiance",
            lambda x: x,
        ),  # M10: 1.61 μm (SWIR) - Snow/cloud discrimination, fire detection
        "viirs11m": (
            "M",
            "VIIRS-M11-SDR",
            "Radiance",
            lambda x: x,
        ),  # M11: 2.25 μm (SWIR) - Cloud effective radius, drought monitoring
        # Thermal Emissive Bands (TEB): Thermal emission, day/night capable
        "viirs12m": (
            "M",
            "VIIRS-M12-SDR",
            "Radiance",
            lambda x: x,
        ),  # M12: 3.7 μm (MWIR) - Sea surface temp, fire detection, cloud properties
        "viirs13m": (
            "M",
            "VIIRS-M13-SDR",
            "Radiance",
            lambda x: x,
        ),  # M13: 4.05 μm (MWIR) - Volcanic ash, fire detection, cloud top temp
        "viirs14m": (
            "M",
            "VIIRS-M14-SDR",
            "Radiance",
            lambda x: x,
        ),  # M14: 8.55 μm (LWIR) - Cloud top properties, atmospheric profile
        "viirs15m": (
            "M",
            "VIIRS-M15-SDR",
            "Radiance",
            lambda x: x,
        ),  # M15: 10.76 μm (LWIR) - Sea/land surface temp, cloud properties
        "viirs16m": (
            "M",
            "VIIRS-M16-SDR",
            "Radiance",
            lambda x: x,
        ),  # M16: 12.01 μm (LWIR) - Cloud properties, atmospheric water vapor
        # === EDR Level 2 Products: Environmental Data Records ===
        # Surface/Land Products (JPSS RR - Rapid Refresh algorithms)
        # Resolution: 750m at nadir, accuracy: ±2-3K for LST, ±0.02 for albedo
        "lst": (
            "L2",
            "JPSSRR_LST",
            "LST",
            lambda x: x,
        ),  # Land Surface Temperature from split-window algorithm (M15/M16)
        "salb": (
            "L2",
            "JPSSRR_SurfAlb",
            "SurfaceAlbedo",
            lambda x: x,
        ),  # Surface Albedo from BRDF model using visible/NIR bands
        "snc": (
            "L2",
            "JPSSRR_SnowCover",
            "SnowCover",
            lambda x: x,
        ),  # Snow Cover fraction using NDSI from M3/M10 bands
        # Active Fire Products (Enterprise algorithm)
        # Resolution: 375m (I-band), detection sensitivity: ~4K temperature anomaly
        "afire": (
            "L2",
            "VIIRS_EFIRE_VIIRSI_EDR",
            "Fire",
            lambda x: x,
        ),  # Active Fire Detection using contextual algorithm (I4/I5 thermal anomaly)
        "fmask": (
            "L2",
            "VIIRS_EFIRE_VIIRSI_EDR",
            "FireMask",
            lambda x: x,
        ),  # Fire Mask with confidence levels (low/nominal/high/water/cloud)
        # Cloud Products (JRR - Joint Radio and Radar algorithms)
        # Resolution: 750m, temporal: every orbit (~100 minutes global coverage)
        "cmask": (
            "L2",
            "VIIRS-JRR-CloudMask",
            "CloudMask",
            lambda x: x,
        ),  # Cloud Mask using multispectral tests (clear/probably_clear/probably_cloudy/cloudy)
        "cphase": (
            "L2",
            "VIIRS-JRR-CloudPhase",
            "CloudPhase",
            lambda x: x,
        ),  # Cloud Phase from IR split-window (ice/water/mixed/unknown)
        "cth": (
            "L2",
            "VIIRS-JRR-CloudHeight",
            "CldTopHght",
            lambda x: x,
        ),  # Cloud Top Height from CO2 slicing method using M13/M16
        "cbh": (
            "L2",
            "VIIRS-JRR-CloudBase",
            "CldBaseHght",
            lambda x: x,
        ),  # Cloud Base Height from atmospheric profiles and adiabatic assumptions
        "cdcomp": (
            "L2",
            "VIIRS-JRR-CloudDCOMP",
            "CloudMicroVisOD",
            lambda x: x,
        ),  # Cloud Optical Depth from visible reflectance (day only)
        "cncomp": (
            "L2",
            "VIIRS-JRR-CloudNCOMP",
            "CloudMicroVisOD",
            lambda x: x,
        ),  # Cloud Optical Properties from thermal IR (night/day)
        "ccl": (
            "L2",
            "VIIRS-JRR-CloudCoverLayers",
            "Total_Cloud_Fraction",
            lambda x: x,
        ),  # Cloud Cover by pressure layers (low/mid/high)
        # Atmospheric Products
        # Resolution: 750m for aerosol, volcanic ash detection
        "aod": (
            "L2",
            "VIIRS-JRR-ADP",
            "AerosolOpticalThickness",
            lambda x: x,
        ),  # Aerosol Optical Depth at 550nm using Dark Target/Deep Blue algorithms
        "vash": (
            "L2",
            "VIIRS-JRR-VolcanicAsh",
            "VolcanicAsh",
            lambda x: x,
        ),  # Volcanic Ash detection using BTD algorithm (M14-M15/M15-M16)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, str, str, Callable[[Any], Any]]:
        """Get VIIRS product details for a standardized variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (e.g., 'viirs01i', 'viirs05m', 'lst', 'aod')

        Returns
        -------
        tuple[str, str, str, Callable]
            Tuple containing:
            - Product type ("I", "M", or "L2")
            - Folder name for S3 path
            - Dataset name within HDF5 file
            - Modifier function for data transformation
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in VIIRS lexicon")
        return cls.VOCAB[val]

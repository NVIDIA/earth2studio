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
    product codes and HDF5 datasets used in AWS. It follows the same pattern as the
    GOES lexicon for consistency. For more information on VIIRS bands, see:
    https://ncc.nesdis.noaa.gov/VIIRS/VIIRSChannelWavelengths.php

    Parameters
    ----------
    val : str
        Standardized variable name (e.g., 'viirs1i', 'viirs2i', 'viirs5m', 'viirs15m', 'viirs_lst', 'viirs_cloud_mask')

    Notes
    -----
    SDR Products (Level 1):
    - I-bands (375 m): viirs1i-viirs5i (higher spatial resolution imagery bands)  
    - M-bands (750 m): viirs1m-viirs16m (moderate resolution multi-spectral bands)
    - Band numbering follows VIIRS instrument naming (I1-I5, M1-M16)
    - Examples: viirs1i = I1 red (0.64 μm), viirs2i = I2 near-IR (0.86 μm), viirs15m = M15 thermal (10.76 μm)
    
    EDR Products (Level 2):
    - Environmental Data Records at swath resolution
    - Examples: viirs_lst (Land Surface Temperature), viirs_cloud_mask (Cloud Mask), viirs_active_fire (Active Fire)
    """

    # Mapping of VIIRS band numbers to product codes and modifiers
    # Format: "viirs{band_number}{band_type}": ("product_code/dataset", modifier_function)
    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # I-bands (375m resolution) - High spatial resolution
        "viirs1i": ("SVI01/Radiance", lambda x: x),   # I1: 0.64 μm (Red)
        "viirs2i": ("SVI02/Radiance", lambda x: x),   # I2: 0.864 μm (Near-IR)
        "viirs3i": ("SVI03/Radiance", lambda x: x),   # I3: 1.58 μm (SWIR)
        "viirs4i": ("SVI04/Radiance", lambda x: x),   # I4: 3.74 μm (MWIR)
        "viirs5i": ("SVI05/Radiance", lambda x: x),   # I5: 11.45 μm (LWIR/Thermal)
        
        # M-bands (750m resolution) - Reflective Solar Bands
        "viirs1m": ("SVM01/Radiance", lambda x: x),   # M1: 0.412 μm (Violet)
        "viirs2m": ("SVM02/Radiance", lambda x: x),   # M2: 0.445 μm (Blue)
        "viirs3m": ("SVM03/Radiance", lambda x: x),   # M3: 0.488 μm (Blue-Green)
        "viirs4m": ("SVM04/Radiance", lambda x: x),   # M4: 0.555 μm (Green)
        "viirs5m": ("SVM05/Radiance", lambda x: x),   # M5: 0.672 μm (Red)
        "viirs6m": ("SVM06/Radiance", lambda x: x),   # M6: 0.746 μm (Red Edge)
        "viirs7m": ("SVM07/Radiance", lambda x: x),   # M7: 0.865 μm (Near-IR)
        "viirs8m": ("SVM08/Radiance", lambda x: x),   # M8: 1.24 μm (Near-IR)
        "viirs9m": ("SVM09/Radiance", lambda x: x),   # M9: 1.378 μm (Cirrus)
        "viirs10m": ("SVM10/Radiance", lambda x: x),  # M10: 1.61 μm (SWIR)
        "viirs11m": ("SVM11/Radiance", lambda x: x),  # M11: 2.25 μm (SWIR)
        
        # M-bands - Thermal Emissive Bands
        "viirs12m": ("SVM12/Radiance", lambda x: x),  # M12: 3.7 μm (MWIR)
        "viirs13m": ("SVM13/Radiance", lambda x: x),  # M13: 4.05 μm (MWIR)
        "viirs14m": ("SVM14/Radiance", lambda x: x),  # M14: 8.55 μm (LWIR)
        "viirs15m": ("SVM15/Radiance", lambda x: x),  # M15: 10.76 μm (LWIR)
        "viirs16m": ("SVM16/Radiance", lambda x: x),  # M16: 12.01 μm (LWIR)
        
        # L2 EDR Products - Enterprise/JRR Products  
        "viirs_lst": ("JPSSRR_LST/LST", lambda x: x),  # Land Surface Temperature
        "viirs_surf_alb": ("JPSSRR_SurfAlb/SurfaceAlbedo", lambda x: x),  # Surface Albedo
        "viirs_snow_cover": ("JPSSRR_SnowCover/SnowCover", lambda x: x),  # Snow Cover
        
        # L2 EDR Products - Surface Reflectance (multiple bands available)
        "viirs_surf_refl_m1": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M1", lambda x: x),  # Surface Reflectance M1
        "viirs_surf_refl_m2": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M2", lambda x: x),  # Surface Reflectance M2  
        "viirs_surf_refl_m3": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M3", lambda x: x),  # Surface Reflectance M3
        "viirs_surf_refl_m4": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M4", lambda x: x),  # Surface Reflectance M4
        "viirs_surf_refl_m5": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M5", lambda x: x),  # Surface Reflectance M5
        "viirs_surf_refl_m7": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M7", lambda x: x),  # Surface Reflectance M7
        "viirs_surf_refl_m8": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M8", lambda x: x),  # Surface Reflectance M8
        "viirs_surf_refl_m10": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M10", lambda x: x),  # Surface Reflectance M10
        "viirs_surf_refl_m11": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_M11", lambda x: x),  # Surface Reflectance M11
        "viirs_surf_refl_i1": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_I1", lambda x: x),  # Surface Reflectance I1
        "viirs_surf_refl_i2": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_I2", lambda x: x),  # Surface Reflectance I2
        "viirs_surf_refl_i3": ("VIIRS_SurfaceReflectance_EDR/SurfRefl_I3", lambda x: x),  # Surface Reflectance I3
        
        # L2 EDR Products - Active Fire
        "viirs_active_fire": ("VIIRS_EFIRE_VIIRSI_EDR/Fire", lambda x: x),  # Active Fire Detection
        "viirs_fire_mask": ("VIIRS_EFIRE_VIIRSI_EDR/FireMask", lambda x: x),  # Fire Mask
        
        # L2 EDR Products - Cloud Suite
        "viirs_cloud_mask": ("VIIRS-JRR-CloudMask/CloudMask", lambda x: x),  # Cloud Mask
        "viirs_cloud_phase": ("VIIRS-JRR-CloudPhase/CloudPhase", lambda x: x),  # Cloud Phase
        "viirs_cloud_height": ("VIIRS-JRR-CloudHeight/CldTopHght", lambda x: x),  # Cloud Top Height
        "viirs_cloud_base": ("VIIRS-JRR-CloudBase/CldBaseHght", lambda x: x),  # Cloud Base Height
        "viirs_cloud_dcomp": ("VIIRS-JRR-CloudDCOMP/CloudMicroVisOD", lambda x: x),  # Cloud Optical Properties (Day)
        "viirs_cloud_ncomp": ("VIIRS-JRR-CloudNCOMP/CloudMicroVisOD", lambda x: x),  # Cloud Optical Properties (Night)
        "viirs_cloud_cover": ("VIIRS-JRR-CloudCoverLayers/Total_Cloud_Fraction", lambda x: x),  # Cloud Cover Layers
        
        # L2 EDR Products - Aerosol and Atmospheric
        "viirs_aerosol": ("VIIRS-JRR-ADP/AerosolOpticalThickness", lambda x: x),  # Aerosol Detection Product
        "viirs_volcanic_ash": ("VIIRS-JRR-VolcanicAsh/VolcanicAsh", lambda x: x),  # Volcanic Ash Detection
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Get VIIRS SDR product identifier and modifier for a standardized variable name.

        Parameters
        ----------
        val : str
            Standardized variable name (e.g., 'viirs1i', 'viirs2i', 'viirs5m', 'viirs15m')

        Returns
        -------
        tuple[str, Callable]
            Tuple containing:
            - VIIRS product identifier (e.g., 'SVI01/Radiance', 'SVM05/Radiance')
            - Modifier function for data transformation
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in VIIRS lexicon")
        return cls.VOCAB[val]



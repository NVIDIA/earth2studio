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
# Spectral axis and radiance unit conversions for hyperspectral IR sounders.
#
# IASI and CrIS are archived as radiance and must be converted to brightness
# temperature before use (via earth2studio.data.utils.radiance_to_bt; pure
# Planck inversion with no band correction, matching the convention used by
# the MetOp IASI and JPSS CrIS granule sources). AIRS is already in kelvin in
# the NNJA aggregate BUFR files.

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

# IASI: uniform spectral grid, EUMETSAT L1C. Verified against GSI diagnostics.
IASI_FIRST_WAVENUMBER_CM = 645.0
IASI_SPACING_CM = 0.25
IASI_CHANNELS = 8461

# CrIS FSR: three bands (start wavenumber cm⁻¹, first channel, last channel)
# numbered contiguously on the CRTM/GSI sensor_chan grid; the same grid as
# jpss_cris.py builds from its per-band science channel counts. Band ranges
# match the NFQF/NCQF per-band quality struct in the crisf4 BUFR (0-33-077 /
# 0-33-076).
CRIS_SPACING_CM = 0.625
CRIS_BANDS = (
    (650.0, 1, 713),  # long wave
    (1210.0, 714, 1578),  # mid wave
    (2155.0, 1579, 2211),  # short wave
)

# AIRS: log10 of the centre wavenumber in m⁻¹ (LOGRCW) for each channel of the
# NCEP 281-channel subset, as encoded in the NNJA airs_disc_final aggregates.
# fmt: off
AIRS_LOG10_WAVENUMBER_M = {
    1: 4.81265938, 6: 4.8134569, 7: 4.81361633, 10: 4.81409568,
    11: 4.8142556, 15: 4.81489654, 16: 4.81505682, 17: 4.81521699,
    20: 4.81569859, 21: 4.81585923, 22: 4.8160198, 24: 4.81634141,
    27: 4.81682439, 28: 4.81698594, 30: 4.81730817, 36: 4.81827737,
    39: 4.81876215, 40: 4.81892431, 42: 4.8192484, 51: 4.82071037,
    52: 4.82087308, 54: 4.82119833, 55: 4.82136154, 56: 4.82152468,
    59: 4.82201436, 62: 4.82250412, 63: 4.82266748, 68: 4.82348597,
    69: 4.82365027, 71: 4.82397801, 72: 4.82414213, 73: 4.82430614,
    74: 4.82447081, 75: 4.8246347, 76: 4.8247992, 77: 4.82496368,
    78: 4.82512806, 79: 4.82529301, 80: 4.82545793, 82: 4.82578693,
    83: 4.82595226, 84: 4.82611694, 86: 4.82644737, 92: 4.82743905,
    93: 4.82760446, 98: 4.82843366, 99: 4.82859933, 101: 4.82893173,
    104: 4.82943052, 105: 4.82959706, 108: 4.83009634, 110: 4.83043018,
    111: 4.83059701, 113: 4.83093047, 116: 4.83143205, 117: 4.83159915,
    123: 4.83260406, 124: 4.83277196, 128: 4.83344357, 129: 4.83361177,
    138: 4.83852862, 139: 4.83869928, 144: 4.83955284, 145: 4.83972372,
    150: 4.84057967, 151: 4.8407514, 156: 4.84160908, 157: 4.84178105,
    159: 4.84212537, 162: 4.84264169, 165: 4.84315861, 168: 4.84367618,
    169: 4.84384919, 170: 4.84402154, 172: 4.84436719, 173: 4.84454054,
    174: 4.84471382, 175: 4.84488642, 177: 4.84523328, 179: 4.84557983,
    180: 4.84575333, 182: 4.84610071, 185: 4.84662222, 186: 4.84679591,
    190: 4.84749245, 192: 4.84784153, 198: 4.84888898, 201: 4.8494139,
    204: 4.84993939, 207: 4.85046611, 210: 4.85099278, 215: 4.85187266,
    216: 4.85204855, 221: 4.85293053, 226: 4.85381436, 227: 4.85399129,
    232: 4.85487722, 252: 4.85843975, 253: 4.8586184, 256: 4.85915571,
    257: 4.85933528, 261: 4.86005277, 262: 4.86023196, 267: 4.86113158,
    272: 4.86203233, 295: 4.86578482, 299: 4.86651239, 300: 4.86669485,
    305: 4.86760655, 310: 4.86851985, 321: 4.87053674, 325: 4.8712722,
    333: 4.87274696, 338: 4.87367105, 355: 4.87682785, 362: 4.87813387,
    375: 4.88056952, 453: 4.89936684, 475: 4.90368619, 484: 4.9054645,
    497: 4.90804514, 528: 4.91425533, 587: 4.92629769, 672: 4.94016175,
    787: 4.96251425, 791: 4.96319594, 843: 4.97216025, 870: 4.97689263,
    914: 4.98472079, 950: 4.99083947, 1003: 5.00060064, 1012: 5.0022797,
    1019: 5.00358976, 1024: 5.00452771, 1030: 5.00565618, 1038: 5.00716506,
    1048: 5.0090585, 1069: 5.01305978, 1079: 5.01497769, 1082: 5.01555463,
    1083: 5.01574691, 1088: 5.01671044, 1090: 5.01709639, 1092: 5.01748286,
    1095: 5.01806274, 1104: 5.02369474, 1111: 5.02506613, 1115: 5.02585127,
    1116: 5.02604763, 1119: 5.02663783, 1120: 5.02683469, 1123: 5.02742546,
    1130: 5.02880747, 1138: 5.03039122, 1142: 5.0311854, 1178: 5.03838966,
    1199: 5.04264164, 1206: 5.04406721, 1221: 5.04713577, 1237: 5.05042965,
    1252: 5.05353788, 1260: 5.0552032, 1263: 5.0852813, 1266: 5.0858241,
    1285: 5.08927792, 1301: 5.09220745, 1304: 5.09275923, 1329: 5.09738125,
    1371: 5.10906363, 1382: 5.11116468, 1415: 5.11752516, 1424: 5.11927502,
    1449: 5.12417021, 1455: 5.12535274, 1466: 5.12700335, 1477: 5.12882302,
    1500: 5.13265536, 1519: 5.1358489, 1538: 5.13906798, 1545: 5.14026066,
    1565: 5.14368695, 1574: 5.14523838, 1583: 5.14679539, 1593: 5.14853262,
    1614: 5.15220422, 1627: 5.15449366, 1636: 5.15608582, 1644: 5.15750631,
    1652: 5.15893151, 1669: 5.16697065, 1674: 5.16788127, 1681: 5.16915901,
    1694: 5.17154188, 1708: 5.17412233, 1717: 5.17578918, 1723: 5.17690409,
    1740: 5.18007769, 1748: 5.18157892, 1751: 5.18214319, 1756: 5.18308528,
    1763: 5.18821081, 1766: 5.18878287, 1771: 5.18973756, 1777: 5.19088611,
    1780: 5.19146162, 1783: 5.1920375, 1794: 5.19415593, 1800: 5.19531559,
    1803: 5.19589634, 1806: 5.19647797, 1812: 5.19764329, 1826: 5.20037384,
    1843: 5.20371102, 1852: 5.20548749, 1865: 5.338754, 1866: 5.33893433,
    1868: 5.33929497, 1869: 5.33947551, 1872: 5.34001756, 1873: 5.34019834,
    1876: 5.34074109, 1881: 5.34164716, 1882: 5.34182846, 1883: 5.34201008,
    1911: 5.34712229, 1917: 5.34822522, 1918: 5.34840922, 1924: 5.34951552,
    1928: 5.35025441, 1937: 5.35192128, 1941: 5.35266423, 2099: 5.37628965,
    2100: 5.37646655, 2101: 5.37664355, 2103: 5.37699771, 2104: 5.37717507,
    2106: 5.37752973, 2107: 5.37770719, 2108: 5.37788462, 2109: 5.37806234,
    2110: 5.37823998, 2111: 5.37841769, 2112: 5.37859554, 2113: 5.37877354,
    2114: 5.37895143, 2115: 5.37912959, 2116: 5.37930769, 2117: 5.37948593,
    2118: 5.37966423, 2119: 5.37984246, 2120: 5.38002101, 2121: 5.38019949,
    2122: 5.38037807, 2123: 5.38055671, 2128: 5.3814512, 2134: 5.38252709,
    2141: 5.38378575, 2145: 5.38849091, 2149: 5.38921875, 2153: 5.38994783,
    2164: 5.39195854, 2189: 5.39656161, 2197: 5.39804441, 2209: 5.40027767,
    2226: 5.40345993, 2234: 5.40496502, 2280: 5.40843127, 2318: 5.41505653,
    2321: 5.41558409, 2325: 5.4162886, 2328: 5.41681784, 2333: 5.41770133,
    2339: 5.41876385, 2348: 5.42036278, 2353: 5.42125373, 2355: 5.42161067,
    2357: 5.42196781, 2363: 5.42304129, 2370: 5.42429709, 2371: 5.42447686,
    2377: 5.42555671,
}
# fmt: on
AIRS_CHANNELS = 2378
_AIRS_WAVENUMBER_CM = np.full(AIRS_CHANNELS + 1, np.nan)
_AIRS_WAVENUMBER_CM[list(AIRS_LOG10_WAVENUMBER_M)] = (
    10.0 ** np.array(list(AIRS_LOG10_WAVENUMBER_M.values())) / 100.0
)


def wavenumber_cm_inverse(sensor: str, channels: Sequence[int] | Any) -> Any:
    """Centre wavenumber in cm⁻¹ for each channel number.

    AIRS has no formulaic spectral grid; its wavenumbers come from the
    ``LOGRCW`` table of the NCEP channel subset.

    Parameters
    ----------
    sensor : str
        One of ``"airs"``, ``"iasi"``, ``"cris"``.
    channels : sequence of int
        Instrument channel numbers (1-based).
    """
    channels = np.asarray(channels, dtype=np.int64)
    if sensor == "iasi":
        _check_range(sensor, channels, 1, IASI_CHANNELS)
        return IASI_FIRST_WAVENUMBER_CM + IASI_SPACING_CM * (channels - 1)
    if sensor == "cris":
        _check_range(sensor, channels, CRIS_BANDS[0][1], CRIS_BANDS[-1][2])
        result = np.zeros(channels.shape, dtype=np.float64)
        for start, first, last in CRIS_BANDS:
            inside = (channels >= first) & (channels <= last)
            result[inside] = start + CRIS_SPACING_CM * (channels[inside] - first)
        return result
    if sensor == "airs":
        _check_range(sensor, channels, 1, AIRS_CHANNELS)
        result = _AIRS_WAVENUMBER_CM[channels]
        if np.isnan(result).any():
            missing = np.unique(channels[np.isnan(result)])
            raise ValueError(
                f"airs channels {missing.tolist()} are not in the NCEP subset"
            )
        return result
    raise ValueError(f"no spectral axis defined for sensor {sensor!r}")


def iasi_radiance_mw(scra: Any, chsf: Any) -> Any:
    """Convert IASI SCRA integer codes to mW m⁻² sr⁻¹ (cm⁻¹)⁻¹.

    The NNJA IASI BUFR stores per-footprint SCRA integer values and CHSF
    scale exponents declared per channel group (``mtiasi`` carries ten
    STCH/ENCH/CHSF groups per footprint; the caller resolves each channel's
    group). The physical radiance in W m⁻² sr⁻¹ (m⁻¹)⁻¹ (per unit
    wavenumber) is ``SCRA × 10^(-CHSF)``. Converting to
    mW m⁻² sr⁻¹ (cm⁻¹)⁻¹ adds a factor of 10⁵ (×1000 for mW, ×100 for
    per-cm⁻¹ vs per-m⁻¹).

    Parameters
    ----------
    scra : array-like
        Raw SCRA integer values from the BUFR (decoded, not yet scaled).
    chsf : array-like
        CHSF exponent already resolved to each channel, same shape as ``scra``.
    """
    scra = np.asarray(scra, dtype=np.float64)
    chsf = np.asarray(chsf, dtype=np.float64)
    return scra * np.power(10.0, -chsf) * 1e5


def cris_radiance_mw(srad: Any) -> Any:
    """Convert CrIS SRAD (W m⁻² sr⁻¹ (cm⁻¹)⁻¹) to mW m⁻² sr⁻¹ (cm⁻¹)⁻¹."""
    return np.asarray(srad, dtype=np.float64) * 1000.0


def _check_range(sensor: str, channels: Any, low: int, high: int) -> None:
    if channels.size and (channels.min() < low or channels.max() > high):
        raise ValueError(
            f"{sensor} channels must be in [{low}, {high}]; "
            f"got [{channels.min()}, {channels.max()}]"
        )

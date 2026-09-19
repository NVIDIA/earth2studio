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
from typing import Any

from earth2studio.lexicon.base import LexiconType


class MeteosatFCILexicon(metaclass=LexiconType):
    """Lexicon for MTG-I FCI Level-1C Full Disk data source.

    Maps Earth2Studio variable names to MTG FCI channel identifiers.
    The 16 spectral channels cover visible (VIS), near-infrared (NIR),
    water vapour (WV) and infrared (IR) bands. Variable names follow
    the pattern ``fci{wavelength}{band}``, e.g. ``fci87ir``.

    Note
    ----
    Channel documentation:

    - https://user.eumetsat.int/resources/user-guides/mtg-fci-level-1c-data-guide
    - https://data.eumetsat.int/product/EO:EUM:DAT:0662
    """

    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # VIS bands
        "fci04vis": ("vis_04", lambda x: x),  # VIS 0.444 µm (1 km)
        "fci05vis": ("vis_05", lambda x: x),  # VIS 0.510 µm (1 km)
        "fci06vis": ("vis_06", lambda x: x),  # VIS 0.640 µm (0.5 km / 1 km)
        "fci08vis": ("vis_08", lambda x: x),  # VIS 0.865 µm (1 km)
        "fci09vis": ("vis_09", lambda x: x),  # VIS 0.914 µm (1 km)
        # NIR bands
        "fci13nir": ("nir_13", lambda x: x),  # NIR 1.380 µm (1 km)
        "fci16nir": ("nir_16", lambda x: x),  # NIR 1.610 µm (1 km)
        "fci22nir": ("nir_22", lambda x: x),  # NIR 2.250 µm (0.5 km / 1 km)
        # IR bands
        "fci38ir": ("ir_38", lambda x: x),  # IR 3.800 µm (1 km / 2 km)
        "fci63wv": ("wv_63", lambda x: x),  # WV 6.300 µm (2 km)
        "fci73wv": ("wv_73", lambda x: x),  # WV 7.350 µm (2 km)
        "fci87ir": ("ir_87", lambda x: x),  # IR 8.700 µm (2 km)
        "fci97ir": ("ir_97", lambda x: x),  # IR 9.660 µm (2 km)
        "fci105ir": ("ir_105", lambda x: x),  # IR 10.500 µm (1 km / 2 km)
        "fci123ir": ("ir_123", lambda x: x),  # IR 12.300 µm (2 km)
        "fci133ir": ("ir_133", lambda x: x),  # IR 13.300 µm (2 km)
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Return the FCI channel name and modifier for a variable.

        Parameters
        ----------
        val : str
            Variable name (e.g. ``'fci87ir'``)

        Returns
        -------
        tuple[str, Callable]
            FCI channel key and identity modifier
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in Meteosat FCI lexicon")
        return cls.VOCAB[val]


class MeteosatLILexicon(metaclass=LexiconType):
    """Lexicon for MTG-I LI Level-2 pointed (per-detection) lightning products.

    Maps Earth2Studio variable names to a ``{product}::{field}`` key, where
    ``product`` selects the EUMETSAT Data Store collection to read from and
    ``field`` is the native NetCDF variable name within it:

    - ``LFL`` — Lightning Flashes, one record per detected flash
    - ``LGR`` — Lightning Groups, one record per detected group
    - ``LEF`` — Lightning Events Filtered, one record per detected event

    Variable names follow the instrument-agnostic ``lightning_{level}_{quantity}``
    convention shared with :py:class:`GOESGLMLexicon`, e.g.
    ``lightning_flash_radiance`` is the optical radiance of a flash. The
    ``count`` fields are synthetic: the data source fills them with 1.0 per
    record so users can sum or histogram them to obtain flash/group/event
    density during downstream regridding.

    Radiance is reported in ``mW m-2 sr-1`` in all three products.

    The ``*_footprint_pixels`` fields are the detection's footprint size,
    which lets a caller grid detections as an extent density rather than as
    centroid counts. LI reports footprint as a count of contributing
    detector pixels rather than as an area, so converting to an area needs
    the viewing-geometry-dependent pixel size and is left to the caller.
    Events are single pixels, so there is no event-level footprint field.

    Note
    ----
    Variable documentation:

    - https://user.eumetsat.int/resources/user-guides/mtg-li-level-2-data-guide
    - https://data.eumetsat.int/product/EO:EUM:DAT:0691
    """

    VOCAB: dict[str, tuple[str, Callable[[Any], Any]]] = {
        # Lightning Flashes (LFL)
        "lightning_flash_radiance": ("LFL::radiance", lambda x: x),
        "lightning_flash_count": ("LFL::_count", lambda x: x),
        "lightning_flash_duration": ("LFL::flash_duration", lambda x: x),
        "lightning_flash_footprint_pixels": ("LFL::flash_footprint", lambda x: x),
        # Lightning Groups (LGR)
        "lightning_group_radiance": ("LGR::radiance", lambda x: x),
        "lightning_group_count": ("LGR::_count", lambda x: x),
        "lightning_group_footprint_pixels": ("LGR::number_of_events", lambda x: x),
        # Lightning Events Filtered (LEF)
        "lightning_event_radiance": ("LEF::radiance", lambda x: x),
        "lightning_event_count": ("LEF::_count", lambda x: x),
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable[[Any], Any]]:
        """Return the LI product/field key and modifier for a variable.

        Parameters
        ----------
        val : str
            Variable name (e.g. ``'lightning_flash_radiance'``)

        Returns
        -------
        tuple[str, Callable]
            ``(key, modifier)`` where ``key`` is ``'{product}::{field}'``
            and ``modifier`` is the identity function; values are returned
            in their native physical units.
        """
        if val not in cls.VOCAB:
            raise KeyError(f"Variable {val} not found in Meteosat LI lexicon")
        return cls.VOCAB[val]

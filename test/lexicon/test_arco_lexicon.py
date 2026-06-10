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

import numpy as np
import pytest
import torch

from earth2studio.lexicon import ARCOLexicon
from earth2studio.lexicon.base import E2STUDIO_VOCAB


@pytest.mark.parametrize(
    "variable",
    [
        ["t2m"],
        ["u10m", "v200"],
        ["hcc", "z500", "q700"],
        ["sdor", "slor", "tcw", "tp06"],
        ["stl1", "swvl2", "ssrd06", "strd06", "ro"],
        ["cdww", "mwp", "swh", "wmb"],
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_arco_lexicon(variable, device):
    input = torch.randn(len(variable), 8).to(device)
    for v in variable:
        label, modifier = ARCOLexicon[v]
        output = modifier(input)
        assert isinstance(label, str)
        assert input.shape == output.shape
        assert input.device == output.device


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        ("cp06", "convective_precipitation::"),
        ("hcc", "high_cloud_cover::"),
        ("lcc", "low_cloud_cover::"),
        ("mcc", "medium_cloud_cover::"),
        ("ro", "runoff::"),
        ("sdor", "standard_deviation_of_orography::"),
        ("sf", "snowfall::"),
        ("slor", "slope_of_sub_gridscale_orography::"),
        ("ssrd06", "surface_solar_radiation_downwards::"),
        ("stl1", "soil_temperature_level_1::"),
        ("stl2", "soil_temperature_level_2::"),
        ("strd06", "surface_thermal_radiation_downwards::"),
        ("swvl1", "volumetric_soil_water_layer_1::"),
        ("swvl2", "volumetric_soil_water_layer_2::"),
        ("tcc", "total_cloud_cover::"),
        ("tcw", "total_column_water::"),
        ("tp06", "total_precipitation::"),
    ],
)
def test_arco_aifs_aliases(variable, expected):
    assert ARCOLexicon.VOCAB[variable] == expected


def test_new_arco_aliases_are_in_base_vocab():
    aliases = [
        "cdww",
        "cp06",
        "cos_mwd",
        "hcc",
        "lcc",
        "mcc",
        "mwp",
        "ro",
        "sd",
        "sdor",
        "sf",
        "sin_mwd",
        "slor",
        "ssrd06",
        "stl1",
        "stl2",
        "strd06",
        "swh",
        "swvl1",
        "swvl2",
        "tcc",
        "tcw",
        "tp06",
        "wmb",
    ]

    assert not set(aliases) - set(E2STUDIO_VOCAB)


def test_arco_wave_direction_modifiers():
    degrees = np.array([0.0, 90.0, 180.0, 270.0])
    _, cos_modifier = ARCOLexicon["cos_mwd"]
    _, sin_modifier = ARCOLexicon["sin_mwd"]

    assert np.allclose(cos_modifier(degrees), np.array([1.0, 0.0, -1.0, 0.0]))
    assert np.allclose(sin_modifier(degrees), np.array([0.0, 1.0, 0.0, -1.0]))

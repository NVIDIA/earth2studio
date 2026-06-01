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

from earth2studio.lexicon import CFSFluxLexicon, CFSLexicon


def test_cfs_lexicon_pgbf_surface():
    # MSL pressure stays in raw units; identity modifier.
    key, mod = CFSLexicon["msl"]
    assert key == "pgbf::PRMSL::mean sea level"
    assert mod(np.array([1.0])) == np.array([1.0])

    # 2 m dewpoint and RH are pgbf records, not flxf.
    key, _ = CFSLexicon["d2m"]
    assert key == "pgbf::DPT::2 m above ground"
    key, _ = CFSLexicon["r2m"]
    assert key == "pgbf::RH::2 m above ground"


def test_cfs_lexicon_pgbf_pressure_levels():
    # Geopotential modifier converts HGT (m) -> geopotential (m^2 s^-2).
    key, mod = CFSLexicon["z500"]
    assert key == "pgbf::HGT::500 mb"
    assert mod(np.array([1.0])) == pytest.approx(np.array([9.81]))

    # u/v winds reach the lexicon via vector-sibling .idx records but the
    # lexicon entry itself just names the parameter.
    key, _ = CFSLexicon["u850"]
    assert key == "pgbf::UGRD::850 mb"
    key, _ = CFSLexicon["v850"]
    assert key == "pgbf::VGRD::850 mb"

    # Spot-check a representative subset of the programmatic pressure-level
    # entries to catch off-by-one bugs in the level loop.
    for letter, param in (("t", "TMP"), ("q", "SPFH"), ("r", "RH")):
        for level in (50, 250, 1000):
            key, _ = CFSLexicon[f"{letter}{level}"]
            assert key == f"pgbf::{param}::{level} mb"


def test_cfs_lexicon_pgbf_invalid_key():
    with pytest.raises(KeyError):
        CFSLexicon["does_not_exist"]
    # Surface fields that the user might *expect* but pgbf does not carry:
    for missing in ("t2m", "u10m", "v10m", "sp", "q2m", "tcwv"):
        with pytest.raises(KeyError):
            CFSLexicon[missing]


def test_cfs_flux_lexicon():
    key, mod = CFSFluxLexicon["t2m"]
    assert key == "flxf::TMP::2 m above ground"
    assert mod(np.array([300.0])) == np.array([300.0])

    key, _ = CFSFluxLexicon["u10m"]
    assert key == "flxf::UGRD::10 m above ground"

    key, _ = CFSFluxLexicon["tpf"]
    assert key == "flxf::PRATE::surface"

    # Cloud-cover layers map to distinct flxf records.
    key, _ = CFSFluxLexicon["tcc"]
    assert key.endswith("entire atmosphere (considered as a single layer)")
    key, _ = CFSFluxLexicon["hcc"]
    assert key == "flxf::TCDC::high cloud layer"


def test_cfs_flux_lexicon_invalid_key():
    with pytest.raises(KeyError):
        CFSFluxLexicon["d2m"]  # 2 m dewpoint is a pgbf record, not flxf

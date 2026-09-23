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

"""Completeness gate for native DataArray model conformance coverage.

Every concrete export needs an existing mock-model conformance test or a specific
exemption pinned by its test. Discovery uses introspection rather than __all__,
which omits some public models. New exports cannot silently bypass this inventory.
Optional backend skips describe execution availability, not contract exemptions.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import earth2studio.models.dx as dx
import earth2studio.models.px as px
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.base import PrognosticModel


def _model_classes(module: ModuleType, base: type) -> dict[str, type]:
    return {
        name: obj
        for name in dir(module)
        if not name.startswith("_")
        for obj in [getattr(module, name)]
        if inspect.isclass(obj)
        and obj is not base
        and obj.__module__.startswith(f"{module.__name__}.")
    }


_PROGNOSTIC_CLASSES = _model_classes(px, PrognosticModel)
_DIAGNOSTIC_CLASSES = _model_classes(dx, DiagnosticModel)

_PROGNOSTIC_CONFORMANT: set[str] = {
    "ACE2ERA5",
    "AIFS",
    "AIFS2",
    "AIFS2ENS",
    "AIFSENS",
    "Atlas",
    "AtlasCRPS",
    "Aurora",
    "Aurora1p5",
    "Aurora1p5Ensemble",
    "CBottleVideo",
    "DiagnosticWrapper",
    "DLESyM",
    "DLESyMLatLon",
    "DLESyMv0_ISCCP_ERA5",
    "DLESyMv0_ISCCP_ERA5LatLon",
    "DLWP",
    "FCN",
    "FengWu",
    "FuXi",
    "GenCastMini",
    "GraphCastOperational",
    "GraphCastSmall",
    "InterpModAFNO",
    "Pangu3",
    "Pangu6",
    "Pangu24",
    "Persistence",
    "SamudrACE",
    "SFNO",
    "StormCast",
    "StormCastCONUS",
    "StormScopeGOES",
    "StormScopeMRMS",
    "StormScopeMeteosatEU",
    "UCast",
    "WeatherNext2Cyclones",
    "WeatherNext2CyclonesMini",
}

_PROGNOSTIC_EXEMPT: dict[str, str] = {
    "FCN3": (
        "P14: core noise-state refresh draws from global RNG; pinned by "
        "test/models/px/test_fcn3.py::test_fcn3_conformance. "
        "Native input/yield ownership violations are fixed."
    ),
    "FuXiS2S": (
        "P13: ONNX graph samples internal perturbations without a seed API; "
        "test/models/px/test_fuxi_s2s.py::test_fuxi_s2s_conformance."
    ),
    "DataReplay": (
        "P13 with an uncached random source: repeated fetches differ; "
        "test/models/px/test_datareplay.py::test_datareplay_conformance. "
        "Replay determinism depends on its configured source."
    ),
}

_DIAGNOSTIC_CONFORMANT: set[str] = {
    "CBottleInfill",
    "CBottleSR",
    "CBottleTCGuidance",
    "ClimateNet",
    "CorrDiff",
    "CorrDiffCMIP6",
    "CorrDiffCosmoEra5",
    "CorrDiffTaiwan",
    "DLESyMv0_ISCCP_ERA5Precip",
    "DerivedRH",
    "DerivedRHDewpoint",
    "DerivedSurfacePressure",
    "DerivedTCWV",
    "DerivedVPD",
    "DerivedWS",
    "Identity",
    "OrbitGlobalPrecip",
    "PrecipitationAFNO",
    "PrecipitationAFNOv2",
    "SolarRadiationAFNO1H",
    "SolarRadiationAFNO6H",
    "StormScopeDxNSRDB",
    "TCTrackerVitart",
    "TCTrackerWuDuan",
    "WindgustAFNO",
}
_DIAGNOSTIC_EXEMPT: dict[str, str] = {}


def _check_registration(
    discovered: dict[str, type],
    conformant: set[str],
    exempt: dict[str, str],
    registry_path: str,
) -> None:
    known = conformant | set(exempt)
    missing = set(discovered) - known
    assert not missing, (
        f"{sorted(missing)} are exported but not registered in {registry_path}. "
        "Add a model conformance test and a CONFORMANT entry, or a pinned exemption."
    )
    stale = known - set(discovered)
    assert not stale, f"{sorted(stale)} no longer exist; remove stale inventory entries"
    overlap = conformant & set(exempt)
    assert not overlap, f"{sorted(overlap)} are both conformant and exempt"
    assert all(reason.strip() for reason in exempt.values())


def test_prognostic_models_are_registered() -> None:
    _check_registration(
        _PROGNOSTIC_CLASSES,
        _PROGNOSTIC_CONFORMANT,
        _PROGNOSTIC_EXEMPT,
        "test/models/test_model_conformance.py",
    )


def test_diagnostic_models_are_registered() -> None:
    _check_registration(
        _DIAGNOSTIC_CLASSES,
        _DIAGNOSTIC_CONFORMANT,
        _DIAGNOSTIC_EXEMPT,
        "test/models/test_model_conformance.py",
    )

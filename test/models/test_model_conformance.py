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
    "DLESyM",
    "DLESyMLatLon",
    "DLESyMv0_ISCCP_ERA5",
    "DLESyMv0_ISCCP_ERA5LatLon",
    "DLWP",
    "FCN",
    "FengWu",
    "FuXi",
    "GraphCastOperational",
    "GraphCastSmall",
    "InterpModAFNO",
    "Pangu3",
    "Pangu6",
    "Pangu24",
    "Persistence",
    "SamudrACE",
    "SFNO",
    "UCast",
}

_PROGNOSTIC_EXEMPT: dict[str, str] = {
    "Aurora1p5Ensemble": (
        "P12/P14: retains set_rng(seed) and global seeding; each iterator reapplies "
        "the constructor seed and resets noise. "
        "test/models/px/test_aurora1p5.py::test_aurora1p5_ensemble_conformance."
    ),
    "CBottleVideo": (
        "P13: undeclared backend sampling; "
        "test/models/px/test_cbottle_video.py::TestCBottleVideoMock::test_cbottle_video_conformance."
    ),
    "DiagnosticWrapper": (
        "P13 when wrapping an unseeded diagnostic: retains component randomness "
        "without seeding dispatch; test/models/px/test_dxwrapper.py::test_diagnosticwrapper_conformance."
    ),
    "GenCastMini": (
        "P13 with seed=None: fresh per-time keys, without stochastic declaration; "
        "test/models/px/test_gencast_mini.py::test_gencast_mini_conformance."
    ),
    "StormCast": (
        "P13: global diffusion draws; test/models/px/test_stormcast.py::test_stormcast_conformance."
    ),
    "StormCastCONUS": (
        "P13: global diffusion draws; test/models/px/test_stormcastconus.py."
    ),
    "StormScopeGOES": (
        "P13: global diffusion draws; test/models/px/test_stormscope.py."
    ),
    "StormScopeMRMS": (
        "P13: global diffusion draws; test/models/px/test_stormscope.py."
    ),
    "StormScopeMeteosatEU": (
        "P13: global diffusion draws; "
        "test/models/px/test_stormscope_meteosat.py::test_stormscope_meteosat_conformance."
    ),
    "WeatherNext2Cyclones": (
        "P13: advancing functional keys without stochastic declaration; "
        "test/models/px/test_weathernext2.py::test_weathernext2_conformance."
    ),
    "WeatherNext2CyclonesMini": (
        "P13: same key progression as WeatherNext2Cyclones; "
        "test/models/px/test_weathernext2.py::test_weathernext2_conformance."
    ),
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
    "ClimateNet",
    "CorrDiffCMIP6",
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
    "TCTrackerVitart",
    "TCTrackerWuDuan",
    "WindgustAFNO",
}
_DIAGNOSTIC_EXEMPT: dict[str, str] = {
    "CorrDiff": (
        "D9 with seed=None and a noise-producing sampler: undeclared randomness; "
        "test/models/dx/test_corrdiff.py::TestCorrDiffForward::test_corrdiff_conformance."
    ),
    "CBottleInfill": (
        "D9: backend infill has no seed support; "
        "test/models/dx/test_cbottle_infill.py::TestCBottleMock::test_cbottleinfill_conformance."
    ),
    "CBottleSR": (
        "D9 with seed=None: undeclared sampler randomness; "
        "test/models/dx/test_cbottle_sr.py::TestCBottleSRMock::test_cbottle_sr_conformance."
    ),
    "CBottleTCGuidance": (
        "D9 with seed=None: undeclared backend sampling; "
        "test/models/dx/test_cbottle_tc.py::TestCBottleTCMock::test_cbottletcguidance_conformance."
    ),
    "CorrDiffCosmoEra5": (
        "D9 with seed=None: fresh diffusion noise; "
        "test/models/dx/test_corrdiff_cosmo_era5.py::test_corrdiff_cosmo_era5_conformance."
    ),
    "CorrDiffTaiwan": (
        "D9 with seed=None: fresh sampler seeds; "
        "test/models/dx/test_corrdiff_taiwan.py::test_corrdiff_taiwan_conformance."
    ),
    "StormScopeDxNSRDB": (
        "D9 with seed=None: undeclared global draws; "
        "test/models/dx/test_stormscope_dx_nsrdb.py::test_stormscope_dx_nsrdb_conformance."
    ),
}


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

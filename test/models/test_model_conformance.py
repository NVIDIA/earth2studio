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

"""Completeness gate for the Earth2Studio model contract.

Every model class reachable from ``earth2studio.models.px`` /
``earth2studio.models.dx`` must be accounted for below, either as
CONFORMANT (its own ``test/models/{px,dx}/test_<name>.py`` has a
``test_<model>_conformance`` asserting ``check_prognostic_contract`` /
``check_diagnostic_contract`` against a mock instance — see the
``earth2studio-create-prognostic`` / ``earth2studio-create-diagnostic``
skills) or EXEMPT with a specific, reviewable reason. A model that becomes
reachable through the public namespace without a matching entry fails
``test_prognostic_models_are_registered`` / ``test_diagnostic_models_are_registered``
below, so a new model cannot silently ship without conformance coverage or a
documented reason it is missing.

Models are discovered by introspection rather than ``px.__all__`` /
``dx.__all__``: those lists are curated for documentation and already omit
real model classes — ``dx.__all__`` has 13 entries against 25 concrete
diagnostic classes, missing ``Identity``, the ``Derived*`` diagnostics, the
``TCTracker*`` trackers, and the ``CBottle*`` diagnostics. A hand-curated list
is exactly the kind of registry that goes stale silently (see
``test/models/test_auto_models.py``, whose model list is already behind the
44 model test files); introspection cannot drift from what the namespace
actually exports.

Adding a model here:
- If its test file has ``test_<model>_conformance``, add the class name to
  the matching ``_CONFORMANT`` set.
- Otherwise, add it to the matching ``_EXEMPT`` dict with a reason specific
  enough to review — "pending backfill" is acceptable for models that
  predate this gate, but a model added after this gate exists should have a
  structural reason (e.g. output is not a gridded coordinate system) rather
  than a deferral.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import earth2studio.models.dx as dx
import earth2studio.models.px as px
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.base import PrognosticModel


def _model_classes(module: ModuleType, base: type) -> dict[str, type]:
    """Concrete model classes reachable from a models namespace, by name.

    Excludes the protocol base itself and anything not defined inside the
    package (re-exported utilities, standard library names picked up by
    ``dir()``).
    """
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

_PENDING_BACKFILL = (
    "pending conformance backfill: added before the model contract "
    "(dev/spec/MODEL_CONTRACT_SPEC.md) landed and has no "
    "test_<model>_conformance yet"
)

# Models with a mock-model conformance test in their own test file.
_PROGNOSTIC_CONFORMANT: set[str] = set()

_PROGNOSTIC_EXEMPT: dict[str, str] = {
    name: _PENDING_BACKFILL
    for name in [
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
        "DLESyM",
        "DLESyMLatLon",
        "DLESyMv0_ISCCP_ERA5",
        "DLESyMv0_ISCCP_ERA5LatLon",
        "DLWP",
        "DataReplay",
        "DiagnosticWrapper",
        "FCN",
        "FCN3",
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
        "SFNO",
        "SamudrACE",
        "StormCast",
        "StormCastCONUS",
        "StormScopeGOES",
        "StormScopeMRMS",
        "StormScopeMeteosatEU",
        "UCast",
        "WeatherNext2CyclonesMini",
    ]
}

_DIAGNOSTIC_CONFORMANT: set[str] = set()

_DIAGNOSTIC_EXEMPT: dict[str, str] = {
    name: _PENDING_BACKFILL
    for name in [
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
    ]
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
        f"{sorted(missing)} are reachable from the models namespace but not "
        f"registered in {registry_path}. Add a test_<model>_conformance test "
        "and list the class under the CONFORMANT set, or add it to the "
        "EXEMPT dict with a reason."
    )

    stale = known - set(discovered)
    assert not stale, (
        f"{sorted(stale)} are registered in {registry_path} but no longer "
        "exist in the models namespace; remove the stale entries."
    )

    overlap = conformant & set(exempt)
    assert not overlap, f"{sorted(overlap)} are listed as both conformant and exempt"

    assert all(
        reason.strip() for reason in exempt.values()
    ), "exemption reasons must not be blank"


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

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

# Models with a mock-model conformance test in their own test file.
_PROGNOSTIC_CONFORMANT: set[str] = {
    "AIFS",
    "AIFS2",
    "AIFS2ENS",
    "AIFSENS",
    "Atlas",
    "AtlasCRPS",
    "Aurora",
    "CBottleVideo",
    "DLWP",
    "DiagnosticWrapper",
    "FCN",
    "FCN3",
    "FengWu",
    "Pangu3",
    "Pangu6",
    "Pangu24",
    "Persistence",
    "SFNO",
    "UCast",
}

# Models with a real, verified contract violation, found by the backfill effort
# that added test_<model>_conformance across every model test file. Each is
# tracked here with the specific rule(s) it fails and the test that pins the
# exact violation via pytest.raises(ContractException) (so it documents the
# gap without leaving a permanently red test). Fixing these is a follow-up:
# making every existing wrapper actually compliant with the contract.
_PROGNOSTIC_EXEMPT: dict[str, str] = {
    "ACE2ERA5": (
        "fails P16: create_iterator() yield 0 changes after later steps are "
        "produced, so the yields alias one buffer — "
        "test/models/px/test_ace2.py::test_ace2era5_conformance"
    ),
    "FuXi": (
        "fails P15 (both create_iterator() and __call__ mutate the input "
        "tensor in place) and P16 (yield 0 changes after later steps are "
        "produced, so the yields alias one buffer) — "
        "test/models/px/test_fuxi.py::TestFuXiMock::test_fuxi_conformance"
    ),
    "DLESyM": (
        "fails P16 (create_iterator() yields alias one buffer), P7 (0th "
        "yield lead_time is wrong), and P13 (two rollouts from one input "
        "disagree despite declaring stochastic=False) — "
        "test/models/px/test_dlesym.py::test_dlesym_conformance"
    ),
    "DLESyMLatLon": (
        "shares DLESyM's create_iterator()/rollout logic, which is "
        "confirmed non-conformant (P16/P7/P13, see DLESyM above); not "
        "independently executed here because earth2grid's CPU regridder "
        "segfaults in this sandbox regardless of device — "
        "test/models/px/test_dlesym.py::test_dlesym_latlon_conformance"
    ),
    "DLESyMv0_ISCCP_ERA5": (
        "inherits DLESyM's rollout logic; fails P7 (0th yield lead_time is "
        "wrong) and P13 (two rollouts from one input disagree despite "
        "declaring stochastic=False) — "
        "test/models/px/test_dlesym_v0_isccp_era5.py::test_dlesym_v0_isccp_era5_conformance"
    ),
    "DLESyMv0_ISCCP_ERA5LatLon": (
        "shares DLESyMv0_ISCCP_ERA5's rollout logic, confirmed "
        "non-conformant above (P7/P13); not independently executed here "
        "because earth2grid's CPU regridder segfaults in this sandbox "
        "regardless of device — "
        "test/models/px/test_dlesym_v0_isccp_era5.py::test_dlesym_v0_isccp_era5_latlon_conformance"
    ),
    "Aurora1p5": (
        "fails P16: create_iterator() yields alias one buffer (yield 0 "
        "changes after later steps are produced) — "
        "test/models/px/test_aurora1p5.py::test_aurora1p5_conformance"
    ),
    "Aurora1p5Ensemble": (
        "fails P14 (bare torch.manual_seed in set_rng, no fork_rng — the "
        "spec's documented known deviation) and P16 (same aliased-yield bug "
        "as Aurora1p5) — "
        "test/models/px/test_aurora1p5.py::test_aurora1p5ensemble_conformance"
    ),
    "DataReplay": (
        "fails P13: declares stochastic=False but replays from an unseeded "
        "Random/Random_FX data source, so two rollouts from one input "
        "disagree — test/models/px/test_datareplay.py::test_datareplay_conformance"
    ),
    "GenCastMini": (
        "fails P10: create_iterator() applies rear_hook but never "
        "front_hook, so a hook a caller sets is silently dropped (reasoned "
        "from source, matches the GraphCast deviation; not executed in this "
        "sandbox — no CUDA jax wheel) — "
        "test/models/px/test_gencast_mini.py::test_gencastmini_conformance"
    ),
    "GraphCastOperational": (
        "fails P10: create_iterator() applies rear_hook but never "
        "front_hook — documented known deviation, see "
        "dev/spec/MODEL_CONTRACT_SPEC.md — "
        "test/models/px/test_graphcast.py::test_graphcastoperational_conformance"
    ),
    "GraphCastSmall": (
        "fails P10: same rear_hook-only gap as GraphCastOperational — "
        "test/models/px/test_graphcast.py::test_graphcastsmall_conformance"
    ),
    "InterpModAFNO": (
        "fails P5 (invalid coordinate system with swapped final two "
        "dimensions is silently accepted instead of raising ValueError) and "
        "P10 (create_iterator() applies neither hook chain) — "
        "test/models/px/test_interpmodafno.py::test_interpmodafno_conformance"
    ),
    "SamudrACE": (
        "check_prognostic_contract() cannot complete: __call__ "
        "unconditionally raises NotImplementedError ('use create_iterator') "
        "and the checker's P15/P10 probes call __call__ directly with no "
        "guard — needs either a single-step __call__ or checker support for "
        "iterator-only models — "
        "test/models/px/test_samudrace.py::test_samudrace_conformance"
    ),
    "StormCast": (
        "fails P13: unseeded torch.randn_like diffusion latents, declares "
        "stochastic=False — "
        "test/models/px/test_stormcast.py::test_stormcast_conformance"
    ),
    "StormCastCONUS": (
        "fails P13: same unseeded diffusion-latent gap as StormCast — "
        "test/models/px/test_stormcastconus.py::test_stormcastconus_conformance"
    ),
    "StormScopeGOES": (
        "fails P13: same unseeded diffusion-latent gap as StormCast — "
        "test/models/px/test_stormscope.py::test_stormscopegoes_conformance"
    ),
    "StormScopeMRMS": (
        "fails P13: same unseeded diffusion-latent gap as StormCast — "
        "test/models/px/test_stormscope.py::test_stormscopemrms_conformance"
    ),
    "StormScopeMeteosatEU": (
        "fails P15 (create_iterator() mutates the input coordinate system "
        "in place), P7 (0th yield lead_time is wrong), P10 (neither hook "
        "chain applied), and P13 (same unseeded diffusion-latent gap as "
        "StormCast) — "
        "test/models/px/test_stormscope_meteosat.py::test_stormscopemeteosateu_conformance"
    ),
    "WeatherNext2CyclonesMini": (
        "fails P10: same rear_hook-only gap as GraphCast/GenCastMini "
        "(reasoned from source, not executed — no CUDA jax wheel in this "
        "sandbox) — "
        "test/models/px/test_weathernext2.py::test_weathernext2cyclonesmini_conformance"
    ),
}

_DIAGNOSTIC_CONFORMANT: set[str] = {
    "CBottleInfill",
    "CBottleSR",
    "CBottleTCGuidance",
    "ClimateNet",
    "CorrDiff",
    "DLESyMv0_ISCCP_ERA5Precip",
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
    "CorrDiffCMIP6": (
        "fails D6: preprocess_input() transposes to a view and "
        "_apply_sai_cover() writes into that view in place, mutating the "
        "caller's input tensor (same bug class as issue #1133 / PR #1134) — "
        "test/models/dx/test_corrdiff_cmip6.py::test_corrdiffcmip6_conformance"
    ),
    "CorrDiffCosmoEra5": (
        "fails D9: diffusion latents draw from an unseeded global RNG when "
        "seed=None (the default); declares neither stochastic nor set_rng — "
        "test/models/dx/test_corrdiff_cosmo_era5.py::test_corrdiff_cosmo_era5_conformance"
    ),
    "CorrDiffTaiwan": (
        "fails D9: sampler seeds default to np.random.randint per call when "
        "seed=None; declares neither stochastic nor set_rng — "
        "test/models/dx/test_corrdiff_taiwan.py::test_corrdiff_taiwan_conformance"
    ),
    "DerivedRH": (
        "fails D9: an unseeded probe input can land within float distance "
        "of the es_w formula's t=32.19K singularity, producing NaN on one "
        "call and not the other (torch.allclose treats NaN != NaN) — "
        "test/models/dx/test_derived.py::test_derivedrh_conformance"
    ),
    "StormScopeDxNSRDB": (
        "fails D9: declares neither stochastic nor set_rng, but draws fresh "
        "unseeded torch.randn noise per call — "
        "test/models/dx/test_stormscope_dx_nsrdb.py::test_stormscope_dx_nsrdb_conformance"
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

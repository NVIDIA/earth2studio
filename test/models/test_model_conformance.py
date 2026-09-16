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
    "DLWP",
    "FCN",
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
    "CBottleVideo": (
        "fails P13: declares stochastic=False but samples diffusion latents "
        "per call, so two rollouts from one input disagree; declares neither "
        "stochastic nor set_rng despite already passing a seed through to the "
        "core model's sample() — "
        "test/models/px/test_cbottle_video.py::TestCBottleVideoMock::test_cbottle_video_conformance"
    ),
    "DiagnosticWrapper": (
        "fails P7 (the wrapped diagnostic's 'sample' dimension leaves the 0th "
        "yield carrying dimensions the contract does not allow there), P10 "
        "(create_iterator() applies neither hook) and P13 (inherits the "
        "wrapped CorrDiffTaiwan's unseeded sampler while declaring "
        "stochastic=False) — "
        "test/models/px/test_dxwrapper.py::test_diagnosticwrapper_conformance"
    ),
    "FCN3": (
        "fails P15 and P16 (_forward() squeezes the input to a view and writes "
        "into it in place, so both call paths mutate the caller's tensor and "
        "the tensor already yielded as step 0 changes under the caller) and "
        "P14 (refreshing the core model's internal noise state draws from the "
        "global generator, so stepping a seeded model perturbs global RNG "
        "state) — test/models/px/test_fcn3.py::test_fcn3_conformance"
    ),
    "FuXi": (
        "fails P15 (both create_iterator() and __call__ mutate the input "
        "tensor in place) and P16 (yield 0 changes after later steps are "
        "produced, so the yields alias one buffer) — "
        "test/models/px/test_fuxi.py::TestFuXiMock::test_fuxi_conformance"
    ),
    "DLESyM": (
        "fails P7 (the 0th yield's lead_time carries the whole input history "
        "rather than the analysis time), always; and, from two independent, "
        "confirmed bugs in prepare_output_data() — it allocates its output "
        "tensor with torch.empty and only partially writes it (ocean output "
        "covers 2 of this mock's 16 atmos_output_times lead-time slots per "
        "variable, leaving the rest uninitialized), and separately a yielded "
        "tensor's storage is genuinely mutated after being yielded (confirmed "
        "by direct storage-identity inspection, not just value comparison; "
        "likely site is _next_step_inputs feeding a slice of the previous "
        "yield back in as the next step's input) — P13 and/or P16, whose "
        "combination varies by run since it depends on allocator state — "
        "test/models/px/test_dlesym.py::test_dlesym_conformance"
    ),
    "DLESyMLatLon": (
        "shares DLESyM's create_iterator()/rollout logic, which is "
        "confirmed non-conformant (P7, plus P13 and/or P16, see DLESyM "
        "above); not independently executed here because earth2grid's CPU "
        "regridder segfaults in this sandbox regardless of device — "
        "test/models/px/test_dlesym.py::test_dlesym_latlon_conformance"
    ),
    "DLESyMv0_ISCCP_ERA5": (
        "inherits DLESyM's rollout logic; fails P7 (0th yield lead_time is "
        "wrong), always; and, like DLESyM, P13 and/or P16 from the same two "
        "confirmed prepare_output_data() bugs — "
        "test/models/px/test_dlesym_v0_isccp_era5.py::test_dlesym_v0_isccp_era5_conformance"
    ),
    "DLESyMv0_ISCCP_ERA5LatLon": (
        "shares DLESyMv0_ISCCP_ERA5's rollout logic, confirmed "
        "non-conformant above (P7, plus P13 and/or P16); not independently "
        "executed here because earth2grid's CPU regridder segfaults in this "
        "sandbox regardless of device — "
        "test/models/px/test_dlesym_v0_isccp_era5.py::test_dlesym_v0_isccp_era5_latlon_conformance"
    ),
    "Aurora1p5": (
        "fails P16: create_iterator() yields alias one buffer (yield 0 "
        "changes after later steps are produced) — "
        "test/models/px/test_aurora1p5.py::test_aurora1p5_conformance"
    ),
    "Aurora1p5Ensemble": (
        "fails P12 (set_rng(seed) does not accept reset=True, so a generic "
        "caller following the documented set_rng(seed, reset=True) interface "
        "raises TypeError), P14 (bare torch.manual_seed in set_rng, no "
        "fork_rng — the spec's documented known deviation), and P16 (same "
        "aliased-yield bug as Aurora1p5); create_iterator() also re-applies "
        "the constructor seed on every call, silently overriding a seed a "
        "caller already set via set_rng() (see MODEL_CONTRACT_SPEC.md's "
        "'Seeding is the only entry point') — "
        "test/models/px/test_aurora1p5.py::test_aurora1p5ensemble_conformance"
    ),
    "DataReplay": (
        "fails P13: declares stochastic=False but replays from an unseeded "
        "Random/Random_FX data source, so two rollouts from one input "
        "disagree — test/models/px/test_datareplay.py::test_datareplay_conformance"
    ),
    "GenCastMini": (
        "fails P5 (output_coords() accepts a coordinate system with its final "
        "two dimensions swapped instead of raising ValueError), P10 "
        "(create_iterator() applies rear_hook but never front_hook, so a hook "
        "a caller sets is silently dropped — the documented known deviation, "
        "see dev/spec/MODEL_CONTRACT_SPEC.md), P13 (declares stochastic=False "
        "but two rollouts from one input do not compare equal) and P16 (the "
        "yields alias one buffer) — "
        "test/models/px/test_gencast_mini.py::test_gencast_mini_conformance"
    ),
    "GraphCastOperational": (
        "fails the same four rules as GenCastMini above (P5/P10/P13/P16); "
        "this wrapper family shares its structure — "
        "test/models/px/test_graphcast.py::test_graphcast_operational_conformance"
    ),
    "GraphCastSmall": (
        "fails the same four rules as GraphCastOperational — "
        "test/models/px/test_graphcast.py::test_graphcast_small_conformance"
    ),
    "InterpModAFNO": (
        "fails P5 (invalid coordinate system with swapped final two "
        "dimensions is silently accepted instead of raising ValueError) and "
        "P10 (create_iterator() applies neither hook) — "
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
        "fails the same four rules as GraphCast/GenCastMini "
        "(P5/P10/P13/P16) — "
        "test/models/px/test_weathernext2.py::test_weathernext2_conformance"
    ),
}

_DIAGNOSTIC_CONFORMANT: set[str] = {
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
    "WindgustAFNO",
}

_DIAGNOSTIC_EXEMPT: dict[str, str] = {
    "CBottleInfill": (
        "fails D9: the sampler call takes no seed argument at all ('NO SEED "
        "SUPPORT!' in cbottle_infill.py), so diffusion latents come from the "
        "unseeded global RNG; declares neither stochastic nor set_rng — "
        "test/models/dx/test_cbottle_infill.py::TestCBottleMock::test_cbottleinfill_conformance"
    ),
    "CBottleSR": (
        "fails D9: with the default seed=None the diffusion latents come from "
        "the unseeded global RNG; declares neither stochastic nor set_rng, and "
        "its seeded path uses a bare torch.manual_seed that needs forking — "
        "test/models/dx/test_cbottle_sr.py::TestCBottleSRMock::test_cbottle_sr_conformance"
    ),
    "CBottleTCGuidance": (
        "fails D9: same unseeded-latent gap as CBottleInfill — "
        "test/models/dx/test_cbottle_tc.py::TestCBottleTCMock::test_cbottletcguidance_conformance"
    ),
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
    "TCTrackerVitart": (
        "fails D9 by design: the tracker accumulates a path_buffer across "
        "calls (that is what turns per-frame centers into tracks, hence "
        "reset_path_buffer()), so a second call on one input returns a larger "
        "tensor. The contract has no notion of a stateful diagnostic yet — "
        "test/models/dx/test_tc_tracking.py::test_tc_tracker_vitart_conformance"
    ),
    "TCTrackerWuDuan": (
        "fails D9 for the same reason as TCTrackerVitart: the shared "
        "path_buffer accumulates across calls — "
        "test/models/dx/test_tc_tracking.py::test_tc_tracker_wu_duan_conformance"
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

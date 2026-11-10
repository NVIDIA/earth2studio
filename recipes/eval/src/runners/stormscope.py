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

"""Coupled GOES to MRMS nowcasting on the StormScope wrappers.

The ``stormscope`` builder wires a :class:`~earth2studio.models.px.StormScopeGOES`
model and a :class:`~earth2studio.models.px.StormScopeMRMS` model into an
:class:`earth2studio.coupling.CoupledRunner` through one
:class:`~earth2studio.coupling.Connector`: the GOES model's frames feed the radar
model's GOES conditioning slot.  The schedule follows the contract the MRMS
wrapper documents.  The radar model initializes from the GOES history the GOES
model holds, and each radar step takes the GOES forecast at the lead time of its
previous forecast, published one step earlier.  The runner yields the two
forecasts as the streams ``goes`` and ``mrms``, both on the shared HRRR projected
grid.  It batches ensemble members, seeds each model per member and draws an
optional per-model initial-condition perturbation for each member.

Regridding onto the model grid and the radar and lightning composition are the
models' own ``default_sources()``.  This module resolves one source per role,
couples the models and records the roles for ``predownload.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import hydra
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.coupling import Connector, CoupledRunner
from earth2studio.data import CompositeSource
from earth2studio.grids import Region
from earth2studio.models.utils import recommended_sources, step_interval
from earth2studio.perturbation import Perturbation
from earth2studio.run.runner import Source
from earth2studio.utils.type import CoordinateSystem

from ..data import resolve_source
from ..models import load_from_node

COMPONENTS = ("goes", "mrms")
"""Component names of the coupled runner, which also name its streams."""


def _default_source(model: Any, slot: int, label: str) -> Source:
    """Return the model's recommended source for input slot ``slot``."""
    recommended = recommended_sources(model)
    if isinstance(recommended, tuple):
        recommended = recommended[slot] if slot < len(recommended) else None
    elif slot:
        recommended = None
    if recommended is None:
        raise ValueError(
            f"StormScope {label} needs a source; the model recommends none"
        )
    return cast(Source, recommended)


def coupled_step(*models: Any) -> np.timedelta64:
    """Return the step of a runner coupling ``models``.

    One step of the coupled runner is the shortest interval that advances
    every model a whole number of its own steps.  Models sharing a cadence, as
    the StormScope pair does, give that cadence, so ``nsteps`` keeps counting
    model steps.

    Parameters
    ----------
    *models : Any
        Stateful models declaring ``input_coords`` and ``output_coords``.

    Returns
    -------
    np.timedelta64
        The coupled step at nanosecond resolution.
    """
    steps = []
    for model in models:
        inputs = cast(CoordinateSystem, model.input_coords())
        outputs = cast(CoordinateSystem, model.output_coords(inputs))
        interval = step_interval(
            inputs["lead_time"].values, outputs["lead_time"].values
        )
        steps.append(int(np.timedelta64(interval, "ns").astype("int64")))
    return np.timedelta64(math.lcm(*steps), "ns")


def couple(
    goes: Any,
    mrms: Any,
    goes_source: Source | None = None,
    mrms_source: Source | None = None,
    conditioning_source: Source | None = None,
    goes_forcing_source: Source | None = None,
    seed: int = 0,
    perturbation: Mapping[str, Perturbation] | None = None,
) -> CoupledRunner:
    """Couple a GOES model and an MRMS model into one runner.

    Parameters
    ----------
    goes : StormScopeGOES
        GOES model; an unforced variant, or a conditioned variant that reads
        its forcing from ``goes_forcing_source``.
    mrms : StormScopeMRMS
        Radar model conditioned on GOES; its grid must equal the GOES grid.
    goes_source : DataSource, optional
        GOES observation history on the model grid, by default the GOES
        model's recommendation
    mrms_source : DataSource, optional
        Radar (and lightning) history on the model grid, by default the MRMS
        model's recommendation
    conditioning_source : DataSource, optional
        GOES history for the radar model's conditioning slot.  The connector
        feeds that slot from the GOES model, so this source only serves frames
        at or before the initial time that the GOES model does not hold, as
        with a radar history longer than the GOES one; by default
        ``goes_source``
    goes_forcing_source : DataSource | ForecastSource, optional
        Forcing of a conditioned GOES variant, by default its recommendation
    seed : int, optional
        Base seed for per-member seeding, offset per model, by default 0
    perturbation : Mapping[str, Perturbation], optional
        Initial-condition perturbation per model (``goes``, ``mrms``), drawn
        for each member alone, by default none

    Returns
    -------
    CoupledRunner
        Runner yielding the ``goes`` and ``mrms`` streams.  Its ``step`` is
        the coupled step of :func:`coupled_step`.
    """
    if goes_source is None:
        goes_source = _default_source(goes, 0, "GOES")
    if mrms_source is None:
        mrms_source = _default_source(mrms, 0, "MRMS")
    if conditioning_source is None:
        conditioning_source = goes_source
    goes_slots: list[Source] = [goes_source]
    if goes.forcing_coords() is not None:
        if goes_forcing_source is None:
            goes_forcing_source = _default_source(goes, 1, "GOES forcing")
        goes_slots.append(goes_forcing_source)
    runner = CoupledRunner(
        {"goes": goes, "mrms": mrms},
        [Connector("goes", "mrms")],
        sources={"goes": tuple(goes_slots), "mrms": (mrms_source, conditioning_source)},
        seed=seed,
        perturbation=perturbation,
    )
    runner.step = coupled_step(goes, mrms)  # type: ignore[attr-defined]
    return runner


def _region(node: Any) -> Region | None:
    """Build a :class:`~earth2studio.grids.Region` from ``cfg.model.region``."""
    if not node:
        return None
    return Region(
        kind=str(node["kind"]), params=tuple(float(p) for p in node["params"])
    )


_SLOT_KEYS = ("ic_source", "conditioning_source", "forcing_source", "perturbation")


def _model_node(node: DictConfig) -> dict[str, Any]:
    """The model-loading part of a StormScope node, without its slot settings."""
    return {
        k: v
        for k, v in OmegaConf.to_container(node, resolve=True).items()
        if k not in _SLOT_KEYS
    }


def _perturbations(cfg: DictConfig, node: DictConfig) -> dict[str, Perturbation]:
    """Per-model initial-condition perturbations of an ensemble campaign.

    The builder instantiates ``cfg.model.goes.perturbation`` and
    ``cfg.model.mrms.perturbation`` when ``ensemble_size`` exceeds one, as the
    ``forecast`` builder treats ``cfg.perturbation``; deterministic runs are
    never perturbed.
    """
    if (cfg.get("ensemble_size", 1) or 1) <= 1:
        return {}
    return {
        name: hydra.utils.instantiate(node[name].perturbation)
        for name in COMPONENTS
        if node[name].get("perturbation")
    }


def stormscope(cfg: DictConfig, prefer_predownloaded: bool = True) -> CoupledRunner:
    """Build the coupled GOES to MRMS runner from ``cfg.model.{goes, mrms}``.

    Each model node loads through :func:`src.models.load_from_node` (its
    ``load_args`` such as ``model_name``, ``amp`` and ``compile`` pass to
    ``load_model``).  Both models take an optional ``cfg.model.region`` box
    (``{kind: box, params: [west, south, east, north]}``).  Sources resolve
    per role: ``goes`` and ``mrms`` from each node's ``ic_source``,
    ``mrms_conditioning`` from ``cfg.model.mrms.conditioning_source`` and
    ``goes_forcing`` from ``cfg.model.goes.forcing_source``; an unset node
    falls back to the predownloaded store, then the model default.  The GOES
    model feeds the radar model's conditioning slot, so without a configured
    conditioning source the GOES source serves any history frame the GOES
    model does not hold, and the roles omit ``mrms_conditioning``.  For
    ensembles, ``cfg.model.<name>.perturbation`` perturbs that model's initial
    condition.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config.
    prefer_predownloaded : bool, optional
        Read predownloaded stores when present, by default True

    Returns
    -------
    CoupledRunner
        Runner yielding the ``goes`` and ``mrms`` streams, carrying
        ``source_roles`` and the composed ``verification_source``.
    """
    node = cfg.model
    if "goes" not in node or "mrms" not in node:
        raise ValueError(
            "The stormscope runner needs cfg.model.goes and cfg.model.mrms nodes; "
            "see cfg/model/stormscope_goes_mrms.yaml"
        )
    goes = load_from_node(_model_node(node.goes), "StormScope GOES model")
    mrms = load_from_node(_model_node(node.mrms), "StormScope MRMS model")
    region = _region(node.get("region"))
    if region is not None:
        logger.info(f"StormScope: restricting both models to region {region}")
        goes = goes.with_region(region)
        mrms = mrms.with_region(region)

    def resolve(role: str, live: Any) -> Any:
        return resolve_source(
            cfg, role, live=live, prefer_predownloaded=prefer_predownloaded
        )

    roles: dict[str, Any] = {
        "goes": resolve("goes", node.goes.get("ic_source")),
        "mrms": resolve("mrms", node.mrms.get("ic_source")),
    }
    for name in COMPONENTS:
        if roles[name] is None:
            roles[name] = _default_source(
                goes if name == "goes" else mrms, 0, name.upper()
            )
    conditioning = resolve("mrms_conditioning", node.mrms.get("conditioning_source"))
    if conditioning is not None:
        roles["mrms_conditioning"] = conditioning
    else:
        logger.info(
            "Slot 'mrms_conditioning': fed by the GOES model; the goes source "
            "serves any history frame it does not hold"
        )
    forcing = None
    if goes.forcing_coords() is not None:
        forcing = resolve("goes_forcing", node.goes.get("forcing_source"))
        if forcing is None:
            forcing = _default_source(goes, 1, "GOES forcing")
        roles["goes_forcing"] = forcing

    runner = couple(
        goes,
        mrms,
        goes_source=roles["goes"],
        mrms_source=roles["mrms"],
        conditioning_source=conditioning,
        goes_forcing_source=forcing,
        seed=int(cfg.get("random_seed", 42)),
        perturbation=_perturbations(cfg, node),
    )
    runner.source_roles = roles  # type: ignore[attr-defined]
    runner.verification_source = _verification_source(  # type: ignore[attr-defined]
        cfg, node, goes, mrms
    )
    return runner


def _verification_source(
    cfg: DictConfig, node: DictConfig, goes: Any, mrms: Any
) -> Source:
    """Compose the live GOES and MRMS observations over both models' outputs.

    Verification covers every forecast valid time, so it reads the live
    sources (the configured ``ic_source`` nodes or the models' defaults), not
    the predownloaded history stores.
    """

    def live(role: str, model: Any, slot_node: Any, label: str) -> Any:
        source = resolve_source(cfg, role, live=slot_node, prefer_predownloaded=False)
        return source if source is not None else _default_source(model, 0, label)

    def outputs(model: Any) -> list[str]:
        oc = cast(CoordinateSystem, model.output_coords(model.input_coords()))
        return [str(v) for v in oc["variable"].values]

    return CompositeSource(
        {
            "goes": live("goes", goes, node.goes.get("ic_source"), "GOES"),
            "mrms": live("mrms", mrms, node.mrms.get("ic_source"), "MRMS"),
        },
        {"goes": outputs(goes), "mrms": outputs(mrms)},
    )

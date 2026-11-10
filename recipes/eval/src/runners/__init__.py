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

"""Runner builders: from a Hydra config to a :class:`earth2studio.run.Runner`.

``cfg.runner`` names a builder.  Select a built-in by name::

    runner: forecast          # prognostic rollout, members batched per item
    runner: diagnostic        # diagnostics applied to source data
    runner: assimilation      # DA analyses scored directly
    runner: assimilation_forecast   # DA analyses initialize a prognostic
    runner: stormscope        # coupled GOES and MRMS nowcast
    runner:                   # a built-in with options
        name: window_mean
        window: 30D

A custom builder is any importable callable ``(cfg, **options) -> Runner``::

    runner: {_target_: my_pkg.my_runner, option: value}

Every builder returns a runner that also carries ``source_roles``: the role
name of each bound source (``data``, ``forcing0`` ...), which
``predownload.py`` uses to name the stores it writes and
:func:`src.data.resolve_source` to find them again.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.run import DiagnosticRunner, PrognosticRunner, Runner

from ..data import resolve_source
from ..models import load_diagnostics, load_prognostic

BUILTIN_RUNNERS: dict[str, str] = {
    "forecast": "src.runners.forecast",
    "diagnostic": "src.runners.diagnostic",
    "assimilation": "src.runners.assimilation.assimilation",
    "assimilation_forecast": "src.runners.assimilation.assimilation_forecast",
    "stormscope": "src.runners.stormscope.stormscope",
    "window_mean": "src.runners.seasonal.window_mean",
}
"""Built-in runner names and the builder callables they resolve to."""


def build_runner(cfg: DictConfig, prefer_predownloaded: bool = True) -> Runner:
    """Build the campaign's runner from ``cfg.runner``.

    Parameters
    ----------
    cfg : DictConfig
        Full campaign config.
    prefer_predownloaded : bool, optional
        Whether bound sources read predownloaded stores when present, by
        default True.  ``predownload.py`` passes False to bind live sources.

    Returns
    -------
    Runner
        A runner on CPU; :class:`earth2studio.run.Pipeline` moves it.
    """
    spec: Any = cfg.get("runner", "forecast") or "forecast"
    if isinstance(spec, str):
        spec = {"_target_": BUILTIN_RUNNERS.get(spec, spec)}
    else:
        spec = OmegaConf.to_container(spec, resolve=True)
        if not isinstance(spec, dict):
            raise ValueError(f"cfg.runner must be a name or a mapping, got {spec!r}")
        if "name" in spec:
            name = spec.pop("name")
            if name not in BUILTIN_RUNNERS:
                raise ValueError(
                    f"Unknown built-in runner {name!r}; choose from "
                    f"{sorted(BUILTIN_RUNNERS)} or give a _target_"
                )
            spec = {"_target_": BUILTIN_RUNNERS[name], **spec}
        if "_target_" not in spec:
            raise ValueError("cfg.runner mapping needs 'name' or '_target_'")
    builder: Callable[..., Runner] = hydra.utils.instantiate(spec, _partial_=True)
    runner = builder(cfg, prefer_predownloaded=prefer_predownloaded)
    if not hasattr(runner, "source_roles"):
        runner.source_roles = {}  # type: ignore[attr-defined]
    logger.info(f"Built runner {type(runner).__name__}")
    return runner


def _forcing_sources(
    cfg: DictConfig, nslots: int, prefer_predownloaded: bool
) -> tuple[dict[str, Any], tuple[Any, ...] | None]:
    """Resolve one source per forcing slot from ``cfg.forcing_sources``.

    ``forcing_sources`` is a list with one node (or ``null`` for the model's
    recommendation) per forcing slot.  Predownloaded stores take the name
    ``forcing<k>.zarr``.
    """
    nodes = cfg.get("forcing_sources") or [None] * nslots
    if len(nodes) != nslots:
        raise ValueError(
            f"forcing_sources lists {len(nodes)} entries for a model with "
            f"{nslots} forcing slot(s)"
        )
    roles: dict[str, Any] = {}
    sources: list[Any] = []
    for k, node in enumerate(nodes):
        source = resolve_source(
            cfg, f"forcing{k}", live=node, prefer_predownloaded=prefer_predownloaded
        )
        roles[f"forcing{k}"] = source
        sources.append(source)
    if all(s is None for s in sources):
        return roles, None
    if any(s is None for s in sources):
        raise ValueError(
            "forcing_sources must set every slot or none; mixing configured and "
            "recommended forcing sources is not supported"
        )
    return roles, tuple(sources)


def forecast(
    cfg: DictConfig,
    prefer_predownloaded: bool = True,
    model_node: DictConfig | None = None,
    source: Any = None,
) -> PrognosticRunner:
    """Build the prognostic rollout runner.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config: ``model``, optional ``diagnostics``, ``perturbation``
        (applied to ensembles only), ``data_source`` / ``ic_source`` /
        ``forcing_sources`` and ``random_seed``.
    prefer_predownloaded : bool, optional
        Read predownloaded stores when present, by default True
    model_node : DictConfig, optional
        Prognostic node other than ``cfg.model``, by default None
    source : Any, optional
        Initial-condition source replacing the configured one, for builders
        that assemble their own (data assimilation), by default None

    Returns
    -------
    PrognosticRunner
        Runner yielding the ``forecast`` stream plus one per diagnostic.
    """
    prognostic = load_prognostic(cfg, model_node)
    diagnostics = load_diagnostics(cfg)
    roles: dict[str, Any] = {}
    if source is None:
        source = resolve_source(
            cfg,
            "data",
            byo=cfg.get("ic_source"),
            live=cfg.get("data_source"),
            prefer_predownloaded=prefer_predownloaded,
        )
    roles["data"] = source
    forcing_signatures = prognostic.forcing_coords()
    nslots = (
        0
        if forcing_signatures is None
        else len(forcing_signatures) if isinstance(forcing_signatures, tuple) else 1
    )
    forcing: tuple[Any, ...] | None = None
    if nslots:
        forcing_roles, forcing = _forcing_sources(cfg, nslots, prefer_predownloaded)
        roles.update(forcing_roles)
    perturbation = None
    if (cfg.get("ensemble_size", 1) or 1) > 1 and cfg.get("perturbation"):
        perturbation = hydra.utils.instantiate(cfg.perturbation)
    runner = PrognosticRunner(
        prognostic,
        source=source,
        forcing=forcing,
        diagnostics=diagnostics,
        seed=int(cfg.get("random_seed", 42)),
        perturbation=perturbation,
    )
    runner.source_roles = {  # type: ignore[attr-defined]
        "data": runner.source,
        **{f"forcing{k}": runner.forcing[k] for k in range(len(runner.forcing))},
    }
    return runner


def diagnostic(cfg: DictConfig, prefer_predownloaded: bool = True) -> DiagnosticRunner:
    """Build the diagnostic-only runner: diagnostics applied to source data.

    A configured source (``ic_source`` or ``data_source``) feeds every
    diagnostic under the role ``data``.  Without one, each diagnostic reads its
    own recommended source under the role ``data_<stream>``, so
    ``predownload.py`` caches each and inference finds the stores again.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config with a ``diagnostics`` block and the source settings.
    prefer_predownloaded : bool, optional
        Read predownloaded stores when present, by default True

    Returns
    -------
    DiagnosticRunner
        Runner yielding one stream per diagnostic at lead time zero.
    """
    diagnostics = load_diagnostics(cfg)
    if not diagnostics:
        raise ValueError("The diagnostic runner needs at least one 'diagnostics' entry")
    source = resolve_source(
        cfg,
        "data",
        byo=cfg.get("ic_source"),
        live=cfg.get("data_source"),
        prefer_predownloaded=prefer_predownloaded,
    )
    runner = DiagnosticRunner(
        diagnostics, source=source, seed=int(cfg.get("random_seed", 42))
    )
    roles: dict[str, Any] = {}
    if source is not None:
        roles["data"] = source
    else:
        for name, recommended in runner.sources.items():
            role = f"data_{name}"
            runner.sources[name] = resolve_source(
                cfg, role, live=recommended, prefer_predownloaded=prefer_predownloaded
            )
            roles[role] = runner.sources[name]
    runner.source_roles = roles  # type: ignore[attr-defined]
    return runner

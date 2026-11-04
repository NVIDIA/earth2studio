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

"""Model loading from Hydra config nodes.

A model node names an ``architecture`` (an ``AutoModelMixin`` class) and may
add ``package_path`` and ``load_args``; Hydra instantiates ``_target_`` nodes
directly.  Every load runs on rank 0 first, which fills a shared package
cache once before the other ranks read it.
"""

from __future__ import annotations

from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig

from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.run.work import rank0_first


def load_from_node(node: DictConfig | dict, kind: str = "model") -> Any:
    """Load a model described by a config node.

    Parameters
    ----------
    node : DictConfig | dict
        Either a Hydra ``_target_`` block, or a node with ``architecture``
        (fully qualified class name), optional ``package_path`` and optional
        ``load_args`` forwarded to ``load_model``.
    kind : str, optional
        Label for log messages, by default "model"

    Returns
    -------
    Any
        The loaded model, not yet placed on a device.
    """
    if "_target_" in node:
        model = rank0_first(lambda: hydra.utils.instantiate(node))
        logger.success(f"Instantiated {kind}: {type(model).__name__}")
        return model
    if "architecture" not in node:
        raise ValueError(
            f"A {kind} node needs '_target_' or 'architecture'; got keys "
            f"{list(node.keys())}"
        )
    cls = hydra.utils.get_class(node["architecture"])
    load_args: dict[str, Any] = dict(node.get("load_args", {}) or {})

    def load() -> Any:
        # Packages are lazy: the download happens inside load_model, so the
        # whole load runs rank 0 first.
        if node.get("package_path"):
            from earth2studio.models.auto import Package

            package = Package(node["package_path"])
        else:
            package = cls.load_default_package()
        return cls.load_model(package=package, **load_args)

    model = rank0_first(load)
    logger.success(f"Loaded {kind}: {cls.__name__}")
    return model


def load_prognostic(cfg: DictConfig, node: DictConfig | None = None) -> PrognosticModel:
    """Load the prognostic model from ``cfg.model`` or an explicit node."""
    model: PrognosticModel = load_from_node(
        node if node is not None else cfg.model, "prognostic model"
    )
    return model


def load_diagnostics(cfg: DictConfig) -> dict[str, DiagnosticModel]:
    """Load the ``diagnostics`` block as stream name to model.

    Parameters
    ----------
    cfg : DictConfig
        Config with an optional ``diagnostics`` mapping of name to model node.

    Returns
    -------
    dict[str, DiagnosticModel]
        Diagnostics keyed by the stream name the runner yields them under.
    """
    if not cfg.get("diagnostics"):
        return {}
    diagnostics: dict[str, DiagnosticModel] = {}
    for name, node in cfg.diagnostics.items():
        if str(name) == "forecast":
            raise ValueError("Diagnostic stream name 'forecast' is reserved")
        diagnostics[str(name)] = load_from_node(node, f"diagnostic '{name}'")
    return diagnostics


def load_assimilation(node: DictConfig) -> Any:
    """Load a data assimilation model from its config node (``cfg.model.da``)."""
    return load_from_node(node, "assimilation model")

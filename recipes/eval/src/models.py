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

from __future__ import annotations

from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig

from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel

from .distributed import run_on_rank0_first


def load_prognostic(cfg: DictConfig) -> PrognosticModel:
    """Load a prognostic model from the Hydra config.

    The config's ``model`` section must contain an ``architecture`` key whose
    value is the fully-qualified class name of a prognostic model (e.g.
    ``earth2studio.models.px.DLWP``).  The class is expected to expose the
    standard ``load_default_package`` / ``load_model`` classmethods from the
    ``AutoModelMixin`` protocol.

    Any extra keyword arguments under ``model.load_args`` are forwarded to
    ``load_model``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with a ``model`` section.

    Returns
    -------
    PrognosticModel
        Loaded (but not yet device-placed) prognostic model.
    """
    model_cfg = cfg.model
    cls = hydra.utils.get_class(model_cfg.architecture)

    if model_cfg.get("package_path"):
        from earth2studio.models.auto import Package

        pkg = Package(model_cfg.package_path)
    else:
        pkg = run_on_rank0_first(cls.load_default_package)

    load_kwargs: dict[str, Any] = dict(model_cfg.get("load_args", {}))
    model: PrognosticModel = cls.load_model(package=pkg, **load_kwargs)

    logger.success(f"Loaded prognostic model: {cls.__name__}")
    return model


def load_diagnostics(cfg: DictConfig) -> list[DiagnosticModel]:
    """Load diagnostic models listed in the config.

    Each entry under ``diagnostics`` should be a Hydra-instantiable object
    (``_target_`` style) or have an ``architecture`` key following the same
    pattern as the prognostic model.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with an optional ``diagnostics`` section.

    Returns
    -------
    list[DiagnosticModel]
        Loaded diagnostic models (may be empty).
    """
    if "diagnostics" not in cfg:
        return []

    models: list[DiagnosticModel] = []
    for name, dx_cfg in cfg.diagnostics.items():
        if "_target_" in dx_cfg:
            dx = hydra.utils.instantiate(dx_cfg)
        elif "architecture" in dx_cfg:
            cls = hydra.utils.get_class(dx_cfg.architecture)
            pkg = run_on_rank0_first(cls.load_default_package)
            load_kwargs: dict[str, Any] = dict(dx_cfg.get("load_args", {}))
            dx = cls.load_model(package=pkg, **load_kwargs)
        else:
            raise ValueError(
                f"Diagnostic '{name}' must have '_target_' or 'architecture'. "
                f"Got keys: {list(dx_cfg.keys())}"
            )
        models.append(dx)
        logger.success(f"Loaded diagnostic model: {name}")

    return models

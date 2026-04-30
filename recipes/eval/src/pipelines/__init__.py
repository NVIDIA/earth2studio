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

"""Eval recipe inference pipelines."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .base import Pipeline, PredownloadStore
from .dlesym import DLESyMPipeline
from .forecast import DiagnosticPipeline, ForecastPipeline
from .stormscope import StormScopePipeline

__all__ = [
    "DLESyMPipeline",
    "DiagnosticPipeline",
    "ForecastPipeline",
    "Pipeline",
    "PredownloadStore",
    "StormScopePipeline",
    "build_pipeline",
]


def build_pipeline(cfg: DictConfig) -> Pipeline:
    """Instantiate the pipeline declared by ``cfg.pipeline``.

    ``cfg.pipeline`` may be either:

    * A :class:`~omegaconf.DictConfig` with a ``_target_`` entry (standard
      Hydra instantiation; lets callers pass construction kwargs).
    * A string containing a fully qualified class path
      (e.g. ``"src.pipelines.forecast.ForecastPipeline"`` or
      ``"my_pkg.MyPipeline"``), resolved via
      :func:`hydra.utils.get_class` and constructed with no arguments.

    The resulting object must be a :class:`Pipeline` subclass.  Callers
    invoke :meth:`Pipeline.setup` before use.
    """
    spec = cfg.get("pipeline")
    if spec is None:
        raise ValueError(
            "cfg.pipeline is required (fully qualified class path or "
            "{_target_: ...} block)."
        )

    if isinstance(spec, str):
        cls = hydra.utils.get_class(spec)
        if not (isinstance(cls, type) and issubclass(cls, Pipeline)):
            raise TypeError(f"Pipeline '{spec}' must be a subclass of Pipeline.")
        return cls()

    pipeline = hydra.utils.instantiate(spec)
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"cfg.pipeline resolved to {type(pipeline).__name__}, which is "
            "not a Pipeline subclass."
        )
    return pipeline

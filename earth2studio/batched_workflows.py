# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from loguru import logger

__all__ = [
    "DeterministicBatchRequest",
    "DeterministicBatchResponse",
    "DeterministicBatchRuntime",
    "run_deterministic_batch",
]


@dataclass(frozen=True)
class DeterministicBatchRequest:
    """Single deterministic forecast request in a shared-resource batch.

    Parameters
    ----------
    model : str
        Name of the prognostic model to run.
    start_time : str
        Forecast initialization time in ISO 8601 format.
    nsteps : int
        Number of forecast steps to execute.
    output_path : str | Path
        Path where the request's forecast dataset will be written. The path must not
        already exist.
    run_id : str | None, optional
        Identifier used to correlate request logs and results, by default None
    """

    model: str
    start_time: str
    nsteps: int
    output_path: str | Path
    run_id: str | None = None


@dataclass(frozen=True)
class DeterministicBatchResponse:
    """Per-request outcome from a deterministic forecast batch.

    Parameters
    ----------
    model : str
        Name of the prognostic model used for the request.
    start_time : str
        Forecast initialization time in ISO 8601 format.
    nsteps : int
        Number of forecast steps requested.
    dataset_path : str
        Path assigned to the request's forecast dataset.
    status : Literal["succeeded", "failed"], optional
        Request outcome, by default "succeeded"
    error : str | None, optional
        Failure details when ``status`` is ``"failed"``, by default None
    """

    model: str
    start_time: str
    nsteps: int
    dataset_path: str
    status: Literal["succeeded", "failed"] = "succeeded"
    error: str | None = None


ModelLoader = Callable[[str], Any]
DataFactory = Callable[[], Any]
ForecastRunner = Callable[[DeterministicBatchRequest, Any, Any, Any], None]


class DeterministicBatchRuntime:
    """Reusable deterministic inference resources for grouped requests.

    This runtime deliberately has no dependency on the Earth2Studio serve stack.
    Callers such as PhysicsNeMo-Serve own queueing, grouping, scheduling, and
    output registration; this object owns model/data setup and forecast calls.

    Parameters
    ----------
    device : Any | None, optional
        Device used for model inference. When None, CUDA is selected when available
        and CPU otherwise, by default None
    model_loader : ModelLoader | None, optional
        Callable that loads a model from a normalized model name. Uses the built-in
        DLWP loader when None, by default None
    data_factory : DataFactory | None, optional
        Callable that creates the shared data source. Uses GFS when None, by default
        None.
    runner : ForecastRunner | None, optional
        Callable that executes one request with the shared model, data source, and
        device. Uses the built-in deterministic workflow when None, by default None
    """

    def __init__(
        self,
        *,
        device: Any | None = None,
        model_loader: ModelLoader | None = None,
        data_factory: DataFactory | None = None,
        runner: ForecastRunner | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self._model_loader = model_loader or _load_default_deterministic_model
        self._data_factory = data_factory or _load_default_data_source
        self._runner = runner
        self._loaded_model_name: str | None = None
        self._model: Any | None = None
        self._data: Any | None = None

    def _ensure_loaded(self, model_name: str) -> tuple[Any, Any]:
        normalized_model = _normalize_model_name(model_name)
        if self._loaded_model_name is not None:
            if normalized_model != self._loaded_model_name:
                raise ValueError(
                    "DeterministicBatchRuntime can only hold one model at a time; "
                    f"loaded {self._loaded_model_name!r}, requested {normalized_model!r}"
                )
            if self._model is not None and self._data is not None:
                return self._model, self._data

        model = self._model_loader(normalized_model)
        model = model.to(self.device)
        data = self._data_factory()
        self._model = model
        self._data = data
        self._loaded_model_name = normalized_model
        return self._model, self._data

    def close(self) -> None:
        """Release the model and data resources cached by this runtime.

        The runtime remains reusable after closing. Its model and data resources are
        loaded again when the next request runs.
        """

        self._model = None
        self._data = None
        self._loaded_model_name = None

    def run(self, request: DeterministicBatchRequest) -> DeterministicBatchResponse:
        """Run one deterministic forecast using the cached resources.

        Parameters
        ----------
        request : DeterministicBatchRequest
            Deterministic forecast request to execute.

        Returns
        -------
        DeterministicBatchResponse
            Request outcome. Execution errors produce failed responses.
        """
        logger.info(
            "Earth2 deterministic request start run_id={} model={} start_time={} nsteps={} output_path={}",
            request.run_id,
            request.model,
            request.start_time,
            request.nsteps,
            request.output_path,
        )
        final_path = Path(request.output_path)
        staging_path = final_path.with_name(
            f".{final_path.name}.tmp-{uuid.uuid4().hex}"
        )
        staged_request = replace(request, output_path=staging_path)
        try:
            if final_path.exists():
                raise FileExistsError(f"output path already exists: {final_path}")

            model, data = self._ensure_loaded(request.model)
            if self._runner is not None:
                self._runner(staged_request, model, data, self.device)
            else:
                _run_default_deterministic_forecast(
                    request=staged_request,
                    model=model,
                    data=data,
                    device=self.device,
                )
            os.replace(staging_path, final_path)
        except Exception as exc:
            shutil.rmtree(staging_path, ignore_errors=True)
            logger.exception(
                "Earth2 deterministic request failed run_id={} model={} start_time={} nsteps={}",
                request.run_id,
                request.model,
                request.start_time,
                request.nsteps,
            )
            return DeterministicBatchResponse(
                model=request.model,
                start_time=request.start_time,
                nsteps=request.nsteps,
                dataset_path=str(request.output_path),
                status="failed",
                error=str(exc),
            )

        logger.info(
            "Earth2 deterministic request succeeded run_id={} model={} start_time={} nsteps={} output_path={}",
            request.run_id,
            request.model,
            request.start_time,
            request.nsteps,
            request.output_path,
        )
        return DeterministicBatchResponse(
            model=request.model,
            start_time=request.start_time,
            nsteps=request.nsteps,
            dataset_path=str(request.output_path),
        )


def run_deterministic_batch(
    requests: Sequence[DeterministicBatchRequest],
    *,
    runtime: DeterministicBatchRuntime | None = None,
    device: Any | None = None,
) -> list[DeterministicBatchResponse]:
    """Run deterministic forecast requests with shared model and data resources.

    Parameters
    ----------
    requests : Sequence[DeterministicBatchRequest]
        Compatible deterministic forecast requests to execute sequentially.
    runtime : DeterministicBatchRuntime | None, optional
        Runtime that owns shared inference resources. A runtime is created when None,
        by default None
    device : Any | None, optional
        Device for a newly created runtime. Ignored when ``runtime`` is supplied, by
        default None

    Returns
    -------
    list[DeterministicBatchResponse]
        Per-request outcomes in input order. An empty request sequence returns an
        empty list.
    """
    if not requests:
        return []

    batch_runtime = runtime or DeterministicBatchRuntime(device=device)
    return [batch_runtime.run(request) for request in requests]


def _normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().lower()
    if not normalized:
        raise ValueError("model name cannot be empty")
    return normalized


def _resolve_device(device: Any | None) -> Any:
    if device is not None:
        return device

    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_default_deterministic_model(model_name: str) -> Any:
    if model_name != "dlwp":
        raise ValueError("default deterministic batching supports only model='dlwp'")

    from earth2studio.models.px import DLWP

    package = DLWP.load_default_package()
    return DLWP.load_model(package)


def _load_default_data_source() -> Any:
    from earth2studio.data import GFS

    return GFS()


def _run_default_deterministic_forecast(
    *,
    request: DeterministicBatchRequest,
    model: Any,
    data: Any,
    device: Any,
) -> None:
    from earth2studio.io import ZarrBackend
    from earth2studio.run import deterministic
    from earth2studio.utils.time import to_time_array

    deterministic(
        time=to_time_array([request.start_time]),
        nsteps=request.nsteps,
        prognostic=model,
        data=data,
        io=ZarrBackend(str(request.output_path)),
        device=device,
    )

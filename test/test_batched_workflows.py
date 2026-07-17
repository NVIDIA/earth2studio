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

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from earth2studio.batched_workflows import (
    DeterministicBatchRequest,
    DeterministicBatchRuntime,
    run_deterministic_batch,
)


class FakeModel:
    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self.calls = calls

    def to(self, device: Any) -> FakeModel:
        self.calls.append(("model.to", device))
        return self


def test_run_deterministic_batch_reuses_runtime_resources(tmp_path: Path) -> None:
    calls: list[tuple[Any, ...]] = []

    def model_loader(model_name: str) -> FakeModel:
        calls.append(("load_model", model_name))
        return FakeModel(calls)

    def data_factory() -> object:
        calls.append(("load_data", None))
        return object()

    def runner(
        request: DeterministicBatchRequest,
        _model: object,
        _data: object,
        device: Any,
    ) -> None:
        Path(request.output_path).mkdir(parents=True)
        calls.append(("run", request.run_id, device))

    runtime = DeterministicBatchRuntime(
        device="cpu",
        model_loader=model_loader,
        data_factory=data_factory,
        runner=runner,
    )
    requests = [
        DeterministicBatchRequest(
            model="custom",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            output_path=tmp_path / f"forecast-{index}.zarr",
            run_id=f"run-{index}",
        )
        for index in range(2)
    ]

    responses = run_deterministic_batch(requests, runtime=runtime)

    assert [response.status for response in responses] == ["succeeded", "succeeded"]
    assert calls == [
        ("load_model", "custom"),
        ("model.to", "cpu"),
        ("load_data", None),
        ("run", "run-0", "cpu"),
        ("run", "run-1", "cpu"),
    ]
    assert [response.dataset_path for response in responses] == [
        str(tmp_path / "forecast-0.zarr"),
        str(tmp_path / "forecast-1.zarr"),
    ]
    assert all(Path(response.dataset_path).is_dir() for response in responses)
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_run_deterministic_batch_uses_default_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import earth2studio.data as data_module
    import earth2studio.io as io_module
    import earth2studio.models.px as model_module
    import earth2studio.run as run_module
    import earth2studio.utils.time as time_module

    output_path = tmp_path / "default.zarr"
    request = DeterministicBatchRequest(
        model="dlwp",
        start_time="2026-01-01T00:00:00Z",
        nsteps=2,
        output_path=output_path,
    )
    package = object()
    data = object()
    converted_time = object()
    model = FakeModel([])
    load_package = Mock(return_value=package)
    load_model = Mock(return_value=model)
    load_data = Mock(return_value=data)
    to_time_array = Mock(return_value=converted_time)

    def deterministic(**kwargs: Any) -> None:
        Path(kwargs["io"]).mkdir(parents=True)

    forecast = Mock(side_effect=deterministic)

    monkeypatch.setattr(model_module.DLWP, "load_default_package", load_package)
    monkeypatch.setattr(model_module.DLWP, "load_model", load_model)
    monkeypatch.setattr(data_module, "GFS", load_data)
    monkeypatch.setattr(time_module, "to_time_array", to_time_array)
    monkeypatch.setattr(io_module, "ZarrBackend", Path)
    monkeypatch.setattr(run_module, "deterministic", forecast)

    response = run_deterministic_batch([request], device="cpu")[0]

    assert response.status == "succeeded"
    assert response.dataset_path == str(output_path)
    assert model.calls == [("model.to", "cpu")]
    load_package.assert_called_once_with()
    load_model.assert_called_once_with(package)
    load_data.assert_called_once_with()
    to_time_array.assert_called_once_with([request.start_time])
    staging_path = forecast.call_args.kwargs["io"]
    assert staging_path.parent == output_path.parent
    assert staging_path.name.startswith(f".{output_path.name}.tmp-")
    forecast.assert_called_once_with(
        time=converted_time,
        nsteps=2,
        prognostic=model,
        data=data,
        io=staging_path,
        device="cpu",
    )
    assert output_path.is_dir()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_run_deterministic_batch_returns_item_failure(tmp_path: Path) -> None:
    def model_loader(_model_name: str) -> FakeModel:
        return FakeModel([])

    def runner(
        request: DeterministicBatchRequest,
        _model: object,
        _data: object,
        _device: Any,
    ) -> None:
        output_path = Path(request.output_path)
        output_path.mkdir(parents=True)
        (output_path / "data").write_text(str(request.run_id))
        if request.run_id == "bad":
            raise RuntimeError("boom")

    runtime = DeterministicBatchRuntime(
        device="cpu",
        model_loader=model_loader,
        data_factory=object,
        runner=runner,
    )

    responses = run_deterministic_batch(
        [
            DeterministicBatchRequest(
                model="dlwp",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
                output_path=tmp_path / "good.zarr",
                run_id="good",
            ),
            DeterministicBatchRequest(
                model="dlwp",
                start_time="2026-01-01T00:00:00Z",
                nsteps=1,
                output_path=tmp_path / "bad.zarr",
                run_id="bad",
            ),
        ],
        runtime=runtime,
    )

    assert responses[0].status == "succeeded"
    assert responses[1].status == "failed"
    assert responses[1].error is not None
    assert "boom" in responses[1].error
    assert (tmp_path / "good.zarr" / "data").read_text() == "good"
    assert not (tmp_path / "bad.zarr").exists()
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_run_deterministic_batch_preserves_existing_output(tmp_path: Path) -> None:
    output_path = tmp_path / "existing.zarr"
    output_path.mkdir()
    marker = output_path / "marker"
    marker.write_text("original")
    runner_called = False

    def runner(
        _request: DeterministicBatchRequest,
        _model: object,
        _data: object,
        _device: Any,
    ) -> None:
        nonlocal runner_called
        runner_called = True

    runtime = DeterministicBatchRuntime(device="cpu", runner=runner)
    response = runtime.run(
        DeterministicBatchRequest(
            model="dlwp",
            start_time="2026-01-01T00:00:00Z",
            nsteps=1,
            output_path=output_path,
        )
    )

    assert response.status == "failed"
    assert response.error == f"output path already exists: {output_path}"
    assert marker.read_text() == "original"
    assert not runner_called
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_runtime_does_not_commit_partial_state_on_data_failure() -> None:
    calls: list[tuple[Any, ...]] = []
    data_load_attempt = 0

    def model_loader(model_name: str) -> FakeModel:
        calls.append(("load_model", model_name))
        return FakeModel(calls)

    def data_factory() -> object:
        nonlocal data_load_attempt
        data_load_attempt += 1
        calls.append(("load_data", data_load_attempt))
        if data_load_attempt == 1:
            raise RuntimeError("data boom")
        return object()

    runtime = DeterministicBatchRuntime(
        device="cpu",
        model_loader=model_loader,
        data_factory=data_factory,
    )

    with pytest.raises(RuntimeError, match="data boom"):
        runtime._ensure_loaded("dlwp")

    assert runtime._model is None
    assert runtime._data is None
    assert runtime._loaded_model_name is None

    model, data = runtime._ensure_loaded("dlwp")

    assert runtime._model is model
    assert runtime._data is data
    assert runtime._loaded_model_name == "dlwp"
    assert calls == [
        ("load_model", "dlwp"),
        ("model.to", "cpu"),
        ("load_data", 1),
        ("load_model", "dlwp"),
        ("model.to", "cpu"),
        ("load_data", 2),
    ]

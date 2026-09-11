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

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import main as recipe
import numpy as np
import pytest
import torch


def _config(tmp_path: Path, *, members: int) -> recipe.ForecastConfig:
    """Create a minimal test configuration."""
    return recipe.ForecastConfig(
        issue_time=datetime(2020, 6, 3),
        nsteps=3,
        members=members,
        batch_size=1,
        variables=("t2m", "tp"),
        output=tmp_path / "forecast.zarr",
        overwrite=False,
        cache=True,
        async_timeout=90,
        device="cuda:0",
    )


def _mock_components(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock]:
    """Replace heavyweight model, data, and IO construction."""
    model = MagicMock(name="model")
    package = MagicMock(name="package")
    source = MagicMock(name="source")
    data = MagicMock(name="data")
    io = MagicMock(name="io")
    model.input_coords.return_value = {
        "variable": np.array(["t2m", "z500", "msl", "tp"])
    }

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(recipe.FuXiS2S, "load_default_package", lambda: package)
    monkeypatch.setattr(recipe.FuXiS2S, "load_model", lambda value: model)
    arco = MagicMock(return_value=source)
    adapter = MagicMock(return_value=data)
    backend = MagicMock(return_value=io)
    monkeypatch.setattr(recipe, "ARCO_ERA5", arco)
    monkeypatch.setattr(recipe, "FuXiS2SERA5", adapter)
    monkeypatch.setattr(recipe, "ZarrBackend", backend)
    return {
        "model": model,
        "source": source,
        "data": data,
        "io": io,
        "arco": arco,
        "adapter": adapter,
        "backend": backend,
    }


def test_parse_args_converts_issue_time_and_member_count() -> None:
    config = recipe.parse_args(
        [
            "--issue-time",
            "2020-06-03T00:00:00+00:00",
            "--nsteps",
            "42",
            "--members",
            "11",
            "--batch-size",
            "2",
            "--variables",
            "t2m",
            "tp",
        ]
    )

    assert config.issue_time == datetime(2020, 6, 3)
    assert config.workflow_time == datetime(2020, 6, 2)
    assert config.nsteps == 42
    assert config.members == 11
    assert config.batch_size == 2
    assert config.variables == ("t2m", "tp")


@pytest.mark.parametrize(
    "argv",
    [
        ["--members", "0"],
        ["--nsteps", "43"],
        ["--members", "2", "--batch-size", "3"],
        ["--variables", "all", "tp"],
        ["--variables", "tp", "tp"],
        ["--issue-time", "2020-06-03T06:00:00Z"],
    ],
)
def test_parse_args_rejects_invalid_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        recipe.parse_args(argv)


def test_single_member_uses_deterministic_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _mock_components(monkeypatch)
    model = components["model"]
    io = components["io"]
    deterministic = MagicMock(return_value=io)
    ensemble = MagicMock()
    monkeypatch.setattr(recipe.run, "deterministic", deterministic)
    monkeypatch.setattr(recipe.run, "ensemble", ensemble)

    result = recipe.run_forecast(_config(tmp_path, members=1))

    assert result is io
    deterministic.assert_called_once()
    ensemble.assert_not_called()
    kwargs = deterministic.call_args.kwargs
    assert kwargs["time"] == ["2020-06-02T00:00:00"]
    assert kwargs["nsteps"] == 3
    assert kwargs["prognostic"] is model
    assert kwargs["data"] is components["data"]
    assert kwargs["io"] is io
    np.testing.assert_array_equal(kwargs["output_coords"]["variable"], ["t2m", "tp"])
    components["arco"].assert_called_once_with(
        cache=True,
        verbose=True,
        async_timeout=90,
    )
    components["adapter"].assert_called_once_with(components["source"])
    io.root.attrs.update.assert_called_once_with(
        {
            "forecast_issue_time_utc": "2020-06-03T00:00:00Z",
            "latest_complete_daily_mean_utc": "2020-06-02T00:00:00Z",
            "ensemble_members": 1,
            "forecast_steps": 3,
        }
    )


def test_multiple_members_use_zero_perturbation_ensemble(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _mock_components(monkeypatch)
    model = components["model"]
    io = components["io"]
    deterministic = MagicMock()
    ensemble = MagicMock(return_value=io)
    zero = MagicMock(name="zero")
    monkeypatch.setattr(recipe.run, "deterministic", deterministic)
    monkeypatch.setattr(recipe.run, "ensemble", ensemble)
    monkeypatch.setattr(recipe, "Zero", MagicMock(return_value=zero))

    result = recipe.run_forecast(_config(tmp_path, members=5))

    assert result is io
    deterministic.assert_not_called()
    ensemble.assert_called_once()
    kwargs = ensemble.call_args.kwargs
    assert kwargs["nensemble"] == 5
    assert kwargs["batch_size"] == 1
    assert kwargs["perturbation"] is zero
    assert kwargs["prognostic"] is model
    assert kwargs["data"] is components["data"]


def test_invalid_output_variable_fails_before_arco_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    components = _mock_components(monkeypatch)
    config = replace(_config(tmp_path, members=1), variables=("t2",))

    with pytest.raises(ValueError, match="Unknown FuXi-S2S output variables: t2"):
        recipe.run_forecast(config)

    components["arco"].assert_not_called()
    components["adapter"].assert_not_called()


def test_existing_output_requires_explicit_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, members=1)
    config.output.mkdir()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(FileExistsError, match="--overwrite"):
        recipe.run_forecast(config)


def test_all_variables_omits_output_filter() -> None:
    assert not recipe._output_coords(None)

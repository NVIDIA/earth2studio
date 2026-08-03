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

import io
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.px import FuXiS2S
from earth2studio.models.px.fuxi_s2s import (
    VARIABLES,
    _extract_tar_member,
)


class PhooFuXiS2S(torch.nn.Module):
    """Small deterministic stand-in for the FuXi-S2S ONNX graph."""

    def forward(
        self,
        x: torch.Tensor,
        step: torch.Tensor,
        day_of_year: torch.Tensor,
    ) -> torch.Tensor:
        delta = (step + day_of_year).reshape(1, 1, 1, 1, 1)
        prediction = x[:, -1:] + delta
        return torch.cat((x[:, -1:], prediction), dim=1)


class PhooStochasticSession:
    """Small ORT stand-in that returns a distinct sample on every call."""

    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list[Any]:
        return [
            type("OrtValue", (), {"name": name})() for name in ("input", "step", "doy")
        ]

    def get_outputs(self) -> list[Any]:
        return [type("OrtValue", (), {"name": "output"})()]

    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        self.calls += 1
        prediction = inputs["input"][:, -1:] + self.calls
        return [np.concatenate((inputs["input"][:, -1:], prediction), axis=1)]


@pytest.fixture(scope="module")
def fuxi_s2s_test_package(tmp_path_factory) -> Package:
    tmp_path = tmp_path_factory.mktemp("fuxi_s2s")
    torch.onnx.export(
        PhooFuXiS2S(),
        (
            torch.ones(1, 2, len(VARIABLES), 121, 240),
            torch.zeros(1),
            torch.ones(1),
        ),
        str(tmp_path / "fuxi_s2s.onnx"),
        input_names=["input", "step", "doy"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    (tmp_path / "fuxi_s2s").touch()
    return Package(str(tmp_path))


def _identity_model() -> FuXiS2S:
    return FuXiS2S("")


def test_fuxi_s2s_coords() -> None:
    model = _identity_model()
    input_coords = model.input_coords()
    input_coords["batch"] = np.array([0])
    input_coords["time"] = np.array([np.datetime64("2020-06-02")])

    output_coords = model.output_coords(input_coords)

    assert len(input_coords["variable"]) == 76
    assert input_coords["lead_time"].tolist() == [
        np.timedelta64(-1, "D"),
        np.timedelta64(0, "D"),
    ]
    assert output_coords["lead_time"].tolist() == [np.timedelta64(1, "D")]
    np.testing.assert_allclose(input_coords["lat"], np.linspace(90, -90, 121))
    np.testing.assert_allclose(
        input_coords["lon"], np.linspace(0, 360, 240, endpoint=False)
    )


def test_fuxi_s2s_unit_conversions() -> None:
    model = _identity_model()
    x = torch.ones(2, len(VARIABLES), 2, 3)
    x[:, VARIABLES.index("ttr")] = 7200.0
    x[:, VARIABLES.index("tp")] = 0.001
    x[0, VARIABLES.index("tp"), 0, 0] = torch.nan
    x[0, VARIABLES.index("tp"), 0, 1] = -0.001
    x[0, VARIABLES.index("tp"), 0, 2] = 2.0
    x[:, VARIABLES.index("sst"), 0, 0] = torch.nan

    model_input = model._prepare_input(x)

    torch.testing.assert_close(
        model_input[:, VARIABLES.index("ttr")],
        torch.full((2, 2, 3), 2.0),
    )
    expected_tp = torch.ones(2, 2, 3)
    expected_tp[0, 0] = torch.tensor([0.0, 0.0, 1000.0])
    torch.testing.assert_close(model_input[:, VARIABLES.index("tp")], expected_tp)
    assert torch.isnan(model_input[:, VARIABLES.index("sst"), 0, 0]).all()

    model_output = torch.ones_like(x)
    model_output[:, VARIABLES.index("ttr")] = 2.0
    model_output[:, VARIABLES.index("tp")] = 1.0
    output = model._prepare_output(model_output)

    torch.testing.assert_close(
        output[:, VARIABLES.index("ttr")],
        torch.full((2, 2, 3), 7200.0),
    )
    torch.testing.assert_close(
        output[:, VARIABLES.index("tp")],
        torch.full((2, 2, 3), 0.001),
    )


def test_fuxi_s2s_call(fuxi_s2s_test_package) -> None:
    model = FuXiS2S.load_model(fuxi_s2s_test_package)
    coords = model.input_coords()
    del coords["batch"]
    coords["time"] = np.array([np.datetime64("2020-01-01")])
    x = torch.ones(1, 2, len(VARIABLES), 121, 240)
    x[:, :, VARIABLES.index("ttr")] = 3600.0
    x[:, :, VARIABLES.index("tp")] = 0.001

    output, output_coords = model(x, coords)

    expected_model_output = model._prepare_input(x[:, -1:]) + 1.0 / 365.0
    expected = model._prepare_output(expected_model_output)
    torch.testing.assert_close(output, expected)
    assert output.shape == (1, 1, len(VARIABLES), 121, 240)
    assert output_coords["lead_time"].tolist() == [np.timedelta64(1, "D")]


def test_fuxi_s2s_call_invalid_coords(fuxi_s2s_test_package) -> None:
    model = FuXiS2S.load_model(fuxi_s2s_test_package)
    coords = model.input_coords()
    del coords["batch"]
    coords["time"] = np.array([np.datetime64("2020-01-01")])
    coords["lat"] = coords["lat"][::-1]
    x = torch.ones(1, 2, len(VARIABLES), 121, 240)

    with pytest.raises(ValueError):
        model(x, coords)


def test_fuxi_s2s_iter(fuxi_s2s_test_package) -> None:
    model = FuXiS2S.load_model(fuxi_s2s_test_package)
    coords = model.input_coords()
    del coords["batch"]
    coords["time"] = np.array([np.datetime64("2020-01-01")])
    coords["ensemble"] = np.arange(2)
    coords.move_to_end("ensemble", last=False)
    x = torch.ones(2, 1, 2, len(VARIABLES), 121, 240)
    model_iterator = model.create_iterator(x, coords)

    initial, initial_coords = next(model_iterator)
    first, first_coords = next(model_iterator)
    second, second_coords = next(model_iterator)

    assert initial.shape == first.shape == second.shape
    assert initial.shape == (2, 1, 1, len(VARIABLES), 121, 240)
    np.testing.assert_array_equal(initial_coords["ensemble"], np.arange(2))
    np.testing.assert_array_equal(first_coords["ensemble"], np.arange(2))
    np.testing.assert_array_equal(second_coords["ensemble"], np.arange(2))
    assert initial_coords["lead_time"].tolist() == [np.timedelta64(0, "D")]
    assert first_coords["lead_time"].tolist() == [np.timedelta64(1, "D")]
    assert second_coords["lead_time"].tolist() == [np.timedelta64(2, "D")]
    torch.testing.assert_close(
        first[..., 0, :, :],
        torch.full_like(first[..., 0, :, :], 1.0 + 1.0 / 365.0),
    )
    torch.testing.assert_close(
        second[..., 0, :, :],
        torch.full_like(second[..., 0, :, :], 2.0 + 3.0 / 365.0),
    )


def test_fuxi_s2s_ensemble_members_use_independent_ort_calls() -> None:
    model = _identity_model()
    session = PhooStochasticSession()
    model.ort = session  # type: ignore[assignment]
    coords = model.input_coords()
    del coords["batch"]
    coords["time"] = np.array([np.datetime64("2020-01-01")])
    coords["ensemble"] = np.arange(2)
    coords.move_to_end("ensemble", last=False)
    x = torch.ones(2, 1, 2, len(VARIABLES), 121, 240)

    iterator = model.create_iterator(x, coords)
    next(iterator)
    prediction, prediction_coords = next(iterator)

    assert session.calls == 2
    assert prediction.shape == (2, 1, 1, len(VARIABLES), 121, 240)
    assert not torch.equal(prediction[0], prediction[1])
    np.testing.assert_array_equal(prediction_coords["ensemble"], np.arange(2))


def test_fuxi_s2s_shifted_leads_use_matching_step(fuxi_s2s_test_package) -> None:
    model = FuXiS2S.load_model(fuxi_s2s_test_package)
    coords = model.input_coords()
    del coords["batch"]
    coords["time"] = np.array([np.datetime64("2020-01-01")])
    coords["lead_time"] += np.timedelta64(10, "D")
    x = torch.ones(1, 2, len(VARIABLES), 121, 240)

    output, output_coords = model(x, coords)

    expected = 1.0 + 10.0 + 11.0 / 365.0
    torch.testing.assert_close(
        output[:, :, 0],
        torch.full_like(output[:, :, 0], expected),
    )
    assert output_coords["lead_time"].tolist() == [np.timedelta64(11, "D")]


def test_fuxi_s2s_rejects_fractional_or_negative_latest_lead() -> None:
    model = _identity_model()
    coords = model.input_coords()
    coords["batch"] = np.array([0])
    coords["time"] = np.array([np.datetime64("2020-01-01")])

    coords["lead_time"] = np.array([np.timedelta64(-12, "h"), np.timedelta64(12, "h")])
    with pytest.raises(ValueError, match="non-negative whole number of days"):
        model.output_coords(coords)

    coords["lead_time"] = np.array([np.timedelta64(-2, "D"), np.timedelta64(-1, "D")])
    with pytest.raises(ValueError, match="non-negative whole number of days"):
        model.output_coords(coords)


def test_fuxi_s2s_extracts_only_requested_archive_member(tmp_path) -> None:
    tar_path = tmp_path / "model.tar"
    with tarfile.open(tar_path, mode="w") as archive:
        member = tarfile.TarInfo("model-1.0/fuxi_s2s.onnx")
        member.size = len(b"onnx")
        archive.addfile(member, io.BytesIO(b"onnx"))

    onnx_path = _extract_tar_member(
        str(tar_path),
        "model-1.0/fuxi_s2s.onnx",
        tmp_path / "assets" / "fuxi_s2s.onnx",
    )

    assert onnx_path.read_bytes() == b"onnx"


def test_fuxi_s2s_load_model_resolves_external_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = Package(str(tmp_path))
    resolved: list[str] = []

    def resolve(file_path: str) -> str:
        resolved.append(file_path)
        return str(tmp_path / file_path)

    monkeypatch.setattr(package, "resolve", resolve)

    model = FuXiS2S.load_model(package)

    assert model.onnx_path == str(tmp_path / "fuxi_s2s.onnx")
    assert resolved == ["fuxi_s2s", "fuxi_s2s.onnx"]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_fuxi_s2s_package(device: str) -> None:
    torch.cuda.empty_cache()
    package = FuXiS2S.load_default_package()
    model = FuXiS2S.load_model(package).to(device)
    coords = model.input_coords()
    del coords["batch"]
    coords["time"] = np.array([np.datetime64("2020-06-02")])

    with zipfile.ZipFile(package.resolve("data.zip?download=1")) as archive:
        datasets = {}
        for name in ("input", "mean", "std"):
            with archive.open(f"data/{name}.nc") as stream:
                datasets[name] = xr.open_dataset(io.BytesIO(stream.read())).load()
        with archive.open("data/sample/total_precipitation.nc") as stream:
            official_tp = xr.open_dataarray(io.BytesIO(stream.read())).load()

    official_names = {
        "u10m": "10u",
        "v10m": "10v",
        "u100m": "100u",
        "v100m": "100v",
    }
    official_variables = [
        official_names.get(variable, variable) for variable in VARIABLES
    ]
    normalized = datasets["input"]["data"].sel(level=official_variables).values
    center = datasets["mean"]["data"].sel(level=official_variables).values
    scale = datasets["std"]["data"].sel(level=official_variables).values
    model_input = normalized * scale[None, :, None, None]
    model_input += center[None, :, None, None]
    x = torch.from_numpy(model_input).unsqueeze(0).to(device)
    x[:, :, VARIABLES.index("ttr")].mul_(3600.0)
    x[:, :, VARIABLES.index("tp")].expm1_().clamp_(min=0.0).div_(1000.0)

    ort_input = model._prepare_input(x.float())
    expected_ort_input = torch.from_numpy(model_input).unsqueeze(0)
    expected_ort_input[:, :, VARIABLES.index("tp")] = torch.from_numpy(
        official_tp.values[:, 0] * 1000.0
    )
    torch.testing.assert_close(
        ort_input.cpu(),
        expected_ort_input,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    output, output_coords = model(x, coords)

    assert output.shape == (1, 1, len(VARIABLES), 121, 240)
    assert torch.isfinite(output).all()
    assert torch.all(
        (output[:, :, VARIABLES.index("t2m")] > 100.0)
        & (output[:, :, VARIABLES.index("t2m")] < 400.0)
    )
    assert torch.all(output[:, :, VARIABLES.index("tp")] >= 0.0)
    assert torch.all(output[:, :, VARIABLES.index("tp")] < 1.1)
    assert torch.max(torch.abs(output[:, :, VARIABLES.index("ttr")])) > 3600.0
    np.testing.assert_array_equal(
        output_coords["lead_time"],
        np.array([np.timedelta64(1, "D")]),
    )

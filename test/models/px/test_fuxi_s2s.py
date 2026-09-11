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
import zipfile
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import fsspec
import numpy as np
import onnx
import pytest
import torch
import xarray as xr
from onnx import TensorProto, helper, numpy_helper

from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px import FuXiS2S
from earth2studio.models.px.fuxi_s2s import VARIABLES
from earth2studio.utils import handshake_dim


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


class _PhooBinding:
    """Minimal IO-binding mock for :class:`PhooStochasticSession`."""

    def __init__(self) -> None:
        self.inputs: dict[str, torch.Tensor] = {}
        self.outputs: dict[str, torch.Tensor] = {}

    def bind_input(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: Any,
        shape: tuple[int, ...],
        buffer_ptr: int,
    ) -> None:
        self.inputs[name] = _tensor_from_ptr(buffer_ptr, shape, device_type, device_id)

    def bind_output(
        self,
        name: str,
        device_type: str,
        device_id: int,
        element_type: Any,
        shape: tuple[int, ...],
        buffer_ptr: int,
    ) -> None:
        self.outputs[name] = _tensor_from_ptr(buffer_ptr, shape, device_type, device_id)


def _tensor_from_ptr(
    ptr: int, shape: tuple[int, ...], device_type: str, device_id: int
) -> torch.Tensor:
    """Reconstruct a tensor view from a raw data pointer (CPU only in tests)."""
    import ctypes

    numel = 1
    for s in shape:
        numel *= s
    arr = (ctypes.c_float * numel).from_address(ptr)
    return torch.frombuffer(arr, dtype=torch.float32).reshape(shape)


class PhooStochasticSession:
    """Small ORT stand-in that returns a distinct sample on every call.

    Supports both the legacy ``run()`` API and the ``io_binding`` API
    used by the production ``_forward`` implementation.
    """

    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list[Any]:
        return [
            type("OrtValue", (), {"name": name})() for name in ("input", "step", "doy")
        ]

    def get_outputs(self) -> list[Any]:
        return [type("OrtValue", (), {"name": "output"})()]

    def io_binding(self) -> _PhooBinding:
        return _PhooBinding()

    def run_with_iobinding(self, binding: _PhooBinding) -> None:
        self.calls += 1
        inp = binding.inputs["input"]
        prediction = inp[:, -1:] + self.calls
        result = torch.cat((inp[:, -1:], prediction), dim=1)
        out = binding.outputs["output"]
        out.copy_(result)


@pytest.fixture(scope="class")
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


class TestFuXiS2SMock:

    @pytest.mark.parametrize(
        "time",
        [
            np.array([np.datetime64("2020-01-01T00:00")]),
            np.array(
                [
                    np.datetime64("2020-01-01T00:00"),
                    np.datetime64("2020-06-15T00:00"),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_fuxi_s2s_call(self, time, fuxi_s2s_test_package, device) -> None:
        model = FuXiS2S.load_model(fuxi_s2s_test_package).to(device)

        dc = model.input_coords()
        del dc["batch"]
        del dc["time"]
        del dc["lead_time"]
        del dc["variable"]
        r = Random(dc)

        lead_time = model.input_coords()["lead_time"]
        variable = model.input_coords()["variable"]
        x, coords = fetch_data(r, time, variable, lead_time, device=device)

        out, out_coords = model(x, coords)

        if not isinstance(time, Iterable):
            time = [time]

        assert out.shape == torch.Size([len(time), 1, len(VARIABLES), 121, 240])
        assert (out_coords["time"] == time).all()
        np.testing.assert_array_equal(
            out_coords["lead_time"],
            np.array([np.timedelta64(1, "D")]),
        )
        handshake_dim(out_coords, "lon", 4)
        handshake_dim(out_coords, "lat", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "lead_time", 1)
        handshake_dim(out_coords, "time", 0)

    @pytest.mark.parametrize(
        "dc",
        [
            OrderedDict({"lat": np.linspace(-90, 90, 121)}),
            OrderedDict(
                {"lat": np.linspace(90, -90, 121), "phoo": np.random.randn(240)}
            ),
            OrderedDict(
                {
                    "lat": np.linspace(90, -90, 121),
                    "lon": np.random.randn(1),
                }
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_fuxi_s2s_exceptions(self, dc, fuxi_s2s_test_package, device) -> None:
        time = np.array([np.datetime64("2020-01-01T00:00")])
        model = FuXiS2S.load_model(fuxi_s2s_test_package).to(device)

        r = Random(dc)

        lead_time = model.input_coords()["lead_time"]
        variable = model.input_coords()["variable"]
        x, coords = fetch_data(r, time, variable, lead_time, device=device)

        with pytest.raises((KeyError, ValueError)):
            model(x, coords)

    @pytest.mark.parametrize("ensemble", [2])
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fuxi_s2s_iter(self, ensemble, fuxi_s2s_test_package, device) -> None:
        time = np.array([np.datetime64("2020-01-01T00:00")])
        model = FuXiS2S.load_model(fuxi_s2s_test_package).to(device)

        dc = model.input_coords()
        del dc["batch"]
        del dc["time"]
        del dc["lead_time"]
        del dc["variable"]
        r = Random(dc)

        lead_time = model.input_coords()["lead_time"]
        variable = model.input_coords()["variable"]
        x, coords = fetch_data(r, time, variable, lead_time, device=device)

        x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
        coords.update({"ensemble": np.arange(ensemble)})
        coords.move_to_end("ensemble", last=False)

        p_iter = model.create_iterator(x, coords)

        if not isinstance(time, Iterable):
            time = [time]

        # Initial yield should return the input
        out, out_coords = next(p_iter)
        assert out.shape[0] == ensemble

        for i, (out, out_coords) in enumerate(p_iter):
            assert len(out.shape) == 6
            assert out.shape[0] == ensemble
            assert (out_coords["time"] == time).all()
            assert out_coords["lead_time"][0] == np.timedelta64(i + 1, "D")
            handshake_dim(out_coords, "lon", 5)
            handshake_dim(out_coords, "lat", 4)
            handshake_dim(out_coords, "variable", 3)
            handshake_dim(out_coords, "lead_time", 2)
            handshake_dim(out_coords, "time", 1)
            handshake_dim(out_coords, "ensemble", 0)

            if i > 3:
                break


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


def test_fuxi_s2s_default_package_is_pinned() -> None:
    package = FuXiS2S.load_default_package()

    assert package.root == (
        "hf://Artamta/FuXi-S2S-ONNX@" "5d7a6b132aaaaa070d2856d002f95911140db0ff"
    )


def test_fuxi_s2s_load_model_stages_remote_external_weights(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source"
    source_path.mkdir()
    model_path = source_path / "fuxi_s2s.onnx"

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    weight = numpy_helper.from_array(np.array([2.0], dtype=np.float32), "weight")
    graph = helper.make_graph(
        [helper.make_node("Add", ["input", "weight"], ["output"])],
        "external-data-test",
        [input_info],
        [output_info],
        [weight],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_model.ir_version = 9
    onnx.save_model(
        onnx_model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="fuxi_s2s",
        size_threshold=0,
    )

    remote_name = tmp_path.name
    remote_root = f"memory://{remote_name}"
    memory_fs = fsspec.filesystem("memory")
    for file_name in ("fuxi_s2s", "fuxi_s2s.onnx"):
        memory_fs.pipe(
            f"/{remote_name}/{file_name}",
            (source_path / file_name).read_bytes(),
        )
    package = Package(
        remote_root,
        cache_options={"cache_storage": "TMP"},
    )

    external_path = Path(package.resolve("fuxi_s2s"))
    cached_model_path = Path(package.resolve("fuxi_s2s.onnx"))
    assert external_path != cached_model_path.with_name("fuxi_s2s")

    model = FuXiS2S.load_model(package)

    staged_model_path = Path(model.onnx_path)
    assert staged_model_path.name == "fuxi_s2s.onnx"
    assert staged_model_path.is_absolute()
    assert staged_model_path.is_relative_to(Path(package.fs.storage[-1]).resolve())
    staged_external_path = staged_model_path.with_name("fuxi_s2s")
    assert staged_external_path.is_file()
    output = model._get_ort_session().run(
        None,
        {"input": np.array([3.0], dtype=np.float32)},
    )[0]
    np.testing.assert_array_equal(output, np.array([5.0], dtype=np.float32))

    staged_external_data = staged_external_path.read_bytes()
    external_path.write_bytes(b"refreshed")
    assert staged_external_path.read_bytes() == staged_external_data


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_fuxi_s2s_package(device: str) -> None:
    torch.cuda.empty_cache()
    package = FuXiS2S.load_default_package()
    model = FuXiS2S.load_model(package).to(device)
    coords = model.input_coords()
    del coords["batch"]
    coords["time"] = np.array([np.datetime64("2020-06-02")])

    sample_package = Package(
        "https://zenodo.org/records/15718402/files",
        cache_options={
            "cache_storage": Package.default_cache("fuxi_s2s"),
            "same_names": True,
        },
    )
    with zipfile.ZipFile(sample_package.resolve("data.zip?download=1")) as archive:
        datasets = {}
        for name in ("input", "mean", "std"):
            with archive.open(f"data/{name}.nc") as stream:
                datasets[name] = xr.open_dataset(io.BytesIO(stream.read())).load()
        official_samples = {}
        for variable, file_name in {
            "sst": "sea_surface_temperature.nc",
            "t2m": "2m_temperature.nc",
            "tp": "total_precipitation.nc",
            "ttr": "top_net_thermal_radiation.nc",
            "z500": "geopotential.nc",
        }.items():
            with archive.open(f"data/sample/{file_name}") as stream:
                official_samples[variable] = xr.open_dataarray(
                    io.BytesIO(stream.read())
                ).load()

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
    official_sst = official_samples["sst"].values[:, 0]
    assert np.isnan(official_sst).any()
    assert np.isfinite(official_sst).any()
    model_input[:, VARIABLES.index("sst")] = official_sst
    np.testing.assert_allclose(
        model_input[:, VARIABLES.index("t2m")],
        official_samples["t2m"].values[:, 0],
    )
    np.testing.assert_allclose(
        model_input[:, VARIABLES.index("z500")],
        official_samples["z500"].sel(level=500).values,
    )
    np.testing.assert_array_equal(
        np.isnan(model_input[:, VARIABLES.index("sst")]),
        np.isnan(official_sst),
    )
    x = torch.from_numpy(model_input).unsqueeze(0).to(device)
    x[:, :, VARIABLES.index("ttr")].mul_(3600.0)
    x[:, :, VARIABLES.index("tp")].expm1_().clamp_(min=0.0).div_(1000.0)
    np.testing.assert_allclose(
        x[0, :, VARIABLES.index("ttr")].cpu().numpy(),
        official_samples["ttr"].values[:, 0],
        rtol=1.0e-6,
        atol=0.1,
    )

    ort_input = model._prepare_input(x.float())
    expected_ort_input = torch.from_numpy(model_input).unsqueeze(0)
    expected_ort_input[:, :, VARIABLES.index("tp")] = torch.from_numpy(
        official_samples["tp"].values[:, 0] * 1000.0
    )
    torch.testing.assert_close(
        ort_input.cpu(),
        expected_ort_input,
        rtol=1.0e-5,
        atol=1.0e-5,
        equal_nan=True,
    )
    np.testing.assert_array_equal(
        np.isnan(ort_input[0, :, VARIABLES.index("sst")].cpu().numpy()),
        np.isnan(official_sst),
    )
    providers = model._get_ort_session().get_providers()
    assert providers[0] == "CUDAExecutionProvider"
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

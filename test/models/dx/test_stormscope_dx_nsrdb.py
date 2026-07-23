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

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

import earth2studio.models.dx.stormscope_dx_nsrdb as stormscope_module
from earth2studio.models.dx import StormScopeDxNSRDB
from earth2studio.utils import handshake_dim


class PhooDiffusionModel(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return x[:, :1]


class PhooRegressionModel(torch.nn.Module):
    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return condition[:, :1] * 0.0


class LocalPackage:
    def __init__(self, root: Path):
        self.root = root

    def resolve(self, path: str) -> str:
        resolved = self.root / path
        if not resolved.exists():
            raise FileNotFoundError(path)
        return str(resolved)


class PhooModule:
    @staticmethod
    def from_checkpoint(path: str) -> torch.nn.Module:
        if "regression" in path:
            return PhooRegressionModel()
        return PhooDiffusionModel()


def create_model(
    device: str = "cpu",
    number_of_samples: int = 1,
    seed: int | None = None,
    partial_mask: bool = False,
) -> StormScopeDxNSRDB:
    height, width = 32, 64
    latitudes = torch.linspace(25, 50, height).unsqueeze(1).repeat(1, width)
    longitudes = torch.linspace(-120, -80, width).unsqueeze(0).repeat(height, 1)
    lat_radians = torch.deg2rad(latitudes)
    lon_radians = torch.deg2rad(longitudes)
    invariants = torch.stack(
        [
            torch.sin(lat_radians),
            torch.cos(lat_radians),
            torch.sin(lon_radians),
            torch.cos(lon_radians),
        ]
    )
    valid_mask = torch.ones(height, width, dtype=torch.bool)
    if partial_mask:
        valid_mask[: height // 2] = False

    return StormScopeDxNSRDB(
        diffusion_model=PhooDiffusionModel(),
        regression_model=PhooRegressionModel(),
        sigma_min=0.004,
        sigma_max=0.25,
        conditioning_means=torch.zeros(1, 8, 1, 1),
        conditioning_stds=torch.ones(1, 8, 1, 1),
        conditioning_variables=np.array([f"abi{index:02d}c" for index in range(1, 9)]),
        output_variables=np.array(["ghi"]),
        latitudes=latitudes,
        longitudes=longitudes,
        invariants=invariants,
        valid_mask=valid_mask,
        y_coords=np.arange(height),
        x_coords=np.arange(width),
        number_of_samples=number_of_samples,
        seed=seed,
        num_steps=2,
        amp=False,
    ).to(device)


def make_input(
    model: StormScopeDxNSRDB,
    batch: int = 1,
    time: int = 1,
    device: str = "cpu",
) -> tuple[torch.Tensor, OrderedDict]:
    input_coords = model.input_coords()
    tensor = torch.linspace(
        0,
        1,
        steps=batch
        * time
        * len(input_coords["variable"])
        * len(input_coords["y"])
        * len(input_coords["x"]),
        device=device,
    ).reshape(
        batch,
        time,
        len(input_coords["variable"]),
        len(input_coords["y"]),
        len(input_coords["x"]),
    )
    coords = OrderedDict(
        {
            "batch": np.arange(batch),
            "time": np.array([np.datetime64("2024-07-15T18:00")] * time),
            "variable": input_coords["variable"],
            "y": input_coords["y"],
            "x": input_coords["x"],
        }
    )
    return tensor, coords


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_dx_nsrdb_call(device):
    model = create_model(device=device, partial_mask=True)
    input_tensor, input_coords = make_input(model, batch=2, device=device)

    output, output_coords = model(input_tensor, input_coords)

    assert output.shape == (2, 1, 1, 1, 32, 64)
    assert output_coords["variable"].tolist() == ["ghi"]
    assert "lead_time" not in input_coords
    assert "lead_time" not in output_coords
    handshake_dim(output_coords, "batch", 0)
    handshake_dim(output_coords, "sample", 1)
    handshake_dim(output_coords, "time", 2)
    handshake_dim(output_coords, "variable", 3)
    handshake_dim(output_coords, "y", 4)
    handshake_dim(output_coords, "x", 5)
    valid = model.valid_mask.cpu().numpy()
    output_numpy = output.detach().cpu().numpy()
    assert np.isnan(output_numpy[..., ~valid]).all()
    assert np.isfinite(output_numpy[..., valid]).all()
    assert (output_numpy[..., valid] >= 0).all()


def test_stormscope_dx_nsrdb_seed_and_samples():
    model = create_model(number_of_samples=2, seed=42)
    input_tensor, input_coords = make_input(model)

    first, first_coords = model(input_tensor, input_coords)
    second, _ = model(input_tensor, input_coords)

    torch.testing.assert_close(first, second, equal_nan=True)
    assert first.shape == (1, 2, 1, 1, 32, 64)
    np.testing.assert_array_equal(first_coords["sample"], np.arange(2))


def test_stormscope_dx_nsrdb_defaults_and_name():
    model = create_model()
    default_model = StormScopeDxNSRDB(
        diffusion_model=PhooDiffusionModel(),
        regression_model=PhooRegressionModel(),
        sigma_min=model.sigma_min,
        sigma_max=model.sigma_max,
        conditioning_means=model.conditioning_means,
        conditioning_stds=model.conditioning_stds,
        conditioning_variables=model.conditioning_variables,
        output_variables=model.output_variables,
        latitudes=model.latitudes,
        longitudes=model.longitudes,
        num_steps=2,
        amp=False,
    )

    assert str(default_model) == "StormScopeDxNSRDB"
    assert default_model.invariants is None
    assert default_model.valid_mask.all()
    np.testing.assert_array_equal(default_model.y, np.arange(32))
    np.testing.assert_array_equal(default_model.x, np.arange(64))
    assert default_model.load_default_package() is not None


def test_stormscope_dx_nsrdb_input_interpolation():
    model = create_model()
    model.build_input_interpolator(model.latitudes, model.longitudes)
    input_tensor, input_coords = make_input(model)
    input_coords["latitude"] = input_coords.pop("y")
    input_coords["longitude"] = input_coords.pop("x")

    output, output_coords = model(input_tensor, input_coords)

    assert output.shape == (1, 1, 1, 1, 32, 64)
    assert list(output_coords) == ["batch", "sample", "time", "variable", "y", "x"]


def test_stormscope_dx_nsrdb_invalid_tensor_rank():
    model = create_model()
    _, input_coords = make_input(model)

    with pytest.raises(ValueError, match=r"\[batch, time, variable, y, x\]"):
        model._forward_sample(torch.zeros(1, 8, 32, 64), input_coords)


def test_stormscope_dx_nsrdb_local_package(tmp_path, monkeypatch):
    height, width = 32, 64
    variables = [f"abi{index:02d}c" for index in range(1, 9)]
    registry = {
        "normalization": {
            "goes": {"order": variables, "file_prefix": "goes"},
            "identity": {"order": ["identity"], "file_prefix": None},
        },
        "stormscope_solar_goes_nsrdb": {
            "checkpoints": [
                {
                    "path": "diffusion.mdlus",
                    "sigma_min": 0.004,
                    "sigma_max": 0.25,
                }
            ],
            "regression_checkpoint": {"path": "regression.mdlus"},
            "image_size": [height, width],
            "spatial_downsample": 1,
            "conditioning_vars": variables,
            "variables": ["ghi"],
        },
    }
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    (tmp_path / "diffusion.mdlus").touch()
    (tmp_path / "regression.mdlus").touch()
    np.save(tmp_path / "goes_means.npy", np.arange(8, dtype=np.float32))
    np.save(tmp_path / "goes_stds.npy", np.arange(1, 9, dtype=np.float32))
    latitudes = np.linspace(25, 50, height, dtype=np.float32)[:, None]
    latitudes = np.repeat(latitudes, width, axis=1)
    longitudes = np.linspace(-120, -80, width, dtype=np.float32)[None, :]
    longitudes = np.repeat(longitudes, height, axis=0)
    np.save(tmp_path / "lat.npy", latitudes)
    np.save(tmp_path / "lon.npy", longitudes)
    np.save(tmp_path / "altitude.npy", np.ones((height, width), dtype=np.float32))
    np.save(tmp_path / "elev_std.npy", np.ones((height, width), dtype=np.float32))
    np.save(tmp_path / "nsrdb_mask.npy", np.ones((height, width), dtype=np.float32))
    package = LocalPackage(tmp_path)
    monkeypatch.setattr(stormscope_module, "Module", PhooModule)

    model = StormScopeDxNSRDB.load_model(
        package, number_of_samples=2, seed=7, amp=False
    )
    identity_means, identity_stds = model._build_normalization(
        package, registry, np.array(["identity"])
    )

    assert model.number_of_samples == 2
    assert model.seed == 7
    torch.testing.assert_close(
        model.conditioning_means.flatten(), torch.arange(8, dtype=torch.float32)
    )
    torch.testing.assert_close(
        model.conditioning_stds.flatten(), torch.arange(1, 9, dtype=torch.float32)
    )
    torch.testing.assert_close(identity_means, torch.zeros_like(identity_means))
    torch.testing.assert_close(identity_stds, torch.ones_like(identity_stds))
    assert model.latitudes.shape == (height, width)
    assert model.longitudes.min() >= 0

    with pytest.raises(KeyError, match="missing"):
        model._build_normalization(package, registry, np.array(["missing"]))

    registry["stormscope_solar_goes_nsrdb"]["checkpoints"].append(
        registry["stormscope_solar_goes_nsrdb"]["checkpoints"][0]
    )
    (tmp_path / "registry.json").write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="one diffusion checkpoint"):
        StormScopeDxNSRDB.load_model(package)


@pytest.mark.parametrize(
    "coordinate, value",
    [
        ("variable", np.array(["wrong"])),
        ("y", np.arange(33)),
    ],
)
def test_stormscope_dx_nsrdb_exceptions(coordinate, value):
    model = create_model()
    input_tensor, input_coords = make_input(model)
    input_coords[coordinate] = value

    with pytest.raises((KeyError, ValueError)):
        model(input_tensor, input_coords)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"number_of_samples": 0}, "number_of_samples"),
        ({"num_steps": 1}, "num_steps"),
    ],
)
def test_stormscope_dx_nsrdb_constructor_exceptions(kwargs, match):
    model = create_model()
    constructor_args = {
        "diffusion_model": PhooDiffusionModel(),
        "regression_model": PhooRegressionModel(),
        "sigma_min": model.sigma_min,
        "sigma_max": model.sigma_max,
        "conditioning_means": model.conditioning_means,
        "conditioning_stds": model.conditioning_stds,
        "conditioning_variables": model.conditioning_variables,
        "output_variables": model.output_variables,
        "latitudes": model.latitudes,
        "longitudes": model.longitudes,
    }
    constructor_args.update(kwargs)

    with pytest.raises(ValueError, match=match):
        StormScopeDxNSRDB(**constructor_args)


@pytest.mark.package
@pytest.mark.timeout(600)
def test_stormscope_dx_nsrdb_package():
    model = StormScopeDxNSRDB.load_model(
        StormScopeDxNSRDB.load_default_package(),
        seed=42,
        amp=True,
    ).to("cuda:0")
    model.num_steps = 2
    input_tensor, input_coords = make_input(model, device="cuda:0")

    output, output_coords = model(input_tensor, input_coords)

    assert output.shape[0] == 1
    assert output.shape[1] == 1
    assert output_coords["variable"].tolist() == ["ghi"]
    assert torch.isfinite(output[..., model.valid_mask]).all()

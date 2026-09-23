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
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.dx.stormscope_dx_nsrdb as stormscope_module
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import StormScopeDxNSRDB
from earth2studio.utils import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import OptionalDependencyFailure


@pytest.fixture(autouse=True)
def optional_backend(request, monkeypatch):
    if (
        stormscope_module.Module is None
        and request.node.get_closest_marker("package") is None
    ):
        monkeypatch.delitem(
            OptionalDependencyFailure.failures,
            stormscope_module.__file__,
            raising=False,
        )
        monkeypatch.setattr(
            stormscope_module,
            "pnm_insolation",
            lambda dates, lat, lon, scale, **kw: np.ones(
                (len(dates), *lat.shape), dtype=np.float32
            )
            * scale,
        )
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")


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
) -> xr.DataArray:
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
    coords = coord_array_like(
        input_coords,
        {
            "batch": np.arange(batch),
            "time": np.array([np.datetime64("2024-07-15T18:00")] * time),
        },
    )
    return from_torch(tensor, coords)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_dx_nsrdb_call(device):
    model = create_model(device=device, partial_mask=True)
    input_tensor = make_input(model, batch=2)
    input_tensor.name = "imagery"
    input_tensor.attrs["nested"] = {"owner": ["caller"]}
    input_tensor.encoding = {"nested": {"owner": ["caller"]}}
    input_tensor = input_tensor.assign_coords(
        units=("variable", ["input"] * 8), auxiliary=("batch", [4, 5])
    )
    before = input_tensor.copy(deep=True)
    input_coords = input_tensor.coords

    output = model(input_tensor)
    output_coords = output.coords
    handshake_dataarray(output, model.output_coords(input_tensor))

    assert output.shape == (2, 1, 1, 1, 32, 64)
    assert output_coords["variable"].values.tolist() == ["ghi"]
    assert "lead_time" not in input_coords
    assert "lead_time" not in output_coords
    assert output.dims == ("batch", "time", "sample", "variable", "y", "x")
    valid = model.valid_mask.cpu().numpy()
    output_numpy = output.e2s.to_torch()[0].cpu().numpy()
    assert np.isnan(output_numpy[..., ~valid]).all()
    assert np.isfinite(output_numpy[..., valid]).all()
    assert (output_numpy[..., valid] >= 0).all()
    assert output.name == "imagery" and output.encoding == input_tensor.encoding
    assert "units" not in output.coords
    np.testing.assert_array_equal(output.auxiliary, input_tensor.auxiliary)
    output.attrs["nested"]["owner"].append("output")
    output.encoding["nested"]["owner"].append("output")
    xr.testing.assert_identical(input_tensor, before)
    assert input_tensor.encoding == before.encoding


def test_stormscope_dx_nsrdb_seed_and_samples():
    model = create_model(number_of_samples=2, seed=42)
    input_tensor = make_input(model)

    rng = torch.get_rng_state().clone()
    first = model(input_tensor)
    second = model(input_tensor)
    first_coords = first.coords

    xr.testing.assert_identical(first, second)
    assert torch.equal(rng, torch.get_rng_state())
    model.set_rng(43, reset=False)
    xr.testing.assert_identical(first, model(input_tensor))
    model.set_rng(43)
    assert not np.array_equal(first.values, model(input_tensor).values)
    assert first.shape == (1, 1, 2, 1, 32, 64)
    np.testing.assert_array_equal(first_coords["sample"], np.arange(2))
    forecast = (
        input_tensor.expand_dims(member=["a", "b"], lead_time=[np.timedelta64(3, "h")])
        .transpose("member", "batch", "time", "lead_time", "variable", "y", "x")
        .copy(deep=True)
    )
    forecast = forecast.assign_coords(valid_time=forecast.time + forecast.lead_time)
    result = model(forecast)
    np.testing.assert_array_equal(result.valid_time, forecast.valid_time)
    observed = input_tensor.assign_coords(
        time=input_tensor.time + np.timedelta64(3, "h")
    )
    expected = model(observed)
    model.set_rng(43)
    one = model(
        forecast.isel(member=0, lead_time=0, drop=True).assign_coords(
            time=observed.time
        )
    )
    np.testing.assert_array_equal(one.values, expected.values)
    assert result.dims == (*forecast.dims[:-3], "sample", "variable", "y", "x")
    assert model.output_coords(model.input_coords()).attrs[
        "earth2studio_dynamic_dims"
    ] == ("batch", "time")


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


def test_stormscope_dx_nsrdb_input_interpolation(monkeypatch):
    import importlib.util

    if importlib.util.find_spec("earth2grid") is None:

        class OfflineNearest(torch.nn.Module):
            def __init__(
                self, source_lats, source_lons, target_lats, target_lons, max_dist_km
            ):
                super().__init__()
                self.register_buffer(
                    "valid_mask", torch.ones_like(target_lats, dtype=torch.bool)
                )

            def forward(self, x):
                return x[..., ::2, ::2]

        monkeypatch.setattr(
            stormscope_module, "NearestNeighborInterpolator", OfflineNearest
        )
    model = create_model()
    source_lat = model.latitudes.repeat_interleave(2, 0).repeat_interleave(2, 1)
    source_lon = model.longitudes.repeat_interleave(2, 0).repeat_interleave(2, 1)
    model.build_input_interpolator(source_lat, source_lon)
    from earth2studio.grids import CurvilinearGrid

    source_grid = CurvilinearGrid(
        source_lat.numpy(),
        source_lon.numpy(),
        y=3000 * np.arange(source_lat.shape[0]) - 1587306,
        x=3000 * np.arange(source_lat.shape[1]) - 2697520,
    )
    model.build_input_interpolator(source_lat, source_lon, input_grid=source_grid)
    np.testing.assert_array_equal(model.input_coords().y, source_grid.y)
    np.testing.assert_array_equal(model.input_coords().x, source_grid.x)
    source_signature = model.input_coords()
    model.build_input_interpolator(source_lat, source_lon, input_grid=source_signature)
    xr.testing.assert_identical(
        model.input_coords().coords.to_dataset(), source_signature.coords.to_dataset()
    )
    assert model.input_coords().attrs == source_signature.attrs
    with pytest.raises(ValueError, match="input_grid"):
        model.build_input_interpolator(
            source_lat + 1, source_lon, input_grid=source_grid
        )
    xr.testing.assert_identical(
        model.input_coords().coords.to_dataset(), source_signature.coords.to_dataset()
    )
    assert model.input_coords().attrs == source_signature.attrs
    input_tensor = make_input(model)
    input_tensor.attrs["earth2studio_grid_id"] = "source-only"
    input_tensor = input_tensor.assign_coords(
        source_aux=(("y", "x"), np.zeros(source_lat.shape))
    )

    output = model(input_tensor)

    assert output.shape == (1, 1, 1, 1, 32, 64)
    assert list(output.dims) == ["batch", "time", "sample", "variable", "y", "x"]
    assert "source_aux" not in output.coords
    assert "earth2studio_grid_id" not in output.attrs
    np.testing.assert_array_equal(output.lat, model.latitudes.cpu())
    np.testing.assert_array_equal(output.lon, model.longitudes.cpu())


def test_stormscope_dx_nsrdb_invalid_tensor_rank():
    model = create_model()
    input_coords = make_input(model)

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

    model = StormScopeDxNSRDB.load_model(package, number_of_samples=2, seed=7)
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
    input_tensor = make_input(model)

    with pytest.raises((KeyError, ValueError)):
        model.output_coords(coord_array_like(input_tensor, {coordinate: value}))


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


def test_stormscope_dx_nsrdb_conformance():
    model = create_model()
    check_diagnostic_contract(model)


@pytest.mark.package
@pytest.mark.timeout(600)
def test_stormscope_dx_nsrdb_package():
    model = StormScopeDxNSRDB.load_model(
        StormScopeDxNSRDB.load_default_package(),
        seed=42,
    ).to("cuda:0")
    model.num_steps = 2
    input_tensor = make_input(model, device="cuda:0")

    output = model(input_tensor)

    assert output.shape[0] == 1
    assert output.shape[1] == 1
    assert output.coords["variable"].values.tolist() == ["ghi"]
    assert torch.isfinite(output.e2s.to_torch()[0][..., model.valid_mask]).all()

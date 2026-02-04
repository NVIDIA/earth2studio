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
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.dx import CorrDiff
from earth2studio.utils import handshake_dim


class MockPhysicsNemoModule(torch.nn.Module):
    """Mock model for testing CorrDiff residual and regression models."""

    # List of all instances created by from_checkpoint(). Tests can inspect this
    # to verify how many models were loaded and check their .to_calls history.
    created: ClassVar[list["MockPhysicsNemoModule"]] = []

    def __init__(self, img_out_channels=4, device="cpu"):
        super().__init__()
        self.img_out_channels = img_out_channels
        self.sigma_min = 0.0
        self.sigma_max = float("inf")
        self.device = torch.device(device)
        self.profile_mode = False  # For inference optimization tests
        self.to_calls: list[tuple[object | None, object | None]] = []

    def forward(self, x, img_lr=None, sigma=None, class_labels=None, **kwargs):
        # Return tensor with expected output shape
        batch_size = x.shape[0] if len(x.shape) > 0 else 1
        return torch.zeros(
            batch_size, self.img_out_channels, 320, 320, device=self.device
        )

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

    def device(self):
        return self.device

    def to(self, device=None, memory_format=None):
        # Record every `.to(...)` call so tests can assert ordering:
        # 1) `.to(device=...)` must happen before
        # 2) `.to(memory_format=torch.channels_last)`
        self.to_calls.append((device, memory_format))
        if device is not None:
            dev = device if isinstance(device, torch.device) else torch.device(device)
            super().to(dev)
            self.device = dev
        return self

    @classmethod
    def from_checkpoint(cls, path, strict=False):
        inst = cls()
        cls.created.append(inst)
        return inst


@pytest.fixture
def mock_residual_model():
    """Create a mock residual model for testing."""
    return MockPhysicsNemoModule(img_out_channels=4)


@pytest.fixture
def mock_regression_model():
    """Create a mock regression model for testing."""
    return MockPhysicsNemoModule(img_out_channels=4)


@pytest.fixture
def mock_package():
    """Create a mock package for testing."""
    package = MagicMock(spec=Package)
    return package


@pytest.fixture(params=["rectangular", "curvilinear"])
def sample_model_params(request):
    """Create sample model parameters for testing."""
    input_variables = ["t2m", "u10m", "v10m", "z500"]
    output_variables = ["mrr", "t2m", "u10m", "v10m"]

    # Create sample grids
    input_lat = torch.linspace(19.25, 28, 36)
    input_lon = torch.linspace(116, 126, 41)[:-1]
    if request.param == "rectangular":
        input_lat_grid, input_lon_grid = input_lat, input_lon
    else:
        # 2D curvilinear input grid: start from rectilinear meshgrid and add
        # a small longitude-dependent perturbation to latitude so the grid is
        # no longer strictly rectilinear.
        base_lat_grid, base_lon_grid = torch.meshgrid(
            input_lat, input_lon, indexing="ij"
        )
        input_lat_grid = base_lat_grid + 0.01 * torch.sin(torch.deg2rad(base_lon_grid))
        input_lon_grid = base_lon_grid
    output_lat = torch.linspace(24.25, 27.75, 320)
    output_lon = torch.linspace(116.25, 119.75, 320)

    output_lat_grid, output_lon_grid = torch.meshgrid(
        output_lat, output_lon, indexing="ij"
    )

    # Create normalization parameters
    in_center = torch.zeros(len(input_variables))
    in_scale = torch.ones(len(input_variables))
    out_center = torch.zeros(len(output_variables))
    out_scale = torch.ones(len(output_variables))

    return {
        "input_variables": input_variables,
        "output_variables": output_variables,
        "lat_input_grid": input_lat_grid,
        "lon_input_grid": input_lon_grid,
        "lat_output_grid": output_lat_grid,
        "lon_output_grid": output_lon_grid,
        "in_center": in_center,
        "in_scale": in_scale,
        "out_center": out_center,
        "out_scale": out_scale,
    }


@pytest.fixture(params=["rectangular", "curvilinear"])
def temp_model_files(request):
    """Create temporary model files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create metadata.json
        metadata = {
            "input_variables": ["t2m", "u10m", "v10m", "z500"],
            "output_variables": ["mrr", "t2m", "u10m", "v10m"],
        }
        with open(temp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create stats.json
        stats = {
            "input": {
                "t2m": {"mean": 0.0, "std": 1.0},
                "u10m": {"mean": 0.0, "std": 1.0},
                "v10m": {"mean": 0.0, "std": 1.0},
                "z500": {"mean": 0.0, "std": 1.0},
            },
            "output": {
                "mrr": {"mean": 0.0, "std": 1.0},
                "t2m": {"mean": 0.0, "std": 1.0},
                "u10m": {"mean": 0.0, "std": 1.0},
                "v10m": {"mean": 0.0, "std": 1.0},
            },
        }
        with open(temp_path / "stats.json", "w") as f:
            json.dump(stats, f)

        # Create latlon_grid.nc
        input_lat = np.linspace(19.25, 28, 36)
        input_lon = np.linspace(116, 126, 40, endpoint=False)
        if request.param == "rectangular":
            ds = xr.Dataset({}, coords={"lat": input_lat, "lon": input_lon})
        else:
            # 2D curvilinear input grid
            base_lat_grid, base_lon_grid = np.meshgrid(
                input_lat, input_lon, indexing="ij"
            )
            input_lat_grid = base_lat_grid + 0.01 * np.sin(np.deg2rad(base_lon_grid))
            input_lon_grid = base_lon_grid
            ds = xr.Dataset(
                {
                    "lat": (["x", "y"], input_lat_grid),
                    "lon": (["x", "y"], input_lon_grid),
                },
                coords={"x": np.arange(36), "y": np.arange(40)},
            )
        ds.to_netcdf(temp_path / "input_latlon_grid.nc")

        output_lat = np.linspace(24.25, 27.75, 320)
        output_lon = np.linspace(116.25, 119.75, 320)

        if request.param == "rectangular":
            # Store as 1D arrays for rectilinear grids
            ds = xr.Dataset({}, coords={"lat": output_lat, "lon": output_lon})
        else:
            # Store as 2D arrays for truly curvilinear grids
            base_lat_grid, base_lon_grid = np.meshgrid(
                output_lat, output_lon, indexing="ij"
            )
            # Add a small longitude-dependent perturbation to latitude so the grid is
            # no longer strictly rectilinear but still within input bounds
            output_lat_grid = base_lat_grid + 0.01 * np.sin(np.deg2rad(base_lon_grid))
            output_lon_grid = base_lon_grid
            ds = xr.Dataset(
                {
                    "lat": (["x", "y"], output_lat_grid),
                    "lon": (["x", "y"], output_lon_grid),
                },
                coords={"x": np.arange(320), "y": np.arange(320)},
            )
        ds.to_netcdf(temp_path / "output_latlon_grid.nc")

        yield temp_path


@pytest.fixture(params=["rectangular", "curvilinear"])
def temp_model_files_with_invariants(request):
    """Create temporary model files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create metadata.json
        metadata = {
            "input_variables": ["t2m", "u10m", "v10m", "z500"],
            "output_variables": ["mrr", "t2m", "u10m", "v10m"],
        }
        with open(temp_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create stats.json
        stats = {
            "input": {
                "t2m": {"mean": 0.0, "std": 1.0},
                "u10m": {"mean": 0.0, "std": 1.0},
                "v10m": {"mean": 0.0, "std": 1.0},
                "z500": {"mean": 0.0, "std": 1.0},
            },
            "output": {
                "mrr": {"mean": 0.0, "std": 1.0},
                "t2m": {"mean": 0.0, "std": 1.0},
                "u10m": {"mean": 0.0, "std": 1.0},
                "v10m": {"mean": 0.0, "std": 1.0},
            },
            "invariants": {
                "orography": {"mean": 0.0, "std": 1.0},
                "landsea_mask": {"mean": 0.0, "std": 1.0},
            },
        }
        with open(temp_path / "stats.json", "w") as f:
            json.dump(stats, f)

        # Create latlon_grid.nc
        input_lat = np.linspace(19.25, 28, 36)
        input_lon = np.linspace(116, 126, 40, endpoint=False)
        if request.param == "rectangular":
            ds = xr.Dataset({}, coords={"lat": input_lat, "lon": input_lon})
        else:
            # 2D curvilinear input grid
            base_lat_grid, base_lon_grid = np.meshgrid(
                input_lat, input_lon, indexing="ij"
            )
            input_lat_grid = base_lat_grid + 0.01 * np.sin(np.deg2rad(base_lon_grid))
            input_lon_grid = base_lon_grid
            ds = xr.Dataset(
                {
                    "lat": (["x", "y"], input_lat_grid),
                    "lon": (["x", "y"], input_lon_grid),
                },
                coords={"x": np.arange(36), "y": np.arange(40)},
            )
        ds.to_netcdf(temp_path / "input_latlon_grid.nc")

        output_lat = np.linspace(24.25, 27.75, 320)
        output_lon = np.linspace(116.25, 119.75, 320)

        if request.param == "rectangular":
            # Store as 1D arrays for rectilinear grids
            ds = xr.Dataset({}, coords={"lat": output_lat, "lon": output_lon})
            ds_inv = xr.Dataset(
                {
                    "orography": (["lat", "lon"], np.random.randn(320, 320)),
                    "landsea_mask": (["lat", "lon"], np.random.randn(320, 320)),
                },
                coords={"lat": output_lat, "lon": output_lon},
            )
        else:
            # Store as 2D arrays for truly curvilinear grids
            base_lat_grid, base_lon_grid = np.meshgrid(
                output_lat, output_lon, indexing="ij"
            )
            output_lat_grid = base_lat_grid + 0.01 * np.sin(np.deg2rad(base_lon_grid))
            output_lon_grid = base_lon_grid
            ds = xr.Dataset(
                {
                    "lat": (["x", "y"], output_lat_grid),
                    "lon": (["x", "y"], output_lon_grid),
                },
                coords={"x": np.arange(320), "y": np.arange(320)},
            )
            ds_inv = xr.Dataset(
                {
                    "orography": (["lat", "lon"], np.random.randn(320, 320)),
                    "landsea_mask": (["lat", "lon"], np.random.randn(320, 320)),
                    "lat": (["x", "y"], output_lat_grid),
                    "lon": (["x", "y"], output_lon_grid),
                },
                coords={"x": np.arange(320), "y": np.arange(320)},
            )
        ds.to_netcdf(temp_path / "output_latlon_grid.nc")
        ds_inv.to_netcdf(temp_path / "invariants.nc")

        yield temp_path


class TestCorrDiffForward:
    @pytest.mark.parametrize(
        "x",
        [
            # Single batch, single sample
            torch.randn(1, 4, 36, 40),
            # Multiple batch, single sample
            torch.randn(2, 4, 36, 40),
            # Single batch, multiple samples
            torch.randn(1, 4, 36, 40),
            # Multiple batch, multiple samples
            torch.randn(3, 4, 36, 40),
        ],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    @pytest.mark.parametrize("number_of_samples", [1])  # , 2])
    @pytest.mark.parametrize("solver", ["euler"])  # , "heun"])
    @pytest.mark.parametrize("sampler_type", ["deterministic"])  # , "stochastic"])
    @pytest.mark.parametrize(
        "inference_mode", ["regression"]
    )  # , "diffusion", "both"])
    def test_corrdiff_basic(
        self,
        x,
        device,
        number_of_samples,
        solver,
        sampler_type,
        inference_mode,
        mock_residual_model,
        mock_regression_model,
        sample_model_params,
    ):
        """Test basic functionality of CorrDiff model with various configurations."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Skip deterministic sampler with hr_mean_conditioning=True
        if sampler_type == "deterministic" and inference_mode != "regression":
            pytest.skip(
                "Deterministic sampler not implemented with hr_mean_conditioning"
            )

        params = sample_model_params.copy()
        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            number_of_samples=number_of_samples,
            solver=solver,
            sampler_type=sampler_type,
            inference_mode=inference_mode,
            hr_mean_conditioning=False,  # Disable for deterministic sampler compatibility
            **params,
        )
        model.to(device)

        x = x.to(device)

        coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "variable": model.input_coords()["variable"],
                "lat": model.input_coords()["lat"],
                "lon": model.input_coords()["lon"],
            }
        )

        # Run model
        out, out_coords = model(x, coords)

        # Check output shape
        expected_shape = (
            x.shape[0],
            number_of_samples,
            len(params["output_variables"]),
            320,
            320,
        )
        assert out.shape == expected_shape

        # Check output coordinates
        assert out_coords["variable"].shape == (len(params["output_variables"]),)
        assert out_coords["sample"].shape == (number_of_samples,)
        assert out_coords["lat"].shape == (320, 320)
        assert out_coords["lon"].shape == (320, 320)

        # Verify coordinate dimensions
        handshake_dim(out_coords, "lon", 4)
        handshake_dim(out_coords, "lat", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "sample", 1)
        handshake_dim(out_coords, "batch", 0)

    @pytest.mark.parametrize(
        "invalid_coords",
        [
            OrderedDict({"batch": np.array([0]), "variable": np.array(["wrong_var"])}),
            OrderedDict(
                {"batch": np.array([0]), "variable": np.array(["t2m"])}
            ),  # Missing variables
            OrderedDict(
                {"batch": np.array([0]), "variable": np.array(["t2m", "u10m"])}
            ),  # Incomplete variables
        ],
    )
    def test_corrdiff_invalid_coords(
        self,
        invalid_coords,
        mock_residual_model,
        mock_regression_model,
        sample_model_params,
    ):
        """Test CorrDiff model with invalid coordinates."""
        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            **sample_model_params,
        )

        x = torch.randn((1, len(invalid_coords["variable"]), 36, 40))
        with pytest.raises(ValueError):
            model(x, invalid_coords)

    def test_corrdiff_time_coord_type_validation(
        self,
        mock_residual_model,
        mock_regression_model,
        sample_model_params,
    ):
        """Validate ``coords["time"]`` type when provided.

        CorrDiff interprets ``coords["time"]`` as a per-sample timestamp and passes it
        to subclasses as ``valid_time`` (to avoid confusion with forecast init times).
        """
        params = sample_model_params.copy()
        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            number_of_samples=1,
            solver="euler",
            sampler_type="stochastic",
            inference_mode="regression",
            hr_mean_conditioning=False,
            **params,
        )

        x = torch.randn((1, len(params["input_variables"]), 36, 40))
        coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "variable": model.input_coords()["variable"],
                "lat": model.input_coords()["lat"],
                "lon": model.input_coords()["lon"],
            }
        )

        # Wrong dtype: ints are not datetime-like
        bad_coords = coords.copy()
        bad_coords["time"] = np.array([0])
        with pytest.raises(TypeError):
            # `CorrDiff.__call__` is decorated with `@batch_func()`, which enforces that
            # `len(coords)` matches `x.ndim`. Since "time" is an optional *metadata*
            # key (not a tensor dimension), we call the undecorated implementation
            # here to unit-test the time validation logic directly.
            model.__call__.__wrapped__(model, x, bad_coords)

        # Accepted dtype: numpy datetime64 per batch element
        ok_coords = coords.copy()
        ok_coords["time"] = np.array(["2020-01-01T00:00:00"], dtype="datetime64[ns]")
        out, _ = model.__call__.__wrapped__(model, x, ok_coords)
        assert out.shape[0] == 1

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_corrdiff_exceptions(
        self, device, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test exception handling for invalid inputs."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            **sample_model_params,
        ).to(device)

        x = torch.randn((1, 4, 36, 40), device=device)
        coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": model.input_coords()["variable"],
                "lat": model.input_coords()["lat"],
                "lon": model.input_coords()["lon"],
            }
        )

        # Test invalid coordinate dimensions
        wrong_coords = coords.copy()
        wrong_coords["lat"] = np.linspace(90, -90, 720)  # Wrong lat dimension
        with pytest.raises(ValueError):
            model(x, wrong_coords)

        # Test missing required coordinates
        wrong_coords = coords.copy()
        del wrong_coords["lat"]
        with pytest.raises(ValueError):
            model(x, wrong_coords)

        # Test incorrect latlon_grid
        with pytest.raises(ValueError):
            lat_input_grid = model.lat_input_grid + 100
            model._check_latlon_grid(
                lat_input_grid,
                model.lon_input_grid,
                model.lat_output_grid,
                model.lon_output_grid,
            )

        with pytest.raises(ValueError):
            lon_input_grid = model.lon_input_grid + 100
            model._check_latlon_grid(
                model.lat_input_grid,
                lon_input_grid,
                model.lat_output_grid,
                model.lon_output_grid,
            )

        # For regular (1D) grids, corrupting a single value should break validation.
        # For curvilinear (2D) grids we currently only enforce global bounds, so this
        # local corruption is not guaranteed to raise.
        lat_input_grid = model.lat_input_grid.clone()
        lat_input_grid[0] = 100
        if lat_input_grid.ndim == 1:
            with pytest.raises(ValueError):
                model._check_latlon_grid(
                    lat_input_grid,
                    model.lon_input_grid,
                    model.lat_output_grid,
                    model.lon_output_grid,
                )
        else:
            # Just call to ensure the function runs without crashing
            model._check_latlon_grid(
                lat_input_grid,
                model.lon_input_grid,
                model.lat_output_grid,
                model.lon_output_grid,
            )

    def test_corrdiff_infer_input_latlon_grid(
        self, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test input lat/lon grid inference."""
        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            **sample_model_params,
        )

        lat_input_grid, lon_input_grid = model._infer_input_latlon_grid(
            model.lat_output_grid, model.lon_output_grid, 0.25
        )
        true_lat_input_grid = torch.arange(24, 28.0 + 0.25, 0.25)
        true_lon_input_grid = torch.arange(116, 120.0 + 0.25, 0.25)
        assert torch.allclose(lat_input_grid, true_lat_input_grid)
        assert torch.allclose(lon_input_grid, true_lon_input_grid)

    def test_corrdiff_parameter_validation(
        self, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test parameter validation in CorrDiff constructor."""

        # Test invalid number_of_samples
        with pytest.raises(ValueError, match="must be a positive integer"):
            CorrDiff(
                residual_model=mock_residual_model,
                regression_model=mock_regression_model,
                number_of_samples=0,
                **sample_model_params,
            )

        # Test invalid number_of_steps
        with pytest.raises(ValueError, match="must be a positive integer"):
            CorrDiff(
                residual_model=mock_residual_model,
                regression_model=mock_regression_model,
                number_of_steps=0,
                **sample_model_params,
            )

        # Test invalid solver
        with pytest.raises(ValueError, match="is not supported"):
            CorrDiff(
                residual_model=mock_residual_model,
                regression_model=mock_regression_model,
                solver="invalid_solver",
                **sample_model_params,
            )

    def test_corrdiff_with_invariants(
        self, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test CorrDiff model with invariant features."""
        params = sample_model_params.copy()

        # Add invariants
        invariants = OrderedDict(
            {"orography": torch.randn(320, 320), "landsea_mask": torch.randn(320, 320)}
        )
        params["invariants"] = invariants
        params["invariant_center"] = torch.zeros(2)
        params["invariant_scale"] = torch.ones(2)

        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            **params,
        )

        x = torch.randn(1, 4, 36, 40)
        coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": model.input_coords()["variable"],
                "lat": model.input_coords()["lat"],
                "lon": model.input_coords()["lon"],
            }
        )

        out, out_coords = model(x, coords)

        # Check that invariants are properly handled
        assert model.invariants is not None
        assert model.invariant_variables == ["orography", "landsea_mask"]
        assert out.shape == (1, 1, 4, 320, 320)

    def test_corrdiff_preprocessing_postprocessing(
        self, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test preprocessing and postprocessing methods."""
        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            **sample_model_params,
        )

        # Test preprocessing with 3D input [C, H_in, W_in]
        x_3d = torch.randn(4, 36, 40)
        preprocessed_3d = model.preprocess_input(x_3d)
        assert preprocessed_3d.shape == (1, 4, 320, 320)

        # Test preprocessing with 4D input [B, C, H_in, W_in] for backwards compatibility
        x_4d = torch.randn(2, 4, 36, 40)
        preprocessed_4d = model.preprocess_input(x_4d)
        assert preprocessed_4d.shape == (2, 4, 320, 320)

        # Test postprocessing - operates on output grid shape
        output = torch.randn(1, 4, 320, 320)
        postprocessed = model.postprocess_output(output)
        assert postprocessed.shape == (1, 4, 320, 320)

    def test_corrdiff_input_output_coords(
        self, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test input and output coordinate systems."""
        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            **sample_model_params,
        )

        # Test input coordinates
        input_coords = model.input_coords()
        assert "batch" in input_coords
        assert "variable" in input_coords
        assert "lat" in input_coords
        assert "lon" in input_coords
        assert len(input_coords["variable"]) == len(
            sample_model_params["input_variables"]
        )

        # Test output coordinates
        output_coords = model.output_coords(input_coords)
        assert "batch" in output_coords
        assert "sample" in output_coords
        assert "variable" in output_coords
        assert "lat" in output_coords
        assert "lon" in output_coords
        assert len(output_coords["variable"]) == len(
            sample_model_params["output_variables"]
        )

    def test_corrdiff_seed_reproducibility(
        self, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test that seed parameter provides reproducible results."""
        params = sample_model_params.copy()
        seed = 42

        # Create two models with the same seed
        model1 = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            seed=seed,
            **params,
        )

        model2 = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            seed=seed,
            **params,
        )

        x = torch.randn(1, 4, 36, 40)
        coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": model1.input_coords()["variable"],
                "lat": model1.input_coords()["lat"],
                "lon": model1.input_coords()["lon"],
            }
        )

        # Results should be identical with same seed
        out1, _ = model1(x, coords)
        out2, _ = model2(x, coords)

        # Note: This test might fail if the mock models have non-deterministic behavior
        # In practice, with real models, this should pass
        torch.testing.assert_close(out1, out2)

    def test_corrdiff_load_default_package(
        self,
    ):
        """Test that load_default_package raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            CorrDiff.load_default_package()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_corrdiff_interpolation(
        self, mock_residual_model, mock_regression_model, sample_model_params, device
    ):
        """Test interpolation functionality."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            **sample_model_params,
        ).to(device)

        x = torch.randn(1, 4, 36, 40, device=device)

        # Test interpolation
        interpolated = model._interpolate(x)
        assert interpolated.shape == (1, 4, 320, 320)
        assert interpolated.device == x.device

    def test_corrdiff_sampler_setup(
        self, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test sampler setup with different sampler types."""

        # Test deterministic sampler
        model_det = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            sampler_type="deterministic",
            hr_mean_conditioning=False,  # Must be False for deterministic
            **sample_model_params,
        )
        assert model_det.sampler is not None

        # Test stochastic sampler
        model_stoch = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            sampler_type="stochastic",
            **sample_model_params,
        )
        assert model_stoch.sampler is not None

        # Test invalid sampler type
        with pytest.raises(ValueError, match="Unknown sampler type"):
            CorrDiff(
                residual_model=mock_residual_model,
                regression_model=mock_regression_model,
                sampler_type="invalid",
                **sample_model_params,
            )

    def test_corrdiff_inference_modes(
        self, mock_residual_model, mock_regression_model, sample_model_params
    ):
        """Test different inference modes."""
        x = torch.randn(1, 4, 36, 40)

        lat_grid, lon_grid = (
            sample_model_params["lat_input_grid"].cpu().numpy(),
            sample_model_params["lon_input_grid"].cpu().numpy(),
        )
        coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": np.array(sample_model_params["input_variables"]),
                "lat": lat_grid,
                "lon": lon_grid,
            }
        )

        # Test regression mode
        model_reg = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            inference_mode="regression",
            **sample_model_params,
        )
        out_reg, _ = model_reg(x, coords)
        assert out_reg.shape == (1, 1, 4, 320, 320)

        # Test diffusion mode
        model_diff = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            inference_mode="diffusion",
            **sample_model_params,
        )
        out_diff, _ = model_diff(x, coords)
        assert out_diff.shape == (1, 1, 4, 320, 320)

        # Test both mode
        model_both = CorrDiff(
            residual_model=mock_residual_model,
            regression_model=mock_regression_model,
            inference_mode="both",
            **sample_model_params,
        )
        out_both, _ = model_both(x, coords)
        assert out_both.shape == (1, 1, 4, 320, 320)


class TestCorrDiffLoadModel:
    @patch("earth2studio.models.dx.corrdiff.PhysicsNemoModule", MockPhysicsNemoModule)
    @patch("earth2studio.models.dx.corrdiff.StackedRandomGenerator", MagicMock())
    @patch("earth2studio.models.dx.corrdiff.deterministic_sampler", MagicMock())
    def test_corrdiff_load_model_basic(self, mock_package, temp_model_files):
        """Test basic load_model functionality."""

        # Configure mock package to return our temp files.
        # Simulate missing invariants.nc (no invariants for basic case).
        def resolve_side_effect(path: str) -> str:
            if Path(path).name == "invariants.nc":
                raise FileNotFoundError
            return str(temp_model_files / Path(path).name)

        mock_package.resolve.side_effect = resolve_side_effect

        # Load model
        model = CorrDiff.load_model(mock_package)

        # Verify model was created correctly
        assert isinstance(model, CorrDiff)
        assert model.input_variables == ["t2m", "u10m", "v10m", "z500"]
        assert model.output_variables == ["mrr", "t2m", "u10m", "v10m"]
        assert model.invariants is None
        assert model.invariant_variables == []

    @patch("earth2studio.models.dx.corrdiff.PhysicsNemoModule", MockPhysicsNemoModule)
    @patch("earth2studio.models.dx.corrdiff.StackedRandomGenerator", MagicMock())
    @patch("earth2studio.models.dx.corrdiff.deterministic_sampler", MagicMock())
    def test_corrdiff_load_model_device_calls_model_to_before_channels_last(
        self, mock_package, temp_model_files
    ):
        """Ensure `load_model(device=...)` applies `.to(device)` before `channels_last`.
        Why this matters:
        - `channels_last` conversion should happen after moving the model to the target
          device, because the memory-format optimization is device-dependent.
        - This test makes that behavior explicit by checking the order of `.to(...)`
          calls on the mocked PhysicsNemo models (residual + regression).
        """
        MockPhysicsNemoModule.created.clear()

        def resolve_side_effect(path: str) -> str:
            if Path(path).name == "invariants.nc":
                raise FileNotFoundError
            return str(temp_model_files / Path(path).name)

        mock_package.resolve.side_effect = resolve_side_effect

        _ = CorrDiff.load_model(mock_package, device="cpu")

        assert len(MockPhysicsNemoModule.created) == 2
        for m in MockPhysicsNemoModule.created:
            # Expect: to(device) first, then to(memory_format=channels_last)
            assert m.to_calls[0] == ("cpu", None)
            assert m.to_calls[1][0] is None
            assert m.to_calls[1][1] is torch.channels_last

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @patch("earth2studio.models.dx.corrdiff.PhysicsNemoModule", MockPhysicsNemoModule)
    @patch("earth2studio.models.dx.corrdiff.StackedRandomGenerator", MagicMock())
    @patch("earth2studio.models.dx.corrdiff.deterministic_sampler", MagicMock())
    def test_corrdiff_load_model_device_places_buffers_on_cuda(
        self, mock_package, temp_model_files
    ):
        """If CUDA is available, verify `load_model(device='cuda:0')` places buffers on GPU.
        This complements the CPU-only call-order test by validating actual tensor placement
        on CUDA (when available). We check representative buffers that must match the
        model's device to avoid implicit CPU↔GPU transfers during inference.
        """
        MockPhysicsNemoModule.created.clear()

        def resolve_side_effect(path: str) -> str:
            if Path(path).name == "invariants.nc":
                raise FileNotFoundError
            return str(temp_model_files / Path(path).name)

        mock_package.resolve.side_effect = resolve_side_effect

        model = CorrDiff.load_model(mock_package, device="cuda:0")
        assert model.in_center.device.type == "cuda"
        assert model.lat_output_grid.device.type == "cuda"

    @patch("earth2studio.models.dx.corrdiff.PhysicsNemoModule", MockPhysicsNemoModule)
    @patch("earth2studio.models.dx.corrdiff.StackedRandomGenerator", MagicMock())
    @patch("earth2studio.models.dx.corrdiff.deterministic_sampler", MagicMock())
    def test_corrdiff_load_model_with_invariants(
        self, mock_package, temp_model_files_with_invariants
    ):
        """Test load_model functionality with invariant features."""
        # Configure mock package to return our temp files
        mock_package.resolve.side_effect = lambda path: str(
            temp_model_files_with_invariants / Path(path).name
        )

        # Load model
        model = CorrDiff.load_model(mock_package)

        # Verify model was created correctly
        assert isinstance(model, CorrDiff)
        assert model.input_variables == ["t2m", "u10m", "v10m", "z500"]
        assert model.output_variables == ["mrr", "t2m", "u10m", "v10m"]
        assert model.invariants is not None
        assert model.invariant_variables == ["orography", "landsea_mask"]
        assert model.invariants.shape == (2, 320, 320)

    @patch("earth2studio.models.dx.corrdiff.PhysicsNemoModule", None)
    @patch("earth2studio.models.dx.corrdiff.StackedRandomGenerator", None)
    @patch("earth2studio.models.dx.corrdiff.deterministic_sampler", None)
    def test_corrdiff_load_model_missing_dependencies(self, mock_package):
        """Test load_model raises ImportError when dependencies are missing."""
        with pytest.raises(
            ImportError,
            match="Additional CorrDiff model dependencies are not installed",
        ):
            CorrDiff.load_model(mock_package)

    @patch("earth2studio.models.dx.corrdiff.PhysicsNemoModule", MockPhysicsNemoModule)
    @patch("earth2studio.models.dx.corrdiff.StackedRandomGenerator", MagicMock())
    @patch("earth2studio.models.dx.corrdiff.deterministic_sampler", MagicMock())
    def test_corrdiff_load_model_missing_files(self, mock_package, temp_model_files):
        """Test load_model handles missing files gracefully."""
        # Configure mock package to return non-existent files
        mock_package.resolve.side_effect = lambda path: "/non/existent/path"

        with pytest.raises(FileNotFoundError):
            CorrDiff.load_model(mock_package)

    @patch("earth2studio.models.dx.corrdiff.PhysicsNemoModule", MockPhysicsNemoModule)
    @patch("earth2studio.models.dx.corrdiff.StackedRandomGenerator", MagicMock())
    @patch("earth2studio.models.dx.corrdiff.deterministic_sampler", MagicMock())
    def test_corrdiff_load_model_invalid_metadata(self, mock_package, temp_model_files):
        """Test load_model with invalid metadata."""
        # Modify metadata to be invalid
        with open(temp_model_files / "metadata.json", "w") as f:
            json.dump({"invalid": "metadata"}, f)

        mock_package.resolve.side_effect = lambda path: str(
            temp_model_files / Path(path).name
        )

        with pytest.raises(KeyError):
            CorrDiff.load_model(mock_package)

    @patch("earth2studio.models.dx.corrdiff.PhysicsNemoModule", MockPhysicsNemoModule)
    @patch("earth2studio.models.dx.corrdiff.StackedRandomGenerator", MagicMock())
    @patch("earth2studio.models.dx.corrdiff.deterministic_sampler", MagicMock())
    def test_corrdiff_load_model_invalid_stats(self, mock_package, temp_model_files):
        """Test load_model with invalid stats."""
        # Modify stats to be invalid
        with open(temp_model_files / "stats.json", "w") as f:
            json.dump({"invalid": "stats"}, f)

        mock_package.resolve.side_effect = lambda path: str(
            temp_model_files / Path(path).name
        )

        with pytest.raises(KeyError):
            CorrDiff.load_model(mock_package)

    def test_corrdiff_hr_mean_conditioning_validation(self):
        """Test validation of hr_mean_conditioning parameter."""
        # Test that hr_mean_conditioning=True with deterministic sampler raises error
        lat_input_grid = torch.linspace(19.25, 28, 36)
        lon_input_grid = torch.linspace(116, 126, 41)[:-1]
        lat_output_grid = torch.linspace(24.25, 27.75, 320)
        lon_output_grid = torch.linspace(116.25, 119.75, 320)
        lat_output_grid, lon_output_grid = torch.meshgrid(
            lat_output_grid, lon_output_grid, indexing="ij"
        )
        with pytest.raises(
            NotImplementedError,
            match="High-res mean conditioning is not yet implemented",
        ):
            CorrDiff(
                input_variables=["t2m", "u10m"],
                output_variables=["mrr", "t2m"],
                residual_model=MockPhysicsNemoModule(),
                regression_model=MockPhysicsNemoModule(),
                lat_input_grid=lat_input_grid,
                lon_input_grid=lon_input_grid,
                lat_output_grid=lat_output_grid,
                lon_output_grid=lon_output_grid,
                in_center=torch.zeros(2),
                in_scale=torch.ones(2),
                invariant_center=torch.zeros(0),
                invariant_scale=torch.ones(0),
                out_center=torch.zeros(2),
                out_scale=torch.ones(2),
                sampler_type="deterministic",
                hr_mean_conditioning=True,
            )

    def test_corrdiff_buffer_registration(self):
        """Test that model buffers are properly registered."""

        lat_input_grid = torch.linspace(19.25, 28, 36)
        lon_input_grid = torch.linspace(116, 126, 41)[:-1]
        lat_output_grid = torch.linspace(24.25, 27.75, 320)
        lon_output_grid = torch.linspace(116.25, 119.75, 320)
        lat_output_grid, lon_output_grid = torch.meshgrid(
            lat_output_grid, lon_output_grid, indexing="ij"
        )
        model = CorrDiff(
            input_variables=["t2m", "u10m"],
            output_variables=["mrr", "t2m"],
            residual_model=MockPhysicsNemoModule(),
            regression_model=MockPhysicsNemoModule(),
            lat_input_grid=lat_input_grid,
            lon_input_grid=lon_input_grid,
            lat_output_grid=lat_output_grid,
            lon_output_grid=lon_output_grid,
            in_center=torch.zeros(2),
            in_scale=torch.ones(2),
            invariant_center=torch.zeros(0),
            invariant_scale=torch.ones(0),
            out_center=torch.zeros(2),
            out_scale=torch.ones(2),
        )

        # Check that buffers are registered
        assert hasattr(model, "lat_input_grid")
        assert hasattr(model, "lon_input_grid")
        assert hasattr(model, "lat_output_grid")
        assert hasattr(model, "lon_output_grid")
        assert hasattr(model, "in_center")
        assert hasattr(model, "in_scale")
        assert hasattr(model, "out_center")
        assert hasattr(model, "out_scale")

        # Check buffer shapes
        assert model.in_center.shape == (1, 2, 1, 1)
        assert model.in_scale.shape == (1, 2, 1, 1)
        assert model.out_center.shape == (1, 2, 1, 1)
        assert model.out_scale.shape == (1, 2, 1, 1)

    def test_corrdiff_buffer_registration_with_invariants(self):
        """Test that model buffers are properly registered with invariants."""
        invariants = {
            "orography": torch.randn(128, 128),
            "landsea_mask": torch.randn(128, 128),
        }

        lat_input_grid = torch.linspace(19.25, 28, 36)
        lon_input_grid = torch.linspace(116, 126, 41)[:-1]
        lat_output_grid = torch.linspace(24.25, 27.75, 320)
        lon_output_grid = torch.linspace(116.25, 119.75, 320)
        lat_output_grid, lon_output_grid = torch.meshgrid(
            lat_output_grid, lon_output_grid, indexing="ij"
        )
        model = CorrDiff(
            input_variables=["t2m", "u10m"],
            output_variables=["mrr", "t2m"],
            residual_model=MockPhysicsNemoModule(),
            regression_model=MockPhysicsNemoModule(),
            lat_input_grid=lat_input_grid,
            lon_input_grid=lon_input_grid,
            lat_output_grid=lat_output_grid,
            lon_output_grid=lon_output_grid,
            in_center=torch.zeros(2),
            in_scale=torch.ones(2),
            invariant_center=torch.zeros(2),
            invariant_scale=torch.ones(2),
            out_center=torch.zeros(2),
            out_scale=torch.ones(2),
            invariants=invariants,
        )

        # Check that invariants are registered
        assert hasattr(model, "invariants")
        assert model.invariants is not None
        assert model.invariant_variables == ["orography", "landsea_mask"]

        # Check that input normalization includes invariants
        assert model.in_center.shape == (1, 4, 1, 1)  # 2 input + 2 invariant
        assert model.in_scale.shape == (1, 4, 1, 1)  # 2 input + 2 invariant

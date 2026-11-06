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

"""Tests for SuperResolution generative diagnostic model."""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import SuperResolution
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


class PhooSuperResolution(torch.nn.Module):
    """Dummy super-resolution model for testing."""

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Input: [1, 4, 91, 180] + noise -> Output: [1, 4, 361, 720]
        out = torch.nn.functional.interpolate(
            x, size=(361, 720), mode="bilinear", align_corners=True
        )
        return out + 0.01 * noise


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create dummy package for testing."""
    tmp_path = tmp_path_factory.mktemp("model_data")

    torch.save(PhooSuperResolution(), tmp_path / "model.pt")

    # Save normalization parameters
    np.save(tmp_path / "in_center.npy", np.zeros(4))
    np.save(tmp_path / "in_scale.npy", np.ones(4))
    np.save(tmp_path / "out_center.npy", np.zeros(4))
    np.save(tmp_path / "out_scale.npy", np.ones(4))

    return Package(str(tmp_path))


def make_input(model, batch: int = 1, device: str = "cpu") -> xr.DataArray:
    signature = coord_array_like(model.input_coords(), {"batch": np.arange(batch)})
    return from_torch(torch.randn(signature.shape, device=device), signature)


class TestSuperResolutionMock:
    """Mock tests using dummy model weights."""

    @pytest.mark.parametrize("number_of_samples", [1, 3])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_super_resolution_samples(self, test_package, number_of_samples, device):
        """Test that correct number of samples are generated."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = SuperResolution.load_model(
            test_package, number_of_samples=number_of_samples
        ).to(device)
        x = make_input(dx, 1, device)

        out = dx(x)

        # Output dims: [batch, sample, variable, lat, lon] on the fine grid
        assert out.dims == ("batch", "sample", "variable", "lat", "lon")
        assert out.shape == (1, number_of_samples, 4, 361, 720)
        np.testing.assert_array_equal(
            out.coords["sample"].values, np.arange(number_of_samples)
        )
        handshake_dataarray(out, dx.output_coords(x))

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_super_resolution_deterministic_seed(self, test_package, device):
        """Test reproducibility with set_rng."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = SuperResolution.load_model(test_package, number_of_samples=2).to(device)
        x = make_input(dx, 1, device)

        dx.set_rng(42)
        out1 = dx(x).e2s.as_numpy()
        dx.set_rng(42)
        out2 = dx(x).e2s.as_numpy()
        # Same seed should produce same output
        xr.testing.assert_identical(out1, out2)

        # Different seed should produce different output
        dx.set_rng(123)
        out3 = dx(x).e2s.as_numpy()
        assert not np.array_equal(out1.values, out3.values)

        # Samples within one call differ
        assert not np.array_equal(out1.values[:, 0], out1.values[:, 1])

    def test_super_resolution_no_seed_argument(self, test_package):
        """Seeding is only available through set_rng."""
        with pytest.raises(TypeError):
            SuperResolution.load_model(test_package, seed=1)

    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_super_resolution_call(self, test_package, batch, device):
        """Test forward pass with different batch sizes."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = SuperResolution.load_model(test_package, number_of_samples=1).to(device)
        x = make_input(dx, batch, device)
        before = x.copy(deep=True)

        out = dx(x)

        assert out.shape == (batch, 1, 4, 361, 720)
        assert list(out.coords["variable"].values) == ["t2m", "u10m", "v10m", "msl"]
        xr.testing.assert_identical(x.e2s.as_numpy(), before.e2s.as_numpy())

    def test_super_resolution_exceptions(self, test_package):
        """Test SuperResolution raises on invalid inputs."""
        dx = SuperResolution.load_model(test_package)
        x = make_input(dx)

        # Wrong variable names
        with pytest.raises((KeyError, ValueError)):
            dx(x.assign_coords(variable=["a", "b", "c", "d"]))

        # Wrong dimension order
        with pytest.raises(ValueError):
            dx(x.transpose(*reversed(x.dims)))

        # Wrong grid size
        with pytest.raises(ValueError):
            dx(x.isel(lat=slice(0, 45)))

        # Invalid sample count
        with pytest.raises(ValueError):
            SuperResolution.load_model(test_package, number_of_samples=0)

    def test_super_resolution_conformance(self, test_package):
        """Test SuperResolution against the native diagnostic contract."""
        dx = SuperResolution.load_model(test_package, number_of_samples=2)
        assert isinstance(dx, DiagnosticModel)
        assert dx.stochastic is True
        assert check_diagnostic_contract(dx) == []

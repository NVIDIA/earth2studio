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

"""Tests for PrecipEstimator diagnostic model."""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import PrecipEstimator
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.utils import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


class PhooPrecipEstimator(torch.nn.Module):
    """Dummy model for testing PrecipEstimator."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch, 5, lat, lon] -> Output: [batch, 1, lat, lon]
        return x[:, :1, :, :]


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create dummy package for testing."""
    tmp_path = tmp_path_factory.mktemp("model_data")

    # Save dummy model
    model = PhooPrecipEstimator()
    torch.save(model, tmp_path / "model.pt")

    # Save normalization parameters
    np.save(tmp_path / "center.npy", np.zeros(5))
    np.save(tmp_path / "scale.npy", np.ones(5))

    return Package(str(tmp_path))


def make_input(model, batch: int = 1, device: str = "cpu") -> xr.DataArray:
    signature = coord_array_like(model.input_coords(), {"batch": np.arange(batch)})
    return from_torch(torch.randn(signature.shape, device=device), signature)


class TestPrecipEstimatorMock:
    """Mock tests using dummy model weights."""

    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_precip_estimator_call(self, test_package, batch, device):
        """Test forward pass of PrecipEstimator."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")

        dx = PrecipEstimator.load_model(test_package).to(device)
        x = make_input(dx, batch, device)
        before = x.copy(deep=True)

        out = dx(x)

        assert out.shape == (batch, 1, 361, 720)
        assert list(out.coords["variable"].values) == ["tp"]
        handshake_dataarray(out, dx.output_coords(x))
        # Dummy model passes through the first (t2m) channel
        np.testing.assert_allclose(
            out.e2s.as_numpy().values[:, 0], x.e2s.as_numpy().values[:, 0]
        )
        xr.testing.assert_identical(x.e2s.as_numpy(), before.e2s.as_numpy())

    def test_precip_estimator_exceptions(self, test_package):
        """Test PrecipEstimator raises on invalid inputs."""
        dx = PrecipEstimator.load_model(test_package)
        x = make_input(dx)

        # Wrong variable names
        with pytest.raises((KeyError, ValueError)):
            dx(x.assign_coords(variable=["a", "b", "c", "d", "e"]))

        # Wrong dimension order
        with pytest.raises(ValueError):
            dx(x.transpose(*reversed(x.dims)))

        # Wrong grid size
        with pytest.raises(ValueError):
            dx(x.isel(lat=slice(0, 180)))

    def test_precip_estimator_normalization(self, test_package):
        """Test that normalization buffers are registered with the right shape."""
        dx = PrecipEstimator.load_model(test_package)

        assert dx.center.shape == torch.Size([5, 1, 1])
        assert dx.scale.shape == torch.Size([5, 1, 1])
        assert {"center", "scale"} <= {name for name, _ in dx.named_buffers()}

    def test_precip_estimator_conformance(self, test_package):
        """Test PrecipEstimator against the native diagnostic contract."""
        dx = PrecipEstimator.load_model(test_package)
        assert isinstance(dx, DiagnosticModel)
        assert check_diagnostic_contract(dx) == [
            "D10: model does not declare itself stochastic"
        ]

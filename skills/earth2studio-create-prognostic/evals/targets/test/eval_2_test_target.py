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

"""Tests for AdditiveModel prognostic model."""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import AdditiveModel
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


class PhooAdditiveModel(torch.nn.Module):
    """Dummy additive model for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.01


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create dummy package with model and normalization files."""
    tmp_path = tmp_path_factory.mktemp("model_data")
    torch.save(PhooAdditiveModel(), tmp_path / "model.pt")
    np.save(tmp_path / "center.npy", np.zeros(4))
    np.save(tmp_path / "scale.npy", np.ones(4))
    return Package(str(tmp_path))


def make_input(model, batch: int = 1, device: str = "cpu") -> xr.DataArray:
    """Concretize the configured signature with a time axis."""
    signature = coord_array_like(model.input_coords(), {"batch": np.arange(batch)})
    signature = signature.expand_dims(
        time=np.array(["2024-01-01"], dtype="datetime64[ns]"), axis=1
    )
    return from_torch(torch.randn(signature.shape, device=device), signature)


class TestAdditiveModelMock:
    """Mock tests using dummy model weights."""

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_additive_call(self, test_package, device):
        """Test single forward pass adds constant."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")
        model = AdditiveModel.load_model(test_package).to(device)
        x = make_input(model, 1, device)
        before = x.copy(deep=True)

        out = model(x)

        assert out.shape == x.shape
        handshake_dataarray(out, model.output_coords(x))
        np.testing.assert_allclose(
            out.e2s.as_numpy().values,
            x.e2s.as_numpy().values + 0.01,
            rtol=1e-5,
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            out.lead_time.values, x.lead_time.values + np.timedelta64(6, "h")
        )
        xr.testing.assert_identical(x.e2s.as_numpy(), before.e2s.as_numpy())

    @pytest.mark.parametrize("ensemble", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_additive_iter(self, test_package, ensemble, device):
        """Test iterator produces correct sequence with accumulating additions."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")
        model = AdditiveModel.load_model(test_package).to(device)
        x = make_input(model, ensemble, device)
        original = x.copy(deep=True)

        iterator = model.create_iterator(x)
        # The iterator yields forecasts only: +6h, +12h, ...
        for i in range(5):
            out = next(iterator)
            assert out.sizes["batch"] == ensemble
            np.testing.assert_array_equal(
                out.lead_time.values, np.array([np.timedelta64((i + 1) * 6, "h")])
            )
            # Additive model accumulates: x + (i + 1) * 0.01
            np.testing.assert_allclose(
                out.e2s.as_numpy().values,
                x.e2s.as_numpy().values + (i + 1) * 0.01,
                rtol=1e-4,
                atol=1e-5,
            )
        iterator.close()
        xr.testing.assert_identical(x.e2s.as_numpy(), original.e2s.as_numpy())

    def test_additive_step_matches_iterator(self, test_package):
        """Test step() on the preceding forecast matches the next iterator yield."""
        model = AdditiveModel.load_model(test_package)
        x = make_input(model)
        y, state = model.initialize(x)
        iterator = model.create_iterator(x)
        next(iterator)
        expected, _ = model.step(y, state=state)
        xr.testing.assert_identical(next(iterator), expected)

    def test_additive_exceptions(self, test_package):
        """Test model raises on invalid inputs."""
        model = AdditiveModel.load_model(test_package)
        x = make_input(model)

        # Wrong variable names
        with pytest.raises((KeyError, ValueError)):
            model(x.assign_coords(variable=["a", "b", "c", "d"]))

        # Wrong dimension order
        with pytest.raises(ValueError):
            model(x.transpose(*reversed(x.dims)))

        # Wrong grid size
        with pytest.raises(ValueError):
            model(x.isel(lat=slice(0, 180)))

    def test_additive_normalization_buffers(self, test_package):
        """Test normalization buffers are loaded from package as buffers."""
        model = AdditiveModel.load_model(test_package)
        assert model.center.shape == (4, 1, 1)
        assert model.scale.shape == (4, 1, 1)
        assert {"center", "scale"} <= {name for name, _ in model.named_buffers()}

    def test_additive_grid_size(self, test_package):
        """Test grid size is 361x720 (0.5 degree)."""
        model = AdditiveModel.load_model(test_package)
        signature = model.input_coords()
        assert signature.sizes["lat"] == 361
        assert signature.sizes["lon"] == 720

    def test_additive_conformance(self, test_package):
        """Test AdditiveModel against the native prognostic contract."""
        model = AdditiveModel.load_model(test_package)
        assert isinstance(model, PrognosticModel)
        assert check_prognostic_contract(model) == [
            "P14: model does not declare itself stochastic",
            "P21: checkpoint serialization requires component-specific tests",
        ]

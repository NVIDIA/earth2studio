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

"""Tests for IdentityModel prognostic model."""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import IdentityModel
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils import coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


def make_input(model, batch: int = 1, device: str = "cpu") -> xr.DataArray:
    """Concretize the configured signature with a time axis."""
    signature = coord_array_like(
        model.input_coords(),
        {"batch": np.arange(batch)},
    )
    signature = signature.expand_dims(
        time=np.array(["2024-01-01"], dtype="datetime64[ns]"), axis=1
    )
    return from_torch(torch.randn(signature.shape, device=device), signature)


class TestIdentityModelMock:
    """Mock tests using the model's built-in identity behaviour."""

    @pytest.mark.parametrize("batch", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_identity_call(self, batch, device):
        """Test single forward pass."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")
        model = IdentityModel.load_model(IdentityModel.load_default_package())
        model = model.to(device)
        x = make_input(model, batch, device)
        before = x.copy(deep=True)

        out = model(x)

        assert out.shape == x.shape
        assert out.dims == x.dims
        handshake_dataarray(out, model.output_coords(x))
        np.testing.assert_array_equal(
            out.lead_time.values, x.lead_time.values + np.timedelta64(6, "h")
        )
        # Identity: values are unchanged and the input is not modified
        np.testing.assert_array_equal(
            out.e2s.as_numpy().values, x.e2s.as_numpy().values
        )
        xr.testing.assert_identical(x.e2s.as_numpy(), before.e2s.as_numpy())

    @pytest.mark.parametrize("ensemble", [1, 2])
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_identity_iter(self, ensemble, device):
        """Test iterator yields forecasts at +6h, +12h, ... (no initial state)."""
        if device == "cuda:0" and not torch.cuda.is_available():
            pytest.skip("No GPU available")
        model = IdentityModel.load_model(IdentityModel.load_default_package())
        model = model.to(device)
        x = make_input(model, ensemble, device)
        original = x.copy(deep=True)

        expected, state = model.initialize(x)
        iterator = model.create_iterator(x)
        first = next(iterator)
        # The iterator yields forecasts only; its first yield is initialize()
        xr.testing.assert_identical(first.e2s.as_numpy(), expected.e2s.as_numpy())
        assert first.sizes["batch"] == ensemble
        np.testing.assert_array_equal(
            first.lead_time.values, np.array([np.timedelta64(6, "h")])
        )

        for i in range(1, 5):
            out = next(iterator)
            assert out.sizes["batch"] == ensemble
            np.testing.assert_array_equal(
                out.lead_time.values, np.array([np.timedelta64((i + 1) * 6, "h")])
            )
            np.testing.assert_array_equal(
                out.e2s.as_numpy().values, x.e2s.as_numpy().values
            )
        iterator.close()
        xr.testing.assert_identical(x.e2s.as_numpy(), original.e2s.as_numpy())

    def test_identity_step_matches_iterator(self):
        """Test step() on the preceding forecast matches the next iterator yield."""
        model = IdentityModel.load_model(IdentityModel.load_default_package())
        x = make_input(model)
        y, state = model.initialize(x)
        iterator = model.create_iterator(x)
        next(iterator)
        expected, _ = model.step(y, state=state)
        xr.testing.assert_identical(next(iterator), expected)

    def test_identity_exceptions(self):
        """Test model raises on invalid inputs."""
        model = IdentityModel.load_model(IdentityModel.load_default_package())
        x = make_input(model)

        # Wrong variable names
        with pytest.raises((KeyError, ValueError)):
            model(x.assign_coords(variable=["a", "b", "c", "d"]))

        # Wrong dimension order
        with pytest.raises(ValueError):
            model(x.transpose(*reversed(x.dims)))

        # Wrong grid size
        with pytest.raises(ValueError):
            model(x.isel(lat=slice(0, 90)))

        # Multiple lead times are not accepted by a one-frame model
        with pytest.raises(ValueError):
            model(xr.concat([x, x], dim="lead_time"))

    def test_identity_conformance(self):
        """Test IdentityModel against the native prognostic contract."""
        model = IdentityModel.load_model(IdentityModel.load_default_package())
        assert isinstance(model, PrognosticModel)
        assert check_prognostic_contract(model) == [
            "P14: model does not declare itself stochastic",
            "P21: checkpoint serialization requires component-specific tests",
        ]

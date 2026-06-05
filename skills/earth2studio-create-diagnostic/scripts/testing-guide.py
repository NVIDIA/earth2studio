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

"""Testing Guide for Diagnostic Model Wrappers.

This file provides test templates for Earth2Studio diagnostic models.
Copy and adapt for your specific model.

Test file location: test/models/dx/test_<model_name>.py

Required tests:
1. test_<model>_call - Forward pass with mock model (parametrize batch, device)
2. test_<model>_exceptions - Invalid coordinates raise errors
3. test_<model>_package - Integration test with real weights (@pytest.mark.package)

Additional tests for generative models:
4. test_<model>_samples - Correct number of samples produced
5. test_<model>_deterministic_seed - Reproducibility with seed

See real examples:
- test/models/dx/test_precip_afno.py (deterministic)
- test/models/dx/test_corrdiff.py (generative)

Use `uv run pytest` to execute tests
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.models.auto import Package
from earth2studio.models.dx import MyDiagnostic  # TODO: Import your model
from earth2studio.utils import handshake_dim


# =============================================================================
# DUMMY MODEL (Phoo Pattern)
# =============================================================================


class PhooDiagnostic(torch.nn.Module):
    """Dummy model that mimics the actual model's interface.

    The Phoo model should:
    1. Accept input in the same shape the real model expects
    2. Produce output in the same shape the real model produces
    3. Perform a simple, deterministic operation

    This allows testing the wrapper's logic without real weights.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 1) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Simple deterministic forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor [batch, in_channels, lat, lon].

        Returns
        -------
        torch.Tensor
            Output tensor [batch, out_channels, lat, lon].
        """
        # Return subset of channels (simulates variable transformation)
        return x[:, : self.out_channels, :, :]


# =============================================================================
# MOCK TESTS: Deterministic Diagnostic
# =============================================================================


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(1, 4, 720, 1440),  # Single batch
        torch.randn(2, 4, 720, 1440),  # Multi batch
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_diagnostic_call(x, device):
    """Test forward pass of diagnostic with mock model.

    Validates:
    - Model accepts input tensor
    - Output has correct shape
    - Output coordinates are valid
    """
    # Skip CUDA tests if not available
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    # Create mock model
    core_model = PhooDiagnostic(in_channels=4, out_channels=1)
    center = torch.zeros(4, 1, 1)
    scale = torch.ones(4, 1, 1)

    # Instantiate wrapper
    dx = MyDiagnostic(core_model, center, scale).to(device)
    x = x.to(device)

    # Create input coordinates
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    # Run forward pass
    out, out_coords = dx(x, coords)

    # Validate output shape
    expected_shape = torch.Size(
        [x.shape[0], 1, 720, 1440]
    )  # [batch, out_var, lat, lon]
    assert (
        out.shape == expected_shape
    ), f"Shape mismatch: {out.shape} != {expected_shape}"

    # Validate output coordinates
    expected_out_var = dx.output_coords(coords)["variable"]
    assert np.array_equal(out_coords["variable"], expected_out_var)

    # Validate dimension ordering
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_diagnostic_exceptions(device):
    """Test model raises on invalid coordinates.

    Validates:
    - Wrong variable names raise KeyError or ValueError
    - Wrong dimension order raises ValueError
    - Wrong coordinate values raise ValueError
    """
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    core_model = PhooDiagnostic()
    center = torch.zeros(4, 1, 1)
    scale = torch.ones(4, 1, 1)

    dx = MyDiagnostic(core_model, center, scale).to(device)
    x = torch.randn(1, 4, 720, 1440).to(device)

    # Test 1: Wrong variable name
    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": np.array(["wrong_var"]),
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )
    with pytest.raises((KeyError, ValueError)):
        dx(x, wrong_coords)

    # Test 2: Wrong dimension order (lon before lat)
    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lon": dx.input_coords()["lon"],  # Wrong order
            "lat": dx.input_coords()["lat"],
        }
    )
    with pytest.raises(ValueError):
        dx(x, wrong_coords)

    # Test 3: Wrong coordinate values (inverted lat)
    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": np.linspace(-90, 90, 720),  # Wrong direction
            "lon": dx.input_coords()["lon"],
        }
    )
    with pytest.raises(ValueError):
        dx(x, wrong_coords)


# =============================================================================
# PACKAGE TEST (Real weights, slow)
# =============================================================================


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_diagnostic_package(device):
    """Integration test with real model weights.

    This test:
    1. Downloads real checkpoint via load_default_package
    2. Runs forward pass with random input
    3. Validates output shape and coordinates

    Marked with @pytest.mark.package to skip in normal test runs.
    Run with: uv run pytest -m package
    """
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    package = MyDiagnostic.load_default_package()
    dx = MyDiagnostic.load_model(package).to(device)

    # Create random input
    x = torch.randn(2, len(dx.input_coords()["variable"]), 720, 1440).to(device)
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    # Run forward pass
    out, out_coords = dx(x, coords)

    # Validate output
    assert out.shape[0] == x.shape[0]  # Batch preserved
    assert np.array_equal(out_coords["variable"], dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)


# =============================================================================
# MOCK TESTS: Generative Diagnostic (CorrDiff-like)
# =============================================================================


class PhooGenerative(torch.nn.Module):
    """Dummy generative model for testing."""

    def __init__(self, out_channels: int = 4, out_lat: int = 320, out_lon: int = 320):
        super().__init__()
        self.out_channels = out_channels
        self.out_lat = out_lat
        self.out_lon = out_lon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor with output shape."""
        batch = x.shape[0]
        return torch.randn(batch, self.out_channels, self.out_lat, self.out_lon)


@pytest.mark.parametrize("number_of_samples", [1, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_generative_samples(number_of_samples, device):
    """Test generative diagnostic produces correct number of samples.

    Validates:
    - Output has sample dimension
    - Sample dimension has correct size
    - All samples have correct shape
    """
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    # TODO: Instantiate your generative diagnostic
    # dx = MyGenerative(..., number_of_samples=number_of_samples).to(device)

    # Create input
    # x = torch.randn(1, 4, 36, 40).to(device)
    # coords = OrderedDict({...})

    # Run forward
    # out, out_coords = dx(x, coords)

    # Validate sample dimension
    # assert "sample" in out_coords
    # assert len(out_coords["sample"]) == number_of_samples
    # assert out.shape[1] == number_of_samples  # [batch, sample, var, lat, lon]
    pass  # TODO: Implement


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_generative_deterministic_seed(device):
    """Test generative diagnostic is reproducible with seed.

    Validates:
    - Same seed produces same output
    - Different seeds produce different outputs
    """
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("No GPU available")

    # TODO: Instantiate model twice with same seed
    # dx1 = MyGenerative(..., seed=42).to(device)
    # dx2 = MyGenerative(..., seed=42).to(device)

    # x = torch.randn(1, 4, 36, 40).to(device)
    # coords = OrderedDict({...})

    # out1, _ = dx1(x, coords)
    # out2, _ = dx2(x, coords)

    # torch.testing.assert_close(out1, out2)

    # dx3 = MyGenerative(..., seed=123).to(device)
    # out3, _ = dx3(x, coords)
    # assert not torch.allclose(out1, out3)
    pass  # TODO: Implement


# =============================================================================
# FIXTURE PATTERN: Creating Test Package
# =============================================================================


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create a dummy model package for testing.

    This fixture creates a temporary directory with:
    - model.pt: Serialized PhooDiagnostic
    - center.npy: Zero center normalization
    - scale.npy: Unit scale normalization
    """
    tmp_path = tmp_path_factory.mktemp("model_data")

    # Save dummy model
    model = PhooDiagnostic()
    torch.save(model, tmp_path / "model.pt")

    # Save normalization parameters
    np.save(tmp_path / "center.npy", np.zeros(4))
    np.save(tmp_path / "scale.npy", np.ones(4))

    return Package(str(tmp_path))


# =============================================================================
# OPTIONAL: Additional Test Patterns
# =============================================================================


class TestDiagnosticAdditional:
    """Additional test patterns (optional)."""

    def test_diagnostic_dtype(self, test_package):
        """Test model preserves dtype."""
        dx = MyDiagnostic.load_model(test_package)

        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(1, 4, 720, 1440, dtype=dtype)
            coords = OrderedDict(
                {
                    "batch": np.ones(1),
                    "variable": dx.input_coords()["variable"],
                    "lat": dx.input_coords()["lat"],
                    "lon": dx.input_coords()["lon"],
                }
            )
            out, _ = dx(x, coords)
            assert out.dtype == dtype

    def test_diagnostic_batch_independence(self, test_package):
        """Test batch elements are processed independently."""
        dx = MyDiagnostic.load_model(test_package)

        # Single batch
        x1 = torch.randn(1, 4, 720, 1440)
        coords1 = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )
        out1, _ = dx(x1, coords1)

        # Double batch (same input repeated)
        x2 = x1.repeat(2, 1, 1, 1)
        coords2 = OrderedDict(
            {
                "batch": np.ones(2),
                "variable": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )
        out2, _ = dx(x2, coords2)

        # Both batch elements should be identical to single batch result
        torch.testing.assert_close(out2[0], out1[0])
        torch.testing.assert_close(out2[1], out1[0])

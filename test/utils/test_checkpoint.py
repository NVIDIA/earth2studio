# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import os
import tempfile
from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.utils.checkpoint import (
    load_checkpoint,
    save_checkpoint,
    should_checkpoint,
    validate_checkpoint_compatibility,
)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_checkpoint_save_load_cycle(device):
    """Test complete save/load cycle preserves data"""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)

    # Create test data
    x = torch.randn(2, 3, 4, 5).to(device)
    coords = OrderedDict(
        {
            "batch": np.array([0, 1]),
            "variable": np.array(["u", "v", "t"]),
            "lat": np.linspace(-90, 90, 4),
            "lon": np.linspace(0, 360, 5),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        # Save checkpoint
        save_checkpoint(10, x, coords, checkpoint_path, "deterministic")

        # Load checkpoint
        loaded = load_checkpoint(checkpoint_path, device)

        # Verify data integrity
        assert loaded["step"] == 10
        assert loaded["workflow_type"] == "deterministic"
        assert torch.allclose(loaded["state"], x)

        # Verify coordinates
        loaded_coords = loaded["coords"]
        assert loaded_coords.keys() == coords.keys()
        for key in coords.keys():
            np.testing.assert_array_equal(loaded_coords[key], coords[key])

    finally:
        os.unlink(checkpoint_path)


@pytest.mark.parametrize(
    "step,interval,path,expected",
    [
        (5, None, None, False),
        (5, 10, None, False),
        (5, None, "/path", False),
        (0, 5, "/path", True),
        (5, 5, "/path", True),
        (10, 5, "/path", True),
        (3, 5, "/path", False),
        (7, 5, "/path", False),
    ],
)
def test_should_checkpoint(step, interval, path, expected):
    """Test checkpoint decision logic"""
    assert should_checkpoint(step, interval, path) == expected


def test_validate_checkpoint_compatibility():
    """Test checkpoint compatibility validation"""

    # Create mock prognostic model
    class MockPrognostic:
        def input_coords(self):
            return OrderedDict(
                {
                    "batch": np.array([]),
                    "variable": np.array(["u", "v", "t"]),
                    "lat": np.linspace(-90, 90, 4),
                    "lon": np.linspace(0, 360, 5),
                }
            )

    prognostic = MockPrognostic()

    # Compatible coordinates (batch can be different size)
    compatible_coords = OrderedDict(
        {
            "batch": np.array([0, 1]),
            "variable": np.array(["u", "v", "t"]),
            "lat": np.linspace(-90, 90, 4),
            "lon": np.linspace(0, 360, 5),
        }
    )

    assert validate_checkpoint_compatibility(compatible_coords, prognostic)

    # Incompatible coordinates (wrong variables)
    incompatible_coords = OrderedDict(
        {
            "batch": np.array([0, 1]),
            "variable": np.array(["u", "v"]),  # Missing 't'
            "lat": np.linspace(-90, 90, 4),
            "lon": np.linspace(0, 360, 5),
        }
    )

    assert not validate_checkpoint_compatibility(incompatible_coords, prognostic)


@pytest.mark.parametrize("workflow_type", ["deterministic", "diagnostic", "ensemble"])
def test_checkpoint_workflow_type(workflow_type):
    """Test checkpoint saves workflow type correctly"""
    device = torch.device("cpu")
    x = torch.randn(2, 3, 4, 5)
    coords = OrderedDict(
        {
            "batch": np.array([0, 1]),
            "variable": np.array(["u", "v", "t"]),
            "lat": np.linspace(-90, 90, 4),
            "lon": np.linspace(0, 360, 5),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(5, x, coords, checkpoint_path, workflow_type)
        loaded = load_checkpoint(checkpoint_path, device)
        assert loaded["workflow_type"] == workflow_type
    finally:
        os.unlink(checkpoint_path)


def test_checkpoint_contains_rng_state():
    """Test checkpoint includes RNG states for reproducibility"""
    device = torch.device("cpu")
    x = torch.randn(2, 3)
    coords = OrderedDict(
        {"batch": np.array([0, 1]), "variable": np.array(["u", "v", "t"])}
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(0, x, coords, checkpoint_path, "deterministic")
        loaded = load_checkpoint(checkpoint_path, device)

        # Should contain RNG states
        assert "torch_rng_state" in loaded
        assert isinstance(loaded["torch_rng_state"], torch.ByteTensor)

        # CUDA RNG state only if CUDA available
        if torch.cuda.is_available():
            assert "cuda_rng_state" in loaded

    finally:
        os.unlink(checkpoint_path)


def test_checkpoint_coordinate_types():
    """Test checkpoint handles different coordinate types"""
    device = torch.device("cpu")
    x = torch.randn(2, 3, 4)
    coords = OrderedDict(
        {
            "batch": np.array([0, 1]),
            "variable": np.array(["u", "v", "t"]),
            "time": np.array(
                [
                    np.datetime64("2024-01-01"),
                    np.datetime64("2024-01-02"),
                    np.datetime64("2024-01-03"),
                    np.datetime64("2024-01-04"),
                ]
            ),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(0, x, coords, checkpoint_path, "deterministic")
        loaded = load_checkpoint(checkpoint_path, device)

        # Verify datetime handling
        loaded_coords = loaded["coords"]
        np.testing.assert_array_equal(loaded_coords["time"], coords["time"])

    finally:
        os.unlink(checkpoint_path)


def test_load_checkpoint_file_not_found():
    """Test load_checkpoint raises FileNotFoundError for missing files"""
    device = torch.device("cpu")

    with tempfile.NamedTemporaryFile(delete=False) as f:
        nonexistent_path = f.name + "_nonexistent"

    with pytest.raises(FileNotFoundError):
        load_checkpoint(nonexistent_path, device)


def test_validate_checkpoint_compatibility_missing_dims():
    """Test validation fails for missing dimensions"""

    class MockPrognostic:
        def input_coords(self):
            return OrderedDict(
                {
                    "variable": np.array(["u", "v", "t"]),
                    "lat": np.linspace(-90, 90, 4),
                    "lon": np.linspace(0, 360, 5),
                }
            )

    prognostic = MockPrognostic()

    # Missing required dimension
    incomplete_coords = OrderedDict(
        {
            "variable": np.array(["u", "v", "t"]),
            "lat": np.linspace(-90, 90, 4),
            # Missing 'lon'
        }
    )

    assert not validate_checkpoint_compatibility(incomplete_coords, prognostic)


def test_validate_checkpoint_compatibility_shape_mismatch():
    """Test validation fails for shape mismatches"""

    class MockPrognostic:
        def input_coords(self):
            return OrderedDict(
                {
                    "variable": np.array(["u", "v", "t"]),
                    "lat": np.linspace(-90, 90, 4),
                    "lon": np.linspace(0, 360, 5),
                }
            )

    prognostic = MockPrognostic()

    # Wrong shape for lat dimension
    mismatched_coords = OrderedDict(
        {
            "variable": np.array(["u", "v", "t"]),
            "lat": np.linspace(-90, 90, 6),  # Different size
            "lon": np.linspace(0, 360, 5),
        }
    )

    assert not validate_checkpoint_compatibility(mismatched_coords, prognostic)


@pytest.mark.parametrize("interval", [1, 2, 5, 10])
def test_checkpoint_interval_patterns(interval):
    """Test checkpoint saving at different intervals"""
    path = "/dummy/path"

    # Test steps 0-20
    expected_saves = [i for i in range(0, 21, interval)]

    actual_saves = [
        step for step in range(21) if should_checkpoint(step, interval, path)
    ]

    assert actual_saves == expected_saves


def test_checkpoint_ensemble_coordinates():
    """Test checkpoint handling of ensemble coordinates"""
    device = torch.device("cpu")
    x = torch.randn(4, 3, 8, 16)  # 4 ensemble members
    coords = OrderedDict(
        {
            "ensemble": np.array([0, 1, 2, 3]),
            "variable": np.array(["u", "v", "t"]),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16),
        }
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(2, x, coords, checkpoint_path, "ensemble")
        loaded = load_checkpoint(checkpoint_path, device)

        # Verify ensemble dimension preserved
        assert "ensemble" in loaded["coords"]
        assert len(loaded["coords"]["ensemble"]) == 4
        np.testing.assert_array_equal(loaded["coords"]["ensemble"], coords["ensemble"])

    finally:
        os.unlink(checkpoint_path)


def test_checkpoint_metadata_structure():
    """Test checkpoint contains expected metadata fields"""
    device = torch.device("cpu")
    x = torch.randn(2, 3)
    coords = OrderedDict(
        {"batch": np.array([0, 1]), "variable": np.array(["u", "v", "t"])}
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = f.name

    try:
        save_checkpoint(7, x, coords, checkpoint_path, "diagnostic")
        loaded = load_checkpoint(checkpoint_path, device)

        # Check all required fields
        required_fields = [
            "step",
            "state",
            "coords",
            "workflow_type",
            "torch_rng_state",
        ]
        for field in required_fields:
            assert field in loaded, f"Missing required field: {field}"

        # Check field types
        assert isinstance(loaded["step"], int)
        assert isinstance(loaded["state"], torch.Tensor)
        assert isinstance(loaded["coords"], OrderedDict)
        assert isinstance(loaded["workflow_type"], str)

    finally:
        os.unlink(checkpoint_path)

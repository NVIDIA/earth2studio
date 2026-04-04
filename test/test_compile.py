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

from unittest.mock import MagicMock, patch

import pytest
import torch

from earth2studio.run import _maybe_compile_model


class DummyNetwork(torch.nn.Module):
    """A minimal nn.Module for testing torch.compile."""

    def __init__(self, channels: int = 4):
        super().__init__()
        self.linear = torch.nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MockPrognosticWithModel:
    """Mock prognostic model that exposes a .model attribute."""

    def __init__(self):
        self.model = DummyNetwork(channels=4)


class MockPrognosticWithoutModel:
    """Mock prognostic model without a .model attribute."""

    pass


class MockPrognosticNonModuleModel:
    """Mock prognostic model where .model is not an nn.Module."""

    def __init__(self):
        self.model = "not_a_module"


def test_compile_disabled():
    """When compile=False, model should be returned unchanged."""
    prog = MockPrognosticWithModel()
    original_model = prog.model
    result = _maybe_compile_model(prog, compile=False)
    assert result is prog
    assert result.model is original_model


def test_compile_enabled_with_model():
    """When compile=True and model has .model attribute, it should be compiled."""
    prog = MockPrognosticWithModel()
    original_model = prog.model
    result = _maybe_compile_model(prog, compile=True)
    assert result is prog
    # After compilation, model should be wrapped (not the same object)
    assert result.model is not original_model


def test_compile_without_model_attribute():
    """When model lacks .model attribute, should warn and return unchanged."""
    prog = MockPrognosticWithoutModel()
    result = _maybe_compile_model(prog, compile=True)
    assert result is prog


def test_compile_non_module_model():
    """When .model is not an nn.Module, should warn and return unchanged."""
    prog = MockPrognosticNonModuleModel()
    result = _maybe_compile_model(prog, compile=True)
    assert result is prog
    assert result.model == "not_a_module"


def test_compile_fallback_on_error():
    """If torch.compile raises, should fall back gracefully."""
    prog = MockPrognosticWithModel()

    with patch("torch.compile", side_effect=RuntimeError("compilation failed")):
        result = _maybe_compile_model(prog, compile=True)
        # Should return the model unchanged on error
        assert result is prog


def test_compiled_model_produces_output():
    """Verify that a compiled model still produces correct output."""
    prog = MockPrognosticWithModel()

    # Get reference output before compilation
    x = torch.randn(2, 4)
    with torch.no_grad():
        ref_output = prog.model(x)

    # Compile
    _maybe_compile_model(prog, compile=True)

    # Get output after compilation (first call triggers compilation)
    with torch.no_grad():
        compiled_output = prog.model(x)

    assert torch.allclose(ref_output, compiled_output, atol=1e-5)

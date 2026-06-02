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

"""Tests for HistoryModel prognostic model."""

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import HistoryModel
from earth2studio.utils import handshake_dim


class PhooHistoryModel(torch.nn.Module):
    """Dummy history model for testing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2, keepdim=True)


class TestHistoryModelMock:
    """Mock tests using dummy model weights."""

    @pytest.mark.parametrize(
        "time",
        [
            np.array([np.datetime64("2024-01-01T00:00")]),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="No GPU"
                ),
            ),
        ],
    )
    def test_history_call(self, time, device):
        """Test single forward pass with history input."""
        model = HistoryModel.load_model(HistoryModel.load_default_package())
        model = model.to(device)

        dc = model.input_coords()
        dc["time"] = time

        # Manually create input with 2 lead_time steps
        shape = (
            1,  # batch
            len(time),
            len(dc["lead_time"]),  # 2 lead_time steps
            len(dc["variable"]),
            len(dc["lat"]),
            len(dc["lon"]),
        )
        x = torch.randn(shape, device=device)
        coords = dc.copy()
        coords["batch"] = np.array([0])

        out, out_coords = model(x, coords)

        # Output should have 1 lead_time step
        assert out.shape[2] == 1
        assert isinstance(out_coords, OrderedDict)
        handshake_dim(out_coords, "variable", 3)

        # Output lead_time should be 0h + 6h = 6h
        expected_lead = np.array([np.timedelta64(6, "h")])
        np.testing.assert_array_equal(out_coords["lead_time"], expected_lead)

    def test_history_input_coords(self):
        """Test input_coords has two lead_time steps."""
        model = HistoryModel.load_model(HistoryModel.load_default_package())
        coords = model.input_coords()

        assert len(coords["lead_time"]) == 2
        np.testing.assert_array_equal(
            coords["lead_time"],
            np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
        )

    @pytest.mark.parametrize("ensemble", [1, 2])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="No GPU"
                ),
            ),
        ],
    )
    def test_history_iter(self, ensemble, device):
        """Test iterator produces correct sequence."""
        model = HistoryModel.load_model(HistoryModel.load_default_package())
        model = model.to(device)

        dc = model.input_coords()
        time = np.array([np.datetime64("2024-01-01T00:00")])
        dc["time"] = time

        shape = (
            ensemble,
            len(time),
            len(dc["lead_time"]),
            len(dc["variable"]),
            len(dc["lat"]),
            len(dc["lon"]),
        )
        x = torch.randn(shape, device=device)
        coords = dc.copy()
        coords["batch"] = np.arange(ensemble)
        coords["ensemble"] = np.arange(ensemble)
        coords.move_to_end("ensemble", last=False)

        iterator = model.create_iterator(x, coords)
        assert isinstance(iterator, Iterable)

        # Initial condition (step 0) should be at lead_time=0h
        step0_x, step0_coords = next(iterator)
        assert step0_x.shape[0] == ensemble
        np.testing.assert_array_equal(
            step0_coords["lead_time"], np.array([np.timedelta64(0, "h")])
        )

        # Subsequent steps
        for i, (step_x, step_coords) in enumerate(iterator):
            assert step_x.shape[0] == ensemble
            assert step_x.shape[2] == 1  # Single lead_time output
            expected_lead = np.array([np.timedelta64((i + 1) * 6, "h")])
            np.testing.assert_array_equal(step_coords["lead_time"], expected_lead)
            if i >= 4:
                break

    @pytest.mark.parametrize(
        "coords",
        [
            OrderedDict(
                {
                    "batch": np.empty(0),
                    "time": np.empty(0),
                    "lead_time": np.array([np.timedelta64(0, "h")]),  # Wrong: needs 2
                    "variable": np.array(["t2m"]),
                    "lat": np.linspace(90, -90, 10),
                    "lon": np.linspace(0, 360, 20),
                }
            ),
        ],
    )
    def test_history_exceptions(self, coords):
        """Test model raises on invalid coordinates."""
        model = HistoryModel.load_model(HistoryModel.load_default_package())
        x = torch.randn(
            1,
            1,
            len(coords["lead_time"]),
            len(coords["variable"]),
            len(coords["lat"]),
            len(coords["lon"]),
        )
        with pytest.raises((KeyError, ValueError)):
            model(x, coords)

    def test_history_averages_inputs(self):
        """Test model averages the two history steps."""
        model = HistoryModel.load_model(HistoryModel.load_default_package())
        dc = model.input_coords()
        dc["time"] = np.array([np.datetime64("2024-01-01T00:00")])

        # Create input where averaging is testable
        shape = (1, 1, 2, len(dc["variable"]), len(dc["lat"]), len(dc["lon"]))
        x = torch.zeros(shape)
        x[:, :, 0, :, :, :] = 1.0  # -6h state
        x[:, :, 1, :, :, :] = 3.0  # 0h state
        coords = dc.copy()
        coords["batch"] = np.array([0])

        out, _ = model(x, coords)

        # Output should be average: (1 + 3) / 2 = 2
        torch.testing.assert_close(out, torch.ones_like(out) * 2.0)

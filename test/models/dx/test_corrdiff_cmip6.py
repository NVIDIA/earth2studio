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

from collections import OrderedDict
from datetime import datetime
from typing import ClassVar

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.dx import CorrDiffCMIP6
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
            batch_size,
            self.img_out_channels,
            x.shape[-2],
            x.shape[-1],
            device=self.device,
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
def cmip6_model_minimal(mock_residual_model, mock_regression_model):
    """Minimal CorrDiffCMIP6 instance suitable for unit tests."""
    n = 128
    input_variables = ["siconc", "snc", "t2m"]
    output_variables = ["u10m", "v10m", "t2m", "sp"]

    lat_in = torch.linspace(-1.0, 1.0, n)
    lon_in = torch.linspace(0.0, 7.0, n)
    lat_out = torch.linspace(-1.0, 1.0, n)
    lon_out = torch.linspace(0.0, 7.0, n)

    # One invariant is required by the CMIP6 channel-reordering logic (coslat slot).
    invariants = OrderedDict({"coslat": torch.zeros(n, n)})

    return CorrDiffCMIP6(
        input_variables=input_variables,
        output_variables=output_variables,
        residual_model=mock_residual_model,
        regression_model=mock_regression_model,
        lat_input_grid=lat_in,
        lon_input_grid=lon_in,
        lat_output_grid=lat_out,
        lon_output_grid=lon_out,
        in_center=torch.zeros(len(input_variables)),
        in_scale=torch.ones(len(input_variables)),
        out_center=torch.zeros(len(output_variables)),
        out_scale=torch.ones(len(output_variables)),
        invariants=invariants,
        invariant_center=torch.zeros(1),
        invariant_scale=torch.ones(1),
        time_feature_center=torch.zeros(2),
        time_feature_scale=torch.ones(2),
        number_of_samples=1,
        number_of_steps=1,
        solver="euler",
        sampler_type="stochastic",
        inference_mode="regression",
        hr_mean_conditioning=False,
    )


class TestCorrDiffCMIP6Utils:
    """Focused unit tests for CorrDiffCMIP6 model utils."""

    def test_preprocess_requires_valid_time(self, cmip6_model_minimal):
        x = torch.zeros((len(cmip6_model_minimal.input_variables), 128, 128))
        with pytest.raises(ValueError):
            cmip6_model_minimal.preprocess_input(x, valid_time=None)

    def test_preprocess_accepts_chw_and_returns_bchw(
        self, monkeypatch, cmip6_model_minimal
    ):
        def _fake_cos_sza(dt, lon, lat):
            return np.zeros_like(lat, dtype=np.float32)

        monkeypatch.setattr(
            "earth2studio.models.dx.corrdiff_cmip6.cos_zenith_angle",
            _fake_cos_sza,
            raising=True,
        )

        coords = cmip6_model_minimal.input_coords()
        x = torch.zeros(
            (1, coords["lead_time"].shape[0], coords["variable"].shape[0], 8, 8)
        )
        out = cmip6_model_minimal.preprocess_input(
            x, valid_time=datetime(2020, 1, 1, 12)
        )
        assert out.ndim == 4
        assert out.shape[0] == 1

    def _patch_minimal_dependencies(self, monkeypatch):
        """Patch optional/slow dependencies used in CorrDiffCMIP6 internals."""

        def _fake_cos_sza(dt, lon, lat):
            return np.zeros_like(lat, dtype=np.float32)

        monkeypatch.setattr(
            "earth2studio.models.dx.corrdiff_cmip6.cos_zenith_angle",
            _fake_cos_sza,
            raising=True,
        )

        def _fake_regression_step(*, net, img_lr, latents_shape):
            # Match expected shape (including preprocess padding), on the correct device.
            return torch.zeros(latents_shape, device=img_lr.device, dtype=torch.float32)

        def _fake_diffusion_step(
            *,
            net,
            sampler_fn,
            img_shape,
            img_out_channels,
            rank_batches,
            img_lr,
            rank,
            device,
            mean_hr=None,
        ):
            return torch.zeros(
                (1, img_out_channels, img_shape[0], img_shape[1]),
                device=device,
                dtype=torch.float32,
            )

        monkeypatch.setattr(
            "earth2studio.models.dx.corrdiff.regression_step",
            _fake_regression_step,
            raising=True,
        )
        monkeypatch.setattr(
            "earth2studio.models.dx.corrdiff.diffusion_step",
            _fake_diffusion_step,
            raising=True,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_stream_samples_to_cpu(self, monkeypatch, cmip6_model_minimal):
        self._patch_minimal_dependencies(monkeypatch)

        model = cmip6_model_minimal.to("cuda")
        model.inference_mode = "both"
        model.number_of_samples = 2

        coords = model.input_coords()
        x = torch.zeros(
            (1, coords["lead_time"].shape[0], coords["variable"].shape[0], 128, 128),
            device="cuda",
        )
        valid_time = datetime(2020, 1, 1, 12)

        model.stream_samples_to_cpu = True
        y_cpu = model._forward(x, valid_time=valid_time)
        assert y_cpu.device.type == "cpu"
        assert y_cpu.shape[0] == 1
        assert y_cpu.shape[1] == 2
        assert y_cpu.shape[2] == len(model.output_variables)

        model.stream_samples_to_cpu = False
        y_gpu = model._forward(x, valid_time=valid_time)
        assert y_gpu.device.type == "cuda"
        assert y_gpu.shape == y_cpu.shape

    def test_postprocess_output(self, cmip6_model_minimal):
        model = cmip6_model_minimal

        # Build a padded tensor with a known per-row pattern in the crop region.
        # Shape here matches the padded shape that preprocess introduces:
        # H_pad = 128 + 23 + 24 = 175, W_pad = 128 + 48 + 48 = 224
        h_pad, w_pad = 175, 224
        x = torch.zeros(
            (1, len(model.output_variables), h_pad, w_pad), dtype=torch.float32
        )
        # Fill crop region with negative values and a row-wise ramp to observe flipping.
        crop = x[:, :, 23:-24, 48:-48]  # [1,1,128,128]
        for r in range(crop.shape[2]):
            crop[:, :, r, :] = float(-(r + 1))  # negative so clamp will hit

        y = model.postprocess_output(x)
        assert y.shape == (1, len(model.output_variables), 128, 128)
        # Clamp: t2m is in _NONNEGATIVE_VARS, so negatives should become 0
        assert torch.all(y[:, model.output_variables.index("t2m")] >= 0)


@pytest.mark.parametrize("inference_mode", ["regression", "both"])
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [np.datetime64("1993-04-05T00:00"), np.datetime64("1993-04-06T00:00")]
        ),
    ],
)
@pytest.mark.parametrize(
    "output_lead_times",
    [
        np.array([np.timedelta64(-6, "h")]),
        np.array([np.timedelta64(-12, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize("number_of_samples", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_corrdiff_cmip6_forward(
    inference_mode,
    time,
    output_lead_times,
    number_of_samples,
    device,
    cmip6_model_minimal,
):

    dx = cmip6_model_minimal.to(device)
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package AIFS
    dx.output_lead_times = output_lead_times
    dx.number_of_samples = number_of_samples
    dx.number_of_steps = 1

    # Create "domain coords"
    dc = {k: dx.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = dx.input_coords()["lead_time"]
    variable = dx.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = dx(x, coords)

    assert out.shape[0] == number_of_samples
    assert out.shape[1] == time.shape[0]
    assert out.shape[2] == output_lead_times.shape[0]
    assert out.shape[3] == len(dx.output_variables)

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "sample", 0)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_corrdiff_cmip6_package(device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package
    package = CorrDiffCMIP6.load_default_package()
    dx = CorrDiffCMIP6.load_model(package, device=device)
    dx.number_of_samples = 2
    dx.number_of_steps = 1

    # Create "domain coords"
    dc = {k: dx.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = dx.input_coords()["lead_time"]
    variable = dx.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)
    out, out_coords = dx(x, coords)

    assert out.shape[0] == 2
    assert out.shape[1] == 1
    assert out.shape[2] == 1
    assert out.shape[3] == len(dx.output_variables)

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "sample", 0)

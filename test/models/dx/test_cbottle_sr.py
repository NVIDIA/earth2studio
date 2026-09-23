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

from collections import OrderedDict

import numpy as np
import pytest
import torch

try:
    import cbottle
except ImportError:
    cbottle = None

from types import SimpleNamespace

from test_cbottle_infill import _field

from earth2studio.models.conformance import (
    ContractException,
    check_diagnostic_contract,
)
from earth2studio.models.dx import CBottleSR
from earth2studio.models.dx.cbottle_sr import CHANNEL_TO_VARIABLE
from earth2studio.utils import handshake_dim


@pytest.fixture(autouse=True)
def offline_sr(monkeypatch):
    if cbottle is not None:
        return

    def initialize(
        self,
        core,
        lat_lon=True,
        output_resolution=(2161, 4320),
        super_resolution_window=None,
        seed=None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        self.seed = seed
        self.sampler_steps = kwargs.get("sampler_steps", 18)
        self._sample_index = 0
        self.input_type = self.output_type = "latlon" if lat_lon else "healpix"
        self.register_buffer("_device_buffer", torch.empty(0))
        self.input_grid = SimpleNamespace(
            lat=np.linspace(90, -90, 721, endpoint=False), lon=np.arange(1440) / 4
        )
        if super_resolution_window is None:
            lat = np.linspace(90, -90, output_resolution[0], endpoint=False)
            lon = np.linspace(0, 360, output_resolution[1], endpoint=False)
        else:
            s, w, n, e = super_resolution_window
            lat, lon = np.linspace(s, n, output_resolution[0]), np.linspace(
                w, e, output_resolution[1]
            )
        self.output_grid = SimpleNamespace(lat=lat, lon=lon)

    def forward(self, x):
        # Mirror the EDM schedule denominator; one step produces NaN, not a
        # valid diffusion trajectory, even in the lightweight offline fixture.
        steps = torch.arange(self.sampler_steps, device=x.device, dtype=torch.float64)
        schedule = (
            800 ** (1 / 7)
            + steps / (self.sampler_steps - 1) * (0.02 ** (1 / 7) - 800 ** (1 / 7))
        ) ** 7
        shape = (
            (12, len(self.output_grid.lat), len(self.output_grid.lon))
            if self.output_type == "latlon"
            else (12, 12582912)
        )
        gen = (
            torch.Generator(device=x.device).manual_seed(self.seed + self._sample_index)
            if self.seed is not None
            else None
        )
        self._sample_index += 1
        return (
            (torch.rand((), device=x.device, generator=gen) * schedule[0] / 800)
            .float()
            .expand(shape)
        )

    monkeypatch.setattr(CBottleSR, "__init__", initialize)
    monkeypatch.setattr(CBottleSR, "_forward", forward)


@pytest.fixture(scope="class")
def mock_cbottle_core_model() -> torch.nn.Module:
    if cbottle is None:
        return torch.nn.Identity()
    """Create a mock core model similar to the actual cbottle model"""
    # Create a more realistic mock using cbottle config like in test_cbottle.py
    # Actual parameters,
    # "architecture": "unet_hpx1024_patch"
    # "model_channels": 128,
    # "label_dim": 0,
    # "out_channels": 12,
    # "condition_channels": 24,
    # "time_length": 1,
    # "label_dropout": 0.0,
    # "position_embed_channels": 20,
    # "img_resolution": 128

    model_config = cbottle.config.models.ModelConfigV1()
    model_config.architecture = "unet_hpx1024_patch"
    model_config.model_channels = 8  # Reduced for testing
    model_config.label_dim = 0
    model_config.out_channels = 12  # Number of variables
    model_config.condition_channels = 24
    model_config.time_length = 1
    model_config.label_dropout = 0.0
    model_config.position_embed_channels = 20
    model_config.img_resolution = 128
    model_config.level = 10  # SR has positional embedding, so HPX level needs to be 10

    model = cbottle.models.get_model(model_config)

    batch_info = cbottle.datasets.base.BatchInfo(
        channels=list(CHANNEL_TO_VARIABLE.keys())
    )
    return cbottle.inference.SuperResolutionModel(model, batch_info)


class TestCBottleSRMock:

    @pytest.mark.parametrize(
        "x",
        [
            torch.randn(1, 12, 721, 1440),
            torch.randn(2, 12, 721, 1440),
        ],
    )
    @pytest.mark.parametrize("output_resolution", [(721, 1440)])
    @pytest.mark.parametrize(
        "device,window",
        [
            ("cuda:0", (0, -120, 50, -40)),
            ("cuda:0", None),
            ("cuda:0", (0, -120, 50, -40)),
        ],  # Skipping CPU tests, should work be too slow
    )
    def test_cbottle_sr_latlon(
        self, x, device, output_resolution, window, mock_cbottle_core_model
    ):
        # Create CBottleSR model with mock core model
        dx = CBottleSR(
            mock_cbottle_core_model,
            lat_lon=True,
            output_resolution=output_resolution,
            super_resolution_window=window,
            sampler_steps=2,  # Smallest valid EDM schedule
            sigma_max=800,  # Reduced for testing
        ).to(device)

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

        # Forward pass
        field = _field(dx, x, coords)
        out = dx(field)
        assert torch.isfinite(out.e2s.to_torch()[0]).all()
        out_coords = {k: out.coords[k].values for k in out.dims}

        # Check output shape
        expected_shape = torch.Size(
            [
                x.shape[0],
                len(dx.input_coords()["variable"]),
                output_resolution[0],
                output_resolution[1],
            ]
        )
        assert out.shape == expected_shape

        # Check output coordinates
        assert all(out_coords["variable"] == dx.output_coords(field)["variable"])
        handshake_dim(out_coords, "lon", 3)
        handshake_dim(out_coords, "lat", 2)
        handshake_dim(out_coords, "variable", 1)
        handshake_dim(out_coords, "batch", 0)

        # Check coordinate values
        assert len(out_coords["lat"]) == output_resolution[0]
        assert len(out_coords["lon"]) == output_resolution[1]

    @pytest.mark.parametrize(
        "x",
        [
            torch.randn(1, 12, 49152),  # HEALPix level 6
            torch.randn(2, 12, 49152),
        ],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_cbottle_sr_healpix(self, x, device, mock_cbottle_core_model):
        """Test HEALPix input to HEALPix output (native grids)"""
        # Create CBottleSR model with HEALPix input and output
        dx = CBottleSR(
            mock_cbottle_core_model,
            lat_lon=False,
            sampler_steps=2,
            sigma_max=800,
        ).to(device)

        x = x.to(device)

        # Create HEALPix input coordinates
        coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "variable": dx.input_coords()["variable"],
                "hpx": dx.input_coords()["hpx"],
            }
        )

        # Forward pass
        field = _field(dx, x, coords)
        out = dx(field)
        assert torch.isfinite(out.e2s.to_torch()[0]).all()
        out_coords = {k: out.coords[k].values for k in out.dims}

        # Check output shape - HEALPix level 10: 1024^2 * 12 = 12,582,912
        expected_shape = torch.Size(
            [
                x.shape[0],
                len(dx.input_coords()["variable"]),
                12582912,  # HEALPix level 10 pixels
            ]
        )
        assert out.shape == expected_shape

        # Check output coordinates are HEALPix
        assert all(out_coords["variable"] == dx.output_coords(field)["variable"])
        handshake_dim(out_coords, "hpx", 2)
        handshake_dim(out_coords, "variable", 1)
        handshake_dim(out_coords, "batch", 0)

        # Check HEALPix coordinate values
        assert len(out_coords["hpx"]) == 12582912  # Level 10 HEALPix

    @pytest.mark.parametrize(
        "x",
        [
            torch.randn(1, 12, 721, 1440),
            torch.randn(2, 12, 721, 1440),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_sr_exceptions(self, x, device, mock_cbottle_core_model):
        """Test CBottleSR exception handling"""

        dx = CBottleSR(mock_cbottle_core_model).to(device)
        x = x.to(device)

        # Wrong coordinate keys
        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "wrong": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )

        with pytest.raises((KeyError, ValueError)):
            dx(_field(dx, x, wrong_coords))

        # Wrong coordinate order
        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "variable": dx.input_coords()["variable"],
                "lon": dx.input_coords()["lon"],
                "lat": dx.input_coords()["lat"],  # Wrong order
            }
        )

        with pytest.raises(ValueError):
            dx(_field(dx, x.transpose(-1, -2), wrong_coords))

        # Wrong coordinate values
        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "variable": dx.input_coords()["variable"],
                "lat": np.linspace(-90, 90, 720),  # Wrong size
                "lon": dx.input_coords()["lon"],
            }
        )

        with pytest.raises(ValueError):
            dx(_field(dx, x, wrong_coords))

    def test_cbottle_sr_conformance(self, mock_cbottle_core_model, monkeypatch):
        dx = CBottleSR(
            mock_cbottle_core_model,
            lat_lon=True,
            output_resolution=(721, 1440),
            sampler_steps=2,  # Smallest valid EDM schedule
            sigma_max=800,  # Reduced for testing
        )
        forward = dx._forward

        def finite_forward(x):
            out = forward(x)
            assert torch.isfinite(out).all()
            return out

        monkeypatch.setattr(dx, "_forward", finite_forward)
        with pytest.raises(ContractException) as exc_info:
            check_diagnostic_contract(dx)
        assert exc_info.value.violations == [
            "D9: repeated runs with the same input and seed disagree"
        ]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cbottle_sr_package(device):
    """Test the cached model package CBottleSR"""
    # Only cuda supported for full model
    package = CBottleSR.load_default_package()
    dx = CBottleSR.load_model(
        package,
        lat_lon=True,
        sampler_steps=2,  # Smallest valid EDM schedule
        output_resolution=(721, 1440),  # Reduced for testing
        seed=42,  # Set seed for reproducibility
    ).to(device)

    x = torch.randn(1, 12, 721, 1440).to(device)
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    field = _field(dx, x, coords)
    out = dx(field)
    assert torch.isfinite(out.e2s.to_torch()[0]).all()
    out_coords = {k: out.coords[k].values for k in out.dims}
    assert out.shape == torch.Size([x.shape[0], 12, 721, 1440])

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(field)["variable"])
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)

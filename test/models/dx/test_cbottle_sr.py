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

import numpy as np
import pytest
import torch

try:
    import cbottle
except ImportError:
    pytest.skip("cbottle dependencies not installed", allow_module_level=True)

from earth2studio.models.dx import CBottleSR
from earth2studio.utils import handshake_dim


@pytest.fixture(scope="class")
def mock_cbottle_core_model() -> torch.nn.Module:
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
    model.sigma_min = 0.002
    model.sigma_max = 80.0

    return model


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
    def test_cbottle_sr_latlon_to_latlon(
        self, x, device, output_resolution, window, mock_cbottle_core_model
    ):
        # Create CBottleSR model with mock core model
        dx = CBottleSR(
            mock_cbottle_core_model,
            output_resolution=output_resolution,
            super_resolution_window=window,
            sampler_steps=1,  # Reduced for testing speed
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
        out, out_coords = dx(x, coords)

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
        assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
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
    def test_cbottle_sr_healpix_to_healpix(
        self, x, device, mock_cbottle_core_model
    ):
        """Test HEALPix input to HEALPix output (native grids)"""
        # Create CBottleSR model with HEALPix input and output
        dx = CBottleSR(
            mock_cbottle_core_model,
            input_type="healpix",
            output_type="healpix",
            sampler_steps=1,
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
        out, out_coords = dx(x, coords)

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
        assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
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
            dx(x, wrong_coords)

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
            dx(x, wrong_coords)

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
            dx(x, wrong_coords)

    def test_cbottle_sr_healpix_exceptions(self, mock_cbottle_core_model):
        """Test CBottleSR HEALPix-specific exception handling"""
        
        # Test invalid input/output type combinations
        with pytest.raises(ValueError, match="input_type must be one of"):
            CBottleSR(mock_cbottle_core_model, input_type="invalid")
            
        with pytest.raises(ValueError, match="output_type must be one of"):
            CBottleSR(mock_cbottle_core_model, output_type="invalid")
            
        # Test super_resolution_window with HEALPix output
        with pytest.raises(ValueError, match="super_resolution_window is not supported with healpix output_type"):
            CBottleSR(
                mock_cbottle_core_model,
                output_type="healpix",
                super_resolution_window=(0, -120, 50, -40)
            )

    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_cbottle_sr_healpix_coordinate_validation(self, device, mock_cbottle_core_model):
        """Test coordinate validation for HEALPix inputs/outputs"""
        
        # Test HEALPix input with wrong coordinates
        dx_healpix_in = CBottleSR(
            mock_cbottle_core_model,
            input_type="healpix",
            output_type="latlon",
            sampler_steps=1,
        ).to(device)
        
        x_healpix = torch.randn(1, 12, 49152).to(device)
        
        # Wrong coordinate key (lat/lon instead of hpx)
        wrong_coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": dx_healpix_in.input_coords()["variable"],
                "lat": np.linspace(90, -90, 721),  # Wrong for HEALPix input
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        
        with pytest.raises((KeyError, ValueError)):
            dx_healpix_in(x_healpix, wrong_coords)
            
        # Wrong HEALPix coordinate size
        wrong_coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": dx_healpix_in.input_coords()["variable"],
                "hpx": np.arange(1000),  # Wrong size for HEALPix level 6
            }
        )
        
        with pytest.raises(ValueError):
            dx_healpix_in(x_healpix, wrong_coords)

    def test_cbottle_sr_input_output_combinations(self, mock_cbottle_core_model):
        """Test all valid input/output type combinations"""
        
        # All valid combinations should instantiate without error
        combinations = [
            ("latlon", "latlon"),
            ("latlon", "healpix"),
            ("healpix", "latlon"), 
            ("healpix", "healpix"),
        ]
        
        for input_type, output_type in combinations:
            # Should not raise any exceptions
            dx = CBottleSR(
                mock_cbottle_core_model,
                input_type=input_type,
                output_type=output_type,
                sampler_steps=1,
            )
            
            # Check that the input/output types are stored correctly
            assert dx.input_type == input_type
            assert dx.output_type == output_type
            
            # Check coordinate systems are consistent
            input_coords = dx.input_coords()
            if input_type == "latlon":
                assert "lat" in input_coords and "lon" in input_coords
                assert "hpx" not in input_coords
            else:  # healpix
                assert "hpx" in input_coords
                assert "lat" not in input_coords and "lon" not in input_coords


@pytest.mark.ci_cache
@pytest.mark.timeout(60)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cbottle_sr_package(device, model_cache_context):
    """Test the cached model package CBottleSR"""
    # Only cuda supported for full model
    with model_cache_context():
        package = CBottleSR.load_default_package()
        dx = CBottleSR.load_model(
            package,
            sampler_steps=1,  # Reduced for testing
            output_resolution=(721, 1440),  # Reduced for testing
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

    out, out_coords = dx(x, coords)
    assert out.shape == torch.Size([x.shape[0], 12, 721, 1440])

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.ci_cache
@pytest.mark.timeout(60) 
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize(
    "input_type,output_type,expected_input_shape,expected_output_shape",
    [
        ("latlon", "healpix", (1, 12, 721, 1440), (1, 12, 12582912)),
        ("healpix", "latlon", (1, 12, 49152), (1, 12, 721, 1440)),
        ("healpix", "healpix", (1, 12, 49152), (1, 12, 12582912)),
    ]
)
def test_cbottle_sr_package_healpix(device, input_type, output_type, expected_input_shape, expected_output_shape, model_cache_context):
    """Test the cached model package CBottleSR with HEALPix configurations"""
    # Only cuda supported for full model
    with model_cache_context():
        package = CBottleSR.load_default_package()
        dx = CBottleSR.load_model(
            package,
            input_type=input_type,
            output_type=output_type,
            sampler_steps=1,  # Reduced for testing
            output_resolution=(721, 1440) if output_type == "latlon" else None,
        ).to(device)

    # Create appropriate input tensor and coordinates
    x = torch.randn(*expected_input_shape).to(device)
    
    if input_type == "latlon":
        coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "variable": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )
    else:  # healpix
        coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "variable": dx.input_coords()["variable"], 
                "hpx": dx.input_coords()["hpx"],
            }
        )

    out, out_coords = dx(x, coords)
    assert out.shape == torch.Size(expected_output_shape)

    # Check variables
    assert all(out_coords["variable"] == dx.output_coords(coords)["variable"])
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)
    
    # Check output coordinate structure based on output type
    if output_type == "latlon":
        handshake_dim(out_coords, "lon", 3)
        handshake_dim(out_coords, "lat", 2)
    else:  # healpix
        handshake_dim(out_coords, "hpx", 2)
        assert len(out_coords["hpx"]) == 12582912  # Level 10 HEALPix

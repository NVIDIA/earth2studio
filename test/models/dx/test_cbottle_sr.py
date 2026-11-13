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

from earth2studio.grids import Region
from earth2studio.models.base import DomainAdaptable
from earth2studio.models.conformance import (
    check_diagnostic_contract,
)
from earth2studio.models.dx import CBottleSR
from earth2studio.models.dx.cbottle_sr import CHANNEL_TO_VARIABLE
from earth2studio.utils import handshake_dim


class _PointRegridder(torch.nn.Module):
    """Offline stand-in for the HEALPix to lat/lon regridder."""

    def __init__(self, grid):
        super().__init__()
        self.shape = (len(np.ravel(grid.lat)), len(np.ravel(grid.lon)))

    def forward(self, x):
        return x.mean(-1)[..., None, None].expand(*x.shape[:-1], *self.shape)


@pytest.fixture(autouse=True)
def offline_sr(monkeypatch):
    if cbottle is not None:
        return
    import earth2studio.models.dx.cbottle_sr as sr_module

    def equiangular(nlat, nlon, includes_south_pole=True):
        return SimpleNamespace(
            lat=np.linspace(90, -90, nlat)[:, None],
            lon=np.linspace(0, 360, nlon, endpoint=False),
        )

    monkeypatch.setattr(
        sr_module,
        "earth2grid",
        SimpleNamespace(
            latlon=SimpleNamespace(
                LatLonGrid=lambda lat, lon: SimpleNamespace(
                    lat=np.asarray(lat)[:, None], lon=np.asarray(lon)
                ),
                equiangular_lat_lon_grid=equiangular,
            ),
            get_regridder=lambda src, dest: _PointRegridder(dest),
        ),
    )

    def initialize(
        self,
        core,
        lat_lon=True,
        output_resolution=(2161, 4320),
        region=None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        self.sampler_steps = kwargs.get("sampler_steps", 18)
        self.input_type = self.output_type = "latlon" if lat_lon else "healpix"
        self.register_buffer("_device_buffer", torch.empty(0))
        self.input_grid = SimpleNamespace(
            lat=np.linspace(90, -90, 721), lon=np.arange(1440) / 4
        )
        self.hpx_high_res_grid = "healpix-l10-nested"
        self.sigma_max = 800
        self._coords = None
        self.regrid_input_to_hpx_low_res = lambda x: x
        self._reorder_to_sr_channels = lambda x: x
        self._reorder_from_sr_channels = lambda x: x
        output_shape = (12, 48) if self.output_type == "latlon" else (12, 12582912)

        class Core:
            extents = []

            def __call__(self, x, **kwargs):
                self.extents.append(kwargs.get("extents"))
                steps = torch.arange(
                    self.num_steps, device=x.device, dtype=torch.float64
                )
                schedule = (
                    800 ** (1 / 7)
                    + steps / (self.num_steps - 1) * (0.02 ** (1 / 7) - 800 ** (1 / 7))
                ) ** 7
                out = (
                    (torch.rand((), device=x.device) * schedule[0] / 800)
                    .float()
                    .expand(output_shape)
                )
                return out[None, :, None], None

        self.sr_model = Core()
        window, full_longitude = (
            self._region_window(region) if region is not None else (None, False)
        )
        self._configure_window(window, output_resolution, full_longitude)

    monkeypatch.setattr(CBottleSR, "__init__", initialize)
    monkeypatch.setattr(sr_module, "replace", lambda x: x)


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
        "device,region",
        [
            ("cuda:0", Region.box(-120, 0, -40, 50)),
            ("cuda:0", None),
            ("cuda:0", Region.band(-30, 30)),
        ],  # Skipping CPU tests, should work be too slow
    )
    def test_cbottle_sr_latlon(
        self, x, device, output_resolution, region, mock_cbottle_core_model
    ):
        # Create CBottleSR model with mock core model
        dx = CBottleSR(
            mock_cbottle_core_model,
            lat_lon=True,
            output_resolution=output_resolution,
            region=region,
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
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dx = CBottleSR(
            mock_cbottle_core_model,
            lat_lon=True,
            output_resolution=(721, 1440),
            sampler_steps=2,  # Smallest valid EDM schedule
            sigma_max=800,  # Reduced for testing
        ).to(device)
        forward = dx._forward

        def finite_forward(x):
            out = forward(x)
            assert torch.isfinite(out).all()
            return out

        monkeypatch.setattr(dx, "_forward", finite_forward)
        assert check_diagnostic_contract(dx, device=device) == []


def _same_signature(a, b):
    return (
        a.dims == b.dims
        and a.shape == b.shape
        and a.attrs == b.attrs
        and all(np.array_equal(a[k].values, b[k].values) for k in a.coords)
        and set(a.coords) == set(b.coords)
    )


class TestCBottleSRRegion:
    def test_cbottle_sr_with_region(self, mock_cbottle_core_model):
        dx = CBottleSR(mock_cbottle_core_model, output_resolution=(16, 32))
        assert isinstance(dx, DomainAdaptable)
        signature = dx.output_coords(dx.input_coords())
        lat, lon = dx.output_lat.copy(), dx.output_lon.copy()
        regridder = dx.regrid_hpx_high_res_to_output

        view = dx.with_region(Region.box(-120, 0, -40, 50))
        coords = view.output_coords(view.input_coords())
        assert coords.sizes["lat"] == 16 and coords.sizes["lon"] == 32
        np.testing.assert_allclose(coords.lat[[0, -1]], [0, 50])
        np.testing.assert_allclose(coords.lon[[0, -1]], [-120, -40])
        assert view.super_resolution_extents == (-120, -40, 0, 50)
        assert view.sr_model is dx.sr_model
        assert view.regrid_hpx_high_res_to_output is not regridder

        # The parent is unchanged
        assert dx.super_resolution_extents is None
        np.testing.assert_array_equal(dx.output_lat, lat)
        np.testing.assert_array_equal(dx.output_lon, lon)
        assert dx.regrid_hpx_high_res_to_output is regridder
        assert _same_signature(dx.output_coords(dx.input_coords()), signature)

        # The constructor region selects the same grid as with_region
        built = CBottleSR(
            mock_cbottle_core_model,
            output_resolution=(16, 32),
            region=Region.box(-120, 0, -40, 50),
        )
        assert _same_signature(built.output_coords(built.input_coords()), coords)
        assert built.super_resolution_extents == view.super_resolution_extents

        # A view reaches the whole globe, as the parent does
        for region in (Region.box(10, -20, 30, 20), Region.band(-90, 90)):
            a = view.with_region(region).output_coords(dx.input_coords())
            b = dx.with_region(region).output_coords(dx.input_coords())
            assert _same_signature(a, b)
        globe = dx.with_region(Region.band(-90, 90))
        assert _same_signature(globe.output_coords(dx.input_coords()), signature)

    def test_cbottle_sr_with_region_antimeridian(self, mock_cbottle_core_model):
        dx = CBottleSR(mock_cbottle_core_model, output_resolution=(8, 21))
        view = dx.with_region(Region.box(170, -10, -170, 10))
        lon = np.asarray(view.output_coords(view.input_coords()).lon)
        np.testing.assert_allclose(lon[[0, -1]], [170, 190])
        assert np.all(np.diff(lon) > 0)
        west, east = view.super_resolution_extents[:2]
        assert (west % 360, east % 360) == (170, 190)

    def test_cbottle_sr_with_region_band(self, mock_cbottle_core_model):
        dx = CBottleSR(mock_cbottle_core_model, output_resolution=(8, 16))
        view = dx.with_region(Region.band(-30, 30))
        coords = view.output_coords(view.input_coords())
        np.testing.assert_allclose(coords.lat[[0, -1]], [-30, 30])
        np.testing.assert_allclose(coords.lon, np.arange(16) * 22.5)
        west, east, south, north = view.super_resolution_extents
        # cBottle reduces longitudes modulo 360, so the extents must not collapse
        assert (south, north) == (-30, 30)
        assert west % 360 < east % 360 and east - west > 359.99

    @pytest.mark.parametrize(
        "region,match",
        [
            (Region.rectangle(40, -100, 1e5), "box and band"),
            (Region.circle(40, -100, 1e5), "box and band"),
            (Region.box(-100, 30, -100, 40), "positive extent"),
            (Region.box(-110, 30, -100, 30), "positive extent"),
            (Region.band(30, 30), "positive extent"),
        ],
    )
    def test_cbottle_sr_with_region_rejects(
        self, mock_cbottle_core_model, region, match
    ):
        dx = CBottleSR(mock_cbottle_core_model, output_resolution=(8, 16))
        with pytest.raises(ValueError, match=match):
            dx.with_region(region)

    def test_cbottle_sr_with_region_healpix(self, mock_cbottle_core_model):
        dx = CBottleSR(mock_cbottle_core_model, lat_lon=False)
        view = dx.with_region(Region.box(-120, 0, -40, 50))
        assert view.super_resolution_extents == (-120, -40, 0, 50)
        assert _same_signature(
            view.output_coords(view.input_coords()),
            dx.output_coords(dx.input_coords()),
        )

    @pytest.mark.skipif(cbottle is not None, reason="inspects the offline backend")
    def test_cbottle_sr_with_region_forward(self, mock_cbottle_core_model):
        dx = CBottleSR(mock_cbottle_core_model, output_resolution=(8, 16))
        dx.set_rng(3)
        view = dx.with_region(Region.box(-120, 0, -40, 50))
        x = torch.randn(1, 12, 721, 1440)
        coords = OrderedDict(
            {
                "batch": np.ones(1),
                "variable": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )
        dx.sr_model.extents.clear()
        out_view = view(_field(view, x, coords))
        out_dx = dx(_field(dx, x, coords))
        assert dx.sr_model.extents == [(-120, -40, 0, 50), None]
        assert out_view.shape == out_dx.shape == (1, 12, 8, 16)
        # The view's stream is a copy: it reproduces the parent's first draw
        torch.testing.assert_close(out_view.e2s.to_torch()[0], out_dx.e2s.to_torch()[0])

    @pytest.mark.skipif(cbottle is not None, reason="inspects the offline backend")
    def test_cbottle_sr_views_follow_shared_network(self, mock_cbottle_core_model):
        dx = CBottleSR(mock_cbottle_core_model, output_resolution=(8, 16))
        # A shared module network, so its device is observable
        dx.sr_model = torch.nn.Linear(1, 1)
        view = dx.with_region(Region.box(-120, 0, -40, 50))
        view.to("meta")
        assert dx.sr_model.weight.device.type == "meta"
        assert dx.device.type == "cpu"
        dx._follow_network()
        assert dx.device.type == "meta"


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
    ).to(device)
    dx.set_rng(42)

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

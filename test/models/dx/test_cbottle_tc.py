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
from datetime import datetime, timedelta

import numpy as np
import pytest
import torch
import xarray as xr

try:
    import cbottle
    from cbottle.datasets import base
    from cbottle.inference import MixtureOfExpertsDenoiser
except ImportError:
    cbottle = None

from types import SimpleNamespace

from test_cbottle_infill import _field

from earth2studio.models.conformance import (
    ContractException,
    check_diagnostic_contract,
)
from earth2studio.models.dx import CBottleTCGuidance
from earth2studio.utils import handshake_dim, handshake_metadata


@pytest.fixture(autouse=True)
def offline_tc(monkeypatch):
    if cbottle is not None:
        return

    def initialize(
        self, core_model, classifier, sst, lat_lon=True, seed=None, **kwargs
    ):
        torch.nn.Module.__init__(self)
        self.sst = sst
        self.lat_lon = lat_lon
        self.seed = seed
        self.sigma_max = 200
        self.sampler_steps = 2
        self.batch_size = 2
        self.guidance_scale = 1
        self.dataset_modality = 1
        self.register_buffer("device_buffer", torch.empty(0))
        self.register_buffer(
            "lat_grid", torch.linspace(90, -90, 721, dtype=torch.float64)
        )
        self.register_buffer("lon_grid", torch.arange(1440, dtype=torch.float64) / 4)
        grid = SimpleNamespace(
            shape=(768,), ang2pix=lambda lon, lat: (lon.abs().long() % 768)
        )

        def sample(batch, seed=None, **kwargs):
            target = batch["target"]
            gen = (
                torch.Generator(device=target.device).manual_seed(seed)
                if seed is not None
                else None
            )
            return torch.randn(
                target.shape, device=target.device, generator=gen
            ), SimpleNamespace(grid=grid)

        self.core_model = SimpleNamespace(
            sample=sample,
            classifier_grid=grid,
            net=SimpleNamespace(domain=SimpleNamespace(_grid=grid)),
            _normalize=lambda x: x,
            _reorder=lambda x: x,
            translate=lambda batch, **kw: (batch["target"], None),
        )

    def prepare(self, times, **kwargs):
        self._validate_sst_time(times)
        x = torch.zeros(len(times), 45, 1, 49152, device=self.device_buffer.device)
        return {
            "target": x,
            "labels": x[:, :1, 0, :1],
            "condition": x[:, :1],
            "second_of_day": x[:, 0, 0, :1],
            "day_of_year": x[:, 0, 0, :1],
        }

    monkeypatch.setattr(CBottleTCGuidance, "__init__", initialize)
    monkeypatch.setattr(CBottleTCGuidance, "get_cbottle_input", prepare)
    monkeypatch.setattr(
        CBottleTCGuidance,
        "regrid_hpx_to_latlon",
        lambda self, x, grid: x.mean(-1, keepdim=True)
        .unsqueeze(-1)
        .expand(-1, -1, -1, 721, 1440),
    )


@pytest.fixture(scope="class")
def mock_sst_ds() -> torch.nn.Module:
    times = [np.datetime64("1870-01-16T12:00:00"), np.datetime64("2022-12-16T12:00:00")]
    lats = np.arange(-89.5, 90, 1.0)
    lons = np.arange(0.5, 360, 1.0)
    data = np.full((2, 180, 360), -1.8)  # In Celcius
    return xr.Dataset(
        data_vars=dict(
            tosbcs=xr.DataArray(
                data=data,
                dims=["time", "lat", "lon"],
                coords={"time": times, "lat": lats, "lon": lons},
            )
        )
    )


@pytest.fixture(scope="class")
def mock_core_model() -> torch.nn.Module:
    if cbottle is None:
        return torch.nn.Identity()
    # Real model checkpoint has
    # {"model_channels": 192, "label_dim": 1024, "out_channels": 45, "condition_channels": 1}
    model_config = cbottle.config.models.ModelConfigV1()
    model_config.model_channels = 4
    model_config.label_dim = 1024
    model_config.out_channels = 45
    model_config.condition_channels = 1
    model_config.level = 2
    model1 = cbottle.models.get_model(model_config)
    model2 = cbottle.models.get_model(model_config)
    return MixtureOfExpertsDenoiser(
        [model1, model2],
        (100.0, 10.0),
        batch_info=base.BatchInfo(CBottleTCGuidance.output_variables),
    )


@pytest.fixture(scope="class")
def mock_classifier_model() -> torch.nn.Module:
    if cbottle is None:
        return torch.nn.Identity()
    # Real model checkpoint has
    # {"model_channels": 192, "label_dim": 1024, "out_channels": 45, "condition_channels": 1}
    model_config = cbottle.config.models.ModelConfigV1()
    model_config.model_channels = 4
    model_config.label_dim = 1024
    model_config.out_channels = 45
    model_config.condition_channels = 1
    model_config.level = 2
    model_config.enable_classifier = True
    return cbottle.models.get_model(model_config)


class TestCBottleTCMock:
    @pytest.mark.parametrize(
        "lat_coords,lon_coords",
        [
            (torch.tensor([30.0]), torch.tensor([120.0])),  # Single point
            (torch.tensor([30.0, 45.0]), torch.tensor([120.0, -80.0])),  # Two points
            (torch.tensor([-30.0]), torch.tensor([0.0])),  # Southern hemisphere
        ],
    )
    @pytest.mark.parametrize(
        "times", [[datetime(1990, 1, 1)], [datetime(1990, 1, 1), datetime(1990, 1, 2)]]
    )
    def test_create_guidance_tensor(
        self,
        lat_coords,
        lon_coords,
        times,
        mock_core_model,
        mock_classifier_model,
        mock_sst_ds,
    ):
        """Test guidance tensor creation with different coordinate combinations"""
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds)
        field = dx.create_guidance_tensor(lat_coords, lon_coords, times)
        coords = {k: v.values for k, v in field.coords.items()}
        guidance = field.e2s.to_torch()[0]

        assert guidance.shape == (len(times), 1, 1, 721, 1440)
        assert guidance.dtype == torch.float32
        assert torch.sum(torch.nan_to_num(guidance, nan=0.0)) == len(times) * len(
            lat_coords
        )  # One point per coordinate pair

        assert "time" in coords
        assert "lead_time" in coords
        assert "variable" in coords
        assert "lat" in coords
        assert "lon" in coords
        assert coords["variable"] == ["tc_guidance"]

    @pytest.mark.parametrize(
        "x,time,dataset_modality",
        [
            (torch.zeros(1, 1, 1, 1, 721, 1440), np.array([datetime(2020, 1, 1)]), 1),
            (
                torch.zeros(1, 2, 1, 1, 721, 1440),
                np.array([datetime(2000, 1, 2, 3, 4, 5), datetime(1980, 8, 1)]),
                1,
            ),
            (
                torch.zeros(2, 2, 1, 1, 721, 1440),
                np.array([datetime(2000, 1, 2, 3, 4, 5), datetime(1980, 8, 1)]),
                0,
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_tc_forward(
        self,
        x,
        time,
        dataset_modality,
        device,
        mock_core_model,
        mock_classifier_model,
        mock_sst_ds,
        monkeypatch,
    ):
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds).to(
            device
        )
        dx.sampler_steps = 2  # Speed up sampler
        dx.batch_size = 2
        dx.dataset_modality = dataset_modality

        coords = OrderedDict(
            {
                "batch": np.arange(x.shape[0]),
                "time": time,
                "lead_time": np.array([timedelta(hours=6)]),
                "variable": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )
        x = x.to(device)
        field = _field(dx, x, coords)
        seen_times = []
        with pytest.raises(
            ValueError, match="lead_time must contain nonempty finite timedeltas"
        ):
            dx.output_coords(field.assign_coords(lead_time=[6]))
        with pytest.raises(
            ValueError, match="lead_time must contain nonempty finite timedeltas"
        ):
            dx.output_coords(
                field.assign_coords(lead_time=np.array(["NaT"], dtype="timedelta64[h]"))
            )
        with pytest.raises(
            ValueError, match="lead_time must contain nonempty finite timedeltas"
        ):
            dx.output_coords(field.isel(lead_time=slice(0, 0)))
        prepare = dx.get_cbottle_input

        def record_times(times, *args, **kwargs):
            seen_times.extend(times)
            return prepare(times, *args, **kwargs)

        monkeypatch.setattr(dx, "get_cbottle_input", record_times)
        output = dx(field)
        np.testing.assert_array_equal(output.lead_time, field.lead_time)
        np.testing.assert_array_equal(
            np.asarray(seen_times, dtype="datetime64[ns]"),
            np.tile(field.time.values + np.timedelta64(6, "h"), x.shape[0]),
        )
        out_coords = {k: output.coords[k].values for k in output.dims}
        out = output.e2s.to_torch()[0]

        assert out.shape == torch.Size(
            [x.shape[0], x.shape[1], x.shape[2], 45, 721, 1440]
        )
        assert np.all(out_coords["variable"] == dx.output_coords(field)["variable"])
        handshake_dim(out_coords, "lon", 5)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "batch", 0)

    @pytest.mark.parametrize(
        "x,time",
        [
            (torch.zeros(1, 1, 1, 768), np.array([datetime(2020, 1, 1)])),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_tc_forward_hpx(
        self, x, time, device, mock_core_model, mock_classifier_model, mock_sst_ds
    ):
        dx = CBottleTCGuidance(
            mock_core_model, mock_classifier_model, mock_sst_ds, lat_lon=False, seed=0
        ).to(device)
        dx.sampler_steps = 2  # Speed up sampler
        dx.batch_size = 2

        coords = OrderedDict(
            {
                "time": time,
                "lead_time": np.array([timedelta(hours=6), timedelta(hours=12)]),
                "variable": dx.input_coords()["variable"],
                "hpx": np.arange(x.shape[-1]),
            }
        )
        x = x.to(device).repeat(1, 2, 1, 1)
        field = _field(dx, x, coords)
        assert field.attrs["ordering"] == "xy"
        assert field.attrs["layout"] == "flat"
        assert field.attrs["origin"] == "north"
        assert field.attrs["clockwise"] is True
        bad_grid = field.copy(deep=False)
        bad_grid.attrs = {**field.attrs, "ordering": "nested"}
        with pytest.raises(ValueError):
            handshake_metadata(bad_grid, dx.input_coords(), ("ordering",))
        output = dx(field)
        np.testing.assert_array_equal(output.lead_time, field.lead_time)
        out_coords = {k: output.coords[k].values for k in output.dims}
        out = output.e2s.to_torch()[0]

        assert out.shape == torch.Size([x.shape[0], x.shape[1], 45, 49152])
        assert np.all(out_coords["variable"] == dx.output_coords(field)["variable"])
        handshake_dim(out_coords, "hpx", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "lead_time", 1)
        handshake_dim(out_coords, "time", 0)

    def test_calculate_odds_ratio_unit_contract_forwards_to_core(
        self, mock_core_model, mock_classifier_model, mock_sst_ds, monkeypatch
    ):
        """Unit contract: wrapper forwards inputs/kwargs and returns core outputs."""
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds).to(
            "cpu"
        )
        guidance = dx.create_guidance_tensor(
            torch.tensor([30.0]),
            torch.tensor([120.0]),
            [datetime(2000, 1, 1)],
        )

        called: dict = {}

        def _fake_calculate_odds_ratio(batch, guidance_pixels, **kwargs):
            called["batch"] = batch
            called["guidance_pixels"] = guidance_pixels
            called["kwargs"] = kwargs
            return 1.5, torch.zeros_like(batch["target"])

        def _fake_regrid(x, grid):
            # Return a lat-lon shaped tensor
            return x.new_zeros(*x.shape[:-1], 721, 1440)

        monkeypatch.setattr(
            dx.core_model,
            "calculate_odds_ratio",
            _fake_calculate_odds_ratio,
            raising=False,
        )
        monkeypatch.setattr(dx, "regrid_hpx_to_latlon", _fake_regrid)

        log_odds_ratio, forward_latents = dx.calculate_odds_ratio(guidance)
        latent_coords = forward_latents.coords

        assert log_odds_ratio == pytest.approx(1.5)
        assert isinstance(called["batch"], dict)
        assert called["guidance_pixels"].ndim == 1
        assert called["guidance_pixels"].numel() > 0
        assert called["kwargs"]["num_steps"] == dx.sampler_steps
        assert called["kwargs"]["guidance_scale"] == 128
        assert called["kwargs"]["compute_forward_divergences"] is False
        assert "variable" in latent_coords
        assert "lat" in latent_coords
        assert "lon" in latent_coords

    def test_calculate_odds_ratio_rejects_multiple_samples(
        self, mock_core_model, mock_classifier_model, mock_sst_ds
    ):
        """Unit guardrail: odds-ratio path is defined for one flattened sample."""
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds).to(
            "cpu"
        )
        guidance = dx.create_guidance_tensor(
            torch.tensor([30.0]),
            torch.tensor([120.0]),
            [datetime(2000, 1, 1), datetime(2000, 1, 2)],
        )

        with pytest.raises(ValueError, match="required dim time is not of size 1"):
            dx.calculate_odds_ratio(guidance)

    def test_calculate_odds_ratio_rejects_empty_guidance(
        self, mock_core_model, mock_classifier_model, mock_sst_ds, monkeypatch
    ):
        """Unit guardrail: all-NaN guidance map should be rejected."""
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds).to(
            "cpu"
        )
        guidance = dx.create_guidance_tensor(
            torch.tensor([30.0]),
            torch.tensor([120.0]),
            [datetime(2000, 1, 1)],
        )
        guidance[:] = torch.nan

        def _fake_calculate_odds_ratio(*args, **kwargs):
            raise AssertionError("core calculate_odds_ratio should not be reached")

        monkeypatch.setattr(
            dx.core_model,
            "calculate_odds_ratio",
            _fake_calculate_odds_ratio,
            raising=False,
        )

        with pytest.raises(ValueError, match="No guidance pixels set"):
            dx.calculate_odds_ratio(guidance)

    def test_validate_sst_time_valid(
        self, mock_core_model, mock_classifier_model, mock_sst_ds
    ):
        valid_times = [
            datetime(1950, 6, 15),
            datetime(2000, 12, 31),
            datetime(2022, 12, 15),
        ]
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds)
        # Should not raise any exceptions
        dx._validate_sst_time(valid_times)

        invalid_times = [datetime(1939, 12, 31)]
        with pytest.raises(ValueError):
            dx._validate_sst_time(invalid_times)

        invalid_times = [datetime(2022, 12, 16, 12)]
        with pytest.raises(ValueError):
            dx._validate_sst_time(invalid_times)

    def test_cbottletcguidance_conformance(
        self, mock_core_model, mock_classifier_model, mock_sst_ds
    ):
        """Check the mock CBottleTCGuidance model against the model contract.

        Probed at a time inside the default AMIP mid-month SST range: with no
        input SST fields the model rejects anything from 2022-12-16 on, and the
        checker's default probe time (2024-01-01) is outside it.

        CBottleTCGuidance does not currently declare `stochastic` or implement
        `set_rng()`; constructed without a seed — the default — its diffusion
        latents come from the unseeded global generator, so two calls on one
        input disagree and D9 is violated. Pinned here until the wrapper
        declares stochastic and implements set_rng().

        Explicitly moved to and probed on cuda:0 rather than left on whatever
        device the class-scoped fixtures happen to be on: earlier tests in this
        class move the shared fixture modules onto cuda:0 via ``.to(device)``
        without moving them back, so this test would otherwise inherit a CUDA
        model against the checker's CPU-default probe tensor depending on test
        execution order.
        """
        dx = CBottleTCGuidance(mock_core_model, mock_classifier_model, mock_sst_ds).to(
            "cuda:0"
        )
        dx.sampler_steps = 2  # Speed up sampler
        with pytest.raises(ContractException) as exc_info:
            check_diagnostic_contract(
                dx, device="cuda:0", time=np.datetime64("2022-01-01T00:00:00")
            )
        assert exc_info.value.violations == [
            "D9: repeated runs with the same input and seed disagree"
        ]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cbottle_tc_package(device):
    # Only cuda used here to speed things up, but CPU also works
    package = CBottleTCGuidance.load_default_package()
    dx = CBottleTCGuidance.load_model(package, seed=0).to(device)

    # Guidance over florida
    lat = 27
    lon = -82
    time = np.array(
        [datetime(2000, 8, 9, 10), datetime(2005, 10, 11, 12)], dtype=np.datetime64
    )
    guidance = dx.create_guidance_tensor(
        torch.tensor([lat]),
        torch.tensor([lon]),
        time,
    )
    guidance = guidance.e2s.as_cupy(0)

    output = dx(guidance)
    out_coords = {k: output.coords[k].values for k in output.dims}
    out = output.e2s.to_torch()[0]
    assert out.shape == torch.Size(
        [
            out_coords["time"].shape[0],
            out_coords["lead_time"].shape[0],
            out_coords["variable"].shape[0],
            721,
            1440,
        ]
    )
    assert np.all(out_coords["variable"] == dx.output_coords(guidance)["variable"])
    assert np.all(out_coords["time"] == time)
    handshake_dim(out_coords, "lon", -1)
    handshake_dim(out_coords, "lat", -2)
    handshake_dim(out_coords, "variable", -3)
    handshake_dim(out_coords, "lead_time", -4)
    handshake_dim(out_coords, "time", -5)

    # For a physical sanity check, see if tcwv is high at the guidance location
    vidx = np.where(out_coords["variable"] == "tcwv")[0]
    lat_idx = 4 * (90 - lat)
    lon_idx = 4 * (360 + lon)
    assert (out[:, :, vidx, lat_idx, lon_idx] >= 60).all()

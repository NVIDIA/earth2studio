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

from earth2studio.models.conformance import (
    ContractException,
    check_diagnostic_contract,
)
from earth2studio.models.dx import CBottleInfill
from earth2studio.utils import handshake_dim
from earth2studio.utils.cupy import from_torch


def _field(model, tensor, coords):
    signature = model.input_coords()
    dims = tuple(coords)
    labels = {k: np.asarray(v) for k, v in coords.items()}
    if "time" in labels:
        labels["time"] = labels["time"].astype("datetime64[ns]")
    if "lead_time" in labels:
        labels["lead_time"] = labels["lead_time"].astype("timedelta64[ns]")
    # Coordinate construction is separate from field conversion so invalid labels
    # still reach the wrapper's native validation.
    from earth2studio.utils.coords import coord_array

    sig = coord_array(dims, labels, attrs=signature.attrs)
    return from_torch(tensor, sig)


@pytest.fixture(autouse=True)
def offline_infill(monkeypatch):
    if cbottle is not None:
        return

    def initialize(self, core_model, sst_ds, input_variables, **kwargs):
        torch.nn.Module.__init__(self)
        self.sst = sst_ds
        self.input_variables = input_variables
        self.seed = None
        self.sigma_max = 200
        self.sampler_steps = 2
        self.batch_size = 4
        self.register_buffer("device_buffer", torch.empty(0))

        def infill(batch):
            target = batch["target"]
            return (
                torch.where(torch.isnan(target), torch.randn_like(target), target),
                None,
            )

        self.core_model = SimpleNamespace(infill=infill)

    def prepare(self, time, x, label=1):
        if "sst" not in self.input_variables:
            self._validate_sst_time(time)
        target = x.new_full((len(time), 45, 1, 1), torch.nan)
        target[:, self.input_variable_idx, 0, 0] = x.mean((-1, -2))
        return {
            "target": target,
            "labels": x.new_zeros(len(time), 1),
            "condition": x.new_zeros(len(time), 1),
            "second_of_day": x.new_zeros(len(time), 1),
            "day_of_year": x.new_zeros(len(time), 1),
        }

    monkeypatch.setattr(CBottleInfill, "__init__", initialize)
    monkeypatch.setattr(CBottleInfill, "get_cbottle_input", prepare)
    monkeypatch.setattr(
        CBottleInfill, "_regrid_outputs", lambda self, x: x.expand(-1, -1, 721, 1440)
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
    return MixtureOfExpertsDenoiser(
        [model1],
        (),
        batch_info=base.BatchInfo(CBottleInfill.output_variables),
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


class TestCBottleMock:
    @pytest.mark.parametrize(
        "input_variables",
        [
            np.array(["u10m", "v10m"]),
            np.array(["t2m", "z1000", "sic"]),
        ],
    )
    @pytest.mark.parametrize(
        "time,lead_time",
        [
            (
                np.array(
                    [datetime(2020, 1, 1, 6, 2, 3), datetime(1990, 5, 6, 7, 8, 9)]
                ),
                np.array([timedelta(0)]),
            ),
            (
                np.array([datetime(2006, 12, 13, 12, 36)]),
                np.array([timedelta(hours=6), timedelta(hours=12, minutes=5)]),
            ),
        ],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_infill(
        self, input_variables, time, lead_time, device, mock_core_model, mock_sst_ds
    ):
        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        x = torch.randn(
            time.shape[0], lead_time.shape[0], input_variables.shape[0], 721, 1440
        ).to(device)
        coords = OrderedDict(
            {
                "time": time,
                "lead_time": lead_time,
                "variable": input_variables,
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        field = _field(dx, x, coords)
        field.attrs["user"] = {"notes": ["input"]}
        field.encoding["user"] = {"notes": ["input"]}
        field.time.attrs["user"] = {"notes": ["input"]}
        planned = dx.output_coords(field)
        assert planned.data.nbytes == 0
        assert planned.attrs["user"] == field.attrs["user"]
        assert planned.time.attrs["user"] == field.time.attrs["user"]
        assert field.attrs["user"]["notes"] == ["input"]
        assert field.encoding["user"]["notes"] == ["input"]
        assert field.time.attrs["user"]["notes"] == ["input"]
        output = dx(field)
        out_coords = {k: output.coords[k].values for k in output.dims}
        out = output.e2s.to_torch()[0]

        assert out.shape == torch.Size(
            [
                time.shape[0],
                lead_time.shape[0],
                out_coords["variable"].shape[0],
                721,
                1440,
            ]
        )
        assert np.all(out_coords["variable"] == dx.output_coords(field)["variable"])
        assert np.all(out_coords["time"] == time.astype("datetime64[ns]"))
        assert np.all(out_coords["lead_time"] == lead_time.astype("timedelta64[ns]"))
        handshake_dim(out_coords, "lon", 4)
        handshake_dim(out_coords, "lat", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "lead_time", 1)
        handshake_dim(out_coords, "time", 0)
        assert not torch.isnan(out).any()
        # Assert the provided fields the same (fairly close, theres still interpolation)
        torch.allclose(out[:, :, dx.input_variable_idx], x, rtol=0.05)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_cbottle_infill_exceptions(self, device, mock_core_model, mock_sst_ds):

        dx = CBottleInfill(mock_core_model, mock_sst_ds, ["t2m"]).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        x = torch.randn(1).to(device)
        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "time": dx.input_coords()["time"],
                "lead_time": dx.input_coords()["lead_time"],
                "wrong": dx.input_coords()["variable"],
                "lat": dx.input_coords()["lat"],
                "lon": dx.input_coords()["lon"],
            }
        )

        with pytest.raises((KeyError, ValueError)):
            dx(_field(dx, x, wrong_coords))

        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "time": dx.input_coords()["time"],
                "variable": dx.input_coords()["variable"],
                "lon": dx.input_coords()["lon"],
                "lat": dx.input_coords()["lat"],
            }
        )

        with pytest.raises(ValueError):
            dx(_field(dx, x, wrong_coords))

        wrong_coords = OrderedDict(
            {
                "batch": np.ones(x.shape[0]),
                "time": dx.input_coords()["time"],
                "lead_time": dx.input_coords()["lead_time"],
                "variable": dx.input_coords()["variable"],
                "lat": np.linspace(-90, 90, 720),
                "lon": dx.input_coords()["lon"],
            }
        )
        with pytest.raises(ValueError):
            dx(_field(dx, x, wrong_coords))

    @pytest.mark.parametrize(
        "input_variables",
        [
            np.array(["u10m"]),
            np.array(["sst", "z1000", "sic"]),
            np.array(["v1000", "sst"]),
        ],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_cbottle_infill_sst(
        self, input_variables, device, mock_core_model, mock_sst_ds
    ):

        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        # With AMIP time range
        time = np.array([datetime(2020, 1, 1, 1), datetime(2021, 1, 1, 1)])
        lead_time = np.array([timedelta(hours=1)])

        x = torch.randn(
            1, time.shape[0], lead_time.shape[0], input_variables.shape[0], 721, 1440
        ).to(device)
        coords = OrderedDict(
            {
                "ensemble": np.array([1]),
                "time": time,
                "lead_time": lead_time,
                "variable": input_variables,
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        dx(_field(dx, x, coords))

        # Outside of AMIP time range
        coords["time"] = np.array([datetime(2023, 1, 1, 1), datetime(2002, 2, 2, 2)])

        if "sst" in input_variables:
            dx(_field(dx, x, coords))
        else:
            with pytest.raises(ValueError):
                dx(_field(dx, x, coords))

    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_cbottle_infill_invariant_inputs(
        self, device, mock_core_model, mock_sst_ds
    ):
        # Checks a few invariant inputs that should produce the same result
        input_variables = np.array(["u10m", "v10m"])
        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        time = np.array([datetime(1995, 8, 2, 3, 12)])
        lead_time = np.array([timedelta(hours=6)])

        x = torch.randn(
            1, time.shape[0], lead_time.shape[0], input_variables.shape[0], 721, 1440
        ).to(device)
        coords = OrderedDict(
            {
                "ensemble": np.array([1]),
                "time": time,
                "lead_time": lead_time,
                "variable": input_variables,
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        out0 = dx(_field(dx, x, coords)).e2s.to_torch()[0]

        # Adjust time and lead time dim so data is at same timestamp
        coords["time"] = np.array([datetime(1995, 8, 2, 9, 12)])
        coords["lead_time"] = np.array([timedelta(hours=0)])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        out1 = dx(_field(dx, x, coords)).e2s.to_torch()[0]

        # Permute variables
        input_variables = np.array(["v10m", "u10m"])
        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to(device)
        dx.sampler_steps = 2  # Speed up sampler

        coords["variable"] = input_variables
        x = torch.flip(x, dims=[-3])
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        np.random.seed(0)
        out2 = dx(_field(dx, x, coords)).e2s.to_torch()[0]

        assert torch.allclose(out0, out1)
        assert torch.allclose(out0, out2)

    def test_cbottleinfill_conformance(self, mock_core_model, mock_sst_ds):
        """Check the mock CBottleInfill model against the model contract.

        Probed at a time inside the default AMIP mid-month SST range: with no
        input SST fields the model rejects anything from 2022-12-16 on, and the
        checker's default probe time (2024-01-01) is outside it.

        Explicitly moved to and probed on cuda:0 rather than left on whatever
        device the class-scoped ``mock_core_model``/``mock_sst_ds`` fixtures
        happen to be on: earlier tests in this class move the shared fixture
        modules onto cuda:0 via ``.to(device)`` without moving them back, so
        this test would otherwise inherit a CUDA model against the checker's
        CPU-default probe tensor depending on test execution order.

        CBottleInfill does not currently declare `stochastic` or implement
        `set_rng()` — the sampler call has no seed argument to pass one to at
        all ("NO SEED SUPPORT!" in cbottle_infill.py) — so its diffusion latents
        come from the unseeded global generator and two calls on one input
        disagree, violating D9. Pinned here until the wrapper can be seeded.
        """
        input_variables = np.array(["u10m", "v10m"])
        dx = CBottleInfill(mock_core_model, mock_sst_ds, input_variables).to("cuda:0")
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
def test_cbottle_package(device):
    # Test the cached model package
    # Only cuda supported
    input_variables = np.array(["tpf"])
    package = CBottleInfill.load_default_package()
    dx = CBottleInfill.load_model(package, input_variables=input_variables).to(device)

    time = np.array([datetime(2020, 1, 1, 1), datetime(2021, 1, 1, 1)])
    lead_time = np.array([timedelta(hours=1)])
    x = torch.zeros(
        1, time.shape[0], lead_time.shape[0], input_variables.shape[0], 721, 1440
    ).to(device)
    coords = OrderedDict(
        {
            "ensemble": np.array([1]),
            "time": time,
            "lead_time": lead_time,
            "variable": input_variables,
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )

    field = _field(dx, x, coords)
    output = dx(field)
    out_coords = {k: output.coords[k].values for k in output.dims}
    out = output.e2s.to_torch()[0]

    assert out.shape == torch.Size(
        [
            1,
            time.shape[0],
            lead_time.shape[0],
            out_coords["variable"].shape[0],
            721,
            1440,
        ]
    )
    assert np.all(out_coords["variable"] == dx.output_coords(field)["variable"])
    assert np.all(out_coords["time"] == time.astype("datetime64[ns]"))
    assert np.all(out_coords["lead_time"] == lead_time.astype("timedelta64[ns]"))
    handshake_dim(out_coords, "lon", -1)
    handshake_dim(out_coords, "lat", -2)
    handshake_dim(out_coords, "variable", -3)
    handshake_dim(out_coords, "lead_time", -4)
    handshake_dim(out_coords, "time", -5)

    # Check physical ranges
    vidx = np.where(out_coords["variable"] == "sic")[0]
    assert (out[:, :, :, vidx] >= -0.1).all() and (out[:, :, :, vidx] <= 1.1).all()
    vidx = np.where(out_coords["variable"] == "u10m")[0]
    assert (out[:, :, :, vidx] >= -40).all() and (out[:, :, :, vidx] <= 40).all()
    vidx = np.where(out_coords["variable"] == "t2m")[0]
    assert (out[:, :, :, vidx] >= 184).all() and (out[:, :, :, vidx] <= 330).all()

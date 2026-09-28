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

import inspect
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.data import Random, fetch_data
from earth2studio.grids import LatLonGrid
from earth2studio.models.conformance import (
    ContractException,
    check_prognostic_contract,
)
from earth2studio.models.px import FCN3
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


@pytest.fixture(autouse=True)
def optional_backend(request):
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")
    if request.node.originalname != "test_fcn3_iter":
        pytest.importorskip("makani")


class PhooFCN3Preprocessor(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.register_buffer(
            "state",
            torch.randn(
                10,
            ),
        )
        self.refreshes = 0

    def set_internal_state(self, state: torch.Tensor):
        self.state = state.to(self.state.device)

    def get_internal_state(self, tensor=True):
        return self.state

    def update_internal_state(self, replace_state=True):
        self.refreshes += 1
        self.state = torch.randn((10,), device=self.state.device)


class PhooFCN3Model(torch.nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor


class PhooFCN3ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._generator = None

    def forward(self, x, t, normalized_data: bool = False, replace_state: bool = False):
        # Deterministic (identity) unless set_rng() has seeded a local generator,
        # mirroring the real core model's noise-conditioned forward so that FCN3's
        # declared stochastic=True is exercisable by the conformance rollout rules.
        if self._generator is None:
            return x
        noise = torch.randn(x.shape, generator=self._generator).to(x.device)
        return x + noise

    def set_rng(self, reset: bool = True, seed: int = 333):
        if reset or self._generator is None:
            self._generator = torch.Generator().manual_seed(seed)


@pytest.fixture(scope="function")
def dummy_model():
    preprocessor = PhooFCN3Preprocessor()
    model = PhooFCN3Model(preprocessor)
    return model


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn3_call(time, device, dummy_model):

    # Spoof model
    model = PhooFCN3ModelWrapper(dummy_model)
    p = FCN3(model).to(device)

    # Create "domain coords"
    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(r, time, variable, lead_time, device=device)
    x.attrs.update(earth2studio_grid_id="latlon-0.25deg", earth2studio_crs="EPSG:4326")
    coords = x
    out = p(x)
    out_coords = out.coords

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 72, 721, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    assert out.dims == ("time", "lead_time", "variable", "lat", "lon")


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn3_iter(ensemble, device, dummy_model, monkeypatch):

    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooFCN3ModelWrapper(dummy_model)
    p = FCN3.__new__(FCN3)
    inspect.unwrap(FCN3.__init__)(p, model, variables=np.array(["t2m"]))
    p.to(device)
    declared = p.input_coords()
    assert (
        declared.data.nbytes == 0
        and declared.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    )
    signature = coord_array(
        declared.dims,
        {"lead_time": declared.lead_time, "variable": declared.coords["variable"]},
        dynamic=("batch", "time"),
        grid=LatLonGrid([45, -45], [0, 120, 240]),
    )
    monkeypatch.setattr(p, "input_coords", lambda: signature.copy())

    # Create "domain coords"
    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(r, time, variable, lead_time, device=device)
    x.attrs.update(earth2studio_crs="EPSG:4326")
    x = x.expand_dims(ensemble=np.arange(ensemble))
    x = x.rename("weather")
    x = from_torch(x.e2s.to_torch()[0].to(device), coord_array_like(x), name=x.name)
    x.encoding = {"source": "fixture"}
    original = x.copy(deep=True)
    monkeypatch.setattr(
        model,
        "forward",
        lambda value, t, **kwargs: value + dummy_model.preprocessor.state[0],
    )
    p_iter = p.create_iterator(x)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    initial = next(p_iter)
    assert dummy_model.preprocessor.refreshes == 0
    retained = []
    for i, out in enumerate(p_iter):
        out_coords = out.coords
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 1, 2, 3])
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
        assert out.name == x.name and out.encoding == x.encoding
        retained.append((out, out.copy(deep=True)))
        if i == 1:
            torch.testing.assert_close(
                out.e2s.to_torch()[0] - retained[0][0].e2s.to_torch()[0],
                retained[0][0].e2s.to_torch()[0] - x.e2s.to_torch()[0],
            )

        if i > 5:
            break
    # The iterator draws one noise state per member, then restores it each step.
    assert dummy_model.preprocessor.refreshes == ensemble
    for out, saved in retained:
        xr.testing.assert_identical(out, saved)
    xr.testing.assert_identical(initial, original)
    xr.testing.assert_identical(x, original)
    assert retained[-1][0].e2s.to_torch()[0].device == torch.device(device)
    p_iter.close()
    monkeypatch.setattr(model, "forward", PhooFCN3ModelWrapper.forward.__get__(model))
    p.set_rng(17)
    first = p(x)
    generator = model._generator
    p.set_rng(18, reset=False)
    assert model._generator is generator
    p.set_rng(17)
    torch.testing.assert_close(
        first.e2s.to_torch()[0], p(x).e2s.to_torch()[0], rtol=0, atol=0
    )
    p.set_rng(18)
    assert not torch.equal(first.e2s.to_torch()[0], p(x).e2s.to_torch()[0])


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn3_exceptions(dc, device, dummy_model):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooFCN3ModelWrapper(dummy_model)
    p = FCN3(model).to(device)

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x)


def test_fcn3_conformance(dummy_model):
    """Check the mock FCN3 model against the Earth2Studio model contract.

    FCN3 declares stochastic=True and delegates set_rng to its core model. The
    Phoo core model seeds a local torch.Generator and adds noise from it once
    seeded, so P13 (reproducibility) and P14 (RNG isolation) are exercisable.

    Not conformant. Genuine wrapper bugs, tracked in
    test/models/test_model_conformance.py pending a fix:
    - P14: refreshing the core model's internal noise state draws from the
      global generator, so stepping a seeded model perturbs global RNG state.
    """
    model = PhooFCN3ModelWrapper(dummy_model)
    p = FCN3(model)
    with pytest.raises(ContractException) as exc_info:
        check_prognostic_contract(p)
    assert {v.split(":")[0] for v in exc_info.value.violations} == {
        "P14",
    }


@pytest.fixture(scope="function")
def model() -> FCN3:
    package = FCN3.load_default_package()
    p = FCN3.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn3_load_package(device, model):
    torch.cuda.empty_cache()
    # Test the cached model package FCN3
    model.to(device)


# Will not test while we do not have 80GB GPU cards
# in CI
# @pytest.mark.package
# @pytest.mark.timeout(360)
# @pytest.mark.parametrize("device", ["cuda:0"])
# def test_fcn3_package(device, model):
#     torch.cuda.empty_cache()
#     time = np.array([np.datetime64("1993-04-05T00:00")])
#     # Test the cached model package FCN3
#     p = model.to(device)

#     # Create "domain coords"
#     dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

#     # Initialize Data Source
#     r = Random(dc)

#     # Get Data and convert to tensor, coords
#     lead_time = p.input_coords()["lead_time"]
#     variable = p.input_coords()["variable"]
#     x, coords = fetch_data(r, time, variable, lead_time, device=device)

#     out, out_coords = p(x, coords)

#     if not isinstance(time, Iterable):
#         time = [time]

#     assert out.shape == torch.Size([len(time), 1, 72, 721, 1440])
#     assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
#     handshake_dim(out_coords, "lon", 4)
#     handshake_dim(out_coords, "lat", 3)
#     handshake_dim(out_coords, "variable", 2)
#     handshake_dim(out_coords, "lead_time", 1)
#     handshake_dim(out_coords, "time", 0)

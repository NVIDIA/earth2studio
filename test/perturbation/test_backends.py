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

"""Perturbation methods should work on NumPy, CuPy and Torch backed DataArrays"""

from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.data import Random
from earth2studio.models.batch import batch_func
from earth2studio.perturbation import (
    BredVector,
    Brown,
    CorrelatedSphericalGaussian,
    Gaussian,
    HemisphericCentredBredVector,
    LaggedEnsemble,
    SphericalGaussian,
    Zero,
)
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.cupy import _TorchArray, from_torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")

BACKENDS = [
    pytest.param(("numpy", "cpu"), id="numpy"),
    pytest.param(("cupy", "cuda:0"), id="cupy", marks=cuda),
    pytest.param(("torch", "cpu"), id="torch-cpu"),
    pytest.param(("torch", "cuda:0"), id="torch-cuda", marks=cuda),
]


def _make(shape, dims, backend):
    name, device = backend
    x = torch.randn(*shape)
    da = from_torch(
        x,
        OrderedDict((d, np.arange(n)) for d, n in zip(dims, shape)),
        attrs={"a": 1},
        backend="numpy",
    )
    return da.e2s.to_backend(name, device=device)


def _check(out, x, dtype=True):
    assert out.dims == x.dims
    assert out.shape == x.shape
    assert out.attrs == x.attrs
    assert isinstance(out.data, _TorchArray) == isinstance(x.data, _TorchArray)
    assert out.e2s.is_cupy == x.e2s.is_cupy
    out_t = out.e2s.to_torch()[0]
    x_t = x.e2s.to_torch()[0]
    assert out_t.device == x_t.device
    if dtype:
        assert out_t.dtype == x_t.dtype
    assert torch.isfinite(out_t).all()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "perturbation",
    [
        Zero(),
        Brown(),
        Gaussian(),
        SphericalGaussian(),
        CorrelatedSphericalGaussian(noise_amplitude=0.1),
    ],
    ids=lambda p: type(p).__name__,
)
def test_backend_simple(perturbation, backend):
    # N:2N lat/lon grid for spherical methods
    x = _make((2, 1, 3, 16, 32), ("ensemble", "time", "variable", "lat", "lon"), backend)
    before = x.e2s.to_torch()[0].clone()
    out = perturbation(x)
    _check(out, x)
    # Input is not modified in place
    assert torch.equal(x.e2s.to_torch()[0], before)
    if not isinstance(perturbation, Zero):
        assert not torch.equal(out.e2s.to_torch()[0], before)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_brown_numpy_seed(backend):
    if backend[0] == "torch":
        pytest.skip("Brown noise is generated with numpy/cupy RNG")
    x = _make((2, 1, 3, 16, 16), ("ensemble", "time", "variable", "lat", "lon"), backend)
    p = Brown()
    if backend[0] == "numpy":
        np.random.seed(7)
        a = p(x)
        np.random.seed(7)
        b = p(x)
    else:
        import cupy as cp

        cp.random.seed(7)
        a = p(x)
        cp.random.seed(7)
        b = p(x)
    assert torch.equal(a.e2s.to_torch()[0], b.e2s.to_torch()[0])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("model_backend", ["auto", "torch"])
@pytest.mark.parametrize("seeding", [Brown(), Gaussian()], ids=lambda p: type(p).__name__)
def test_backend_bred_vector(seeding, model_backend, backend):
    # Model output backend can differ from perturbation input backend
    def model(x: xr.DataArray) -> xr.DataArray:
        t, _ = x.e2s.to_torch()
        return from_torch(t * 0.1, x, backend=model_backend)

    x = _make((2, 1, 3, 16, 16), ("ensemble", "time", "variable", "lat", "lon"), backend)
    prtb = BredVector(model, noise_amplitude=0.1, integration_steps=2)
    prtb.seeding_perturbation_method = seeding
    _check(prtb(x), x)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("seeding", [Brown(), Gaussian()], ids=lambda p: type(p).__name__)
def test_backend_hcbv(seeding, backend):
    class FooModel(torch.nn.Module):
        def __init__(self, coords):
            super().__init__()
            self._input_coords = coords

        def input_coords(self):
            return self._input_coords

        def output_coords(self, input_coords):
            return coord_array_like(
                input_coords, {"lead_time": np.array([np.timedelta64(1, "s")])}
            )

        @batch_func()
        def forward(self, x):
            return x * 0.1

    variable = ["tcwv", "u200"]
    dc = OrderedDict([("lat", np.arange(16)), ("lon", np.arange(16))])
    fc = OrderedDict(
        [
            ("batch", np.empty(0)),
            ("lead_time", np.array([np.timedelta64(0, "h")])),
            ("variable", np.array(variable)),
        ]
    )
    fc.update(dc)
    model = FooModel(coord_array(tuple(fc), fc, dynamic=("batch",)))
    if backend[1] != "cpu":
        model = model.to(backend[1])

    x = _make(
        (2, 1, 1, 2, 16, 16),
        ("ensemble", "time", "lead_time", "variable", "lat", "lon"),
        backend,
    )
    x = x.assign_coords(
        time=np.array([np.datetime64("2021-11-23T12:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(variable),
    )
    prtb = HemisphericCentredBredVector(
        model=model,
        data=Random(dc),
        seeding_perturbation_method=seeding,
        noise_amplitude=1.0,
        integration_steps=2,
    )
    # Output dtype follows the data source in HCBV
    _check(prtb(x), x, dtype=False)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_lagged(backend):
    dc = OrderedDict([("lat", np.arange(8)), ("lon", np.arange(16))])
    lags = np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")])
    x = _make(
        (2, 1, 1, 2, 8, 16),
        ("ensemble", "time", "lead_time", "variable", "lat", "lon"),
        backend,
    )
    x = x.assign_coords(
        time=np.array([np.datetime64("2021-11-23T12:00")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(["t2m", "tcwv"]),
        lat=dc["lat"],
        lon=dc["lon"],
    )
    out = LaggedEnsemble(Random(dc), lags=lags)(x)
    _check(out, x)

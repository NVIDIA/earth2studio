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
from collections.abc import Iterable

import numpy as np
import pytest
import torch

try:
    import anemoi  # noqa: F401
except ImportError:
    pytest.skip("anemoi not installed", allow_module_level=True)

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import AIFS
from earth2studio.utils import handshake_dim


def make_two_nnz_per_first_row_csr(n_rows, n_cols, device):
    # crow_indices must be length n_rows+1 and monotone non‑decreasing
    crow = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    crow[1:] = 2  # row0 --> 2 nnz, others 0 nnz

    col = torch.tensor([0, 1], dtype=torch.int64, device=device)
    val = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)

    return torch.sparse_csr_tensor(
        crow, col, val, size=(n_rows, n_cols), dtype=torch.float32
    )


class DotDict(dict):
    """Minimal DotDict replacement with recursive dot-notation access."""

    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)  # recursively wrap dicts
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


class PhooAIFSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Mock data indices, annoying
        data_indices = DotDict()
        data_indices.data = DotDict()
        data_indices.data.input = DotDict()
        data_indices.data.output = DotDict()

        all = torch.arange(0, 116)

        data_indices = DotDict()
        data_indices.data = DotDict()
        data_indices.data.input = DotDict()
        data_indices.data.output = DotDict()

        data_indices.data.input.prognostic = torch.cat(
            [
                all[0:82],
                all[[83, 85, 87, 88, 109, 110, 112, 113]],
            ]
        )

        data_indices.data.input.forcing = torch.cat(
            [all[[82, 84, 86, 89]], all[92:101]]
        )

        data_indices.data.output.full = torch.cat(
            [all[0:82], all[[83, 85, 87, 88, 90, 91]], all[101:115]]
        )

        data_indices.data.input.full = torch.cat(
            [all[0:90], all[92:101], all[[109, 110, 112, 113]]]
        )

        # Model indices
        data_indices.model = DotDict()
        data_indices.model.input = DotDict()
        data_indices.model.input.forcing = torch.cat(
            [all[[82, 84, 86, 89]], all[90:99]]
        )

        self.data_indices = data_indices

    def predict_step(self, x):
        return torch.ones(x.shape[0], x.shape[2], 102, device=x.device)


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
def test_aifs_call(time, device):
    # Spoof model
    model = PhooAIFSModel()

    # Create tensors with the correct shapes
    # For latitudes and longitudes, we'll create a smaller tensor for testing
    # but maintain the same structure
    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    # Create sparse tensors for interpolation matrices
    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    ).to(torch.float64)

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    ).to(torch.float64)

    # Initialize AIFS
    p = AIFS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
    ).to(device)

    # Create "domain coords"
    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 106, 721, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aifs_iter(ensemble, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooAIFSModel()

    # Create tensors with the correct shapes
    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    # Create sparse tensors for interpolation matrices
    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    ).to(torch.float64)

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    ).to(torch.float64)

    p = AIFS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
    ).to(device)

    # Create "domain coords"
    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    next(p_iter)  # Skip first which should return the input
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 106, 721, 1440])
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")

        if i > 5:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aifs_exceptions(dc, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooAIFSModel()

    # Create tensors with the correct shapes
    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    # Create sparse tensors for interpolation matrices
    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    ).to(torch.float64)

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    ).to(torch.float64)

    p = AIFS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
    ).to(device)

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.fixture(scope="function")
def model(model_cache_context) -> AIFS:
    # Test only on cuda device
    with model_cache_context():
        package = AIFS.load_default_package()
        p = AIFS.load_model(package)
        return p


@pytest.mark.ci_cache
@pytest.mark.timeout(360)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aifs_package(device, model):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package AIFS
    p = model.to(device)

    assert len(p.input_variables) == 94
    assert len(p.output_variables) == 106

    # Create "domain coords"
    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 106, 721, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)

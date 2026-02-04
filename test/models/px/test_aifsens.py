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
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import AIFSENS
from earth2studio.models.px.aifsens import VARIABLES
from earth2studio.utils import handshake_dim


def make_two_nnz_per_first_row_csr(n_rows, n_cols, device):
    # crow_indices must be length n_rows+1 and monotone non-decreasing
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


class PhooAIFSENSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        data_indices = DotDict()
        data_indices.data = DotDict()
        data_indices.data.input = DotDict()
        data_indices.data.output = DotDict()

        all_idx = torch.arange(0, len(VARIABLES))

        data_indices.data.input.prognostic = torch.cat(
            [
                all_idx[0:82],
                all_idx[[83, 85, 87, 88, 101, 102]],
            ]
        )

        data_indices.data.input.forcing = torch.cat(
            [all_idx[[82, 84, 86, 89]], all_idx[92:101]]
        )

        data_indices.data.output.forcing = torch.cat(
            [all_idx[[82, 84, 86, 89]], all_idx[92:101]]
        )

        data_indices.data.input.full = torch.cat(
            [all_idx[0:90], all_idx[92:101], all_idx[[101, 102]]]
        )

        data_indices.data.output.full = torch.cat(
            [all_idx[0:82], all_idx[[83, 85, 87, 88, 90, 91]], all_idx[101:113]]
        )

        data_indices.model = DotDict()
        data_indices.model.input = DotDict()
        data_indices.model.input.forcing = torch.cat(
            [all_idx[[82, 84, 86, 89]], all_idx[90:99]]
        )

        self.data_indices = data_indices

    def predict_step(self, x, fcstep=1):
        del fcstep
        return torch.ones(x.shape[0], 1, x.shape[2], 100, device=x.device)


EXPECTED_OUTPUT_VARIABLES = len(VARIABLES) - 13
EXPECTED_INPUT_VARIABLES = 88


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
def test_aifsens_call(time, device):
    model = PhooAIFSENSModel()

    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    ).to(torch.float64)

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    ).to(torch.float64)

    invariants = torch.randn(
        4,
        721,
        1440,
        device=device,
    )

    p = AIFSENS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
        invariants=invariants,
    ).to(device)

    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, EXPECTED_OUTPUT_VARIABLES, 721, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize("ensemble", [1])  # Batch size of 2 is too large
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aifsens_iter(ensemble, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    model = PhooAIFSENSModel()

    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    ).to(torch.float64)

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    ).to(torch.float64)

    invariants = torch.randn(
        4,
        721,
        1440,
        device=device,
    )

    p = AIFSENS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
        invariants=invariants,
    ).to(device)

    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size(
            [ensemble, len(time), 1, EXPECTED_OUTPUT_VARIABLES, 721, 1440]
        )
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i), "h")

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
def test_aifsens_exceptions(dc, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    model = PhooAIFSENSModel()

    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    ).to(torch.float64)

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    ).to(torch.float64)

    invariants = torch.randn(
        4,
        721,
        1440,
        device=device,
    )

    p = AIFSENS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
        invariants=invariants,
    ).to(device)

    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.fixture(scope="function")
def model() -> AIFSENS:
    package = AIFSENS.load_default_package()
    p = AIFSENS.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aifsens_package(device, ensemble, model):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = model.to(device)

    assert len(p.input_variables) == EXPECTED_INPUT_VARIABLES
    assert len(p.output_variables) == EXPECTED_OUTPUT_VARIABLES

    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    r = Random(dc)

    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    coords = {"ensemble": np.arange(ensemble, dtype=int)} | coords
    x = x.unsqueeze(0).repeat(ensemble, *([1] * x.ndim))

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size(
        [ensemble, len(time), 1, EXPECTED_OUTPUT_VARIABLES, 721, 1440]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "ensemble", 0)

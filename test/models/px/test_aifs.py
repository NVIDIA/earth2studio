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

        data_indices = DotDict()
        data_indices.data = DotDict()
        data_indices.data.input = DotDict()
        data_indices.data.output = DotDict()
        data_indices.data._name_to_index = {
            "10u": 0,
            "10v": 1,
            "2d": 2,
            "2t": 3,
            "cos_julian_day": 4,
            "cos_latitude": 5,
            "cos_local_time": 6,
            "cos_longitude": 7,
            "cp": 8,
            "insolation": 9,
            "lsm": 10,
            "msl": 11,
            "q_100": 12,
            "q_1000": 13,
            "q_150": 14,
            "q_200": 15,
            "q_250": 16,
            "q_300": 17,
            "q_400": 18,
            "q_50": 19,
            "q_500": 20,
            "q_600": 21,
            "q_700": 22,
            "q_850": 23,
            "q_925": 24,
            "sdor": 25,
            "sin_julian_day": 26,
            "sin_latitude": 27,
            "sin_local_time": 28,
            "sin_longitude": 29,
            "skt": 30,
            "slor": 31,
            "sp": 32,
            "t_100": 33,
            "t_1000": 34,
            "t_150": 35,
            "t_200": 36,
            "t_250": 37,
            "t_300": 38,
            "t_400": 39,
            "t_50": 40,
            "t_500": 41,
            "t_600": 42,
            "t_700": 43,
            "t_850": 44,
            "t_925": 45,
            "tcw": 46,
            "tp": 47,
            "u_100": 48,
            "u_1000": 49,
            "u_150": 50,
            "u_200": 51,
            "u_250": 52,
            "u_300": 53,
            "u_400": 54,
            "u_50": 55,
            "u_500": 56,
            "u_600": 57,
            "u_700": 58,
            "u_850": 59,
            "u_925": 60,
            "v_100": 61,
            "v_1000": 62,
            "v_150": 63,
            "v_200": 64,
            "v_250": 65,
            "v_300": 66,
            "v_400": 67,
            "v_50": 68,
            "v_500": 69,
            "v_600": 70,
            "v_700": 71,
            "v_850": 72,
            "v_925": 73,
            "w_100": 74,
            "w_1000": 75,
            "w_150": 76,
            "w_200": 77,
            "w_250": 78,
            "w_300": 79,
            "w_400": 80,
            "w_50": 81,
            "w_500": 82,
            "w_600": 83,
            "w_700": 84,
            "w_850": 85,
            "w_925": 86,
            "z": 87,
            "z_100": 88,
            "z_1000": 89,
            "z_150": 90,
            "z_200": 91,
            "z_250": 92,
            "z_300": 93,
            "z_400": 94,
            "z_50": 95,
            "z_500": 96,
            "z_600": 97,
            "z_700": 98,
            "z_850": 99,
            "z_925": 100,
            "swvl1": 101,
            "swvl2": 102,
            "stl1": 103,
            "stl2": 104,
            "ssrd": 105,
            "strd": 106,
            "sf": 107,
            "tcc": 108,
            "mcc": 109,
            "hcc": 110,
            "lcc": 111,
            "100u": 112,
            "100v": 113,
            "ro": 114,
        }

        data_indices.data.input.prognostic = torch.IntTensor(
            [
                0,
                1,
                2,
                3,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                30,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
            ]
        )

        data_indices.data.input.diagnostic = torch.IntTensor(
            [8, 47, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        )

        data_indices.data.output.full = torch.IntTensor(
            [
                0,
                1,
                2,
                3,
                8,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                30,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
            ]
        )

        data_indices.data.input.full = torch.IntTensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                87,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
            ]
        )

        data_indices.data.output.prognostic = torch.IntTensor(
            [
                0,
                1,
                2,
                3,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                30,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
                60,
                61,
                62,
                63,
                64,
                65,
                66,
                67,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                75,
                76,
                77,
                78,
                79,
                80,
                81,
                82,
                83,
                84,
                85,
                86,
                88,
                89,
                90,
                91,
                92,
                93,
                94,
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
            ]
        )

        data_indices.data.output.diagnostic = torch.IntTensor(
            [8, 47, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]
        )

        # Model indices
        data_indices.model = DotDict()
        data_indices.model.input = DotDict()

        self.data_indices = data_indices

    def predict_step(self, x, fcstep=1):
        del fcstep
        return torch.ones(x.shape[0], 1, x.shape[2], 102, device=x.device)


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

    invariants = torch.zeros(4, 721, 1440, device=device)

    # Initialize AIFS
    p = AIFS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
        invariants=invariants,
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

    assert out.shape == torch.Size(
        [len(time), 1, out_coords["variable"].shape[0], 721, 1440]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize("ensemble", [1])  # Batch size of 2 is too large
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

    invariants = torch.zeros(4, 721, 1440, device=device)

    p = AIFS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
        invariants=invariants,
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
        assert out.shape == torch.Size(
            [ensemble, len(time), 1, out_coords["variable"].shape[0], 721, 1440]
        )
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

    invariants = torch.zeros(4, 721, 1440, device=device)

    p = AIFS(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
        invariants=invariants,
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
def model() -> AIFS:
    # Test only on cuda device
    package = AIFS.load_default_package()
    p = AIFS.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aifs_package(device, model):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package AIFS
    p = model.to(device)

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

    assert out.shape == torch.Size(
        [len(time), 1, out_coords["variable"].shape[0], 721, 1440]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)

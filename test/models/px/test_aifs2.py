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
from earth2studio.models.px import AIFS2
from earth2studio.utils import handshake_dim


def make_two_nnz_per_first_row_csr(n_rows, n_cols, device):
    """Create a minimal CSR sparse tensor for testing interpolation."""
    crow = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    crow[1:] = 2

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
            value = DotDict(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

class PhooAIFS2Model(torch.nn.Module):
    """Mock AIFS2 model for unit testing."""

    def __init__(self):
        super().__init__()

        # Mock data indices following AIFS2 checkpoint structure
        data_indices = DotDict()
        data_indices.data = DotDict()
        data_indices.data.input = DotDict()
        data_indices.data.output = DotDict()

        # AIFS2 uses checkpoint variable names (e.g., "10u" instead of "u10m")
        # This is the full variable mapping sorted alphabetically by checkpoint name
        # Variable order MUST match the real checkpoint exactly
        data_indices.data._name_to_index = {
            "100u": 0,
            "100v": 1,
            "10u": 2,
            "10v": 3,
            "2d": 4,
            "2t": 5,
            "cdww": 6,
            "cos_julian_day": 7,
            "cos_latitude": 8,
            "cos_local_time": 9,
            "cos_longitude": 10,
            "cos_mwd": 11,
            "cp": 12,
            "h1012": 13,
            "h1214": 14,
            "h1417": 15,
            "h1721": 16,
            "h2125": 17,
            "h2530": 18,
            "hcc": 19,
            "insolation": 20,
            "lcc": 21,
            "lsm": 22,
            "mcc": 23,
            "msl": 24,
            "mwp": 25,
            "q_100": 26,
            "q_1000": 27,
            "q_150": 28,
            "q_200": 29,
            "q_250": 30,
            "q_300": 31,
            "q_400": 32,
            "q_50": 33,
            "q_500": 34,
            "q_600": 35,
            "q_700": 36,
            "q_850": 37,
            "q_925": 38,
            "ro": 39,
            "sd": 40,
            "sdor": 41,
            "sf": 42,
            "sin_julian_day": 43,
            "sin_latitude": 44,
            "sin_local_time": 45,
            "sin_longitude": 46,
            "sin_mwd": 47,
            "skt": 48,
            "slor": 49,
            "snowc": 50,
            "sp": 51,
            "ssrd": 52,
            "stl1": 53,
            "stl2": 54,
            "strd": 55,
            "swh": 56,
            "swvl1": 57,
            "swvl2": 58,
            "t_10": 59,
            "t_100": 60,
            "t_1000": 61,
            "t_150": 62,
            "t_200": 63,
            "t_250": 64,
            "t_300": 65,
            "t_400": 66,
            "t_50": 67,
            "t_500": 68,
            "t_600": 69,
            "t_700": 70,
            "t_850": 71,
            "t_925": 72,
            "tcc": 73,
            "tcw": 74,
            "tp": 75,
            "u_10": 76,
            "u_100": 77,
            "u_1000": 78,
            "u_150": 79,
            "u_200": 80,
            "u_250": 81,
            "u_300": 82,
            "u_400": 83,
            "u_50": 84,
            "u_500": 85,
            "u_600": 86,
            "u_700": 87,
            "u_850": 88,
            "u_925": 89,
            "v_10": 90,
            "v_100": 91,
            "v_1000": 92,
            "v_150": 93,
            "v_200": 94,
            "v_250": 95,
            "v_300": 96,
            "v_400": 97,
            "v_50": 98,
            "v_500": 99,
            "v_600": 100,
            "v_700": 101,
            "v_850": 102,
            "v_925": 103,
            "w_10": 104,
            "w_100": 105,
            "w_1000": 106,
            "w_150": 107,
            "w_200": 108,
            "w_250": 109,
            "w_300": 110,
            "w_400": 111,
            "w_50": 112,
            "w_500": 113,
            "w_600": 114,
            "w_700": 115,
            "w_850": 116,
            "w_925": 117,
            "wmb": 118,
            "z": 119,
            "z_10": 120,
            "z_100": 121,
            "z_1000": 122,
            "z_150": 123,
            "z_200": 124,
            "z_250": 125,
            "z_300": 126,
            "z_400": 127,
            "z_50": 128,
            "z_500": 129,
            "z_600": 130,
            "z_700": 131,
            "z_850": 132,
            "z_925": 133,
        }

        # Input prognostic variables (not forcings, invariants, or diagnostics)
        # These indices match the real checkpoint exactly
        input_prognostic_indices = [
            2,
            3,
            4,
            5,
            6,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            40,
            47,
            48,
            51,
            53,
            54,
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
            74,
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
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
        ]
        data_indices.data.input.prognostic = torch.IntTensor(input_prognostic_indices)

        # Input diagnostic variables (accumulations + diagnostics from previous step)
        input_diagnostic_indices = [
            0,
            1,
            12,
            19,
            21,
            23,
            33,
            39,
            42,
            50,
            52,
            55,
            73,
            75,
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
            115,
            116,
            117,
        ]
        data_indices.data.input.diagnostic = torch.IntTensor(input_diagnostic_indices)

        # Input full = prognostic + forcings + invariants (but not diagnostics)
        input_full_indices = [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            13,
            14,
            15,
            16,
            17,
            18,
            20,
            22,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            34,
            35,
            36,
            37,
            38,
            40,
            41,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            51,
            53,
            54,
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
            74,
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
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
        ]
        data_indices.data.input.full = torch.IntTensor(input_full_indices)

        # Output prognostic = same as input prognostic
        data_indices.data.output.prognostic = torch.IntTensor(input_prognostic_indices)

        # Output diagnostic = same as input diagnostic
        data_indices.data.output.diagnostic = torch.IntTensor(input_diagnostic_indices)

        # Output full = prognostic + diagnostic
        output_full_indices = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            21,
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
            42,
            47,
            48,
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
            115,
            116,
            117,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
        ]
        data_indices.data.output.full = torch.IntTensor(output_full_indices)

        # Model indices (not used in wrapper but needed for structure)
        data_indices.model = DotDict()
        data_indices.model.input = DotDict()

        self.data_indices = data_indices

    def predict_step(self, x, fcstep=1):
        """Mock prediction step - returns output with correct shape."""
        del fcstep
        # Output shape: (batch, 1, n_points, n_output_vars)
        # n_output_vars = len(output_full)
        n_output_vars = len(self.data_indices.data.output.full)
        return torch.ones(x.shape[0], 1, x.shape[2], n_output_vars, device=x.device)

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
def test_aifs2_call(time, device):
    """Test AIFS2 single step forward pass."""
    # Spoof model
    model = PhooAIFS2Model()

    # Create tensors with the correct shapes for O96 octahedral grid
    # O96 has 542,080 points
    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    # Create sparse tensors for interpolation matrices
    # ERA5 0.25deg: 721*1440 = 1,038,240 points
    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    )

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    )

    # AIFS2 has 5 invariants: lsm, sdor, slor, z, wmb
    invariants = torch.zeros(5, 721, 1440, device=device)

    # Initialize AIFS2
    p = AIFS2(
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

@pytest.mark.parametrize("ensemble", [1])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aifs2_iter(ensemble, device):
    """Test AIFS2 iterator produces correct sequence."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    # Spoof model
    model = PhooAIFS2Model()

    # Create tensors
    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    )

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    )

    invariants = torch.zeros(5, 721, 1440, device=device)

    p = AIFS2(
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

    # Get Data
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
def test_aifs2_exceptions(dc, device):
    """Test AIFS2 raises on invalid coordinates."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    model = PhooAIFS2Model()

    latitudes = torch.randn(1, 1, 542080, 1, device=device)
    longitudes = torch.randn(1, 1, 542080, 1, device=device)

    interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=542_080, n_cols=1_038_240, device=device
    )

    inverse_interpolation_matrix = make_two_nnz_per_first_row_csr(
        n_rows=1_038_240, n_cols=542_080, device=device
    )

    invariants = torch.zeros(5, 721, 1440, device=device)

    p = AIFS2(
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
        invariants=invariants,
    ).to(device)

    # Initialize Data Source
    r = Random(dc)

    # Get Data
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)

@pytest.fixture(scope="function")
def model() -> AIFS2:
    """Load real AIFS2 model from package, mocking IFS fetch if needed."""
    from unittest.mock import patch

    # Mock fetch_data to return fake invariants if IFS would be called
    def mock_fetch_data(source, time, variable, *args, **kwargs):
        # Return fake invariants tensor (5 variables: lsm, sdor, slor, z, wmb)
        fake_invariants = torch.zeros(1, 1, 1, len(variable), 721, 1440)
        fake_coords = {"time": time, "variable": np.array(variable)}
        return fake_invariants, fake_coords

    package = AIFS2.load_default_package()
    with patch("earth2studio.models.px.aifs2.fetch_data", side_effect=mock_fetch_data):
        p = AIFS2.load_model(package)
    return p

@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aifs2_package(device, model):
    """Integration test with real AIFS2 model weights."""
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])

    # Test the cached model package AIFS2
    p = model.to(device)

    # Create "domain coords"
    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data
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

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
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import AIFSENS
from earth2studio.models.px.aifsens import VARIABLES
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


@pytest.fixture
def backend():
    from importlib.metadata import version

    pytest.importorskip("anemoi.models")

    anemoi_version = version("anemoi-models")
    # AIFSENS requires anemoi-models version specified by pyproject.toml.
    if anemoi_version != "0.5.1":
        pytest.skip(
            (
                f"anemoi-models {anemoi_version} not compatible with AIFSENS "
                "(requires 0.5.1)"
            ),
        )


@pytest.fixture(autouse=True)
def optional_device(request):
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")


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
        np.array([np.datetime64("1993-04-05T00:00")]),  # Multiple times not supported
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aifsens_call(time, device, backend):
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
    x = fetch_data(
        r, time, variable, lead_time, device=device, delta_t=np.timedelta64(1, "h")
    )
    x.attrs.update(earth2studio_grid_id="latlon-0.25deg", earth2studio_crs="EPSG:4326")
    coords = x
    out = p(x)
    out_coords = out.coords

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, EXPECTED_OUTPUT_VARIABLES, 721, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    assert out.dims == ("time", "lead_time", "variable", "lat", "lon")


@pytest.mark.parametrize("ensemble", [1])  # Batch size of 2 is too large
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aifsens_iter(ensemble, device, backend):
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
    x = fetch_data(
        r, time, variable, lead_time, device=device, delta_t=np.timedelta64(1, "h")
    )
    x.attrs.update(earth2studio_grid_id="latlon-0.25deg", earth2studio_crs="EPSG:4326")
    x = x.expand_dims(ensemble=np.arange(ensemble))
    p_iter = p.create_iterator(x)

    if not isinstance(time, Iterable):
        time = [time]

    initial = next(p_iter)
    assert initial.equals(x.isel(lead_time=slice(-1, None)))
    for i, out in enumerate(p_iter):
        out_coords = out.coords
        assert len(out.shape) == 6
        assert out.shape == torch.Size(
            [ensemble, len(time), 1, EXPECTED_OUTPUT_VARIABLES, 721, 1440]
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
def test_aifsens_exceptions(dc, device, backend):
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
    x = fetch_data(
        r, time, variable, lead_time, device=device, delta_t=np.timedelta64(1, "h")
    )

    with pytest.raises((KeyError, ValueError)):
        p(x)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aifsens_conformance(monkeypatch, device):
    assert "_fill_input" in AIFSENS.__dict__
    model = PhooAIFSENSModel()

    latitudes = torch.tensor([45, 45, -45, -45]).reshape(1, 1, 4, 1).float()
    longitudes = torch.tensor([0, 180, 0, 180]).reshape(1, 1, 4, 1).float()
    interpolation_matrix = torch.eye(4, dtype=torch.float64).to_sparse_csr()
    inverse_interpolation_matrix = interpolation_matrix
    invariants = torch.zeros(4, 2, 2)

    p = AIFSENS.__new__(AIFSENS)
    inspect.unwrap(AIFSENS.__init__)(
        p,
        model=model,
        latitudes=latitudes,
        longitudes=longitudes,
        interpolation_matrix=interpolation_matrix,
        inverse_interpolation_matrix=inverse_interpolation_matrix,
        invariants=invariants,
    )
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
        grid=LatLonGrid([45, -45], [0, 180]),
    )
    monkeypatch.setattr(p, "input_coords", lambda: signature.copy())

    assert check_prognostic_contract(p) == [
        "P14: model does not declare itself stochastic"
    ]
    coords = coord_array_like(
        signature,
        {"batch": [0, 1], "time": np.array(["2000-01-01"], dtype="datetime64[ns]")},
    )
    x = from_torch(torch.randn(coords.shape), coords, name="weather").rename(
        batch="member"
    )
    original = x.copy(deep=True)
    torch.testing.assert_close(
        p(x).e2s.to_torch()[0], torch.ones(p.output_coords(x).shape, device=device)
    )
    iterator = p.create_iterator(x)
    initial = next(iterator)
    retained = []
    for step in range(1, 4):
        out = next(iterator)
        retained.append((out, out.copy(deep=True)))
        torch.testing.assert_close(
            out.e2s.to_torch()[0], torch.ones(out.shape, device=device)
        )
        assert out.lead_time.values[0] == np.timedelta64(step * 6, "h")
        assert out.name == x.name and out.dims == x.dims
    for out, saved in retained:
        xr.testing.assert_identical(out, saved)
    xr.testing.assert_identical(x, original)
    xr.testing.assert_identical(initial, original.isel(lead_time=slice(-1, None)))

    iterator.close()
    # Non-identity interpolation exposes an accidental public-grid round trip.
    eye = torch.eye(4, dtype=torch.float64, device=device)
    p.interpolation_matrix = ((eye + eye.roll(1, dims=1)) / 2).to_sparse_csr()
    data = p.model.data_indices.data
    indices = [
        data.input.full.tolist().index(i) if i in data.input.full else 0
        for i in data.output.full.tolist()
    ]
    monkeypatch.setattr(
        p.model, "predict_step", lambda value, fcstep: value[:, -1:, :, indices] + 1
    )
    native_coords = {d: coords.coords[d].values for d in coords.dims}
    state = p._prepare_input(x.e2s.to_torch()[0].to(device), native_coords)
    expected = []
    for step in range(1, 4):
        state, output_coords = p._forward(
            state,
            coord_array_like(coords, {"lead_time": native_coords["lead_time"]}),
            step,
        )
        expected.append(
            p._prepare_output(
                state, {d: output_coords.coords[d].values for d in output_coords.dims}
            ).clone()
        )
        native_coords["lead_time"] = native_coords["lead_time"] + np.timedelta64(6, "h")
        state = p._update_input(state, native_coords)
    preparations = []
    prepare = p._prepare_input

    def capture(value, coords):
        preparations.append(1)
        return prepare(value, coords)

    monkeypatch.setattr(p, "_prepare_input", capture)
    p.front_hook = lambda value: value.assign_attrs(source="front")
    p.rear_hook = lambda value: value.rename(None)
    iterator = p.create_iterator(x)
    next(iterator)
    for reference in expected:
        out = next(iterator)
        torch.testing.assert_close(out.e2s.to_torch()[0], reference)
        assert out.name is None and out.attrs["source"] == "front"
    assert len(preparations) == 1
    iterator.close()


@pytest.fixture(scope="function")
def model(backend) -> AIFSENS:
    """Load real AIFSENS model from package, mocking IFS fetch if needed."""
    from unittest.mock import patch

    # Mock fetch_data to return fake invariants if IFS would be called
    def mock_fetch_data(source, time, variable, *args, **kwargs):
        # Return fake invariants tensor (4 variables: lsm, sdor, slor, z)
        return xr.DataArray(
            np.zeros((len(variable), 721, 1440), dtype=np.float32),
            dims=("variable", "lat", "lon"),
            coords={"variable": variable},
        )

    package = AIFSENS.load_default_package()
    with patch(
        "earth2studio.models.px.aifsens.fetch_data", side_effect=mock_fetch_data
    ):
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
    x = fetch_data(
        r, time, variable, lead_time, device=device, delta_t=np.timedelta64(1, "h")
    )
    x.attrs.update(earth2studio_grid_id="latlon-0.25deg", earth2studio_crs="EPSG:4326")
    x = x.expand_dims(ensemble=np.arange(ensemble))
    coords = x
    out = p(x)
    out_coords = out.coords

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size(
        [ensemble, len(time), 1, EXPECTED_OUTPUT_VARIABLES, 721, 1440]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert out.dims == ("ensemble", "time", "lead_time", "variable", "lat", "lon")

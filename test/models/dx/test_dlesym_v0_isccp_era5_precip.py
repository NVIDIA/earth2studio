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


import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.dx.dlesym_v0_isccp_era5_precip as precip_src
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import DLESyMv0_ISCCP_ERA5Precip
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import OptionalDependencyFailure


@pytest.fixture(autouse=True)
def optional_backend(monkeypatch):
    if precip_src.insolation is None:
        monkeypatch.delitem(
            OptionalDependencyFailure.failures, precip_src.__file__, raising=False
        )
        monkeypatch.setattr(
            precip_src,
            "insolation",
            lambda times, lat, lon: np.zeros(
                (len(times), *lat.shape), dtype=np.float32
            ),
        )


_PRECIP_VARIABLES = [
    "z500",
    "tau300-700",
    "z1000",
    "t2m",
    "tcwv",
    "t850",
    "z250",
    "rlut",
    "ws10m",
    "sst",
]
_PRECIP_INPUT_TIMES = np.array([-6, 0], dtype="timedelta64[h]")


class PhooPrecipModel(torch.nn.Module):
    """Mock HEALPixUNet diagnostic; emits ones with output_time_dim=1."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = 1
        self.input_time_dim = 2

    def forward(self, in_list):
        state = in_list[0]  # (B, F=12, T, V, H, W)
        b, f = state.shape[:2]
        _, _, _, _, h, w = state.shape
        return torch.zeros(b, f, self.output_time_dim, 1, h, w, device=state.device)


def _build_climatology(nside: int, n_doy: int = 366) -> dict:
    rng = np.random.default_rng(0)
    return {
        "ttr_clim_mean": rng.standard_normal((n_doy, 12, nside, nside)).astype(
            "float32"
        ),
        "ttr_clim_std": np.ones((n_doy, 12, nside, nside), dtype="float32"),
        "olr_clim_mean": rng.standard_normal((n_doy, 12, nside, nside)).astype(
            "float32"
        ),
        "olr_clim_std": np.ones((n_doy, 12, nside, nside), dtype="float32"),
    }


def _build_model(
    device, nside: int = 16, use_ttr: bool = True
) -> DLESyMv0_ISCCP_ERA5Precip:
    n_vars = len(_PRECIP_VARIABLES)
    hpx_lat = np.random.randn(12, nside, nside)
    hpx_lon = np.random.randn(12, nside, nside)
    center = np.zeros((1, 1, 1, n_vars, 1, 1, 1))
    scale = np.ones((1, 1, 1, n_vars, 1, 1, 1))
    constants = np.random.randn(12, 2, nside, nside)
    clim = _build_climatology(nside) if use_ttr else {}

    return DLESyMv0_ISCCP_ERA5Precip(
        core_model=PhooPrecipModel(),
        hpx_lat=hpx_lat,
        hpx_lon=hpx_lon,
        nside=nside,
        center=center,
        scale=scale,
        constants=constants,
        input_times=_PRECIP_INPUT_TIMES,
        variables=_PRECIP_VARIABLES,
        output_variable="tp06",
        log_epsilon=1e-8,
        use_ttr=use_ttr,
        **clim,
    ).to(device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("use_ttr", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_v0_isccp_era5_precip_forward(device, use_ttr, batch_size):
    """Forward pass shape + output coords are correct (both TTR modes)."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 16
    model = _build_model(device, nside=nside, use_ttr=use_ttr)
    in_coords = model.input_coords()
    assert isinstance(in_coords, xr.DataArray)
    assert in_coords.data.nbytes == 0
    # Input variable should be ``ttr`` when use_ttr=True, else ``rlut``.
    expected_input_var = "ttr" if use_ttr else "rlut"
    assert expected_input_var in list(in_coords["variable"])
    in_coords = coord_array_like(
        in_coords,
        {
            "batch": np.arange(batch_size),
            "time": np.array([np.datetime64("2020-01-01T00:00")]),
        },
    )

    x = torch.randn(
        batch_size,
        len(in_coords["time"]),
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        12,
        nside,
        nside,
        device=device,
    )

    field = from_torch(x, in_coords, name="climate")
    field.encoding = {"test": "precip"}
    field = field.assign_coords(
        terrain=(("face", "height", "width"), np.zeros((12, nside, nside))),
        valid_time=(
            ("time", "lead_time"),
            field.time.values[:, None] + field.lead_time.values[None, :],
        ),
    )
    field = field.rename(batch="member").drop_vars("member")
    for key, value in {
        "origin": "south",
        "clockwise": False,
        "ordering": "nested",
        "layout": "flat",
        "level": 0,
        "nside": 1,
        "type": "LatLonGrid",
        "topology": "rectilinear",
        "dims": ["hpx"],
        "shape": [12 * nside**2],
    }.items():
        with pytest.raises(ValueError, match="metadata"):
            model.output_coords(field.assign_attrs({key: value}))
    bad = field.copy(deep=False)
    bad.attrs.pop("clockwise")
    with pytest.raises(ValueError, match="metadata"):
        model(bad)
    out = model(field)
    out_coords = out.coords
    expected = model.output_coords(in_coords)

    assert out.shape == (batch_size, 1, 1, 1, 12, nside, nside)
    assert list(out_coords["variable"].values) == ["tp06"]
    assert out.name == field.name and out.encoding == field.encoding
    assert out.dims[0] == "member" and "member" not in out.coords
    assert "terrain" in out.coords and "valid_time" not in out.coords
    assert out.attrs["ordering"] == "xy"
    assert out.attrs["origin"] == "north"
    assert out.attrs["clockwise"] is True
    assert len(out_coords["lead_time"]) == 1
    assert out_coords["lead_time"][0] == in_coords["lead_time"][-1]
    for key in out_coords:
        if key in expected.coords:
            np.testing.assert_array_equal(out_coords[key], expected[key])


@pytest.mark.parametrize("device", ["cpu"])
def test_dlesym_v0_isccp_era5_precip_log_epsilon_inverse(device):
    """log_epsilon path: model emits zeros, so output should be exactly 0.

    The denormalization is ``exp(out + log(eps)) - eps``. For ``out=0`` that
    evaluates to ``eps - eps == 0``, so a fully-zero model output should yield
    an all-zero precipitation prediction (the floor of the log-transform).
    """
    nside = 16
    model = _build_model(device, nside=nside)
    in_coords = model.input_coords()
    in_coords = coord_array_like(
        in_coords,
        {"batch": np.array([0]), "time": np.array([np.datetime64("2020-06-15T12:00")])},
    )

    x = torch.randn(
        1,
        1,
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        12,
        nside,
        nside,
        device=device,
    )
    out = model(from_torch(x, in_coords)).e2s.to_torch()[0]
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_dlesym_v0_isccp_era5_precip_conformance():
    """Check the mock model against the Earth2Studio model contract.

    The model does not declare itself stochastic, so D10 (RNG isolation) is
    structurally inapplicable and reported as a skip rather than passed.
    """
    model = _build_model("cpu")
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_dlesym_v0_isccp_era5_precip_package(device):
    """Load the real HuggingFace package and run one precip diagnostic forward pass."""
    torch.cuda.empty_cache()
    package = DLESyMv0_ISCCP_ERA5Precip.load_default_package()
    model = DLESyMv0_ISCCP_ERA5Precip.load_model(package).to(device)

    nside = model.nside
    batch_size = 1
    time = np.array([np.datetime64("2020-01-01T00:00")])
    in_coords = model.input_coords()
    in_coords = coord_array_like(
        in_coords, {"batch": np.arange(batch_size), "time": time}
    )

    x = torch.randn(
        batch_size,
        len(time),
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        12,
        nside,
        nside,
        device=device,
    )

    out = model(from_torch(x, in_coords))
    out_coords = out.coords
    expected_coords = model.output_coords(in_coords)

    assert out.shape == (batch_size, len(time), 1, 1, 12, nside, nside)
    assert list(out_coords["variable"].values) == ["tp06"]
    assert len(out_coords["lead_time"]) == 1
    assert out_coords["lead_time"][0] == in_coords["lead_time"][-1]
    for key in out_coords:
        np.testing.assert_array_equal(out_coords[key], expected_coords[key])

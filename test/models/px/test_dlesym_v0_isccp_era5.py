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

from earth2studio.models.px import DLESyMv0_ISCCP_ERA5, DLESyMv0_ISCCP_ERA5LatLon
from earth2studio.utils import handshake_coords

# Upstream-DLESyM variable layout (model space). The wrapper swaps ``rlut`` ->
# ``ttr`` in input_coords when use_ttr=True.
_ATMOS_VARIABLES = [
    "z500",
    "tau300-700",
    "z1000",
    "t2m",
    "tcwv",
    "t850",
    "z250",
    "rlut",
    "ws10m",
]
_OCEAN_VARIABLES = ["sst"]
_ATMOS_COUPLING_VARIABLES = ["sst"]
_OCEAN_COUPLING_VARIABLES = ["z1000", "ws10m", "rlut"]

# The atmos model is a HEALPixRecUNet with presteps=1 and input_time_dim=2, so
# it consumes (presteps + 1) * input_time_dim = 4 history timesteps (the first
# two initialize the recurrent hidden state). The ocean model is a plain
# HEALPixUNet (presteps=0) and consumes only its input_time_dim window.
_ATMOS_INPUT_TIMES = np.array([-18, -12, -6, 0], dtype="timedelta64[h]")
_OCEAN_INPUT_TIMES = np.array([-48, 0], dtype="timedelta64[h]")

_ATMOS_OUTPUT_TIMES = np.array(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],
    dtype="timedelta64[h]",
)
_OCEAN_OUTPUT_TIMES = np.array([48, 96], dtype="timedelta64[h]")


class PhooAtmosModel(torch.nn.Module):
    """Mock atmosphere model. Output ones; shape mirrors HEALPixRecUNet."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = len(_ATMOS_OUTPUT_TIMES)
        self.input_time_dim = len(_ATMOS_INPUT_TIMES)

    def forward(self, in_list):
        x = in_list[0]
        b, t = x.shape[:2]
        return torch.ones(b, t, self.output_time_dim, *x.shape[3:], device=x.device)


class PhooOceanModel(torch.nn.Module):
    """Mock ocean model."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = len(_OCEAN_OUTPUT_TIMES)
        self.input_time_dim = len(_OCEAN_INPUT_TIMES)

    def forward(self, in_list):
        x = in_list[0]
        b, t = x.shape[:2]
        return torch.ones(b, t, self.output_time_dim, *x.shape[3:], device=x.device)


def _build_climatology(nside: int, n_doy: int = 366) -> dict:
    """Return synthetic per-doy climatology arrays of the right shape."""
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


# Lat/lon input variables: the derived ``tau300-700`` / ``ws10m`` are replaced by
# their base inputs, and ``rlut`` is advertised as ``ttr`` when use_ttr=True.
_LATLON_BASE_VARIABLES = ["u10m", "v10m", "z300", "z700"]


def _build_latlon_model(
    device, nside: int = 16, use_ttr: bool = True
) -> DLESyMv0_ISCCP_ERA5LatLon:
    """Build DLESyMv0_ISCCP_ERA5LatLon with mock components.

    Uses a small per-doy climatology so the test stays light even though the
    lat/lon regridders target the fixed 721x1440 grid.
    """
    n_vars = len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES)
    hpx_lat = np.random.randn(12, nside, nside)
    hpx_lon = np.random.randn(12, nside, nside)
    center = np.zeros((1, 1, 1, n_vars, 1, 1, 1))
    scale = np.ones((1, 1, 1, n_vars, 1, 1, 1))
    atmos_constants = np.random.randn(12, 2, nside, nside)
    ocean_constants = np.random.randn(12, 2, nside, nside)

    clim = _build_climatology(nside, n_doy=8) if use_ttr else {}

    return DLESyMv0_ISCCP_ERA5LatLon(
        atmos_model=PhooAtmosModel(),
        ocean_model=PhooOceanModel(),
        hpx_lat=hpx_lat,
        hpx_lon=hpx_lon,
        nside=nside,
        center=center,
        scale=scale,
        atmos_constants=atmos_constants,
        ocean_constants=ocean_constants,
        atmos_input_times=_ATMOS_INPUT_TIMES,
        ocean_input_times=_OCEAN_INPUT_TIMES,
        atmos_output_times=_ATMOS_OUTPUT_TIMES,
        ocean_output_times=_OCEAN_OUTPUT_TIMES,
        atmos_variables=_ATMOS_VARIABLES,
        ocean_variables=_OCEAN_VARIABLES,
        atmos_coupling_variables=_ATMOS_COUPLING_VARIABLES,
        ocean_coupling_variables=_OCEAN_COUPLING_VARIABLES,
        use_ttr=use_ttr,
        **clim,
    ).to(device)


def _build_model(device, nside: int = 16, use_ttr: bool = True) -> DLESyMv0_ISCCP_ERA5:
    """Build DLESyMv0_ISCCP_ERA5 with mock components. Uses nside=16 for fast tests."""
    n_vars = len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES)
    hpx_lat = np.random.randn(12, nside, nside)
    hpx_lon = np.random.randn(12, nside, nside)
    center = np.zeros((1, 1, 1, n_vars, 1, 1, 1))
    scale = np.ones((1, 1, 1, n_vars, 1, 1, 1))
    atmos_constants = np.random.randn(12, 2, nside, nside)
    ocean_constants = np.random.randn(12, 2, nside, nside)

    clim = _build_climatology(nside) if use_ttr else {}

    model = DLESyMv0_ISCCP_ERA5(
        atmos_model=PhooAtmosModel(),
        ocean_model=PhooOceanModel(),
        hpx_lat=hpx_lat,
        hpx_lon=hpx_lon,
        nside=nside,
        center=center,
        scale=scale,
        atmos_constants=atmos_constants,
        ocean_constants=ocean_constants,
        atmos_input_times=_ATMOS_INPUT_TIMES,
        ocean_input_times=_OCEAN_INPUT_TIMES,
        atmos_output_times=_ATMOS_OUTPUT_TIMES,
        ocean_output_times=_OCEAN_OUTPUT_TIMES,
        atmos_variables=_ATMOS_VARIABLES,
        ocean_variables=_OCEAN_VARIABLES,
        atmos_coupling_variables=_ATMOS_COUPLING_VARIABLES,
        ocean_coupling_variables=_OCEAN_COUPLING_VARIABLES,
        use_ttr=use_ttr,
        **clim,
    ).to(device)
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("use_ttr", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_v0_isccp_era5_forward(device, use_ttr, batch_size):
    """Forward pass with mock atmos/ocean models, both with and without TTR."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 16
    model = _build_model(device, nside=nside, use_ttr=use_ttr)

    # When use_ttr=True the wrapper advertises ``ttr`` in input_coords.
    in_coords = model.input_coords()
    expected_input_var = "ttr" if use_ttr else "rlut"
    assert expected_input_var in list(in_coords["variable"])

    time = np.array([np.datetime64("2020-01-01T00:00")])
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

    in_coords["batch"] = np.arange(batch_size)
    in_coords["time"] = time

    out, out_coords = model(x, in_coords)
    expected_coords = model.output_coords(in_coords)

    # Output should always be in model variable space (rlut), regardless of
    # whether the input was provided as ttr.
    assert "rlut" in list(out_coords["variable"])
    assert "ttr" not in list(out_coords["variable"])
    assert out.shape == (
        batch_size,
        len(time),
        len(_ATMOS_OUTPUT_TIMES),
        len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES),
        12,
        nside,
        nside,
    )
    for key in out_coords:
        handshake_coords(out_coords, expected_coords, key)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_dlesym_v0_isccp_era5_ttr_transform_changes_values(device):
    """The TTR->OLR transform must actually run when use_ttr=True."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 16
    model = _build_model(device, nside=nside, use_ttr=True)

    in_coords = model.input_coords()
    in_coords["batch"] = np.array([0])
    in_coords["time"] = np.array([np.datetime64("2020-07-15T00:00")])

    x_input = torch.zeros(
        1,
        1,
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        12,
        nside,
        nside,
        device=device,
    )
    x_transformed = model._apply_ttr_to_olr(x_input, in_coords)

    # Find the ttr channel and confirm the transform pulled values away from zero.
    ttr_idx = list(in_coords["variable"]).index("ttr")
    transformed_channel = x_transformed[:, :, :, ttr_idx]
    other_channels = torch.cat(
        [
            x_transformed[:, :, :, :ttr_idx],
            x_transformed[:, :, :, ttr_idx + 1 :],
        ],
        dim=3,
    )
    assert not torch.allclose(
        transformed_channel, torch.zeros_like(transformed_channel)
    )
    assert torch.allclose(other_channels, torch.zeros_like(other_channels))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_v0_isccp_era5_iterator(device, batch_size):
    """Iterator runs through several coupled rollout steps without errors."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 16
    model = _build_model(device, nside=nside, use_ttr=True)
    in_coords = model.input_coords()

    time = np.array([np.datetime64("2020-01-01T00:00")])
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
    in_coords["batch"] = np.arange(batch_size)
    in_coords["time"] = time

    iterator = model.create_iterator(x, in_coords)

    coupler_step = _ATMOS_OUTPUT_TIMES[-1]
    initial_x, initial_coords = next(iterator)
    # Initial yield is in model variable space (post-transform).
    assert "rlut" in list(initial_coords["variable"])

    for i in range(2):
        out, coords = next(iterator)
        assert out.shape == (
            batch_size,
            len(time),
            len(_ATMOS_OUTPUT_TIMES),
            len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES),
            12,
            nside,
            nside,
        )
        assert np.all(coords["lead_time"] == _ATMOS_OUTPUT_TIMES + coupler_step * i)


def test_dlesym_v0_isccp_era5_missing_clim_raises():
    """Direct construction without climatology should raise."""
    nside = 16
    n_vars = len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES)
    hpx_lat = np.random.randn(12, nside, nside)
    hpx_lon = np.random.randn(12, nside, nside)
    center = np.zeros((1, 1, 1, n_vars, 1, 1, 1))
    scale = np.ones((1, 1, 1, n_vars, 1, 1, 1))
    atmos_constants = np.random.randn(12, 2, nside, nside)
    ocean_constants = np.random.randn(12, 2, nside, nside)

    with pytest.raises(ValueError, match="climatology"):
        DLESyMv0_ISCCP_ERA5(
            atmos_model=PhooAtmosModel(),
            ocean_model=PhooOceanModel(),
            hpx_lat=hpx_lat,
            hpx_lon=hpx_lon,
            nside=nside,
            center=center,
            scale=scale,
            atmos_constants=atmos_constants,
            ocean_constants=ocean_constants,
            atmos_input_times=_ATMOS_INPUT_TIMES,
            ocean_input_times=_OCEAN_INPUT_TIMES,
            atmos_output_times=_ATMOS_OUTPUT_TIMES,
            ocean_output_times=_OCEAN_OUTPUT_TIMES,
            atmos_variables=_ATMOS_VARIABLES,
            ocean_variables=_OCEAN_VARIABLES,
            atmos_coupling_variables=_ATMOS_COUPLING_VARIABLES,
            ocean_coupling_variables=_OCEAN_COUPLING_VARIABLES,
            use_ttr=True,
            # climatology omitted
        )


def test_dlesym_v0_isccp_era5_missing_rlut_raises():
    """Atmos variables must contain rlut (model space name)."""
    nside = 16
    bad_atmos = list(_ATMOS_VARIABLES)
    bad_atmos[bad_atmos.index("rlut")] = "ttr"  # wrong; should be model space rlut
    n_vars = len(bad_atmos) + len(_OCEAN_VARIABLES)

    with pytest.raises(ValueError, match="rlut"):
        DLESyMv0_ISCCP_ERA5(
            atmos_model=PhooAtmosModel(),
            ocean_model=PhooOceanModel(),
            hpx_lat=np.random.randn(12, nside, nside),
            hpx_lon=np.random.randn(12, nside, nside),
            nside=nside,
            center=np.zeros((1, 1, 1, n_vars, 1, 1, 1)),
            scale=np.ones((1, 1, 1, n_vars, 1, 1, 1)),
            atmos_constants=np.random.randn(12, 2, nside, nside),
            ocean_constants=np.random.randn(12, 2, nside, nside),
            atmos_input_times=_ATMOS_INPUT_TIMES,
            ocean_input_times=_OCEAN_INPUT_TIMES,
            atmos_output_times=_ATMOS_OUTPUT_TIMES,
            ocean_output_times=_OCEAN_OUTPUT_TIMES,
            atmos_variables=bad_atmos,
            ocean_variables=_OCEAN_VARIABLES,
            atmos_coupling_variables=_ATMOS_COUPLING_VARIABLES,
            ocean_coupling_variables=_OCEAN_COUPLING_VARIABLES,
            use_ttr=False,
        )


@pytest.mark.parametrize("use_ttr", [True, False])
def test_dlesym_v0_isccp_era5_latlon_input_coords(use_ttr):
    """LatLon variant advertises lat/lon dims and base (non-derived) variables."""
    model = _build_latlon_model("cpu", nside=16, use_ttr=use_ttr)
    in_coords = model.input_coords()

    # Lat/lon dims present, HEALPix dims absent.
    for dim in ["lat", "lon"]:
        assert dim in in_coords
    for dim in ["face", "height", "width"]:
        assert dim not in in_coords

    variables = list(in_coords["variable"])
    # Derived variables are replaced by their base inputs.
    assert "tau300-700" not in variables
    assert "ws10m" not in variables
    for v in _LATLON_BASE_VARIABLES:
        assert v in variables
    # Radiation channel is advertised as ttr / rlut depending on use_ttr.
    if use_ttr:
        assert "ttr" in variables
        assert "rlut" not in variables
    else:
        assert "rlut" in variables
        assert "ttr" not in variables


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("use_ttr", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_v0_isccp_era5_latlon_forward(device, use_ttr, batch_size):
    """LatLon forward: lat/lon in, lat/lon out, in model variable space."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 64  # lat/lon regridders target the fixed 721x1440 grid
    model = _build_latlon_model(device, nside=nside, use_ttr=use_ttr)

    in_coords = model.input_coords()
    expected_input_var = "ttr" if use_ttr else "rlut"
    assert expected_input_var in list(in_coords["variable"])

    time = np.array([np.datetime64("2020-07-15T00:00")])
    x = torch.randn(
        batch_size,
        len(time),
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        721,
        1440,
        device=device,
    )
    in_coords["batch"] = np.arange(batch_size)
    in_coords["time"] = time

    out, out_coords = model(x, in_coords)
    expected_coords = model.output_coords(in_coords)

    # Output is on the lat/lon grid and in model variable space (rlut).
    assert "rlut" in list(out_coords["variable"])
    assert "ttr" not in list(out_coords["variable"])
    assert out.shape == (
        batch_size,
        len(time),
        len(_ATMOS_OUTPUT_TIMES),
        len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES),
        721,
        1440,
    )
    for key in out_coords:
        handshake_coords(out_coords, expected_coords, key)


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_v0_isccp_era5_latlon_iterator(device, batch_size):
    """LatLon iterator runs through several coupled rollout steps on lat/lon."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 64
    model = _build_latlon_model(device, nside=nside, use_ttr=True)
    in_coords = model.input_coords()

    time = np.array([np.datetime64("2020-01-01T00:00")])
    x = torch.randn(
        batch_size,
        len(time),
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        721,
        1440,
        device=device,
    )
    in_coords["batch"] = np.arange(batch_size)
    in_coords["time"] = time

    iterator = model.create_iterator(x, in_coords)

    coupler_step = _ATMOS_OUTPUT_TIMES[-1]
    next(iterator)  # initial condition

    for i in range(2):
        out, coords = next(iterator)
        assert out.shape == (
            batch_size,
            len(time),
            len(_ATMOS_OUTPUT_TIMES),
            len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES),
            721,
            1440,
        )
        assert "rlut" in list(coords["variable"])
        assert np.all(coords["lead_time"] == _ATMOS_OUTPUT_TIMES + coupler_step * i)

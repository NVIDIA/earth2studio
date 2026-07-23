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
import pandas as pd
import pytest
import torch

from earth2studio.utils.obs import ObsGridMapping

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

VARS = np.array(["t2m", "u10", "v10"])
LAT_1D = np.linspace(20.0, 50.0, 8, dtype=np.float32)  # shape (8,)
LON_1D = np.linspace(230.0, 300.0, 10, dtype=np.float32)  # shape (10,)
LAT_2D, LON_2D = np.meshgrid(LAT_1D, LON_1D, indexing="ij")  # shape (8, 10)


def make_mapping(grid_type: str = "regular", device: str = "cpu") -> ObsGridMapping:
    """Return an ObsGridMapping over a small test grid (regular or irregular)."""
    if grid_type == "regular":
        return ObsGridMapping(VARS, LAT_1D, LON_1D, device=device)
    return ObsGridMapping(VARS, LAT_2D, LON_2D, device=device)


def make_obs_df(
    variables=("t2m",),
    lats=(35.0,),
    lons=(265.0,),
    values=(1.0,),
    times=None,
) -> pd.DataFrame:
    """Build a minimal observation DataFrame; pass times to include a time column."""
    data = {
        "variable": list(variables),
        "lat": list(lats),
        "lon": list(lons),
        "observation": list(values),
    }
    if times is not None:
        data["time"] = list(times)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# device fixture — parametrizes CPU and GPU; skips GPU when unavailable
# ---------------------------------------------------------------------------


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request) -> str:
    """Parametrize over CPU and CUDA; skip CUDA when CuPy or CUDA is not available."""
    d = request.param
    if d != "cpu":
        import earth2studio.utils.obs as _obs_mod

        if _obs_mod.cp is None:
            pytest.skip("CuPy not available")
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    return d


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_init_regular_shape(device):
    """grid_shape, grid_type, and device are set correctly for a 1-D lat/lon grid."""
    m = make_mapping("regular", device)
    assert m.grid_shape == (len(VARS), len(LAT_1D), len(LON_1D))
    assert m.grid_type == "regular"
    assert m.device == torch.device(device)


def test_init_irregular_shape(device):
    """grid_shape and grid_type are set correctly for a 2-D lat/lon grid."""
    m = make_mapping("irregular", device)
    assert m.grid_shape == (len(VARS), *LAT_2D.shape)
    assert m.grid_type == "irregular"


def test_init_bad_lat_ndim():
    """ValueError is raised when grid_lat is neither 1-D nor 2-D."""
    lat_3d = np.zeros((2, 3, 4), dtype=np.float32)
    lon_3d = np.zeros((2, 3, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        ObsGridMapping(VARS, lat_3d, lon_3d)


def test_init_device_from_tensor():
    """Device is inferred from the grid_lat tensor when device is not given explicitly."""
    lat_t = torch.tensor(LAT_1D, device="cpu")
    lon_t = torch.tensor(LON_1D, device="cpu")
    m = ObsGridMapping(VARS, lat_t, lon_t)
    assert m.device == torch.device("cpu")


def test_init_uses_cupy():
    """On a CUDA device with CuPy available, m.xp is the cupy module for both grid types."""
    import earth2studio.utils.obs as _obs_mod

    if _obs_mod.cp is None:
        pytest.skip("CuPy not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    import cupy as cp

    for grid_type in ("regular", "irregular"):
        m = make_mapping(grid_type, "cuda:0")
        assert m.xp is cp


# ---------------------------------------------------------------------------
# obs_coords
# ---------------------------------------------------------------------------


def test_obs_coords_grid_corners(device):
    """An obs at the grid origin maps to fractional indices (0, 0) on the correct device."""
    m = make_mapping("regular", device)
    obs_var = np.array(["t2m"])
    obs_lat = np.array([LAT_1D[0]], dtype=np.float32)
    obs_lon = np.array([LON_1D[0]], dtype=np.float32)
    c, i, j, valid = m.obs_coords(obs_var, obs_lat, obs_lon)
    assert valid.all()
    assert c.item() == 0
    assert i.item() == pytest.approx(0.0, abs=1e-4)
    assert j.item() == pytest.approx(0.0, abs=1e-4)
    assert c.device.type == torch.device(device).type


def test_obs_coords_unknown_variable_filtered(device):
    """Observations with a variable not in grid_variables are removed."""
    m = make_mapping("regular", device)
    obs_var = np.array(["unknown"])
    obs_lat = np.array([35.0], dtype=np.float32)
    obs_lon = np.array([265.0], dtype=np.float32)
    c, i, j, valid = m.obs_coords(obs_var, obs_lat, obs_lon)
    assert not valid.any()
    assert len(c) == 0


def test_obs_coords_out_of_bounds_removed(device):
    """Observations outside the grid extent are removed when remove_out_of_bounds is True."""
    m = make_mapping("regular", device)
    obs_var = np.array(["t2m"])
    obs_lat = np.array([0.0], dtype=np.float32)  # below grid
    obs_lon = np.array([265.0], dtype=np.float32)
    c, i, j, valid = m.obs_coords(obs_var, obs_lat, obs_lon)
    assert not valid.any()
    assert len(c) == 0


def test_obs_coords_remove_out_of_bounds_false(device):
    """With remove_out_of_bounds=False all rows are returned; the mask flags invalid ones."""
    m = make_mapping("regular", device)
    obs_var = np.array(["t2m", "unknown"])
    obs_lat = np.array([35.0, 35.0], dtype=np.float32)
    obs_lon = np.array([265.0, 265.0], dtype=np.float32)
    c, i, j, valid = m.obs_coords(obs_var, obs_lat, obs_lon, remove_out_of_bounds=False)
    assert len(c) == 2
    assert valid[0].item() is True
    assert valid[1].item() is False


def test_obs_coords_gpu_matches_cpu():
    """GPU and CPU obs_coords produce bit-identical channel indices and matching fractional indices."""
    import earth2studio.utils.obs as _obs_mod

    if _obs_mod.cp is None:
        pytest.skip("CuPy not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    obs_var = np.array(["t2m", "u10", "v10"])
    obs_lat = np.array([LAT_1D[1], LAT_1D[4], 0.0], dtype=np.float32)
    obs_lon = np.array([LON_1D[2], LON_1D[7], LON_1D[3]], dtype=np.float32)

    c_cpu, i_cpu, j_cpu, v_cpu = make_mapping(device="cpu").obs_coords(
        obs_var, obs_lat, obs_lon
    )
    c_gpu, i_gpu, j_gpu, v_gpu = make_mapping(device="cuda:0").obs_coords(
        obs_var, obs_lat, obs_lon
    )

    assert torch.equal(c_cpu, c_gpu.cpu())
    assert torch.allclose(i_cpu, i_gpu.cpu(), atol=1e-4)
    assert torch.allclose(j_cpu, j_gpu.cpu(), atol=1e-4)
    assert torch.equal(v_cpu, v_gpu.cpu())


# ---------------------------------------------------------------------------
# obs_to_grid
# ---------------------------------------------------------------------------


def test_obs_to_grid_none_input(device):
    """None input returns (None, None) when return_empty_grid is False."""
    y, mask = make_mapping(device=device).obs_to_grid(None)
    assert y is None and mask is None


def test_obs_to_grid_empty_df(device):
    """An empty DataFrame returns (None, None) when return_empty_grid is False."""
    y, mask = make_mapping(device=device).obs_to_grid(pd.DataFrame())
    assert y is None and mask is None


def test_obs_to_grid_return_empty_grid(device):
    """return_empty_grid=True yields zero-filled tensors of the correct shape and device."""
    m = make_mapping(device=device)
    y, mask = m.obs_to_grid(None, return_empty_grid=True)
    assert y is not None and mask is not None
    assert y.shape == (1, *m.grid_shape)
    assert y.sum() == 0
    assert y.device.type == torch.device(device).type


def test_obs_to_grid_single_obs(device):
    """A single observation lands in the correct grid cell with value and mask == 1."""
    m = make_mapping(device=device)
    obs = make_obs_df(["t2m"], [LAT_1D[3]], [LON_1D[5]], [42.0])
    y, mask = m.obs_to_grid(obs)
    assert y is not None
    assert y.shape == (1, *m.grid_shape)
    assert mask.sum().item() == 1
    assert y[0, 0, 3, 5].item() == pytest.approx(42.0, abs=1e-4)
    assert y.device.type == torch.device(device).type


def test_obs_to_grid_averaging(device):
    """Two obs in the same cell should be averaged."""
    m = make_mapping(device=device)
    lat, lon = float(LAT_1D[2]), float(LON_1D[4])
    obs = make_obs_df(["t2m", "t2m"], [lat, lat], [lon, lon], [10.0, 20.0])
    y, mask = m.obs_to_grid(obs)
    assert y[0, 0, 2, 4].item() == pytest.approx(15.0, abs=1e-3)
    assert mask[0, 0, 2, 4].item() == pytest.approx(1.0)


def test_obs_to_grid_variable_filter(device):
    """Observations for variables not in the filter list are excluded from the grid."""
    m = make_mapping(device=device)
    obs = make_obs_df(
        ["t2m", "u10"], [LAT_1D[1], LAT_1D[2]], [LON_1D[1], LON_1D[2]], [5.0, 9.0]
    )
    y, mask = m.obs_to_grid(obs, variables=["t2m"])
    assert mask[0, 1].sum().item() == 0  # u10 channel empty


def test_obs_to_grid_time_filter_symmetric():
    """A scalar time_tolerance creates a symmetric window; obs outside it are dropped."""
    m = make_mapping()
    t0 = np.datetime64("2024-01-01T00:00", "ns")
    obs = make_obs_df(
        ["t2m", "t2m"],
        [LAT_1D[1], LAT_1D[2]],
        [LON_1D[1], LON_1D[2]],
        [1.0, 2.0],
        times=[t0 + np.timedelta64(30, "m"), t0 + np.timedelta64(90, "m")],
    )
    y, mask = m.obs_to_grid(
        obs, request_time=t0, time_tolerance=np.timedelta64(60, "m")
    )
    assert mask.sum().item() == 1  # only the 30-min obs is within ±60 min


def test_obs_to_grid_time_filter_asymmetric():
    """A (lower, upper) tuple time_tolerance supports one-sided windows."""
    m = make_mapping()
    t0 = np.datetime64("2024-01-01T00:00", "ns")
    obs = make_obs_df(
        ["t2m", "t2m"],
        [LAT_1D[1], LAT_1D[2]],
        [LON_1D[1], LON_1D[2]],
        [1.0, 2.0],
        times=[t0 - np.timedelta64(10, "m"), t0 + np.timedelta64(10, "m")],
    )
    tol = (np.timedelta64(0, "m"), np.timedelta64(30, "m"))
    y, mask = m.obs_to_grid(obs, request_time=t0, time_tolerance=tol)
    assert mask.sum().item() == 1  # only the forward obs


def test_obs_to_grid_gpu_matches_cpu():
    """GPU and CPU obs_to_grid produce identical gridded values and masks."""
    import earth2studio.utils.obs as _obs_mod

    if _obs_mod.cp is None:
        pytest.skip("CuPy not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    obs = make_obs_df(
        ["t2m", "u10"],
        [LAT_1D[2], LAT_1D[5]],
        [LON_1D[3], LON_1D[7]],
        [10.0, 20.0],
    )
    y_cpu, mask_cpu = make_mapping(device="cpu").obs_to_grid(obs)
    y_gpu, mask_gpu = make_mapping(device="cuda:0").obs_to_grid(obs)

    assert torch.allclose(y_cpu, y_gpu.cpu(), atol=1e-4)
    assert torch.equal(mask_cpu, mask_gpu.cpu())


# ---------------------------------------------------------------------------
# grid_to_obs
# ---------------------------------------------------------------------------


def test_grid_to_obs_nearest(device):
    """Nearest-neighbour sampling returns the value of the closest grid node."""
    m = make_mapping(device=device)
    x = torch.zeros(m.grid_shape, device=device)
    x[0, 3, 5] = 7.0
    c, i, j, _ = m.obs_coords(
        np.array(["t2m"]),
        np.array([LAT_1D[3]], dtype=np.float32),
        np.array([LON_1D[5]], dtype=np.float32),
    )
    result = m.grid_to_obs(x, c, i, j, method="nearest")
    assert result.item() == pytest.approx(7.0, abs=1e-4)
    assert result.device.type == torch.device(device).type


def test_grid_to_obs_linear_exact_node(device):
    """Bilinear sampling at an exact grid node returns that node's value."""
    m = make_mapping(device=device)
    x = torch.arange(float(m.C * m.H * m.W), device=device).view(m.grid_shape)
    c, i, j, _ = m.obs_coords(
        np.array(["u10"]),
        np.array([LAT_1D[4]], dtype=np.float32),
        np.array([LON_1D[2]], dtype=np.float32),
    )
    result = m.grid_to_obs(x, c, i, j, method="linear")
    assert result.item() == pytest.approx(x[c.item(), 4, 2].item(), abs=1e-3)


def test_grid_to_obs_linear_interpolation(device):
    """Bilinear midpoint between two cells should average them."""
    m = make_mapping(device=device)
    x = torch.zeros(m.grid_shape, device=device)
    x[0, 2, 4] = 4.0
    c = torch.tensor([0], device=device)
    i = torch.tensor([2.0], device=device)
    j = torch.tensor([3.5], device=device)  # midpoint between col 3 and 4
    result = m.grid_to_obs(x, c, i, j, method="linear")
    assert result.item() == pytest.approx(2.0, abs=1e-4)


def test_grid_to_obs_invalid_method():
    """An unknown interpolation method name raises ValueError."""
    m = make_mapping()
    x = torch.zeros(m.grid_shape)
    c, i, j = torch.tensor([0]), torch.tensor([1.0]), torch.tensor([1.0])
    with pytest.raises(ValueError, match="Unknown interpolation method"):
        m.grid_to_obs(x, c, i, j, method="cubic")


def test_grid_to_obs_bad_x_shape():
    """Passing a 4-D grid tensor (instead of 3-D) raises ValueError."""
    m = make_mapping()
    x = torch.zeros(1, *m.grid_shape)  # 4-D — should raise
    c, i, j = torch.tensor([0]), torch.tensor([1.0]), torch.tensor([1.0])
    with pytest.raises(ValueError):
        m.grid_to_obs(x, c, i, j)


# ---------------------------------------------------------------------------
# Irregular grid
# ---------------------------------------------------------------------------


def test_obs_to_grid_irregular(device):
    """An obs at an exact 2-D grid node is binned into a non-empty cell on the correct device."""
    m = make_mapping("irregular", device)
    obs = make_obs_df(["t2m"], [float(LAT_2D[3, 5])], [float(LON_2D[3, 5])], [99.0])
    y, mask = m.obs_to_grid(obs)
    assert y is not None
    assert mask.sum().item() >= 1
    assert y.device.type == torch.device(device).type

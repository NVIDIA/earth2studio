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

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import earth2studio.models.dx.hirov2 as hiro_src
from earth2studio.grids import resolve_grid
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx.hirov2 import HiROv2, _select_extent
from earth2studio.utils.coords import coord_array
from earth2studio.utils.imports import OptionalDependencyFailure

_FACTOR = 2
_IN_NAMES = ["PRATEsfc", "UGRD10m", "VGRD10m", "PRMSL"]


@pytest.fixture(autouse=True)
def optional_backend(monkeypatch):
    if hiro_src.__file__ in OptionalDependencyFailure.failures:
        monkeypatch.delitem(OptionalDependencyFailure.failures, hiro_src.__file__)


class _Predictor(torch.nn.Module):
    """Stand-in for the fme predictor with the same public surface."""

    def __init__(self, static: bool = False, noisy: bool = True) -> None:
        super().__init__()
        self.coarse_shape = (4, 4)
        self.downscale_factor = _FACTOR
        ref_lat = np.asarray(resolve_grid("gaussian-f90").coords()["lat"].values)
        ref_lon = np.asarray(resolve_grid("gaussian-f90").coords()["lon"].values)
        # Ascending fine grid with the same number of points as fme's.
        fine_lat = np.linspace(-89.9, 89.9, ref_lat.size * _FACTOR)
        fine_lon = np.linspace(0.1, 359.9, ref_lon.size * _FACTOR)
        self.full_fine_coords = SimpleNamespace(
            lat=torch.as_tensor(fine_lat, dtype=torch.float32),
            lon=torch.as_tensor(fine_lon, dtype=torch.float32),
        )
        self.static_inputs = None
        if static:
            global_field = torch.arange(
                fine_lat.size * fine_lon.size, dtype=torch.float32
            ).reshape(fine_lat.size, fine_lon.size)
            self.static_inputs = SimpleNamespace(
                fields=[SimpleNamespace(data=global_field)]
            )
        self.noisy = noisy
        self.static_seen: list = []

    def generate(self, coarse, static_inputs, n_samples=1):
        self.static_seen.append(static_inputs)
        sample = next(iter(coarse.values()))
        n_batch, height, width = sample.shape
        factor = self.downscale_factor
        noise = torch.randn(
            n_batch, n_samples, height * factor, width * factor, device=sample.device
        )
        if not self.noisy:
            noise = torch.zeros_like(noise)
        fields = {}
        for name, value in coarse.items():
            up = value.repeat_interleave(factor, dim=-1).repeat_interleave(
                factor, dim=-2
            )
            fields[name] = up.unsqueeze(1) + noise
        return fields, None, None


def _build(model=None, lat=None, lon=None, **kwargs) -> HiROv2:
    model = _Predictor() if model is None else model
    if lat is None or lon is None:
        lat, lon = _region()
    return HiROv2(model, lat, lon, in_names=_IN_NAMES, out_names=_IN_NAMES, **kwargs)


def _region(n_lat: int = 6, n_lon: int = 6, lat0: int = 60, lon0: int = 100):
    """North-to-south latitude and increasing longitude from the reference grid."""
    coords = resolve_grid("gaussian-f90").coords()
    ref_lat = np.asarray(coords["lat"].values)
    ref_lon = np.asarray(coords["lon"].values)
    lat_asc = ref_lat[lat0 : lat0 + n_lat]
    lon = ref_lon[lon0 : lon0 + n_lon]
    return lat_asc[::-1].copy(), lon.copy()


def _frame(model: HiROv2, values: np.ndarray):
    signature = model.input_coords()
    frame = coord_array(
        ("time", "variable", "lat", "lon"),
        {
            "time": np.array(["2023-01-01T00:00"], dtype="datetime64[ns]"),
            "variable": signature.coords["variable"].values,
        },
        grid=model.grid,
        dtype=np.float32,
    )
    return frame.copy(data=values)


def test_region_limits():
    lat, lon = _region()
    with pytest.raises(ValueError, match="88S-88N"):
        HiROv2(
            _Predictor(),
            np.array([89.0, 80.0]),
            lon[:2],
            in_names=_IN_NAMES,
            out_names=_IN_NAMES,
        )
    with pytest.raises(ValueError, match="north-to-south"):
        _build(lat=lat[::-1], lon=lon)
    with pytest.raises(ValueError, match="smaller than"):
        _build(lat=lat[:2], lon=lon)


def test_extent_selection_is_inside_the_box():
    lat, lon = _select_extent("gaussian-f90", (34.0, 50.0), (233.0, 250.0))
    assert lat[0] > lat[-1]
    assert np.all(np.diff(lat) < 0)
    assert np.all(np.diff(lon) > 0)
    assert np.all((lat >= 34.0) & (lat <= 50.0))
    assert np.all((lon >= 233.0) & (lon <= 250.0))


def test_off_grid_coordinates_are_rejected():
    lat, lon = _region()
    lat = lat.copy()
    lat[0] = lat[0] + 0.25
    with pytest.raises(ValueError, match="reference grid"):
        _build(lat=lat, lon=lon)


def test_output_grid_comes_from_the_checkpoint_fine_grid():
    model = _Predictor()
    hiro = _build(model)
    lat, lon = _region()
    fine = hiro.fine_grid
    assert fine.latitude.size == lat.size * _FACTOR
    assert fine.longitude.size == lon.size * _FACTOR
    assert np.all(np.diff(fine.latitude) < 0)
    assert np.all(np.diff(fine.longitude) > 0)


def test_overlap_average_matches_a_constant_field():
    model = _Predictor(noisy=False)
    hiro = _build(model, overlap=1)
    lat, lon = _region()
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)[None, :, None, None]
    frame = _frame(hiro, np.broadcast_to(values, (1, 4, lat.size, lon.size)).copy())
    out = hiro(frame)
    assert out.sizes["sample"] == 1
    assert out.sizes["lat"] == lat.size * _FACTOR
    assert out.sizes["lon"] == lon.size * _FACTOR
    np.testing.assert_allclose(out.sel(variable="tp").values, 1.0)
    np.testing.assert_allclose(out.sel(variable="msl").values, 4.0)


def test_static_windows_are_sliced_from_the_global_fields():
    model = _Predictor(static=True, noisy=False)
    hiro = _build(model, overlap=1)
    lat, lon = _region()
    frame = _frame(hiro, np.zeros((1, 4, lat.size, lon.size), dtype=np.float32))
    hiro(frame)
    # Coarse positions are ascending in latitude inside the model.
    ref = resolve_grid("gaussian-f90").coords()
    ref_lat = np.asarray(ref["lat"].values)
    ref_lon = np.asarray(ref["lon"].values)
    lat_asc = np.sort(np.argmin(np.abs(ref_lat[None] - lat[:, None]), axis=1))
    lon_idx = np.argmin(np.abs(ref_lon[None] - lon[:, None]), axis=1)
    full = model.static_inputs.fields[0].data
    # Windows are 4 coarse cells with overlap 1 over 6 cells: starts 0 and 2.
    first = model.static_seen[0]
    fine_rows = (lat_asc[0:4, None] * _FACTOR + np.arange(_FACTOR)).reshape(-1)
    fine_cols = (lon_idx[0:4, None] * _FACTOR + np.arange(_FACTOR)).reshape(-1)
    expected = full[torch.as_tensor(fine_rows)][:, torch.as_tensor(fine_cols)]
    torch.testing.assert_close(first.fields[0].data, expected)
    assert first.fields[0].data.shape == (4 * _FACTOR, 4 * _FACTOR)


def test_hirov2_conformance():
    hiro = _build(_Predictor(), number_of_samples=2)
    assert check_diagnostic_contract(hiro) == []


def test_same_seed_reproduces_and_a_new_seed_changes_the_draw():
    hiro = _build(_Predictor(), number_of_samples=2)
    lat, lon = _region()
    frame = _frame(hiro, np.ones((1, 4, lat.size, lon.size), dtype=np.float32))
    hiro.set_rng(0)
    first = hiro(frame)
    hiro.set_rng(0)
    second = hiro(frame)
    hiro.set_rng(1)
    third = hiro(frame)
    np.testing.assert_allclose(first.values, second.values)
    assert not np.allclose(first.values, third.values)

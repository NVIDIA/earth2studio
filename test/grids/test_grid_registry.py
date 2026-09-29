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

import runpy
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

import earth2studio.grids as grids
from earth2studio.grids import (
    E2S_CRS,
    E2S_GRID_ID,
    GridDefinition,
    HEALPixGrid,
    ProjectedGrid,
    infer_grid,
    list_grids,
    register_grid,
    resolve_grid,
)

GRID = ProjectedGrid([0, 3000], [0, 3000, 6000], "EPSG:3857")


def test_builtin_grids_are_lazy_and_cached(monkeypatch):
    constructors = []
    for module, name in (
        (grids.latlon, "LatLonGrid"),
        (grids.projected, "ProjectedGrid"),
        (grids.healpix, "HEALPixGrid"),
    ):
        constructor = Mock(wraps=getattr(module, name))
        monkeypatch.setattr(module, name, constructor)
        constructors.append(constructor)
    registry = runpy.run_path(grids.__file__)
    names = registry["list_grids"]()
    assert "healpix-l10-nested" in names
    assert all(constructor.call_count == 0 for constructor in constructors)
    resolve = registry["resolve_grid"]
    grid = resolve("hpx3")
    assert grid.level == 3
    assert resolve("healpix-l3-nested") is resolve("hpx3") is grid
    assert [constructor.call_count for constructor in constructors] == [0, 0, 1]
    assert registry["list_grids"]() == names
    # Pending built-ins reserve names/aliases and retain idempotent registration.
    registry["register_grid"]("latlon-1deg", resolve("latlon-1deg"))
    with pytest.raises(ValueError, match="conflicts"):
        registry["register_grid"]("custom-grid", GRID, aliases=("hpx10",))
    with pytest.raises(ValueError, match="conflicts"):
        registry["register_grid"]("healpix-l6-nested", GRID)


def test_grid_registry_builtins():
    assert {"latlon-0.25deg", "hrrr-conus-3km", "healpix-l6-nested"} <= set(
        list_grids()
    )
    assert resolve_grid("fcn1").shape == (720, 1440)
    assert isinstance(resolve_grid("hrrr"), GridDefinition)


def test_grid_registry_registration():
    register_grid("test-registry-grid", GRID, aliases=("test-registry-alias",))
    register_grid("test-registry-grid", GRID, aliases=("test-registry-alias",))
    assert resolve_grid("test-registry-alias") is GRID
    for name, definition, message in (
        ("test-registry-alias", GRID, "conflicts"),
        ("EPSG:4326", GRID, "valid CRS"),
        ("invalid-grid", object(), "implement GridDefinition"),
    ):
        with pytest.raises((TypeError, ValueError), match=message):
            register_grid(name, definition)  # type: ignore[arg-type]


def test_grid_registry_inference_validation():
    register_grid("test-explicit-grid", GRID)
    array = xr.DataArray(
        np.ones(GRID.shape),
        dims=GRID.dims,
        coords={"y": GRID.y, "x": GRID.x},
        attrs={E2S_CRS: "EPSG:3857", E2S_GRID_ID: "test-explicit-grid"},
    )
    assert infer_grid(array) is GRID
    assert infer_grid(array.expand_dims(time=[0])) is GRID
    assert infer_grid(array.isel(x=slice(2))).shape == (2, 2)


def test_grid_registry_errors():
    for operation, message in (
        (lambda: resolve_grid("missing"), "Unknown"),
        (lambda: resolve_grid("EPSG:4326"), "does not define"),
        (lambda: infer_grid(xr.DataArray(np.ones(1), dims="x")), "Cannot infer"),
    ):
        with pytest.raises(ValueError, match=message):
            operation()


@pytest.mark.parametrize("resolution,shape", [(1, (181, 360)), (1.5, (121, 240))])
def test_global_latlon_builtins(resolution, shape):
    grid = resolve_grid(f"latlon-{resolution}deg")
    coords = grid.coords()
    assert grid.shape == shape
    np.testing.assert_array_equal(
        coords["lat"], np.arange(90, -90 - resolution, -resolution)
    )
    np.testing.assert_array_equal(coords["lon"], np.arange(0, 360, resolution))


def test_gaussian_f90_builtin():
    from scipy.special import roots_legendre

    grid = resolve_grid("gaussian-f90")
    coords = grid.coords()
    assert grid.shape == (180, 360)
    np.testing.assert_allclose(
        coords["lat"], np.degrees(np.arcsin(roots_legendre(180)[0])), rtol=0, atol=1e-12
    )
    np.testing.assert_array_equal(coords["lon"], np.arange(0.5, 360, 1))
    assert np.all(np.diff(coords["lat"]) > 0)
    assert resolve_grid("ace2") is grid


@pytest.mark.parametrize("level", [3, 6, 10])
def test_healpix_builtin_orderings(level):
    fingerprints = set()
    for suffix, kwargs in (
        ("nested", {"ordering": "nested"}),
        ("ring", {"ordering": "ring"}),
        (
            "xy-north-clockwise",
            {"ordering": "xy", "xy_origin": "north", "xy_clockwise": True},
        ),
        (
            "xy-north-clockwise-face",
            {
                "ordering": "xy",
                "xy_origin": "north",
                "xy_clockwise": True,
                "layout": "face",
            },
        ),
    ):
        grid = resolve_grid(f"healpix-l{level}-{suffix}")
        expected = HEALPixGrid(level, **kwargs)
        assert grid.shape == expected.shape
        assert grid.attrs == expected.attrs
        indexes = {
            dim: np.array([0, size - 1]) for dim, size in zip(grid.dims, grid.shape)
        }
        xr.testing.assert_identical(
            grid.coords(indexes).to_dataset(), expected.coords(indexes).to_dataset()
        )
        fingerprints.add(grid.fingerprint())
    assert len(fingerprints) == 4


@pytest.mark.parametrize(
    "y,x", [(slice(273, 785), slice(579, 1219)), (slice(17, 1041), slice(3, 1795))]
)
def test_hrrr_model_domain_selection(y, x):
    parent = resolve_grid("hrrr")
    indexes = parent.coords(only_index=True).to_dataset().isel(y=y, x=x).coords
    selected = parent.coords({dim: indexes[dim].values for dim in parent.dims})
    expected = ProjectedGrid(parent.y[y], parent.x[x], parent.crs)
    xr.testing.assert_identical(selected.to_dataset(), expected.coords().to_dataset())

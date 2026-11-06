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
import xarray as xr
import zarr

from earth2studio.grids import CurvilinearGrid, LatLonGrid
from earth2studio.io import ZarrBackend
from earth2studio.utils.coords import coord_array

# Protocol behavior is covered by test_io_contract.py

TIMES = np.array(["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[h]")
LEADS = np.arange(4) * np.timedelta64(6, "h")
VARIABLES = ["t2m", "tcwv"]
GRID = LatLonGrid(np.linspace(90, -90, 8), np.linspace(0, 360, 16, endpoint=False))


def make_template(
    variables: list[str] = VARIABLES, grid: object = GRID, **coords: np.ndarray
) -> xr.DataArray:
    return coord_array(
        ("time", "lead_time", "variable", *grid.dims),
        {
            "time": coords.get("time", TIMES),
            "lead_time": coords.get("lead_time", LEADS),
            "variable": variables,
        },
        grid=grid,
        attrs={"units": "K"},
    )


def make_field(
    lead_time: np.ndarray = LEADS,
    variables: list[str] = VARIABLES,
    seed: int = 0,
) -> xr.DataArray:
    shape = (len(TIMES), len(lead_time), len(variables), *GRID.shape)
    data = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("time", "lead_time", "variable", "lat", "lon"),
        coords={
            "time": TIMES,
            "lead_time": lead_time,
            "variable": variables,
            "lat": GRID.latitude,
            "lon": GRID.longitude,
        },
    )


def test_zarr_memory_store() -> None:
    z = ZarrBackend()
    assert isinstance(z.store, zarr.storage.MemoryStore)
    assert isinstance(z.root, zarr.Group)
    assert len(z) == 0

    z.add_array(make_template())
    # One array per variable label, plus one array per coordinate
    assert len(z) == 6
    assert sorted(z) == ["lat", "lead_time", "lon", "t2m", "tcwv", "time"]
    assert "t2m" in z and "time" in z and "u10m" not in z
    assert z["t2m"].shape == (2, 4, 8, 16)
    assert z["t2m"].dtype == np.float32
    # Default chunks isolate each time and lead time; unlisted dims are whole
    assert z["t2m"].chunks == (1, 1, 8, 16)
    assert z["t2m"].attrs["units"] == "K"
    assert z.coords["time"].dtype == np.dtype("datetime64[ns]")

    field = make_field(lead_time=LEADS[1:3])
    z.write(field)
    stored = z.to_xarray()
    np.testing.assert_array_equal(
        stored["t2m"].sel(lead_time=LEADS[1:3]).values,
        field.sel(variable="t2m", drop=True).values,
    )
    assert np.isnan(stored["t2m"].sel(lead_time=LEADS[0]).values).all()

    read = z.read(make_template(["t2m"], lead_time=LEADS[1:3]))
    np.testing.assert_array_equal(
        read.sel(variable="t2m", drop=True).values,
        field.sel(variable="t2m", drop=True).values,
    )


def test_zarr_file_store(tmp_path: str) -> None:
    file_name = str(tmp_path / "forecast.zarr")
    z = ZarrBackend(file_name=file_name)
    assert isinstance(z.store, zarr.storage.LocalStore)
    z.add_array(make_template())
    field = make_field()
    z.write(field)

    # Reopening the store recovers its coordinates and chunk layout
    reopened = ZarrBackend(file_name=file_name)
    np.testing.assert_array_equal(reopened.coords["lead_time"], LEADS)
    assert reopened.chunks["time"] == 1
    assert reopened.chunks["lon"] == 16

    # Resumed runs may call add_array unconditionally, keeping written data
    reopened.add_array(make_template())
    np.testing.assert_array_equal(
        reopened.to_xarray()["t2m"].values,
        field.sel(variable="t2m", drop=True).values,
    )
    reopened.write(make_field(variables=["t2m"], seed=1))
    reopened.close()

    stored = xr.open_zarr(file_name, consolidated=False)
    np.testing.assert_array_equal(
        stored["t2m"].values,
        make_field(variables=["t2m"], seed=1).sel(variable="t2m", drop=True).values,
    )
    np.testing.assert_array_equal(stored["time"].values, TIMES)

    with pytest.raises(ValueError, match="time"):
        reopened.add_array(make_template(time=TIMES + np.timedelta64(1, "h")))


def test_zarr_overwrite(tmp_path: str) -> None:
    file_name = str(tmp_path / "overwrite.zarr")
    ZarrBackend(file_name=file_name).add_array(make_template())

    z = ZarrBackend(file_name=file_name, backend_kwargs={"overwrite": True})
    assert len(z) == 0
    assert z.coords == {}
    z.add_array(make_template(["u10m"]))
    assert sorted(z) == ["lat", "lead_time", "lon", "time", "u10m"]


def test_zarr_chunks_and_codecs() -> None:
    z = ZarrBackend(
        chunks={"time": 2, "lead_time": 1, "lat": 4},
        zarr_codecs=zarr.codecs.BloscCodec(cname="zstd"),
    )
    z.add_array(make_template(["t2m"]))
    assert z["t2m"].chunks == (2, 1, 4, 16)
    assert any("blosc" in str(codec).lower() for codec in z["t2m"].compressors)

    field = make_field(variables=["t2m"])
    z.write(field)
    np.testing.assert_array_equal(
        z.to_xarray()["t2m"].values, field.sel(variable="t2m", drop=True).values
    )


def test_zarr_curvilinear_coords() -> None:
    latitude = np.arange(12.0).reshape(3, 4)
    grid = CurvilinearGrid(latitude, latitude + 100.0)
    z = ZarrBackend()
    z.add_array(make_template(["t2m"], grid=grid))

    # Curvilinear stores use the grid's own dims, not synthetic ilat/ilon
    assert z["t2m"].metadata.dimension_names == ("time", "lead_time", "y", "x")
    assert "ilat" not in z and "ilon" not in z
    assert z["lat"].metadata.dimension_names == ("y", "x")
    assert z["t2m"].attrs["coordinates"] == "lat lon"

    stored = z.to_xarray()["t2m"]
    assert stored.coords["lat"].dims == ("y", "x")
    np.testing.assert_array_equal(stored.coords["lat"].values, latitude)


def test_zarr_reopen_skips_coordinate_arrays_when_loading_chunks(
    tmp_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lat = np.arange(4)
    lon = np.arange(6)
    file_name = tmp_path / "chunks.zarr"
    root = zarr.group(zarr.storage.LocalStore(file_name), overwrite=True)

    root.create_array(
        "lat",
        shape=lat.shape,
        chunks=lat.shape,
        dtype=lat.dtype,
        dimension_names=["lat"],
    )
    root["lat"][:] = lat
    root.create_array(
        "fields",
        shape=(len(lat), len(lon)),
        chunks=(2, 3),
        dtype="float32",
        dimension_names=["lat", "lon"],
    )
    root.create_array(
        "lon",
        shape=lon.shape,
        chunks=lon.shape,
        dtype=lon.dtype,
        dimension_names=["lon"],
    )
    root["lon"][:] = lon

    root_type = type(root)
    original_iter = root_type.__iter__

    def iter_coords_first(group: zarr.Group):
        names = list(original_iter(group))
        ordered_names = ["lat", "fields", "lon"]
        if set(ordered_names).issubset(names):
            return iter(
                ordered_names + [name for name in names if name not in ordered_names]
            )
        return iter(names)

    monkeypatch.setattr(root_type, "__iter__", iter_coords_first)
    assert list(root)[:3] == ["lat", "fields", "lon"]

    z = ZarrBackend(file_name)

    assert z.chunks["lat"] == 2
    assert z.chunks["lon"] == 3


def test_zarr_reopen_with_scalar_array(tmp_path: str) -> None:
    file_name = tmp_path / "scalar.zarr"
    z = ZarrBackend(file_name, backend_kwargs={"overwrite": True})
    z.add_array(make_template())
    # Scalar array, e.g. a CF grid_mapping variable; zarr stores () as None
    z.root.create_array(
        "crs", shape=(), dtype="int32", fill_value=0, dimension_names=()
    )
    assert z.root["crs"].metadata.dimension_names is None
    z.root.create_group("diagnostics")

    z = ZarrBackend(file_name)
    assert {"time", "lead_time", "lat", "lon"} <= set(z.coords)
    assert "crs" in z
    assert "diagnostics" in z.root
    assert "diagnostics" not in z.coords

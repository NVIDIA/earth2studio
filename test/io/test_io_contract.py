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

"""Shared IO backend contract tests (rules I1-I11 in dev/spec/IO_SPEC.md).

Add each migrated backend to ``BACKENDS`` as a factory and a reader returning the
store as an :class:`xarray.Dataset` after ``flush``. Backends offering ``read`` are
also checked against it.
"""

from collections.abc import Callable

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import CurvilinearGrid, LatLonGrid, infer_grid
from earth2studio.io import IOBackend, XarrayBackend
from earth2studio.utils.coords import (
    E2S_DYNAMIC_DIMS,
    E2S_KIND,
    E2S_SCHEMA_VERSION,
    E2S_STATISTICS,
    coord_array,
)
from earth2studio.utils.cupy import from_torch

BACKENDS: dict[str, tuple[Callable[[], IOBackend], Callable[[IOBackend], xr.Dataset]]]
BACKENDS = {
    "xarray": (XarrayBackend, lambda io: io.root),  # type: ignore[attr-defined]
}

TIMES = np.array(["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[h]")
LEADS = np.arange(3) * np.timedelta64(6, "h")
VARIABLES = ["t2m", "tp:sum:6h"]
GRID = LatLonGrid(np.array([1.0, 0.0, -1.0]), np.arange(4.0))


@pytest.fixture(params=list(BACKENDS))
def backend(request: pytest.FixtureRequest) -> tuple[IOBackend, Callable]:
    factory, read = BACKENDS[request.param]
    return factory(), read


def make_template(
    variables: list[str] = VARIABLES, grid: object = GRID, **attrs: object
) -> xr.DataArray:
    return coord_array(
        ("time", "lead_time", "variable", "lat", "lon"),
        {"time": TIMES, "lead_time": LEADS, "variable": variables},
        grid=grid,
        attrs={"units": "K", **attrs},
    )


def make_field(
    time: np.ndarray = TIMES,
    lead_time: np.ndarray = LEADS,
    variables: list[str] = VARIABLES,
    seed: int = 0,
) -> xr.DataArray:
    shape = (len(time), len(lead_time), len(variables), 3, 4)
    data = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("time", "lead_time", "variable", "lat", "lon"),
        coords={
            "time": time,
            "lead_time": lead_time,
            "variable": variables,
            **GRID.coords(),
        },
    )


def test_protocol(backend: tuple[IOBackend, Callable]) -> None:
    io, _ = backend
    assert isinstance(io, IOBackend)


def test_add_array_from_signature(backend: tuple[IOBackend, Callable]) -> None:
    # I1, I3: allocation-free signatures work, labels name arrays verbatim
    io, read = backend
    io.add_array(make_template())
    io.flush()
    stored = read(io)
    assert set(stored.data_vars) == set(VARIABLES)
    for name in VARIABLES:
        assert stored[name].dims == ("time", "lead_time", "lat", "lon")
        assert stored[name].dtype == np.float32
        assert np.isnan(stored[name].values).all()


def test_add_array_named(backend: tuple[IOBackend, Callable]) -> None:
    # I3: without a variable dimension, the template name names the array
    io, read = backend
    seed = coord_array(("ensemble",), {"ensemble": [0, 1]}, dtype=np.int64, name="seed")
    io.add_array(seed)
    io.write(
        xr.DataArray(
            [17, 29], dims="ensemble", coords={"ensemble": [0, 1]}, name="seed"
        )
    )
    io.flush()
    np.testing.assert_array_equal(read(io)["seed"].values, [17, 29])

    with pytest.raises(ValueError, match="named"):
        io.add_array(coord_array(("ensemble",), {"ensemble": [0, 1]}))


@pytest.mark.parametrize(
    "template",
    [
        coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": ["t2m"]},
            dynamic=("batch",),
            grid=GRID,
        ),
        coord_array(("time", "variable"), {"variable": ["t2m"]}, sizes={"time": 0}),
    ],
    ids=["dynamic", "empty"],
)
def test_add_array_rejects_unresolved(
    backend: tuple[IOBackend, Callable], template: xr.DataArray
) -> None:
    # I2
    io, _ = backend
    with pytest.raises(ValueError):
        io.add_array(template)


def test_add_array_shared_coords(backend: tuple[IOBackend, Callable]) -> None:
    # I4: identical re-adds keep data; conflicting coordinates raise
    io, read = backend
    io.add_array(make_template())
    field = make_field()
    io.write(field)
    io.add_array(make_template())
    io.add_array(make_template(["u10m"]))
    io.flush()
    stored = read(io)
    np.testing.assert_array_equal(
        stored["t2m"].values, field.sel(variable="t2m").values
    )
    assert "u10m" in stored.data_vars

    shifted = make_template().assign_coords(time=TIMES + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="time"):
        io.add_array(shifted)
    with pytest.raises(ValueError, match="dimensions"):
        io.add_array(make_template().isel(lead_time=0, drop=True))


def test_write_by_label(backend: tuple[IOBackend, Callable]) -> None:
    # I5, I6: unordered, noncontiguous subsets with several lead times
    io, read = backend
    io.add_array(make_template())
    field = make_field(
        time=TIMES[::-1], lead_time=LEADS[[2, 0]], variables=VARIABLES[::-1]
    )
    io.write(field)
    io.flush()
    stored = read(io)
    for name in VARIABLES:
        expected = field.sel(variable=name, drop=True).transpose(*stored[name].dims)
        actual = stored[name].sel(time=field.time, lead_time=field.lead_time)
        np.testing.assert_array_equal(actual.values, expected.values)
    assert np.isnan(stored["t2m"].sel(lead_time=LEADS[1]).values).all()


@pytest.mark.parametrize(
    "field",
    [
        make_field(variables=["t2m", "u10m"]),
        make_field(time=TIMES + np.timedelta64(1, "h")),
        make_field().transpose("lead_time", "time", "variable", "lat", "lon"),
        make_field().isel(lead_time=0, drop=True),
        make_field(lead_time=LEADS[[0, 0]]),
    ],
    ids=["unknown-array", "unknown-label", "reordered", "missing-dim", "duplicate"],
)
def test_write_rejects_before_writing(
    backend: tuple[IOBackend, Callable], field: xr.DataArray
) -> None:
    # I5: invalid writes raise and leave the store unchanged
    io, read = backend
    io.add_array(make_template())
    with pytest.raises(ValueError):
        io.write(field)
    io.flush()
    stored = read(io)
    for name in VARIABLES:
        assert np.isnan(stored[name].values).all()


def test_write_borrows(backend: tuple[IOBackend, Callable]) -> None:
    # I7: the input is not modified, and the store does not alias it
    io, read = backend
    io.add_array(make_template())
    field = make_field(time=TIMES[::-1])
    original = field.copy(deep=True)
    io.write(field)
    xr.testing.assert_identical(field, original)
    field.values[:] = 0.0
    io.flush()
    stored = read(io)["t2m"].sel(time=TIMES[::-1])
    np.testing.assert_array_equal(
        stored.values, original.sel(variable="t2m", drop=True).values
    )


def test_write_torch_backed(backend: tuple[IOBackend, Callable]) -> None:
    # I8: Torch-adapter payloads are transferred by the backend
    io, read = backend
    io.add_array(make_template())
    field = make_field()
    torch_field = from_torch(torch.from_numpy(field.values), field, backend="torch")
    io.write(torch_field)
    io.flush()
    np.testing.assert_array_equal(
        read(io)["t2m"].values, field.sel(variable="t2m").values
    )


def test_write_cupy_backed(backend: tuple[IOBackend, Callable]) -> None:
    # I8: CuPy payloads are transferred by the backend
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("cupy")
    io, read = backend
    io.add_array(make_template())
    field = make_field()
    io.write(field.e2s.as_cupy())
    io.flush()
    np.testing.assert_array_equal(
        read(io)["t2m"].values, field.sel(variable="t2m").values
    )


def test_round_trip_metadata(backend: tuple[IOBackend, Callable]) -> None:
    # I9, I10: coordinates, auxiliaries and persisted attributes survive
    latitude = np.arange(12.0).reshape(3, 4)
    grid = CurvilinearGrid(latitude, latitude + 100.0)
    template = coord_array(
        ("time", "lead_time", "variable", "y", "x"),
        {"time": TIMES, "lead_time": LEADS, "variable": VARIABLES},
        grid=grid,
        attrs={"units": "K"},
    )
    template.coords["lead_time"].attrs["long_name"] = "forecast lead time"
    io, read = backend
    io.add_array(template)
    io.flush()
    stored = read(io)
    assert stored["time"].dtype == np.dtype("datetime64[ns]")
    assert stored["lead_time"].dtype == np.dtype("timedelta64[ns]")
    np.testing.assert_array_equal(stored["time"].values, TIMES)
    np.testing.assert_array_equal(stored["lead_time"].values, LEADS)
    assert stored["lead_time"].attrs["long_name"] == "forecast lead time"
    for name in VARIABLES:
        array = stored[name]
        np.testing.assert_array_equal(array.coords["lat"].values, latitude)
        assert array.coords["lat"].dims == ("y", "x")
        assert array.attrs["units"] == "K"
        for key in (E2S_KIND, E2S_SCHEMA_VERSION, E2S_DYNAMIC_DIMS, E2S_STATISTICS):
            assert key not in array.attrs
        assert infer_grid(array).fingerprint() == grid.fingerprint()


def test_close(backend: tuple[IOBackend, Callable]) -> None:
    # I11
    io, read = backend
    io.add_array(make_template())
    io.write(make_field())
    io.close()
    io.close()
    with pytest.raises(RuntimeError):
        io.write(make_field())
    assert not np.isnan(read(io)["t2m"].values).any()


@pytest.fixture
def readable(backend: tuple[IOBackend, Callable]) -> IOBackend:
    io, _ = backend
    if not hasattr(io, "read"):
        pytest.skip("Backend does not support reading")
    return io


def test_read_round_trip(readable: IOBackend) -> None:
    # Reading a template returns what was written, with stored metadata
    io = readable
    template = make_template()
    io.add_array(template)
    field = make_field()
    io.write(field)
    result = io.read(template)  # type: ignore[attr-defined]
    assert result.dims == field.dims
    np.testing.assert_array_equal(result.values, field.values)
    np.testing.assert_array_equal(result.coords["variable"], VARIABLES)
    assert result.attrs["units"] == "K"
    assert infer_grid(result).fingerprint() == GRID.fingerprint()


def test_read_by_label(readable: IOBackend) -> None:
    # Mapping selections choose dimension order, label order and arrays
    io = readable
    io.add_array(make_template())
    field = make_field()
    io.write(field)
    selection = {
        "variable": ["tp:sum:6h"],
        "lead_time": LEADS[[2, 0]],
        "time": TIMES[::-1],
        "lat": [-1.0, 1.0],
        "lon": np.arange(4.0),
    }
    result = io.read(selection, dtype=np.float64)  # type: ignore[attr-defined]
    expected = field.sel(selection).transpose(*selection)
    assert result.dims == tuple(selection)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result.values, expected.values)

    with pytest.raises(ValueError):
        io.read({**selection, "variable": ["u10m"]})  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        io.read({**selection, "lead_time": LEADS + 1})  # type: ignore[attr-defined]


def test_read_named(readable: IOBackend) -> None:
    io = readable
    seed = coord_array(("ensemble",), {"ensemble": [0, 1]}, dtype=np.int64, name="seed")
    io.add_array(seed)
    io.write(
        xr.DataArray(
            [17, 29], dims="ensemble", coords={"ensemble": [0, 1]}, name="seed"
        )
    )
    result = io.read(seed)  # type: ignore[attr-defined]
    assert result.name == "seed"
    np.testing.assert_array_equal(result.values, [17, 29])


def test_read_cupy(readable: IOBackend) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    pytest.importorskip("cupy")
    io = readable
    io.add_array(make_template())
    io.write(make_field())
    result = io.read(make_template(), device="cuda:0")  # type: ignore[attr-defined]
    assert result.e2s.is_cupy


def test_add_array_rejects_name_collisions(backend: tuple[IOBackend, Callable]) -> None:
    # Array names share a namespace with coordinates
    io, read = backend
    io.add_array(make_template())
    with pytest.raises(ValueError, match="collide"):
        io.add_array(coord_array(("lat",), {"lat": GRID.coords()["lat"]}, name="lat"))
    io.flush()
    np.testing.assert_array_equal(read(io)["lat"].values, GRID.coords()["lat"].values)


def test_add_array_rejects_repeated_labels(backend: tuple[IOBackend, Callable]) -> None:
    io, _ = backend
    with pytest.raises(ValueError, match="unique"):
        io.add_array(make_template().assign_coords(time=TIMES[[0, 0]]))


def test_write_unlabelled_dims_in_full(backend: tuple[IOBackend, Callable]) -> None:
    # Fields without labels must span the stored axis, never broadcast into it
    io, read = backend
    io.add_array(coord_array(("sample",), sizes={"sample": 3}, name="seed"))
    with pytest.raises(ValueError, match="span all 3"):
        io.write(xr.DataArray([7.0], dims="sample", name="seed"))
    io.write(xr.DataArray([1.0, 2.0, 3.0], dims="sample", name="seed"))
    io.flush()
    np.testing.assert_array_equal(read(io)["seed"].values, [1.0, 2.0, 3.0])

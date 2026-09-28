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
from inspect import Parameter, signature
from typing import Any

import dask.array as da
import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.utils as utils
from earth2studio.utils import (
    convert_multidim_to_singledim,
    coord_array,
    handshake_coords,
    handshake_dataarray,
    handshake_dataarrays,
    handshake_dim,
    handshake_metadata,
    handshake_nonempty,
    handshake_size,
    handshake_time,
)
from earth2studio.utils.coords import (
    cat_coords,
    map_coords,
    split_coords,
    tile_coords,
)
from earth2studio.utils.cupy import Earth2StudioAccessor


def test_handshake_device_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    data = np.arange(12).reshape(3, 4)[:, ::2]
    array = xr.DataArray(data, dims=("y", "x"))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Device validation must not convert, copy, or read field values")

    monkeypatch.setattr(Earth2StudioAccessor, "to_torch", forbidden)
    monkeypatch.setattr(xr.DataArray, "copy", forbidden)
    monkeypatch.setattr(xr.DataArray, "values", property(forbidden))
    monkeypatch.setattr(np, "copy", forbidden)
    monkeypatch.setattr(np, "array", forbidden)
    monkeypatch.setattr(np, "asarray", forbidden)
    monkeypatch.setattr(torch, "as_tensor", forbidden)
    monkeypatch.setattr(torch, "from_numpy", forbidden)
    monkeypatch.setattr(torch.cuda, "current_device", forbidden)
    for device in ("cpu", torch.device("cpu:0")):
        assert utils.handshake_device(array, device) is None
    with pytest.raises(ValueError, match="Expected.*cuda:0.*got.*cpu"):
        utils.handshake_device(array, "cuda:0")
    assert array.data is data


def test_handshake_device_api() -> None:
    parameters = signature(utils.handshake_device).parameters
    assert tuple(parameters) == ("array", "expected_device")
    assert all(p.default is Parameter.empty for p in parameters.values())
    assert all(p.kind is Parameter.POSITIONAL_OR_KEYWORD for p in parameters.values())
    assert (
        utils.handshake_device(array=xr.DataArray(np.zeros(1)), expected_device="cpu")
        is None
    )
    with pytest.raises(TypeError, match="DataArray"):
        utils.handshake_device(np.zeros(1), "cpu")
    with pytest.raises(TypeError, match="Unsupported.*device"):
        utils.handshake_device(xr.DataArray(np.zeros(1)), "meta")


def test_handshake_device_cuda_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    # Portable even without a CUDA runtime; resolving an explicit index needs none.
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    array = xr.DataArray(np.zeros(1))
    with pytest.raises(ValueError, match="Expected.*cuda:3.*got.*cpu"):
        utils.handshake_device(array, torch.device("cuda"))


def test_handshake_device_rejects_lazy_and_signature_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xarray.core.indexing import LazilyIndexedArray, NumpyIndexingAdapter

    lazy = xr.DataArray(da.zeros(2, chunks=1), dims="x")
    declaration = coord_array(("x",), sizes={"x": 2})
    backend = LazilyIndexedArray(NumpyIndexingAdapter(np.zeros(2)))
    backend_array = xr.DataArray(xr.Variable("x", backend))

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Unsupported storage must be rejected without materialization")

    monkeypatch.setattr(da.Array, "compute", forbidden)
    monkeypatch.setattr(type(declaration.data), "__array__", forbidden)
    monkeypatch.setattr(LazilyIndexedArray, "get_duck_array", forbidden)
    for array in (lazy, declaration, backend_array):
        with pytest.raises(TypeError, match="only NumPy- or CuPy"):
            utils.handshake_device(array, "cpu")


def test_handshake_device_without_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    from earth2studio.utils import cupy as cupy_utils

    def missing_cupy(name: str) -> None:
        raise ImportError

    monkeypatch.setattr(cupy_utils, "import_module", missing_cupy)
    assert utils.handshake_device(xr.DataArray(np.zeros(1)), "cpu") is None
    with pytest.raises(TypeError, match="only NumPy- or CuPy"):
        utils.handshake_device(coord_array(("x",), sizes={"x": 1}), "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_handshake_device_cupy(monkeypatch: pytest.MonkeyPatch) -> None:
    cp = pytest.importorskip("cupy")
    index = torch.cuda.current_device()
    with cp.cuda.Device(index):
        data = cp.arange(12).reshape(3, 4)[:, ::2]
    array = xr.DataArray(data, dims=("y", "x"))
    pointer = data.data.ptr

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Device validation must not convert, copy, or read field values")

    monkeypatch.setattr(Earth2StudioAccessor, "to_torch", forbidden)
    monkeypatch.setattr(xr.DataArray, "copy", forbidden)
    monkeypatch.setattr(xr.DataArray, "values", property(forbidden))
    monkeypatch.setattr(cp, "copy", forbidden)
    monkeypatch.setattr(cp, "array", forbidden)
    monkeypatch.setattr(cp, "asarray", forbidden)
    monkeypatch.setattr(cp, "asnumpy", forbidden)
    monkeypatch.setattr(torch, "from_dlpack", forbidden)
    for device in (torch.device("cuda", index), "cuda"):
        assert utils.handshake_device(array, device) is None
    with pytest.raises(ValueError, match=f"Expected.*cpu.*got.*cuda:{index}"):
        utils.handshake_device(array, "cpu")
    # Comparing another index does not require allocating on a second GPU.
    with pytest.raises(
        ValueError, match=f"Expected.*cuda:{index + 1}.*got.*cuda:{index}"
    ):
        utils.handshake_device(array, f"cuda:{index + 1}")
    assert array.data is data
    assert data.data.ptr == pointer


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("batch", []), ("variable", []), ("lat", []), ("lon", [])]),
        OrderedDict([("time", []), ("lat", []), ("lon", [])]),
    ],
)
def test_handshake_dim(coords):
    # Check dims no index
    for dim in list(coords.keys()):
        handshake_dim(coords, dim)
    # Check dims with index
    for i, dim in enumerate(list(coords.keys())):
        handshake_dim(coords, dim, i)
    # Check dims with reverse index
    for i, dim in enumerate(list(coords.keys())[::-1]):
        handshake_dim(coords, dim, -(i + 1))


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("a", []), ("b", []), ("lat", []), ("lon", [])]),
        OrderedDict([("lat", []), ("lon", [])]),
    ],
)
def test_handshake_dim_failure(coords):

    with pytest.raises(KeyError):
        handshake_dim(coords, "fake_dim")

    with pytest.raises(ValueError):
        handshake_dim(coords, "lat", -1)

    with pytest.raises(ValueError):
        handshake_dim(coords, "lat", 5)


def test_handshakes_dataarray_dimensions_and_labels():
    signature = coord_array(
        ("member", "variable", "y"),
        {"variable": ["t2m"], "latitude": ("y", [10, 20])},
        sizes={"member": 3},
    )
    handshake_dim(signature, "member", 0)
    handshake_size(signature, "member", 3)
    handshake_dim(signature, ("member", "variable", "y"))
    handshake_coords(signature, signature.assign_coords(source="test"), "latitude")
    with pytest.raises(KeyError):
        handshake_dim(signature, "latitude")
    with pytest.raises(ValueError):
        handshake_coords(
            signature, signature.assign_coords(latitude=("y", [20, 10])), "latitude"
        )
    handshake_coords(signature, signature.isel(y=[1]), "latitude", subset=True)
    with pytest.raises(ValueError):
        handshake_coords(
            signature,
            signature.assign_coords(latitude=("y", [20, 30])),
            "latitude",
            subset=True,
        )


def test_handshake_relative_history_and_runtime():
    declaration = coord_array(
        ("batch", "time", "lead_time", "variable"),
        {"lead_time": np.array([-6, 0], dtype="timedelta64[h]"), "variable": ["t2m"]},
        dynamic=("batch", "time"),
    )
    handshake_dataarray(declaration, declaration)
    handshake_time(declaration, allow_dynamic=True)
    with pytest.raises(ValueError, match="nonempty"):
        handshake_nonempty(declaration)
    actual = coord_array(
        ("member", "time", "lead_time", "variable"),
        {
            "time": np.array(["2020-01-01"], dtype="datetime64[D]"),
            "lead_time": np.array([6, 12], dtype="timedelta64[h]"),
            "variable": ["t2m"],
        },
        sizes={"member": 2},
    )
    handshake_time(actual)
    handshake_nonempty(actual)
    handshake_time(actual, "lead_time")
    lead = actual.lead_time.values
    relative = actual.assign_coords(lead_time=lead - lead[-1])
    handshake_dataarray(relative, declaration)
    with pytest.raises(ValueError):
        handshake_time(declaration)
    for bad in (
        np.array([0, 6]),
        np.array(["NaT", "2020-01-01"], dtype="datetime64[D]"),
        np.array(["NaT", 6], dtype="timedelta64[h]"),
        np.array([0, 3], dtype="timedelta64[h]"),
    ):
        with pytest.raises(ValueError):
            changed = actual.assign_coords(lead_time=bad)
            handshake_time(changed, "lead_time")
            handshake_dataarray(
                changed.assign_coords(lead_time=bad - bad[-1]), declaration
            )
    with pytest.raises(ValueError):
        handshake_time(actual.assign_coords(time=[0]))
    handshake_time(
        actual, "lead_time", step=np.timedelta64(6, "h"), minimum=np.timedelta64(0, "h")
    )
    with pytest.raises(ValueError, match="align"):
        handshake_time(actual, "lead_time", step=np.timedelta64(1, "D"))
    with pytest.raises(ValueError, match="at least"):
        handshake_time(actual, "lead_time", minimum=np.timedelta64(12, "h"))
    scalar = actual.isel(time=0)
    handshake_time(
        actual, minimum=np.datetime64("2020-01-01"), maximum=np.datetime64("2020-01-02")
    )
    with pytest.raises(ValueError, match="before"):
        handshake_time(actual, maximum=np.datetime64("2020-01-01"))
    handshake_time(scalar, dimension=False)
    with pytest.raises(ValueError):
        handshake_time(scalar)


def test_handshake_healpix_metadata():
    from earth2studio.grids import HEALPixGrid

    signature = coord_array(("hpx",), grid=HEALPixGrid(0, ordering="nested"))
    handshake_dataarray(signature, signature)
    changed = signature.copy(deep=False)
    changed.attrs = {**signature.attrs, "ordering": "ring"}
    handshake_dataarray(changed, signature)
    with pytest.raises(ValueError, match="metadata"):
        handshake_metadata(changed, signature, ("ordering",))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_map_nearest(device):
    coords = OrderedDict(
        [("variable", np.array(["a", "b", "c"])), ("lat", np.array([1, 2, 3]))]
    )
    data = torch.randn(3, 3).to(device)

    # No change
    out, outc = map_coords(data, coords, coords)
    assert torch.allclose(out, data)
    assert np.all(outc["variable"] == outc["variable"])

    # Select slice in 1D
    out, outc = map_coords(data, coords, OrderedDict([("variable", np.array(["a"]))]))
    assert torch.allclose(out, data[:1])
    assert np.all(outc["variable"] == np.array(["a"]))

    # Select slice in 1D
    out, outc = map_coords(
        data, coords, OrderedDict([("batch", None), ("variable", np.array(["b", "c"]))])
    )
    assert torch.allclose(out, data[1:])
    assert np.all(outc["variable"] == np.array(["b", "c"]))

    # Select slice in 2D
    out, outc = map_coords(
        data,
        coords,
        OrderedDict([("variable", np.array(["b", "c"])), ("lat", np.array([1]))]),
    )
    assert torch.allclose(out, data[1:, :1])

    # Select index 1D
    out, outc = map_coords(data, coords, OrderedDict([("lat", np.array([1, 3]))]))
    assert torch.allclose(out, torch.cat([data[:, :1], data[:, 2:]], dim=-1))
    assert np.all(outc["lat"] == np.array([1, 3]))

    # Select index 2D
    out, outc = map_coords(
        data,
        coords,
        OrderedDict([("variable", np.array(["a", "c"])), ("lat", np.array([1, 3]))]),
    )
    assert out.shape == torch.Size((2, 2))
    assert np.all(outc["variable"] == np.array(["a", "c"]))
    assert np.all(outc["lat"] == np.array([1, 3]))

    # Select index 1D reverse
    out, outc = map_coords(
        data, coords, OrderedDict([("variable", np.array(["c", "a"]))])
    )
    truth = torch.cat((data[-1:], data[:1]), dim=0)
    assert torch.allclose(out, truth)
    assert np.all(outc["variable"] == np.array(["c", "a"]))

    out, outc = map_coords(
        data,
        coords,
        OrderedDict([("variable", np.array(["b", "c"])), ("lat", np.array([1, 2]))]),
    )
    assert torch.allclose(out, data[1:, :2])

    out, outc = map_coords(
        data,
        coords,
        OrderedDict(
            [("variable", np.array(["b", "c"])), ("lat", np.array([1.1, 2.4]))]
        ),
    )
    assert torch.allclose(out, data[1:, :2])

    out, outc = map_coords(
        data,
        coords,
        OrderedDict([("variable", np.array(["c"])), ("lat", np.array([1.8, 2.4]))]),
    )
    assert torch.allclose(out, torch.stack([data[2:, 1], data[2:, 1]], dim=1))
    # Test out of bounds of coordinate system
    out, outc = map_coords(data, coords, OrderedDict([("lat", np.array([1.8, 4.0]))]))
    assert torch.allclose(out, data[:, 1:])

    out, outc = map_coords(data, coords, OrderedDict([("lat", np.array([-0.1, 1.6]))]))
    assert torch.allclose(out, data[:, :2])


def test_map_roll_condition():
    input_coords = OrderedDict({"lon": np.array([0, 90, 180, 270])})
    output_coords = OrderedDict({"lon": np.array([180, 270, 0, 90])})
    x = torch.arange(4).float()

    mapped_x, mapped_coords = map_coords(x, input_coords, output_coords)
    expected_x = torch.tensor([2, 3, 0, 1], dtype=torch.float32)
    expected_coords = OrderedDict({"lon": np.array([180, 270, 0, 90])})

    assert torch.equal(mapped_x, expected_x)
    assert np.all(mapped_coords["lon"] == expected_coords["lon"])

    input_coords = OrderedDict({"lat": np.array([0, 30, 60, 90])})
    output_coords = OrderedDict({"lat": np.array([60, 90, 0, 30])})
    x = torch.arange(4).float()

    mapped_x, mapped_coords = map_coords(x, input_coords, output_coords)
    expected_x = torch.tensor([2, 3, 0, 1], dtype=torch.float32)
    expected_coords = OrderedDict({"lat": np.array([60, 90, 0, 30])})

    assert torch.equal(mapped_x, expected_x)
    assert np.all(mapped_coords["lat"] == expected_coords["lat"])

    input_coords = OrderedDict(
        {"lon": np.array([0, 90, 180, 270]), "lat": np.array([0, 30, 60, 90])}
    )
    output_coords = OrderedDict(
        {"lon": np.array([180, 270, 0, 90]), "lat": np.array([60, 90, 0, 30])}
    )
    x = torch.arange(16).reshape(4, 4).float()

    mapped_x, mapped_coords = map_coords(x, input_coords, output_coords)
    expected_x = torch.tensor(
        [[10, 11, 8, 9], [14, 15, 12, 13], [2, 3, 0, 1], [6, 7, 4, 5]],
        dtype=torch.float32,
    )
    expected_coords = OrderedDict(
        {"lon": np.array([180, 270, 0, 90]), "lat": np.array([60, 90, 0, 30])}
    )

    assert torch.equal(mapped_x, expected_x)
    assert np.all(mapped_coords["lon"] == expected_coords["lon"])
    assert np.all(mapped_coords["lat"] == expected_coords["lat"])


def test_map_errors():
    coords = OrderedDict(
        [("variable", np.array(["a", "b", "c"])), ("lat", np.array([1, 2, 3]))]
    )
    data = torch.arange(0, 9).reshape((3, 3))

    with pytest.raises(KeyError):
        map_coords(data, coords, OrderedDict([("foo", np.array(["c"]))]))

    with pytest.raises(ValueError):
        map_coords(data, coords, OrderedDict([("variable", np.array(["d"]))]))

    curv_coords = OrderedDict(
        [
            ("variable", np.array(["a", "b", "c"])),
            ("lat", np.array([[1, 2, 3], [4, 5, 6]])),
        ]
    )
    with pytest.raises(ValueError):
        map_coords(data, coords, curv_coords)


def check_coord_equivalence(a: OrderedDict, b: OrderedDict) -> None:
    for ka, kb in zip(a, b):
        assert np.allclose(a[ka], b[ka])


def test_convert_multidim_to_singledim():
    lat = np.linspace(0, 1, 20)
    lon = np.linspace(0, 1, 40)

    LON, LAT = np.meshgrid(lon, lat)

    dc = OrderedDict(dict(lat=LAT, lon=LON))

    true_converted = OrderedDict(dict(ilat=np.arange(20), ilon=np.arange(40)))

    # Test simple case
    c = dc
    out, _ = convert_multidim_to_singledim(c)
    check_coord_equivalence(out, true_converted)

    # Test with leading coordinates
    c = OrderedDict(
        {
            "e": np.arange(1),
            "d": np.arange(4),
        }
    )
    out, _ = convert_multidim_to_singledim(c | dc)
    check_coord_equivalence(out, c | true_converted)

    # Test with training coordinates
    out, _ = convert_multidim_to_singledim(dc | c)
    check_coord_equivalence(out, true_converted | c)

    # Test with multiple multi-dim coordinates
    dc1 = OrderedDict(dict(lat1=LAT, lon1=LON))
    true_converted2 = OrderedDict(dict(ilat1=np.arange(20), ilon1=np.arange(40)))
    out, _ = convert_multidim_to_singledim(dc | dc1)
    check_coord_equivalence(out, true_converted | true_converted2)

    out, _ = convert_multidim_to_singledim(dc | c | dc1)
    check_coord_equivalence(out, true_converted | c | true_converted2)

    # Test with 3 dims
    ff = np.linspace(0, 1, 5)
    LON, LAT, ff = np.meshgrid(lon, lat, ff)
    dc = OrderedDict(dict(lat=LAT, lon=LON, ff=ff))
    true_converted = OrderedDict(
        dict(ilat=np.arange(20), ilon=np.arange(40), iff=np.arange(5))
    )
    out, mapping = convert_multidim_to_singledim(dc)
    check_coord_equivalence(out, true_converted)

    assert mapping["lat"] == ["ilat", "ilon", "iff"]
    assert mapping["lon"] == ["ilat", "ilon", "iff"]
    assert mapping["ff"] == ["ilat", "ilon", "iff"]


def test_convert_multidim_to_singledim_error():
    lat = np.linspace(0, 1, 20)
    lon = np.linspace(0, 1, 40)

    _, LAT = np.meshgrid(lon, lat)

    dc = OrderedDict(
        dict(
            lat=LAT,
        )
    )

    with pytest.raises(ValueError):
        convert_multidim_to_singledim(dc)

    dc = OrderedDict(dict(lat=LAT, e=np.arange(40)))

    with pytest.raises(ValueError):
        convert_multidim_to_singledim(dc)


def test_handshake_coords():
    """Test handshake_coords function"""
    coords1 = OrderedDict(
        [
            ("lat", np.array([1, 2, 3])),
            ("lon", np.array([4, 5, 6])),
        ]
    )
    coords2 = OrderedDict(
        [
            ("lat", np.array([1, 2, 3])),
            ("lon", np.array([4, 5, 6])),
        ]
    )

    # Should pass
    handshake_coords(coords1, coords2, "lat")
    handshake_coords(coords1, coords2, ["lat", "lon"])

    # Test missing dimension in input coords
    with pytest.raises(KeyError, match="not found in input coordinates"):
        handshake_coords(coords1, coords2, "variable")

    # Test missing dimension in target coords
    coords2 = OrderedDict([("lon", np.array([1, 2, 3]))])
    with pytest.raises(KeyError, match="not found in target coordinates"):
        handshake_coords(coords1, coords2, "lat")

    # Test different shapes
    coords2 = OrderedDict([("lat", np.array([1, 2, 3, 4]))])
    with pytest.raises(ValueError, match="are not the same"):
        handshake_coords(coords1, coords2, "lat")

    # Test different values
    coords2 = OrderedDict([("lat", np.array([1, 2, 4]))])
    with pytest.raises(ValueError, match="are not the same"):
        handshake_coords(coords1, coords2, "lat")


def test_handshake_size():
    """Test handshake_size function"""
    coords = OrderedDict([("lat", np.array([1, 2, 3]))])

    # Should pass
    handshake_size(coords, "lat", 3)

    # Test missing dimension
    with pytest.raises(KeyError, match="not found in input coordinates"):
        handshake_size(coords, "lon", 3)

    # Test wrong size
    with pytest.raises(ValueError, match="is not of size"):
        handshake_size(coords, "lat", 4)


def test_map_coords_additional():
    """Test additional map_coords scenarios"""
    # Test time coordinate handling
    coords = OrderedDict([("time", np.array([1, 2, 3])), ("lat", np.array([1, 2, 3]))])
    data = torch.randn(3, 3)
    output_coords = OrderedDict([("lat", np.array([1.5, 2]))])
    out, outc = map_coords(data, coords, output_coords)
    assert out.shape == (3, 2)

    # Test unsupported method
    with pytest.raises(ValueError, match="not supported"):
        map_coords(data, coords, output_coords, method="quadratic")

    # Test non-numeric coordinate error
    coords = OrderedDict([("var", np.array(["a", "b", "c"]))])
    output_coords = OrderedDict([("var", np.array(["d", "e"]))])
    with pytest.raises(ValueError, match="must be in the input coordinates"):
        map_coords(torch.randn(3), coords, output_coords)


def test_split_coords():
    """Test split_coords function"""
    x = torch.randn(2, 3, 4)
    coords = OrderedDict(
        [
            ("batch", np.array([0, 1])),
            ("variable", np.array(["a", "b", "c"])),
            ("time", np.array([1, 2, 3, 4])),
        ]
    )

    # Test normal split
    xs, reduced_coords, values = split_coords(x, coords, "variable")
    assert len(xs) == 3
    assert all(t.shape == (2, 4) for t in xs)
    assert "variable" not in reduced_coords
    assert np.array_equal(values, np.array(["a", "b", "c"]))

    # Test invalid dimension
    with pytest.raises(ValueError, match="is not in coords"):
        split_coords(x, coords, "invalid_dim")


def test_convert_multidim_to_singledim_additional():
    """Test additional convert_multidim_to_singledim scenarios"""
    # Test incomplete multidimensional coordinates
    lat = np.linspace(0, 1, 20)
    lon = np.linspace(0, 1, 40)
    LON, LAT = np.meshgrid(lon, lat)

    # Missing matching coordinate
    coords = OrderedDict([("lat", LAT)])
    with pytest.raises(
        ValueError, match="Assumed that if an n-dimensional coordinate exists"
    ):
        convert_multidim_to_singledim(coords)

    # Test mismatched shapes
    coords = OrderedDict([("lat", LAT), ("lon", np.zeros((30, 50)))])  # Different shape
    with pytest.raises(
        ValueError, match="Assumed that if an n-dimensional coordinate exists"
    ):
        convert_multidim_to_singledim(coords)

    # Test with 3D coordinates but missing matching coordinates
    ff = np.linspace(0, 1, 5)
    LON, LAT, FF = np.meshgrid(lon, lat, ff)
    coords = OrderedDict(
        [
            ("lat", LAT),
            ("lon", LON),
            # Missing FF coordinate
        ]
    )
    with pytest.raises(
        ValueError, match="Assumed that if an n-dimensional coordinate exists"
    ):
        convert_multidim_to_singledim(coords)


def test_tile_coords():
    """Test tiling function for expanding dimensions"""

    x = torch.randn(2, 721, 1440)
    coords = OrderedDict(
        {
            "variable": np.array(["z", "lsm"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    target_coords = OrderedDict(
        {
            "ensemble": np.array([0, 1, 2]),
            "time": np.array([1, 2, 3, 4]),
            "lead_time": np.array([0, 1, 2, 3, 4]),
            "variable": np.array(["z", "lsm"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    result, result_coords = tile_coords(x, coords, target_coords)
    assert result.shape == (3, 4, 5, 2, 721, 1440)
    assert "variable" in result_coords
    assert "time" in result_coords


def test_tile_coords_ignores_common_dims():
    x = torch.randn(2, 721, 1440)
    coords = OrderedDict(
        {
            "variable": np.array(["z", "lsm"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    # target_coords has "variable", "lat", "lon" which should be ignored
    # Only "ensemble", "time", "lead_time" should be used for tiling
    target_coords = OrderedDict(
        {
            "ensemble": np.array([0, 1, 2]),
            "time": np.array([1, 2, 3, 4]),
            "lead_time": np.array([0, 1, 2, 3, 4]),
            "variable": np.array(["different", "values"]),  # Should be ignored
            "lat": np.linspace(0, 90, 100),  # Should be ignored
            "lon": np.linspace(0, 180, 200),  # Should be ignored
        }
    )

    result, result_coords = tile_coords(x, coords, target_coords)
    # Should tile to (3, 4, 5, 2, 721, 1440) - leading dims from target_coords (not in coords)
    assert result.shape == (3, 4, 5, 2, 721, 1440)
    # Should use coords values, not target_coords values for common dims
    assert np.array_equal(result_coords["variable"], coords["variable"])
    assert np.array_equal(result_coords["lat"], coords["lat"])
    assert np.array_equal(result_coords["lon"], coords["lon"])


def test_tile_coords_edge_cases():
    # Test: no leading dimensions (all target_coords dims exist in coords)
    x = torch.randn(2, 3, 4)
    coords = OrderedDict(
        {
            "a": np.array([0, 1]),
            "b": np.array([0, 1, 2]),
            "c": np.array([0, 1, 2, 3]),
        }
    )
    target_coords = OrderedDict(
        {
            "a": np.array([0, 1]),
            "b": np.array([0, 1, 2]),
            "c": np.array([0, 1, 2, 3]),
        }
    )
    result, result_coords = tile_coords(x, coords, target_coords)
    assert result.shape == x.shape
    assert result_coords == coords

    # Test: common dimensions in target_coords are ignored (uses coords values)
    x = torch.randn(2, 721, 1440)
    coords = OrderedDict(
        {
            "variable": np.array(["z", "lsm"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )
    target_coords = OrderedDict(
        {
            "ensemble": np.array([0, 1, 2]),
            "time": np.array([1, 2, 3, 4]),
            "variable": np.array(["different"]),  # Should be ignored
            "lat": np.linspace(0, 90, 100),  # Should be ignored
            "lon": np.linspace(0, 180, 200),  # Should be ignored
        }
    )
    # Only "ensemble" and "time" from target_coords should be used for tiling
    result, result_coords = tile_coords(x, coords, target_coords)
    assert result.shape == (3, 4, 2, 721, 1440)
    assert np.array_equal(result_coords["variable"], coords["variable"])
    assert np.array_equal(result_coords["lat"], coords["lat"])
    assert np.array_equal(result_coords["lon"], coords["lon"])

    # Test: single dimension x
    x = torch.randn(721)
    coords = OrderedDict({"lat": np.linspace(90, -90, 721)})
    target_coords = OrderedDict(
        {
            "ensemble": np.array([0, 1, 2]),
            "time": np.array([1, 2, 3, 4]),
            "lat": np.linspace(0, 90, 100),  # Should be ignored
        }
    )
    result, result_coords = tile_coords(x, coords, target_coords)
    assert result.shape == (3, 4, 721)
    assert "lat" in result_coords
    assert np.array_equal(result_coords["lat"], coords["lat"])

    # Test: empty tensors (0-sized dimensions)
    x = torch.randn(0, 721, 1440)
    coords = OrderedDict(
        {
            "variable": np.array([]),  # Empty array
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )
    target_coords = OrderedDict(
        {
            "ensemble": np.array([0, 1, 2]),
            "time": np.array([1, 2, 3, 4]),
            "variable": np.array([1, 2, 3]),  # Should be ignored
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )
    result, result_coords = tile_coords(x, coords, target_coords)
    assert result.shape == (3, 4, 0, 721, 1440)
    assert len(result_coords["variable"]) == 0

    # Test: mismatched coordinate values (common dims use coords values)
    x = torch.randn(2, 10)
    coords = OrderedDict(
        {
            "a": np.array([0, 1]),
            "b": np.arange(10),
        }
    )
    target_coords = OrderedDict(
        {
            "c": np.array([0, 1, 2]),
            "a": np.array([10, 20]),  # Should be ignored
            "b": np.arange(10, 20),  # Should be ignored
        }
    )
    result, result_coords = tile_coords(x, coords, target_coords)
    assert result.shape == (3, 2, 10)
    assert np.array_equal(result_coords["a"], coords["a"])
    assert np.array_equal(result_coords["b"], coords["b"])


def test_tile_coords_validation():
    """Test that tile_coords validates new dims lead common dims"""
    x = torch.randn(2, 10)
    coords = OrderedDict(
        {
            "a": np.array([0, 1]),
            "b": np.arange(10),
        }
    )

    # Valid: new dims before common dims
    target_coords = OrderedDict(
        {
            "c": np.array([0, 1, 2]),
            "a": np.array([10, 20]),
            "b": np.arange(10, 20),
        }
    )
    result, _ = tile_coords(x, coords, target_coords)
    assert result.shape == (3, 2, 10)

    # Invalid: new dim after common dim
    target_coords_bad = OrderedDict(
        {
            "a": np.array([10, 20]),
            "c": np.array([0, 1, 2]),  # New dim after common dim
            "b": np.arange(10, 20),
        }
    )
    with pytest.raises(ValueError):
        tile_coords(x, coords, target_coords_bad)


def test_cat_coords():
    xx = torch.randn(1, 2, 721, 1440)
    cox = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["u10m", "v10m"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    yy = torch.randn(1, 1, 721, 1440)
    coy = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    result, result_coords = cat_coords((xx, yy), (cox, coy), dim="variable")

    assert result.shape == (1, 3, 721, 1440)
    assert len(result_coords["variable"]) == 3
    assert np.array_equal(result_coords["variable"], ["u10m", "v10m", "msl"])


def test_cat_coords_different_dims():
    """Test concatenation along different dimensions"""
    # Test concatenation along "time" dimension
    xx = torch.randn(2, 3, 721, 1440)
    cox = OrderedDict(
        {
            "time": np.array([0, 1]),
            "variable": np.array(["u10m", "v10m", "msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    yy = torch.randn(1, 3, 721, 1440)
    coy = OrderedDict(
        {
            "time": np.array([2]),
            "variable": np.array(["u10m", "v10m", "msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    result, result_coords = cat_coords((xx, yy), (cox, coy), dim="time")
    assert result.shape == (3, 3, 721, 1440)
    assert len(result_coords["time"]) == 3
    assert np.array_equal(result_coords["time"], [0, 1, 2])

    # Test concatenation along "batch" dimension
    xx = torch.randn(2, 3, 721, 1440)
    cox = OrderedDict(
        {
            "batch": np.array([0, 1]),
            "variable": np.array(["u10m", "v10m", "msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    yy = torch.randn(1, 3, 721, 1440)
    coy = OrderedDict(
        {
            "batch": np.array([2]),
            "variable": np.array(["u10m", "v10m", "msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    result, result_coords = cat_coords((xx, yy), (cox, coy), dim="batch")
    assert result.shape == (3, 3, 721, 1440)
    assert len(result_coords["batch"]) == 3
    assert np.array_equal(result_coords["batch"], [0, 1, 2])


def test_cat_coords_multiple():
    xx = torch.randn(1, 1, 721, 1440)
    cox = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["u10m"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    yy = torch.randn(1, 1, 721, 1440)
    coy = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["v10m"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    zz = torch.randn(1, 1, 721, 1440)
    coz = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    result, result_coords = cat_coords((xx, yy, zz), (cox, coy, coz), dim="variable")
    assert result.shape == (1, 3, 721, 1440)
    assert len(result_coords["variable"]) == 3
    assert np.array_equal(result_coords["variable"], ["u10m", "v10m", "msl"])


def test_cat_coords_single():
    """Test concatenation with a single tensor (should return as-is)"""
    xx = torch.randn(1, 2, 721, 1440)
    cox = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["u10m", "v10m"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    result, result_coords = cat_coords((xx,), (cox,), dim="variable")
    assert torch.equal(result, xx)  # Should return the same tensor
    for dim in cox.keys():
        assert np.all(result_coords[dim] == cox[dim])


def test_cat_coords_errors():
    """Test error cases for cat_coords"""
    # Test length mismatch
    xx = torch.randn(1, 2, 721, 1440)
    cox = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["u10m", "v10m"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )
    yy = torch.randn(1, 1, 721, 1440)
    coy = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )
    with pytest.raises(ValueError):
        cat_coords((xx, yy), (cox,), dim="variable")

    # Test empty input
    with pytest.raises(ValueError):
        cat_coords((), (), dim="variable")

    xx = torch.randn(1, 2, 721, 1440)
    cox = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["u10m", "v10m"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    yy = torch.randn(1, 1, 721, 1440)
    coy = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    # Test missing dimension in first coords (handshake_dim on cox)
    with pytest.raises(KeyError):
        cat_coords((xx, yy), (cox, coy), dim="nonexistent")

    # Test cox has extra dim (key-equality check catches mismatched dimension sets)
    cox_with_extra = cox.copy()
    cox_with_extra["extra_dim"] = np.array([0])
    xx_extra = torch.randn(1, 2, 1, 721, 1440)
    with pytest.raises(KeyError):
        cat_coords((xx_extra, yy), (cox_with_extra, coy), dim="extra_dim")

    # Test mismatched non-cat coords (handshake_coords catches shape mismatch)
    yy_bad = torch.randn(1, 1, 361, 1440)
    coy_bad = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["msl"]),
            "lat": np.linspace(90, -90, 361),
            "lon": np.linspace(0, 360, 1440),
        }
    )
    with pytest.raises(ValueError):
        cat_coords((xx, yy_bad), (cox, coy_bad), dim="variable")

    # Test mismatched dimension names
    yy_bad = torch.randn(1, 1, 361, 1440)
    coy_bad = OrderedDict(
        {
            "lead_time": np.array([0]),
            "variable": np.array(["msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )
    with pytest.raises(ValueError):
        cat_coords((xx, yy_bad), (cox, coy_bad), dim="variable")


def test_coordinate_system_signature():
    signature = coord_array(
        ("batch", "lead_time", "variable", "x"),
        {
            "lead_time": [np.timedelta64(0, "h")],
            "variable": ["a:mean:24h"],
            "x": [0, 1],
        },
        dynamic=("batch",),
    )
    assert signature.shape == (0, 1, 1, 2)
    assert signature.data.nbytes == 0
    assert signature.attrs["earth2studio_dynamic_dims"] == ("batch",)
    assert (
        signature.attrs["earth2studio_statistics"]["a:mean:24h"]["modifier"]
        == "mean:24h"
    )

    array = xr.DataArray(
        np.zeros((3, 1, 1, 2)),
        dims=("time", "lead_time", "variable", "x"),
        coords={
            "lead_time": signature.lead_time,
            "variable": ["a:mean:24h"],
            "x": [0, 1],
        },
        attrs=signature.attrs,
    )
    handshake_dataarray(array, signature)
    with pytest.raises(ValueError, match="trailing dimensions"):
        handshake_dataarray(
            array.transpose("time", "lead_time", "x", "variable"), signature
        )


def test_coordinate_system_grid_and_collection():
    signature = coord_array(
        ("batch", "lead_time", "variable", "lat", "lon"),
        {"lead_time": [np.timedelta64(0, "h")], "variable": ["a"]},
        dynamic=("batch",),
        grid="latlon-0.25deg-south-pole-excluded",
    )
    assert signature.shape == (0, 1, 1, 720, 1440)
    assert (
        signature.attrs["earth2studio_grid_id"] == "latlon-0.25deg-south-pole-excluded"
    )
    array = xr.DataArray(
        np.zeros((1, 1, 1, 720, 1440), dtype=np.float32),
        dims=signature.dims,
        coords=signature.coords,
        attrs=signature.attrs,
    )
    handshake_dataarrays((array, array), (signature, signature))
    with pytest.raises(ValueError, match="Expected 2 DataArrays"):
        handshake_dataarrays((array,), (signature, signature))


def test_coordinate_projected_grid_metadata():
    from earth2studio.grids import E2S_CRS, ProjectedGrid, infer_grid

    grid = ProjectedGrid(np.arange(2) * 1000, np.arange(3) * 1000, "EPSG:3857")
    signature = coord_array(
        ("y", "x"),
        grid=grid,
        attrs={"earth2studio_grid_id": "source-only", E2S_CRS: "EPSG:4326"},
    )
    assert "earth2studio_grid_id" not in signature.attrs
    assert signature.attrs[E2S_CRS] == grid.crs.to_string()
    assert infer_grid(signature).fingerprint() == grid.fingerprint()
    named = coord_array(
        ("hrrr_y", "hrrr_x"),
        grid=grid,
        grid_dims={"y": "hrrr_y", "x": "hrrr_x"},
    )
    assert named.attrs["dims"] == ["hrrr_y", "hrrr_x"]
    xr.testing.assert_identical(named.rename(hrrr_y="y", hrrr_x="x").y, signature.y)


def test_coordinate_auxiliary_geometry_validation():
    from earth2studio.grids import CurvilinearGrid

    signature = coord_array(
        ("y", "x"), grid=CurvilinearGrid(np.zeros((2, 3)), np.ones((2, 3)))
    )
    with pytest.raises(ValueError, match="lat"):
        handshake_dataarray(signature.assign_coords(lat=signature.lat + 1), signature)
    with pytest.raises(ValueError, match="lon"):
        handshake_dataarray(signature.drop_vars("lon"), signature)


def test_coordinate_array_like_replaces_dependent_coordinates():
    from earth2studio.utils.coords import coord_array_like

    array = xr.DataArray(
        np.zeros((2, 2, 3)),
        dims=("member", "variable", "x"),
        coords={
            "member": [0, 1],
            "variable": ["a", "b"],
            "x": [0, 1, 2],
            "units": ("variable", ["K", "m"]),
            "height": ("x", [1, 2, 3]),
        },
        name="forecast",
        attrs={"source": "test", "earth2studio_dynamic_dims": ("batch",)},
    )
    output = coord_array_like(array, {"variable": ["tp:sum:6h"]})
    assert output.shape == (2, 1, 3)
    assert output.data.nbytes == 0
    assert output.name == array.name
    assert output.attrs["source"] == "test"
    assert output.attrs["earth2studio_dynamic_dims"] == ()
    assert "units" not in output.coords
    assert output.attrs["earth2studio_statistics"]["tp:sum:6h"]["modifier"] == "sum:6h"
    xr.testing.assert_identical(output.height, array.height)
    assert "units" in array.coords
    copied = coord_array_like(output)
    assert copied.attrs == output.attrs
    replaced = coord_array_like(output, {"variable": ["other"]})
    assert "earth2studio_statistics" not in replaced.attrs


def test_coordinate_statistics_derived_from_labels() -> None:
    signature = coord_array(
        ("variable",),
        {"variable": ["tp:sum:6h", "t2m", "tp:mean:1day"]},
        attrs={"earth2studio_statistics": {"tp:sum:6h": "mean:12h"}},
    )
    statistics = signature.attrs["earth2studio_statistics"]
    assert set(statistics) == {"tp:sum:6h", "tp:mean:1day"}
    assert statistics["tp:sum:6h"]["modifier"] == "sum:6h"
    assert statistics["tp:mean:1day"]["modifier"] == "mean:24h"
    np.testing.assert_array_equal(
        signature.coords["variable"], ["tp:sum:6h", "t2m", "tp:mean:1day"]
    )


@pytest.mark.parametrize(
    "dynamic", [("time",), ("sample", "sensor"), ("batch", "time", "member")]
)
def test_coordinate_dynamic_dimensions_are_model_independent(
    dynamic: tuple[str, ...],
) -> None:
    signature = coord_array(
        (*dynamic, "variable", "x"), {"variable": ["t2m"], "x": [0, 1]}, dynamic=dynamic
    )
    concrete = coord_array(
        (*dynamic, "variable", "x"),
        {
            **{
                dim: (
                    np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
                    if dim == "time"
                    else [0, 1]
                )
                for dim in dynamic
            },
            "variable": ["t2m"],
            "x": [0, 1],
        },
    )
    handshake_dataarray(concrete, signature)
    assert signature.attrs["earth2studio_dynamic_dims"] == dynamic
    assert signature.data.nbytes == 0


@pytest.mark.parametrize(
    "labels",
    [["tp:sum:6h", "tp:sum:6h"], ["tp:mean:1month"], ["tp:unknown:6h"], [":sum:6h"]],
)
def test_coordinate_invalid_statistic_labels(labels: list[str]) -> None:
    with pytest.raises(ValueError):
        coord_array(("variable",), {"variable": labels})


def test_coordinate_array_like_rejects_implicit_resize():
    from earth2studio.utils.coords import coord_array_like

    array = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"))
    with pytest.raises(ValueError, match="size"):
        coord_array_like(array, {"height": ("x", [4, 5])})


def test_coordinate_array_like_partial_dynamic_dimensions():
    from earth2studio.utils.coords import coord_array_like

    signature = coord_array(
        ("batch", "time", "x"), {"x": [0, 1]}, dynamic=("batch", "time")
    )
    # Resolve the rightmost wildcard first to preserve a leading dynamic prefix.
    partial = coord_array_like(signature, {"time": [np.datetime64("2026-09-17")]})
    assert partial.attrs["earth2studio_dynamic_dims"] == ("batch",)
    assert partial.shape == (0, 1, 2)
    concrete = coord_array_like(partial, {"batch": [0, 1]})
    assert concrete.attrs["earth2studio_dynamic_dims"] == ()
    assert concrete.shape == (2, 1, 2)
    with pytest.raises(ValueError, match="Dynamic dimensions must lead"):
        coord_array_like(signature, {"batch": [0, 1]})


def test_coordinate_array_like_spatial_replacement_requires_new_grid():
    from earth2studio.grids import CurvilinearGrid, infer_grid
    from earth2studio.utils.coords import coord_array_like

    signature = coord_array(("lat", "lon"), grid="latlon-0.25deg-south-pole-excluded")
    with pytest.raises(ValueError, match="grid"):
        coord_array_like(signature, {"lat": np.asarray(signature.lat) + 1})
    with pytest.raises(ValueError, match="grid"):
        coord_array_like(signature, {"lon": np.asarray(signature.lon)[:3]})

    # A custom crop must not inherit the source registry identity or CRS.
    grid = CurvilinearGrid(np.zeros((2, 3)), np.ones((2, 3)))
    cropped = coord_array(("y", "x"), grid=grid, attrs=signature.attrs)
    assert "earth2studio_grid_id" not in cropped.attrs
    assert "earth2studio_crs" not in cropped.attrs
    assert "crs" not in cropped.attrs
    assert infer_grid(cropped).fingerprint() == grid.fingerprint()
    assert "earth2studio_grid_id" in signature.attrs

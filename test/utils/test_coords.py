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

import numpy as np
import pytest
import torch

from earth2studio.utils import (
    convert_multidim_to_singledim,
    handshake_coords,
    handshake_dim,
    handshake_size,
)
from earth2studio.utils.coords import map_coords, split_coords


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
    coords1 = OrderedDict([("lat", np.array([1, 2, 3]))])
    coords2 = OrderedDict([("lat", np.array([1, 2, 3]))])

    # Should pass
    handshake_coords(coords1, coords2, "lat")

    # Test missing dimension in input coords
    with pytest.raises(KeyError, match="not found in input coordinates"):
        handshake_coords(coords1, coords2, "lon")

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

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import os
import tempfile
from collections import OrderedDict
from importlib.metadata import version

import fsspec
import numpy as np
import pytest
import torch
import zarr

try:
    zarr_version = version("zarr")
    zarr_major_version = int(zarr_version.split(".")[0])
except Exception:
    zarr_major_version = 2
if zarr_major_version < 3:
    pytest.skip("Zarr version 2 not supported")

from earth2studio.io import AsyncZarrBackend
from earth2studio.utils.coords import convert_multidim_to_singledim, split_coords


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [
            np.datetime64("1971-06-01T06:00:00"),
            np.datetime64("2021-11-23T18:00:00"),
            np.datetime64("2021-11-24T00:00:00"),
        ],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize(
    "fs",
    [fsspec.filesystem("memory"), fsspec.filesystem("file")],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_async_zarr_write(
    time: list[np.datetime64],
    variable: list[str],
    fs: fsspec.spec.AbstractFileSystem,
    device: str,
    tmp_path: str,
) -> None:

    index_coords = {
        "time": np.asarray(time),
    }
    z = AsyncZarrBackend(f"{tmp_path}/output.zarr", index_coords=index_coords, fs=fs)

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    for i, time0 in enumerate(time):
        total_coords["time"] = np.array([time0])
        z.write(x[i : i + 1], total_coords, "fields_1")
        assert "fields_1" in z.zs
        assert z.zs["fields_1"].shape == x.shape
        assert np.allclose(z.zs["fields_1"][i], x[i].to("cpu").numpy())
    z.close()
    assert np.allclose(z.zs["fields_1"], x.to("cpu").numpy())

    total_coords = OrderedDict(
        {
            "variable": np.asarray(variable),
            "time": np.asarray(time),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    for i, time0 in enumerate(time):
        total_coords["time"] = np.array([time0])
        z.write(x[:, i : i + 1], total_coords, "fields_2")
        assert "fields_2" in z.zs
        assert z.zs["fields_2"].shape == x.shape
        assert np.allclose(z.zs["fields_2"][:, i], x[:, i].to("cpu").numpy())
    z.close()
    assert np.allclose(z.zs["fields_2"], x.to("cpu").numpy())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fs",
    [fsspec.filesystem("memory"), fsspec.filesystem("file")],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_async_write(
    fs: fsspec.spec.AbstractFileSystem, device: str, tmp_path: str
) -> None:
    time = [
        np.datetime64("1971-06-01T06:00:00"),
        np.datetime64("2021-11-23T18:00:00"),
        np.datetime64("2021-11-24T00:00:00"),
    ]
    index_coords = {
        "time": np.asarray(time),
    }
    z = AsyncZarrBackend(f"{tmp_path}/output.zarr", index_coords=index_coords, fs=fs)

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(["t2m", "tcwv"]),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    for i, time0 in enumerate(time):
        total_coords["time"] = np.array([time0])
        await z.async_write(x[i : i + 1], total_coords, "fields_1")
        assert "fields_1" in z.zs
        assert z.zs["fields_1"].shape == x.shape
        assert np.allclose(z.zs["fields_1"][i], x[i].to("cpu").numpy())
    z.close()
    assert np.allclose(z.zs["fields_1"], x.to("cpu").numpy())


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "time",
#     [
#         [np.datetime64("1958-01-31")],
#         [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
#     ],
# )
# @pytest.mark.parametrize(
#     "variable",
#     [["t2m"], ["t2m", "tcwv"]],
# )
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# async def test_zarr_variable(
#     time: list[np.datetime64], variable: list[str], device: str
# ) -> None:

#     total_coords = OrderedDict(
#         {
#             "time": np.asarray(time),
#             "variable": np.asarray(variable),
#             "lat": np.linspace(-90, 90, 180),
#             "lon": np.linspace(0, 360, 360, endpoint=False),
#         }
#     )

#     # Remove var names
#     coords = total_coords.copy()
#     var_names = coords.pop("variable")

#     # Test Memory Store
#     z = AsyncZarrBackend("memory://", index_coords=coords)
#     assert isinstance(z.zstore, zarr.storage.MemoryStore)

#     # Test writing
#     partial_coords = OrderedDict(
#         {
#             "time": np.asarray(time)[:1],
#             "variable": np.asarray(variable)[:1],
#             "lat": total_coords["lat"],
#             "lon": total_coords["lon"][:180],
#         }
#     )
#     partial_data = torch.randn((1, 1, 180, 180), device=device)
#     await z.async_write(*split_coords(partial_data, partial_coords, "variable"))
#     assert np.allclose(z.zs[variable[0]][0, :, :180], partial_data.to("cpu").numpy())

#     # Test Directory Store
#     with tempfile.TemporaryDirectory() as td:
#         file_name = os.path.join(td, "temp_zarr.zarr")
#         z = AsyncZarrBackend(file_name, index_coords=coords)
#         assert os.path.exists(file_name)
#         if zarr_major_version >= 3:
#             assert isinstance(z.zstore, zarr.storage.LocalStore)
#         else:
#             assert isinstance(z.zstore, zarr.storage.DirectoryStore)

#         z._initialize_arrays(coords, var_names, [np.float32] * len(var_names))
#         # Check instantiation
#         for dim in coords:
#             assert z.zs[dim].shape == coords[dim].shape

#         for var_name in var_names:
#             assert var_name in z.zs
#             assert z.zs[var_name].shape == tuple(
#                 [len(values) for values in coords.values()]
#             )

#         # Test writing
#         partial_coords = OrderedDict(
#             {
#                 "time": np.asarray(time)[:1],
#                 "variable": np.asarray(variable)[:1],
#                 "lat": total_coords["lat"],
#                 "lon": total_coords["lon"][:180],
#             }
#         )
#         partial_data = torch.randn((1, 1, 180, 180), device=device)
#         await z.async_write(*split_coords(partial_data, partial_coords, "variable"))
#         assert np.allclose(
#             z.zs[variable[0]][0, :, :180], partial_data.to("cpu").numpy()
#         )

#     # Cleanup
#     z.close()


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "overwrite",
#     [True, False],
# )
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# async def test_zarr_file(overwrite: bool, device: str, tmp_path: str) -> None:
#     time = [np.datetime64("1958-01-31T00:00:00")]
#     variable = ["t2m", "tcwv"]
#     total_coords = OrderedDict(
#         {
#             "time": np.asarray(time),
#             "variable": np.asarray(variable),
#             "lat": np.linspace(-90, 90, 180),
#             "lon": np.linspace(0, 360, 360, endpoint=False),
#         }
#     )

#     # Test File Store
#     z = AsyncZarrBackend(tmp_path / "test.zarr", index_coords=total_coords)

#     shape = tuple([len(values) for values in total_coords.values()])
#     array_name = "fields"
#     dummy = torch.randn(shape, device=device, dtype=torch.float32)
#     z._initialize_arrays(total_coords, [array_name], [np.float32])
#     await z.async_write(dummy, total_coords, array_name)

#     # Check to see if write overwrite in add array works
#     if overwrite:
#         z._initialize_arrays(total_coords, [array_name], [np.float32])
#         await z.async_write(dummy, total_coords, array_name)
#     else:
#         with pytest.raises(RuntimeError):
#             z._initialize_arrays(total_coords, [array_name], [np.float32])

#     z = AsyncZarrBackend(tmp_path / "test.zarr", index_coords=total_coords)
#     # Check to see if write overwrite in constructor allows redefinition Zarr
#     if overwrite:
#         z._initialize_arrays(total_coords, [array_name], [np.float32])
#         await z.async_write(dummy, total_coords, array_name)
#     else:
#         with pytest.raises(RuntimeError):
#             z._initialize_arrays(total_coords, [array_name], [np.float32])

#     # Cleanup
#     z.close()


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "time",
#     [
#         [np.datetime64("1958-01-31")],
#         [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
#     ],
# )
# @pytest.mark.parametrize(
#     "variable",
#     [["t2m"], ["t2m", "tcwv"]],
# )
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# async def test_zarr_exceptions(
#     time: list[np.datetime64], variable: list[str], device: str
# ) -> None:

#     total_coords = OrderedDict(
#         {
#             "time": np.asarray(time),
#             "variable": np.asarray(variable),
#             "lat": np.linspace(-90, 90, 180),
#             "lon": np.linspace(0, 360, 360, endpoint=False),
#         }
#     )

#     # Test Memory Store
#     z = AsyncZarrBackend("memory://", index_coords=total_coords)
#     assert isinstance(z.zstore, zarr.storage.MemoryStore)

#     # Test mismatch between len(array_names) and len(data)
#     shape = tuple([len(values) for values in total_coords.values()])
#     array_name = "fields"
#     dummy = torch.randn(shape, device=device, dtype=torch.float32)
#     with pytest.raises(ValueError):
#         z._initialize_arrays(total_coords, [array_name], [np.float32, np.float32])

#     # Test trying to add the same array twice.
#     z._initialize_arrays(total_coords, ["dummy_1"], [np.float32])
#     with pytest.raises(RuntimeError):
#         z._initialize_arrays(total_coords, ["dummy_1"], [np.float32])

#     # Try to write with bad coords
#     bad_coords = {"ensemble": np.arange(0)} | total_coords
#     bad_shape = (1,) + shape
#     dummy = torch.randn(bad_shape, device=device, dtype=torch.float32)
#     with pytest.raises(ValueError):
#         await z.async_write(dummy, bad_coords, "dummy_1")

#     # Try to write with too many array names
#     with pytest.raises(ValueError):
#         await z.async_write([dummy, dummy], bad_coords, "dummy_1")

#     # Cleanup
#     z.close()


# @pytest.mark.asyncio
# @pytest.mark.parametrize(
#     "time",
#     [
#         [np.datetime64("1958-01-31")],
#         [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
#     ],
# )
# @pytest.mark.parametrize(
#     "variable",
#     [["t2m"], ["t2m", "tcwv"]],
# )
# @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
# async def test_zarr_field_multidim(
#     time: list[np.datetime64], variable: list[str], device: str
# ) -> None:

#     lat = np.linspace(-90, 90, 180)
#     lon = np.linspace(0, 360, 360, endpoint=False)
#     LON, LAT = np.meshgrid(lon, lat)

#     total_coords = OrderedDict(
#         {
#             "time": np.asarray(time),
#             "variable": np.asarray(variable),
#             "lat": LAT,
#             "lon": LON,
#         }
#     )

#     adjusted_coords, _ = convert_multidim_to_singledim(total_coords)

#     # Test Memory Store
#     z = AsyncZarrBackend("memory://", index_coords=adjusted_coords)
#     assert isinstance(z.zstore, zarr.storage.MemoryStore)

#     # Instantiate
#     array_name = "fields"
#     z._initialize_arrays(adjusted_coords, [array_name], [np.float32])

#     # Check instantiation
#     for dim in adjusted_coords:
#         assert dim in z.zs
#         assert z.zs[dim].shape == adjusted_coords[dim].shape

#     # Test __contains__
#     assert array_name in z.zs

#     # Test __getitem__
#     shape = tuple([len(dim) for dim in adjusted_coords.values()])
#     assert z.zs[array_name].shape == shape

#     # Test __len__
#     assert len(z.zs) == 7

#     # Test __iter__
#     for array in z.zs:
#         assert array in ["fields", "time", "variable", "lat", "lon", "ilat", "ilon"]

#     # Test add_array with torch.Tensor
#     z._initialize_arrays(adjusted_coords, ["dummy_1"], [np.float32])
#     await z.async_write(
#         torch.randn(shape, device=device, dtype=torch.float32),
#         adjusted_coords,
#         "dummy_1",
#     )

#     assert "dummy_1" in z.zs
#     assert z.zs["dummy_1"].shape == shape

#     # Test add_array with kwarg (overwrite)
#     await z.async_write(
#         torch.randn(shape, device=device, dtype=torch.float32),
#         adjusted_coords,
#         "dummy_1",
#         overwrite=True,
#     )

#     assert "dummy_1" in z.zs
#     assert z.zs["dummy_1"].shape == shape

#     # Test add_array with list and kwarg (overwrite)
#     await z.async_write(
#         [torch.randn(shape, device=device, dtype=torch.float32)],
#         adjusted_coords,
#         "dummy_1",
#         overwrite=True,
#     )

#     assert "dummy_1" in z.zs
#     assert z.zs["dummy_1"].shape == shape

#     await z.async_write(
#         torch.randn(shape, device=device, dtype=torch.float32),
#         adjusted_coords,
#         "dummy_1",
#         overwrite=True,
#         fill_value=None,
#     )

#     assert "dummy_1" in z.zs
#     assert z.zs["dummy_1"].shape == shape

#     # Test writing

#     # Test full write
#     x = torch.randn(shape, device=device, dtype=torch.float32)
#     await z.async_write(x, adjusted_coords, "fields")

#     xx, _ = await z.async_read(adjusted_coords, "fields", device=device)
#     assert torch.allclose(x, xx)

#     # Test separate write
#     await z.async_write(x, total_coords, "fields_1")
#     assert "fields_1" in z.zs
#     assert z.zs["fields_1"].shape == x.shape

#     xx, _ = await z.async_read(total_coords, "fields_1", device=device)
#     assert torch.allclose(x, xx)

#     # Cleanup
#     z.close()

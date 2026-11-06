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

import warnings
from collections import OrderedDict
from contextvars import copy_context

import dask.array as da
import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.utils import cupy as cupy_utils
from earth2studio.utils.cupy import from_torch


def assert_values_and_gradients(actual, expected, leaf):
    torch.testing.assert_close(actual, expected, equal_nan=True)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), leaf, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), leaf)[0],
        equal_nan=True,
    )


def test_numpy_torch_and_batch_round_trip():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    array = xr.DataArray(
        data,
        dims=("member", "time", "variable"),
        coords={
            "member": [0, 1],
            "time": np.arange(3),
            "variable": ["a", "b", "c", "d"],
            "valid_time": ("time", np.arange(3) + 10),
        },
        name="state",
        attrs={"units": "K"},
    )

    tensor, coords = array.e2s.to_torch()
    assert tensor.data_ptr() == data.ctypes.data
    assert list(coords) == list(array.dims)
    tensor[0, 0, 0] = -1
    assert data[0, 0, 0] == -1

    restored = from_torch(tensor, coords, name=array.name, attrs=array.attrs)
    assert restored.data.ctypes.data == tensor.data_ptr()
    assert restored.name == array.name and restored.attrs == array.attrs
    assert array.e2s.as_numpy().data is array.data

    _, generated_coords = xr.DataArray(np.ones((2, 3)), dims=("x", "y")).e2s.to_torch()
    np.testing.assert_array_equal(generated_coords["x"], np.arange(2))

    batched = array.e2s.batch(("member", "time"), contiguous=False)
    assert batched.dims == ("batch", "variable")
    assert np.shares_memory(batched.data, array.data)
    unbatched = batched.e2s.unbatch(contiguous=False)
    assert np.shares_memory(unbatched.data, batched.data)
    xr.testing.assert_identical(unbatched, array)


def test_batch_copy_and_validation(monkeypatch):
    array = xr.DataArray(
        np.arange(24).reshape(2, 3, 4),
        dims=("a", "b", "c"),
        coords={"a": np.arange(2), "b": np.arange(3), "c": np.arange(4)},
    )

    with pytest.raises(ValueError, match="requires a copy"):
        array.e2s.batch(("a", "c"), contiguous=False)
    batched = array.e2s.batch(("a", "c"))
    assert batched.data.flags.c_contiguous
    xr.testing.assert_identical(batched.e2s.unbatch(), array)

    lazy = xr.DataArray(da.arange(2, chunks=1), dims=("a",))
    assert isinstance(lazy.e2s.as_numpy().data, np.ndarray)
    for operation in (lambda: lazy.e2s.batch(("a",)), lazy.e2s.to_torch):
        with pytest.raises(TypeError, match="only NumPy- or CuPy"):
            operation()

    def missing_cupy(_: str) -> None:
        raise ImportError

    monkeypatch.setattr(cupy_utils, "import_module", missing_cupy)
    with pytest.raises(ImportError, match="CuPy is required"):
        array.e2s.as_cupy()
    assert not array.e2s.is_cupy

    with pytest.raises(ValueError, match="At least one"):
        array.e2s.batch(())
    with pytest.raises(ValueError, match="unique"):
        array.e2s.batch(("a", "a"))
    with pytest.raises(ValueError, match="not found"):
        array.e2s.batch(("missing",))
    mixed_coord = array.assign_coords(mixed=(("a", "c"), np.ones((2, 4))))
    with pytest.raises(NotImplementedError, match="batched and unbatched"):
        mixed_coord.e2s.batch(("a", "b"))
    with pytest.raises(ValueError, match="Recursive batching"):
        array.e2s.batch(("a",), batch_dim="b")
    with pytest.raises(ValueError, match="coordinate 'batch' already exists"):
        array.assign_coords(batch=1).e2s.batch(("a",))
    with pytest.raises(ValueError, match="does not contain"):
        array.e2s.unbatch()
    coords = OrderedDict((("a", np.arange(2)),))
    with pytest.warns(UserWarning, match="autograd"):
        from_torch(torch.zeros(2, requires_grad=True), coords, requires_grad=True)
    with pytest.raises(ValueError, match="floating.*complex"):
        array.e2s.to_torch(requires_grad=True)
    with pytest.raises(ValueError, match="rank"):
        from_torch(torch.zeros(2, 3), coords)
    with pytest.raises(ValueError, match="dimension size"):
        from_torch(
            torch.zeros(2, 3),
            OrderedDict((("a", np.arange(2)), ("b", np.arange(2)))),
        )
    with pytest.raises(TypeError, match="Unsupported Torch device"):
        from_torch(torch.empty(2, device="meta"), coords)

    batched = array.e2s.batch(("a",))
    with pytest.raises(ValueError, match="Recursive batching"):
        batched.e2s.batch(("missing",))
    with pytest.raises(ValueError, match="leading dimension"):
        batched.transpose("b", "batch", "c").e2s.unbatch()
    with pytest.raises(ValueError, match="size does not match"):
        batched.isel(batch=slice(1)).e2s.unbatch()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_cupy_torch_and_batch_round_trip():
    cp = pytest.importorskip(
        "cupy", reason="CuPy is required for GPU integration tests"
    )
    array = xr.DataArray(
        np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        dims=("member", "time", "variable"),
        coords={
            "member": np.arange(2),
            "time": np.arange(3),
            "variable": ["a", "b", "c", "d"],
        },
    ).e2s.as_cupy(device=0)
    assert array.e2s.is_cupy
    assert array.e2s.as_cupy().data is array.data

    tensor, coords = array.e2s.to_torch()
    assert tensor.data_ptr() == array.data.data.ptr
    tensor[0, 0, 0] = -1
    assert int(array.data[0, 0, 0]) == -1

    restored = from_torch(tensor, coords)
    assert restored.data.data.ptr == tensor.data_ptr()
    assert restored.e2s.is_cupy

    batched = restored.e2s.batch(("member", "time"), contiguous=False)
    assert cp.shares_memory(batched.data, restored.data)
    unbatched = batched.e2s.unbatch(contiguous=False)
    assert cp.shares_memory(unbatched.data, batched.data)
    cp.testing.assert_array_equal(unbatched.data, restored.data)

    host = unbatched.e2s.as_numpy()
    assert isinstance(host.data, np.ndarray)
    np.testing.assert_array_equal(host.data, cp.asnumpy(restored.data))

    reordered = restored.transpose("variable", "member", "time")
    copied = reordered.e2s.batch(("variable", "time"))
    assert copied.data.flags.c_contiguous
    cp.testing.assert_array_equal(copied.e2s.unbatch().data, reordered.data)


def test_backend_scope_precedence_and_context_isolation():
    tensor = torch.arange(3.0)
    coords = {"x": np.arange(3)}
    outside = copy_context()
    with cupy_utils.backend("torch"):
        assert from_torch(tensor, coords).e2s.to_torch()[0] is tensor
        assert isinstance(from_torch(tensor, coords, backend="numpy").data, np.ndarray)
        assert isinstance(outside.run(from_torch, tensor, coords).data, np.ndarray)
        with pytest.raises(RuntimeError), cupy_utils.backend("numpy"):
            assert isinstance(from_torch(tensor, coords).data, np.ndarray)
            raise RuntimeError("restore scope")
        assert from_torch(tensor, coords).e2s.to_torch()[0] is tensor
    assert isinstance(from_torch(tensor, coords).data, np.ndarray)
    with pytest.raises(ValueError, match="backend"):
        from_torch(tensor, coords, backend="invalid")
    with pytest.raises(ValueError, match="backend"), cupy_utils.backend("invalid"):
        pass


@pytest.mark.parametrize("policy", [None, "torch", "numpy", "auto", "invalid"])
def test_backend_environment(policy, monkeypatch):
    if policy is None:
        monkeypatch.delenv("EARTH2STUDIO_ARRAY_BACKEND", raising=False)
    else:
        monkeypatch.setenv("EARTH2STUDIO_ARRAY_BACKEND", policy)
    if policy == "invalid":
        with pytest.raises(ValueError, match="backend"):
            cupy_utils._backend_from_environment()
        return
    monkeypatch.setattr(
        cupy_utils, "_DEFAULT_BACKEND", cupy_utils._backend_from_environment()
    )
    tensor = torch.ones(2)
    coords = {"x": np.arange(2)}
    array = from_torch(tensor, coords)
    if policy == "torch":
        assert array.e2s.to_torch()[0] is tensor
    else:
        assert isinstance(array.data, np.ndarray)


def test_torch_nonleaf_round_trip_and_export_warnings():
    leaf = torch.arange(6.0, requires_grad=True)
    tensor = leaf.square()
    coords = {"x": np.arange(6)}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        array = from_torch(tensor, coords, backend="torch")
        assert array.e2s.to_torch()[0] is tensor
        assert array.e2s.to_torch(requires_grad=True)[0] is tensor
    assert not caught
    array.e2s.to_torch()[0].sum().backward()
    torch.testing.assert_close(leaf.grad, 2 * leaf.detach())
    for export in (
        array.e2s.as_numpy,
        lambda: array.e2s.to_backend("numpy"),
        lambda: from_torch(tensor, coords, backend="auto"),
    ):
        with pytest.warns(UserWarning, match="autograd") as caught:
            result = export()
        assert len(caught) == 1
        assert isinstance(result.data, np.ndarray)
        assert not result.e2s.to_torch()[0].requires_grad
        np.testing.assert_array_equal(result.data, tensor.detach().numpy())


def test_torch_wrapping_ignores_default_device():
    tensor = torch.arange(6.0, requires_grad=True)
    cupy_utils._numpy_dtype.cache_clear()
    with torch.device("meta"):
        array = from_torch(tensor, {"x": np.arange(6)}, backend="torch")
        result = (array + 2).e2s.to_torch()[0]
    assert array.e2s.to_torch()[0] is tensor
    assert_values_and_gradients(result, tensor + 2, tensor)


def test_backend_tracking_metadata_and_devices(monkeypatch):
    tensor = torch.ones(3)
    signature = xr.DataArray(
        np.zeros(3),
        dims="x",
        coords={"x": [1, 2, 3]},
        name="state",
        attrs={"units": "K"},
    )
    signature.encoding["test"] = "encoding"
    array = from_torch(tensor, signature, backend="torch", requires_grad=True)
    assert array.e2s.to_torch()[0].requires_grad
    assert not tensor.requires_grad
    assert array.name == "state" and array.attrs == {"units": "K"}
    original = from_torch(tensor, signature, backend="numpy")
    original.encoding = signature.encoding.copy()
    converted = original.e2s.to_backend("torch", device="cpu")
    assert converted.encoding == original.encoding
    xr.testing.assert_identical(converted.e2s.as_numpy(), original)
    tracked, _ = original.e2s.to_torch(requires_grad=True)
    assert tracked.is_leaf and tracked.requires_grad
    with pytest.raises(ValueError, match="CPU|cpu"):
        converted.e2s.to_backend("numpy", device="cuda:0")
    with pytest.raises(TypeError, match="Unsupported Torch device"):
        converted.e2s.to_backend("torch", device="meta")
    with pytest.raises(ValueError, match="floating.*complex"):
        from_torch(
            torch.ones(3, dtype=torch.int64),
            signature,
            backend="torch",
            requires_grad=True,
        )

    def missing_cupy(_):
        raise ImportError

    monkeypatch.setattr(cupy_utils, "import_module", missing_cupy)
    assert from_torch(tensor, signature, backend="torch").e2s.to_torch()[0] is tensor
    assert isinstance(from_torch(tensor, signature, backend="numpy").data, np.ndarray)
    with pytest.raises(ImportError, match="CuPy is required"):
        from_torch(tensor, signature, backend="cupy")


@pytest.mark.parametrize(
    "operation,reference",
    [
        (
            lambda a: a.sel(a=[1, 0], c=[3, 1]).transpose("c", "b", "a"),
            lambda t: t[[1, 0]][:, :, [3, 1]].permute(2, 1, 0),
        ),
        (
            lambda a: a.isel(
                a=xr.DataArray([1, 0], dims="points"),
                c=xr.DataArray([3, 1], dims="points"),
            ),
            lambda t: t[[1, 0], :, [3, 1]],
        ),
        (
            lambda a: a.isel(a=1, b=slice(None, None, -1), c=slice(1, None, 2)),
            lambda t: t[1].flip(0)[:, 1::2],
        ),
        (
            lambda a: a.copy(deep=True)
            .rename(a="member")
            .assign_coords(member=[10, 20]),
            lambda t: t.clone(),
        ),
        (lambda a: xr.concat([a, a * 2], dim="a"), lambda t: torch.cat([t, t * 2])),
        (
            lambda a: a.expand_dims(sample=[0, 1])
            .isel(sample=0)
            .expand_dims(single=[0])
            .squeeze("single"),
            lambda t: t,
        ),
        (
            lambda a: (2 * a + xr.DataArray([1.0, 2.0, 3.0, 4.0], dims="c")) ** 2 / 3
            - a,
            lambda t: (2 * t + torch.arange(1, 5, dtype=torch.float64)) ** 2 / 3 - t,
        ),
        (lambda a: a.mean(["a", "c"]), lambda t: t.mean((0, 2))),
        (lambda a: a.sum("b", skipna=False), lambda t: t.sum(1)),
        (
            lambda a: xr.concat([a, a * 2], dim="sample")
            .mean("c", keepdims=True)
            .sum(dim=[]),
            lambda t: torch.stack([t, t * 2]).mean(3, keepdim=True),
        ),
    ],
    ids="index vectorized reverse copy concat expand arithmetic mean sum stack".split(),
)
def test_torch_xarray_operations_preserve_gradients(operation, reference):
    leaf = torch.arange(24.0, requires_grad=True)
    source = leaf.reshape(2, 3, 4)
    array = from_torch(
        source,
        {"a": np.arange(2), "b": np.arange(3), "c": np.arange(4)},
        backend="torch",
    )
    assert_values_and_gradients(
        operation(array).e2s.to_torch()[0], reference(source), leaf
    )


@pytest.mark.parametrize("skipna", [True, False])
@pytest.mark.parametrize("operation", ["sum", "mean"])
def test_torch_nan_reductions(operation, skipna):
    tensor = torch.tensor(
        [[1.0, float("nan"), 3.0], [float("nan"), float("nan"), float("nan")]],
        requires_grad=True,
    )
    array = from_torch(tensor, {"x": [0, 1], "y": [0, 1, 2]}, backend="torch")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = getattr(array, operation)("y", skipna=skipna).e2s.to_torch()[0]
    # xarray builds nondifferentiable masks/constants for skipna sum; the
    # selected data still retains its gradient, checked below.
    if operation == "sum" and skipna:
        assert any("isnan" in str(w.message) for w in caught)
        assert any("zeros_like" in str(w.message) for w in caught)
    else:
        assert not caught
    expected = getattr(torch, ("nan" if skipna else "") + operation)(tensor, dim=1)
    assert_values_and_gradients(actual, expected, tensor)


def test_torch_comparisons_cast_and_unsupported_operations():
    tensor = torch.arange(3.0, requires_grad=True)
    array = from_torch(tensor, {"x": [0, 1, 2]}, backend="torch")
    with pytest.warns(UserWarning, match="greater.*gradient"):
        torch.testing.assert_close((array > 1).e2s.to_torch()[0], tensor > 1)
    converted = array.astype(np.float64).e2s.to_torch()[0]
    assert converted.dtype == torch.float64 and converted.requires_grad
    assert array.copy(deep=True).e2s.to_torch()[0].data_ptr() != tensor.data_ptr()
    with pytest.raises(TypeError, match="Implicit NumPy"):
        np.asarray(array.data)
    for operation in (
        lambda: np.linalg.inv(array.data),
        lambda: np.add(array.data, 1, out=(array.data,)),
        lambda: np.reshape(array.data, (3, 1), "F"),
        lambda: np.concatenate([array.data] * 2, 0, np.empty(6)),
        lambda: np.broadcast_to(array.data, (2, 3), True),
        lambda: np.sum(array.data, where=True),
    ):
        with pytest.raises(NotImplementedError, match="Torch backend"):
            operation()
    torch.testing.assert_close(
        np.stack([array.data] * 2, axis=-1).tensor, torch.stack([tensor] * 2, dim=-1)
    )


def test_torch_batch_views_and_copy_gradients():
    leaf = torch.arange(24.0, requires_grad=True)
    tensor = leaf.reshape(2, 3, 4)
    array = from_torch(
        tensor,
        {"a": np.arange(2), "b": np.arange(3), "c": np.arange(4)},
        backend="torch",
    )
    batched = array.e2s.batch(("a", "b"), contiguous=False)
    assert batched.e2s.to_torch()[0].data_ptr() == tensor.data_ptr()
    torch.testing.assert_close(
        batched.e2s.unbatch(contiguous=False).e2s.to_torch()[0], tensor
    )
    with pytest.raises(ValueError, match="requires a copy"):
        array.e2s.batch(("a", "c"), contiguous=False)
    copied = array.e2s.batch(("a", "c"))
    assert copied.e2s.to_torch()[0].is_contiguous()
    restored = copied.e2s.unbatch().e2s.to_torch()[0]
    torch.testing.assert_close(restored, tensor)
    restored.sum().backward()
    torch.testing.assert_close(leaf.grad, torch.ones_like(leaf))


@pytest.mark.parametrize("leading", [(), (2,), (2, 3)])
def test_unchanged_batched_component_chain(leading):
    from earth2studio.models.batch import batch_func
    from earth2studio.utils.coords import coord_array

    class Component:
        def input_coords(self):
            return coord_array(
                ("batch", "variable"), {"variable": ["a", "b"]}, dynamic=("batch",)
            )

        @batch_func()
        def __call__(self, x):
            tensor, _ = x.e2s.to_torch()
            return from_torch(tensor * 2, x)

    coords = {f"lead{i}": np.arange(size) for i, size in enumerate(leading)}
    coords["variable"] = ["a", "b"]
    tensor = torch.ones((*leading, 2), requires_grad=True)
    with cupy_utils.backend("torch"):
        array = from_torch(tensor, coords)
        result = Component()(Component()(array))
        assert result.dims == array.dims
        result.e2s.to_torch()[0].sum().backward()
    torch.testing.assert_close(tensor.grad, torch.full_like(tensor, 4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_cuda_backend_transfers_and_exports():
    cp = pytest.importorskip("cupy")
    leaf = torch.arange(6.0, requires_grad=True)
    array = from_torch(
        leaf.square().reshape(2, 3), {"x": [0, 1], "y": [0, 1, 2]}, backend="torch"
    )
    device = array.e2s.to_backend("torch", device="cuda:0")
    actual = (
        (device.isel(y=[2, 0]) * 2)
        .mean("x")
        .e2s.to_backend("torch", device="cpu")
        .e2s.to_torch()[0]
    )
    expected = (leaf.square().reshape(2, 3)[:, [2, 0]] * 2).mean(0)
    assert_values_and_gradients(actual, expected, leaf)
    for export in (
        lambda: array.e2s.as_cupy(device=0),
        device.e2s.as_cupy,
        lambda: device.e2s.to_backend("auto"),
        lambda: from_torch(device.e2s.to_torch()[0], device, backend="cupy"),
    ):
        with pytest.warns(UserWarning, match="autograd") as caught:
            result = export()
        assert len(caught) == 1
        assert isinstance(result.data, cp.ndarray)
        assert result.e2s.to_torch()[0].is_cuda
        assert not result.e2s.to_torch()[0].requires_grad
        fresh = result.e2s.to_torch(requires_grad=True)[0]
        assert fresh.requires_grad and fresh.is_leaf
    with pytest.warns(UserWarning, match="autograd"):
        host = from_torch(device.e2s.to_torch()[0], device, backend="numpy")
    assert isinstance(host.data, np.ndarray)
    with pytest.raises(ValueError, match="CUDA"):
        device.e2s.to_backend("cupy", device="cpu")
    untracked = from_torch(torch.ones(2), {"x": [0, 1]}, backend="cupy")
    assert isinstance(untracked.data, cp.ndarray)


@pytest.mark.parametrize(
    "operation,dtype,axis,output_dtype",
    [
        (np.nansum, np.float32, (), None),
        (np.nanmean, np.float32, (), None),
        (np.sum, np.int32, (), None),
        (np.sum, np.int32, None, np.int32),
        (np.nansum, np.int32, 0, np.int32),
        (np.mean, np.int32, 0, None),
    ],
)
def test_torch_reduction_dtypes_and_empty_axes(operation, dtype, axis, output_dtype):
    values = np.array([1, np.nan if dtype == np.float32 else 2], dtype=dtype)
    array = from_torch(torch.from_numpy(values), {"x": [0, 1]}, backend="torch")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = operation(values, axis=axis, dtype=output_dtype)
    actual = operation(array.data, axis=axis, dtype=output_dtype).tensor
    torch.testing.assert_close(actual, torch.as_tensor(expected), equal_nan=True)


@pytest.mark.parametrize(
    "operation",
    [np.less, np.less_equal, np.greater, np.greater_equal, np.equal, np.not_equal],
)
def test_torch_scalar_left_comparisons(operation):
    array = from_torch(torch.arange(3.0), {"x": [0, 1, 2]}, backend="torch")
    np.testing.assert_array_equal(
        operation(1, array.data).tensor.numpy(), operation(1, np.arange(3.0))
    )


@pytest.mark.parametrize(
    "name,operation",
    [
        ("floor_divide", lambda a: np.floor_divide(5, a)),
        ("equal", lambda a: np.equal(a, 2)),
        ("less", lambda a: np.less(2, a)),
        ("isnan", np.isnan),
        ("isfinite", np.isfinite),
        ("logical_not", np.logical_not),
        ("logical_and", lambda a: np.logical_and(a, a)),
        ("zeros_like", np.zeros_like),
        ("astype", lambda a: a.astype(np.int64)),
        ("astype", lambda a: a.astype(bool)),
        ("sum", lambda a: np.sum(a, dtype=np.int64)),
        ("nansum", lambda a: np.nansum(a, dtype=np.int64)),
        ("imag", lambda a: a.imag),
    ],
)
@pytest.mark.parametrize("mode", ["tracked", "untracked", "no_grad", "inference_mode"])
def test_torch_nondifferentiable_operations_warn(name, operation, mode):
    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=mode != "untracked")
    data = from_torch(tensor, {"x": [0, 1, 2]}, backend="torch").data
    expected = operation(
        from_torch(tensor.detach(), {"x": [0, 1, 2]}, backend="torch").data
    )
    context = (
        getattr(torch, mode)()
        if mode in ("no_grad", "inference_mode")
        else torch.enable_grad()
    )
    with context, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if mode == "tracked" and name in ("sum", "nansum"):
            with pytest.raises(RuntimeError, match="Autograd not support dtype"):
                operation(data)
        else:
            actual = operation(data)
            torch.testing.assert_close(actual.tensor, expected.tensor)
    assert len(caught) == (1 if mode == "tracked" else 0)
    if caught:
        assert issubclass(caught[0].category, UserWarning)
        assert name in str(caught[0].message)
        assert "gradient" in str(caught[0].message)


def test_torch_differentiable_operations_do_not_warn():
    tensor = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    array = from_torch(tensor, {"x": [0, 1, 2]}, backend="torch")
    with warnings.catch_warnings(record=True) as caught:
        result = ((array + 1) ** 2).astype(np.float64).sum(skipna=False)
        result.e2s.to_torch()[0].backward()
    assert not caught
    torch.testing.assert_close(tensor.grad, 2 * (tensor.detach() + 1))

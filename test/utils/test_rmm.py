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

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import xarray as xr

from earth2studio.utils import rmm as memory
from earth2studio.utils.cupy import backend, from_torch


@pytest.fixture
def allocators(monkeypatch):
    class DefaultResource:
        pass

    resources = [DefaultResource(), object()]
    cp = Mock()
    cp.cuda.Device.side_effect = lambda device: nullcontext()
    mr = Mock(CudaMemoryResource=DefaultResource)
    mr.get_current_device_resource.side_effect = resources
    torch_allocator, cupy_allocator = object(), object()
    for name, module in {
        "cupy": cp,
        "rmm": SimpleNamespace(mr=mr),
        "rmm.allocators.cupy": SimpleNamespace(rmm_cupy_allocator=cupy_allocator),
        "rmm.allocators.torch": SimpleNamespace(rmm_torch_allocator=torch_allocator),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(memory._configure_rmm, "configured", False)
    monkeypatch.setattr(memory._configure_rmm, "disabled", False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    switch = Mock()
    monkeypatch.setattr(torch.cuda.memory, "change_current_allocator", switch)
    return SimpleNamespace(
        cp=cp,
        mr=mr,
        switch=switch,
        resources=resources,
        torch_allocator=torch_allocator,
        cupy_allocator=cupy_allocator,
    )


@pytest.mark.parametrize("policy", ["auto", "cupy"])
def test_configuration_preserves_resources_and_is_idempotent(allocators, policy):
    with backend(policy):
        memory._configure_rmm()
    assert memory._configure_rmm.configured
    allocators.switch.assert_called_once_with(allocators.torch_allocator)
    allocators.mr.PoolMemoryResource.assert_called_once_with(
        allocators.resources[0], initial_pool_size=0
    )
    allocators.mr.set_per_device_resource.assert_called_once_with(
        0, allocators.mr.PoolMemoryResource.return_value
    )
    allocators.cp.cuda.set_allocator.assert_called_with(allocators.cupy_allocator)


@pytest.mark.parametrize(
    "disabled,cuda_available,platform,missing",
    [
        (True, True, "linux", None),
        (False, False, "linux", None),
        (False, True, "win32", None),
        (False, True, "darwin", None),
        (False, True, "linux", "rmm"),
        (False, True, "linux", "cupy"),
        (False, True, "linux", "rmm.allocators.torch"),
    ],
)
def test_configuration_skips_without_changing_allocators(
    monkeypatch, allocators, disabled, cuda_available, platform, missing
):
    monkeypatch.setattr(memory._configure_rmm, "disabled", disabled)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(sys, "platform", platform)
    if missing:
        monkeypatch.setitem(sys.modules, missing, None)
    memory._configure_rmm()
    allocators.switch.assert_not_called()
    allocators.cp.cuda.set_allocator.assert_not_called()
    allocators.mr.get_current_device_resource.assert_not_called()


def test_late_configuration_preserves_allocators(allocators):
    allocators.switch.side_effect = RuntimeError("Already initialized")
    with pytest.raises(RuntimeError, match="Restart.*before"):
        memory._configure_rmm()
    allocators.mr.set_per_device_resource.assert_not_called()
    allocators.cp.cuda.set_allocator.assert_called_once_with(
        allocators.cp.cuda.get_allocator.return_value
    )
    assert not memory._configure_rmm.configured


def test_allocator_context_failure_allows_retry(allocators):
    allocators.cp.cuda.set_allocator.side_effect = ValueError("using_allocator")
    with pytest.raises(RuntimeError, match="using_allocator"):
        memory._configure_rmm()
    allocators.switch.assert_not_called()
    allocators.mr.set_per_device_resource.assert_not_called()
    allocators.cp.cuda.set_allocator.side_effect = None
    memory._configure_rmm()
    assert memory._configure_rmm.configured


@pytest.fixture
def shared_cuda():
    if not torch.cuda.is_available() or not memory._configure_rmm.configured:
        pytest.skip("Shared RMM CUDA allocator not enabled")
    return pytest.importorskip("cupy")


def test_torch_and_cupy_share_allocations(shared_cuda):
    cp = shared_cuda
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    assert cp.cuda.get_allocator() is rmm_cupy_allocator
    for device in range(torch.cuda.device_count()):
        with cp.cuda.Device(device):
            resource = rmm.mr.get_current_device_resource()
            x = torch.ones(1024, device=f"cuda:{device}")
            view = cp.from_dlpack(x)
            assert view.data.ptr == x.data_ptr()
            cp.testing.assert_array_equal(view, 1)
            del view
            torch.cuda.synchronize(device)
            pointer = x.data_ptr()
            del x
            y = cp.empty(1024, dtype=cp.float32)
            assert y.data.ptr == pointer
            with backend("numpy"), backend("cupy"):
                assert rmm.mr.get_current_device_resource() is resource


@pytest.mark.parametrize("disabled", [False, True])
def test_cupy_conversions_preserve_allocators(monkeypatch, shared_cuda, disabled):
    cp = shared_cuda
    monkeypatch.setattr(memory._configure_rmm, "disabled", disabled)
    switch = Mock()
    monkeypatch.setattr(torch.cuda.memory, "change_current_allocator", switch)
    allocator = cp.cuda.get_allocator()
    with backend("cupy"):
        array = from_torch(torch.ones(1), {"x": [0]})
    for result in (
        array,
        xr.DataArray([1.0]).e2s.as_cupy(),
        xr.DataArray([1.0]).e2s.to_backend("cupy"),
        array.e2s.to_backend("auto"),
        array.e2s.as_numpy().e2s.as_cupy(),
    ):
        cp.testing.assert_array_equal(result.data, 1)
    switch.assert_not_called()
    assert cp.cuda.get_allocator() is allocator


def test_cudf_shares_rmm_resource(shared_cuda):
    cp = shared_cuda
    import rmm

    resource = rmm.mr.get_current_device_resource()
    cudf = pytest.importorskip("cudf")
    assert rmm.mr.get_current_device_resource() is resource
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    y = cp.from_dlpack(x)
    cp.testing.assert_array_equal(cudf.Series(y).to_cupy(), y)

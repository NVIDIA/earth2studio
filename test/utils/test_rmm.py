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

import importlib.util
import os
import subprocess
import sys
import textwrap

import pytest
import torch


def run_script(script: str, backend: str = "auto", disable_rmm: str = "0") -> None:
    result = subprocess.run(  # noqa: S603 - scripts are static test inputs
        [sys.executable, "-c", textwrap.dedent(script)],
        env={
            **os.environ,
            "EARTH2STUDIO_ARRAY_BACKEND": backend,
            "EARTH2STUDIO_DISABLE_RMM": disable_rmm,
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
@pytest.mark.parametrize("disabled", ["1", "true", "YES", "on"])
def test_disable_rmm_preserves_allocators_and_conversions(disabled: str) -> None:
    run_script(
        """
        import sys
        import torch
        import cupy as cp
        x = torch.ones(1, device="cuda")
        allocator = cp.cuda.get_allocator()
        import earth2studio
        from earth2studio.utils.cupy import backend, from_torch
        assert "rmm" not in sys.modules
        with backend("cupy"):
            array = from_torch(x, {"x": [0]})
            assert array.e2s.is_cupy
            assert array.e2s.to_backend("cupy").data.data.ptr == x.data_ptr()
            cp.testing.assert_array_equal(array.e2s.as_numpy().e2s.as_cupy().data, 1)
        assert cp.cuda.get_allocator() is allocator
        assert torch.cuda.memory.get_allocator_backend() == "native"
        assert "rmm" not in sys.modules
        """,
        disable_rmm=disabled,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
@pytest.mark.parametrize("policy", ["auto", "cupy"])
def test_import_shares_rmm_pool(policy: str) -> None:
    run_script(
        """
        import earth2studio
        import cupy as cp
        import rmm
        import torch
        from rmm.allocators.cupy import rmm_cupy_allocator
        from earth2studio.utils.cupy import backend

        assert cp.cuda.get_allocator() is rmm_cupy_allocator
        assert torch.cuda.memory.get_allocator_backend() == "pluggable"
        for device in range(torch.cuda.device_count()):
            with cp.cuda.Device(device):
                pool = rmm.mr.get_current_device_resource()
                assert isinstance(pool, rmm.mr.PoolMemoryResource)
                assert pool.pool_size() == 0
                x = torch.empty(1024, device=f"cuda:{device}")
                pointer = x.data_ptr()
                del x
                y = cp.empty(1024, dtype=cp.float32)
                assert y.data.ptr == pointer
                del y
                x = torch.ones(1024, device=f"cuda:{device}")
                assert x.data_ptr() == pointer
                view = cp.from_dlpack(x)
                assert view.data.ptr == pointer
                cp.testing.assert_array_equal(view, 1)
                with backend("numpy"), backend("cupy"):
                    assert rmm.mr.get_current_device_resource() is pool
                assert rmm.mr.get_current_device_resource() is pool
                del view, x
        """,
        policy,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_preserves_custom_rmm_resource() -> None:
    run_script("""
        import rmm
        upstream = rmm.mr.CudaMemoryResource()
        resource = rmm.mr.TrackingResourceAdaptor(upstream)
        rmm.mr.set_current_device_resource(resource)
        import earth2studio
        import torch
        import cupy as cp
        assert rmm.mr.get_current_device_resource() is resource
        x = torch.empty(1024, device="cuda")
        y = cp.empty(1024, dtype=cp.float32)
        assert resource.get_allocated_bytes() == 8192
        del x, y
        assert resource.get_allocated_bytes() == 0
        """)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
@pytest.mark.parametrize(
    "selection",
    [
        'with backend("cupy"): pass',
        'with backend("auto"): pass',
        'xr.DataArray([1.0], dims="x").e2s.as_cupy()',
        'from_torch(torch.ones(1), {"x": [0]}, backend="cupy")',
    ],
)
def test_late_selection_fails_without_partial_configuration(selection: str) -> None:
    run_script(
        """
        import earth2studio
        import cupy as cp
        import rmm
        import torch
        import xarray as xr
        from earth2studio.utils.cupy import backend, from_torch
        allocator = cp.cuda.get_allocator()
        resource = rmm.mr.get_current_device_resource()
        x = torch.ones(1, device="cuda")
        try:
            exec(SELECTION)
        except RuntimeError as error:
            assert "before" in str(error) and "restart" in str(error).lower()
        else:
            raise AssertionError("Late CuPy selection must fail")
        assert cp.cuda.get_allocator() is allocator
        assert rmm.mr.get_current_device_resource() is resource
        assert torch.cuda.memory.get_allocator_backend() == "native"
        """.replace("SELECTION", repr(selection)),
        "torch",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
@pytest.mark.parametrize("policy", ["numpy", "torch"])
def test_backend_scope_configures_before_first_cuda_tensor(policy: str) -> None:
    run_script(
        """
        import earth2studio
        import sys
        import torch
        import xarray as xr
        from earth2studio.utils.cupy import backend
        assert "rmm" not in sys.modules
        assert not xr.DataArray([1]).e2s.is_cupy
        assert "rmm" not in sys.modules
        with backend("cupy"):
            x = torch.ones(1, device="cuda")
        assert torch.cuda.memory.get_allocator_backend() == "pluggable"
        """,
        policy,
    )


def test_cpu_only_import_does_not_load_rmm() -> None:
    run_script("""
        import sys
        import torch
        torch.cuda.is_available = lambda: False
        import earth2studio
        assert "rmm" not in sys.modules
        assert not torch.cuda.is_initialized()
        """)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
@pytest.mark.xfail(
    strict=True,
    reason="RMM's Torch adapter does not implement record_stream lifetime tracking",
)
def test_record_stream_prevents_premature_reuse() -> None:
    run_script("""
        import earth2studio
        import torch
        warm = torch.ones(1024 * 1024, device="cuda")
        warm2 = warm + 1
        torch.cuda.synchronize()
        del warm, warm2
        stream = torch.cuda.Stream()
        x = torch.ones(1024, device="cuda")
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(1000000000)
            y = x + 1
        x.record_stream(stream)
        del x
        z = torch.full((1024,), 99.0, device="cuda")
        torch.cuda.synchronize()
        assert y[0].item() == 2, f"Premature allocation reuse: {y[0].item()}"
        """)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_cupy_allocator_context_failure_allows_retry() -> None:
    run_script(
        """
        import earth2studio
        import cupy as cp
        import rmm
        import torch
        from earth2studio.utils.cupy import backend
        allocator = cp.cuda.get_allocator()
        resource = rmm.mr.get_current_device_resource()
        with cp.cuda.using_allocator(cp.cuda.MemoryPool().malloc):
            try:
                with backend("cupy"):
                    pass
            except RuntimeError as error:
                assert "using_allocator" in str(error)
            else:
                raise AssertionError("Cannot configure in an allocator context")
        assert cp.cuda.get_allocator() is allocator
        assert rmm.mr.get_current_device_resource() is resource
        assert torch.cuda.memory.get_allocator_backend() == "native"
        with backend("cupy"):
            x = torch.ones(1, device="cuda")
        assert torch.cuda.memory.get_allocator_backend() == "pluggable"
        """,
        "numpy",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
@pytest.mark.parametrize(
    "conversion",
    [
        "xr.DataArray([1.0]).e2s.as_cupy()",
        'xr.DataArray([1.0]).e2s.to_backend("cupy")',
        'xr.DataArray(cp.ones(1)).e2s.to_backend("cupy")',
        'xr.DataArray(cp.ones(1)).e2s.to_backend("auto")',
        'from_torch(torch.ones(1), {"x": [0]}, backend="cupy")',
    ],
)
def test_explicit_conversion_configures_rmm(conversion: str) -> None:
    run_script(
        """
        import earth2studio
        import cupy as cp
        import torch
        import xarray as xr
        from earth2studio.utils.cupy import from_torch
        result = eval(CONVERSION)
        assert result.e2s.is_cupy
        assert torch.cuda.memory.get_allocator_backend() == "pluggable"
        cp.testing.assert_array_equal(result.data, 1)
        """.replace("CONVERSION", repr(conversion)),
        "numpy",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_missing_rmm_has_actionable_error() -> None:
    run_script("""
        import sys
        import cupy as cp
        allocator = cp.cuda.get_allocator()
        sys.modules["rmm"] = None
        try:
            import earth2studio
        except ImportError as error:
            assert "rmm-cu13" in str(error)
        else:
            raise AssertionError("Missing RMM must fail on CUDA")
        assert cp.cuda.get_allocator() is allocator
        """)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
@pytest.mark.skipif(importlib.util.find_spec("cudf") is None, reason="cudf missing")
@pytest.mark.parametrize("cudf_first", [False, True])
def test_cudf_shares_rmm_resource(cudf_first: bool) -> None:
    run_script(("import cudf\n" if cudf_first else "") + textwrap.dedent("""
            import earth2studio
            import cupy as cp
            import rmm
            import torch
            resource = rmm.mr.get_current_device_resource()
            import cudf
            assert rmm.mr.get_current_device_resource() is resource
            x = torch.arange(1024, device="cuda", dtype=torch.float32)
            y = cp.from_dlpack(x)
            series = cudf.Series(y)
            cp.testing.assert_array_equal(series.to_cupy(), y)
            assert torch.cuda.memory.get_allocator_backend() == "pluggable"
            from rmm.allocators.cupy import rmm_cupy_allocator
            assert cp.cuda.get_allocator() is rmm_cupy_allocator
            """))

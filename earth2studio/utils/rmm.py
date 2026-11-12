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

import os
import sys
from threading import Lock

import torch


class _RMMConfiguration:
    def __init__(self) -> None:
        self._lock = Lock()
        self.configured = False
        self.disabled = os.environ.get(
            "EARTH2STUDIO_DISABLE_RMM", "0"
        ).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def __call__(self) -> None:
        """Install RMM allocators before Torch initializes its CUDA allocator."""
        with self._lock:
            if (
                self.disabled
                or self.configured
                or sys.platform != "linux"
                or not torch.cuda.is_available()
            ):
                return
            self._configure()

    def _configure(self) -> None:
        try:
            import cupy as cp
            import rmm
            from rmm.allocators.cupy import rmm_cupy_allocator
            from rmm.allocators.torch import rmm_torch_allocator
        except ImportError:
            return

        if rmm_torch_allocator is None:
            raise RuntimeError("RMM's Torch CUDA allocator is unavailable.")

        # CuPy forbids global changes inside using_allocator(). Probe with the
        # existing allocator before making the irreversible Torch switch.
        try:
            cp.cuda.set_allocator(cp.cuda.get_allocator())
        except ValueError as error:
            raise RuntimeError(
                "Configure Earth2Studio's CuPy backend outside a CuPy "
                "using_allocator() context, then retry."
            ) from error

        # Prepare resources without publishing them. A failed Torch allocator
        # switch must not leave CuPy using a different allocation policy.
        pools = {}
        for device in range(torch.cuda.device_count()):
            with cp.cuda.Device(device):
                resource = rmm.mr.get_current_device_resource()
                # Only wrap RMM's default; preserve existing pools/custom resources.
                if type(resource) is rmm.mr.CudaMemoryResource:
                    pools[device] = rmm.mr.PoolMemoryResource(
                        resource, initial_pool_size=0
                    )

        try:
            torch.cuda.memory.change_current_allocator(rmm_torch_allocator)
        except RuntimeError as error:
            raise RuntimeError(
                "Earth2Studio cannot configure shared RMM memory after Torch's "
                "CUDA allocator has initialized. Restart the process and import "
                "earth2studio with EARTH2STUDIO_ARRAY_BACKEND=auto or cupy before "
                "creating CUDA tensors, or enter backend('cupy') before CUDA use. "
                "Alternatively, set EARTH2STUDIO_DISABLE_RMM=1 before import "
                "to keep the existing allocators."
            ) from error

        for device, pool in pools.items():
            rmm.mr.set_per_device_resource(device, pool)
        cp.cuda.set_allocator(rmm_cupy_allocator)
        self.configured = True


_configure_rmm = _RMMConfiguration()

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
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

try:
    import onnxruntime as ort

    ort.preload_dlls()
    from onnxruntime import InferenceSession
except ImportError:
    ort = None
    InferenceSession = TypeVar("InferenceSession")  # type: ignore
import torch


def create_ort_session(
    onnx_file: str,
    device: torch.device = torch.device("cpu", 0),
) -> InferenceSession:
    """Create ORT session on specified device

    Parameters
    ----------
    onnx_file : str
        ONNX file
    device : torch.device, optional
        Device for session to run on, by default "cpu"

    Returns
    -------
    ort.InferenceSession
        ORT inference session
    """
    if ort is None:
        raise ImportError(
            "onnxruntime (onnxruntime-gpu) is required for this model. See model install notes for details.\n"
            + "https://nvidia.github.io/earth2studio/userguide/about/install.html#model-dependencies"
        )
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    options.intra_op_num_threads = 1
    options.log_severity_level = 3

    # That will trigger a FileNotFoundError
    os.stat(onnx_file)
    if device.type == "cuda":
        if device.index is None:
            device_index = torch.cuda.current_device()
        else:
            device_index = device.index

        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": device_index,
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        providers = [
            "CPUExecutionProvider",
        ]

    ort_session = ort.InferenceSession(
        onnx_file,
        sess_options=options,
        providers=providers,
    )

    return ort_session


@contextmanager
def fork_rng(
    states: dict[str, torch.Tensor] | None,
    seed: int | None,
    device: torch.device,
) -> Iterator[None]:
    """Resume model-owned Torch RNG state for a numerical sampling call.

    Parameters
    ----------
    states : dict[str, torch.Tensor] | None
        Model-owned states keyed by ``cpu`` and ``cuda:N``. Updated in place
        with the advanced states, including when sampling raises an exception.
    seed : int | None
        Initial seed for a previously unused device. None leaves global RNG
        behavior unchanged. Seeded calls require an initialized CPU state.
    device : torch.device
        Sampling device. CPU state is preserved alongside CUDA state.

    Notes
    -----
    Restores the caller's global state on exit. Do not span iterator yields or
    use concurrently with other code accessing the same global generators.
    """
    if seed is None:
        yield
        return
    if states is None:
        raise ValueError("Seeded sampling requires initialized RNG states")
    devices = []
    if device.type == "cuda":
        devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(states["cpu"])
        for index in devices:
            key = f"cuda:{index}"
            if key not in states:
                states[key] = torch.Generator(device=key).manual_seed(seed).get_state()
            torch.cuda.set_rng_state(states[key], index)
        try:
            yield
        finally:
            states["cpu"] = torch.get_rng_state()
            for index in devices:
                states[f"cuda:{index}"] = torch.cuda.get_rng_state(index)

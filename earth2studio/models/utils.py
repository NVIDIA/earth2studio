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
from typing import TypeVar

try:
    import onnxruntime as ort
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

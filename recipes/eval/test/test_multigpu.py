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

"""Multi-GPU tests for the eval recipe.

Each test launches a ``torchrun`` subprocess with the worker script
``_multigpu_worker.py``.  Tests are skipped when fewer than the required
number of GPUs are available.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_WORKER = str(Path(__file__).with_name("_multigpu_worker.py"))
_RECIPE_ROOT = str(Path(__file__).resolve().parents[1])

_requires_2_gpus = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA GPUs",
)


def _run_worker(
    test_name: str, nproc: int, output_dir: str, timeout: int = 120
) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": _RECIPE_ROOT}
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--standalone",
        _WORKER,
        "--test",
        test_name,
        "--output-dir",
        output_dir,
    ]
    return subprocess.run(
        cmd,  # noqa: S603
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_RECIPE_ROOT,
        env=env,
    )


@_requires_2_gpus
class TestMultiGPU2:
    def test_distribute_work(self, tmp_path):
        result = _run_worker("distribute_work", nproc=2, output_dir=str(tmp_path))
        assert result.returncode == 0, (
            f"Worker failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_run_on_rank0_first(self, tmp_path):
        result = _run_worker("run_on_rank0_first", nproc=2, output_dir=str(tmp_path))
        assert result.returncode == 0, (
            f"Worker failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    def test_end_to_end_inference(self, tmp_path):
        result = _run_worker(
            "end_to_end_inference", nproc=2, output_dir=str(tmp_path), timeout=180
        )
        assert result.returncode == 0, (
            f"Worker failed (rc={result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

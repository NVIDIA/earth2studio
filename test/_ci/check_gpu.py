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

import subprocess
import sys


def main() -> int:
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover - CI precheck utility
        print(f"[gpu-check] torch import failed: {e}")
        return 123

    try:
        ok = torch.cuda.is_available()
    except Exception as e:  # pragma: no cover - CI precheck utility
        print(f"[gpu-check] torch.cuda.is_available() errored: {e}")
        return 123

    if not ok:
        # Log nvidia-smi output if available to aid debugging
        try:
            subprocess.run(["nvidia-smi"], check=False)  # noqa: S603 S607
        except Exception:  # noqa: S603 S110
            pass
        print(
            "[gpu-check] No GPU detected by torch (torch.cuda.is_available() == False)"
        )
        return 123

    print("[gpu-check] GPU detected by torch")
    return 0


if __name__ == "__main__":
    sys.exit(main())

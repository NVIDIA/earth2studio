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
from pathlib import Path

DIRECTORY_ENV_VARS = (
    "EARTH2STUDIO_CACHE",
    "EARTH2STUDIO_DATA_CACHE",
    "EARTH2STUDIO_MODEL_CACHE",
    "PRE_COMMIT_HOME",
    "TESTMON_CACHE",
    "UV_CACHE_DIR",
)
FILE_ENV_VARS = ("TESTMON_DATAFILE",)


def main() -> int:
    paths: set[Path] = set()

    for env_var in DIRECTORY_ENV_VARS:
        if value := os.environ.get(env_var):
            paths.add(Path(value).expanduser())

    for env_var in FILE_ENV_VARS:
        if value := os.environ.get(env_var):
            paths.add(Path(value).expanduser().parent)

    for path in sorted(paths, key=str):
        path.mkdir(parents=True, exist_ok=True)
        print(f"[ci-paths] ensured {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

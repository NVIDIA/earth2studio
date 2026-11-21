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

import configparser
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE_TESTS = sorted(
    path.relative_to(ROOT)
    for directory in ("test/utils", "test/models")
    for path in (ROOT / directory).glob("test_*.py")
)


@pytest.mark.parametrize("test_file", CORE_TESTS, ids=str)
def test_core_file_has_tox_selection(test_file):
    """Core tests must be selected before dependency/testmon filtering can apply."""
    config = configparser.ConfigParser(interpolation=None)
    config.read(ROOT / "tox.ini")
    selected = set()
    for section in config.sections():
        if not section.startswith("testenv:"):
            continue
        commands = config[section].get("commands", "").replace("\\\n", " ")
        for command in commands.splitlines():
            args = shlex.split(command)
            if not args or args[0] != "pytest":
                continue
            selected.update(Path(arg) for arg in args[1:] if arg.startswith("test/"))

    assert any(path == test_file or path in test_file.parents for path in selected), (
        f"{test_file} is not selected by any pytest command in tox.ini; "
        "assign it to an environment with the required dependencies"
    )

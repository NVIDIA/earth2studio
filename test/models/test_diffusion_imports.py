# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys

import pytest


@pytest.mark.parametrize("missing_module", ["physicsnemo", "physicsnemo.diffusion"])
def test_optional_diffusion_import(missing_module: str) -> None:
    script = f"""
import sys
sys.modules[{missing_module!r}] = None

from earth2studio.models._diffusion import SOLVERS, seeded_rng, solver_options
from earth2studio.utils.imports import OptionalDependencyError

assert not SOLVERS
with seeded_rng(None):
    pass
try:
    solver_options(None, "heun")
except OptionalDependencyError:
    pass
else:
    raise AssertionError("Missing PhysicsNeMo must fail when sampling is requested")
"""
    subprocess.run([sys.executable, "-c", script], check=True)  # noqa: S603

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

"""Make the recipe root importable as `src` for the torch-free test suite.

Also bridges loguru into pytest's ``caplog``: the connector logs via loguru (matching
earth2studio), which does not feed the stdlib-``logging`` handler ``caplog`` captures, so the
fixture below adds ``caplog``'s handler as a loguru sink for tests that assert on log output.
"""

import sys
from pathlib import Path

import pytest
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def caplog(caplog):  # noqa: F811 — intentionally override pytest's caplog
    """Route loguru records into pytest's ``caplog`` for the duration of a test."""
    handler_id = logger.add(
        caplog.handler,
        level=0,
        format="{message}",
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    yield caplog
    logger.remove(handler_id)

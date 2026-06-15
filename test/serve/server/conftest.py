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

import logging

import pytest
from loguru import logger


@pytest.fixture(autouse=True)
def propagate_loguru_to_caplog(caplog):
    """Route loguru messages into pytest's caplog so assertions on caplog.text work."""

    class _Sink:
        def __init__(self):
            self._active = True

        def write(self, message):
            if not self._active:
                return
            record = message.record
            level = record["level"].no
            exc = record["exception"]
            exc_info = (exc.type, exc.value, exc.traceback) if exc else None
            logging.getLogger(record["name"]).handle(
                logging.LogRecord(
                    name=record["name"],
                    level=level,
                    pathname=record["file"].path,
                    lineno=record["line"],
                    msg=record["message"],
                    args=(),
                    exc_info=exc_info,
                )
            )

    sink = _Sink()
    handler_id = logger.add(sink, format="{message}", level=0, enqueue=False)
    yield caplog
    sink._active = False
    try:
        logger.remove(handler_id)
    except ValueError:
        pass

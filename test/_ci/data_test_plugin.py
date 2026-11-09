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

"""Incremental reports that survive a killed pytest child."""

import faulthandler
import json
import os
import signal

import pytest


def emit(event: str, **fields: object) -> None:
    with open(os.environ["DATA_TEST_EVENTS"], "a") as stream:
        stream.write(json.dumps({"event": event, **fields}) + "\n")


def pytest_configure() -> None:
    # pytest redirects fd 2 during capture; use a separate descriptor so a killed
    # test cannot take its diagnostic stack dump with the capture buffer.
    stack_fd = os.open(
        os.environ["DATA_TEST_STACKS"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    faulthandler.register(signal.SIGUSR1, file=stack_fd, all_threads=True)
    emit("ready")


def pytest_collection_finish(session: pytest.Session) -> None:
    emit("collected", nodeids=[item.nodeid for item in session.items])


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    emit("phase", nodeid=item.nodeid, phase="setup")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item: pytest.Item) -> None:
    emit("phase", nodeid=item.nodeid, phase="call")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item: pytest.Item) -> None:
    emit("phase", nodeid=item.nodeid, phase="teardown")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    outcome = report.outcome.upper()
    if hasattr(report, "wasxfail"):
        outcome = "XFAIL" if report.skipped else "XPASS"
    emit(
        "report",
        nodeid=report.nodeid,
        phase=report.when,
        outcome=outcome,
        duration=report.duration,
        detail=report.longreprtext,
    )


def pytest_runtest_logfinish(nodeid: str) -> None:
    emit("finished", nodeid=nodeid)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(exitstatus: int) -> None:
    emit("sessionfinish", exitstatus=int(exitstatus))

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
import math
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest


def emit(event: str, **fields: object) -> None:
    if "DATA_TEST_EVENTS" not in os.environ:
        return
    with open(os.environ["DATA_TEST_EVENTS"], "a") as stream:
        stream.write(json.dumps({"event": event, **fields}) + "\n")


def positive_timeout(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("live timeout must be finite and positive")
    return seconds


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--live",
        choices=("changed", "all", "none"),
        default=os.getenv("DATA_TEST_LIVE", "all"),
    )
    parser.addoption("--live-timeout", type=positive_timeout, default=None)


def changed_files() -> list[str]:
    if "DATA_TEST_CHANGED_FILES" in os.environ:
        return os.environ["DATA_TEST_CHANGED_FILES"].splitlines()
    base = os.getenv("DATA_TEST_BASE", "HEAD")
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", "diff", "--name-only", base, "--"],  # noqa: S607
            text=True,
            timeout=10,
        ).splitlines()
    except (subprocess.SubprocessError, OSError) as error:
        raise pytest.UsageError(
            f"Cannot determine live-test changes against {base}: {error}"
        ) from error


def source_changed(file: str, changes: list[str]) -> bool:
    name = Path(file).stem.removeprefix("test_")
    for change in changes:
        path = Path(change)
        if change.startswith(("earth2studio/data/", "earth2studio/lexicon/")):
            # Source families also cover ace2_era5, ghcn_daily, wb2era5, etc.
            if name.startswith(path.stem):
                return True
            if path.stem.startswith(("utils", "_")) or path.stem in {
                "base",
                "stac",
                "time_window",
                "composite",
            }:
                return True
        elif change.startswith("test/data/test_"):
            if path.stem == Path(file).stem:
                return True
        elif change.startswith(
            ("earth2studio/utils/", "test/_ci/", ".github/")
        ) or change in {
            "test/conftest.py",
            "test/data/conftest.py",
            "pyproject.toml",
            "uv.lock",
            "tox.ini",
        }:
            return True
    return False


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "live(timeout=60): slow live data-source call; requires --slow",
    )
    supervised = "DATA_TEST_EVENTS" in os.environ
    expected_file = os.getenv("DATA_TEST_FILE")
    if expected_file and config.args != [expected_file]:
        raise pytest.UsageError(
            "Do not forward test paths/node IDs after --; use DATA_TEST_PATHS "
            "or supervisor path arguments instead."
        )
    config._live_changes = (
        changed_files() if supervised and config.getoption("live") == "changed" else []
    )
    # Disable selection before testmon can omit entire files. Keep collection/coverage.
    if config.getoption("live") == "all" or (
        config.getoption("live") == "changed"
        and any(source_changed(file, config._live_changes) for file in config.args)
    ):
        config.option.testmon_noselect = True
    if "DATA_TEST_EVENTS" not in os.environ:
        return
    # pytest redirects fd 2 during capture; use a separate descriptor so a killed
    # test cannot take its diagnostic stack dump with the capture buffer.
    stack_fd = os.open(
        os.environ["DATA_TEST_STACKS"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
    )
    faulthandler.register(signal.SIGUSR1, file=stack_fd, all_threads=True)
    emit("ready")


def pytest_itemcollected(item) -> None:
    # Add before collection modifiers so pytest-skip-slow and -m see the marker.
    if item.get_closest_marker("live"):
        item.add_marker(pytest.mark.slow)


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items) -> None:
    if (
        "DATA_TEST_EVENTS" not in os.environ
        and config.getoption("live") == "changed"
        and any(item.get_closest_marker("live") for item in items)
    ):
        config._live_changes = changed_files()
    completed = set(json.loads(os.getenv("DATA_TEST_COMPLETED", "[]")))
    items[:] = [item for item in items if item.nodeid not in completed]
    for item in items:
        marker = item.get_closest_marker("live")
        if marker is None:
            continue
        timeout = config.getoption("live_timeout")
        item._live_timeout = positive_timeout(
            str(timeout if timeout is not None else marker.kwargs.get("timeout", 60))
        )
        mode = config.getoption("live")
        if mode == "none" or (
            mode == "changed"
            and not source_changed(str(item.path), config._live_changes)
        ):
            item.add_marker(pytest.mark.skip(reason="live source not selected"))
        # Supervised runs own the deadline; direct pytest uses pytest-timeout.
        timeout = 0 if "DATA_TEST_EVENTS" in os.environ else item._live_timeout
        item.add_marker(pytest.mark.timeout(timeout), append=False)


def pytest_collection_finish(session: pytest.Session) -> None:
    emit("collected", nodeids=[item.nodeid for item in session.items])


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    emit("phase", nodeid=item.nodeid, phase="setup")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item: pytest.Item) -> None:
    emit(
        "phase",
        nodeid=item.nodeid,
        phase="call",
        started=time.monotonic(),
        timeout=getattr(item, "_live_timeout", None),
    )


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


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session):
    outcome = yield
    outcome.get_result()
    emit("sessionfinish", exitstatus=int(session.exitstatus), finished=time.monotonic())

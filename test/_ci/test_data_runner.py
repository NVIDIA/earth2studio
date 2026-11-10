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

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

RUNNER = Path(__file__).with_name("run_data_tests.py")


@pytest.fixture(autouse=True)
def isolated_live_selection(monkeypatch):
    for name in ("DATA_TEST_LIVE", "DATA_TEST_BASE", "DATA_TEST_CHANGED_FILES"):
        monkeypatch.delenv(name, raising=False)


def run_files(tmp_path: Path, files: dict[str, str], *options: str, pytest_args=()):
    for name, content in files.items():
        (tmp_path / name).write_text(content)
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", GITHUB_ACTIONS="true")
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(RUNNER),
            "--output",
            str(tmp_path / "results"),
            "--file-timeout",
            "2",
            "--timeout",
            "10",
            "--grace",
            "0.1",
            *options,
            *files,
            "--",
            "-q",
            *pytest_args,
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    summary = json.loads((tmp_path / "results/summary.json").read_text())
    return result, summary


def test_live_timeout_restarts_and_only_timeout_is_xfailed(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {"test_live.py": """
import os, time, pytest
from pathlib import Path
@pytest.mark.live(timeout=0.15)
def test_timeout():
    Path('old_pid').write_text(str(os.getpid()))
    print('live download stuck')
    time.sleep(60)
@pytest.mark.live
def test_assertion():
    assert os.getpid() != int(Path('old_pid').read_text())
    assert False, 'real assertion'
@pytest.mark.live
def test_error():
    raise RuntimeError('real error')
def test_later():
    pass
"""},
        pytest_args=("--live=all",),
    )
    assert result.returncode == 1, result.stdout
    assert [case["status"] for case in summary[0]["tests"].values()] == [
        "XFAIL",
        "FAILED",
        "FAILED",
        "PASSED",
    ]
    assert "real assertion" in result.stdout
    assert "real error" in result.stdout
    assert "live download stuck" in result.stdout
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "old_pid").read_text()), 0)


@pytest.mark.parametrize("mode", ["none", "changed"])
def test_live_selection_skips_unselected_sources(tmp_path: Path, monkeypatch, mode):
    monkeypatch.setenv("DATA_TEST_CHANGED_FILES", "docs/index.md")
    result, summary = run_files(
        tmp_path,
        {
            "test_gfs.py": "import pytest\n@pytest.mark.live\ndef test_live():\n assert False\ndef test_mock():\n pass\n"
        },
        pytest_args=(f"--live={mode}",),
    )
    assert result.returncode == 0, result.stdout
    assert [case["status"] for case in summary[0]["tests"].values()] == [
        "SKIPPED",
        "PASSED",
    ]


@pytest.mark.parametrize(
    "changed",
    ["earth2studio/data/gfs.py", "earth2studio/data/utils.py", "test/data/test_gfs.py"],
)
def test_changed_source_enables_live_tests(tmp_path: Path, monkeypatch, changed):
    monkeypatch.setenv("DATA_TEST_CHANGED_FILES", changed)
    result, summary = run_files(
        tmp_path,
        {"test_gfs.py": "import pytest\n@pytest.mark.live\ndef test_live():\n pass\n"},
        pytest_args=("--live=changed",),
    )
    assert result.returncode == 0, result.stdout
    assert next(iter(summary[0]["tests"].values()))["status"] == "PASSED"


def test_live_timeout_override_and_remaining_cases(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {
            "test_live.py": "import pytest, time\n@pytest.mark.live(timeout=60)\ndef test_live():\n time.sleep(60)\ndef test_later():\n pass\n"
        },
        pytest_args=("--live=all", "--live-timeout=0.1"),
    )
    assert result.returncode == 0, result.stdout
    assert [case["status"] for case in summary[0]["tests"].values()] == [
        "XFAIL",
        "PASSED",
    ]


def test_live_teardown_timeout_is_failure(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {
            "test_live.py": "import pytest, time\n@pytest.fixture\ndef stuck():\n yield\n time.sleep(60)\n@pytest.mark.live(timeout=0.1)\ndef test_live(stuck):\n pass\n"
        },
        pytest_args=("--live=all",),
    )
    assert result.returncode == 1, result.stdout
    assert summary[0]["status"] == "TIMEOUT"


def test_enabled_live_tests_bypass_testmon(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATA_TEST_CHANGED_FILES", "earth2studio/data/gfs.py")
    body = "import pytest\n@pytest.mark.live\ndef test_live():\n pass\n"
    for _ in range(2):
        result, summary = run_files(
            tmp_path,
            {"test_gfs.py": body},
            pytest_args=("-p", "testmon.pytest_testmon", "--testmon", "--live=changed"),
        )
        assert result.returncode == 0, result.stdout
        assert next(iter(summary[0]["tests"].values()))["status"] == "PASSED"


def test_live_restarts_keep_cumulative_file_budget(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {
            "test_live.py": "import pytest, time\n@pytest.mark.live(timeout=0.15)\n@pytest.mark.parametrize('i', range(20))\ndef test_live(i):\n time.sleep(60)\n"
        },
        pytest_args=("--live=all",),
    )
    assert result.returncode == 1, result.stdout
    assert summary[0]["status"] == "TIMEOUT"
    outcomes = [case["status"] for case in summary[0]["tests"].values()]
    assert "XFAIL" in outcomes
    assert "NOT RUN" in outcomes
    assert summary[0]["duration"] < 4


def test_reports_failures_and_preserves_success_logs(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {
            "test_good.py": "def test_good():\n print('success detail')\n",
            "test_bad.py": "def test_bad():\n print('failure detail')\n assert False, 'broken source'\n",
        },
    )
    assert result.returncode == 1
    assert "broken source" in result.stdout
    assert "failure detail" in result.stdout
    assert "PASSED test_good.py::test_good" in result.stdout
    assert result.stdout.count("::group::") == result.stdout.count("::endgroup::") == 2
    in_group = False
    for line in result.stdout.splitlines():
        if line.startswith("::group::"):
            assert not in_group, "parallel file groups must not overlap"
            in_group = True
        elif line == "::endgroup::":
            assert in_group
            in_group = False
        elif line.startswith("RUNNING "):
            assert not in_group, "worker startup must not interrupt a file group"
    assert not in_group
    assert {entry["status"] for entry in summary} == {"PASSED", "FAILED"}
    assert "success detail" in (tmp_path / "results/test_good/output.log").read_text()
    assert (tmp_path / "results/junit.xml").exists()


@pytest.mark.parametrize(
    "body",
    [
        "import time\ndef test_hang():\n time.sleep(60)\ndef test_pending():\n pass\n",
        "import threading, time\ndef test_shutdown():\n threading.Thread(target=time.sleep, args=(60,)).start()\n",
        "import os\ndef test_crash():\n os._exit(0)\n",
    ],
)
def test_timeout_and_false_success_are_failures(tmp_path: Path, body: str):
    result, summary = run_files(tmp_path, {"test_stuck.py": body})
    assert result.returncode == 1
    assert summary[0]["status"] in {"TIMEOUT", "ERROR"}
    assert "test_stuck.py" in result.stdout
    if "test_pending" in body:
        assert "NOT RUN test_stuck.py::test_pending" in result.stdout
        assert "TIMEOUT test_stuck.py::test_hang" in result.stdout
        assert "test_hang" in (tmp_path / "results/test_stuck/stacks.log").read_text()


def test_kills_descendants_even_when_parent_exits(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {"test_child.py": """
import subprocess, sys
def test_child():
    subprocess.Popen([sys.executable, '-c',
        "import time; from pathlib import Path; time.sleep(2); Path('escaped').touch()"])
"""},
    )
    assert result.returncode == 0, result.stdout
    time.sleep(2.2)
    assert not (tmp_path / "escaped").exists()


def test_global_budget_marks_queued_files_not_run(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {
            "test_a.py": "import time\ndef test_a():\n time.sleep(60)\n",
            "test_b.py": "def test_b():\n pass\n",
        },
        "--workers",
        "1",
        "--timeout",
        "1",
    )
    assert result.returncode == 1
    assert [entry["status"] for entry in summary] == ["TIMEOUT", "NOT RUN"]


def test_collection_failure_prints_traceback(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {
            "test_import.py": "raise RuntimeError('collection broke')\n",
        },
    )
    assert result.returncode == 1
    assert "collection broke" in result.stdout
    assert summary[0]["status"] == "FAILED"


def test_preserves_pytest_outcomes_and_teardown_errors(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {"test_outcomes.py": """
import pytest
@pytest.mark.xfail
def test_xfail():
    assert False
@pytest.mark.xfail
def test_xpass():
    pass
@pytest.mark.skip
def test_skip():
    pass
@pytest.fixture
def broken():
    yield
    raise RuntimeError('teardown broke')
def test_teardown(broken):
    pass
"""},
    )
    assert result.returncode == 1
    assert [case["status"] for case in summary[0]["tests"].values()] == [
        "XFAIL",
        "XPASS",
        "SKIPPED",
        "FAILED",
    ]
    assert "teardown broke" in result.stdout


def test_runs_files_concurrently_with_separate_state(tmp_path: Path):
    body = """
import os, time
from pathlib import Path
def test_state():
    Path('{own}').touch()
    deadline = time.monotonic() + 1
    while not Path('{other}').exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert Path('{other}').exists(), 'files did not overlap'
    Path('state-{own}').write_text(os.environ['TESTMON_DATAFILE'] + '\\n' + os.environ['COVERAGE_FILE'])
"""
    result, _ = run_files(
        tmp_path,
        {
            "test_a.py": body.format(own="a", other="b"),
            "test_b.py": body.format(own="b", other="a"),
        },
    )
    assert result.returncode == 0, result.stdout
    assert (tmp_path / "state-a").read_text() != (tmp_path / "state-b").read_text()


def test_no_selected_tests_is_success(tmp_path: Path):
    result, summary = run_files(tmp_path, {"test_empty.py": ""})
    assert result.returncode == 0, result.stdout
    assert summary[0]["returncode"] == 5


def test_partial_output_survives_timeout(tmp_path: Path):
    result, _ = run_files(
        tmp_path,
        {
            "test_output.py": "import time\ndef test_output():\n print('download started')\n time.sleep(60)\n"
        },
    )
    assert result.returncode == 1
    assert "download started" in result.stdout


def test_escalates_when_child_ignores_termination(tmp_path: Path):
    result, summary = run_files(
        tmp_path,
        {"test_ignore.py": """
import signal, time
def test_ignore():
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
"""},
    )
    assert result.returncode == 1
    assert summary[0]["returncode"] == -signal.SIGKILL


def test_cancellation_preserves_results_and_kills_children(tmp_path: Path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_cancel.py").write_text(
        "import os, time\nfrom pathlib import Path\n"
        "def test_cancel():\n Path('pid').write_text(str(os.getpid()))\n time.sleep(60)\n"
    )
    process = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            str(RUNNER),
            "--output",
            "results",
            "--grace",
            "0.1",
            "test_cancel.py",
        ],
        cwd=tmp_path,
        env=dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD="1"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "pid").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (tmp_path / "pid").exists()
        child_pid = int((tmp_path / "pid").read_text())
        process.terminate()
        output, _ = process.communicate(timeout=5)
        assert process.returncode == 1, output
        summary = json.loads((tmp_path / "results/summary.json").read_text())
        assert summary[0]["status"] == "INTERRUPTED"
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

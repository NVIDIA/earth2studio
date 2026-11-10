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

"""Run each data-source test file in a bounded, disposable POSIX process group."""

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path


def events(directory: Path) -> list[dict]:
    path = directory / "events.jsonl"
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # The process may have been killed during its final write.
    return records


def kill_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def stop(process: subprocess.Popen, directory: Path, grace: float) -> None:
    if process.poll() is None:
        if any(event["event"] == "ready" for event in events(directory)):
            try:
                os.kill(process.pid, signal.SIGUSR1)
            except ProcessLookupError:
                pass
            time.sleep(min(grace, 0.1))
        kill_group(process.pid, signal.SIGTERM)
        time.sleep(grace)
    # Also remove descendants left behind by a normally exiting pytest process.
    kill_group(process.pid, signal.SIGKILL)
    process.wait()


def report(
    file: str, directory: Path, code: int | None, reason: str, duration: float
) -> dict:
    records = events(directory)
    complete = any(event["event"] == "sessionfinish" for event in records)
    status = reason or (
        "ERROR" if not complete else "PASSED" if code in (0, 5) else "FAILED"
    )
    cases = {}
    for event in records:
        kind = event["event"]
        if kind == "collected":
            for node in event["nodeids"]:
                cases.setdefault(
                    node, {"status": "NOT RUN", "duration": 0.0, "detail": ""}
                )
        elif kind == "phase":
            cases[event["nodeid"]].update(status="INTERRUPTED", phase=event["phase"])
        elif kind == "report":
            case = cases[event["nodeid"]]
            case["duration"] += event["duration"]
            if event["phase"] == "call" or event["outcome"] != "PASSED":
                case["outcome"] = event["outcome"]
                case["detail"] = event["detail"]
        elif kind == "finished":
            case = cases[event["nodeid"]]
            case["status"] = case.get("outcome", "PASSED")
    for case in cases.values():
        if case["status"] == "INTERRUPTED" and status == "TIMEOUT":
            case["status"] = "TIMEOUT"
    if status == "PASSED" and any(
        case["status"] == "FAILED" for case in cases.values()
    ):
        status = "FAILED"
    title = f"{status} {file} ({duration:.1f}s, exit={code})"
    print(title, flush=True)
    github = os.environ.get("GITHUB_ACTIONS") == "true"
    print(f"::group::{title}" if github else f"--- {file} ---", flush=True)
    for nodeid, case in cases.items():
        print(
            f"{case['status']} {nodeid} ({case['duration']:.2f}s)"
            + (
                f" [{case['phase']}]"
                if case["status"] in {"TIMEOUT", "INTERRUPTED"}
                else ""
            )
        )
    if status != "PASSED" or any(event["event"] == "live_timeout" for event in records):
        print(
            f"File result: {status}; deadline includes collection and interpreter shutdown."
        )
        for log in (directory / "output.log", directory / "stacks.log"):
            if log.exists():
                # Indent child output so workflow-command-looking text stays plain text.
                with log.open(errors="replace") as stream:
                    for line in stream:
                        print("  " + line, end="")
    if github:
        print("::endgroup::", flush=True)
    return dict(
        file=file, status=status, returncode=code, duration=duration, tests=cases
    )


def write_summary(output: Path, results: list[dict]) -> None:
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    suites = ET.Element("testsuites")
    for result in results:
        suite = ET.SubElement(suites, "testsuite", name=result["file"])
        for nodeid, case in result["tests"].items():
            test = ET.SubElement(
                suite, "testcase", name=nodeid, time=str(case["duration"])
            )
            status = case["status"]
            if status in {"SKIPPED", "XFAIL", "NOT RUN"}:
                ET.SubElement(test, "skipped", message=status)
            elif status not in {"PASSED", "XPASS"}:
                ET.SubElement(test, "failure", message=status).text = case["detail"]
        if result["status"] != "PASSED":
            test = ET.SubElement(suite, "testcase", name="process")
            ET.SubElement(test, "error", message=result["status"]).text = (
                f"{result['file']}: {result['status']}, exit={result['returncode']}; see output.log"
            )
    ET.ElementTree(suites).write(
        output / "junit.xml", encoding="utf-8", xml_declaration=True
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers", type=int, default=int(os.getenv("DATA_TEST_WORKERS", "8"))
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=float(os.getenv("DATA_TEST_FILE_TIMEOUT", "300")),
    )
    parser.add_argument(
        "--timeout", type=float, default=float(os.getenv("DATA_TEST_TIMEOUT", "1200"))
    )
    parser.add_argument("--grace", type=float, default=2)
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/test-results/test-data/data")
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=shlex.split(os.getenv("DATA_TEST_PATHS", "test/data")),
    )
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = parser.parse_args(argv[:split])
    pytest_args = argv[split + 1 :]
    if any(
        not math.isfinite(value) or value <= 0
        for value in (args.workers, args.file_timeout, args.timeout, args.grace)
    ):
        parser.error("workers and time limits must be positive")
    files = sorted(
        {
            str(file)
            for path in args.paths
            for file in (
                sorted(Path(path).rglob("test_*.py"))
                if Path(path).is_dir()
                else [Path(path)]
            )
        }
    )
    if not files or any(not Path(file).is_file() for file in files):
        parser.error("test paths must contain existing test files")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "selection.json").write_text(
        json.dumps(
            dict(
                files=files,
                pytest_args=pytest_args,
                workers=args.workers,
                file_timeout=args.file_timeout,
                timeout=args.timeout,
                live=os.getenv("DATA_TEST_LIVE", "none"),
                base=os.getenv("DATA_TEST_BASE", "HEAD"),
            ),
            indent=2,
        )
        + "\n"
    )
    pending = deque(files)
    started_files = {}
    active = {}
    results = []
    deadline = time.monotonic() + args.timeout
    interrupted = False

    def interrupt(signum: int, frame: object) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        while pending or active:
            expired = time.monotonic() >= deadline or interrupted
            while pending and len(active) < args.workers and not expired:
                file = pending.popleft()
                key = str(Path(file).with_suffix("")).replace("/", "_")
                directory = output / key
                directory.mkdir(parents=True, exist_ok=True)
                first = file not in started_files
                if first:
                    (directory / "events.jsonl").write_text("")
                    (directory / "stacks.log").write_text("")
                    started_files[file] = time.monotonic()
                env = dict(
                    os.environ,
                    DATA_TEST_EVENTS=str(directory / "events.jsonl"),
                    DATA_TEST_STACKS=str(directory / "stacks.log"),
                    PYTHONUNBUFFERED="1",
                    DATA_TEST_COMPLETED=json.dumps(
                        [
                            event["nodeid"]
                            for event in events(directory)
                            if event["event"] == "finished"
                        ]
                    ),
                )
                env["PYTHONPATH"] = os.pathsep.join(
                    [str(Path(__file__).parent.resolve()), env.get("PYTHONPATH", "")]
                )
                # Never share writable testmon/coverage/pytest-cache files between children.
                database = Path(env.get("TESTMON_DATAFILE", ".testmondata")).resolve()
                database = database.parent / "data-files" / key / database.name
                database.parent.mkdir(parents=True, exist_ok=True)
                env["TESTMON_DATAFILE"] = str(database)
                env["COVERAGE_FILE"] = (
                    str(Path(env.get("COVERAGE_FILE", ".coverage.test-data")).resolve())
                    + "."
                    + key
                )
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-p",
                    "data_test_plugin",
                    *pytest_args,
                    "-s",
                    "-ra",
                    "--color=no",
                    "-o",
                    f"cache_dir={directory / 'pytest-cache'}",
                    file,
                ]
                print(f"RUNNING {file} (limit {args.file_timeout:g}s)", flush=True)
                started = started_files[file]
                with (directory / "output.log").open("w" if first else "a") as log:
                    process = subprocess.Popen(  # noqa: S603
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                active[file] = (process, directory, started)
                expired = time.monotonic() >= deadline or interrupted
            for file, (process, directory, started) in list(active.items()):
                elapsed = time.monotonic() - started
                code = process.poll()
                reason = ""
                live = None
                for event in events(directory):
                    if event["event"] in {"ready", "report", "finished"}:
                        live = None
                    elif event["event"] == "phase":
                        live = event if event.get("timeout") else None
                if code is None:
                    if interrupted:
                        reason = "INTERRUPTED"
                    elif time.monotonic() >= deadline or elapsed >= args.file_timeout:
                        reason = "TIMEOUT"
                    elif live and time.monotonic() >= live["started"] + live["timeout"]:
                        reason = "LIVE TIMEOUT"
                if code is not None or reason:
                    stop(process, directory, args.grace)
                    if reason == "LIVE TIMEOUT":
                        with (directory / "events.jsonl").open("a") as stream:
                            for event in [
                                dict(event="live_timeout", nodeid=live["nodeid"]),
                                dict(
                                    event="report",
                                    nodeid=live["nodeid"],
                                    phase="call",
                                    outcome="XFAIL",
                                    duration=live["timeout"],
                                    detail=f"Live call exceeded {live['timeout']}s; process group terminated",
                                ),
                                dict(event="finished", nodeid=live["nodeid"]),
                            ]:
                                stream.write(json.dumps(event) + "\n")
                        del active[file]
                        pending.appendleft(file)
                        continue
                    results.append(
                        report(file, directory, process.returncode, reason, elapsed)
                    )
                    del active[file]
                    write_summary(output, results)
            if expired:
                while pending:
                    file = pending.popleft()
                    if file in started_files:
                        key = str(Path(file).with_suffix("")).replace("/", "_")
                        results.append(
                            report(
                                file,
                                output / key,
                                None,
                                "INTERRUPTED" if interrupted else "TIMEOUT",
                                time.monotonic() - started_files[file],
                            )
                        )
                        continue
                    print(
                        f"NOT RUN {file} (global deadline or cancellation)", flush=True
                    )
                    results.append(
                        dict(
                            file=file,
                            status="NOT RUN",
                            returncode=None,
                            duration=0,
                            tests={},
                        )
                    )
                write_summary(output, results)
            if active:
                time.sleep(0.05)
    finally:
        for process, directory, _ in active.values():
            stop(process, directory, args.grace)
        write_summary(output, results)
    print(f"Results: {output}", flush=True)
    return int(interrupted or any(result["status"] != "PASSED" for result in results))


if __name__ == "__main__":
    sys.exit(main())

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

"""Benchmark: raw-BUFR NNJA fetch (before) vs nnja-ai Parquet fetch (after).

Compares fetching the same NNJA-archived observations for HealDA through two
different transports:

- "before": :class:`earth2studio.data.NNJAObsSat` /
  :class:`earth2studio.data.NNJAObsConv`, which download raw BUFR/PrepBUFR
  cycle files from the NNJA S3 mirror and decode them message-by-message with
  pybufrkit.
- "after": :class:`earth2studio.data.NNJAAIObsSat` /
  :class:`earth2studio.data.NNJAAIObsConv`, which read the same archive's
  pre-decoded Parquet catalog (published by the ``nnja-ai`` package) directly
  as columnar data -- no BUFR parsing.

Satellite obs (amsua/atms/mhs) are a like-for-like comparison: both sources
read the exact same NCEP aggregate microwave archive, just through different
transports. Conventional obs are *not* like-for-like: ``NNJAObsConv`` reads
the merged, QC'd PrepBUFR product, while ``NNJAAIObsConv`` reads the raw
per-family ADPUPA/ADPSFC dump streams (the nnja-ai catalog does not currently
publish the merged product), so its numbers are reported separately with
that caveat.

Usage
-----
python recipes/eval/nnja_ai_benchmark.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from earth2studio.data import NNJAObsConv, NNJAObsSat
from earth2studio.data.nnja_ai import NNJAAIObsConv, NNJAAIObsSat

ANALYSIS_TIME = datetime(2024, 1, 1, 0)
TOLERANCE = timedelta(minutes=30)
SAT_VARIABLES = ["amsua", "mhs"]  # atms omitted: ~5x the rows of the others


@dataclass
class BenchResult:
    label: str
    variable: str
    seconds: float
    rows: int
    error: str | None = None


def _time_call(label: str, variable: str, fn) -> BenchResult:
    t0 = time.perf_counter()
    try:
        df = fn()
        elapsed = time.perf_counter() - t0
        return BenchResult(label, variable, elapsed, len(df))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the benchmark
        elapsed = time.perf_counter() - t0
        return BenchResult(label, variable, elapsed, 0, error=str(exc)[:200])


def run_satellite_benchmark() -> list[BenchResult]:
    results = []
    for variable in SAT_VARIABLES:
        old = NNJAObsSat(time_tolerance=TOLERANCE, verbose=False)
        results.append(
            _time_call(
                "NNJAObsSat (raw BUFR, before)",
                variable,
                lambda v=variable, ds=old: ds(ANALYSIS_TIME, [v]),
            )
        )
        new = NNJAAIObsSat(time_tolerance=TOLERANCE)
        results.append(
            _time_call(
                "NNJAAIObsSat (nnja-ai Parquet, after)",
                variable,
                lambda v=variable, ds=new: ds(ANALYSIS_TIME, [v]),
            )
        )
    return results


def run_conventional_benchmark() -> list[BenchResult]:
    results = []
    old = NNJAObsConv(time_tolerance=TOLERANCE, verbose=False)
    results.append(
        _time_call(
            "NNJAObsConv (merged PrepBUFR, before)",
            "t",
            lambda: old(ANALYSIS_TIME, ["t"]),
        )
    )
    new = NNJAAIObsConv(time_tolerance=TOLERANCE)
    results.append(
        _time_call(
            "NNJAAIObsConv (raw ADPUPA Parquet, after)",
            "t",
            lambda: new(ANALYSIS_TIME, ["t"]),
        )
    )
    return results


def report(results: list[BenchResult]) -> None:
    df = pd.DataFrame(
        [
            {
                "source": r.label,
                "variable": r.variable,
                "seconds": round(r.seconds, 2),
                "rows": r.rows,
                "rows_per_sec": round(r.rows / r.seconds) if r.seconds > 0 else 0,
                "error": r.error or "",
            }
            for r in results
        ]
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    print(f"NNJA-AI benchmark: analysis_time={ANALYSIS_TIME}, tolerance={TOLERANCE}\n")

    print("=== Satellite (amsua, mhs): like-for-like, same archive ===")
    sat_results = run_satellite_benchmark()
    report(sat_results)

    print(
        "\n=== Conventional (t): NOT like-for-like -- see module docstring ==="
    )
    conv_results = run_conventional_benchmark()
    report(conv_results)

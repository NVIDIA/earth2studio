# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Check that reproduced members follow the tracks of the original run.

For models that are not bitwise reproducible on GPU, such as WeatherNext 2,
a reproduced member starts identical to the original and drifts slowly. Every
track of the original lasting ``--min-days`` has to reappear in the
reproduced member, starting at the same time, and stay within the tolerances
over its first ``--hours``. Weak systems drift most, their centres being
poorly defined: within 48 h, a few hPa and tens of kilometres, against
hundreds of kilometres between different members.

    python compare_tracks.py <original dir> <reproduced dir>
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.plt.data_handling import extract_tracks_from_file  # noqa: E402
from src.tc_hunt_utils import great_circle_distance  # noqa: E402


def compare(
    original: pd.DataFrame, reproduced: pd.DataFrame, args: argparse.Namespace
) -> list[str]:
    """Differences beyond the tolerances, one message each."""
    problems = []
    duration = original.groupby("track_id")["time"].agg(["min", "max"])
    long_tracks = duration[
        duration["max"] - duration["min"] >= pd.Timedelta(days=args.min_days)
    ]
    for track_id, (start, _) in long_tracks.iterrows():
        track = original[original["track_id"] == track_id]
        first = track.iloc[0]
        candidates = reproduced[reproduced["time"] == start]
        if candidates.empty:
            problems.append(f"track {track_id} starting {start} not reproduced")
            continue
        distance = great_circle_distance(
            first["lat"],
            first["lon"],
            candidates["lat"].to_numpy(),
            candidates["lon"].to_numpy(),
        )
        match = reproduced[
            reproduced["track_id"]
            == candidates["track_id"].iloc[int(np.argmin(distance))]
        ]
        window = track[track["time"] < start + pd.Timedelta(hours=args.hours)]
        pairs = window.merge(match, on="time", suffixes=("", "_rep"))
        if len(pairs) < len(window):
            problems.append(f"track {track_id}: reproduced track ends early")
            continue
        km = (
            great_circle_distance(
                pairs["lat"], pairs["lon"], pairs["lat_rep"], pairs["lon_rep"]
            )
            / 1000
        )
        wind = (pairs["wind_speed"] - pairs["wind_speed_rep"]).abs()
        hpa = (pairs["msl"] - pairs["msl_rep"]).abs() / 100
        for name, value, limit in (
            ("position [km]", km.max(), args.km),
            ("wind [m/s]", wind.max(), args.wind),
            ("pressure [hPa]", hpa.max(), args.hpa),
        ):
            if value > limit:
                problems.append(
                    f"track {track_id}: {name} differs by {value:.2f} > {limit}"
                )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("original")
    parser.add_argument("reproduced")
    parser.add_argument("--hours", type=float, default=48.0)
    parser.add_argument("--min-days", type=float, default=2.5)
    parser.add_argument("--km", type=float, default=100.0)
    parser.add_argument("--wind", type=float, default=5.0)
    parser.add_argument("--hpa", type=float, default=5.0)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.reproduced, "tracks_*.csv")))
    if not files:
        print(f"no track files in {args.reproduced}")
        return 1
    failed = False
    for path in files:
        name = os.path.basename(path)
        problems = compare(
            extract_tracks_from_file(os.path.join(args.original, name)),
            extract_tracks_from_file(path),
            args,
        )
        print(f"{name}: {'ok' if not problems else '; '.join(problems)}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

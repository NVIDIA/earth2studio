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
from dataclasses import dataclass, field, fields

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.plotting.data_handling import extract_tracks_from_file  # noqa: E402
from src.tc_hunt_utils import great_circle_distance, match_track  # noqa: E402


@dataclass(frozen=True)
class Tolerances:
    """Largest differences between an original and a reproduced track."""

    hours: float = field(
        default=48.0, metadata={"help": "hours compared from the start of a track"}
    )
    min_days: float = field(
        default=2.5, metadata={"help": "days a track lasts at least to be compared"}
    )
    km: float = field(default=100.0, metadata={"help": "centre position, in km"})
    wind: float = field(default=5.0, metadata={"help": "maximum wind, in m/s"})
    hpa: float = field(default=5.0, metadata={"help": "minimum pressure, in hPa"})


def compare(
    original: pd.DataFrame, reproduced: pd.DataFrame, tolerances: Tolerances
) -> list[str]:
    """Differences between the tracks of a member and its reproduction.

    Parameters
    ----------
    original : pd.DataFrame
        Tracks of the original member, see ``extract_tracks_from_file``
    reproduced : pd.DataFrame
        Tracks of the reproduced member
    tolerances : Tolerances
        Largest differences accepted

    Returns
    -------
    list[str]
        One message per difference beyond the tolerances, empty if none
    """
    problems = []
    duration = original.groupby("track_id")["time"].agg(["min", "max"])
    long_tracks = duration.index[
        duration["max"] - duration["min"] >= pd.Timedelta(days=tolerances.min_days)
    ]
    for track_id in long_tracks:
        track = original[original["track_id"] == track_id]
        window = track[
            track["time"] < track["time"].min() + pd.Timedelta(hours=tolerances.hours)
        ]
        match = match_track(reproduced, window)
        pairs = window.merge(match, on="time", suffixes=("", "_rep"))
        if len(pairs) < len(window):
            problems.append(
                f"track {track_id}: not reproduced over {tolerances.hours} h"
            )
            continue
        differences = {
            "position [km]": (
                great_circle_distance(
                    pairs["lat"], pairs["lon"], pairs["lat_rep"], pairs["lon_rep"]
                )
                / 1000,
                tolerances.km,
            ),
            "wind [m/s]": (
                pairs["wind_speed"] - pairs["wind_speed_rep"],
                tolerances.wind,
            ),
            "pressure [hPa]": ((pairs["msl"] - pairs["msl_rep"]) / 100, tolerances.hpa),
        }
        for name, (difference, limit) in differences.items():
            largest = abs(difference).max()
            if largest > limit:
                problems.append(
                    f"track {track_id}: {name} differs by {largest:.2f} > {limit}"
                )
    return problems


def main() -> int:
    """Compare every reproduced track file with the original one."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("original", help="track directory of the original run")
    parser.add_argument("reproduced", help="track directory of the reproduction")
    for tolerance in fields(Tolerances):
        parser.add_argument(
            f"--{tolerance.name.replace('_', '-')}",
            type=float,
            default=tolerance.default,
            help=tolerance.metadata["help"],
        )
    args = parser.parse_args()
    tolerances = Tolerances(
        **{t.name: getattr(args, t.name) for t in fields(Tolerances)}
    )

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
            tolerances,
        )
        print(f"{name}: {'ok' if not problems else '; '.join(problems)}")
        failed |= bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

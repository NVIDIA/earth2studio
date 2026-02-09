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

import argparse
from pathlib import Path

from compare_tracks import compare_netcdf_files


def find_matching_files(dir1: str, dir2: str) -> list[tuple[str, str]]:
    """Find files with matching names in two directories.

    Parameters
    ----------
    dir1 : str
        Path to first directory
    dir2 : str
        Path to second directory

    Returns
    -------
    list[tuple[str, str]]
        List of tuples containing matching file pairs
    """
    files1 = {f.name: f for f in Path(dir1).glob("*.nc")}
    files2 = {f.name: f for f in Path(dir2).glob("*.nc")}

    # Find common filenames
    common_names = set(files1.keys()) & set(files2.keys())

    # Return pairs of full paths
    return [(str(files1[name]), str(files2[name])) for name in common_names]


def main() -> None:
    """Main function to compare NetCDF files between two directories.

    This function:
    1. Parses command line arguments for directory paths (optional)
    2. Finds matching NetCDF files between the specified directories
    3. For each matching pair, calls compare_netcdf_files to compare their contents
    4. Reports any differences found in the data

    The script will exit if:
    - No matching files are found between the directories
    - An error occurs during file comparison

    Notes
    -----
    - Only compares files with identical names between directories
    - Default directories are outputs_helene/cyclones and outputs_reprod/cyclones
    """
    # Get the directory of this script
    script_dir = Path(__file__).parent

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Compare NetCDF files between two directories"
    )
    parser.add_argument(
        "--dir1",
        type=str,
        default=str(script_dir.parent / "outputs_helene" / "cyclones"),
        help="Path to first directory containing NetCDF files",
    )
    parser.add_argument(
        "--dir2",
        type=str,
        default=str(script_dir.parent / "outputs_reprod" / "cyclones"),
        help="Path to second directory containing NetCDF files",
    )

    args = parser.parse_args()

    # Find matching files
    matching_pairs = find_matching_files(args.dir1, args.dir2)

    if not matching_pairs:
        print("No matching files found between the directories!")
        exit()

    # Compare each pair
    for file1, file2 in matching_pairs:
        print(f"\nComparing {Path(file1).name} with {Path(file2).name}:")
        if not compare_netcdf_files(file1, file2):
            print("Not all files are identical")
            exit()


if __name__ == "__main__":
    main()

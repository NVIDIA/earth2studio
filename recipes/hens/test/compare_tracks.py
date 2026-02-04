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

import sys
from pathlib import Path

import numpy as np
import xarray as xr


def compare_netcdf_files(file1_path: str, file2_path: str) -> bool:
    """Compare data arrays from two NetCDF files.

    This function loads two NetCDF files and compares their data arrays, ignoring metadata.
    It provides detailed information about any differences found, including maximum and mean
    absolute differences, and the number of non-matching values.

    Parameters
    ----------
    file1_path : str
        Path to the first NetCDF file to compare.
    file2_path : str
        Path to the second NetCDF file to compare.

    Notes
    -----
    - The function assumes both files contain a data array named '__xarray_dataarray_variable__'.
    - NaN values are handled appropriately using np.allclose with equal_nan=True.
    - The comparison is done using numpy's allclose function with default tolerances.
    - If the shapes of the data arrays differ, the function will only report this and return.
    """
    # Verify files exist
    if not Path(file1_path).exists() or not Path(file2_path).exists():
        print("Error: One or both files do not exist")
        sys.exit(1)

    # Load the datasets
    ds1 = xr.open_dataset(file1_path)
    ds2 = xr.open_dataset(file2_path)

    # Get the data array (assuming it's the only data array in the file)
    data1 = ds1["__xarray_dataarray_variable__"]
    data2 = ds2["__xarray_dataarray_variable__"]

    # Compare shapes
    if data1.shape != data2.shape:
        print(f"Shapes differ: {data1.shape} vs {data2.shape}")
        return False

    # Compare values
    is_equal = np.allclose(data1.values, data2.values, equal_nan=True)

    if is_equal:
        print("Files are identical in terms of data values")
        return True
    else:
        # Calculate differences
        diff = np.abs(data1.values - data2.values)
        max_diff = np.nanmax(diff)
        mean_diff = np.nanmean(diff)
        print("Files differ:")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")

        # Count non-matching values
        non_matching = np.sum(~np.isclose(data1.values, data2.values, equal_nan=True))
        total_values = np.prod(data1.shape)
        print(f"Number of non-matching values: {non_matching} out of {total_values}")
        return False


def main() -> None:
    """Main entry point for the script.

    This function handles command line arguments and initiates the comparison
    between two NetCDF files. It performs basic validation of the input arguments
    and file existence before proceeding with the comparison.

    Notes
    -----
    - The script expects exactly two command line arguments: paths to the NetCDF files.
    - Both files must exist for the comparison to proceed.
    - The script will exit with status code 1 if the arguments are invalid or files don't exist.
    """
    if len(sys.argv) != 3:
        print("Usage: python compare_tracks.py <file1.nc> <file2.nc>")
        sys.exit(1)

    file1_path = sys.argv[1]
    file2_path = sys.argv[2]

    compare_netcdf_files(file1_path, file2_path)


if __name__ == "__main__":
    main()

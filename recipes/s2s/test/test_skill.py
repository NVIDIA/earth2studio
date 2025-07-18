# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from earth2studio.data import ARCO
from earth2studio.data.utils import fetch_data
from earth2studio.statistics import crps
from earth2studio.utils.coords import CoordSystem

expected_scores = {
    "dlesym": {
        "t2m": torch.tensor([2.1747, 2.0343, 2.3918, 1.6183, 3.0419, 1.9484, 1.7904]),
        "z500": torch.tensor(
            [288.6965, 474.8904, 291.3008, 469.3990, 468.8430, 210.8404, 270.3322]
        ),
    },
    "sfno": {
        "t2m": torch.tensor([4.3031, 4.0860, 3.7687, 4.2791, 5.4535, 3.8613, 4.2332]),
        "z500": torch.tensor(
            [143.6826, 205.8306, 263.3020, 398.4369, 504.6732, 375.4496, 235.7776]
        ),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).parent.parent
    model = args.model
    path = os.path.join(script_dir, args.path)

    if model not in ["dlesym", "sfno"]:
        raise ValueError("Model must be either 'dlesym' or 'sfno'")

    verif_lead_times = np.arange(4, 30, 4, dtype="timedelta64[D]")
    vars = ["t2m", "z500"]
    data_source = ARCO(verbose=False)
    metric = crps(
        ensemble_dimension="ensemble", reduction_dimensions=["lat", "lon"], fair=True
    )
    passed = {var: False for var in vars}

    for var in vars:
        with xr.open_zarr(path) as ds:
            # Load forecast data
            fcst = ds[var].isel(time=[0]).sel(lead_time=verif_lead_times)
            fcst_coords = CoordSystem(
                ensemble=fcst.ensemble.values,
                time=fcst.time.values,
                lead_time=fcst.lead_time.values,
                lat=fcst.lat.values,
                lon=fcst.lon.values,
            )

            # Load verification data
            interp_coords = OrderedDict(
                {
                    "_lat": fcst.lat.values,
                    "_lon": fcst.lon.values,
                }
            )
            verif, verif_coords = fetch_data(
                source=data_source,
                variable=var,
                time=ds.time.values,
                lead_time=verif_lead_times,
                interp_to=interp_coords,
            )
            verif = verif[:, :, 0, :, :]
            verif_coords.pop("variable")
            verif_coords["lat"], verif_coords["lon"] = (
                verif_coords["_lat"],
                verif_coords["_lon"],
            )
            del verif_coords["_lat"], verif_coords["_lon"]

            # Check within 5% of expected scores
            scores, score_coords = metric(
                torch.from_numpy(fcst.values), fcst_coords, verif, verif_coords
            )
            if torch.allclose(scores.squeeze(), expected_scores[model][var], rtol=5e-2):
                passed[var] = True
            else:
                print(f"Expected skill not verified for {model}:")
                for lt in range(len(expected_scores[model][var])):
                    print(f"Lead time {verif_lead_times[lt]}")
                    print(f"Expected: {expected_scores[model][var][lt]}")
                    print(f"Actual: {scores.squeeze()[lt]}")

    if all(passed.values()):
        print(f"Expected skill verified for {model}")


if __name__ == "__main__":
    main()

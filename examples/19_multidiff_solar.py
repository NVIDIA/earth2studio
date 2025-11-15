# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
os.environ["EARTH2STUDIO_PACKAGE_TIMEOUT"] = "10000"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import subprocess as sp
import argparse
from earth2studio.data import GFS, CDS, IFS
from earth2studio.io.netcdf4 import NetCDF4Backend
from earth2studio.models.auto import Package
from earth2studio.models.px import SFNO, InterpModAFNO

from collections import OrderedDict
from datetime import datetime
from math import ceil

import numpy as np
import torch
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import CorrDiffSolarMD
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, map_coords
from earth2studio.utils.time import to_time_array
import matplotlib.pyplot as plt
import copy
import earth2studio.run as run

# Load the default model package which downloads the check point from NGC
forecast_package = SFNO.load_default_package()
forecast_model = SFNO.load_model(forecast_package)

# Load the interpolation model
interp_package = InterpModAFNO.load_default_package()
prognostic_model = InterpModAFNO.load_model(interp_package, px_model=forecast_model)

# Load the downscaling model for SWDR
solarcorrdiff_package_path = CorrDiffSolarMD.load_default_package() 
solarcorrdiff_model = CorrDiffSolarMD.load_model(
        Package(solarcorrdiff_package_path, cache=False),
)

plt_lr = False

os.makedirs('outputs', exist_ok=True)
# Create the data source
data = GFS()


parser = argparse.ArgumentParser(description="Run inference for wind prediction model.")

parser.add_argument(
        '--start_time',
        type=str,  
        default='2024-07-01T06:00:00',
        help="Start time for inference in ISO 8601 format (e.g., 'YYYY-MM-DDTHH:MM:SS')."
)
parser.add_argument(
        '--timesteps',
        type=int,  
        default=1, 
        help="Number of timesteps to run inference for."
)

args = parser.parse_args()
start_time = args.start_time
timesteps = args.timesteps  
time= [start_time]  
output_file="./outputs/ChinaWdiff_wDiff_solar_{}.nc".format(start_time)
io = NetCDF4Backend(output_file)


# run inference
if plt_lr:
    with torch.no_grad():
        io,t_srx = run.diagnostic_solar(time, timesteps, prognostic_model, solarcorrdiff_model, data, io, plt_lr)
else:
    with torch.no_grad():
        io = run.diagnostic_solar(time, timesteps, prognostic_model, solarcorrdiff_model, data, io, plt_lr)

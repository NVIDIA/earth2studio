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

# %% [markdown]
# In this notebook, we will examine ensemble generation for Hurricane Helene, a tropical
# cyclone that made landfall in September 2024.
# The storm posed a challenging case for weather prediction systems and caused
# widespread impacts across the southeastern United States.
#
# The workflow is structured as follows: first, we will set up the most important
# configurations and initialize key objects, then explore their content.
# Following this, we will assemble the perturbation before running the inference.
# Finally, we will visualise the results by plotting the tracks and fields.
#
# **NOTE**: provide links to download (at least two) checkpoints and skill file.
#
# ## Configuring the pipeline
#
# The pipeline requires several configuration parameters to be set.
# We will define the most important ones here and then combine them into a configuration
# object.
#
# The key parameters include:
# - `project`: project name used for output file naming
# - `start_times`: time of initial conditions (multiple ICs can be specified)
# - `nsteps`: number of forecast steps
# - `nensemble`: ensemble size **per checkpoint and IC**
# - `batch_size`: number of forecast steps to run in parallel
# - `model_registry`: path to the registry of model packages
# - `max_num_checkpoints`: maximum number of checkpoints to use
#

# %%
project = "helene"

start_times = ["2024-09-24 12:00:00"]
nsteps = 8
nensemble = 4
batch_size = 1

model_registry = "hens_model_registry"
max_num_checkpoints = 2

output_vars = ["t2m", "u10m", "v10m", "u850", "v850", "msl", "z500"]
out_dir = "./outputs"


# %% [markdown]
# Next we'll load the configuration file using OmegaConf/Hydra.
# The config files are located in the cfg/ folder and use YAML format.
# The helene.yaml config contains the base settings for the Helene case study.
# We'll load it and then override some values programmatically below and apply the
# overrides.

# %%

from dotenv import load_dotenv
from omegaconf import OmegaConf
from physicsnemo.distributed import DistributedManager

load_dotenv()
DistributedManager.initialize()

# Create the configuration dictionary
from pathlib import Path

# Read the config from helene.yaml
cfg_cli = OmegaConf.from_cli()
config_path = cfg_cli.get("--config-path", "cfg")
config_name = cfg_cli.get("--config-name", "helene.yaml")
cfg = OmegaConf.load(Path(config_path) / Path(config_name))

# Overwriting via python
cfg["start_times"] = start_times
cfg["nsteps"] = nsteps
cfg["nensemble"] = nensemble
cfg["batch_size"] = batch_size

cfg["forecast_model"]["registry"] = model_registry
cfg["forecast_model"]["max_num_checkpoints"] = max_num_checkpoints

cfg["file_output"]["output_vars"] = output_vars

# Overrides via CLI
cfg = OmegaConf.merge(cfg, cfg_cli)

# %% [markdown]
# Next, using hydra conf, the required objects needed for the HENs workflow can get
# instantiated automatically using the `initialise`. If you are familiar with
# Earth2Studio workflows, this is essentially automating the creation of the core
# components such as data sources, prognostic and diagnostic models.

# %%

from src.hens_utilities import initialise

(
    ensemble_configs,
    model_dict,
    dx_model_dict,
    cyclone_tracker,
    data_source,
    output_coords_dict,
    base_random_seed,
    writer_executor,
    writer_threads,
) = initialise(cfg)

# %% [markdown]
# The initialisation provides us with a number of objects that will be used throughout
# the inference. Let's have a closer look at two of them:
#
# First, we explore the ensemble configs, a list of tuples containing information on
# which model package and initial conditions to use. Additionally, it contains ensemble
# indices and batch IDs assuring these are unique across all processes.
# Let's have a look at the content and explore how many cases we will run, depending on
# number of ensemble members, batch size, number of checkpoints and initial conditions.
#

# %%
for ii, (pkg, ic, ens_idx, batch_ids_produce) in enumerate(ensemble_configs):
    print(f"ensemble config {ii+1} of {len(ensemble_configs)}:")
    print(f"    package: {pkg}")
    print(f"    initial condition: {ic}")
    print(f"    ensemble index of first member: {ens_idx}")
    print(f"    batch ids to produce: {batch_ids_produce}\n")


# %% [markdown]
# Note that the inference is parallelised across ensemble config elements which
# corresponds to initial conditions.
# As a result, you cannot use more GPUs than number of ICs multiplied
# by number of checkpoints. If more GPUs are available, they remain idle.
#
# The model dict includes information about the model class and which model weights are
# currently loaded. In addition, it also holds a pointer to the model loaded to the GPU.
# During inference, the model dict gets updated according to the information provided in
# the ensemble config. Let's have a look at its contents:

# %%
from termcolor import colored

print(colored("The model class is:", attrs=["bold"]))
print(model_dict["class"], "\n")

print(
    colored("The model package (weights), which is currently loaded:", attrs=["bold"])
)
print(model_dict["package"], "\n")

print(colored("The fully initialised model is provided in:", attrs=["bold"]))
print(model_dict["model"].parameters, "\n")


# %% [markdown]
# Now bring everything together:
# - loop over ensemble configs (models)
# - update model dict (if package has changed)
# - initialise output
# - initialise perturbation (as ICs might have changed)
# - run inference, where all ensemble members are produced for a given checkpoint
# - write to disk
#
# This is implemented in the src/hens_run.py file which we encourage users to look at
# and customize as needed.
# The HENS run file can also be executed directly using Hydra CLI if desired.

# %%

from src.hens_run import run_inference

run_inference(
    cfg,
    ensemble_configs,
    model_dict,
    dx_model_dict,
    cyclone_tracker,
    data_source,
    output_coords_dict,
    base_random_seed,
    writer_executor,
    writer_threads,
)

# %% [markdown]
# After completing the ensemble generation process, the results are stored in the output
# directory. This directory contains both the forecast fields and the cyclone track
# data.
# A separate output file is created for each checkpoint-IC pair, so if you haven't
# changed the configs, there should be two NETCDF files with field data and one CSV file
# with the tracks.
#
# The field data includes:
# - 4 ensemble members
# - 1 initial condition (time)
# - 17 lead times for each forecast field (IC + 16 forecast steps)
#
# The track data includes detailed information about the storm's position, intensity,
# and other relevant meteorological parameters at each time step.
#
#
# We will now visualise these results. First, let us have a quick look at the global
# field after one day.
# Since the HENS models have a timestep size of 6 hours, the 4th lead time index is
# the forecast at 24 hours.

# %%

import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ds = xr.load_dataset(glob.glob(f"{out_dir}/gulf_of_mexico/*.nc")[0])

print(ds)
variable = "u10m"
lead_time = 4
ds[variable].isel(ensemble=0, lead_time=lead_time, time=0).plot(figsize=(16, 10))
plt.savefig(f"{out_dir}/helene_{variable}_{int(lead_time*6)}hours.jpg")

# %% [markdown]
# Now, let's focus on the Gulf of Mexico and plot the tracks of Hurricane Helene.
# Generating a spaghetti plot using cartopy can be done with just a few xarray
# operations.
# The plotting code below demonstrates how to filter our TC tracks that have a length
# greater than 2 steps.
# Additional filtering can be done depending on the use case.

# %%

# Create figure with cartopy projection
plt.close("all")
plt.figure(figsize=(10, 8))
projection = ccrs.LambertConformal(
    central_longitude=280.0, central_latitude=28.0, standard_parallels=(18.0, 38.0)
)
ax = plt.axes(projection=projection)

# Add map features
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, alpha=0.1)
ax.gridlines(draw_labels=True, alpha=0.6)
ax.set_extent([260, 300, 10, 40], crs=ccrs.PlateCarree())

track_files = glob.glob(f"{out_dir}/cyclones/*.nc")
for track_file in track_files:
    tracks = xr.load_dataarray(track_file)
    for ensemble in tracks.coords["ensemble"].values:
        for path in tracks.coords["path_id"].values:
            tracks_path = tracks.sel(ensemble=ensemble).isel(time=0)
            # Get lat/lon coordinates, filtering out nans
            lats = tracks_path.isel(path_id=path, variable=0)[:].values
            lons = tracks_path.isel(path_id=path, variable=1)[:].values
            mask = ~np.isnan(lats) & ~np.isnan(lons)
            if mask.any() and len(lons[mask]) > 2:
                ax.plot(
                    lons[mask],
                    lats[mask],
                    color="b",
                    linestyle="-.",
                    transform=ccrs.PlateCarree(),
                )

plt.savefig(f"{out_dir}/helene_tracks.jpg")


# %% [markdown]
# For the last step, we can create an animation to show the hurricane tracks.
# Since this post processing is a little more involved we've provided a API for plotting
# the Helene tracks.
# Users are encouraged to look a modify the implementation for their own needs.

# %%

from src.plot import create_track_animation_florida

track_files = glob.glob("outputs/cyclones/*.nc")
create_track_animation_florida(track_files, out_dir, fps=2)

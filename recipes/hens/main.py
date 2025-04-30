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

# %% [markdown]
# In this notebook, we will examine ensemble generation for Hurricane Helene, a tropical
# cyclone that made landfall in September 2024.
# The storm posed a challenging case for weather prediction systems and caused
# widespread impacts across the southeastern United States.
#
# The workflow is structured as follows: first, we will set up the most important
# configurations and initialise key objects, then explore their content.
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
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager

load_dotenv()
DistributedManager.initialize()

# Create the configuration dictionary
from pathlib import Path

# Read the config from helene.yaml
cfg_path = Path("cfg/helene.yaml")
with open(cfg_path) as f:
    cfg = DictConfig(f.read())

cfg["start_times"] = start_times
cfg["nsteps"] = nsteps
cfg["nensemble"] = nensemble
cfg["batch_size"] = batch_size

cfg["forecast_model"]["package"] = model_registry
cfg["forecast_model"]["max_num_checkpoints"] = max_num_checkpoints

cfg["file_output"]["output_vars"] = output_vars

# %% [markdown]
# Next, using hydra conf, the required objects needed for the HENs workflow can get
# instantiated automatically using the `initialise`. If you are familiar with
# Earth2Studio workflows, this is essentially automating the creation of the core
# components such as data sources, prognostic and diagnostic models.

# %%

from src.hens_utilities import (
    initialise,
    initialise_output,
    update_model_dict,
    write_to_disk,
)
from src.hens_utilities_ensemble import EnsembleBase
from src.hens_utilities_reproduce import create_base_seed_string

(
    ensemble_configs,
    model_dict,
    dx_model_dict,
    cyclone_tracking,
    data,
    output_coords_dict,
    base_random_seed,
    all_tracks_dict,
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
# Note that the inference is parallelised across enssemble config elements, hence across
# IC-package pairs. As a result, you cannot use more GPUs than number of ICs multiplied
# by number of checkpoints. If more GPUs are available, they remain idle.
#
# The model dict includes information about the model class and which model weights are
# currently loaded. In addtion, it also holds a pointer to the model loaded to the GPU.
# During infernce, the model dict gets updated according to the information provided in
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
# The final piece which is missing before we can run the inference is assembling the
# HENS perturbation. For this, we need to provide:
# - a skill file, which contains the deterministic skill of the forecast model
# (**Note**: provide links to download skill file, best in intro)
# - the variable to perturb in the seeding step of the bred vector perturbation
# - the number of integration steps for breeding the noise vector
# - the noise amplification, by which the noise vector is scaled
#
# With this information, we can now assemble the HENS perturbation using
# CorrelatedSphericalGaussian as seeding perturbation and HemisphericCentredBredVector
# as bred vector perturbation. To see how it is aseembled form basic blocks porvided in
# Earth2Studio, have a look into `11_hens/hens_perturbation.py`.

# %%
# for perturbation
skill_path = "hens_model_registry/d2m_sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16.nc"
noise_amplification = 0.35
perturbed_var = ["z500"]
integration_steps = 3

from numpy import datetime64, ndarray
from src.hens_perturbation import HENSPerturbation

from earth2studio.data import DataSource
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation


def initialise_perturbation(
    model: PrognosticModel,
    data: DataSource,
    start_time: ndarray[datetime64],
) -> Perturbation:
    """Helper method to initialize the perturbation"""
    perturbation = HENSPerturbation(
        model=model,
        data=data,
        start_time=start_time,
        skill_path=skill_path,
        noise_amplification=noise_amplification,
        perturbed_var=perturbed_var,
        integration_steps=integration_steps,
    )

    return perturbation


# %% [markdown]
# now bring everyhting together:
# - loop over ensemble configs
# - update model dict (if package has changed)
# - initialise output
# - initialise perturbation (as ICs might have changed)
# - run inference, where all ensemble members are produced
# - write to disk
#
#
# Now we will bring all components together to execute the ensemble forecasting process:
#
# - iterate through each ensemble configuration, which contains the necessary parameters
# for generating individual ensemble members.
# - at each iteration, update the model dictionary whenever a new package is
# encountered, ensuring the correct model weights are loaded.
# - initialise the output object and set up the perturbation method, taking into account
# any changes in the initial conditions.
# - initialise the perturbation method with updated IC and checkpoint
# - initialise the inference pipeline with updated IC and checkpoint
# - run inference, which generates all ensemble members according to the specified
# configuration.
# - write the results to disk, ensuring that all forecast data and associated metadata
# are properly stored for subsequent analysis.
#

# %%
# loop over ensemble configs
for pkg, ic, ens_idx, batch_ids_produce in ensemble_configs:
    # create seed base string required for reproducibility of individual batches
    base_seed_string = create_base_seed_string(pkg, ic, base_random_seed)

    # load new weights if necessary
    model_dict = update_model_dict(model_dict, pkg)

    # create new io object
    io_dict = initialise_output(cfg, ic, model_dict, output_coords_dict)

    # initialise perturbation with updated IC and checkpoint
    perturbation = initialise_perturbation(
        model=model_dict["model"], data=data, start_time=ic
    )

    # initialise inference pipeline with updated IC and checkpoint
    run_hens = EnsembleBase(
        time=[ic],
        nsteps=cfg.nsteps,
        nensemble=cfg.nensemble,
        prognostic=model_dict["model"],
        data=data,
        io_dict=io_dict,
        perturbation=perturbation,
        output_coords_dict=output_coords_dict,
        dx_model_dict=dx_model_dict,
        cyclone_tracking=cyclone_tracking,
        batch_size=cfg.batch_size,
        ensemble_idx_base=ens_idx,
        batch_ids_produce=batch_ids_produce,
        base_seed_string=base_seed_string,
    )

    # run inference
    io_dict = run_hens()

    # if in-memory flavour of io backend was chosen, write content to disk now
    if io_dict:
        writer_threads, writer_executor = write_to_disk(
            cfg,
            ic,
            model_dict,
            io_dict,
            writer_threads,
            writer_executor,
        )

if writer_executor is not None:
    for thread in list(writer_threads):
        thread.result()
        writer_threads.remove(thread)
    writer_executor.shutdown()


# %% [markdown]
# After completing the ensemble generation process, the results are stored in the output
# directory. This directory contains both the forecast fields and the cyclone track
# data.
# A seperate output file is created for each checkpoint-IC pair, so if you haven't
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

ds = xr.load_dataset(glob.glob(f"{out_dir}/global/*.nc")[0])

variable = "u10m"
lead_time = 4
ds[variable].isel(ensemble=0, lead_time=lead_time, time=0).plot(figsize=(16, 6))
plt.savefig(f"{out_dir}/helene_{variable}_{int(lead_time*6)}hours.jpg")

# %% [markdown]
# Now, let's focus on the Gulf of Mexico and plot the tracks of Hurricane Helene.
# Generating a speghetti plot using cartopy can be done with just a few xarray
# operations.
# The plotting code below

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
            tracks_path = tracks.sel(ensemble=ensemble).isel(time=0, lead_time=0)
            # Get lat/lon coordinates, filtering out nans
            lats = tracks_path.isel(path_id=path, variable=0)[:].values
            lons = tracks_path.isel(path_id=path, variable=1)[:].values
            mask = ~np.isnan(lats) & ~np.isnan(lons)
            if mask.any() and len(lons[mask]) > 2:
                print(tracks.shape, lats[mask].shape)
                print(lons[mask])
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
create_track_animation_florida(track_files, out_dir, fps=10)

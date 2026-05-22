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

# %% [markdown]
# # Analyse Tropical Cyclone Track Ensembles
#
# This notebook demonstrates how to analyse and validate ensemble tropical
# cyclone (TC) track predictions. Use this to compare forecast tracks against
# observations
# ([IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive))
# and reanalysis-based reference tracks (ERA5). We explain the workflow using
# the example of
# [Hurricane Helene](https://en.wikipedia.org/wiki/Hurricane_Helene), but it
# can be easily configured to investigate other storms.
#
# ### Workflow Overview
# 1. Configure case and paths for predicted ensemble and reference track
# 2. Plot track trajectories (spaghetti plot)
# 3. Plot absolute intensities over time (wind speed, MSLP)
# 4. Plot relative intensities over time (normalised by reference)
# 5. Compare ERA5 reference against IBTrACS observations
# 6. Analyse extreme value statistics (histograms)
# 7. Compute error moments over lead time
#
# ### Data Prerequisites
# - Ensemble of predicted TC tracks (CSV files from tracking algorithm)
# - Reference track from ERA5 or IBTrACS observations (CSV file)
#
# Both can be produced by configuring a `tc_hunt.py` run with cyclone tracking
# enabled. The ensemble of predicted tracks can, for example, be produced with
# the config `cfg/helene.yaml`. Reference tracks can be extracted using
# `cfg/extract_era5.yaml`.
#
# ### Analysis Notes
# - Track positions are typically well-represented by the model
# - Intensity metrics (wind speed, pressure) show larger biases due to the
#   coarse (~0.25°) resolution, which cannot fully resolve TC structure
# - IBTrACS provides observed best track data; ERA5 represents the best
#   achievable reference for reanalysis-driven forecasts
#
# ### Running the notebook
# This script (and its sibling helpers) imports from `data_handling.py`,
# `plotting_helpers.py` and `analyse_n_plot.py` using bare module names, so
# the working directory must be `recipes/tc_tracking/plotting/`. From the
# recipe root:
#
# ```bash
# cd plotting
# jupytext --to notebook tracks_slayground_notebook.py
# jupyter notebook tracks_slayground.ipynb
# ```
#
# ### Configuration
# - `case` - named storm to analyse. Can be expanded with additional named
#   storms by following the pattern `{name}_{YYYY}_{basin}`
# - `pred_track_dir` - folder containing predicted track CSVs (if using data
#   produced with `cfg/helene.yaml`, this would be
#   `/path/to/outputs_helene/cyclone_tracks_te`)
# - `tru_track_dir` - folder containing reference track CSV file (if using
#   data produced with `cfg/extract_era5.yaml`, this would be
#   `/path/to/outputs_reference_tracks/`)
# - `out_dir` - path for storing plots
# - `time_step` - cadence of the predictions; defaults to 6 h to match the
#   stock FCN3 / AIFS-ENS configurations

# %%
import numpy as np
from analyse_n_plot import load_tracks
from data_handling import compute_averages_of_errors_over_lead_time
from plotting_helpers import (
    plot_errors_over_lead_time,
    plot_extreme_extremes_histograms,
    plot_ib_era5,
    plot_over_time,
    plot_relative_over_time,
    plot_spaghetti,
)

# case = 'amphan_2020_north_indian'
# case = 'beryl_2024_north_atlantic'
# case = 'debbie_2017_southern_pacific'
# case = 'dorian_2019_north_atlantic'
# case = 'harvey_2017_north_atlantic'
case = "hato_2017_west_pacific"
# case = 'helene_2024_north_atlantic'
# case = 'ian_2022_north_atlantic'
# case = 'iota_2020_north_atlantic'
# case = 'irma_2017_north_atlantic'
# case = 'lan_2017_west_pacific'
# case = 'lee_2023_north_atlantic'
# case = 'lorenzo_2019_north_atlantic'
# case = 'maria_2017_north_atlantic'
# case = 'mawar_2023_west_pacific'
# case = 'michael_2018_north_atlantic'
# case = 'milton_2024_north_atlantic'
# case = 'ophelia_2017_north_atlantic'
# case = 'yagi_2024_west_pacific'
# case = 'erin_2025_north_atlantic'

pred_track_dir = "/path/to/outputs_hato/cyclone_tracks_te"
tru_track_dir = "/path/to/outputs_reference_tracks"
out_dir = "./plots"
time_step = np.timedelta64(6, "h")

tru_track, pred_tracks, ens_mean, n_members, out_dir = load_tracks(
    case=case,
    pred_track_dir=pred_track_dir,
    tru_track_dir=tru_track_dir,
    out_dir=out_dir,
    time_step=time_step,
)

# %% [markdown]
# ### Spaghetti Plot
#
# - **Ensemble members** are shown in grey
# - **Ensemble mean** is displayed in green
# - **ERA5 reference** is shown in red

# %%
plot_spaghetti(
    true_track=tru_track,
    pred_tracks=pred_tracks,
    ensemble_mean=ens_mean["mean"],
    case=case,
    n_members=n_members,
    out_dir=out_dir,
)

# %% [markdown]
# ### Plot Absolute Intensities and Track Distance Over Time
#
# This section examines the temporal evolution of cyclone intensities and the
# distance from the reference track. Intensities are represented by minimum
# sea level pressure and maximum wind speed. The reference track should mostly
# fall within the ensemble spread for intensity predictions. However, some
# storms exhibit phenomena such as rapid intensification that cannot be
# adequately captured by models on quarter-degree resolution. In such cases,
# the reference intensity may lie outside the ensemble spread, indicating
# model limitations in resolving fine-scale processes.

# %%
plot_over_time(
    pred_tracks=pred_tracks,
    tru_track=tru_track,
    ensemble_mean=ens_mean,
    case=case,
    n_members=n_members,
    out_dir=out_dir,
    time_step=time_step,
)

# %% [markdown]
# ### Plot Relative Intensities Over Time
#
# This cell shows the same intensity metrics as in the previous cell, this
# time normalised by the reference. Note that for the pressure field we
# normalise the deviation from normal pressure. This normalisation helps to
# identify systematic biases in the ensemble predictions and highlights
# periods where the model over- or underestimates cyclone intensity relative
# to observations.

# %%
plot_relative_over_time(
    pred_tracks=pred_tracks,
    tru_track=tru_track,
    ensemble_mean=ens_mean,
    case=case,
    n_members=n_members,
    out_dir=out_dir,
    time_step=time_step,
)

# %% [markdown]
# ### Plot ERA5 Against IBTrACS Variables
#
# This cell compares the intensities reached in the ERA5 reanalysis data
# against those obtained from IBTrACS observations. The deviation between
# both datasets is usually larger the more intense the storm becomes. Note
# that there are two separate y-axes for the different intensity metrics
# (pressure and wind speed).

# %%
plot_ib_era5(
    tru_track=tru_track,
    case=case,
    variables=["msl", "wind_speed"],
    out_dir=out_dir,
)

# %% [markdown]
# ### Extreme Values Over Lifetime of Storm
#
# This cell computes the maximum intensity reached along each track
# throughout the storm's lifetime and displays the distribution across
# ensemble members as histograms. For comparison, the extreme values from the
# reference track are shown as vertical lines.

# %%
plot_extreme_extremes_histograms(
    pred_tracks=pred_tracks,
    tru_track=tru_track,
    ensemble_mean=ens_mean,
    case=case,
    out_dir=out_dir,
)

# %% [markdown]
# ### Statistics
#
# This cell computes error metrics as a function of lead time across all
# ensemble members. The following statistics are calculated: mean absolute
# error, root mean square error, and standard deviation for wind speed,
# pressure intensity, and track distance.

# %%
variables = ["wind_speed", "msl", "dist"]

err_dict, _ = compute_averages_of_errors_over_lead_time(
    pred_tracks=pred_tracks,
    tru_track=tru_track,
    variables=variables,
)

plot_errors_over_lead_time(
    err_dict=err_dict,
    case=case,
    ic=pred_tracks[0]["ic"],
    n_members=n_members,
    n_tracks=len(pred_tracks),
    out_dir=out_dir,
    time_step=time_step,
)

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
# # How Strong Is the Storm? Two Trackers on One WeatherNext 2 Ensemble
#
# A forecast model predicts fields; how strong a tropical cyclone is in those
# fields is decided by the *tracker* that reads them. This notebook runs two
# very different trackers on the same ensemble members of
# [WeatherNext 2 Cyclones](https://nvidia.github.io/earth2studio/main/modules/generated/models/px/WeatherNext2Cyclones/)
# (WN2) and compares the storm intensity they report with the observed best
# track. For context, it adds TempestExtremes on two other ensemble models
# started from the same initial conditions,
# [FourCastNet 3](https://nvidia.github.io/earth2studio/main/modules/generated/models/px/FCN3/)
# (FCN3) and
# [AIFS-ENS](https://nvidia.github.io/earth2studio/main/modules/generated/models/px/AIFSENS/).
#
# ### The two trackers
#
# | | TempestExtremes (TE) | WN2 built-in tracker |
# | --- | --- | --- |
# | Reads | the forecast fields: sea-level pressure, 10 m wind, geopotential | cyclone fields the network predicts next to the weather |
# | Finds storms by | pressure minima with a closed contour and a warm core | the predicted probability that a cyclone is present |
# | Pressure | grid-point minimum of the 0.25° field | central pressure as in the best track |
# | Wind | largest grid-point wind within 5° of the centre | 1-minute sustained wind as in the best track |
# | Works with | any model | WN2 only |
#
# TE measures what the forecast fields *contain*. Two things keep that below
# the observed intensity: a 0.25° grid (about 28 km) cannot hold the eyewall of
# an intense hurricane, whose strongest winds sit within 10 to 30 km of the
# centre, and the fields of a machine-learning forecast tend to be smoother
# than those of the analysis it starts from. WN2 takes a shortcut: it was
# trained on
# [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)
# best tracks to predict the quantities forecasters report, and its tracker
# reads those predictions. Comparing both on the same members, and TE on the
# analysis itself, separates these effects.
#
# ### The model
#
# WN2 is a Functional Generative Network (FGN): each ensemble member is one
# pass of the network with its own random perturbation of the network's
# parameters. It steps 6 h at a time on a 0.25° grid. The checkpoint in
# Earth2Studio is the one that ran during the 2025 hurricane season, trained on
# data up to 2024, so the 2025 storms shown here are not part of its training.
#
# - Model and tracker: Alet et al. (2026), *Operational tropical cyclone
#   forecasting with AI*, [Nature](https://doi.org/10.1038/s41586-026-10953-2)
# - FGN: Alet et al. (2025), *Skillful joint probabilistic weather forecasting
#   from marginals*, [arXiv:2506.10772](https://arxiv.org/abs/2506.10772)
# - Code and checkpoints:
#   [google-deepmind/weathernext](https://github.com/google-deepmind/weathernext)
# - TempestExtremes: Ullrich et al. (2021),
#   [GMD](https://doi.org/10.5194/gmd-14-5023-2021)
# - IBTrACS: Knapp et al. (2010),
#   [BAMS](https://doi.org/10.1175/2009BAMS2755.1)
#
# ### The storms
#
# | Storm | Basin | Forecast start | What happened |
# | --- | --- | --- | --- |
# | [Melissa](https://en.wikipedia.org/wiki/Hurricane_Melissa) | North Atlantic | 24 Oct 2025, 00 UTC | Tropical storm that intensified for three days to 82 m/s before landfall in Jamaica on 28 Oct |
# | [Ragasa](https://en.wikipedia.org/wiki/Typhoon_Ragasa) | West Pacific | 18 Sep 2025, 12 UTC | Gained 35 m/s in 24 h on 20 Sep, peaked near 75 m/s east of the Luzon Strait |
#
# ### Data prerequisites
#
# None with `use_sample_tracks = True` below: the tracks of the forecasts shown
# here come with the recipe, in `aux_data/sample_tracks`, and IBTrACS is
# downloaded from NOAA on first use (about 300 MB). To produce the tracks
# yourself, see section 6.1 of the recipe's README:
#
# 1. The WN2 forecast of the storm, with both trackers (an 80 GB GPU per
#    process): `python main.py --config-name=melissa_wn2.yaml`. It writes
#    `outputs_melissa_wn2/cyclone_tracks_wn2/` and `cyclone_tracks_te/`, one
#    file per member, under the same name in both.
# 2. Optional: the same storm with FCN3 and AIFS-ENS, tracked by TE only.
# 3. Optional: TE's track of the storm in the IFS analysis the forecasts start
#    from, `python main.py --config-name=extract_ifs.yaml`, as a measure of
#    what a 0.25° grid can show at best.
#
# ### Running the notebook
# With the analysis environment (`analysis/pyproject.toml`, see section 3 of
# the recipe's README) activated, from the `analysis` directory:
#
# ```bash
# cd analysis
# jupytext --to notebook wn2_tracker_comparison.py
# jupyter lab
# ```
#
# ### Configuration
#
# - `storm` - the storm, one of `STORMS`; add a storm there with its IBTrACS
#   name, season and basin
# - `use_sample_tracks` - use the tracks that come with the recipe, or read
#   your own runs from `runs_dir`
# - `runs_dir` - directory holding your runs' outputs, `outputs_<storm>_wn2`
#   and so on, as written by `main.py`; runs that do not exist are left out
# - `ibtracs_csv` - where IBTrACS is, or will be, stored

# %%
import os

import matplotlib.pyplot as plt

if not os.path.isdir("plotting"):
    raise RuntimeError(
        f"run the notebook from the recipe's analysis directory, not {os.getcwd()}"
    )

# importing plotting also makes the recipe's src package importable
from plotting.comparison_plots import (  # noqa: E402
    plot_over_time,
    plot_rapid_intensification,
    plot_storm_size,
    plot_tracks,
    plot_wind_pressure,
)
from plotting.storm_ensembles import (  # noqa: E402
    Storm,
    load_analysis_track,
    load_runs,
    peak_intensity_table,
)
from src.data.ibtracs import load_ibtracs_storm  # noqa: E402

STORMS = {
    "melissa": Storm("melissa", 2025, "north_atlantic"),
    "ragasa": Storm("ragasa", 2025, "west_pacific"),
}
storm = STORMS["melissa"]

use_sample_tracks = True
runs_dir, run_prefix = "..", "outputs_"  # main.py writes ../outputs_<storm>_<model>
if use_sample_tracks:
    runs_dir, run_prefix = "../aux_data/sample_tracks", ""
ibtracs_csv = "../aux_data/ibtracs.ALL.list.v04r01.csv"

# %% [markdown]
# ### Load the Best Track
#
# IBTrACS collects the best tracks of all agencies, which do not measure wind
# the same way: the US agencies report 1-minute sustained winds, most others
# 10-minute winds, which are 10 to 15 % lower for the same storm.
# `load_ibtracs_storm` reads the US columns in every basin, the convention WN2
# predicts, and converts them to the units of the track files (Pa, m/s).
#
# Best tracks are revised for about a year after a season. Until then they are
# provisional, and the loader says so.

# %%
best_track = load_ibtracs_storm(ibtracs_csv, storm.name, storm.season)
# early points can lack intensities, before the US agency took the storm over
columns = ["time", "lat", "lon", "msl", "wind_speed", "rmw_km", "nature"]
best_track[columns].dropna(subset=["wind_speed"]).head()

# %% [markdown]
# ### Load the Forecast Tracks
#
# Both trackers find every cyclone on the globe, so each member's file holds
# several tracks. `load_runs` keeps the one that follows the storm: the track
# closest to the best track at the first time any track comes within 300 km of
# it. A member in which no track does is counted as a miss.
#
# All track files share one layout, whichever tracker wrote them, so one
# function reads them all. `RUNS` in `plotting/storm_ensembles.py` lists the
# runs, their colours and where their files are.

# %%
ensembles = load_runs(runs_dir, storm, best_track, run_prefix)

reference_csv = os.path.join(
    runs_dir, f"{run_prefix}reference_tracks_ifs", storm.reference_file
)
analysis_track = (
    load_analysis_track(reference_csv) if os.path.isfile(reference_csv) else None
)

for ensemble in ensembles.values():
    found = ensemble.tracks["member"].nunique()
    print(
        f"{ensemble.run.label}: storm found in {found} of {ensemble.n_members} members"
    )

# %% [markdown]
# ### Where Does the Storm Go?
#
# Before comparing intensities, check that the runs follow the same storm. The
# two trackers on WN2 run on the same members, so their tracks should lie on
# top of each other: both locate the same pressure minimum, only by different
# means.

# %%
plot_tracks(ensembles, best_track, storm)
plt.show()

# %% [markdown]
# The distance to the best track over time. In this and the following plots,
# each thin, pale line is one member and the bold line the ensemble mean.

# %%
plot_over_time(ensembles, "track_error_km", storm)
plt.show()

# %% [markdown]
# ### Maximum Wind
#
# The two trackers do not report the same quantity:
#
# - **WN2 tracker:** the 1-minute sustained wind, the best track's own measure.
# - **TE:** the strongest wind at any grid point within 5° of the centre, an
#   average over a grid cell of about 28 km.
#
# The dashed line is TE on the IFS analysis: what a 0.25° field can show at
# best. Where TE on the forecasts stays below it, the forecast fields have lost
# intensity against the analysis they start from; where the analysis itself
# stays below the best track, the grid is the limit. The WN2 tracker reports
# best-track quantities and is subject to neither.

# %%
plot_over_time(ensembles, "wind_speed", storm, best_track, analysis_track)
plt.show()

# %% [markdown]
# ### Minimum Pressure
#
# Pressure varies more smoothly in space than wind, so a coarse grid captures
# more of it: expect TE to come closer to the best track here than for wind,
# and the gap to grow as the storm deepens and its core contracts.

# %%
plot_over_time(ensembles, "msl", storm, best_track, analysis_track)
plt.show()

# %% [markdown]
# ### Peak Intensity in Numbers
#
# The strongest wind and lowest pressure each member reaches during the
# forecast: mean over the members, with the 10th to 90th percentile in
# brackets, against the best track over the same period.

# %%
peak_intensity_table(ensembles, best_track)

# %% [markdown]
# ### Wind-Pressure Relationship
#
# Every track point of every member, wind against pressure. In a real storm
# the two are tightly linked: the deeper the low, the stronger the wind.
#
# - Points **below** the best track have too little wind for their pressure,
#   the signature of fields too smooth for the eyewall.
# - Points **along** it show a tracker that reproduces the observed
#   relationship, here learnt from the best tracks.

# %%
plot_wind_pressure(ensembles, best_track, storm)
plt.show()

# %% [markdown]
# ### Rapid Intensification
#
# A storm intensifies rapidly if its maximum wind rises by at least 15.4 m/s
# within 24 h. The plot shows, at each time, the share of members whose wind
# rose that much over the preceding 24 h; the shaded times are those the storm
# itself did.
#
# A fixed wind threshold is a hard test for a tracker reading grid-scale winds:
# if it sees only half of the storm's wind, it also sees only half of its
# increase.

# %%
plot_rapid_intensification(ensembles, best_track, storm)
plt.show()

# %% [markdown]
# ### Storm Size
#
# The WN2 tracker also predicts the size of the storm, which TE does not
# report: the radius of maximum wind and the radius of 17.5 m/s wind, here
# averaged over the four quadrants. The best track's radius of maximum wind is
# an estimate, reviewed by the agency only in the North Atlantic and East
# Pacific.

# %%
plot_storm_size(ensembles["wn2"], best_track, storm)
plt.show()

# %% [markdown]
# ### Findings with the Sample Tracks
#
# The sample tracks are 10-day forecasts of 32 members, one WN2 network. For
# both storms:
#
# - **Tracks:** all runs find the storm in every member. After four days, the
#   mean track error is about 140 to 170 km for Melissa, similar for all
#   models, and 70 to 90 km for Ragasa, except for FCN3 (about 190 km); after
#   a week, several hundred kilometres.
# - **Peak intensity:** WN2's tracker comes close to the best track, 83 m/s and
#   905 hPa for Melissa against 85 m/s and 892 hPa, 73 m/s against 74 m/s for
#   Ragasa. TE on the three models' fields reaches 28 to 35 m/s and 957 to
#   976 hPa on average.
# - **Grid or forecast:** TE on the IFS analysis finds 46 m/s for Melissa and
#   59 m/s for Ragasa, well above TE on any forecast. So the grid is not the
#   only limit: the forecasts lose intensity against the analysis they start
#   from.
# - **Decay:** Melissa made landfall in Jamaica on 28 Oct and crossed Cuba on
#   29 Oct; the observed wind fell from 80 to about 41 m/s within a day. WN2's
#   tracker weakens the storm far more slowly, keeping it near 60 m/s until
#   30 Oct. For Ragasa, it stays about 5 m/s below the best track from the peak
#   on.
# - **Wind-pressure relationship:** WN2's points follow the observed
#   relationship; TE's points on all three models lie below it.
# - **Rapid intensification:** every WN2 member intensifies rapidly at some
#   point, for both storms; according to TE, a single member does, of WN2 for
#   Ragasa.
#
# ### What to Keep in Mind
#
# - **The trackers answer different questions.** TE reports what the forecast
#   fields contain; the WN2 tracker reports what the network expects a
#   forecaster to analyse. Neither is the other's error.
# - **Two storms are two storms.** The findings above come from two cases,
#   32 members each and one WN2 network; they illustrate the method, they are
#   not a verification.
# - **Wind definitions differ.** Best track and WN2 tracker use 1-minute
#   sustained wind, TE an instantaneous grid-cell wind. Compare TE with the
#   analysis track to judge the forecast, and with the best track to judge the
#   resolution.
# - **Best tracks are estimates.** Atlantic storms are measured by aircraft.
#   West Pacific intensities rest mostly on satellite estimates, and tracks of
#   recent seasons are provisional.
# - **The WN2 tracker starts from scratch.** It is run without the storm's
#   observed position at the start of the forecast, so it has to find the storm
#   itself, and reports it only once it has followed it for 2.5 days.
# - **One checkpoint, a small ensemble.** The operational system combines four
#   independently trained networks and at least 50 members; `model_kwargs` in
#   the config selects the network.
#
# To look at the fields of an interesting member, reproduce it with
# `cfg/reproduce_melissa_wn2.yaml` and plot it with `plot_tracks_n_fields.py`.

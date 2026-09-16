---
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
label: StormCast
category: Prognostic models
domain: CONUS
px_class: StormCast
short: StormCast is NVIDIA's generative km-scale model for the central United States.
---

StormCast is NVIDIA's generative convection-allowing model. It forecasts 99
HRRR variables on a 3 km window of the HRRR grid over the central United
States with a 1-hour step, conditioned on coarse global fields. A regression
network makes a first guess and a diffusion network corrects it, so an
ensemble comes from the sampler's noise rather than from perturbed initial
conditions.

## Event-based scoring

Unlike the global scorecards, this page comes from an event campaign.
Each event is a space-time window over the evaluation data showcasing a physical event.
Initial conditions run every 3 hours from 12 hours before the window to its end.
The headline curves pool every event's initial conditions, and the Event selector
shows one episode at a time. Truth is the HRRR analysis, and scores are uniformly weighted on the model grid. The campaign definition is
[stormcast_2025_events.yaml](https://github.com/NVIDIA/earth2studio/blob/main/recipes/eval/scorecard/cfg/campaign/stormcast_2025_events.yaml)
and the event scoring lives in the
[evaluation recipe](https://github.com/NVIDIA/earth2studio/tree/main/recipes/eval).

## Reference

Pathak, J., Cohen, Y., Garg, P., Harrington, P., Brenowitz, N., Durran,
D., Mardani, M., Vahdat, A., Xu, S., Kashinath, K., and Pritchard, M.
(2026). Kilometer-scale convection-allowing model emulation using
generative diffusion modeling. Science Advances, 12(5), eadv0423.

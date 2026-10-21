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
label: Aurora 1.5 Ensemble
category: Prognostic models
px_class: Aurora1p5Ensemble
short: Aurora 1.5 Ensemble is the stochastic variant of the Aurora 1.5 foundation model from Microsoft.
---

Aurora 1.5 Ensemble is the stochastic checkpoint of the Aurora 1.5
atmospheric foundation model from Microsoft. Each forward pass injects
fresh Gaussian noise into the backbone conditioning. A batch of identical
initial conditions yields independent members. The version scored here
runs 16 members at 0.25° on a 720 × 1440 grid. It consumes the two most
recent analysis frames, t-6 h and t0. Its state advances every 6 hours
while it predicts every hour, so this scorecard carries hourly lead
times, verified against hourly ERA5. It covers the outputs ERA5
provides, including dew point, cloud covers, skin temperature, soil
temperature, soil moisture, sea ice, and snow depth.

## Reference

Bodnar, C., et al. (2025). A foundation model for the Earth system.
Nature, 641, 1180–1187.
[doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y).

Microsoft (2025). Aurora 1.5 model card.
[huggingface.co/microsoft/aurora](https://huggingface.co/microsoft/aurora).

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
label: GraphCast
category: Prognostic models
px_class: GraphCastOperational
short: GraphCast is a graph neural network weather model from Google DeepMind.
---

GraphCast is a graph neural network weather model from Google DeepMind that
runs on a multi-mesh icosahedral representation of the globe. The version
scored here is the operational configuration with 13 pressure levels at
0.25°. It consumes the two most recent analysis frames (t-6 h and t0) plus
solar forcing. Each step advances 6 hours on the native ERA5 721 × 1440
grid.

## Reference

Lam, R., et al. (2023). Learning skillful medium-range global weather
forecasting. Science, 382(6677), 1416-1421.

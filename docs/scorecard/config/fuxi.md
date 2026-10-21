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
label: FuXi
category: Prognostic models
px_class: FuXi
short: FuXi is Fudan University's cascaded Swin-transformer forecast system.
---

FuXi is a medium-range forecasting system from Fudan University. It
cascades three Swin-transformer models, each fine-tuned for a different lead
window: 0 to 5 days, 5 to 10 days, and 10 to 15 days. The rollout hands over
between models as the forecast advances. The version scored here
consumes the two most recent ERA5 analysis frames, 6 hours apart, and forecasts
70 variables at 0.25° resolution with a 6-hour step. Its 13 relative-humidity
levels have no ERA5 counterpart in the verification store and are not scored.

## Reference

Chen, L., et al. (2023). FuXi: A cascade machine learning forecasting
system for 15-day global weather forecast. npj Climate and Atmospheric
Science, 6, 190.
[doi:10.1038/s41612-023-00512-1](https://doi.org/10.1038/s41612-023-00512-1).

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
label: FengWu
category: Prognostic models
px_class: FengWu
short: FengWu is a multi-modal, multi-task transformer from Shanghai AI Laboratory.
---

FengWu is a medium-range forecasting model from Shanghai AI Laboratory. It
treats each atmospheric variable as its own modality with a dedicated encoder
and decoder and fuses them in a shared transformer. Training uses an
uncertainty-weighted multi-task loss and a replay buffer that exposes the
model to its own forecasts. The version scored here consumes the two most
recent ERA5 analysis frames, 6 hours apart, and forecasts 69 variables at
0.25° resolution with a 6-hour step.

## Reference

Chen, K., Han, T., Gong, J., Bai, L., Ling, F., Luo, J.-J., Chen, X., Ma, L.,
Zhang, T., Su, R., Ci, Y., Li, B., Yang, X., and Ouyang, W. (2023). FengWu:
pushing the skillful global medium-range weather forecast beyond 10 days
lead. [arxiv.org/abs/2304.02948](https://arxiv.org/abs/2304.02948).
[arxiv.org/abs/2304.02948](https://arxiv.org/abs/2304.02948).

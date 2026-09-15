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
label: Aurora
category: Prognostic models
px_class: Aurora
short: Aurora is a foundation model of the atmosphere from Microsoft Research.
---

Aurora is a foundation model of the atmosphere from Microsoft Research: a
1.3B-parameter Swin-transformer with Perceiver-style encoders pretrained on
over a million hours of diverse weather and climate data. The version scored
here is the 0.25° deterministic medium-range configuration, which consumes the
two most recent analysis frames (t-6h and t0) and steps forward 6 hours at a
time on a 720x1440 grid (pole-padded onto ERA5's 721x1440 for verification).

## Reference

Bodnar, C., Bruinsma, W. P., Lucic, A., Stanley, M., Brandstetter, J.,
Garvan, P., ... & Perdikaris, P. (2024). Aurora: A foundation model of the
atmosphere. arXiv preprint arXiv:2405.13063, 1(8).

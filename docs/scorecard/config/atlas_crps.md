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
label: Atlas CRPS
category: Prognostic models
px_class: AtlasCRPS
short: Atlas CRPS is the ensemble variant of NVIDIA's Atlas weather model.
---

Atlas CRPS is the CRPS-trained ensemble variant of NVIDIA's Atlas weather
model. Every forward pass draws a fresh noise vector that modulates each
transformer block, so repeated calls from the same initial condition yield
calibrated ensemble members. The version scored here runs a 16-member
ensemble at 0.25°. It consumes the two most recent analysis frames (t-6 h
and t0) and steps forward 6 hours at a time on the native ERA5 721 × 1440
grid.

## Reference

Kossaifi, J., et al. (2026). Demystifying data-driven probabilistic
medium-range weather forecasting.
[arXiv:2601.18111](https://arxiv.org/abs/2601.18111).

NVIDIA (2026). Atlas ERA5 model card.
[huggingface.co/nvidia/atlas-era5](https://huggingface.co/nvidia/atlas-era5).

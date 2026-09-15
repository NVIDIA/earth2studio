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
label: FCN3
category: Prognostic models
px_class: FCN3
short: FourCastNet 3 is NVIDIA's probabilistic machine-learning weather model.
---

FourCastNet 3 is NVIDIA's probabilistic machine-learning weather model, built
on spherical (geometric) signal processing with a hidden-Markov ensemble
formulation: each member evolves its own calibrated stochastic state, so the
ensemble spread is learned rather than imposed by initial-condition
perturbations. It forecasts 72 atmospheric variables globally at 0.25°
resolution with a 6-hour step.

## Reference

Bonev, B., Kurth, T., Mahesh, A., Bisson, M., Kossaifi, J., Kashinath, K.,
... & Keller, A. (2025). FourCastNet 3: A geometric approach to probabilistic
machine-learning weather forecasting at scale. arXiv preprint
arXiv:2507.12144.

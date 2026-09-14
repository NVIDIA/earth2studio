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
label: SFNO
category: Prognostic models
px_class: SFNO
short: SFNO is NVIDIA's Spherical Fourier Neural Operator weather model.
---

SFNO is NVIDIA's Spherical Fourier Neural Operator weather model, the
FourCastNet v2 architecture. It replaces the planar Fourier layers of the
original FourCastNet with spherical harmonic transforms, so the operator
respects the geometry of the globe and stays stable over long forecasts. The
version scored here is the deterministic 73-variable configuration at 0.25°
resolution with a 6-hour step, initialized from a single ERA5 analysis frame.

## Reference

Bonev, B., Kurth, T., Hundt, C., Pathak, J., Baust, M., Kashinath, K., & Anandkumar, A. (2023, July). Spherical fourier neural operators: Learning stable dynamics on the sphere. In International conference on machine learning (pp. 2806-2823). PMLR.
[arxiv.org/abs/2306.03838](https://arxiv.org/abs/2306.03838).

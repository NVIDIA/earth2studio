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
label: DLWP
category: Prognostic models
px_class: DLWP
short: DLWP is a compact convolutional model on the cubed sphere from the University of Washington.
---

Deep Learning Weather Prediction (DLWP) is a compact convolutional model from
the University of Washington. It maps the globe onto a cubed sphere and
applies a U-Net-style convolutional network on the cube faces, which avoids
the polar distortion of a lat/lon grid. The version scored here consumes the
two most recent ERA5 analysis frames, 6 hours apart. It forecasts seven
variables with a 6-hour step: z at 1000, 700, 500, and 300 hPa,
850 hPa temperature, 2 m temperature, and total column water vapour. The cube
faces map back to the 0.25° grid for verification.

## Reference

Weyn, J. A., Durran, D. R., & Caruana, R. (2020). Improving data‐driven global weather prediction using deep convolutional neural networks on a cubed sphere. Journal of Advances in Modeling Earth Systems, 12(9), e2020MS002109.
[doi:10.1029/2020MS002109](https://doi.org/10.1029/2020MS002109).

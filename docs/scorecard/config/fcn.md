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
label: FCN
category: Prognostic models
px_class: FCN
short: FourCastNet is NVIDIA's original Adaptive Fourier Neural Operator weather model.
---

FourCastNet (FCN) is NVIDIA's original data-driven global weather model. It
uses the Adaptive Fourier Neural Operator, a vision-transformer backbone whose
token mixing runs in Fourier space, which made 0.25° global forecasting
tractable at the time. The version scored here is the deterministic
26-variable configuration with a 6-hour step, initialized from a single ERA5
analysis frame. It runs on a 720 by 1440 grid that the pipeline maps onto ERA5's
721 by 1440 for verification. Its two relative-humidity levels have no
counterpart in the verification store, so the scorecard skips them.

## Reference

Kurth, T., Subramanian, S., Harrington, P., Pathak, J., Mardani, M., Hall,
D., Miele, A., Kashinath, K., and Anandkumar, A. (2023). FourCastNet:
accelerating global high-resolution weather forecasting using adaptive
Fourier neural operators. In Proceedings of the Platform for Advanced
Scientific Computing Conference (PASC '23), 1-11.
[arxiv.org/abs/2202.11214](https://arxiv.org/abs/2202.11214).

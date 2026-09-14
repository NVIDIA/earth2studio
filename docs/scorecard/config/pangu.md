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
label: Pangu
category: Prognostic models
px_class: Pangu6
short: Pangu-Weather is Huawei's 3D Earth-specific transformer for medium-range forecasting.
---

Pangu-Weather is Huawei's 3D Earth-specific transformer for medium-range
forecasting. It encodes the 13 pressure levels and the surface fields into a
single 3D token volume and applies Earth-specific positional biases in its
attention blocks. Pangu ships separate networks for 1, 3, 6, and 24-hour
steps. The configuration scored here interleaves the 24-hour and 6-hour
networks. Each whole day advances with 24-hour steps and the 6-hour network
fills the intermediate leads, so daily leads match the 24-hour model exactly.
It forecasts 69 variables at 0.25° resolution from a single ERA5 analysis
frame.

## Reference

Bi, K., Xie, L., Zhang, H., Chen, X., Gu, X., & Tian, Q. (2023). Accurate medium-range global weather forecasting with 3D neural networks. Nature, 619(7970), 533-538.
[doi:10.1038/s41586-023-06185-3](https://doi.org/10.1038/s41586-023-06185-3).

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
# Site-wide scorecard defaults, read by docs/generate_scorecard.py.
#
# labels: fallback display names for models that have no config/<model>.md
#         (a per-model config's `label:` always wins).
# metrics.lower_is_better: metrics whose skill curves read "lower is better";
#         drives the plot's subtitle. Anything not listed reads higher-better
#         (acc) or has its own rule (spread_skill targets 1.0).
# baselines: reference runs (exported like any model) that the plot overlays
#         as dashed lines instead of giving them their own scorecard pages.
baselines:
  persistence: Persistence
  climatology: Climatology
# initializations: alternative initial-condition sources. A model scored
#         from these analyses is exported as eval_scores_<model>_<key>*.json
#         beside its ERA5 run; its page gets an Initialization selector in
#         the plot, an Initialization section (the `description`) ahead of
#         the plot, and the `reference` joins the page's Reference section,
#         rather than a scorecard of its own. A plain string is the label alone.
initializations:
  healda:
    label: HealDA v2
    description: >-
      The Initialization selector in the plot switches between forecasts
      started from ARCO-ERA5 and from HealDA v2. On the HealDA v2 view the
      ERA5-initialized run stays as the dashed curve for reference, and both
      runs are verified against ERA5. HealDA is an AI data assimilation model:
      a neural network on a HEALPix mesh that turns a window of raw
      conventional and satellite observations from the NNJA archive directly
      into a global analysis. So a forecast initialized from it runs end to
      end from observations to prediction.
    reference: >-
      Gupta, A., Subramaniam, A., Pritchard, M. S., Kashinath, K., Frolov, S.,
      Lieberman, K., ... & Brenowitz, N. D. (2026). HealDA: Highlighting the
      importance of initial errors in end-to-end AI weather forecasts. arXiv
      preprint [arXiv:2601.17636](https://arxiv.org/abs/2601.17636).
labels:
  fcn3: FCN3
  aurora: Aurora
  sfno: SFNO
  fengwu: FengWu
  ucast: UCast
  graphcast: GraphCast
  graphcast_small: GraphCast-small
  pangu3: Pangu (3 h)
  pangu6: Pangu (6 h)
  pangu24: Pangu (24 h)
# data_sources: display names for earth2studio data source classes shown in
#         the Evaluation table (falls back to the raw class name).
data_sources:
  ARCO_ERA5: ERA5 (ARCO)
  GFS: GFS
  HRRR: HRRR
  HealDAv2: HealDA v2 analyses
metrics:
  lower_is_better: [rmse, mae, lsd, ensemble_mean_mse, crps, ensemble_variance]
---

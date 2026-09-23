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

"""Building blocks for Earth2Studio workflows that publish site forecasts to DSX.

A producer is the model-specific part of a workflow. It runs or connects to a model and converts
the model output into the shared ``ForecastSeries`` format. The code is divided into these areas:

- ``dsx/`` contains shared DSX publishing code, including authentication, MQTT communication,
  message conversion, validation, queueing, and reconnect handling.
- ``stormcast/`` contains StormCast-specific workflow code, output collection, variable
  definitions, and conditioning setup.
- ``sfno/`` contains SFNO-specific workflow code, output collection, and variable definitions.
- ``shared/`` contains reusable helpers for forecast data, site extraction, lead times, derived
  weather variables, wind rotation, data availability, and cache cleanup.

Three import rules keep these areas independent. The ``dsx`` package must not import model-specific
producers, one producer must not import another, and ``shared`` must not import ``dsx`` or a
producer. ``test/test_architecture.py`` checks all three rules.

Only the producer workflow modules and ``stormcast.conditioning`` import Earth2Studio or PyTorch.
The remaining modules can be tested without a GPU.
"""

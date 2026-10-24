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

"""Couple prognostic models that feed each other, as a single runner.

``earth2studio.run`` never imports this package; a :class:`CoupledRunner` is an
ordinary :class:`~earth2studio.run.Runner` that lives here.
"""

from earth2studio.coupler.runner import CoupledRunner, Link, window_mean

__all__ = ["CoupledRunner", "Link", "window_mean"]

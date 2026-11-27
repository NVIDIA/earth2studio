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

"""Plotting and analysis helpers of the recipe's analysis notebooks.

The helpers use a few functions of the recipe's ``src`` package, such as the
track matcher and the IBTrACS reader. Importing this package makes ``src``
importable, from the notebooks in ``analysis/`` as from anywhere else.
"""

import os
import sys

_RECIPE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _RECIPE_ROOT not in sys.path:
    sys.path.insert(0, _RECIPE_ROOT)

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

__version__ = "0.9.0a0"

# Deprecation warnings
import sys
import warnings

if sys.version_info[:2] == (3, 10):
    warnings.warn(
        "Python 3.10 support will be dropped in Earth2Studio versions after 0.9.0 "
        "in accordance with NEP 29/SPEC 0. Please upgrade to Python 3.11 or newer.",
        DeprecationWarning,
        stacklevel=2,
    )

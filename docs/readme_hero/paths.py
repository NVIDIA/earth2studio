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

import os

# All intermediate data, panels and the final images are written here.
WORK = os.environ.get("E2S_HERO_DIR", os.path.join(os.getcwd(), "hero_out"))
os.makedirs(os.path.join(WORK, "panels"), exist_ok=True)


def P(rel: str) -> str:
    """Resolve a path relative to the hero working directory."""
    return os.path.join(WORK, rel)

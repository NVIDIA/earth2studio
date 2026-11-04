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

from earth2studio.data import Random
from earth2studio.utils.coords import E2S_STATISTICS
from earth2studio.utils.time_statistics import time_statistic_metadata


class RandomStatistics(Random):
    """Synthetic model inputs, including already-aggregated random fields.

    Model tests need correctly labelled inputs, not a time series to reduce.
    """

    lexicon = type("Lexicon", (), {"TIME_STATISTICS": ("mean", "sum", "min", "max")})

    def __call__(self, time, variable):
        array = super().__call__(time, variable)
        array.attrs[E2S_STATISTICS] = {
            label: time_statistic_metadata(label.partition(":")[2])
            for label in array.coords["variable"].values
            if ":" in label
        }
        return array

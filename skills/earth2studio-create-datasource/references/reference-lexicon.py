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

"""Reference lexicon skeleton for new Earth2Studio data sources.

Copy this to ``earth2studio/lexicon/<source_name>.py``, rename the class,
and replace each ``# FILL:`` section.
"""

from collections.abc import Callable

from earth2studio.lexicon.base import LexiconType


class SourceNameLexicon(metaclass=LexiconType):
    """Lexicon for SourceName data source.

    Note
    ----
    Variable documentation: FILL: URL
    """

    VOCAB = {
        # FILL: Map Earth2Studio variable names to remote keys.
        # Surface variables: descriptive abbreviation.
        "t2m": "2m_temperature",
        "u10m": "10m_u_component_of_wind",
        # Pressure-level variables: {short_name}{level_hPa}.
        "t500": "temperature::500",
        "z500": "geopotential::500",
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return remote key and modifier function for a variable.

        Parameters
        ----------
        val : str
            Earth2Studio variable name.

        Returns
        -------
        tuple[str, Callable]
            Remote key string and post-processing function.
        """
        remote_key = cls.VOCAB[val]
        # FILL: Add unit conversions or transforms as needed.
        if val.startswith("z"):
            return remote_key, lambda x: x * 9.81
        return remote_key, lambda x: x

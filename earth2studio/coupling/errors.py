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

"""Errors raised by coupling declarations and coupled runs."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


class CouplingError(ValueError):
    """A coupling declaration is invalid, or a coupled run cannot proceed."""


def _suggest(name: str, options: Iterable[str]) -> str:
    """Return a did-you-mean suffix for ``name``, or an empty string."""
    matches = difflib.get_close_matches(name, list(options), n=1)
    return f"; did you mean {matches[0]!r}?" if matches else ""

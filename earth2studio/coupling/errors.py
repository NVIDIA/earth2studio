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

"""The coupler's one error type."""

import difflib
from collections.abc import Iterable


class CouplingError(ValueError):
    """A coupled system is misconfigured or cannot exchange a field.

    Subclasses :class:`ValueError`, so one ``except ValueError`` covers models,
    regridders and the coupler alike. Messages name the models and variables
    involved and, where possible, the fix.
    """


def _suggest(name: str, candidates: Iterable[str], n: int = 3) -> str:
    """Return a ``Did you mean`` suffix from close matches, or an empty string."""
    matches = difflib.get_close_matches(name, list(candidates), n=n, cutoff=0.5)
    if not matches:
        return ""
    return f" Did you mean: {', '.join(repr(m) for m in matches)}?"

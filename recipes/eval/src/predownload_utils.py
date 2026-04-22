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

"""Shared helpers used by pipelines to declare predownload requirements.

These were previously private module-level functions in ``predownload.py``.
Moving them into a neutral utility module lets concrete ``Pipeline``
subclasses reuse them to build :class:`~src.pipeline.PredownloadStore`
entries without a circular import back into the entry-point script.
``predownload.py`` re-exports these under their original names for
backward compatibility with existing tests and any callers that imported
the private helpers.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from earth2studio.utils.coords import CoordSystem


def infer_step_hours(model: object) -> int:
    """Infer the model's output timestep in hours from its coordinate methods.

    Computed as the difference between the first output lead time and the last
    input lead time, which equals the model's intrinsic step size.
    """
    ic_coords = model.input_coords()  # type: ignore[attr-defined]
    out_coords = model.output_coords(ic_coords)  # type: ignore[attr-defined]
    delta = out_coords["lead_time"][0] - ic_coords["lead_time"][-1]
    return int(delta / np.timedelta64(1, "h"))


def compute_verification_times(
    ic_times: list[np.datetime64],
    nsteps: int,
    step_hours: int,
) -> list[np.datetime64]:
    """Collect all unique model-output valid times across every IC window."""
    step = np.timedelta64(step_hours, "h")
    times: set[np.datetime64] = set()
    for t in ic_times:
        for k in range(nsteps + 1):
            times.add(t + k * step)
    return sorted(times)


def union_variables(*var_lists: list[str]) -> list[str]:
    """Return the ordered union of multiple variable lists."""
    seen: set[str] = set()
    result: list[str] = []
    for vl in var_lists:
        for v in vl:
            if v not in seen:
                result.append(v)
                seen.add(v)
    return result


def squeeze_lead_time(
    x: torch.Tensor, coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Remove the lead_time dimension from a (tensor, coords) pair.

    ``fetch_data()`` always returns a ``lead_time`` dimension even when it
    contains a single ``[0ns]`` entry.  The predownload stores have no
    ``lead_time`` dimension, so we squeeze it out before writing.
    """
    lt_idx = list(coords.keys()).index("lead_time")
    x = x.squeeze(lt_idx)
    coords = OrderedDict((k, v) for k, v in coords.items() if k != "lead_time")
    return x, coords

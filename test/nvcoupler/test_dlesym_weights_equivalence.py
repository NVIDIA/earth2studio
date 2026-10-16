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

"""The real-weights equivalence gate for the DLESyM split adapter.

Every other dlesym_split test runs against a MockDLESyM authored from a
reading of dlesym.py — which makes them structurally circular: a misreading
of the model (normalization order, insolation times, window chunking) would
pass all of them and fail on real weights. THIS test is the actual proof:
driving the split components through nvcoupler must reproduce the native
``DLESyM.__call__`` output on the same input, with real checkpoints.

It is skipped unless the environment can run it (physicsnemo installed and
the hf://nvidia/dlesym-v1-era5 package fetchable — several GB; set
NVCOUPLER_DLESYM_WEIGHTS=1 to opt in). Until it has passed somewhere,
treat "nvcoupler can host DLESyM" as unverified.
"""

import os
from collections import OrderedDict

import numpy as np
import pytest
import torch

requires_weights = pytest.mark.skipif(
    os.environ.get("NVCOUPLER_DLESYM_WEIGHTS") != "1",
    reason=(
        "real-weights equivalence gate: set NVCOUPLER_DLESYM_WEIGHTS=1 with "
        "physicsnemo installed and network/cache access to the "
        "hf://nvidia/dlesym-v1-era5 package (several GB)"
    ),
)


def native_final_lead_slab(
    y_native: torch.Tensor, y_coords: OrderedDict, var_index: int
) -> torch.Tensor:
    """Select variable ``var_index`` at the LAST lead time from a native
    DLESyM output tensor laid out per ``y_coords``.

    The first ``select`` removes the variable axis, which renumbers every
    axis AFTER it by -1 while axes before it keep their indices. In the
    native layout (batch, time, lead_time, variable, ...) lead_time sits
    BEFORE variable, so its index must NOT be shifted — ``lead_axis - 1``
    would silently select the time axis instead.
    """
    var_axis = list(y_coords).index("variable")
    lead_axis = list(y_coords).index("lead_time")
    slab = y_native.select(var_axis, var_index)
    reduced_lead_axis = lead_axis if lead_axis < var_axis else lead_axis - 1
    return slab.select(reduced_lead_axis, -1)


def test_native_final_lead_slab_axis_arithmetic():
    """Cheap structural check of the comparison helper: the selected slab
    must be exactly (var, last-lead) for the native DLESyM axis order, with
    the lead axis NOT shifted after the variable select removes a later axis.
    Runs without weights so the gate's axis math is verified by execution.
    """
    coords = OrderedDict(
        {
            "batch": np.arange(1),
            "time": np.arange(2),
            "lead_time": np.arange(3),
            "variable": np.arange(2),
            "face": np.arange(2),
            "height": np.arange(2),
            "width": np.arange(2),
        }
    )
    shape = tuple(len(v) for v in coords.values())
    x = torch.zeros(shape)
    for lead in range(shape[2]):
        for var in range(shape[3]):
            # encode (var, lead) so any wrong-axis select is detectable
            x[:, :, lead, var] = 100 * var + lead

    for var in range(shape[3]):
        got = native_final_lead_slab(x, coords, var)
        assert got.shape == (1, 2, 2, 2, 2)  # variable and lead_time removed
        assert torch.all(got == 100 * var + (shape[2] - 1))

    # a layout with variable BEFORE lead_time exercises the shifted branch
    coords_vl = OrderedDict(
        {
            "batch": np.arange(1),
            "variable": np.arange(2),
            "lead_time": np.arange(3),
            "face": np.arange(2),
        }
    )
    y = torch.zeros(1, 2, 3, 2)
    for var in range(2):
        for lead in range(3):
            y[:, var, lead] = 100 * var + lead
    for var in range(2):
        got = native_final_lead_slab(y, coords_vl, var)
        assert got.shape == (1, 2)
        assert torch.all(got == 100 * var + 2)


@requires_weights
def test_split_adapter_matches_native_dlesym():
    pytest.importorskip("physicsnemo", reason="DLESyM checkpoints need physicsnemo")
    from earth2studio.models.px import DLESyM
    from earth2studio.nvcoupler.dlesym_split import build_dlesym_driver

    model = DLESyM.load_model(DLESyM.load_default_package())
    model.eval()

    # Equivalence must hold for ANY input, so a random (but finite,
    # reasonably-scaled) state on the native input coords suffices — no data
    # source needed. Batch/time dims of size 1 replace the empty wildcards.
    ic = model.input_coords()
    coords = OrderedDict(
        {
            "batch": np.array([0]),
            "time": np.array([np.datetime64("2024-01-01")]),
            **{k: v for k, v in ic.items() if k not in ("batch", "time")},
        }
    )
    shape = tuple(len(v) for v in coords.values())
    torch.manual_seed(0)
    x = torch.randn(shape)

    # native: one coupled 96h step
    with torch.inference_mode():
        y_native, y_coords = model(x.clone(), coords)

    # nvcoupler: the split components driven for one coupling step
    driver = build_dlesym_driver(model, start="2024-01-01", stop="2024-01-05")
    driver.initialize({"atmos": (x.clone(), coords), "ocean": (x.clone(), coords)})
    states = driver.rollout(driver.clock.n_steps)

    atmos = states["atmos"]
    ocean = states["ocean"]

    # compare every variable at the final lead time (96 h): the atmos
    # component exports its prognostics at the 96 h window end, which is
    # native lead_time[-1]; 96 h is also an ocean output time, so the
    # native SST slice at lead_time[-1] is valid (other atmos leads hold
    # uninitialized memory for ocean variables — never index those).
    for i, raw in enumerate(y_coords["variable"]):
        raw = str(raw)
        if raw in model.atmos_variables:
            std = driver.components["atmos"].dictionary.standard_name(raw)
            got = atmos[std].data
        else:
            std = driver.components["ocean"].dictionary.standard_name(raw)
            got = ocean[std].data
        # align on the last valid lead time for a like-for-like check
        native_last = native_final_lead_slab(y_native, y_coords, i)
        torch.testing.assert_close(
            got.reshape(native_last.shape).float(),
            native_last.float(),
            rtol=1e-4,
            atol=1e-4,
            msg=f"nvcoupler output diverges from native DLESyM for {raw!r}",
        )

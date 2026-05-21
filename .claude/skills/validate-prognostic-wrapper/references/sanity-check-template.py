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

"""Sanity-check plot for <ModelName> prognostic model.

This script is for PR review only — do NOT commit to the repo.
Runs a multi-step forecast using BOTH the original third-party model
and the Earth2Studio wrapper, visualizing outputs side-by-side.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

# =============================================================================
# PART 1: Reference model inference (third-party packages only, NO Earth2Studio)
# =============================================================================
# TODO: Load and run the original model per reference repo instructions
# Uncomment and adapt the following lines:
#
# import <original_package>
# ref_model = <OriginalModel>.from_pretrained("path/to/checkpoint")
# ref_model.eval().cuda()
#
# # Prepare input in the format expected by the original model
# torch.manual_seed(42)
# ref_input = ...  # Shape and format per original model spec
#
# # Run multi-step forecast with reference model
# N_STEPS = 5
# ref_steps = [ref_input.cpu().numpy()]  # Step 0 = initial condition
# current = ref_input
# with torch.no_grad():
#     for _ in range(N_STEPS):
#         current = ref_model(current)
#         ref_steps.append(current.cpu().numpy())

raise NotImplementedError(
    "Fill in the reference model code above, then remove this line."
)

# =============================================================================
# PART 2: Earth2Studio wrapper inference
# =============================================================================
from earth2studio.data import <DataSource>, fetch_data  # Use user's chosen data source
from earth2studio.models.px import ModelName

# Load E2S wrapper
model = ModelName.from_pretrained()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Prepare input via E2S data pipeline
time = np.array([np.datetime64("2024-01-01T00:00")])
input_coords = model.input_coords()
input_coords["time"] = time
ds = <DataSource>(input_coords)  # Use user's chosen data source
x, coords = fetch_data(ds, time, input_coords["variable"], device=device)

# Run multi-step forecast with E2S wrapper
iterator = model.create_iterator(x, coords)
e2s_steps = []
for i, (step_x, step_coords) in enumerate(iterator):
    e2s_steps.append((step_x.cpu().numpy(), dict(step_coords)))
    if i >= N_STEPS:
        break

# =============================================================================
# PART 3: Side-by-side visualization
# =============================================================================
var_list = list(e2s_steps[0][1]["variable"])
plot_vars = var_list[:3]  # First 3 variables, or pick specific ones
n_vars = len(plot_vars)

# Plot: rows = variables, columns = [Reference step0, E2S step0, Ref final, E2S final]
step_indices = [0, N_STEPS]
n_cols = len(step_indices) * 2  # 2 columns per step (ref + e2s)
fig, axes = plt.subplots(n_vars, n_cols, figsize=(4 * n_cols, 4 * n_vars))
if n_vars == 1:
    axes = axes[np.newaxis, :]

for row, var in enumerate(plot_vars):
    var_idx = var_list.index(var)
    for col_idx, si in enumerate(step_indices):
        # Reference model output
        # TODO: Extract the correct variable from ref_steps[si]
        # ref_data_2d = ref_steps[si][..., var_idx, :, :]  # Adapt indexing
        ref_data_2d = np.zeros((100, 100))  # Placeholder — replace with actual

        # E2S wrapper output
        e2s_data, sc = e2s_steps[si]
        e2s_data_2d = e2s_data[0, 0, 0, var_idx, :, :]
        lead = sc["lead_time"]

        # Plot reference
        ax_ref = axes[row, col_idx * 2]
        im_ref = ax_ref.contourf(ref_data_2d, cmap="turbo", levels=20)
        ax_ref.set_title(f"REF: {var} | step={si}")
        plt.colorbar(im_ref, ax=ax_ref, shrink=0.8)

        # Plot E2S
        ax_e2s = axes[row, col_idx * 2 + 1]
        im_e2s = ax_e2s.contourf(e2s_data_2d, cmap="turbo", levels=20)
        ax_e2s.set_title(f"E2S: {var} | lead={lead}")
        plt.colorbar(im_e2s, ax=ax_e2s, shrink=0.8)

plt.suptitle(f"<ModelName> — Reference vs Earth2Studio comparison", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_<model_name>.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_<model_name>.png")

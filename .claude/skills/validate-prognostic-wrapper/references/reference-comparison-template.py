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

"""Reference comparison for <ModelName> prognostic model.

Compares the Earth2Studio wrapper output against the original reference
implementation to verify numerical agreement for both single-step and
multi-step forecasts.

This script is for validation only — do NOT commit to the repo.
"""

import numpy as np
import torch

# --- Reference model ---
# TODO: Load original model per reference repo instructions
# Uncomment and adapt the following lines:
# ref_model = ...
# ref_input = ...
# ref_output_single = ref_model(ref_input)  # single step
# ref_outputs_multi = [ref_output_single]
# current = ref_output_single
# for step in range(N_STEPS):
#     current = ref_model(current)
#     ref_outputs_multi.append(current)
raise NotImplementedError(
    "Fill in the reference model code above, then remove this line."
)

# --- Earth2Studio wrapper ---
from earth2studio.models.px import ModelName

model = ModelName(...)  # or ModelName.load_model(package)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

input_coords = model.input_coords()
# Construct input tensor matching the reference input
# Use the same random seed or identical real data for both
shape = tuple(max(len(v), 1) for v in input_coords.values())
torch.manual_seed(42)
x = torch.randn(shape, device=device)

# --- Single-step comparison ---
e2s_output_single, out_coords = model(x, input_coords)

ref_output_single = ref_output_single.to(e2s_output_single.device)
max_abs_single = (ref_output_single - e2s_output_single).abs().max().item()
max_rel_single = (
    ((ref_output_single - e2s_output_single).abs() / (ref_output_single.abs() + 1e-8))
    .max()
    .item()
)
corr_single = torch.corrcoef(
    torch.stack(
        [
            ref_output_single.flatten(),
            e2s_output_single.flatten(),
        ]
    )
)[0, 1].item()

print("=== Single-step comparison ===")
print(f"Max absolute difference: {max_abs_single:.2e}")
print(f"Max relative difference: {max_rel_single:.2e}")
print(f"Correlation: {corr_single:.8f}")

assert torch.allclose(
    ref_output_single, e2s_output_single, rtol=1e-4, atol=1e-5
), f"Single-step mismatch! Max abs diff: {max_abs_single:.2e}"

# --- Multi-step comparison ---
N_STEPS = 5  # Adapt to model time step (e.g., 5 steps of 6h = 30h)
iterator = model.create_iterator(x, input_coords)

# Skip initial condition (step 0)
step0_x, step0_coords = next(iterator)

print(f"\n=== Multi-step comparison ({N_STEPS} steps) ===")
for step_i in range(N_STEPS):
    e2s_step, e2s_coords = next(iterator)
    ref_step = ref_outputs_multi[step_i + 1].to(e2s_step.device)

    max_abs = (ref_step - e2s_step).abs().max().item()
    corr = torch.corrcoef(torch.stack([ref_step.flatten(), e2s_step.flatten()]))[
        0, 1
    ].item()
    lead = e2s_coords["lead_time"]

    print(
        f"Step {step_i + 1} (lead_time={lead}): "
        f"max_abs={max_abs:.2e}, corr={corr:.8f}"
    )

    assert torch.allclose(
        ref_step, e2s_step, rtol=1e-3, atol=1e-4
    ), f"Multi-step mismatch at step {step_i + 1}! Max abs: {max_abs:.2e}"

print("\nPASS: Reference comparison successful (single + multi-step).")

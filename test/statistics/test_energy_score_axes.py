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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.statistics import energy_score


@pytest.mark.parametrize(
    "order,fair",
    [
        ((0, 1, 2, 3), False),
        ((2, 0, 1, 3), False),
        ((2, 0, 1, 3), True),
        ((1, 2, 3, 0), False),
        ((2, 1, 0, 3), True),
    ],
)
def test_energy_score_reordered_multivariate_axes(
    order: tuple[int, ...], fair: bool
) -> None:
    # Equal ensemble/batch sizes also expose silent reduction over the wrong axis.
    x = torch.arange(72, dtype=torch.float64).reshape(3, 3, 2, 4) / 7
    y = torch.arange(24, dtype=torch.float64).reshape(3, 2, 4) / 11 - 2
    expected = torch.empty(3, dtype=torch.float64)
    for batch in range(3):
        members = x[:, batch].flatten(1)
        truth = y[batch].flatten()
        skill = sum(torch.linalg.vector_norm(member - truth) for member in members) / 3
        spread = sum(
            torch.linalg.vector_norm(a - b) for a in members for b in members
        ) / (2 * 3 * (2 if fair else 3))
        expected[batch] = skill - spread

    names = ["ensemble", "batch", "variable", "lon"]
    x_coords = OrderedDict((names[i], np.arange(x.shape[i])) for i in order)
    y_order = [i - 1 for i in order if i != 0]
    y_coords = OrderedDict((k, v) for k, v in x_coords.items() if k != "ensemble")
    reordered_x, reordered_y = x.permute(order), y.permute(y_order)
    metric = energy_score("ensemble", ["variable", "lon"], fair=fair)
    actual, coords = metric(reordered_x, x_coords, reordered_y, y_coords)
    torch.testing.assert_close(actual, expected)
    assert list(coords) == ["batch"]
    np.testing.assert_array_equal(coords["batch"], np.arange(3))
    for name, values in metric.output_coords(x_coords).items():
        np.testing.assert_array_equal(coords[name], values)

    # Additional weighted reduction must consume the corrected per-batch scores.
    weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    reduced, reduced_coords = energy_score(
        "ensemble", ["variable", "lon"], ["batch"], weights=weights, fair=fair
    )(reordered_x, x_coords, reordered_y, y_coords)
    torch.testing.assert_close(reduced, (expected * weights).sum() / weights.sum())
    assert not reduced_coords


@pytest.mark.parametrize("members,fair", [(1, False), (3, False), (3, True)])
def test_energy_score_trailing_ensemble_scalar_and_gradients(
    members: int, fair: bool
) -> None:
    x = (
        torch.arange(2 * members, dtype=torch.float64).reshape(2, members) + 1
    ).requires_grad_()
    y = torch.tensor([-1.0, -2.0], dtype=torch.float64)
    x_coords = OrderedDict(variable=np.arange(2), ensemble=np.arange(members))
    y_coords = OrderedDict(variable=np.arange(2))
    actual, coords = energy_score("ensemble", ["variable"], fair=fair)(
        x, x_coords, y, y_coords
    )
    skill = sum(torch.linalg.vector_norm(x[:, i] - y) for i in range(members)) / members
    spread = sum(
        torch.linalg.vector_norm(x[:, i] - x[:, j])
        for i in range(members)
        for j in range(members)
    ) / (2 * members * (members - 1 if fair else members))
    expected = skill - spread
    assert actual.ndim == 0
    assert not coords
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, x, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, x)[0]
    torch.testing.assert_close(actual_grad, expected_grad)

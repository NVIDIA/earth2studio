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

import datetime
from collections import OrderedDict
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from earth2studio.data import Random, prep_data_array
from earth2studio.models.batch import batch_coords, batch_func


@pytest.fixture
def PhooModel():
    class BatchModel(torch.nn.Module):
        # Simple fake prognostic model that takes in a tensor of size [B, 2, 1] and
        # returns one of size [B, 1, 1] in the variable dimension.
        def input_coords(self) -> OrderedDict:
            return OrderedDict(
                [
                    ("batch", np.empty(1)),
                    ("variable", np.array(["a", "b"])),
                    ("x", np.ones(1)),
                ]
            )

        @batch_coords()
        def output_coords(self, input_coords: OrderedDict) -> OrderedDict:

            output_coords = input_coords.copy()
            output_coords["variable"] = np.array(["a"])
            output_coords["lead_time"] += 1
            return output_coords

        @batch_func()
        def __call__(self, x: torch.Tensor, coords: OrderedDict) -> tuple:
            return x[:, :1], self.output_coords(coords)

        @batch_func()
        def add_pairs(
            self,
            x1: torch.Tensor,
            coords1: OrderedDict,
            x2: torch.Tensor,
            coords2: OrderedDict,
        ) -> tuple[torch.Tensor, OrderedDict]:
            # Simple op combining two inputs; propagate coords1 through output_coords
            return x1 + x2, self.output_coords(coords1)

        @batch_func()
        def scale_pairs(
            self,
            x1: torch.Tensor,
            coords1: OrderedDict,
            x2: torch.Tensor,
            coords2: OrderedDict,
            alpha: float = 1.0,
        ) -> tuple[torch.Tensor, OrderedDict]:
            return alpha * (x1 + x2), self.output_coords(coords1)

        @batch_func()
        def _default_iterator(
            self, x: torch.Tensor, coords: OrderedDict[str, np.ndarray]
        ) -> Iterator[tuple[torch.Tensor, OrderedDict[str, np.ndarray]]]:
            coords = coords.copy()
            x = x[:, :1]
            while True:
                coords = self.output_coords(coords)
                yield x, coords

        def create_iterator(
            self, x: torch.Tensor, coords: OrderedDict[str, np.ndarray]
        ) -> Iterator[tuple[torch.Tensor, OrderedDict[str, np.ndarray]]]:
            yield from self._default_iterator(x, coords)

    return BatchModel


@pytest.mark.parametrize(
    "bc",
    [
        OrderedDict({"a1": np.random.randn(2)}),
        OrderedDict({"a1": np.random.randn(2), "a2": np.random.randn(2)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_call(PhooModel, bc, device):
    time = datetime.datetime(year=1958, month=1, day=31)
    variable = ["a", "b"]
    dc = {"lead_time": np.ones(1)}
    # Initialize Data Source
    r = Random(dc)

    da = r(time, variable)

    # Prepare data array
    x, coords = prep_data_array(da, device=device)
    # Expand batch dimensions
    for key, value in reversed(bc.items()):
        coords.update({key: value})
        coords.move_to_end(key, last=False)
        x = torch.stack([x for _ in value], dim=0)

    # Forward pass
    model = PhooModel().to(device)
    out, out_coords = model(x, coords)

    assert out.shape[:-2] == x.shape[:-2]
    assert coords.keys() == out_coords.keys()


@pytest.mark.parametrize(
    "bc",
    [
        OrderedDict({"a1": np.random.randn(2)}),
        OrderedDict({"a1": np.random.randn(2), "a2": np.random.randn(2)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_iter(PhooModel, bc, device):
    time = datetime.datetime(year=1958, month=1, day=31)
    variable = ["a", "b"]
    dc = {"lead_time": np.zeros(1)}
    # Initialize Data Source
    r = Random(dc)

    da = r(time, variable)

    # Prepare data array
    x, coords = prep_data_array(da, device=device)
    # Expand batch dimensions
    for key, value in reversed(bc.items()):
        coords.update({key: value})
        coords.move_to_end(key, last=False)
        x = torch.stack([x for _ in value], dim=0)

    # Forward pass
    model = PhooModel().to(device)
    model_iter = model.create_iterator(x, coords)
    # Get generator
    for i, (out, out_coords) in enumerate(model_iter):
        assert out.shape[:-2] == x.shape[:-2]
        assert coords.keys() == out_coords.keys()
        assert out_coords["lead_time"][0] == i + 1

        if i > 5:
            break


@pytest.mark.parametrize(
    "bc",
    [
        OrderedDict({"a1": np.random.randn(2)}),
        OrderedDict({"a1": np.random.randn(2), "a2": np.random.randn(2)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_batch_multiple_pairs(PhooModel, bc, device):
    time = datetime.datetime(year=1958, month=1, day=31)
    variable = ["a", "b"]
    dc = {"lead_time": np.ones(1)}
    # Initialize Data Source
    r = Random(dc)

    da = r(time, variable)

    # Prepare data array for first input
    x1, coords1 = prep_data_array(da, device=device)
    # Create a second input tensor of same shape on the same device
    x2 = torch.randn_like(x1)
    coords2 = coords1.copy()

    # Expand batch dimensions on both inputs consistently
    for key, value in reversed(bc.items()):
        for c in (coords1, coords2):
            c.update({key: value})
            c.move_to_end(key, last=False)
        x1 = torch.stack([x1 for _ in value], dim=0)
        x2 = torch.stack([x2 for _ in value], dim=0)

    # Forward pass on method with two (x, coords) pairs
    model = PhooModel().to(device)
    out, out_coords = model.add_pairs(x1, coords1, x2, coords2)

    assert out.shape[:-2] == x1.shape[:-2]
    assert coords1.keys() == out_coords.keys()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_invalid_ordering_raises(PhooModel, device):
    time = datetime.datetime(year=1958, month=1, day=31)
    variable = ["a", "b"]
    dc = {"lead_time": np.ones(1)}
    r = Random(dc)
    da = r(time, variable)
    x, coords = prep_data_array(da, device=device)
    model = PhooModel().to(device)
    with pytest.raises(ValueError):
        # coords then tensor is invalid for a pair
        _ = model.add_pairs(coords, x, coords, x)  # type: ignore[arg-type]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_invalid_extra_positional_raises(PhooModel, device):
    time = datetime.datetime(year=1958, month=1, day=31)
    variable = ["a", "b"]
    dc = {"lead_time": np.ones(1)}
    r = Random(dc)
    da = r(time, variable)
    x1, coords1 = prep_data_array(da, device=device)
    x2 = torch.randn_like(x1)
    coords2 = coords1.copy()
    model = PhooModel().to(device)
    # Extra positional (alpha) makes odd count of positional args
    with pytest.raises(ValueError):
        _ = model.scale_pairs(x1, coords1, x2, coords2, 0.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_kwargs_allowed_not_batched(PhooModel, device):
    time = datetime.datetime(year=1958, month=1, day=31)
    variable = ["a", "b"]
    dc = {"lead_time": np.ones(1)}
    r = Random(dc)
    da = r(time, variable)
    x1, coords1 = prep_data_array(da, device=device)
    x2 = torch.randn_like(x1)
    coords2 = coords1.copy()
    model = PhooModel().to(device)
    # alpha passed as kwarg should be accepted and not batched
    out, out_coords = model.scale_pairs(x1, coords1, x2, coords2, alpha=0.5)
    assert out.shape[:-2] == x1.shape[:-2]
    assert coords1.keys() == out_coords.keys()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mismatched_batched_dims_raise(PhooModel, device):
    time = datetime.datetime(year=1958, month=1, day=31)
    variable = ["a", "b"]
    dc = {"lead_time": np.ones(1)}
    r = Random(dc)
    da = r(time, variable)
    x1, coords1 = prep_data_array(da, device=device)
    x2 = torch.randn_like(x1)
    coords2 = coords1.copy()

    # Add different leading batch dims to each pair
    coords1.update({"a1": np.arange(2)})
    coords1.move_to_end("a1", last=False)
    x1 = torch.stack([x1 for _ in range(2)], dim=0)

    coords2.update({"b1": np.arange(3)})
    coords2.move_to_end("b1", last=False)
    x2 = torch.stack([x2 for _ in range(3)], dim=0)

    model = PhooModel().to(device)
    with pytest.raises(ValueError):
        _ = model.add_pairs(x1, coords1, x2, coords2)

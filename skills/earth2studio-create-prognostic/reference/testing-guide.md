# Testing Guide — Prognostic Model Wrapper

> **Table of Contents**
>
> 1. [Smoke Test](#smoke-test)
> 2. [Data Fetch Test](#data-fetch-test)
> 3. [Pytest Unit Tests](#pytest-unit-tests)
> 4. [Running Tests](#running-tests)

---

## Smoke Test

Write and run a quick smoke test script to verify basic forward pass:

```python
import torch
import numpy as np
from earth2studio.models.px import ModelName

# Load model (or construct with dummy weights for testing)
model = ModelName(...)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Get input coords
input_coords = model.input_coords()

# Create random input tensor
shape = tuple(max(len(v), 1) for v in input_coords.values())
x = torch.randn(shape)

# Test __call__
output, out_coords = model(x, input_coords)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Output lead_time: {out_coords['lead_time']}")

# Test create_iterator (3 steps)
iterator = model.create_iterator(x, input_coords)
for i, (step_x, step_coords) in enumerate(iterator):
    print(f"Step {i}: shape={step_x.shape}, lead_time={step_coords['lead_time']}")
    if i >= 3:
        break
```

---

## Data Fetch Test

Verify the model's coordinate system works with the data pipeline:

```python
import numpy as np
from earth2studio.data import Random, fetch_data
from earth2studio.models.px import ModelName

model = ModelName(...)

time = np.array([np.datetime64("2024-01-01T00:00")])
input_coords = model.input_coords()
input_coords["time"] = time
ds = Random(input_coords)
x, coords = fetch_data(ds, time, input_coords["variable"])

print(f"Fetched data shape: {x.shape}")
print(f"Variables: {input_coords['variable']}")
```

---

## Pytest Unit Tests

Create `test/models/px/test_<filename>.py` following existing test patterns.

### Complete test file template

```python
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

import gc
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px import ModelName
from earth2studio.utils import handshake_dim


class PhooModelName(torch.nn.Module):
    """Dummy model that performs a simple deterministic operation."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1  # Simple deterministic operation for testing


@pytest.fixture(scope="class")
def test_package(tmp_path_factory):
    """Create a dummy model package for testing."""
    tmp_path = tmp_path_factory.mktemp("data")
    model = PhooModelName()
    torch.save(model, tmp_path / "model.pt")
    return Package(str(tmp_path))


class TestModelNameMock:
    @pytest.mark.parametrize(
        "time",
        [
            np.array([
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="No GPU"
                ),
            ),
        ],
    )
    def test_model_call(self, test_package, time, device):
        """Test single forward pass."""
        model = ModelName.load_model(test_package)
        model = model.to(device)

        dc = model.input_coords()
        dc["time"] = time
        ds = Random(dc)
        x, coords = fetch_data(ds, time, dc["variable"], device=device)

        out, out_coords = model(x, coords)

        assert out.shape == x.shape
        assert isinstance(out_coords, OrderedDict)
        handshake_dim(out_coords, "variable", 3)

    @pytest.mark.parametrize("ensemble", [1, 2])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda:0",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="No GPU"
                ),
            ),
        ],
    )
    def test_model_iter(self, test_package, ensemble, device):
        """Test iterator produces correct sequence."""
        model = ModelName.load_model(test_package)
        model = model.to(device)

        time = np.array([np.datetime64("2024-01-01T00:00")])
        dc = model.input_coords()
        dc["time"] = time
        ds = Random(dc)
        x, coords = fetch_data(ds, time, dc["variable"], device=device)

        # Add ensemble dim
        x = x.unsqueeze(0).repeat(ensemble, *([1] * x.ndim))
        coords["ensemble"] = np.arange(ensemble)
        coords.move_to_end("ensemble", last=False)

        iterator = model.create_iterator(x, coords)
        assert isinstance(iterator, Iterable)

        # Skip initial condition
        next(iterator)

        for i, (step_x, step_coords) in enumerate(iterator):
            assert step_x.shape[0] == ensemble
            if i >= 4:
                break

        del model
        gc.collect()
        if device != "cpu":
            torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "coords",
        [
            OrderedDict({
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["wrong_var"]),
                "lat": np.linspace(90, -90, 10),
                "lon": np.linspace(0, 360, 20),
            }),
        ],
    )
    def test_model_exceptions(self, test_package, coords):
        """Test model raises on invalid coordinates."""
        model = ModelName.load_model(test_package)
        x = torch.randn(
            1, 1, 1,
            len(coords["variable"]),
            len(coords["lat"]),
            len(coords["lon"]),
        )
        with pytest.raises((KeyError, ValueError)):
            model(x, coords)


@pytest.mark.package
def test_model_package():
    """Integration test with real model weights."""
    model = ModelName.from_pretrained()
    input_coords = model.input_coords()
    time = np.array([np.datetime64("2024-01-01T00:00")])
    input_coords["time"] = time
    ds = Random(input_coords)
    x, coords = fetch_data(ds, time, input_coords["variable"])
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(model.device_buffer.device)
    out, out_coords = model(x, coords)
    assert out.shape == x.shape
```

### Key adaptation points

- Adapt `PhooModelName` to match the actual model's input/output interface
  so the wrapper's reshaping logic is exercised
- Match the checkpoint format in `test_package` fixture to what `load_model` expects
- Add model-specific assertions (e.g., lead_time increment, variable count)

---

## Running Tests

### Mock tests (no real checkpoint)

```bash
uv run python -m pytest test/models/px/test_<filename>.py \
    -m "not package" -v
```

All mock tests must pass before proceeding.

### Package integration test

```bash
uv run python -m pytest test/models/px/test_<filename>.py \
    -m "package" -v
```

**Warning:** The package test will:
- Download the model checkpoint (may be several GB)
- Require GPU compute for CUDA models
- Take significantly longer than mock tests

### Coverage report

```bash
uv run python -m pytest test/models/px/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/models/px/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

Target: **>= 90% line coverage** on the new model module.

Common coverage gaps:
- Error handling in `output_coords` (wrong variable names, wrong dims)
- Device management paths (CPU vs CUDA)
- `create_iterator` edge cases (initial condition yield, hook calls)
- `load_model` and `load_default_package` (needs mock or package test)
- ONNX / non-PyTorch backend `.to()` logic

### Full model test suite (optional)

```bash
make pytest TOX_ENV=test-models
```

---
name: create-prognostic-tests
description: Write unit tests for a newly created Earth2Studio prognostic model (px) wrapper including smoke tests, data fetch tests, and pytest unit tests
argument-hint: Name of the prognostic model wrapper class (e.g., Aurora, StormCast)
---

# Test Prognostic Model Wrapper

> **Note:** Creating a complete prognostic model wrapper involves three major phases,
> each with its own skill:
>
> 1. **`create-prognostic-wrapper`** — Steps 0-8: Implement the wrapper
>    class, dependencies, coordinate system, forward pass, model loading, and
>    registration
> 2. **`create-prognostic-tests`** (this skill) — Step 9: Write smoke tests, data fetch tests,
>    pytest unit tests (mock + package), and comparison scripts
> 3. **`validate-prognostic-wrapper`** — Final validation: Run tests with coverage,
>    reference comparison, sanity-check plots, and open a PR with automated review
>
> Complete all steps in this skill first, then invoke the next skill in sequence.

Write unit tests for a newly created Earth2Studio prognostic model wrapper by following
every step below in order.
Each confirmation gate marked by starting with:

```markdown
### **[CONFIRM — <Title>]**
```

requires explicit user approval before proceeding.

**Prerequisites:** This skill assumes the prognostic wrapper has already been implemented
(Steps 0-8 of the `create-prognostic-wrapper` skill are complete).

---

## Step 1 — Test Forward Pass with Random Data

Write and run a quick smoke test script:

```python
import torch
import numpy as np
from earth2studio.models.px import ModelName

# Load model (or construct with dummy weights for testing)
model = ModelName(...)  # Use dummy/test weights if real ones aren't available
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

Report results to the user.

---

## Step 2 — Test Data Fetch with Random Source

Test that the model's coordinate system works with Earth2Studio's data pipeline:

```python
import numpy as np
from earth2studio.data import Random, fetch_data
from earth2studio.models.px import ModelName

model = ModelName(...)

# Create time array
time = np.array([np.datetime64("2024-01-01T00:00")])

# Fetch data using model's input coords
input_coords = model.input_coords()
input_coords["time"] = time
ds = Random(input_coords)
x, coords = fetch_data(ds, time, input_coords["variable"])

print(f"Fetched data shape: {x.shape}")
print(f"Variables: {input_coords['variable']}")
```

Report results to the user. This validates the coordinate system is compatible with the data pipeline.

---

## Step 3 — Write Pytest Unit Tests

Create `test/models/px/test_<filename>.py` following the existing test patterns.

### Test file structure

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
    # Export dummy model to checkpoint format
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

        # Fetch input data
        dc = model.input_coords()
        dc["time"] = time
        ds = Random(dc)
        x, coords = fetch_data(ds, time, dc["variable"], device=device)

        # Run forward
        out, out_coords = model(x, coords)

        # Validate output
        assert out.shape == x.shape
        assert isinstance(out_coords, OrderedDict)
        handshake_dim(out_coords, "variable", 3)
        # Additional model-specific assertions

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

        # Create input
        time = np.array([np.datetime64("2024-01-01T00:00")])
        dc = model.input_coords()
        dc["time"] = time
        ds = Random(dc)
        x, coords = fetch_data(ds, time, dc["variable"], device=device)

        # Add ensemble dim
        x = x.unsqueeze(0).repeat(ensemble, *([1] * x.ndim))
        coords["ensemble"] = np.arange(ensemble)
        coords.move_to_end("ensemble", last=False)

        # Test iterator
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

Adapt the dummy model (`PhooModelName`) to match the
actual model's input/output interface so the wrapper's
reshaping logic is exercised.

---

## Step 4 — Run Tests

### 4a. Run mock tests (no package flag)

First, run the unit tests that use mocked / dummy models.
These do **not** require downloading real checkpoints and
should run quickly on any machine:

```bash
uv run python -m pytest test/models/px/test_<filename>.py \
    -m "not package" -v
```

All mock tests must pass before proceeding. Fix any
failures and re-run until green.

### 4b. Run the package integration test

Once the mock tests pass, run the `@pytest.mark.package`
test which exercises `from_pretrained()` with real model
weights:

```bash
uv run python -m pytest test/models/px/test_<filename>.py \
    -m "package" -v
```

### **[CONFIRM — Package Test]**

Before executing the package test, warn the user:

> The package / integration test will:
>
> - **Download the model checkpoint** (may be several GB)
>   to the local cache
> - **Require GPU compute** for models that need CUDA
>   (the test will skip on CPU-only machines if
>   `torch.cuda.is_available()` is `False`)
> - Take significantly longer than the mock tests
>
> Do you want to proceed with the package test?

Only run the package test after the user confirms. Report
the results back to the user. If the package test fails,
debug and fix the wrapper or test, then re-run.

---

## Next Step — Validate the Wrapper

Once all tests pass, use the **`validate-prognostic-wrapper`** skill to:

1. Run tests with coverage (>=90% required)
2. Create reference comparison scripts (third-party vs E2S)
3. Generate sanity-check plots
4. Open a PR with automated code review

Invoke the skill:

```text
/skill validate-prognostic-wrapper <ModelName>
```

---

## Reminders

- **DO** use `loguru.logger` for logging, never `print()`, inside `earth2studio/`
- **DO** ensure all public functions have full type hints
- **DO** run formatting (`make format`) and linting (`make lint`) before finalizing
- **DO** adapt the dummy model (`PhooModelName`) to match the actual model's interface

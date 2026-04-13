---
name: validate-assimilation-wrapper
description: >-
  Validate a newly created Earth2Studio data assimilation model wrapper by
  writing unit tests (90% coverage required), performing reference comparison
  with DataFrame/DataArray outputs, generating sanity-check plots, and
  opening a PR with automated code review. Use after completing
  create-assimilation-wrapper Steps 0-8.
argument-hint: Name of the DA model class and test file (optional — will be inferred from recent changes if not provided)
---

# Validate Assimilation Model Wrapper

Validate a newly created Earth2Studio data assimilation (DA) model
wrapper by writing unit tests, performing reference comparison,
generating sanity-check outputs, and opening a PR with automated code
review. This skill picks up after the `create-assimilation-wrapper`
skill completes implementation (Steps 0-8).

> **Python Environment:** This project uses **uv** for dependency
> management. Always use the local `.venv` virtual environment
> (`source .venv/bin/activate` or prefix with `uv run python`) for all
> Python commands — installing packages, running tests, executing
> scripts, etc. Use `uv add` / `uv pip install` / `uv lock` instead of
> `pip install`.

Each confirmation gate marked by:

```markdown
### **[CONFIRM — <Title>]**
```

requires **explicit user approval** before proceeding.

---

## Step 1 — Write Pytest Unit Tests

Create a test file at `test/models/da/test_<filename>.py`. DA tests
are fundamentally different from px/dx tests — inputs are DataFrames
(not tensors), outputs are `xr.DataArray` (not tensor + CoordSystem
tuples), and the generator uses the send protocol (not an iterator).

### 1a. PhooModelName dummy class

Create a lightweight dummy model that mimics the DA model under test.
The dummy accepts `pd.DataFrame` input and returns `xr.DataArray`
output. This is used for unit tests that do not require the real
model checkpoint.

```python
class PhooModelName(torch.nn.Module):
    """Dummy DA model for testing."""

    VARIABLES = ["t2m", "u10m"]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("device_buffer", torch.empty(0))
        self._lat = np.linspace(25.0, 50.0, 11, dtype=np.float32)
        self._lon = np.linspace(235.0, 295.0, 13, dtype=np.float32)

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    def __call__(self, obs: pd.DataFrame) -> xr.DataArray:
        request_time = obs.attrs["request_time"]
        data = torch.randn(
            len(request_time),
            len(self.VARIABLES),
            len(self._lat),
            len(self._lon),
        )
        # Return as xr.DataArray (numpy for CPU)
        return xr.DataArray(
            data=data.numpy(),
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": request_time,
                "variable": np.array(self.VARIABLES, dtype=str),
                "lat": self._lat,
                "lon": self._lon,
            },
        )

    def create_generator(self):
        observations = yield None  # Prime
        try:
            while True:
                result = self.__call__(observations)
                observations = yield result
        except GeneratorExit:
            pass

    def to(self, device):
        return super().to(device)
```

### 1b. Test fixtures

Define these fixtures at the top of the test file:

```python
import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

try:
    import cudf
except ImportError:
    cudf = None

try:
    import cupy as cp
except ImportError:
    cp = None

from earth2studio.models.da.<module> import ModelName


@pytest.fixture
def sample_observations_pandas():
    """Create sample pandas DataFrame observations for testing."""
    time1 = np.datetime64("2024-01-01T12:00:00")
    time2 = np.datetime64("2024-01-01T13:00:00")
    return pd.DataFrame(
        {
            "time": [
                time1, time1, time1, time1,
                time2, time2, time2, time2,
            ],
            "lat": [30.0, 30.0, 40.0, 40.0, 30.0, 30.0, 40.0, 40.0],
            "lon": [240.0, 250.0, 240.0, 250.0, 240.0, 250.0, 240.0, 250.0],
            "observation": [10.0, 20.0, 25.0, 35.0, 30.0, 40.0, 35.0, 45.0],
            "variable": [
                "t2m", "t2m", "u10m", "u10m",
                "t2m", "t2m", "u10m", "u10m",
            ],
        }
    )


@pytest.fixture
def sample_observations_cudf():
    """Create sample cudf DataFrame observations for testing."""
    if cudf is None:
        pytest.skip("cudf not available")
    time1 = np.datetime64("2024-01-01T12:00:00")
    time2 = np.datetime64("2024-01-01T13:00:00")
    return cudf.DataFrame(
        {
            "time": [
                time1, time1, time1, time1,
                time2, time2, time2, time2,
            ],
            "lat": [30.0, 30.0, 40.0, 40.0, 30.0, 30.0, 40.0, 40.0],
            "lon": [240.0, 250.0, 240.0, 250.0, 240.0, 250.0, 240.0, 250.0],
            "observation": [10.0, 20.0, 30.0, 40.0, 30.0, 40.0, 50.0, 60.0],
            "variable": [
                "t2m", "t2m", "u10m", "u10m",
                "t2m", "t2m", "u10m", "u10m",
            ],
        }
    )


@pytest.fixture
def test_package(tmp_path):
    """Create a dummy checkpoint directory for integration tests."""
    # Adapt contents to match the model's load_model expectations
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    # Example: create a dummy weights file
    torch.save({"state_dict": {}}, checkpoint_dir / "model.pt")
    return checkpoint_dir
```

### 1c. Test methods

Write the following test methods. Each test must be **complete and
runnable** — no placeholder stubs.

> **Note:** Every test file must start with the SPDX Apache-2.0
> license header (see the create skill for the exact template).

#### test_model_init

```python
def test_model_init():
    """Test constructor parameter validation."""
    model = ModelName(...)  # Fill in required constructor args

    # Verify model attributes are set correctly
    assert hasattr(model, "device_buffer")
    assert hasattr(model, "device")

    # Test invalid constructor args raise errors
    with pytest.raises((ValueError, TypeError)):
        ModelName(invalid_param=...)
```

#### test_model_call

```python
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_model_call(sample_observations_pandas, device):
    """Test stateless __call__ with pandas DataFrame input."""
    model = ModelName(...).to(device)

    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    sample_observations_pandas.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    da = model(sample_observations_pandas)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "variable", "lat", "lon")
    assert da.coords["time"].values[0] == request_time[0]

    # Validate output shape matches model's coordinate system
    n_variables = len(model.VARIABLES) if hasattr(model, "VARIABLES") else da.shape[1]
    assert da.shape[0] == len(request_time)
    assert da.shape[1] == n_variables

    # Validate coordinate values
    assert "t2m" in da.coords["variable"].values  # At least one expected variable

    # Check device-specific return type
    if device == "cuda:0" and torch.cuda.is_available():
        if cp is not None:
            assert isinstance(da.data, cp.ndarray)
            assert not cp.all(cp.isnan(da.data))
    else:
        assert isinstance(da.data, np.ndarray)
        assert not np.all(np.isnan(da.values))
```

#### test_model_call_cudf

```python
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_model_call_cudf(sample_observations_cudf, device):
    """Test __call__ with cudf DataFrame input on GPU."""
    if cudf is None:
        pytest.skip("cudf not available")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model = ModelName(...).to(device)

    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    sample_observations_cudf.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    da = model(sample_observations_cudf)

    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "variable", "lat", "lon")
    assert da.coords["time"].values[0] == request_time[0]

    if cp is not None:
        assert isinstance(da.data, cp.ndarray)
        assert not cp.all(cp.isnan(da.data))
    else:
        assert isinstance(da.data, np.ndarray)
        assert not np.all(np.isnan(da.values))
```

#### test_generator_protocol

```python
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_generator_protocol(sample_observations_pandas, device):
    """Test create_generator prime -> send -> close sequence."""
    model = ModelName(...).to(device)

    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    sample_observations_pandas.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    generator = model.create_generator()

    # Prime the generator
    result = generator.send(None)
    assert result is None

    # Send observations — first step
    da = generator.send(sample_observations_pandas)
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("time", "variable", "lat", "lon")
    assert da.shape[0] == len(request_time)

    # Send observations — second step
    da2 = generator.send(sample_observations_pandas)
    assert isinstance(da2, xr.DataArray)
    assert da2.shape == da.shape

    # Close generator
    generator.close()
```

#### test_init_coords

```python
def test_init_coords():
    """Test init_coords returns correct type for the model."""
    model = ModelName(...)

    result = model.init_coords()

    # Choose ONE of the following patterns based on the model:

    # For stateless models (no initialization data required):
    assert result is None

    # For stateful models (requires initialization data):
    # assert isinstance(result, tuple)
    # assert len(result) > 0
    # for schema in result:
    #     assert isinstance(schema, (dict, OrderedDict))
```

#### test_input_coords

```python
def test_input_coords():
    """Test input_coords returns tuple of FrameSchema."""
    model = ModelName(...)

    result = model.input_coords()

    assert isinstance(result, tuple)
    assert len(result) >= 1

    # Each element should be a FrameSchema (OrderedDict)
    for schema in result:
        assert isinstance(schema, dict)
        # DA observation schemas typically have these columns
        assert "variable" in schema
```

#### test_output_coords

```python
def test_output_coords():
    """Test output_coords returns valid tuple of CoordSystem."""
    model = ModelName(...)

    input_coords = model.input_coords()
    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])

    result = model.output_coords(input_coords, request_time=request_time)

    assert isinstance(result, tuple)
    assert len(result) >= 1

    # Each output should be a CoordSystem with expected dimensions
    for coords in result:
        assert isinstance(coords, dict)
        assert "time" in coords
        assert "variable" in coords
```

#### test_time_tolerance

```python
def test_time_tolerance():
    """Test filter_time_range behavior with time tolerance."""
    model = ModelName(...)

    base_time = np.datetime64("2024-01-01T12:00:00")
    time_within = base_time - np.timedelta64(30, "m")
    time_outside = base_time + np.timedelta64(24, "h")

    obs_df = pd.DataFrame(
        {
            "time": [time_within, time_outside],
            "lat": [30.0, 40.0],
            "lon": [240.0, 250.0],
            "observation": [10.0, 20.0],
            "variable": ["t2m", "t2m"],
        }
    )

    request_time = np.array([base_time])
    obs_df.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    da = model(obs_df)
    assert isinstance(da, xr.DataArray)
    # Verify output is valid — exact assertions depend on model behavior
    assert da.shape[0] == len(request_time)
```

#### test_empty_dataframe

```python
def test_empty_dataframe():
    """Test graceful handling of empty DataFrame."""
    model = ModelName(...)

    empty_df = pd.DataFrame(
        {
            "time": pd.Series([], dtype="datetime64[ns]"),
            "lat": pd.Series([], dtype=np.float32),
            "lon": pd.Series([], dtype=np.float32),
            "observation": pd.Series([], dtype=np.float32),
            "variable": pd.Series([], dtype=str),
        }
    )

    request_time = np.array([np.datetime64("2024-01-01T12:00:00")])
    empty_df.attrs = {
        "request_time": request_time,
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    # Model should either return a DataArray (possibly with NaN) or raise cleanly
    try:
        da = model(empty_df)
        assert isinstance(da, xr.DataArray)
    except (ValueError, RuntimeError):
        pass  # Acceptable to raise on empty input
```

#### test_invalid_attrs

```python
def test_invalid_attrs():
    """Test that missing request_time in attrs raises an error."""
    model = ModelName(...)

    obs_df = pd.DataFrame(
        {
            "time": [np.datetime64("2024-01-01T12:00:00")],
            "lat": [30.0],
            "lon": [240.0],
            "observation": [10.0],
            "variable": ["t2m"],
        }
    )
    # Intentionally do NOT set obs_df.attrs with request_time

    with pytest.raises((ValueError, KeyError, TypeError)):
        model(obs_df)
```

#### test_validate_observation_fields

```python
def test_validate_observation_fields():
    """Test that invalid DataFrame columns raise an error."""
    model = ModelName(...)

    bad_df = pd.DataFrame(
        {
            "wrong_column": [1.0],
            "another_bad": [2.0],
        }
    )
    bad_df.attrs = {
        "request_time": np.array([np.datetime64("2024-01-01T12:00:00")]),
    }

    with pytest.raises((ValueError, KeyError)):
        model(bad_df)
```

#### test_model_exceptions

```python
def test_model_exceptions():
    """Test model raises on invalid inputs."""
    model = ModelName(...)

    # Test with None input (if model requires non-None)
    with pytest.raises((ValueError, TypeError)):
        model(None)
```

#### Integration test (@pytest.mark.package)

```python
@pytest.mark.package
def test_model_package(test_package):
    """Integration test using model checkpoint.

    This test requires the actual model package and is skipped by
    default. Run with --slow to include.
    """
    from earth2studio.models.auto import Package

    package = Package(str(test_package))
    model = ModelName.load_model(package)

    obs_df = pd.DataFrame(
        {
            "time": [np.datetime64("2024-01-01T12:00:00")],
            "lat": [30.0],
            "lon": [240.0],
            "observation": [10.0],
            "variable": ["t2m"],
        }
    )
    obs_df.attrs = {
        "request_time": np.array([np.datetime64("2024-01-01T12:00:00")]),
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }

    da = model(obs_df)
    assert isinstance(da, xr.DataArray)
```

### **[CONFIRM — Package Test]**

Before writing the `@pytest.mark.package` integration test, ask the
user to confirm the model loading path and checkpoint structure:

> The integration test needs to load the actual model checkpoint.
>
> 1. What is the checkpoint path or package URL?
> 2. Does `load_model` require additional arguments beyond `package`?
> 3. What test inputs should the integration test use?
>
> I'll write the `@pytest.mark.package` test based on your answers.

**Do not proceed to Step 2 until the user confirms.**

---

## Step 2 — Run Tests & Achieve 90% Coverage

### 2a. Run the new test file

```bash
uv run python -m pytest test/models/da/test_<filename>.py -v --timeout=60
```

All tests must pass. Fix failures and re-run until green.

### 2b. Run coverage report with `--slow` tests

Run the new test file **with coverage** and the `--slow` flag to
include integration tests. The new DA model file must achieve **at
least 90% line coverage**:

```bash
uv run python -m pytest test/models/da/test_<filename>.py -v \
    --slow --timeout=300 \
    --cov=earth2studio/models/da/<filename> \
    --cov-report=term-missing \
    --cov-fail-under=90
```

- `--slow` enables integration tests marked with `@pytest.mark.package`
  (the `--slow` flag is configured in `conftest.py` to include package
  tests that download real checkpoints and may require GPU)
- `--cov=earth2studio/models/da/<filename>` scopes coverage to the
  new model module only
- `--cov-report=term-missing` shows which lines are not covered
- `--cov-fail-under=90` fails the run if coverage is below 90%

If coverage is below 90%, add additional tests or mock tests to cover
the missing lines. Common DA-specific coverage gaps:

- `GeneratorExit` cleanup path in `create_generator`
- cudf code paths (skipped when cudf is unavailable)
- Empty DataFrame handling branches
- Time tolerance edge cases (observations right at boundary)
- `obs.attrs` validation branches (missing `request_time`)
- cupy vs numpy output paths (GPU vs CPU return types)
- `filter_time_range` with no matching observations
- `dfseries_to_torch` conversion branches

Re-run until coverage is at or above 90%.

### 2c. Run the full model test suite (optional but recommended)

```bash
make pytest TOX_ENV=test-models
```

Confirm no regressions in existing model tests.

---

## Step 3 — Reference Comparison & Sanity-Check

This step validates the DA model wrapper produces correct output by
comparing against the original reference implementation and generating
visual sanity-check plots.

### 3a. Create reference comparison script

Create a **standalone Python script** in the repo root. This is for
validation only and should **NOT** be committed to the repo.

The script loads the reference model and the E2S wrapper side by side,
runs both on identical input, and compares outputs with tolerance.

**For DataArray output** (gridded analysis fields):

```python
"""Reference comparison for <ModelName> assimilation model.

Compares the Earth2Studio wrapper output against the original reference
implementation to verify numerical agreement.

This script is for validation only — do NOT commit to the repo.
"""
import numpy as np
import pandas as pd
import torch

# --- Reference model ---
# TODO: Load original model per reference repo instructions
# Uncomment and adapt the following lines:
# ref_model = ...
# ref_obs = pd.DataFrame({...})
# ref_obs.attrs = {"request_time": ..., "request_lead_time": ...}
# ref_output = ref_model(ref_obs)
raise NotImplementedError(
    "Fill in the reference model code above, then remove this line."
)

# --- Earth2Studio wrapper ---
from earth2studio.models.da import ModelName

model = ModelName(...)  # or ModelName.load_model(package)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Construct identical observation DataFrame
obs_df = pd.DataFrame(
    {
        "time": [...],
        "lat": [...],
        "lon": [...],
        "observation": [...],
        "variable": [...],
    }
)
obs_df.attrs = {
    "request_time": np.array([...], dtype="datetime64[ns]"),
    "request_lead_time": np.array([np.timedelta64(0, "h")]),
}

e2s_output = model(obs_df)

# --- Compare DataArray outputs ---
ref_values = ref_output.values  # numpy array
e2s_values = e2s_output.values  # numpy or cupy array
if hasattr(e2s_values, "get"):
    e2s_values = e2s_values.get()  # cupy -> numpy

max_abs_diff = np.abs(ref_values - e2s_values).max()
max_rel_diff = (
    np.abs(ref_values - e2s_values) / (np.abs(ref_values) + 1e-8)
).max()
correlation = np.corrcoef(ref_values.flatten(), e2s_values.flatten())[0, 1]

print(f"Max absolute difference: {max_abs_diff:.2e}")
print(f"Max relative difference: {max_rel_diff:.2e}")
print(f"Correlation: {correlation:.8f}")

assert np.allclose(ref_values, e2s_values, rtol=1e-4, atol=1e-5), \
    f"Output mismatch! Max abs diff: {max_abs_diff:.2e}"

# --- Compare generator outputs ---
ref_gen = ref_model.create_generator()
ref_gen.send(None)
e2s_gen = model.create_generator()
e2s_gen.send(None)

for step_obs in [obs_df, obs_df]:
    ref_step = ref_gen.send(step_obs)
    e2s_step = e2s_gen.send(step_obs)
    ref_step_vals = ref_step.values
    e2s_step_vals = e2s_step.values
    if hasattr(e2s_step_vals, "get"):
        e2s_step_vals = e2s_step_vals.get()
    step_diff = np.abs(ref_step_vals - e2s_step_vals).max()
    print(f"Generator step max abs diff: {step_diff:.2e}")
    assert np.allclose(ref_step_vals, e2s_step_vals, rtol=1e-4, atol=1e-5)

ref_gen.close()
e2s_gen.close()

print("PASS: Reference comparison successful.")
```

**For DataFrame output** (if the model returns tabular data):

```python
# --- Compare DataFrame outputs ---
assert len(ref_output) == len(e2s_output), \
    f"Row count mismatch: ref={len(ref_output)}, e2s={len(e2s_output)}"

for col in ["lat", "lon", "observation", "variable"]:
    if col in ref_output.columns:
        ref_vals = ref_output[col].values
        e2s_vals = e2s_output[col].values
        if np.issubdtype(ref_vals.dtype, np.floating):
            max_diff = np.abs(ref_vals - e2s_vals).max()
            print(f"Column '{col}' max diff: {max_diff:.2e}")
        else:
            assert np.array_equal(ref_vals, e2s_vals), \
                f"Column '{col}' values differ"

# Spatial coverage check
ref_lat_range = (ref_output["lat"].min(), ref_output["lat"].max())
e2s_lat_range = (e2s_output["lat"].min(), e2s_output["lat"].max())
print(f"Lat range: ref={ref_lat_range}, e2s={e2s_lat_range}")

ref_lon_range = (ref_output["lon"].min(), ref_output["lon"].max())
e2s_lon_range = (e2s_output["lon"].min(), e2s_output["lon"].max())
print(f"Lon range: ref={ref_lon_range}, e2s={e2s_lon_range}")
```

### 3b. Summarize model capabilities to user

Before generating sanity-check plots, **present a summary table** to
the user covering the model's capabilities:

> **Model Summary for `<ClassName>`:**
>
> | Property | Value |
> | ---------------------- | ------------------------------------ |
> | **Model type** | Stateless / Stateful |
> | **Input format** | DataFrame / DataArray / Mixed |
> | **Input schema** | time, lat, lon, observation, variable |
> | **Output format** | DataArray / DataFrame |
> | **Output grid** | lat-lon N x M / HRRR / HealPix |
> | **Output variables** | `var1`, `var2`, ... |
> | **Time tolerance** | (default value) |
> | **cudf/cupy support** | Yes / No |
> | **Observation types** | Surface / Satellite / Radar / Mixed |
> | **Checkpoint source** | NGC / HuggingFace / N/A |

This summary helps the user verify the wrapper matches their
expectations for the model.

### 3c. Generate sanity-check plot scripts

Create **standalone Python scripts** in the repo root. These are for
PR reviewer reference only and should **NOT** be committed to the
repo.

#### Plot 1: Spatial assimilated output

Contourf of gridded DataArray output from `__call__`:

```python
"""Sanity-check plot 1: Spatial assimilated output for <ModelName>.

This script is for PR review only — do NOT commit to the repo.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from earth2studio.models.da import ModelName

# Load model
model = ModelName(...)  # or ModelName.load_model(package)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create observation DataFrame
obs_df = pd.DataFrame(
    {
        "time": [np.datetime64("2024-01-01T12:00:00")] * 4,
        "lat": [30.0, 35.0, 40.0, 45.0],
        "lon": [240.0, 250.0, 260.0, 270.0],
        "observation": [10.0, 15.0, 20.0, 25.0],
        "variable": ["t2m", "t2m", "t2m", "t2m"],
    }
)
obs_df.attrs = {
    "request_time": np.array([np.datetime64("2024-01-01T12:00:00")]),
    "request_lead_time": np.array([np.timedelta64(0, "h")]),
}

# Run forward pass
da = model(obs_df)

# Get coordinate arrays
lat = da.coords["lat"].values
lon = da.coords["lon"].values
variables = da.coords["variable"].values

# Plot contourf for each output variable
n_vars = len(variables)
fig, axes = plt.subplots(1, n_vars, figsize=(6 * n_vars, 5))
if n_vars == 1:
    axes = [axes]

for ax, var in zip(axes, variables):
    data_2d = da.sel(variable=var).isel(time=0).values
    if hasattr(data_2d, "get"):
        data_2d = data_2d.get()  # cupy -> numpy
    im = ax.contourf(lon, lat, data_2d, cmap="turbo", levels=20)
    ax.set_title(f"{var}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle(f"<ModelName> assimilated output", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_da_spatial.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_da_spatial.png")
```

#### Plot 2: Observation overlay (unique to DA)

Scatter of input DataFrame observations overlaid on assimilated grid
output. This visualization is **specific to DA models** — it shows the
sparse-to-dense mapping from observations to analysis field.

```python
"""Sanity-check plot 2: Observation overlay for <ModelName>.

This script is for PR review only — do NOT commit to the repo.
Shows input sparse observations overlaid on the assimilated gridded output.
"""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from earth2studio.models.da import ModelName

# Load model
model = ModelName(...)  # or ModelName.load_model(package)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create observation DataFrame
obs_df = pd.DataFrame(
    {
        "time": [np.datetime64("2024-01-01T12:00:00")] * 8,
        "lat": [30.0, 32.0, 35.0, 38.0, 40.0, 42.0, 45.0, 48.0],
        "lon": [240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 275.0],
        "observation": [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0, 28.0],
        "variable": ["t2m"] * 8,
    }
)
obs_df.attrs = {
    "request_time": np.array([np.datetime64("2024-01-01T12:00:00")]),
    "request_lead_time": np.array([np.timedelta64(0, "h")]),
}

# Run forward pass
da = model(obs_df)

# Get coordinate arrays
lat = da.coords["lat"].values
lon = da.coords["lon"].values
variables = [v for v in da.coords["variable"].values if v in obs_df["variable"].unique()]

n_vars = len(variables)
fig, axes = plt.subplots(
    1, n_vars, figsize=(8 * n_vars, 6),
    subplot_kw={"projection": ccrs.PlateCarree()},
)
if n_vars == 1:
    axes = [axes]

for ax, var in zip(axes, variables):
    # Plot assimilated grid as contourf
    data_2d = da.sel(variable=var).isel(time=0).values
    if hasattr(data_2d, "get"):
        data_2d = data_2d.get()  # cupy -> numpy
    ax.contourf(
        lon, lat, data_2d,
        cmap="turbo", alpha=0.5, levels=20,
        transform=ccrs.PlateCarree(),
    )

    # Overlay input observations as scatter
    obs_var = obs_df[obs_df["variable"] == var]
    scatter = ax.scatter(
        obs_var["lon"].values, obs_var["lat"].values,
        c=obs_var["observation"].values,
        cmap="turbo", edgecolors="k", s=40, zorder=5,
        transform=ccrs.PlateCarree(),
    )
    plt.colorbar(scatter, ax=ax, shrink=0.7, label="Observation value")

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.set_title(f"{var}: grid + observations")

plt.suptitle("<ModelName> — observation overlay", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_da_overlay.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_da_overlay.png")
```

#### Plot 3: Generator sequence

Multi-step assimilation evolution using the generator protocol:

```python
"""Sanity-check plot 3: Generator sequence for <ModelName>.

This script is for PR review only — do NOT commit to the repo.
Shows evolution of assimilated output across multiple generator steps.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from earth2studio.models.da import ModelName

# Load model
model = ModelName(...)  # or ModelName.load_model(package)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create observation sequence — one DataFrame per step
base_time = np.datetime64("2024-01-01T12:00:00")
n_steps = 4
observation_sequence = []

for i in range(n_steps):
    step_time = base_time + np.timedelta64(i, "h")
    obs_step = pd.DataFrame(
        {
            "time": [step_time] * 4,
            "lat": [30.0, 35.0, 40.0, 45.0],
            "lon": [240.0, 250.0, 260.0, 270.0],
            "observation": [10.0 + i * 5, 15.0 + i * 5, 20.0 + i * 5, 25.0 + i * 5],
            "variable": ["t2m"] * 4,
        }
    )
    obs_step.attrs = {
        "request_time": np.array([step_time]),
        "request_lead_time": np.array([np.timedelta64(0, "h")]),
    }
    observation_sequence.append(obs_step)

# Run generator
gen = model.create_generator()
gen.send(None)  # Prime
results = []
for obs_step in observation_sequence:
    result = gen.send(obs_step)
    results.append(result)
gen.close()

# Plot sequence
var = results[0].coords["variable"].values[0]  # Plot first variable
lat = results[0].coords["lat"].values
lon = results[0].coords["lon"].values

fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
if len(results) == 1:
    axes = [axes]

for ax, (i, result) in zip(axes, enumerate(results)):
    data_2d = result.sel(variable=var).isel(time=0).values
    if hasattr(data_2d, "get"):
        data_2d = data_2d.get()  # cupy -> numpy
    im = ax.contourf(lon, lat, data_2d, cmap="turbo", levels=20)
    ax.set_title(f"Step {i}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.colorbar(im, ax=ax, shrink=0.8)

plt.suptitle(f"<ModelName> — generator sequence ({var})", y=1.02)
plt.tight_layout()
plt.savefig("sanity_check_da_generator.png", dpi=150, bbox_inches="tight")
print("Saved: sanity_check_da_generator.png")
```

### 3d. Create side-by-side comparison script

Create a script that runs both the reference implementation and the
E2S wrapper with identical inputs and produces a side-by-side plot:

```python
"""Side-by-side comparison: reference vs Earth2Studio for <ModelName>.

This script is for validation only — do NOT commit to the repo.
Fill in the TODO sections below before running.
"""
import matplotlib.pyplot as plt
import numpy as np

# TODO: Import and initialize both reference and E2S models
# TODO: Prepare identical input observations (DataFrame)
raise NotImplementedError(
    "Fill in the reference and E2S model code above before running."
)

# ref_output = ...  # Run reference model
# e2s_output = ...  # Run E2S wrapper

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: Reference output
# axes[0].contourf(lon, lat, ref_data, cmap="turbo")
# axes[0].set_title("Reference")

# Panel 2: E2S output
# axes[1].contourf(lon, lat, e2s_data, cmap="turbo")
# axes[1].set_title("Earth2Studio")

# Panel 3: Difference
# axes[2].contourf(lon, lat, ref_data - e2s_data, cmap="RdBu_r")
# axes[2].set_title("Difference (Ref - E2S)")

plt.suptitle("<ModelName> — reference vs Earth2Studio")
plt.tight_layout()
plt.savefig("comparison_<model_name>.png", dpi=150, bbox_inches="tight")
print("Saved: comparison_<model_name>.png")
```

### 3e. Run comparison and sanity-check scripts

Execute the scripts:

```bash
uv run python reference_comparison_<model_name>.py
uv run python sanity_check_da_spatial.py
uv run python sanity_check_da_overlay.py
uv run python sanity_check_da_generator.py
```

Verify that:

- The reference comparison passes (all assertions hold)
- All sanity-check scripts run without errors
- Output PNGs are generated
- Metrics are printed (max abs diff, max rel diff, correlation)

### 3f. **[CONFIRM — Sanity-Check & Comparison]**

**You MUST ask the user to visually inspect the generated plot(s)
before proceeding.** Do not skip this step even if the scripts ran
without errors — a successful run does not guarantee the plots are
correct (e.g., empty axes, wrong colorbar range, garbled data).

Tell the user the absolute path to the generated image file(s) and
the reference comparison metrics, then ask them to inspect:

> The reference comparison and sanity-check scripts ran successfully.
>
> **Reference comparison metrics:**
>
> - Max absolute difference: `<value>`
> - Max relative difference: `<value>`
> - Correlation: `<value>`
>
> **Sanity-check plots saved to:**
>
> 1. `/absolute/path/to/sanity_check_da_spatial.png` — gridded output
> 2. `/absolute/path/to/sanity_check_da_overlay.png` — observations on grid
> 3. `/absolute/path/to/sanity_check_da_generator.png` — generator sequence
>
> **Please open these images and confirm they look correct.** Check:
>
> 1. Data is visible on the axes (not blank/empty)
> 2. Values are in physically reasonable ranges
> 3. No obvious artifacts (all-NaN regions, garbled values)
> 4. Spatial patterns look plausible (geographic features visible)
> 5. Observation overlay: scatter points visible on top of grid
> 6. Generator sequence: evolution across steps is coherent
>
> Do the plots look correct and do the reference comparison metrics
> look acceptable?

**Do not proceed to Step 4 until the user explicitly confirms.** If
the user reports problems, debug and fix the issue, re-run the
scripts, and ask the user to inspect again.

---

## Step 4 — Branch, Commit & Open PR

### **[CONFIRM — Ready to Submit]**

Before proceeding, confirm with the user:

> All implementation and validation steps are complete:
>
> - DA model class implemented with correct method ordering
> - `init_coords` returns correct type (None for stateless or tuple for stateful)
> - `input_coords` returns tuple of FrameSchema
> - `output_coords` returns tuple with `request_time`
> - `create_generator` uses send protocol with GeneratorExit handling
> - `validate_observation_fields` used for DataFrame inputs
> - cupy/cudf optional import pattern present
> - Registered in `earth2studio/models/da/__init__.py`
> - Documentation added to `docs/modules/models.rst`
> - Reference URLs included in class docstrings
> - CHANGELOG.md updated
> - Format, lint, and license checks pass
> - Unit tests written and passing with >= 90% coverage
> - Dependencies in pyproject.toml confirmed
> - Reference comparison passes with acceptable tolerance
> - Sanity-check plots generated and confirmed by user
>
> Ready to create a branch, commit, and prepare a PR?

### 4a. Create branch and commit

```bash
git checkout -b feat/da-model-<name>
git add earth2studio/models/da/<filename>.py \
        earth2studio/models/da/__init__.py \
        test/models/da/test_<filename>.py \
        pyproject.toml \
        CHANGELOG.md \
        docs/modules/models.rst
git commit -m "feat: add <ClassName> assimilation model

Add <ClassName> data assimilation model for <brief description>.
Includes unit tests and documentation."
```

Do **NOT** add the sanity-check scripts, comparison scripts, or
their output images.

### 4b. Identify the fork remote and push branch

The working repository is typically a **fork** of
`NVIDIA/earth2studio`. Before pushing, confirm which git remote
points to the user's fork:

```bash
git remote -v
```

Ask the user:

> Which git remote is your fork of `NVIDIA/earth2studio`?
> (Usually `origin` — e.g., `git@github.com:<user>/earth2studio.git`)

Then push the feature branch to the **fork** remote:

```bash
git push -u <fork-remote> feat/da-model-<name>
```

### 4c. Open Pull Request (fork -> NVIDIA/earth2studio)

> **Important:** PRs must be opened **from the fork** to the
> **upstream source repository** `NVIDIA/earth2studio`. The branch
> lives on the fork; the PR targets `main` on the upstream repo.

Use `gh pr create` with explicit `--repo` and `--head` flags:

```bash
gh pr create \
  --repo NVIDIA/earth2studio \
  --base main \
  --head <fork-owner>:feat/da-model-<name> \
  --title "feat: add <ClassName> assimilation model" \
  --body "..."
```

Where `<fork-owner>` is the GitHub username that owns the fork.

The PR body should follow this DA-model-specific template:

````markdown
## Description

Add `<ClassName>` data assimilation model for <brief description>.

Closes #<issue_number> (if applicable)

### Model details

| Property | Value |
|---|---|
| **Model type** | Stateless / Stateful |
| **Input format** | DataFrame / DataArray / Mixed |
| **Output format** | DataArray / DataFrame |
| **Observation schema** | time, lat, lon, observation, variable |
| **Grid specification** | lat-lon / HRRR / HealPix / etc. |
| **Time tolerance** | <default value> |
| **cudf/cupy support** | Yes / No |
| **Reference** | <link to paper/repo> |

### Dependencies added

| Package | Version | License | License URL | Reason |
|---|---|---|---|---|
| `<package>` | `>=X.Y` | <License> | [link](<URL>) | <reason> |

*(or "No new dependencies")*

### Reference comparison

- Max absolute difference: <value>
- Max relative difference: <value>
- Correlation: <value>

### Validation

See sanity-check plots in PR comments below.

## Checklist

- [x] I am familiar with the [Contributing Guidelines][contrib].
- [x] New or existing tests cover these changes.
- [x] The documentation is up to date with these changes.
- [x] The [CHANGELOG.md][changelog] is up to date with these changes.
- [ ] An [issue][issues] is linked to this pull request.
- [ ] Assess and address Greptile feedback (AI code review bot).

[contrib]: https://github.com/NVIDIA/earth2studio/blob/main/CONTRIBUTING.md
[changelog]: https://github.com/NVIDIA/earth2studio/blob/main/CHANGELOG.md
[issues]: https://github.com/NVIDIA/earth2studio/issues
````

### 4d. Post sanity-check as PR comment

After the PR is created, post the sanity-check visualization as a
separate **PR comment** so it is immediately visible to reviewers.

#### Image upload limitation

**GitHub has no CLI or REST API for uploading images to PR comments.**
The only way to embed an image is via the browser's drag-and-drop
editor or by referencing an already-hosted URL.

**Practical workflow:**

1. Write the comment body to a temp file (avoids shell quoting issues
   with heredocs containing backticks and markdown).
2. Post the comment **without** the image — include the validation
   table, reference comparison metrics, the full sanity-check script,
   and a placeholder line.
3. Tell the user to drag the image into the browser editor.

```bash
# 1. Write body to a temp file (use your editor tool, not heredoc)

# 2. Post the comment
gh api -X POST repos/NVIDIA/earth2studio/issues/<PR_NUMBER>/comments \
  -F "body=@/tmp/pr_comment_body.md" \
  --jq '.html_url'
```

Do **not** waste time trying `curl` uploads, GraphQL file mutations,
or the `uploads.github.com` asset endpoint — they do not work for
issue/PR comment images.

#### Comment content template

```markdown
## Sanity-Check Validation

**Model:** `<ClassName>` — <brief description>
**Type:** Stateless / Stateful
**Test environment:** <GPU model or CPU>

### Reference Comparison

| Metric | Value |
|--------|-------|
| Max absolute difference | <value> |
| Max relative difference | <value> |
| Correlation | <value> |

### Model Summary

| Property | Value |
|----------|-------|
| Model type | Stateless / Stateful |
| Input format | DataFrame / DataArray / Mixed |
| Output format | DataArray / DataFrame |
| Observation schema | time, lat, lon, observation, variable |
| Output grid | lat-lon N x M / HRRR / HealPix |
| Output variables | <list or count> |
| Time tolerance | <default value> |
| cudf/cupy support | Yes / No |
| Inference time | ~XX ms |

**Key findings:**
- <bullet summarizing numerical agreement with reference>
- <bullet on output quality / physical reasonableness>
- <bullet on performance or notable behavior>

> **TODO:** Attach sanity-check images by editing this comment in
> the browser.

<details>
<summary>Sanity-check scripts (click to expand)</summary>

```python
PASTE THE FULL WORKING SCRIPTS HERE — not truncated excerpts.
The scripts must be copy-pasteable and produce the plots end-to-end.
```

</details>
```

**Important:** Always paste the **complete, runnable** scripts — not
shortened versions. Reviewers should be able to reproduce the plots
by copying the scripts directly.

#### Finalize

After posting, inform the user of:

1. The comment URL
2. The local paths to the image files for manual attachment
3. Instructions: *"Edit the comment in your browser and drag the
   image files into the editor to embed them."*

> **Note:** The sanity-check images and scripts are for PR review
> purposes only — they must NOT be committed to the repository.

---

## Step 5 — Automated Code Review (Greptile)

After the PR is created and pushed, an automated code review from
**greptile-apps** (Greptile) will be posted as PR review comments.
Wait for this review, then process the feedback.

### 5a. Wait for Greptile review

Poll for review comments from `greptile-apps[bot]` every 30 seconds
for up to **5 minutes**. Time out gracefully if no review arrives:

```bash
# Poll loop — check every 30s, timeout after 5 minutes (10 attempts)
for i in $(seq 1 10); do
  REVIEW_ID=$(gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
    --jq '.[] | select(.user.login == "greptile-apps[bot]") | .id' 2>/dev/null)
  if [ -n "$REVIEW_ID" ]; then
    echo "Greptile review found: $REVIEW_ID"
    break
  fi
  echo "Attempt $i/10 — no review yet, waiting 30s..."
  sleep 30
done
```

If no review after 5 minutes, inform the user:

> Greptile hasn't posted a review after 5 minutes. This can happen if
> the review bot is busy or the PR hasn't triggered it. You can:
>
> 1. Ask me to check again later
> 2. Skip this step and proceed without automated review
> 3. Manually request a review from Greptile on the PR page

### 5b. Pull and parse review comments

Once the review is posted, fetch all comments:

```bash
# Get all review comments on the PR
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") |
    {path: .path, line: .diff_hunk, body: .body}'
```

Also fetch the top-level review body:

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/reviews \
  --jq '.[] | select(.user.login == "greptile-apps[bot]") | .body'
```

### 5c. Categorize and present to user

Parse each comment and categorize it:

| Category | Description | Default action |
| --------------------- | --------------------------------- | -------------- |
| **Bug / correctness** | Logic errors, wrong behavior | Fix |
| **Style / convention** | Naming, formatting, patterns | Fix if valid |
| **Performance** | Inefficiency, resource waste | Evaluate |
| **Documentation** | Missing/wrong docs, docstrings | Fix |
| **Suggestion** | Alternative approach, nice-to-have | User decides |
| **False positive** | Incorrect or irrelevant feedback | Dismiss |

### **[CONFIRM — Review Triage]**

Present each comment to the user in a summary table:

```markdown
| # | File | Line | Category | Summary | Proposed Action |
|---|------|------|----------|---------|-----------------|
| 1 | <model>.py | 142 | Bug | Missing null check | Fix: add guard |
| 2 | <model>.py | 305 | Style | Use f-string | Fix: convert |
| 3 | <model>.py | 45 | Suggestion | Add type alias | Skip: not needed |
| ... | ... | ... | ... | ... | ... |
```

For each comment, briefly explain:

- What Greptile flagged
- Whether you agree or disagree (with reasoning)
- Your proposed fix (or why to skip)

Ask the user to confirm which comments to address. The user may:

- Accept all proposed fixes
- Select specific fixes
- Override your recommendation on any comment
- Add their own fixes

### 5d. Implement fixes

For each accepted fix:

1. Make the code change
2. Run `make format && make lint` after all fixes
3. Run the relevant tests:

   ```bash
   uv run python -m pytest test/models/da/test_<filename>.py -v --timeout=60
   ```

4. Commit with a message like:

   ```bash
   git commit -m "fix: address code review feedback (Greptile)"
   ```

### 5e. Respond to review comments

For each Greptile comment, post a reply on the PR:

**For fixed comments:**

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Fixed in <commit_sha>. <brief description of fix>"
```

**For dismissed comments (false positives / won't fix):**

```bash
gh api repos/NVIDIA/earth2studio/pulls/<PR_NUMBER>/comments/<COMMENT_ID>/replies \
  -f body="Won't fix — <brief justification>"
```

### 5f. Push and resolve

```bash
git push <fork-remote> feat/da-model-<name>
```

After pushing, resolve all addressed review threads if possible.

Inform the user of the final state:

- How many comments were fixed
- How many were dismissed (with reasons)
- Any remaining open threads

---

## Reminders

- **DO** use the repo's local `uv` `.venv` to run Python with
  `uv run python`
- **DO NOT** commit sanity-check/comparison scripts or images to
  the repo
- **DO** use `loguru.logger` for logging, never `print()`, inside
  `earth2studio/`
- **DO** ensure all public functions have full type hints (mypy-clean)
- **DO** maintain alphabetical order in `__init__.py` exports,
  RST file entries, and CHANGELOG entries
- **DO** return tuples from `input_coords` and `output_coords`
- **DO** use `FrameSchema` for tabular inputs, `CoordSystem` for
  gridded outputs
- **DO** validate `request_time` from `obs.attrs`
- **DO** use `validate_observation_fields`, `filter_time_range`,
  `dfseries_to_torch` from `earth2studio.models.da.utils`
- **DO** prime generator with `yield None` and handle `GeneratorExit`
- **DO** return cupy arrays on GPU, numpy on CPU
- **DO** register `device_buffer` and expose `device` property
- **DO** follow the canonical DA method ordering:
  `__init__`, `device` property, `init_coords`, `input_coords`,
  `output_coords`, `load_default_package`, `load_model`, `to`,
  private methods, `__call__`, `create_generator`
- **DO** include reference URLs in class docstrings
- **DO** always update CHANGELOG.md under the current unreleased
  version
- **DO** add the model to `docs/modules/models.rst` in the
  `earth2studio.models.da` section (alphabetical order)
- **DO NOT** use `@batch_func` or `@batch_coords` — these are
  px/dx conventions only and do not apply to DA models
- **DO NOT** use `PrognosticMixin` — DA models do not time-step
- **DO NOT** use `create_iterator` — DA uses `create_generator`
  with the send protocol
- **DO NOT** assume tensor inputs — DA inputs are DataFrames and/or
  DataArrays
- **DO NOT** make a general base class with intent to reuse the
  wrapper across models
- **DO NOT** over-populate the `load_model()` API — only expose
  essential parameters
- **NEVER** commit, hardcode, or include API keys, secrets, tokens,
  or credentials in source code, sample scripts, commit messages,
  PR descriptions, or any file tracked by git

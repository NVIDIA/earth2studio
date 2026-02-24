# Earth2Studio Python Client SDK

A Python client library for the Earth2Studio REST API.

The client SDK provides two interfaces to remotely access the Earth2Studio API server:

1. **`api_client.e2client.RemoteEarth2Workflow`** - High-level interface for seamless
   integration with Earth2Studio and Xarray. Recommended for most use cases.
2. **`api_client.client.Earth2StudioClient`** - Low-level interface for direct API access
   and result file management. Useful for custom workflows and integrations.

Please refer to the server's `README_workflows.md` and `README_earth2workflows.md` for
information about the server-side interface.

## Quick Start

To use a remote workflow, first initialize the `RemoteEarth2Workflow` object and then
call it using the workflow parameters. Here, we access the `DeterministicEarth2Workflow`
included in the server examples and get the results as an Earth2Studio data source.

```python
from datetime import datetime, timedelta
import os
from api_client.e2client import RemoteEarth2Workflow

# Create client (configurable via environment variable)
api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
workflow = RemoteEarth2Workflow(
    api_url,
    workflow_name="deterministic_earth2_workflow",
    device='cuda'  # use 'cpu' if you don't have a GPU on the client machine
)

data_source = workflow(start_time=[datetime(2025, 8, 21, 6)]).as_data_source()
# get temperature forecast 24 h after start time
t2m = data_source(datetime(2025, 8, 22, 6), "t2m")
t2m_np = t2m.values  # get the results as a NumPy array
```

Data access is lazy if the workflow produces Zarr files: In the above example, only the
temperature will be downloaded and the actual download happens on the last line of code.

## Advanced Usage

In the examples below, we initialize `workflow` as in the Quick Start above,
unless otherwise mentioned.

### Access the result dataset directly

You can directly access the underlying Xarray Dataset:

```python
ds = workflow(start_time=[datetime(2025, 8, 21, 6)]).as_dataset()
# same result as getting data from `data_source` in the Quick Start example
t2m = ds["t2m"].sel(lead_time=np.timedelta64(timedelta(hours=24)))
t2m_np = t2m.values  # NumPy array
```

### Iterate over results

Similar to Earth2Studio model iterators, we can iterate over the results as
`(Tensor, CoordSystem)` tuples:

```python
model = workflow(start_time=[datetime(2025, 8, 21, 6)]).as_model()
for (x, coords) in model.create_iterator():
    # computes the mean on GPU if we used `device='cuda'` when creating the Workflow
    print(x.mean().cpu().numpy())
```

This can be used to provide an input to a local workflow that expects a prognostic
model. The example `diagnostic_analysis.py` shows how to use this to run a local
diagnostic.

### Save reference to remote results

You can create a `RemoteEarth2WorkflowResult` manually in order to access the results later.
This allows you to do other work and even close Python while waiting for results to complete.

```python
result = workflow(start_time=[datetime(2025, 8, 21, 6)])
print(result.execution_id)  # write down the execution id
exec_1766159252_5f779460

# *** After restarting Python ***

from datetime import datetime
import os
from api_client.e2client import RemoteEarth2Workflow, RemoteEarth2WorkflowResult

# re-initialize workflow
api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
workflow = RemoteEarth2Workflow(
    api_url,
    workflow_name="deterministic_earth2_workflow",
    device='cuda'  # use 'cpu' if you don't have a GPU on the client machine
)

# manually create result object with the execution_id from earlier
execution_id = "exec_1766159252_5f779460"
result = RemoteEarth2WorkflowResult(workflow, execution_id)

# now we can access the results as before
data_source = result.as_data_source()  # will block until results are available
t2m = data_source(datetime(2025, 8, 22, 6), "t2m")
```

### Use remote workflow to condition a local model

Use the results as a low-resolution conditioning for a StormCast model running locally:

```python
from earth2studio.data import HRRR
from earth2studio.models.px import StormCast
from earth2studio.io import NetCDF4Backend

fcn3_workflow = RemoteEarth2Workflow(
    api_url,
    workflow_name="stormcast_fcn3_workflow",
    device='cuda'  # use 'cpu' if you don't have a GPU on the client machine
)

hrrr_ic = HRRR()
stormcast = StormCast.from_pretrained()
io = NetCDF4Backend("stormcast_result.nc")
conditioning_source = fcn3_workflow(
    start_time=[datetime(2025, 8, 21, 6)],
    num_hours=num_steps,
    run_stormcast=False
).as_data_source()
stormcast.conditioning_data_source = conditioning_source
num_hours=5

run.deterministic(
    [datetime(2025, 8, 21, 6)],
    num_hours,
    stormcast,
    hrrr_ic,
    io,
    device='cuda',
)
```

### Object Storage (S3) Support

The client SDK seamlessly supports both local server storage and object storage
(S3/CloudFront). When the server is configured to use object storage, all the examples
in this README work without any code changes. The client automatically detects the
storage type from the server response and handles downloads appropriately:

- **Server storage**: Files are downloaded directly from the inference server
- **S3 storage**: Files are downloaded from S3 using pre-signed CloudFront URLs

This means you can use the same code regardless of how the server is configured:

```python
# Same whether server uses local or S3 storage
result = workflow(start_time=[datetime(2025, 8, 21, 6)])
ds = result.as_dataset()  # Fetches from S3 if configured
```

### Download result files directly

`Earth2StudioClient` gives lower-level access than `RemoteEarth2Workflow`. Use it to
download individual result files programmatically.

Initialize the client:

```python
import os
from datetime import datetime
from api_client.client import Earth2StudioClient, InferenceRequest

# Create client (configurable via environment variable)
api_url = os.getenv("EARTH2STUDIO_API_URL", "http://localhost:8000")
client = Earth2StudioClient(api_url, workflow_name="deterministic_earth2_workflow")

# Check API health
health = client.health_check()
print(f"API Status: {health.status}")
```

Now, submit a workflow and download result files directly:

```python
parameters = {"start_time": [datetime(2025, 8, 21, 6)]}
request = InferenceRequest(parameters=parameters)
request_result = client.run_inference_sync(request)

# Download individual output files (works with both local and S3 storage)
for file in request_result.output_files[:5]:  # limit to first 5 files
    print(f"Downloading: {file.path}")
    content = client.download_result(request_result, file.path)
    print(f"Downloaded {file.path}: {len(content.getvalue())} bytes")

    # Optionally save to disk
    # with open(file.path.split("/")[-1], "wb") as f:
    #     f.write(content.getvalue())
```

To get the URL of a result file for use with external tools like `wget` (server storage only):

```python
from urllib.parse import urljoin

result_root = client.result_root_path(request_result)
result_path = request_result.result_paths()[0]

result_url = urljoin(api_url, f"{result_root}{result_path}")
print(result_url)
# e.g. http://localhost:8000/v1/infer/.../results/.../results.zarr
```

### Asynchronous usage of `Earth2StudioClient`

For long-running forecasts, you can submit requests and check status separately:

```python
import time
from api_client.models import RequestStatus

# Submit request
response = client.submit_inference_request(request)
execution_id = response.execution_id
print(f"Submitted request: {execution_id}")

# Check status periodically
while True:
    status = client.get_request_status(execution_id)
    print(f"Status: {status.status.value}, Step: {status.progress.current_step}")

    if status.status in [RequestStatus.COMPLETED, RequestStatus.FAILED]:
        break

    time.sleep(10)

# Get results when completed
if status.status == RequestStatus.COMPLETED:
    request_result = client.get_request_results(execution_id)
    print(f"Forecast completed with {len(request_result.output_files)} output files")

# Now we can use ``request_result`` as in the above example
```

### Authentication

If your server requires authentication, you can pass a Bearer token to the client:

```python
# With authentication token
client = Earth2StudioClient(
    base_url="http://localhost:8000",
    workflow_name="deterministic_earth2_workflow",
    token="your-api-token"  # Bearer token for authenticated requests
)

workflow = RemoteEarth2Workflow(
    base_url="http://localhost:8000",
    workflow_name="deterministic_earth2_workflow",
    token="your-api-token"
)
```

The token is sent as a Bearer token in the `Authorization` header with all API requests.

### Custom HTTP Settings

You can modify the settings for HTTP access. The HTTP-related parameters are
identical for `Earth2StudioClient` and `RemoteEarth2Workflow`.

```python
client = Earth2StudioClient(
    base_url="http://localhost:8000",
    timeout=60.0,           # Request timeout
    max_retries=5,          # Maximum retries
    retry_backoff_factor=1.0, # Retry backoff factor
    token="your-api-token"  # Optional authentication token
)

workflow = RemoteEarth2Workflow(
    base_url="http://localhost:8000",
    timeout=60.0,           # Request timeout
    max_retries=5,          # Maximum retries
    retry_backoff_factor=1.0, # Retry backoff factor
    token="your-api-token"  # Optional authentication token
)
```

## API Reference

### RemoteEarth2Workflow

High-level client for Earth2Studio-compatible remote inference workflows.

#### RemoteEarth2Workflow constructor

```python
RemoteEarth2Workflow(
    base_url: str,
    workflow_name: str,
    device: str | torch.device | None = None,
    xr_args: dict[str, Any] | None = None,
    **client_kwargs
)
```

**Args:**

- `base_url`: URL of the Earth2Studio API server
- `workflow_name`: Name of the workflow to execute on the server
- `device`: Device for tensor operations (e.g., "cuda", "cpu").
- `xr_args`: Additional keyword arguments passed to xarray.open_dataset/xarray.open_zarr
- `**client_kwargs`: Additional arguments passed to Earth2StudioClient

#### RemoteEarth2Workflow methods

- `__call__(**kwargs)` → `RemoteEarth2WorkflowResult`: Submit inference with parameters
- `to(device)` → `RemoteEarth2Workflow`: Move workflow to specified device

### RemoteEarth2WorkflowResult

Result object for accessing remote inference outputs.

#### Attributes

- `execution_id`: Unique identifier for the inference execution
- `workflow`: Parent RemoteEarth2Workflow instance

#### RemoteEarth2WorkflowResult methods

- `as_dataset()` → `xr.Dataset`: Wait for completion and return result as xarray Dataset
- `as_data_source()` → `InferenceOutputSource`: Wait for completion and return as Earth2Studio data source
- `as_model(iter_coord="lead_time")` → `InferenceOutputModel`: Model iterable over time steps

### Earth2StudioClient

Low-level client class for direct API access and result file management.

#### Earth2StudioClient constructor

```python
Earth2StudioClient(
    base_url: str = "http://localhost:8000",
    workflow_name: str = "deterministic_earth2_workflow",
    timeout: float = 30.0,
    max_retries: int = 3,
    retry_backoff_factor: float = 0.3,
    token: str | None = None
)
```

**Args:**

- `base_url`: URL of the Earth2Studio API server
- `workflow_name`: Name of the workflow to execute on the server
- `timeout`: Request timeout in seconds
- `max_retries`: Maximum number of retries for failed requests
- `retry_backoff_factor`: Backoff factor between retries
- `token`: Optional Bearer token for authenticated requests

#### Earth2StudioClient methods

- `health_check()` → `HealthStatus`: Check API health
- `submit_inference_request(request)` → `InferenceRequestResponse`: Submit request
- `get_request_status(request_id)` → `InferenceRequestStatus`: Get status
- `get_request_results(request_id, timeout=None)` → `InferenceRequestResults`: Get results
- `wait_for_completion(request_id, poll_interval=5.0, timeout=None)` →
  `InferenceRequestResults`: Wait for completion
- `run_inference_sync(request, poll_interval=5.0, timeout=None)` → `InferenceRequestResults`:
  Submit and wait for completion
- `result_root_path(result)` → `str`: Get root URL path for result files
- `download_result(result, path, timeout=None)` → `io.BytesIO`: Download a result file

### Data Models

- `InferenceRequest`: Inference request payload with workflow parameters
- `InferenceRequestResponse`: Response when submitting a request (contains execution_id)
- `InferenceRequestStatus`: Status information for an inference request
- `InferenceRequestResults`: Results of a completed request with output file paths
- `OutputFile`: Information about an output file (name, path, size)
- `HealthStatus`: API health status response
- `ProgressInfo`: Progress information (progress string, current_step, total_steps)

### Enums

- `RequestStatus`: Request status values
  - Initial: `ACCEPTED`, `QUEUED`
  - Processing: `RUNNING`
  - Final: `COMPLETED`, `FAILED`, `CANCELLED`, `PENDING_RESULTS`

## Examples

The `examples/` directory contains complete, runnable example scripts:

### `basic_forecast.py` - Basic Deterministic Forecast

A comprehensive example showing how to run a simple deterministic forecast using
`RemoteEarth2Workflow`.

### `downscaled_forecast.py` - Downscaled Forecast

Example demonstrating high-resolution downscaled forecasting using a remote global
forecast model with a local downscaling model.

### `diagnostic_analysis.py` - Diagnostic Analysis

Specialized example for running local diagnostic models with remote inference results.

### Running the Examples

Each example can be run independently:

```bash
cd examples/

# Run basic deterministic forecast
python basic_forecast.py

# Run downscaled forecast
python downscaled_forecast.py

# Run diagnostic analysis
python diagnostic_analysis.py
```

**Prerequisites:**

- Earth2Studio API server running
- Client SDK installed with dependencies
- Sufficient disk space for forecast outputs

**Configuration:**

All examples support configuring the API endpoint via environment variable:

```bash
# Set custom API endpoint (optional)
export EARTH2STUDIO_API_URL="http://your-api-server:8000"

# Or use default localhost:8000 if not set
python basic_forecast.py
```

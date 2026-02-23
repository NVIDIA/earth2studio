# Earth2Studio REST API

A REST API interface for running Earth2Studio custom workflows. This API enables programmatic access
to weather forecasting, downscaling, and other atmospheric modeling tasks through custom workflow
definitions.

> **⚠️ IMPORTANT API CHANGE:**
> The legacy inference request API (`POST /v1/infer` with `workflow_type`) has been removed.
> Please use the **Custom Workflow API** instead. See [README_workflows.md](README_workflows.md) for
details.

## Supported Workflows

The API now exclusively supports custom workflow definitions:

### Custom Workflow API

* Define workflows as Python classes that inherit from `E2Workflow`
* Automatic API endpoint generation (`POST /v1/infer/{workflow_name}`)
* Built-in parameter validation using Pydantic models
* Progress tracking and result management
* Full flexibility to implement any Earth2Studio computation

For detailed documentation on creating and using custom workflows, see:

* [README_workflows.md](README_workflows.md) - Complete workflow development guide
* [README_earth2workflows.md](README_earth2workflows.md) - Earth2-specific workflow examples

---

**Note:** The sections below may contain references to the legacy API and are being updated.
For current API usage, please refer to the workflow documentation linked above.

---

## Quick Start

### Using Process Scripts (Recommended)

1. **Clone and navigate to the API directory:**

   ```bash
   cd service/inferenceserver
   ```

2. **Make scripts executable and start services:**

   ```bash

   # Start Redis
   make start-redis

   # Start api-server
   make start-api-server
   ```

3. **Access the API:**

   * API: <http://localhost:8000>
   * Interactive docs: <http://localhost:8000/docs>
   * ReDoc: <http://localhost:8000/redoc>

4. **Check service status:**

   ```bash
   make status
   ```

5. **Test API:**

  ```bash
  make test
  ```

  The test script will validate:

* API health checks
* Workflow submission and monitoring
* **Zip file downloads** with content validation
* Error handling scenarios

  **Environment variables for testing:**

  ```bash
  # Save downloaded zip files locally for inspection
  SAVE_RESULTS_ZIP=true make test

  # Use custom API endpoint
  API_BASE_URL=http://my-server:8000 make test
  ```

1. **Stop all services:**

   ```bash
   make stop-api-server
   make stop-redis
   ```

#### Redis Configuration

The API includes a pre-configured `redis.conf` file that enables:

* **RDB persistence**: Automatic snapshots every 900s, 300s, and 60s
* **AOF persistence**: Append-only file for durability
* **Memory management**: 256MB limit with LRU eviction
* **Local binding**: Only accessible from localhost

#### Verify Redis Connection

```bash
# Test Redis connection
redis-cli ping
# Should return: PONG

# Check Redis info
redis-cli info persistence
```

## API Usage Examples

### 1. Run a Deterministic Forecast

```bash
curl -X POST "http://localhost:8000/v1/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "deterministic",
    "time": "2024-01-01T00:00:00Z",
    "nsteps": 20,
    "prognostic": {
      "model_type": "fcn",
      "model_config": {}
    },
    "data": {
      "source_type": "gfs",
      "source_config": {}
    },
    "io": {
      "backend_type": "zarr",
      "file_name": "forecast.zarr"
    },
    "device": "auto"
  }'
```

**Response:**

```json
{
  "workflow_id": "wf_1704067200_abc12345",
  "status": "accepted",
  "message": "Deterministic workflow accepted and queued for execution",
  "timestamp": "2024-01-01T00:00:00Z",
  "estimated_completion": "2024-01-01T00:20:00Z"
}
```

### 2. Check Inference Request Status

```bash
curl "http://localhost:8000/v1/infer/req_1704067200_abc12345/status"
```

**Response:**

```json
{
  "request_id": "req_1704067200_abc12345",
  "status": "running",
  "progress": {
    "current_step": 15,
    "total_steps": 20,
    "percentage": 75.0
  },
  "message": "Processing forecast step 15 of 20",
  "timestamp": "2024-01-01T00:15:00Z",
  "estimated_completion": "2024-01-01T00:20:00Z"
}
```

### 3. Download Inference Request Results

The results endpoint now streams a zip file containing all output files and metadata instead of
returning JSON with local file paths.

```bash
# Download results as a zip file
curl "http://localhost:8000/v1/infer/req_1704067200_abc12345/results" \
  -o inference_results_req_1704067200_abc12345.zip
```

**Response:**

* **Content-Type**: `application/zip`
* **Content-Disposition**: `attachment; filename="inference_results_req_1704067200_abc12345.zip"`
* **Body**: Binary zip file containing all result files

**Zip File Contents:**

The downloaded zip file contains:

1. **`metadata.json`** - Request metadata and execution details
2. **All output files** - Complete directory structure from the inference

**Example metadata.json structure:**

```json
{
  "request_id": "req_1704067200_abc12345",
  "status": "completed",
  "completion_time": "2024-01-01T00:20:00Z",
  "execution_time_seconds": 1200.5,
  "workflow_type": "deterministic",
  "created_at": "2024-01-01T00:00:00Z",
  "peak_memory_usage": "2.5GB",
  "device": "cuda:0",
  "zip_created_at": "2024-01-01T00:20:05Z",
  "parameters": {
    "workflow_type": "deterministic",
    "time": "2024-01-01T00:00:00Z",
    "nsteps": 20,
    "prognostic": {
      "model_type": "fcn",
      "model_config": {}
    },
    "data": {
      "source_type": "gfs",
      "source_config": {}
    },
    "io": {
      "backend_type": "zarr",
      "file_name": "forecast.zarr"
    }
  },
  "output_files": [
    {
      "path": "forecast.zarr/.zarray",
      "size": 256
    },
    {
      "path": "forecast.zarr/.zattrs",
      "size": 512
    },
    {
      "path": "forecast.zarr/chunks/0.0.0",
      "size": 1048576
    }
  ]
}
```

### Alternative: Using wget or Python

```bash
# Using wget
wget "http://localhost:8000/v1/infer/req_1704067200_abc12345/results" \
  -O results.zip

# Using Python requests
python -c "
import requests
response = requests.get('http://localhost:8000/v1/infer/req_1704067200_abc12345/results')
with open('results.zip', 'wb') as f:
    f.write(response.content)
print(f'Downloaded {len(response.content)} bytes')
"
```

**Error Responses:**
The results endpoint returns different responses based on the request status:

* **200 OK**: Zip file download (request completed successfully)
* **202 Accepted**: Request still running or creating results

  ```json
  {
    "message": "Inference request req_12345 is still running",
    "status": "running",
    "timestamp": "2024-01-01T00:15:00Z"
  }
  ```

  Or during zip file creation:

  ```json
  {
    "message": "Inference request req_12345 is creating results zip file",
    "status": "pending_results",
    "timestamp": "2024-01-01T00:15:00Z"
  }
  ```

* **400 Bad Request**: Results have expired or request is invalid

  ```json
  {
    "error": "Results expired",
    "details": "Results for inference request req_12345 have expired and are no longer available",
    "timestamp": "2024-01-01T00:15:00Z"
  }
  ```

* **404 Not Found**: Request not found or failed

  ```json
  {
    "error": "Inference request not found",
    "details": "No inference request found with ID req_12345",
    "timestamp": "2024-01-01T00:15:00Z"
  }
  ```

### 4. Run an Ensemble Forecast

```bash
curl -X POST "http://localhost:8000/v1/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "ensemble",
    "time": "2024-01-01T00:00:00Z",
    "nsteps": 20,
    "nensemble": 10,
    "prognostic": {
      "model_type": "fcn",
      "model_config": {}
    },
    "data": {
      "source_type": "gfs",
      "source_config": {}
    },
    "io": {
      "backend_type": "zarr",
      "file_name": "ensemble_forecast.zarr"
    },
    "perturbation": {
      "method": "spherical_gaussian",
      "config": {
        "noise_amplitude": 0.15
      }
    },
    "device": "cuda",
    "batch_size": 5
  }'
```

### 5. Run a Diagnostic Workflow

```bash
curl -X POST "http://localhost:8000/v1/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_type": "diagnostic",
    "time": "2024-01-01T00:00:00Z",
    "nsteps": 20,
    "prognostic": {
      "model_type": "fcn",
      "model_config": {}
    },
    "diagnostic": {
      "model_type": "precipitation_afno",
      "model_config": {}
    },
    "data": {
      "source_type": "gfs",
      "source_config": {}
    },
    "io": {
      "backend_type": "zarr",
      "file_name": "diagnostic_forecast.zarr"
    },
    "device": "auto"
  }'
```

## Status Workflow

The inference request follows a well-defined status progression:

### Status States

1. **`queued`**: Request has been submitted and is waiting in the processing queue
2. **`running`**: The inference computation is actively running
3. **`pending_results`**: Inference computation completed successfully, creating results zip file
4. **`completed`**: Both inference and zip file creation completed successfully - results are ready
for download
5. **`failed`**: Request failed during execution or zip file creation
6. **`cancelled`**: Request was cancelled by user (if cancellation is implemented)
7. **`expired`**: Results have passed their TTL and been automatically deleted

### Status Flow

```text
queued → running → pending_results → completed → [expired after TTL]
   ↓        ↓           ↓
 failed ← failed ← ── failed
```

### Key Changes from Previous Versions

* **Pre-created Zip Files**: Results are now packaged into zip files immediately after
  inference completion, rather than being created on-demand during download
* **pending_results State**: This new intermediate state indicates that the inference
  computation is complete but the system is still creating the downloadable zip file
* **Faster Downloads**: Since zip files are pre-created, downloads start immediately
  without processing delays
* **Improved Reliability**: No more timeout issues during large zip file creation

### Monitoring Status Changes

Use the status endpoint to track progress:

```bash
# Check current status
curl "http://localhost:8000/v1/infer/req_12345/status"
```

**Response during pending_results:**

```json
{
  "request_id": "req_1757972814_199f477d",
  "status": "pending_results",
  "progress": {
    "current_step": 20,
    "total_steps": 20,
    "percentage": 100.0
  },
  "message": "Inference completed, creating results zip file",
  "timestamp": "2024-01-01T00:15:00Z"
}
```

## Results Download Format

Starting with this version, the `/v1/infer/{request_id}/results` endpoint streams results as zip
files instead of returning JSON with server file paths. This approach provides several benefits:

### Benefits

* **Complete Results**: Download all output files and metadata in a single request
* **No Path Dependencies**: No need to access server filesystem paths
* **Portable**: Results can be easily shared, archived, or transferred
* **Self-Contained**: Each zip includes metadata about the inference request
* **Bandwidth Efficient**: Compressed format reduces download time

### Zip File Structure

```text
inference_results_req_12345.zip
├── metadata.json              # Request metadata and execution details
├── forecast.zarr/            # Main output directory
│   ├── .zarray               # Zarr metadata files
│   ├── .zattrs
│   └── chunks/               # Data chunks
│       ├── 0.0.0
│       └── ...
└── [additional output files] # Any other generated files
```

### Metadata File Contents

The `metadata.json` file contains:

* Request ID and execution status
* Workflow type and configuration
* Execution time and completion timestamp
* Device information and peak memory usage
* **Original request parameters** - Complete input parameters used for the inference
* **Output file manifest** - List of all files in the zip with their paths and sizes
* Created timestamp and other diagnostic information

This format ensures that all information needed to understand and use the inference results is
included in the download.

## API Schema

### Inference Request Schema

The `/v1/infer` endpoint accepts a unified request schema that maps directly to the function
parameters from `earth2studio.run`:

```yaml
InferenceRequest:
  workflow_type: "deterministic" | "diagnostic" | "ensemble"
  time: string | [string]  # Single timestamp or list of timestamps
  nsteps: integer          # Number of forecast steps
  nensemble: integer       # Required for ensemble workflows
  prognostic:              # Prognostic model configuration
    model_type: "fcn" | "graphcast" | "stormcast" | "dlwp"
    model_config: object   # Additional model parameters
  diagnostic:              # Required for diagnostic workflows
    model_type: "precipitation_afno" | "corrdiff_taiwan" | "cbottle"
    model_config: object   # Additional diagnostic model parameters
  data:                    # Data source configuration
    source_type: "gfs" | "era5" | "custom"
    source_config: object  # Additional data source parameters
  io:                      # IO backend configuration
    backend_type: "zarr" | "netcdf" | "hdf5"
    file_name: string      # Output file name
    backend_kwargs: object # Additional IO parameters
  perturbation:            # Required for ensemble workflows
    method: "spherical_gaussian" | "gaussian" | "custom"
    config: object         # Perturbation configuration
  output_coords: object    # IO output coordinate system override
  device: "cpu" | "cuda" | "auto"
  batch_size: integer      # For ensemble processing
```

## Configuration

### Environment Variables

* `REDIS_HOST`: Redis server hostname (default: "localhost")
* `REDIS_PORT`: Redis server port (default: 6379)
* `REDIS_DB`: Redis database number (default: 0)
* `REDIS_PASSWORD`: Redis password (default: None)
* `CUDA_VISIBLE_DEVICES`: Specify which GPUs to use (e.g., "0,1")
* `DEFAULT_OUTPUT_DIR`: Base directory for output files (default: `/tmp/earth2studio_outputs`)
* `RESULTS_TTL_HOURS`: Time in hours before results expire (default: 24)
* `CLEANUP_WATCHDOG_SEC`: Cleanup check interval in seconds (default: 900)

### Server Configuration (config.yaml)

The API server can be configured via the `config.yaml` file located at
`api_server/conf/config.yaml`:

#### Results Management

* `results_ttl_hours`: Time in hours that inference results (both raw files and zip files) are kept
  before automatic deletion (default: 24)
* `cleanup_watchdog_sec`: Interval in seconds at which the cleanup daemon checks for and removes
  expired results (default: 900, i.e., 15 minutes)

These settings can be configured via `config.yaml` or environment variables (`RESULTS_TTL_HOURS`,
`CLEANUP_WATCHDOG_SEC`).

The cleanup daemon runs as a separate process alongside the API server and RQ workers, and
periodically checks Redis for expired results based on their completion timestamps.

Example configuration:

```yaml
server:
  host: 0.0.0.0
  port: 8000
  results_ttl_hours: 24      # Keep results for 24 hours
  cleanup_watchdog_sec: 900  # Check every 15 minutes
```

Or via environment variables:

```bash
export RESULTS_TTL_HOURS=48
export CLEANUP_WATCHDOG_SEC=1800
```

When results expire:

* Both the zip file and raw result files are automatically deleted
* The inference request status is updated to `expired`
* Attempts to download expired results return a 400 Bad Request error

### Redis Persistence

The API uses Redis for persistent storage with automatic disk persistence:

* **RDB Snapshots**: Automatic backups every 900s, 300s, and 60s
* **AOF Logging**: Append-only file for transaction durability
* **Data Location**: Redis data is stored in `./redis-data/` volume (Docker) or local filesystem
* **TTL**: Inference request metadata expires after 7 days automatically (stored in Redis)
* **Results TTL**: Result files (zip and raw) are deleted based on `results_ttl_hours`
  configuration (default: 24 hours)
* **Memory Limit**: 256MB with LRU eviction policy

### Output File Organization

Output files are organized by workflow type and timestamp:

```text
/tmp/earth2studio_outputs/
├── deterministic_2024-01-01T000000Z/
│   └── forecast.zarr
├── ensemble_2024-01-01T000000Z/
│   └── ensemble_forecast.zarr
└── diagnostic_2024-01-01T000000Z/
    └── diagnostic_forecast.zarr
```

## Model Support

### Prognostic Models

* **FCN**: FourCastNet - Global weather forecasting
* **GraphCast**: Graph-based weather forecasting
* **StormCast**: Tropical cyclone forecasting
* **DLWP**: Deep Learning Weather Prediction

### Diagnostic Models

* **PrecipitationAFNO**: Precipitation prediction
* **CorrDiffTaiwan**: High-resolution downscaling for Taiwan
* **CBottle**: Custom diagnostic models

### Perturbation Methods

* **SphericalGaussian**: Spherical Gaussian noise for ensemble generation
* **Gaussian**: Standard Gaussian noise
* **Custom**: User-defined perturbation methods

### Data Sources

* **GFS**: Global Forecast System
* **ERA5**: ECMWF Reanalysis v5
* **Custom**: User-defined data sources

### IO Backends

* **Zarr**: Best for large datasets, supports chunking
* **NetCDF**: Standard format for climate/weather data
* **HDF5**: Hierarchical data format with compression

## Performance Considerations

### GPU Memory Management

* TBD

### Batch Processing

* TBD

### Model Caching

* TBD

## Error Handling

The API provides detailed error responses:

```json
{
  "error": "Invalid timestamp format",
  "details": "Timestamp must be in ISO 8601 format",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Scenarios

* **400 Bad Request**: Invalid input parameters or expired results
* **404 Not Found**: Inference request ID doesn't exist
* **500 Internal Server Error**: Server-side errors

### Results Endpoint Specific Errors

* **202 Accepted**: Request still running (check status endpoint)
* **400 Bad Request**: Results have expired and are no longer available
* **404 Not Found**: Request failed, not found, or has no output files
* **500 Internal Server Error**: Failed to create results zip file

### Example Error Handling for Results Download

```python
import requests

def download_results(request_id):
    try:
        response = requests.get(f"http://localhost:8000/v1/infer/{request_id}/results")

        if response.status_code == 200:
            # Success - save zip file
            with open(f"results_{request_id}.zip", 'wb') as f:
                f.write(response.content)
            return True

        elif response.status_code == 202:
            # Still running
            error_data = response.json()
            print(f"Request still running: {error_data['message']}")
            return False

        elif response.status_code == 400:
            # Bad request (e.g., expired results)
            error_data = response.json()
            print(f"Error: {error_data['error']} - {error_data['details']}")
            return False

        elif response.status_code == 404:
            # Not found or failed
            error_data = response.json()
            print(f"Error: {error_data['error']} - {error_data['details']}")
            return False

    except Exception as e:
        print(f"Failed to download results: {e}")
        return False
```

## Monitoring and Logging

### Health Check

```bash
curl http://localhost:8000/health
```

### Get Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

The metrics endpoint returns Prometheus-formatted metrics for monitoring the API server, including:

* TBD

### List All Inference Requests

```bash
curl http://localhost:8000/v1/infer
```

### Code Style

The API follows PEP 8 and uses type hints throughout.

### Debug Mode

Enable debug logging by setting the log level:

```python
logging.basicConfig(level=logging.DEBUG)
```

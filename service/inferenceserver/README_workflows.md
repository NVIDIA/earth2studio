# Implement customized REST APIs from Earth2Studio recipes

Transform your custom python Earth2Studio recipes into REST APIs.

## 📋 Table of Contents

1. [What Are Workflows?](#what-are-workflows)
2. [Creating Your First Workflow](#creating-your-first-workflow)
3. [Parameters: From Code to API](#parameters-from-code-to-api)
4. [Example: Deterministic Forecast Workflow](#example-deterministic-forecast-workflow)
5. [REST API Usage](#rest-api-usage)
6. [Auto Discovery System](#auto-discovery-system)
7. [Progress Tracking](#progress-tracking)

---

* *Workflows** are custom Python scripts that get automatically exposed as REST API endpoints.
  Turn any Earth2Studio computation into a scalable, API-accessible service.

### **Architecture Overview**

```text
Your Python Script → Workflow Class → REST API Endpoint
                                       ↓
        Parameters (Pydantic) → JSON API Parameters
                                       ↓
                                Workflow.run() → Redis State Management → File Results
```

---

## Key Concepts

### Status Management

* *Important**: Workflow execution status (queued, running, completed,
  failed) is **automatically managed by the system**. Your workflow code should focus on:

✅ **DO:**

* Set the `name` and `description` (optional) attributes of the class to name and
  description strings
* Set the `Parameters` (optional) and `Config` (optional) attributes of the class
  to the parameters and configuration options of the class
* Implement `validate_parameters()` method to validate inputs
* Use `update_execution_data()` to update progress tracking (see [Progress Tracking](#progress-tracking))
* Use `get_output_path()` to get output directory
* Save results to files in the output directory
* Return results from the `run()` method

❌ **DON'T:**

* Try to set or change the workflow status
* Access private methods (`_get_execution_data`, `_save_execution_data`, `_update_execution_data`)
* Manage execution state directly

### Workflow Lifecycle

1. **User submits request** → Status: `queued`
2. **System starts execution** → Status: `running`
3. **Your `run()` method executes** → You update progress metadata
4. **Execution completes successfully** → Status: `pending_results`
5. **Results are zipped and made available** → Status: `completed`
6. **Or execution fails** → Status: `failed`

The system handles status transitions automatically. You focus on your workflow logic and metadata
updates.

---

## Creating Your First Workflow

### Step 1: Define Your Parameters

Create a **Pydantic model** that defines your workflow's input parameters:

```python
from api_server.workflow import WorkflowParameters
from pydantic import Field
from typing import List

class MyWorkflowParameters(WorkflowParameters):
    """Parameters for my custom workflow"""

    # Required string parameter
    input_text: str = Field(description="Text to process")

    # Optional boolean with default
    uppercase: bool = Field(default=False, description="Convert to uppercase")

    # Numeric parameter with validation
    repeat_count: int = Field(default=1, ge=1, le=10, description="How many times to repeat")

    # List parameter
    tags: List[str] = Field(default=[], description="Tags to apply")
```

### Step 2: Create Your Workflow Class

Implement the **`Workflow`** base class and assign the parameter class to its `Parameters`
attribute:

```python
from api_server.workflow import Workflow, WorkflowProgress, workflow_registry
from typing import Any, Dict, Union
import json

@workflow_registry.register
class MyCustomWorkflow(Workflow):
    """My custom text processing workflow"""

    name = "my_text_processor"
    description = "Processes text with various transformations"
    Parameters = MyWorkflowParameters

    def validate_parameters(
        self,
        parameters: Union[Dict[str, Any], MyWorkflowParameters]
    ) -> MyWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return MyWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters for {self.name}: {e}") from e

    def run(
        self,
        parameters: Union[Dict[str, Any], MyWorkflowParameters],
        execution_id: str
    ) -> Dict[str, Any]:
        """Main workflow logic - this is where your script goes!"""

        # 1. Validate and convert parameters
        parameters = self.validate_parameters(parameters)

        # 2. Initialize metadata for tracking (status is managed by the system)
        metadata = {"parameters": parameters.model_dump()}

        try:
            # Store metadata
            self.update_execution_data(execution_id, {"metadata": metadata})

            # 3. YOUR CUSTOM LOGIC GOES HERE
            # ================================

            # Update progress using WorkflowProgress
            progress = WorkflowProgress(progress="Processing text...")
            self.update_execution_data(execution_id, progress)

            result_text = parameters.input_text

            if parameters.uppercase:
                result_text = result_text.upper()

            result_text = result_text * parameters.repeat_count

            # Add tags if provided
            if parameters.tags:
                result_text += f" [Tags: {', '.join(parameters.tags)}]"

            # 4. Save results to file
            output_dir = self.get_output_path(execution_id)
            output_path = output_dir / "result.txt"
            with open(output_path, "w") as f:
                f.write(result_text)

            # 5. Update final progress and metadata
            progress = WorkflowProgress(progress="Complete!")
            self.update_execution_data(execution_id, progress)

            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": f"Processed {len(result_text)} characters"
                    }
                }
            )

            return {
                "status": "success",
                "result": result_text,
                "output_path": str(output_path)
            }

        except Exception as e:
            # Update error progress
            progress = WorkflowProgress(progress="Failed!")
            self.update_execution_data(execution_id, progress)
            raise
```

### Step 3: Register Your Workflow

The `@workflow_registry.register` decorator will ensure that your workflow gets added to the
workflow registry as long as it is found in a file within the directory set by the `WORKFLOW_DIR`
environment variable.

* *That's it!** Save your file to a directory, set `WORKFLOW_DIR=/path/to/your/workflows`,
  and start the server.
   Your workflow will be automatically registered and available via REST API at
   `/v1/infer/my_text_processor`.

---

## Parameters: From Code to API

### Pydantic → JSON API Mapping

Your **Pydantic fields** automatically become **JSON API parameters**:

```python
class MyParameters(WorkflowParameters):
    # String parameter
    name: str = Field(description="Your name")

    # Numeric with validation
    age: int = Field(ge=0, le=150, description="Your age")

    # Optional with default
    country: str = Field(default="US", description="Country code")

    # Boolean flag
    send_email: bool = Field(default=False, description="Send confirmation email")

    # List/Array
    interests: List[str] = Field(default=[], description="List of interests")

    # Enum choices
    priority: Literal["low", "medium", "high"] = Field(default="medium")
```

**Becomes this JSON API:**

```json
{
  "parameters": {
    "name": "John Doe",
    "age": 30,
    "country": "US",
    "send_email": true,
    "interests": ["AI", "Weather", "Climate"],
    "priority": "high"
  }
}
```

### **Pydantic Field Options:**

| Field Option | Purpose | Example |
|-------------|---------|---------|
| `description` | API documentation | `Field(description="Input text to process")` |
| `default` | Default value | `Field(default="hello", description="...")` |
| `ge`, `le` | Numeric validation | `Field(ge=1, le=100, description="Count")` |
| `min_length`, `max_length` | String validation | `Field(min_length=1, max_length=1000)` |
| `regex` | Pattern validation | `Field(regex=r"^[A-Z]{2}$", description="Country code")` |

---

## Example: Deterministic Forecast Workflow

Let's examine a real Earth2Studio workflow that runs a deterministic weather forecast:

### Parameters Definition

```python
class DeterministicWorkflowParameters(WorkflowParameters):
    """Parameters for deterministic weather forecast"""

    # Forecast configuration
    forecast_times: List[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)"
    )
    nsteps: int = Field(
        default=20, ge=1, le=100,
        description="Number of forecast steps (each step is 6 hours)"
    )

    # Model selection
    model_type: str = Field(
        default="dlwp",
        description="Prognostic model type (dlwp, fcn)"
    )

    # Data source
    data_source: str = Field(
        default="gfs",
        description="Data source for initialization (gfs, era5)"
    )

    # Output options
    output_format: str = Field(
        default="zarr",
        description="Output format (zarr, netcdf)"
    )

    # Visualization
    create_plots: bool = Field(
        default=True,
        description="Whether to create visualization plots"
    )
    plot_variable: str = Field(
        default="t2m",
        description="Variable to plot (t2m, msl, u10m, v10m, etc.)"
    )
    plot_step: int = Field(
        default=4, ge=0,
        description="Forecast step to plot (step 4 = 24 hours)"
    )
```

### Workflow Implementation

```python
from api_server.workflow import WorkflowProgress

@workflow_registry.register()
class DeterministicWorkflow(Workflow):
    """Earth2Studio deterministic forecast workflow"""

    name = "deterministic_workflow"
    description = "Earth2Studio deterministic forecast with visualization"
    Parameters = DeterministicWorkflowParameters

    def validate_parameters(
        self,
        parameters: Union[Dict[str, Any], DeterministicWorkflowParameters]
    ) -> DeterministicWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return DeterministicWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters for {self.name}: {e}") from e

    def run(self, parameters, execution_id):
        """Run deterministic weather forecast"""

        # Validate and convert parameters
        parameters = self.validate_parameters(parameters)

        # Initialize metadata for tracking
        metadata = {"parameters": parameters.model_dump()}

        try:
            # Store metadata
            self.update_execution_data(execution_id, {"metadata": metadata})

            # Import Earth2Studio components
            progress = WorkflowProgress(
                progress="Loading Earth2Studio components...",
                current_step=1,
                total_steps=6
            )
            self.update_execution_data(execution_id, progress)

            from earth2studio import run
            from earth2studio.data import GFS
            from earth2studio.io import ZarrBackend
            from earth2studio.models.px import DLWP, FCN

            # Load model based on parameter
            if parameters.model_type.lower() == "dlwp":
                package = DLWP.load_default_package()
                model = DLWP.load_model(package)
            elif parameters.model_type.lower() == "fcn":
                package = FCN.load_default_package()
                model = FCN.load_model(package)

            # Set up data source
            data = GFS()

            # Configure output
            output_dir = self.get_output_path(execution_id)
            output_path = output_dir / f"forecast.{parameters.output_format}"
            io = ZarrBackend(file_name=str(output_path))

            # Run the forecast!
            progress = WorkflowProgress(
                progress=f"Running {parameters.nsteps}-step forecast...",
                current_step=4,
                total_steps=6
            )
            self.update_execution_data(execution_id, progress)

            io = run.deterministic(
                parameters.forecast_times,
                parameters.nsteps,
                model,
                data,
                io
            )

            # Update final progress and metadata
            progress = WorkflowProgress(progress="Complete!")
            self.update_execution_data(execution_id, progress)

            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": f"Generated {parameters.nsteps}-step forecast"
                    }
                }
            )

            return {
                "status": "success"
            }

        except Exception as e:
            # Update error progress
            progress = WorkflowProgress(progress="Failed!")
            self.update_execution_data(execution_id, progress)
            raise
```

---

## REST API Usage

Once registered, your workflow becomes available via REST API:

### 🌐 **Available Endpoints:**

<!-- markdownlint-disable MD013 -->
| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/v1/workflows` | List all available workflows |
| `POST` | `/v1/infer/{workflow_name}` | Execute a workflow |
| `GET` | `/v1/infer/{workflow_name}/{execution_id}/status` | Check execution status |
| `GET` | `/v1/infer/{workflow_name}/{execution_id}/results` | Get results metadata with file manifest |
| `GET` | `/v1/infer/{workflow_name}/{execution_id}/results/{filepath}` | Download a specific file |
<!-- markdownlint-enable MD013 -->

---

### **1. List Available Workflows**

```bash
curl -X GET "http://localhost:8000/workflows"
```

**Response:**

```json
{
  "workflows": {
    "deterministic_workflow": "Earth2Studio deterministic forecast with visualization"
  }
}
```

---

### **2. Execute a Workflow**

**Deterministic Forecast Example:**

```bash
curl -X POST "http://localhost:8000/v1/infer/deterministic_workflow" \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "forecast_times": ["2024-01-01T00:00:00"],
      "nsteps": 10,
      "model_type": "dlwp",
      "data_source": "gfs",
      "create_plots": true,
      "plot_variable": "t2m",
      "plot_step": 4
    }
  }'
```

**Response:**

```json
{
  "workflow_name": "deterministic_workflow",
  "execution_id": "exec_1703123456_abc12345",
  "status": "queued",
  "position": 2,
  "message": "Workflow 'deterministic_workflow' queued for execution",
  "timestamp": "2024-01-01T12:00:00.000Z"
}
```

* *Note:** The `position` field indicates the job's position in the execution queue (0-indexed).
  This field is only present when `status` is `"queued"`.

---

### **3. Check Execution Status**

```bash
curl -X GET "http://localhost:8000/v1/infer/deterministic_workflow/exec_1703123456_abc12345/status"
```

**Response (Queued):**

```json
{
  "workflow_name": "deterministic_workflow",
  "execution_id": "exec_1703123456_abc12345",
  "status": "queued",
  "position": 2,
  "start_time": "2024-01-01T12:00:00.000Z",
  "metadata": {
    "parameters": {
      "forecast_times": ["2024-01-01T00:00:00"],
      "nsteps": 10,
      "model_type": "dlwp"
    }
  }
}
```

* *Note:** When `status` is `"queued"`,
  the `position` field shows the job's current position in the execution queue (0-indexed).
  This position may change as jobs ahead complete. Once execution starts,
  the `position` field is no longer included.

**Response (Running):**

```json
{
  "workflow_name": "deterministic_workflow",
  "execution_id": "exec_1703123456_abc12345",
  "status": "running",
  "progress": {
    "progress": "Running 10-step forecast...",
    "current_step": 4,
    "total_steps": 6
  },
  "start_time": "2024-01-01T12:00:00.000Z",
  "execution_time_seconds": 45.2,
  "metadata": {
    "parameters": {
      "forecast_times": ["2024-01-01T00:00:00"],
      "nsteps": 10,
      "model_type": "dlwp"
    }
  }
}
```

**Response (Completed):**

```json
{
  "workflow_name": "deterministic_workflow",
  "execution_id": "exec_1703123456_abc12345",
  "status": "completed",
  "progress": {
    "progress": "Complete!",
    "current_step": 6,
    "total_steps": 6
  },
  "start_time": "2024-01-01T12:00:00.000Z",
  "end_time": "2024-01-01T12:15:30.000Z",
  "execution_time_seconds": 930.5,
  "metadata": {
    "parameters": {
      "forecast_times": ["2024-01-01T00:00:00"],
      "nsteps": 10,
      "model_type": "dlwp"
    },
    "results_summary": "Generated 10-step forecast for 1 time(s)",
    "forecast_info": {
      "model_type": "dlwp",
      "nsteps": 10,
      "zarr_tree": "..."
    }
  }
}
```

---

### **4. Get Results Metadata**

The results endpoint returns JSON metadata with an output file manifest:

```bash
curl -X GET "http://localhost:8000/v1/infer/deterministic_workflow/exec_1703123456_abc12345/results"
```

**Response:**

```json
{
  "request_id": "exec_1703123456_abc12345",
  "status": "completed",
  "workflow_name": "deterministic_workflow",
  "completion_time": "2024-01-01T12:15:30.000Z",
  "execution_time_seconds": 930.5,
  "created_at": "2024-01-01T12:00:00.000Z",
  "zip_created_at": "2024-01-01T12:15:35.000Z",
  "parameters": {
    "forecast_times": ["2024-01-01T00:00:00"],
    "nsteps": 10,
    "model_type": "dlwp"
  },
  "output_files": [
    {"path": "exec_1703123456_abc12345/forecast.zarr/.zarray", "size": 245},
    {"path": "exec_1703123456_abc12345/forecast.zarr/t2m/0.0", "size": 1048576},
    {"path": "exec_1703123456_abc12345/forecast_plot.png", "size": 52430},
    {"path": "deterministic_workflow:exec_1703123456_abc12345", "size": 2097152}
  ]
}
```

The `output_files` manifest contains:

* **Individual files**: Output files from the workflow execution
* **Zip file entry**: The complete zip archive (path format: `{workflow_name}:{execution_id}`)

---

### **5. Download Files**

Use the file paths from the manifest to download individual files or the complete zip:

**Download the complete zip file:**

```bash
# The zip file path is "{workflow_name}:{execution_id}"
curl -X GET \
  "http://localhost:8000/v1/infer/deterministic_workflow/exec_1703123456_abc12345/results/\
deterministic_workflow:exec_1703123456_abc12345" \
  --output forecast_results.zip
```

**Download an individual file:**

```bash
curl -X GET \
  "http://localhost:8000/v1/infer/deterministic_workflow/exec_1703123456_abc12345/results/\
exec_1703123456_abc12345/forecast_plot.png" \
  --output forecast_plot.png
```

**Features:**

* **File manifest**: Know exactly what files are available before downloading
* **Individual file access**: Download only the files you need
* **Complete zip**: Download everything at once via the zip entry
* **Streaming**: Large files streamed efficiently

**Access Zarr Results with xarray:**

For workflows that output zarr format, you can open the data directly with xarray using the HTTP
endpoint:

```python
import xarray as xr

# Build the URL to the zarr store
execution_id = "exec_1703123456_abc12345"
workflow_name = "deterministic_workflow"
base_url = "http://localhost:8000"

path = f"{execution_id}/results.zarr"
url = f"{base_url}/v1/infer/{workflow_name}/{execution_id}/results/{path}"

# Open directly with xarray
ds = xr.open_zarr(url)
ds.info()
```

This provides direct access to the zarr data without downloading the entire zip file.

* Note*: Object store integration is a work in progress wherein, results will be saved in object
  store, and available for download.

**Zip Contents:**

```text
forecast_results.zip/
├── exec_1703123456_abc12345/
│   ├── forecast.zarr/
│   │   ├── .zarray
│   │   ├── lat/
│   │   ├── lon/
│   │   ├── time/
│   │   ├── t2m/
│   │   └── ...
│   ├── forecast_metadata.json     # Workflow-specific metadata (if created by workflow)
│   └── forecast_plot_t2m_step4.png
```

Note: The `metadata.json` file is returned via the `/results` endpoint as JSON (not inside the zip).

---

### 🔄 **Complete Workflow Example**

Code for a complete workflow example is available in the file: test/integration/test_workflow.py

---

## Auto-Discovery System

The workflow system automatically discovers and registers workflows from directories.

**Built-in workflows** from `example_workflows/` are **always available**

### Quick Start

To add your own custom workflows:

1. **Create your workflow file** in any directory:

   ```bash
   mkdir ~/my_workflows
   # Create your workflow file (see example_user_workflow.py)
   ```

2. **Set environment variable** pointing to your workflows:

   ```bash
   export WORKFLOW_DIR=~/my_workflows
   ```

3. **Start the server** - workflows are auto-registered:

Your custom workflows will be registered alongside the built-in example workflows as long as they
have the `@workflow_registry.register()` decorator.

### Example Usage

**Single directory:**

```bash
export WORKFLOW_DIR=/home/user/my_workflows
```

**Multiple directories:**

```bash
export WORKFLOW_DIR=/home/user/workflows,/opt/shared_workflows,/data/team_workflows
```

---

## Progress Tracking

Workflows can track progress using the `WorkflowProgress` class, which provides structured progress
updates in the API responses.

### Base Progress Fields

The `WorkflowProgress` class provides four standard fields:

* `progress`: Human-readable progress message
* `current_step`: Current step number
* `total_steps`: Total number of steps
* `error_message`: Error message to report specific errors when workflow fails

### Basic Usage

```python
from api_server.workflow import WorkflowProgress

# Simple progress update
progress = WorkflowProgress(
    progress="Processing data...",
    current_step=3,
    total_steps=10
)
self.update_execution_data(execution_id, progress)

# Error reporting
try:
    # ... workflow logic ...
except Exception as e:
    progress = WorkflowProgress(
        progress="Failed!",
        error_message=str(e)
    )
    self.update_execution_data(execution_id, progress)
    raise
```

### Custom Progress Fields

You can extend `WorkflowProgress` with workflow-specific fields:

```python
from api_server.workflow import WorkflowProgress
from pydantic import Field
from typing import Optional

class MyCustomProgress(WorkflowProgress):
    """Extended progress with custom fields"""

    # Inherited: progress, current_step, total_steps

    # Custom fields specific to your workflow
    data_processed_gb: Optional[float] = Field(None, description="Data processed in GB")
    processing_stage: str = Field("initialization", description="Current processing stage")
    error_count: Optional[int] = Field(None, description="Number of errors encountered")

@workflow_registry.register
class MyWorkflow(Workflow):
    name = "my_workflow"

    def run(self, parameters, execution_id):
        # Use custom progress with additional fields
        progress = MyCustomProgress(
            progress="Processing batch 5",
            current_step=5,
            total_steps=20,
            data_processed_gb=150.5,
            processing_stage="inference",
            error_count=2
        )
        self.update_execution_data(execution_id, progress)
```

### API Response with Progress

Progress is returned as a dedicated `progress` field in API status responses:

```json
{
  "workflow_name": "my_workflow",
  "execution_id": "exec_123",
  "status": "running",
  "progress": {
    "progress": "Processing batch 5",
    "current_step": 5,
    "total_steps": 20,
    "data_processed_gb": 150.5,
    "processing_stage": "inference",
    "error_count": 2
  },
  "start_time": "2024-01-01T12:00:00.000Z",
  "metadata": {
    "parameters": {...}
  }
}
```

**Benefits:**

* ✅ Type-safe and validated via Pydantic
* ✅ Structured data in API responses (not buried in metadata)
* ✅ Extensible with workflow-specific fields
* ✅ Consistent progress tracking across all workflows

---

## Summary

You now have a complete workflow system that can:

**Transform Python scripts** into REST APIs
**Handle parameters** with automatic validation
**Process asynchronously** with status tracking
**Store and serve results** automatically
**Auto-discover workflows** from directories

### Next Steps

* *Create your first workflow** using the templates
  ([example_user_workflow.py](example_user_workflow.py))

---

* For more examples, check out the `deterministic_workflow.py`, `example_user_workflow.py`, and
  workflow discovery documentation in this repository.*

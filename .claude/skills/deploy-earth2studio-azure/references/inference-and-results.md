# Inference And Results

## Invoke With Azure ML CLI

`az ml online-endpoint invoke` always POSTs to the `scoring_route` (configured as `/v1/infer`).
It cannot target workflow-specific paths. Use it only for single-workflow deployments or when
`/v1/infer` is the intended route:

```bash
az ml online-endpoint invoke \
  --name "<endpoint-name>" \
  --deployment-name "<deployment-name>" \
  --request-file .claude/skills/deploy-earth2studio-azure/assets/requests/foundry_fcn3_smoke.json
```

`foundry_fcn3_smoke.json` is the canonical smoke request. For named-workflow or multi-workflow
invocations use the Direct HTTP Pattern below.

When Azure Blob output is desired, add this request parameter:

```json
"container_url": "https://<storage-account>.blob.core.windows.net/<container>"
```

## Direct HTTP Pattern

```bash
SCORING_URI="$(az ml online-endpoint show \
  --name "<endpoint-name>" --query scoring_uri -o tsv)"
KEY="$(az ml online-endpoint get-credentials \
  --name "<endpoint-name>" --query primaryKey -o tsv)"
API_BASE_URL="${SCORING_URI%/v1/infer}"

# Inspect available workflows and parameter schemas:
curl -sS "$API_BASE_URL/v1/infer/workflows" \
  -H "Authorization: Bearer $KEY"
curl -sS "$API_BASE_URL/v1/infer/workflows/<workflow_name>/schema" \
  -H "Authorization: Bearer $KEY"

# Single-workflow deployment (scoring_route is /v1/infer):
curl -sS -X POST "$SCORING_URI" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  --data-binary @.claude/skills/deploy-earth2studio-azure/assets/requests/foundry_fcn3_smoke.json

# Workflow-specific (required when targeting a named workflow directly):
curl -sS -X POST "$API_BASE_URL/v1/infer/<workflow_name>" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  --data-binary @<request.json>
```

The submit response contains `workflow_name` and `execution_id`. Poll status and results:

```bash
curl -sS "$API_BASE_URL/v1/infer/$WORKFLOW_NAME/$EXECUTION_ID/status" \
  -H "Authorization: Bearer $KEY"

curl -sS "$API_BASE_URL/v1/infer/$WORKFLOW_NAME/$EXECUTION_ID/results" \
  -H "Authorization: Bearer $KEY"
```

## Foundry Request Parameters

Parameters are workflow-specific. Retrieve them from the live schema API (requires
`API_BASE_URL` and `KEY` from the Direct HTTP Pattern above):

```bash
curl -sS "$API_BASE_URL/v1/infer/workflows/<workflow_name>/schema" \
  -H "Authorization: Bearer $KEY"
```

Or inspect the workflow's `__call__` signature in `serve/server/example_workflows/`.

`foundry_fcn3_smoke.json` shows the minimal set for `foundry_fcn3_workflow` as a starting point.
Use small smoke values first: `n_steps: 1`, one sample, and one or a few variables.

## Xarray Via Earth2Studio Client

Prefer this when the client can reach the Earth2Studio API base URL:

```python
from datetime import datetime
from earth2studio.serve.client.e2client import RemoteEarth2Workflow

workflow = RemoteEarth2Workflow(
    api_url,
    workflow_name="foundry_fcn3_workflow",
    token=token,
    device="cpu",
)

result = workflow(
    start_time=datetime(2025, 1, 1, 0),
    n_steps=1,
    n_samples=1,
    variables=["t2m"],
    output_format="zarr",
    container_url="https://<storage-account>.blob.core.windows.net/<container>",
)
ds = result.as_dataset()
print(ds)
```

## Xarray Directly From Azure Blob

For Zarr results, use Azure AD credentials from the client side:

```python
from azure.identity import DefaultAzureCredential
import xarray as xr

ds = xr.open_zarr(
    "az://<container>/<prefix>/<execution_id>/results.zarr",
    storage_options={
        "account_name": "<storage-account>",
        "credential": DefaultAzureCredential(),
    },
    consolidated=True,
)
print(ds)
```

For NetCDF results:

```python
from azure.identity import DefaultAzureCredential
import adlfs
import xarray as xr

fs = adlfs.AzureBlobFileSystem(
    account_name="<storage-account>",
    credential=DefaultAzureCredential(),
)

with fs.open("<container>/<prefix>/<execution_id>/results.nc", "rb") as f:
    ds = xr.open_dataset(f)
    print(ds)
```

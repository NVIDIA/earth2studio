# Inference And Results

## Invoke With Azure ML CLI

Submit a smoke request through the deployment's `scoring_route`:

```bash
az ml online-endpoint invoke \
  --name "<endpoint-name>" \
  --deployment-name "<deployment-name>" \
  --request-file .claude/skills/deploy-earth2studio-azure/assets/requests/foundry_fcn3_smoke.json
```

Use the StormScope request asset for `foundry_fcn3_stormscope_goes_workflow`.

When Azure Blob output is desired, add this request parameter:

```json
"container_url": "https://<storage-account>.blob.core.windows.net/<container>"
```

## Direct HTTP Pattern

If using the scoring URI directly:

```bash
SCORING_URI="$(az ml online-endpoint show --name "<endpoint-name>" --query scoring_uri -o tsv)"
KEY="$(az ml online-endpoint get-credentials --name "<endpoint-name>" --query primaryKey -o tsv)"

curl -sS -X POST "$SCORING_URI" \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  --data-binary @.claude/skills/deploy-earth2studio-azure/assets/requests/foundry_fcn3_smoke.json
```

The submit response contains `workflow_name` and `execution_id`. If the Azure ML ingress exposes the Earth2Studio paths, derive `API_BASE_URL` by removing the `/v1/infer` suffix from the scoring URI and poll:

```bash
curl -sS "$API_BASE_URL/v1/infer/$WORKFLOW_NAME/$EXECUTION_ID/status" \
  -H "Authorization: Bearer $KEY"

curl -sS "$API_BASE_URL/v1/infer/$WORKFLOW_NAME/$EXECUTION_ID/results" \
  -H "Authorization: Bearer $KEY"
```

## Foundry Request Parameters

`foundry_fcn3_workflow` supports:

- `start_time`
- `n_steps`
- `n_samples`
- `seeds`
- `variables`
- `output_format`: `zarr` or `netcdf4`
- `container_url`
- `geo_catalog_url`
- `collection_id`

`foundry_fcn3_stormscope_goes_workflow` supports:

- `start_time_fcn3`
- `start_time_stormscope`
- `n_steps`
- `n_samples_fcn3`
- `n_samples_stormscope`
- `seeds_fcn3`
- `seeds_stormscope`
- `variables`
- `output_format`: `zarr` or `netcdf4`
- `container_url`
- `geo_catalog_url`
- `collection_id`

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

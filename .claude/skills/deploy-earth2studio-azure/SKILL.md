---
name: deploy-earth2studio-azure
description: Deploy Earth2Studio serve inference to Azure ML managed online endpoints using serve/Dockerfile, ACR images, Foundry workflows, Azure Blob result storage, inference testing, and xarray result access. Use this skill when building or pushing Earth2Studio inference containers, creating or updating Azure ML endpoint/deployment YAML, invoking Foundry workflows, or reading Azure Blob/Zarr/NetCDF results from Azure-hosted Earth2Studio inference.
---

# Deploy Earth2Studio To Azure

Use this skill to deploy the Earth2Studio inference server in this repo to Azure ML.

## Repo Facts

- Dockerfile: `serve/Dockerfile`
- Built-in workflow directory in the image: `serve/server/example_workflows`
- Azure startup script honors `SERVER_PORT`; the known Azure ML deployments use `8080`.
- Custom workflow API supports `POST /v1/infer/{workflow_name}` and also `POST /v1/infer` when
  exactly one workflow is exposed.
- Foundry workflow names:
  - `foundry_fcn3_workflow`
  - `foundry_fcn3_stormscope_goes_workflow`

## Deployment Assets (source of truth)

All canonical deployment related files live in `serve/server/deployment/azure/`:

| File | Purpose |
|---|---|
| `foundry_fcn3.endpoint.yml` | Azure ML endpoint definition |
| `foundry_fcn3.deployment.yml` | Azure ML deployment definition |
| `requests/foundry_fcn3_smoke.json` | Canonical smoke-test request |
| `earth2studio-serving.md` | Build/push and container behavior notes |
| `azure-ml-managed-online.md` | Endpoint and deployment runbook |
| `inference-and-results.md` | Testing inference and reading results |

## Workflow

1. Build and push the image to ACR.
   - Read `serve/server/deployment/azure/earth2studio-serving.md`.
   - Use `serve/Dockerfile` from the repo root.
   - Tag images with an explicit version, usually a Git SHA or release tag.

2. Create or update the Azure ML endpoint and deployment.
   - Read `serve/server/deployment/azure/azure-ml-managed-online.md`.
   - Start from the canonical templates:
     - `serve/server/deployment/azure/foundry_fcn3.endpoint.yml`
     - `serve/server/deployment/azure/foundry_fcn3.deployment.yml`
   - These templates cover all workflows. To target a different workflow, change
     `EXPOSED_WORKFLOWS` (e.g. `foundry_fcn3_stormscope_goes_workflow`), the `endpoint_name`,
     and the image tag. All other fields (instance type, routing, storage config) stay the same.
   - For single-workflow Azure ML deployments, keep `scoring_route.path: /v1/infer` and set
     exactly one `EXPOSED_WORKFLOWS` value.

3. Test inference.
   - Read `serve/server/deployment/azure/inference-and-results.md`.
   - Start from `serve/server/deployment/azure/requests/foundry_fcn3_smoke.json`
     as the canonical smoke request.
   - For other workflows, infer the request parameters from the live schema API
     (`GET /v1/infer/workflows/{workflow_name}/schema`) or from the workflow's `__call__` signature
     in `serve/server/example_workflows/`.
   - Add `container_url` when results should be uploaded to Azure Blob Storage.

4. Open results with xarray.
   - Prefer `RemoteEarth2Workflow(...).as_dataset()` when using the Earth2Studio client.
   - Use direct Azure Blob access with `adlfs`, `azure-identity`, and `xarray` when working from
     `remote_path` or blob URLs.

## Azure Blob Rules

- Server upload uses Azure `DefaultAzureCredential`.
- The endpoint/deployment managed identity needs `Storage Blob Data Contributor` on the target
  storage account or container.
- Azure reads do not use server-generated SAS URLs; client-side code must authenticate to Blob Storage.
- `container_url` is supplied per inference request, not as an environment variable.
- `geo_catalog_url` requires `container_url` and `output_format: "netcdf4"`.

## IAM / Role Assignment Requirements

- **Model storage (read)**: The endpoint's managed identity must have at minimum
  `Storage Blob Data Reader` on the storage container (or account) that holds
  model weights. Without this, the deployment will fail to load the model at
  startup or at inference time with a permissions error.
- **Inference result storage (write)**: If `container_url` is provided in the
  inference request so that results are uploaded to Azure Blob, the managed
  identity must have `Storage Blob Data Contributor` on that target container
  or storage account. Read-only roles are not sufficient for result upload.
- **Role propagation delay**: Azure RBAC changes can take several minutes to
  propagate. If a freshly assigned role appears correct but inference still
  fails with `AuthorizationPermissionMismatch`, wait a few minutes and retry
  before deeper debugging.

## Networking Requirements

- **Storage account firewall**: Any storage account accessed by the deployment
  (model weights, input data, result output) must have its network settings
  configured to allow access from the Azure ML managed endpoint. By default,
  storage accounts with restricted networking will block the deployment's
  outbound requests, causing silent failures or `AuthorizationFailure` errors
  at inference time.
- **Check first**: Before debugging inference failures, verify that the
  deployment's managed identity can reach all storage accounts and ACR —
  networking misconfigurations are the most common cause of deployments that
  create successfully but fail at runtime.

## Iterating On This Skill

After a real deployment, failed deployment, or changed Azure YAML:

1. Capture the reusable fact, command, or failure mode.
2. Update the relevant file in `serve/server/deployment/azure/` directly — that is the source of truth.
3. Propose a patch to `SKILL.md` only when the workflow steps or rules change.
4. Do not store credentials, tokens, subscription-private secrets, or one-off logs.

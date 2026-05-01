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
- Custom workflow API supports `POST /v1/infer/{workflow_name}` and also `POST /v1/infer` when exactly one workflow is exposed.
- Foundry workflow names:
  - `foundry_fcn3_workflow`
  - `foundry_fcn3_stormscope_goes_workflow`

## Workflow

1. Build and push the image to ACR.
   - Read `references/earth2studio-serving.md`.
   - Use `serve/Dockerfile` from the repo root.
   - Tag images with an explicit version, usually a Git SHA or release tag.

2. Create or update the Azure ML endpoint and deployment.
   - Read `references/azure-ml-managed-online.md`.
   - Start from one of:
     - `assets/azureml/foundry_fcn3.endpoint.yml`
     - `assets/azureml/foundry_fcn3.deployment.yml`
     - `assets/azureml/foundry_fcn3_stormscope_goes.endpoint.yml`
     - `assets/azureml/foundry_fcn3_stormscope_goes.deployment.yml`
   - For single-workflow Azure ML deployments, keep `scoring_route.path: /v1/infer` and set exactly one `EXPOSED_WORKFLOWS` value.

3. Test inference.
   - Read `references/inference-and-results.md`.
   - Start from one of:
     - `assets/requests/foundry_fcn3_smoke.json`
     - `assets/requests/foundry_fcn3_stormscope_goes_smoke.json`
   - Add `container_url` when results should be uploaded to Azure Blob Storage.

4. Open results with xarray.
   - Prefer `RemoteEarth2Workflow(...).as_dataset()` when using the Earth2Studio client.
   - Use direct Azure Blob access with `adlfs`, `azure-identity`, and `xarray` when working from `remote_path` or blob URLs.

## Azure Blob Rules

- Server upload uses Azure `DefaultAzureCredential`.
- The endpoint/deployment managed identity needs `Storage Blob Data Contributor` on the target storage account or container.
- Azure reads do not use server-generated SAS URLs; client-side code must authenticate to Blob Storage.
- `container_url` is supplied per inference request, not as an environment variable.
- `geo_catalog_url` requires `container_url` and `output_format: "netcdf4"`.

## Iterating On This Skill

After a real deployment, failed deployment, or changed Azure YAML:

1. Capture the reusable fact, command, or failure mode.
2. Propose a small patch to this skill.
3. Do not store credentials, tokens, subscription-private secrets, or one-off logs.
4. Keep stable workflow rules in `SKILL.md`; move detailed examples to `references/` or `assets/`.

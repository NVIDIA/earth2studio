# Azure ML Managed Online Deployment

## Prerequisites

Use Azure CLI with the ML extension:

```bash
az extension add --name ml --upgrade
az login
az account set --subscription "<subscription-id-or-name>"
az configure --defaults group="<resource-group>" workspace="<azureml-workspace>"
```

Required Azure resources:

- Azure ML workspace
- ACR containing the Earth2Studio image
- managed online endpoint
- Azure Storage account/container when Azure Blob result upload is used

## Endpoint

Create the endpoint first:

```bash
az ml online-endpoint create \
  -f .claude/skills/deploy-earth2studio-azure/assets/azureml/foundry_fcn3.endpoint.yml
```

or:

```bash
az ml online-endpoint create \
  -f .claude/skills/deploy-earth2studio-azure/assets/azureml/foundry_fcn3_stormscope_goes.endpoint.yml
```

The endpoint assets mirror known working endpoint YAML. Patch the endpoint name when deploying a new endpoint.

If Azure Blob uploads fail with credential or authorization errors, inspect the managed identity available to the deployment and grant Blob write permissions. Azure Blob upload uses `DefaultAzureCredential`.

```bash
ENDPOINT_NAME="<endpoint-name>"
STORAGE_ACCOUNT="<storage-account>"
STORAGE_RESOURCE_GROUP="<storage-resource-group>"

PRINCIPAL_ID="$(az ml online-endpoint show --name "$ENDPOINT_NAME" --query identity.principal_id -o tsv)"
STORAGE_SCOPE="$(az storage account show --name "$STORAGE_ACCOUNT" --resource-group "$STORAGE_RESOURCE_GROUP" --query id -o tsv)"

az role assignment create \
  --assignee "$PRINCIPAL_ID" \
  --role "Storage Blob Data Contributor" \
  --scope "$STORAGE_SCOPE"
```

## Deployment

Start from the deployment asset for the target workflow:

```bash
az ml online-deployment create \
  -f .claude/skills/deploy-earth2studio-azure/assets/azureml/foundry_fcn3.deployment.yml \
  --all-traffic
```

or:

```bash
az ml online-deployment create \
  -f .claude/skills/deploy-earth2studio-azure/assets/azureml/foundry_fcn3_stormscope_goes.deployment.yml \
  --all-traffic
```

For an existing deployment, use `az ml online-deployment update -f <file>`.

## Routing Rule

The deployment assets use:

```yaml
scoring_route:
  port: 8080
  path: /v1/infer
```

This is correct only when `EXPOSED_WORKFLOWS` contains exactly one workflow. Earth2Studio dispatches `POST /v1/infer` to the single exposed workflow. If multiple workflows are exposed, use workflow-specific requests to `/v1/infer/{workflow_name}` or deploy one Azure ML endpoint/deployment per workflow.

## Operations

Show deployment state:

```bash
az ml online-deployment show --name "<deployment-name>" --endpoint-name "<endpoint-name>"
```

Get logs:

```bash
az ml online-deployment get-logs \
  --name "<deployment-name>" \
  --endpoint-name "<endpoint-name>" \
  --lines 200
```

Get scoring URI:

```bash
az ml online-endpoint show --name "<endpoint-name>" --query scoring_uri -o tsv
```

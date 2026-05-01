# Earth2Studio Serving Notes

## Build And Push

Run image builds from the repo root so `serve/Dockerfile` can copy `serve`, `examples`, `pyproject.toml`, `README.md`, and `earth2studio`.

Local Docker build and push:

```bash
ACR_NAME="<acr-name>"
TAG="$(git rev-parse --short HEAD)"
ACR_LOGIN_SERVER="$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)"
IMAGE="$ACR_LOGIN_SERVER/e2s-scicomp:$TAG"

az acr login --name "$ACR_NAME"
docker build -f serve/Dockerfile -t "$IMAGE" .
docker push "$IMAGE"
```

ACR remote build, useful when the local machine cannot build the GPU image:

```bash
ACR_NAME="<acr-name>"
TAG="$(git rev-parse --short HEAD)"
az acr build --registry "$ACR_NAME" --image "e2s-scicomp:$TAG" --file serve/Dockerfile .
```

## Container Behavior

`serve/Dockerfile` sets:

- `SCRIPT_DIR=/workspace/earth2studio-project/serve/server/scripts`
- `CONFIG_DIR=/workspace/earth2studio-project/serve/server/conf`
- `WORKFLOW_DIR=/workspace/earth2studio-project/serve/server/example_workflows`

Azure ML deployments should set:

```yaml
environment_variables:
  SERVER_PORT: 8080
  OBJECT_STORAGE_ENABLED: true
  OBJECT_STORAGE_TYPE: azure
  EXPOSED_WORKFLOWS: "<single-workflow-name>"
```

`EXPOSED_WORKFLOWS` is comma-separated. Use exactly one workflow when Azure ML scoring is configured as `/v1/infer`.

## Useful Checks

Confirm workflow names:

```bash
rg -n 'name = "foundry_' serve/server/example_workflows
```

Confirm service routes:

```bash
rg -n 'v1/infer|liveness|readiness|health' earth2studio/serve/server/main.py
```

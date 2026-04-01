# Object Storage Support

This document describes how to configure and use object storage (AWS S3 with CloudFront or
Azure Blob Storage) for storing workflow results in the Earth2Studio Inference Server.

## Overview

By default, workflow results are stored locally on the inference server. When object storage is
enabled, results are automatically uploaded to your chosen cloud storage provider (AWS S3 or
Azure Blob Storage). For S3, CloudFront signed URLs can be generated; for Azure, the server
uploads with managed identity and does **not** issue read URLs—clients obtain tokens to read blobs.

- **Scalability**: Offload storage from the inference server
- **Performance**: Fast global access via CDN (CloudFront for S3) or direct Azure Blob Storage access
- **Security**: Time-limited CloudFront signed URLs (S3); Azure reads use your own token model
- **Seamless Client Experience**: The Python client SDK automatically handles S3; Azure may require
  client-side auth for reads

## Storage Provider Options

The inference server supports two storage providers:

- **AWS S3**: With optional CloudFront CDN for enhanced performance
- **Azure Blob Storage**: Uploads via managed identity; clients obtain Azure AD (or other) tokens to
  read data

## AWS Prerequisites

Before enabling object storage, you need to set up the following AWS resources:

### 1. S3 Bucket

Create an S3 bucket to store workflow results.
**Must for performance**: Enable S3 Transfer Acceleration for faster uploads:

### 2. CloudFront Distribution

Create a CloudFront distribution to serve content from your S3 bucket.

### 3. CloudFront Key Pair for Signed URLs

To generate signed URLs, you need a CloudFront key pair.

### 4. IAM Credentials

Create IAM credentials with permissions to upload to S3.

## Azure Prerequisites

Before enabling Azure Blob Storage, you need to set up the following Azure resources:

### 1. Azure Storage Account

Create an Azure Storage Account.

### 2. Storage Container

Create a blob container in your storage account.

### 3. Managed identity (or equivalent) for uploads

The inference server writes to the container using **DefaultAzureCredential** (e.g. user-assigned
or system-assigned managed identity). Grant that identity **Storage Blob Data Contributor** (or
equivalent) on the storage account or container.

### 4. Client read access

Downstream clients that read blobs should use **Azure AD** (or your chosen mechanism) to obtain
tokens; the server does not generate SAS or other signed read URLs for Azure.

## Server Configuration

### Environment Variables

Configure object storage using environment variables. Choose either AWS S3 or Azure Blob Storage:

#### AWS S3 Configuration

```bash
# Enable object storage
export OBJECT_STORAGE_ENABLED=true
export OBJECT_STORAGE_TYPE=s3

# S3 Configuration
export OBJECT_STORAGE_BUCKET=your-bucket-name
export OBJECT_STORAGE_REGION=us-east-1
export OBJECT_STORAGE_PREFIX=outputs  # Optional: prefix for uploaded files

# AWS Credentials (or use IAM roles/instance profiles)
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...  # Optional: for temporary credentials

# Optional: S3-compatible endpoint (for MinIO, etc.)
export OBJECT_STORAGE_ENDPOINT_URL=http://localhost:9000

# Transfer Configuration
export OBJECT_STORAGE_TRANSFER_ACCELERATION=true  # Enable S3 Transfer Acceleration
export OBJECT_STORAGE_MAX_CONCURRENCY=16          # Concurrent upload threads
export OBJECT_STORAGE_MULTIPART_CHUNKSIZE=8388608 # 8MB chunk size
export OBJECT_STORAGE_USE_RUST_CLIENT=true        # High-performance Rust client

# CloudFront Signed URL Configuration
export CLOUDFRONT_DOMAIN=https://d30anq61ot046p.cloudfront.net
export CLOUDFRONT_KEY_PAIR_ID=KUCQGLNFR6UH1
# PEM private key *content* (not a file path); use quoting / multiline env as
# supported by your shell
export CLOUDFRONT_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----..."
export SIGNED_URL_EXPIRES_IN=86400  # URL expiration in seconds (S3/CloudFront only)
```

#### Azure Blob Storage Configuration

```bash
# Enable object storage
export OBJECT_STORAGE_ENABLED=true
export OBJECT_STORAGE_TYPE=azure

# Azure Configuration
# Container name (used as bucket equivalent)
export OBJECT_STORAGE_BUCKET=your-container-name
export OBJECT_STORAGE_PREFIX=outputs  # Optional: prefix for uploaded files

# Azure: storage account (managed identity / DefaultAzureCredential for uploads)
export AZURE_STORAGE_ACCOUNT_NAME=mystorageaccount
# Blob service endpoint (optional). Either works; both set `endpoint_url` in config
# (see code order).
# export OBJECT_STORAGE_ENDPOINT_URL=https://mystorageaccount.blob.core.windows.net
# export AZURE_ENDPOINT_URL=https://mystorageaccount.blob.core.windows.net

# Optional: Container name (defaults to OBJECT_STORAGE_BUCKET if not set)
export AZURE_CONTAINER_NAME=workflow-results

# Optional: Planetary Computer Pro / GeoCatalog STAC ingestion after upload
# (queues `geocatalog_ingestion`)
# export AZURE_GEOCATALOG_URL=https://geocatalog.spatio.azure.com/

# Transfer Configuration
export OBJECT_STORAGE_MAX_CONCURRENCY=16          # Concurrent upload threads
export OBJECT_STORAGE_MULTIPART_CHUNKSIZE=8388608 # 8MB chunk size
export OBJECT_STORAGE_USE_RUST_CLIENT=true        # High-performance Rust client

# Blob reads: clients obtain Azure AD (or other) tokens independently;
# the server does not issue SAS/signed read URLs.
```

### YAML Configuration

Alternatively, configure via `config.yaml`:

#### AWS S3 YAML Configuration

```yaml
object_storage:
  enabled: true
  storage_type: s3
  bucket: your-bucket-name
  region: us-east-1
  prefix: outputs
  use_transfer_acceleration: true
  max_concurrency: 16
  multipart_chunksize: 8388608
  use_rust_client: true
  cloudfront_domain: https://d30anq61ot046p.cloudfront.net
  cloudfront_key_pair_id: KUCQGLNFR6UH1
  cloudfront_private_key: null  # PEM private key content
  signed_url_expires_in: 86400
```

#### Azure Blob Storage YAML Configuration

```yaml
object_storage:
  enabled: true
  storage_type: azure
  bucket: your-container-name  # Container name (used as bucket equivalent)
  prefix: outputs
  max_concurrency: 16
  multipart_chunksize: 8388608
  use_rust_client: true
  azure_account_name: mystorageaccount  # Required unless endpoint_url is set
  endpoint_url: null  # Optional: e.g. https://mystorageaccount.blob.core.windows.net
  azure_container_name: workflow-results  # Optional, defaults to bucket
  azure_geocatalog_url: null  # When set, run GeoCatalog ingestion after upload
```

### Configuration Parameters Reference

<!-- markdownlint-disable MD013 -->
| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `enabled` | `OBJECT_STORAGE_ENABLED` | `false` | Enable object storage |
| `storage_type` | `OBJECT_STORAGE_TYPE` | `s3` | Storage provider: `s3` or `azure` |
| `bucket` | `OBJECT_STORAGE_BUCKET` | `null` | S3 bucket name or Azure container name |
| `region` | `OBJECT_STORAGE_REGION` | `us-east-1` | AWS region (S3 only) |
| `prefix` | `OBJECT_STORAGE_PREFIX` | `outputs` | Remote prefix for files |
| `access_key_id` | `OBJECT_STORAGE_ACCESS_KEY_ID` | `null` | AWS access key ID (S3 only) |
| `secret_access_key` | `OBJECT_STORAGE_SECRET_ACCESS_KEY` | `null` | AWS secret access key (S3 only) |
| `session_token` | `OBJECT_STORAGE_SESSION_TOKEN` | `null` | AWS session token (S3 only) |
| `endpoint_url` | `OBJECT_STORAGE_ENDPOINT_URL` or `AZURE_ENDPOINT_URL` | `null` | Custom endpoint (S3-compatible; Azure blob URL). Both env vars map to `endpoint_url`; if both are set, `AZURE_ENDPOINT_URL` wins. |
| `use_transfer_acceleration` | `OBJECT_STORAGE_TRANSFER_ACCELERATION` | `true` | Enable S3 Transfer Acceleration (S3 only) |
| `max_concurrency` | `OBJECT_STORAGE_MAX_CONCURRENCY` | `16` | Max concurrent transfers |
| `multipart_chunksize` | `OBJECT_STORAGE_MULTIPART_CHUNKSIZE` | `8388608` | Multipart chunk size (bytes) |
| `use_rust_client` | `OBJECT_STORAGE_USE_RUST_CLIENT` | `true` | Use high-performance Rust client |
| `cloudfront_domain` | `CLOUDFRONT_DOMAIN` | `null` | CloudFront distribution domain (S3 only) |
| `cloudfront_key_pair_id` | `CLOUDFRONT_KEY_PAIR_ID` | `null` | CloudFront key pair ID (S3 only) |
| `cloudfront_private_key` | `CLOUDFRONT_PRIVATE_KEY` | `null` | PEM private key *content* for CloudFront signing (S3 only) |
| `azure_account_name` | `AZURE_STORAGE_ACCOUNT_NAME` | `null` | Azure storage account name (Azure only; uploads use DefaultAzureCredential) |
| `azure_container_name` | `AZURE_CONTAINER_NAME` | `null` | Azure container name (Azure only, defaults to bucket) |
| `azure_geocatalog_url` | `AZURE_GEOCATALOG_URL` | `null` | When set, enqueue GeoCatalog / Planetary Computer ingestion after Azure upload |
| `signed_url_expires_in` | `SIGNED_URL_EXPIRES_IN` | `86400` | CloudFront signed URL TTL (seconds; S3 only) |
<!-- markdownlint-enable MD013 -->

## Result Metadata

When object storage is enabled, the workflow result metadata includes additional fields:

### AWS S3 Example

```json
{
  "request_id": "exec_1769560728_10ed9d3c",
  "status": "completed",
  "storage_type": "s3",
  "signed_url":
    "https://d30anq61ot046p.cloudfront.net/outputs/exec_1769560728_10ed9d3c/\
*?Policy=...&Signature=...&Key-Pair-Id=...",
  "remote_path": "outputs/exec_1769560728_10ed9d3c",
  "output_files": [
    {"path": "exec_1769560728_10ed9d3c/results.zarr/.zarray", "size": 123},
    {"path": "exec_1769560728_10ed9d3c/results.zarr/t2m/0.0.0", "size": 4567890}
  ]
}
```

### Azure Blob Storage Example

```json
{
  "request_id": "exec_1769560728_10ed9d3c",
  "status": "completed",
  "storage_type": "azure",
  "remote_path": "azure://workflow-results/outputs/exec_1769560728_10ed9d3c",
  "output_files": [
    {"path": "exec_1769560728_10ed9d3c/results.zarr/.zarray", "size": 123},
    {"path": "exec_1769560728_10ed9d3c/results.zarr/t2m/0.0.0", "size": 4567890}
  ]
}
```

There is no `signed_url` for Azure: use `remote_path` (and optional `blob_url` for GeoCatalog-related
flows) and authorize reads with your own Azure token flow.

### Storage Type Values

| Value | Description |
|-------|-------------|
| `server` | Results stored locally on the inference server |
| `s3` | Results stored in S3, accessible via CloudFront signed URL |
| `azure` | Results stored in Azure Blob Storage; reads use client-issued tokens (no server SAS) |

## Client Usage

### Seamless Access (Recommended)

The Python client SDK handles storage type automatically:

```python
from api_client.e2client import RemoteEarth2Workflow

workflow = RemoteEarth2Workflow(api_url, workflow_name="deterministic_earth2_workflow")

# Works the same regardless of storage type
result = workflow(start_time=[datetime(2025, 8, 21, 6)])
ds = result.as_dataset()  # Automatically fetches from S3 if configured
```

### Direct File Download

The `Earth2StudioClient.download_result()` method handles both storage types:

```python
from api_client.client import Earth2StudioClient, InferenceRequest

client = Earth2StudioClient(api_url, workflow_name="deterministic_earth2_workflow")
request_result = client.run_inference_sync(
    InferenceRequest(parameters={"start_time": [datetime(2025, 8, 21, 6)]})
)

# Automatically downloads from S3 or Azure if storage_type is "s3" or "azure"
for file in request_result.output_files[:5]:
    content = client.download_result(request_result, file.path)
    print(f"Downloaded {file.path}: {len(content.getvalue())} bytes")
```

### Using Signed URLs Directly

For advanced use cases, you can use the signed URL directly (S3/CloudFront only).

#### Using CloudFront Signed URLs

```python
import requests

# Get the signed URL from the result
signed_url = request_result.signed_url
# Example:
# https://d30anq61ot046p.cloudfront.net/outputs/exec_123/*?Policy=...&Signature=...&Key-Pair-Id=...

# The signed URL uses a wildcard (*) - construct the actual file URL
base_url = signed_url.split("?")[0].rstrip("/*")
query_params = signed_url.split("?")[1]

# Download a specific file
file_path = "results.zarr/.zarray"
file_url = f"{base_url}/{file_path}?{query_params}"
response = requests.get(file_url)
```

#### Azure blob reads

Use the blob URL from result metadata (or construct it from account, container, and path) and
request an OAuth token from Azure AD with scope for Storage (e.g. `https://storage.azure.com/.default`),
then `GET` the blob with `Authorization: Bearer <token>`. The inference server does not return a
pre-signed Azure URL.

### Using with Xarray and Zarr

The client provides an fsspec mapper for opening Zarr stores directly:

```python
import xarray as xr
from api_client.object_storage import create_cloudfront_mapper

# Create a mapper from the signed URL
mapper = create_cloudfront_mapper(request_result.signed_url, zarr_path="results.zarr")

# Open as xarray Dataset (lazy loading)
ds = xr.open_zarr(mapper, consolidated=True)
print(ds)
```

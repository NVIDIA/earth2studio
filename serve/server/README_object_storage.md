# Object Storage Support

This document describes how to configure and use object storage (AWS S3 with CloudFront) for storing
workflow results in the Earth2Studio Inference Server.

## Overview

By default, workflow results are stored locally on the inference server. When object storage is
enabled, results are automatically uploaded to S3 and served via CloudFront signed URLs. This
provides:

- **Scalability**: Offload storage from the inference server
- **Performance**: CloudFront CDN for fast global access
- **Security**: Time-limited signed URLs for secure access
- **Seamless Client Experience**: The Python client SDK automatically handles both storage types

## AWS Prerequisites

Before enabling object storage, you need to set up the following AWS resources:

### 1. S3 Bucket

Create an S3 bucket to store workflow results:

```bash
aws s3 mb s3://your-bucket-name --region us-east-1
```

**Must for performance**: Enable S3 Transfer Acceleration for faster uploads:

```bash
aws s3api put-bucket-accelerate-configuration \
    --bucket your-bucket-name \
    --accelerate-configuration Status=Enabled
```

### 2. CloudFront Distribution

Create a CloudFront distribution to serve content from your S3 bucket:

1. Go to AWS CloudFront Console → Create Distribution
2. Set Origin Domain to your S3 bucket (`your-bucket-name.s3.amazonaws.com`)
3. Set Origin Access to "Origin access control settings (recommended)"
4. Create a new Origin Access Control (OAC)
5. Update the S3 bucket policy to allow CloudFront access (AWS will provide the policy)

### 3. CloudFront Key Pair for Signed URLs

To generate signed URLs, you need a CloudFront key pair:

1. Go to AWS CloudFront Console → Key Management → Public Keys
2. Create a new public key by uploading a public key you generated:

```bash
# Generate a private key
openssl genrsa -out cloudfront-private-key.pem 2048

# Extract the public key
openssl rsa -in cloudfront-private-key.pem -pubout -out cloudfront-public-key.pem
```

Then:

1. Upload `cloudfront-public-key.pem` to CloudFront
2. Create a Key Group containing your public key
3. Associate the Key Group with your CloudFront distribution's behavior settings (Restrict Viewer
Access → Yes, Trusted Key Groups)
4. Note the **Key Pair ID** (e.g., `KUCQGLNFR6UH1`)
5. Keep `cloudfront-private-key.pem` secure - this is used by the server to sign URLs

### 4. IAM Credentials

Create IAM credentials with permissions to upload to S3. See [Creating IAM
users](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html) for
detailed instructions. The credentials need `s3:PutObject`, `s3:GetObject`,
`s3:DeleteObject`, and `s3:ListBucket` permissions on your bucket.

## Server Configuration

### Environment Variables

Configure object storage using environment variables:

```bash
# Enable object storage
export OBJECT_STORAGE_ENABLED=true

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
export CLOUDFRONT_PRIVATE_KEY_PATH=/path/to/cloudfront-private-key.pem
export OBJECT_STORAGE_SIGNED_URL_EXPIRES_IN=3600  # URL expiration in seconds
```

### YAML Configuration

Alternatively, configure via `config.yaml`:

```yaml
object_storage:
  enabled: true
  bucket: your-bucket-name
  region: us-east-1
  prefix: outputs
  use_transfer_acceleration: true
  max_concurrency: 16
  multipart_chunksize: 8388608
  use_rust_client: true
  cloudfront_domain: https://d30anq61ot046p.cloudfront.net
  cloudfront_key_pair_id: KUCQGLNFR6UH1
  cloudfront_private_key_path: /path/to/cloudfront-private-key.pem
  signed_url_expires_in: 3600
```

### Configuration Parameters Reference

<!-- markdownlint-disable MD013 -->
| Parameter | Environment Variable | Default | Description |
|-----------|---------------------|---------|-------------|
| `enabled` | `OBJECT_STORAGE_ENABLED` | `false` | Enable object storage |
| `bucket` | `OBJECT_STORAGE_BUCKET` | `null` | S3 bucket name |
| `region` | `OBJECT_STORAGE_REGION` | `us-east-1` | AWS region |
| `prefix` | `OBJECT_STORAGE_PREFIX` | `outputs` | Remote prefix for files |
| `access_key_id` | `OBJECT_STORAGE_ACCESS_KEY_ID` | `null` | AWS access key ID |
| `secret_access_key` | `OBJECT_STORAGE_SECRET_ACCESS_KEY` | `null` | AWS secret access key |
| `session_token` | `OBJECT_STORAGE_SESSION_TOKEN` | `null` | AWS session token |
| `endpoint_url` | `OBJECT_STORAGE_ENDPOINT_URL` | `null` | Custom endpoint (S3-compatible) |
| `use_transfer_acceleration` | `OBJECT_STORAGE_TRANSFER_ACCELERATION` | `true` | Enable S3 Transfer Acceleration |
| `max_concurrency` | `OBJECT_STORAGE_MAX_CONCURRENCY` | `16` | Max concurrent transfers |
| `multipart_chunksize` | `OBJECT_STORAGE_MULTIPART_CHUNKSIZE` | `8388608` | Multipart chunk size (bytes) |
| `use_rust_client` | `OBJECT_STORAGE_USE_RUST_CLIENT` | `true` | Use high-performance Rust client |
| `cloudfront_domain` | `CLOUDFRONT_DOMAIN` | `null` | CloudFront distribution domain |
| `cloudfront_key_pair_id` | `CLOUDFRONT_KEY_PAIR_ID` | `null` | CloudFront key pair ID |
| `cloudfront_private_key_path` | `CLOUDFRONT_PRIVATE_KEY_PATH` | `null` | Path to private key PEM file |
| `signed_url_expires_in` | `OBJECT_STORAGE_SIGNED_URL_EXPIRES_IN` | `3600` | Signed URL expiration (seconds) |
<!-- markdownlint-enable MD013 -->

## Result Metadata

When object storage is enabled, the workflow result metadata includes additional fields:

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

### Storage Type Values

| Value | Description |
|-------|-------------|
| `server` | Results stored locally on the inference server |
| `s3` | Results stored in S3, accessible via CloudFront signed URL |

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

# Automatically downloads from S3 if storage_type is "s3"
for file in request_result.output_files[:5]:
    content = client.download_result(request_result, file.path)
    print(f"Downloaded {file.path}: {len(content.getvalue())} bytes")
```

### Using Signed URLs Directly

For advanced use cases, you can use the signed URL directly:

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

## Signed URL Format

CloudFront signed URLs contain three query parameters:

```text
https://d30anq61ot046p.cloudfront.net/outputs/exec_123/*?Policy=eyJTdGF0ZW1lbnQiOl...\
&Signature=ABC123...&Key-Pair-Id=KUCQGLNFR6UH1
```

| Parameter | Description |
|-----------|-------------|
| `Policy` | Base64-encoded JSON policy specifying resource and expiration |
| `Signature` | RSA signature of the policy using the private key |
| `Key-Pair-Id` | CloudFront key pair ID used to verify the signature |

The wildcard (`*`) in the URL path allows access to all files under that prefix.

## Testing

Run object storage integration tests:

```bash
# Set required environment variables
export TEST_S3_BUCKET=my-test-bucket
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...

# Run S3 upload tests
pytest test/integration/test_object_storage.py -v

# Run CloudFront signed URL tests (requires additional config)
export TEST_CLOUDFRONT_DOMAIN=https://d30anq61ot046p.cloudfront.net
export TEST_CLOUDFRONT_KEY_PAIR_ID=KUCQGLNFR6UH1
export TEST_CLOUDFRONT_PRIVATE_KEY_PATH=/path/to/private.pem
pytest test/integration/test_object_storage.py::TestCloudFrontSignedUrl -v
```

## Troubleshooting

### Common Issues

1. **403 Forbidden from CloudFront**
   - Verify the S3 bucket policy allows CloudFront OAC access
   - Check that the CloudFront distribution is configured with the correct origin
   - Ensure the key pair is in a Key Group associated with the distribution

2. **Signed URL expired**
   - Increase `signed_url_expires_in` configuration
   - Request fresh results from the API (URLs are regenerated)

3. **Upload failures**
   - Verify IAM credentials have `s3:PutObject` permission
   - Check bucket name and region are correct
   - If using Transfer Acceleration, ensure it's enabled on the bucket

4. **Slow uploads**
   - Enable `use_rust_client` for better performance
   - Increase `max_concurrency` for more parallel uploads
   - Enable `use_transfer_acceleration` if uploading from distant regions

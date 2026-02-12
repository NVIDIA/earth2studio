# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Object Storage Module

This module provides object storage functionality using NVIDIA Multi-Storage Client (MSC)
with Rust backend for high-performance parallel file transfers.

Example:
    >>> from api_server.object_storage import MSCObjectStorage
    >>> storage = MSCObjectStorage(bucket="my-bucket", region="us-east-1")
    >>> storage.upload_directory("/path/to/local/dir", "remote/prefix")
"""

import base64
import datetime
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class UploadResult:
    """Result of an upload operation"""

    success: bool
    files_uploaded: int
    total_bytes: int
    destination: str
    errors: list[str]

    def __repr__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"UploadResult({status}, files={self.files_uploaded}, "
            f"bytes={self.total_bytes}, dest={self.destination})"
        )


class ObjectStorage(ABC):
    """
    Abstract base class for object storage operations.

    This class defines the interface that all object storage implementations must follow.
    Subclasses should implement the abstract methods to provide functionality for
    specific cloud providers (AWS S3, GCS, Azure Blob Storage, etc.).
    """

    @abstractmethod
    def upload_directory(
        self,
        local_directory: str | Path,
        remote_prefix: str,
        recursive: bool = True,
        overwrite: bool = True,
    ) -> UploadResult:
        """
        Upload a local directory to the object store at the specified path.

        Args:
            local_directory: Path to the local directory to upload.
            remote_prefix: The prefix/path in the bucket where files will be uploaded.
                          Files will maintain their relative paths under this prefix.
            recursive: If True, recursively upload all subdirectories.
            overwrite: If True, overwrite existing files in the destination.

        Returns:
            UploadResult containing details about the upload operation.

        Raises:
            FileNotFoundError: If the local directory does not exist.
            PermissionError: If there are permission issues with the bucket.
            ObjectStorageError: For other storage-related errors.
        """
        pass

    @abstractmethod
    def upload_file(
        self,
        local_file: str | Path,
        remote_key: str,
        overwrite: bool = True,
    ) -> bool:
        """
        Upload a single file to the object store.

        Args:
            local_file: Path to the local file to upload.
            remote_key: The full key/path in the bucket for the file.
            overwrite: If True, overwrite if the file already exists.

        Returns:
            True if upload was successful, False otherwise.
        """
        pass

    @abstractmethod
    def file_exists(self, remote_key: str) -> bool:
        """
        Check if a file exists in the object store.

        Args:
            remote_key: The full key/path in the bucket.

        Returns:
            True if the file exists, False otherwise.
        """
        pass

    @abstractmethod
    def delete_file(self, remote_key: str) -> bool:
        """
        Delete a file from the object store.

        Args:
            remote_key: The full key/path in the bucket.

        Returns:
            True if deletion was successful, False otherwise.
        """
        pass

    @abstractmethod
    def generate_signed_url(self, remote_key: str, expires_in: int = 86400) -> str:
        """
        Generate a signed URL for a file in the object store.

        Args:
            remote_key: The full key/path in the bucket.
            expires_in: The number of seconds the URL will be valid for.
        """
        pass


class ObjectStorageError(Exception):
    """Base exception for object storage errors"""

    pass


class MSCObjectStorage(ObjectStorage):
    """
    Object storage implementation using NVIDIA Multi-Storage Client with Rust backend.

    MSC provides optimized parallel transfers and the Rust client bypasses Python's
    GIL for significantly improved I/O performance (up to 12x faster).

    Uses sync_from for efficient directory uploads with parallel transfers.

    Credentials are picked up from environment variables:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_SESSION_TOKEN (optional)
    - AWS_DEFAULT_REGION

    See: https://nvidia.github.io/multi-storage-client/user_guide/rust.html
    """

    def __init__(
        self,
        bucket: str,
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        endpoint_url: str | None = None,
        use_transfer_acceleration: bool = False,
        max_concurrency: int = 16,
        multipart_chunksize: int = 8 * 1024 * 1024,  # 8 MB
        use_rust_client: bool = False,
        profile_name: str = "e2studio-s3",
        cloudfront_domain: str | None = None,
        cloudfront_key_pair_id: str | None = None,
        cloudfront_private_key: str | None = None,
    ):
        """
        Initialize MSCObjectStorage with AWS credentials and configuration.

        Args:
            bucket: The S3 bucket name.
            region: AWS region (e.g., 'us-east-1').
            access_key_id: AWS access key ID (sets AWS_ACCESS_KEY_ID env var).
            secret_access_key: AWS secret access key (sets AWS_SECRET_ACCESS_KEY env var).
            session_token: AWS session token for temporary credentials.
            endpoint_url: Custom endpoint URL (for S3-compatible services).
            use_transfer_acceleration: Enable S3 Transfer Acceleration (bucket must have it enabled).
            max_concurrency: Maximum number of concurrent transfers (default: 16).
            multipart_chunksize: Chunk size for multipart uploads in bytes (default: 8MB).
            use_rust_client: Enable the high-performance Rust client (default: True).
            profile_name: Name for the MSC profile (default: 'e2studio-s3').
            cloudfront_domain: CloudFront distribution domain for signed URLs.
            cloudfront_key_pair_id: CloudFront key pair ID for signed URLs.
            cloudfront_private_key: PEM private key content as string.
        """
        self.bucket = bucket
        self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.max_concurrency = max_concurrency
        self.multipart_chunksize = multipart_chunksize
        self.use_rust_client = use_rust_client
        self.profile_name = profile_name
        self.use_transfer_acceleration = use_transfer_acceleration

        # Use S3 Transfer Acceleration endpoint if enabled (and no custom endpoint provided)
        if use_transfer_acceleration and not endpoint_url:
            self.endpoint_url = f"https://{bucket}.s3-accelerate.amazonaws.com"
            logger.info(f"S3 Transfer Acceleration enabled: {self.endpoint_url}")
        else:
            self.endpoint_url = endpoint_url or ""

        # CloudFront configuration for signed URLs
        self.cloudfront_domain = cloudfront_domain
        self.cloudfront_key_pair_id = cloudfront_key_pair_id
        self.cloudfront_private_key = cloudfront_private_key

        # Set credentials as environment variables - MSC picks these up automatically
        if access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = access_key_id
        if secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = secret_access_key
        if session_token:
            os.environ["AWS_SESSION_TOKEN"] = session_token
        if region:
            os.environ["AWS_DEFAULT_REGION"] = region

        # Import multi-storage-client
        try:
            import multistorageclient as msc
        except ImportError as e:
            raise ImportError(
                "multi-storage-client is required for MSCObjectStorage. "
                "Install with: pip install multi-storage-client"
            ) from e

        self._msc = msc

        # Build the S3 storage provider options
        s3_storage_provider_options: dict[str, Any] = {
            "base_path": bucket,
            "region_name": self.region,
            "multipart_threshold": multipart_chunksize,
            "multipart_chunksize": multipart_chunksize,
            "max_concurrency": max_concurrency,
        }

        # Add endpoint URL if provided (for S3-compatible services)
        if endpoint_url:
            s3_storage_provider_options["endpoint_url"] = endpoint_url

        # Enable Rust client for high-performance I/O
        if use_rust_client:
            s3_storage_provider_options["rust_client"] = {
                "multipart_chunksize": multipart_chunksize,
                "max_concurrency": max_concurrency,
            }

        # Build the S3 profile config
        s3_profile_config = {
            "profiles": {
                profile_name: {
                    "storage_provider": {
                        "type": "s3",
                        "options": s3_storage_provider_options,
                    }
                }
            }
        }

        # Initialize the S3 StorageClient (target for uploads)
        s3_client_config = msc.StorageClientConfig.from_dict(
            config_dict=s3_profile_config,
            profile=profile_name,
        )
        self._s3_client = msc.StorageClient(config=s3_client_config)

        # Initialize the local filesystem StorageClient (source for uploads)
        local_profile_config = {
            "profiles": {
                "local": {
                    "storage_provider": {
                        "type": "file",
                        "options": {
                            "base_path": "/",
                        },
                    }
                }
            }
        }
        local_client_config = msc.StorageClientConfig.from_dict(
            config_dict=local_profile_config,
            profile="local",
        )
        self._local_client = msc.StorageClient(config=local_client_config)

        rust_status = "enabled" if use_rust_client else "disabled"
        accel_status = "enabled" if use_transfer_acceleration else "disabled"
        logger.info(
            f"MSCObjectStorage initialized: bucket={bucket}, region={self.region}, "
            f"max_concurrency={max_concurrency}, rust_client={rust_status}, "
            f"transfer_acceleration={accel_status}"
        )

    def upload_directory(
        self,
        local_directory: str | Path,
        remote_prefix: str,
        recursive: bool = True,
        overwrite: bool = True,
    ) -> UploadResult:
        """
        Upload a local directory to S3 using Multi-Storage Client sync_from.

        Uses MSC's sync_from for efficient parallel directory uploads.

        Args:
            local_directory: Path to the local directory to upload.
            remote_prefix: The S3 prefix where files will be uploaded.
            recursive: If True, recursively upload all subdirectories.
            overwrite: If True, overwrite existing files.

        Returns:
            UploadResult with details about the upload.
        """
        local_path = Path(local_directory)

        if not local_path.exists():
            raise FileNotFoundError(f"Directory not found: {local_directory}")
        if not local_path.is_dir():
            raise ValueError(f"Path is not a directory: {local_directory}")

        # Normalize paths
        remote_prefix = remote_prefix.strip("/")
        source_path = str(local_path.absolute())

        # Count files for reporting
        if recursive:
            files = [f for f in local_path.rglob("*") if f.is_file()]
        else:
            files = [f for f in local_path.iterdir() if f.is_file()]

        total_bytes = sum(f.stat().st_size for f in files)

        logger.info(
            f"[MSC] Syncing {len(files)} files ({total_bytes / (1024 * 1024):.2f} MB) "
            f"from {local_directory} to s3://{self.bucket}/{remote_prefix}"
        )

        errors: list[str] = []
        start_time = time.time()

        try:
            # Use sync_from for efficient parallel directory upload
            result = self._s3_client.sync_from(
                source_client=self._local_client,
                source_path=source_path,
                target_path=f"/{remote_prefix}" if remote_prefix else "/",
            )

            logger.info(f"[MSC] Sync completed: {result}")
            files_uploaded = len(files)

        except Exception as e:
            error_msg = f"MSC sync_from failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            files_uploaded = 0

        elapsed_time = time.time() - start_time
        success = len(errors) == 0
        destination = f"s3://{self.bucket}/{remote_prefix}"

        result = UploadResult(
            success=success,
            files_uploaded=files_uploaded,
            total_bytes=total_bytes,
            destination=destination,
            errors=errors,
        )

        if success:
            throughput_mbps = (
                (total_bytes / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0
            )
            logger.info(
                f"[MSC] Directory upload completed: {result} "
                f"(elapsed: {elapsed_time:.2f}s, throughput: {throughput_mbps:.2f} MB/s)"
            )
        else:
            logger.warning(f"[MSC] Directory upload completed with errors: {result}")

        return result

    def upload_file(
        self,
        local_file: str | Path,
        remote_key: str,
        overwrite: bool = True,
    ) -> bool:
        """
        Upload a single file to S3 using Multi-Storage Client with Rust backend.

        Args:
            local_file: Path to the local file.
            remote_key: The S3 key for the file.
            overwrite: If True, overwrite if exists.

        Returns:
            True if successful, False otherwise.
        """
        local_path = Path(local_file)

        if not local_path.exists():
            logger.error(f"File not found: {local_file}")
            return False
        if not local_path.is_file():
            logger.error(f"Path is not a file: {local_file}")
            return False

        try:
            remote_key = f"/{remote_key.lstrip('/')}"
            self._s3_client.upload_file(remote_key, str(local_path))
            return True

        except Exception as e:
            logger.error(f"[MSC] Failed to upload {local_file} to {remote_key}: {e}")
            return False

    def file_exists(self, remote_key: str) -> bool:
        """
        Check if a file exists in S3 using MSC info() method.

        Args:
            remote_key: The S3 key to check.

        Returns:
            True if the file exists, False otherwise.
        """
        try:
            remote_path = f"/{remote_key.lstrip('/')}"
            self._s3_client.info(remote_path)
            return True
        except FileNotFoundError:
            return False

    def delete_file(self, remote_key: str) -> bool:
        """
        Delete a file from S3 using MSC delete() method.

        Args:
            remote_key: The S3 key to delete.

        Returns:
            True if successful, False otherwise.
        """
        try:
            remote_path = f"/{remote_key.lstrip('/')}"
            self._s3_client.delete(remote_path)
            return True
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {remote_key}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete {remote_key}: {e}")
            return False

    def _rsa_signer(self, message: bytes) -> bytes:
        """Sign a message using the CloudFront private key."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        # Load key from content string
        if not self.cloudfront_private_key:
            raise ObjectStorageError(
                "No CloudFront private key configured. "
                "Please provide cloudfront_private_key."
            )
        key_data = self.cloudfront_private_key.encode("utf-8")

        private_key = load_pem_private_key(key_data, password=None)
        # SHA-1 is required by CloudFront - cannot use SHA-256
        return private_key.sign(
            message, padding.PKCS1v15(), hashes.SHA1()  # noqa: S303
        )

    @staticmethod
    def _url_safe_b64(data: bytes) -> str:
        """Convert bytes to URL-safe base64."""
        return (
            base64.b64encode(data)
            .decode("utf-8")
            .replace("+", "-")
            .replace("=", "_")
            .replace("/", "~")
        )

    def generate_signed_url(self, remote_key: str, expires_in: int = 86400) -> str:
        """
        Generate a CloudFront signed URL for accessing a file.

        Args:
            remote_key: The S3 key/path to the file. Can include wildcards.
            expires_in: Number of seconds until the URL expires (default: 86400).

        Returns:
            A signed CloudFront URL string.

        Raises:
            ObjectStorageError: If CloudFront configuration is missing.
        """
        if not all(
            [
                self.cloudfront_domain,
                self.cloudfront_key_pair_id,
                self.cloudfront_private_key,
            ]
        ):
            raise ObjectStorageError(
                "CloudFront configuration is required for signed URLs. "
                "Please provide cloudfront_domain, cloudfront_key_pair_id, and "
                "cloudfront_private_key."
            )

        path = remote_key if remote_key.startswith("/") else f"/{remote_key}"
        expire_time = int(
            (
                datetime.datetime.now(datetime.timezone.utc)
                + datetime.timedelta(seconds=expires_in)
            ).timestamp()
        )

        resource = f"{self.cloudfront_domain}{path}"

        policy = f"""
{{
  "Statement": [
    {{
      "Resource": "{resource}",
      "Condition": {{
        "DateLessThan": {{
          "AWS:EpochTime": {expire_time}
        }}
      }}
    }}
  ]
}}
""".strip().encode(
            "utf-8"
        )

        signature = self._rsa_signer(policy)

        signed_url = (
            f"{resource}"
            f"?Policy={self._url_safe_b64(policy)}"
            f"&Signature={self._url_safe_b64(signature)}"
            f"&Key-Pair-Id={self.cloudfront_key_pair_id}"
        )

        logger.debug(f"Generated signed URL for {remote_key}, expires in {expires_in}s")
        return signed_url


def get_object_storage(
    provider: Literal["msc"] = "msc",
    **kwargs: Any,
) -> ObjectStorage:
    """
    Factory function to get an object storage instance.

    This function provides a convenient way to instantiate the MSCObjectStorage
    implementation which uses NVIDIA Multi-Storage Client with Rust backend.

    Args:
        provider: The storage provider to use. Currently only "msc" is supported.
        **kwargs: Provider-specific configuration arguments.

    Returns:
        An ObjectStorage instance.

    Raises:
        ValueError: If an unsupported provider is specified.

    Example:
        >>> storage = get_object_storage(
        ...     "msc",
        ...     bucket="my-bucket",
        ...     region="us-east-1",
        ...     use_rust_client=True
        ... )
    """
    providers = {
        "msc": MSCObjectStorage,
    }

    if provider not in providers:
        raise ValueError(
            f"Unsupported storage provider: {provider}. "
            f"Supported providers: {list(providers.keys())}"
        )

    return providers[provider](**kwargs)

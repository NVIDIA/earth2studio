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
    """
    Result of an upload operation.

    Attributes
    ----------
    success : bool
        Whether the upload completed without errors.
    files_uploaded : int
        Number of files uploaded.
    total_bytes : int
        Total bytes uploaded.
    destination : str
        Destination path or URI.
    errors : list of str
        List of error messages, if any.
    """

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

    Defines the interface that all object storage implementations must follow.
    Subclasses implement the abstract methods for specific cloud providers
    (AWS S3, GCS, Azure Blob Storage, etc.).
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

        Parameters
        ----------
        local_directory : str or Path
            Path to the local directory to upload.
        remote_prefix : str
            Prefix/path in the bucket where files will be uploaded.
            Files maintain their relative paths under this prefix.
        recursive : bool, optional
            If True, recursively upload all subdirectories. Default is True.
        overwrite : bool, optional
            If True, overwrite existing files in the destination. Default is True.

        Returns
        -------
        UploadResult
            Details about the upload operation.

        Raises
        ------
        FileNotFoundError
            If the local directory does not exist.
        PermissionError
            If there are permission issues with the bucket.
        ObjectStorageError
            For other storage-related errors.
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

        Parameters
        ----------
        local_file : str or Path
            Path to the local file to upload.
        remote_key : str
            Full key/path in the bucket for the file.
        overwrite : bool, optional
            If True, overwrite if the file already exists. Default is True.

        Returns
        -------
        bool
            True if upload was successful, False otherwise.
        """
        pass

    @abstractmethod
    def file_exists(self, remote_key: str) -> bool:
        """
        Check if a file exists in the object store.

        Parameters
        ----------
        remote_key : str
            Full key/path in the bucket.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        pass

    @abstractmethod
    def delete_file(self, remote_key: str) -> bool:
        """
        Delete a file from the object store.

        Parameters
        ----------
        remote_key : str
            Full key/path in the bucket.

        Returns
        -------
        bool
            True if deletion was successful, False otherwise.
        """
        pass

    @abstractmethod
    def generate_signed_url(self, remote_key: str, expires_in: int = 86400) -> str:
        """
        Generate a signed URL for a file in the object store.

        Parameters
        ----------
        remote_key : str
            Full key/path in the bucket.
        expires_in : int, optional
            Number of seconds the URL will be valid for. Default is 86400.

        Returns
        -------
        str
            Signed URL string.
        """
        pass


class ObjectStorageError(Exception):
    """
    Base exception for object storage errors.
    """

    pass


class MSCObjectStorage(ObjectStorage):
    """
    Object storage using NVIDIA Multi-Storage Client (MSC) with Rust backend for AWS S3 and Azure Blob Storage.

    MSC provides optimized parallel transfers; the Rust client bypasses Python's
    GIL for improved I/O performance (up to 12x faster). Uses sync_from for
    efficient directory uploads with parallel transfers.

    Supports both AWS S3 and Azure Blob Storage.

    For S3, credentials are read from environment variables: AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN (optional), AWS_DEFAULT_REGION.

    For Azure, credentials can be provided via:
    - Connection string (AzureCredentials): Provide azure_connection_string
    - Managed Identity (DefaultAzureCredentials): Omit azure_connection_string and provide endpoint_url or azure_account_name

    References
    ----------
    https://nvidia.github.io/multi-storage-client/user_guide/rust.html

    Parameters
    ----------
    bucket : str
        S3 bucket name or Azure container name.
    storage_type : str, optional
        Storage provider type, either "s3" or "azure". Default is "s3".
    region : str, optional
        AWS region (e.g. 'us-east-1').
    access_key_id : str, optional
        AWS access key ID (sets AWS_ACCESS_KEY_ID env var).
    secret_access_key : str, optional
        AWS secret access key (sets AWS_SECRET_ACCESS_KEY env var).
    session_token : str, optional
        AWS session token for temporary credentials.
    endpoint_url : str, optional
        Custom endpoint URL for S3-compatible services or Azure Blob Storage.
    use_transfer_acceleration : bool, optional
        Enable S3 Transfer Acceleration (bucket must support it). Default is False.
    max_concurrency : int, optional
        Maximum number of concurrent transfers. Default is 16.
    multipart_chunksize : int, optional
        Chunk size for multipart uploads in bytes. Default is 8 MB.
    use_rust_client : bool, optional
        Use the high-performance Rust client. Default is False.
    profile_name : str, optional
        Name for the MSC profile. Default is 'e2studio-s3' for S3, 'e2studio-azure' for Azure.
    cloudfront_domain : str, optional
        CloudFront distribution domain for signed URLs.
    cloudfront_key_pair_id : str, optional
        CloudFront key pair ID for signed URLs.
    cloudfront_private_key : str, optional
        PEM private key content as string for signed URLs.
    azure_connection_string : str, optional
        Azure connection string (optional if using managed identity).
    azure_account_name : str, optional
        Azure storage account name (required if using managed identity without endpoint_url).
    azure_account_key : str, optional
        Azure storage account key (optional, for SAS token generation).
    azure_container_name : str, optional
        Azure container name.
    """

    def __init__(
        self,
        bucket: str,
        storage_type: Literal["s3", "azure"] = "s3",
        region: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        endpoint_url: str | None = None,
        use_transfer_acceleration: bool = False,
        max_concurrency: int = 16,
        multipart_chunksize: int = 8 * 1024 * 1024,  # 8 MB
        use_rust_client: bool = False,
        profile_name: str | None = None,
        cloudfront_domain: str | None = None,
        cloudfront_key_pair_id: str | None = None,
        cloudfront_private_key: str | None = None,
        # Azure-specific parameters
        azure_connection_string: str | None = None,
        azure_account_name: str | None = None,
        azure_account_key: str | None = None,
        azure_container_name: str | None = None,
    ):
        self.storage_type = storage_type
        self.bucket = bucket
        self.max_concurrency = max_concurrency
        self.multipart_chunksize = multipart_chunksize
        self.use_rust_client = use_rust_client
        self.profile_name = profile_name or (
            "e2studio-s3" if storage_type == "s3" else "e2studio-azure"
        )
        # Initialize endpoint_url as None to allow str | None type
        self.endpoint_url: str | None = None

        # S3-specific configuration
        if storage_type == "s3":
            self.region = region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            self.use_transfer_acceleration = use_transfer_acceleration

            # Use S3 Transfer Acceleration endpoint if enabled (and no custom endpoint provided)
            if use_transfer_acceleration and not endpoint_url:
                self.endpoint_url = f"https://{bucket}.s3-accelerate.amazonaws.com"
                logger.info(f"S3 Transfer Acceleration enabled: {self.endpoint_url}")
            else:
                self.endpoint_url = endpoint_url

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

        # Azure-specific configuration
        elif storage_type == "azure":
            self.azure_container_name = azure_container_name or bucket
            self.azure_account_key = azure_account_key

            # Determine if using managed identity (DefaultAzureCredentials)
            # Managed identity is used when connection string is not provided
            self.use_managed_identity = azure_connection_string is None

            if self.use_managed_identity:
                # Using managed identity - connection string not needed
                logger.info(
                    "Using Azure Managed Identity (DefaultAzureCredentials). "
                    f"Account: {azure_account_name or 'will be determined from endpoint'}, "
                    f"Container: {self.azure_container_name}"
                )
                self.azure_account_name = azure_account_name
            else:
                # Using connection string authentication — azure_connection_string is
                # guaranteed to be not None here (see use_managed_identity above).
                if azure_connection_string is None:
                    raise ObjectStorageError(
                        "Connection string required for non-managed identity mode"
                    )
                # Set Azure connection string
                os.environ["AZURE_CONNECTION_STRING"] = azure_connection_string

                # Extract account name from connection string if not provided directly
                if not azure_account_name:
                    self.azure_account_name = None
                    # Connection string format: DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...
                    for part in azure_connection_string.split(";"):
                        if part.startswith("AccountName="):
                            self.azure_account_name = part.split("=", 1)[1]
                            break
                    if not self.azure_account_name:
                        raise ObjectStorageError(
                            "Could not extract account name from connection string. "
                            "Please provide azure_account_name directly or ensure connection string contains AccountName."
                        )
                else:
                    self.azure_account_name = azure_account_name
        else:
            raise ValueError(
                f"Unsupported storage_type: {storage_type}. Must be 's3' or 'azure'."
            )

        # Import multi-storage-client
        try:
            import multistorageclient as msc
        except ImportError as e:
            raise ImportError(
                "multi-storage-client is required for MSCObjectStorage. "
                "Install with: pip install multi-storage-client"
            ) from e

        self._msc = msc

        # Build storage provider profile config based on storage_type
        if storage_type == "s3":
            # Build the S3 storage provider options
            s3_storage_provider_options: dict[str, Any] = {
                "base_path": bucket,
                "region_name": self.region,
                "multipart_threshold": multipart_chunksize,
                "multipart_chunksize": multipart_chunksize,
                "max_concurrency": max_concurrency,
            }

            # Add endpoint URL if provided (for S3-compatible services)
            if self.endpoint_url:
                s3_storage_provider_options["endpoint_url"] = self.endpoint_url

            # Enable Rust client for high-performance I/O
            if use_rust_client:
                s3_storage_provider_options["rust_client"] = {
                    "multipart_chunksize": multipart_chunksize,
                    "max_concurrency": max_concurrency,
                }

            # Build the S3 profile config
            profile_config = {
                "profiles": {
                    self.profile_name: {
                        "storage_provider": {
                            "type": "s3",
                            "options": s3_storage_provider_options,
                        }
                    }
                }
            }
        elif storage_type == "azure":
            # Build the Azure storage provider options
            # Derive endpoint URL from endpoint_url parameter, connection string, or account name
            azure_endpoint_url = None

            # First, check if endpoint_url was provided directly (for managed identity or custom endpoints)
            if endpoint_url:
                azure_endpoint_url = endpoint_url.rstrip("/")
            # Then, try to extract BlobEndpoint directly from connection string
            elif azure_connection_string:
                for part in azure_connection_string.split(";"):
                    if part.startswith("BlobEndpoint="):
                        azure_endpoint_url = part.split("=", 1)[1].rstrip("/")
                        break

            # If not found, construct from AccountName and EndpointSuffix
            if not azure_endpoint_url:
                account_name = None
                endpoint_suffix = "core.windows.net"  # Default suffix

                # Extract from connection string
                if azure_connection_string:
                    for part in azure_connection_string.split(";"):
                        if part.startswith("AccountName="):
                            account_name = part.split("=", 1)[1]
                        elif part.startswith("EndpointSuffix="):
                            endpoint_suffix = part.split("=", 1)[1]

                # Fall back to provided account_name if not in connection string
                if not account_name:
                    account_name = self.azure_account_name

                if not account_name:
                    raise ObjectStorageError(
                        "Azure endpoint_url cannot be determined. "
                        "Please provide endpoint_url, azure_connection_string (with AccountName or BlobEndpoint), "
                        "or azure_account_name."
                    )

                azure_endpoint_url = f"https://{account_name}.blob.{endpoint_suffix}"
                logger.info(
                    f"Constructed Azure endpoint URL from account name: {azure_endpoint_url}"
                )

            azure_storage_provider_options = {
                "base_path": self.azure_container_name,
                "endpoint_url": azure_endpoint_url,
            }

            # Build the Azure profile config with credentials provider
            profile_config = {
                "profiles": {
                    self.profile_name: {
                        "storage_provider": {
                            "type": "azure",
                            "options": azure_storage_provider_options,
                        }
                    }
                }
            }

            # Add Azure credentials provider
            if self.use_managed_identity:
                # Use DefaultAzureCredentials for managed identity
                profile_config["profiles"][self.profile_name][
                    "credentials_provider"
                ] = {
                    "type": "DefaultAzureCredentials",
                    "options": {},
                }
            elif azure_connection_string:
                # Use AzureCredentials with connection string
                profile_config["profiles"][self.profile_name][
                    "credentials_provider"
                ] = {
                    "type": "AzureCredentials",
                    "options": {"connection": "${AZURE_CONNECTION_STRING}"},
                }

        # Initialize the StorageClient (target for uploads)
        storage_client_config = msc.StorageClientConfig.from_dict(
            config_dict=profile_config,
            profile=self.profile_name,
        )
        self._storage_client = msc.StorageClient(config=storage_client_config)

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
        if storage_type == "s3":
            accel_status = "enabled" if use_transfer_acceleration else "disabled"
            logger.info(
                f"MSCObjectStorage initialized (S3): bucket={bucket}, region={self.region}, "
                f"max_concurrency={max_concurrency}, rust_client={rust_status}, "
                f"transfer_acceleration={accel_status}"
            )
        else:
            logger.info(
                f"MSCObjectStorage initialized (Azure): container={self.azure_container_name}, "
                f"max_concurrency={max_concurrency}, rust_client={rust_status}"
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

        Parameters
        ----------
        local_directory : str or Path
            Path to the local directory to upload.
        remote_prefix : str
            S3 prefix where files will be uploaded.
        recursive : bool, optional
            If True, recursively upload all subdirectories. Default is True.
        overwrite : bool, optional
            If True, overwrite existing files. Default is True.

        Returns
        -------
        UploadResult
            Details about the upload operation.
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

        storage_prefix = (
            f"s3://{self.bucket}"
            if self.storage_type == "s3"
            else f"azure://{self.azure_container_name if self.storage_type == 'azure' else self.bucket}"
        )
        logger.info(
            f"[MSC] Syncing {len(files)} files ({total_bytes / (1024 * 1024):.2f} MB) "
            f"from {local_directory} to {storage_prefix}/{remote_prefix}"
        )

        errors: list[str] = []
        start_time = time.time()

        try:
            # Use sync_from for efficient parallel directory upload
            result = self._storage_client.sync_from(
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
        if self.storage_type == "s3":
            destination = f"s3://{self.bucket}/{remote_prefix}"
        else:
            container = (
                self.azure_container_name
                if self.storage_type == "azure"
                else self.bucket
            )
            destination = f"azure://{container}/{remote_prefix}"

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

        Parameters
        ----------
        local_file : str or Path
            Path to the local file.
        remote_key : str
            S3 key for the file.
        overwrite : bool, optional
            If True, overwrite if exists. Default is True.

        Returns
        -------
        bool
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
            self._storage_client.upload_file(remote_key, str(local_path))
            return True

        except Exception as e:
            logger.error(f"[MSC] Failed to upload {local_file} to {remote_key}: {e}")
            return False

    def file_exists(self, remote_key: str) -> bool:
        """
        Check if a file exists in S3 using MSC info() method.

        Parameters
        ----------
        remote_key : str
            S3 key to check.

        Returns
        -------
        bool
            True if the file exists, False otherwise.
        """
        try:
            remote_path = f"/{remote_key.lstrip('/')}"
            self._storage_client.info(remote_path)
            return True
        except FileNotFoundError:
            return False

    def delete_file(self, remote_key: str) -> bool:
        """
        Delete a file from S3 using MSC delete() method.

        Parameters
        ----------
        remote_key : str
            S3 key to delete.

        Returns
        -------
        bool
            True if successful, False otherwise.
        """
        try:
            remote_path = f"/{remote_key.lstrip('/')}"
            self._storage_client.delete(remote_path)
            return True
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {remote_key}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete {remote_key}: {e}")
            return False

    def _rsa_signer(self, message: bytes) -> bytes:
        """
        Sign a message using the CloudFront private key.

        Parameters
        ----------
        message : bytes
            Message to sign.

        Returns
        -------
        bytes
            RSA signature.
        """
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
        """
        Convert bytes to URL-safe base64.

        Parameters
        ----------
        data : bytes
            Data to encode.

        Returns
        -------
        str
            URL-safe base64 string.
        """
        return (
            base64.b64encode(data)
            .decode("utf-8")
            .replace("+", "-")
            .replace("=", "_")
            .replace("/", "~")
        )

    def generate_signed_url(self, remote_key: str, expires_in: int = 86400) -> str:
        """
        Generate a signed URL for accessing a file.

        For S3, generates a CloudFront signed URL.
        For Azure, generates a SAS (Shared Access Signature) token URL.

        Parameters
        ----------
        remote_key : str
            Storage key/path to the file. Can include wildcards for S3.
        expires_in : int, optional
            Number of seconds until the URL expires. Default is 86400.

        Returns
        -------
        str
            Signed URL string.

        Raises
        ------
        ObjectStorageError
            If required configuration is missing.
        """
        if self.storage_type == "s3":
            return self._generate_cloudfront_signed_url(remote_key, expires_in)
        elif self.storage_type == "azure":
            return self._generate_azure_sas_url(remote_key, expires_in)
        else:
            raise ObjectStorageError(f"Unsupported storage_type: {self.storage_type}")

    def _generate_cloudfront_signed_url(self, remote_key: str, expires_in: int) -> str:
        """Generate a CloudFront signed URL for S3."""
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

        logger.debug(
            f"Generated CloudFront signed URL for {remote_key}, expires in {expires_in}s"
        )
        return signed_url

    def _generate_azure_sas_url(self, remote_key: str, expires_in: int) -> str:
        """Generate an Azure SAS (Shared Access Signature) URL."""
        if not self.azure_account_name or not self.azure_account_key:
            raise ObjectStorageError(
                "Azure account name and account key are required for signed URLs. "
                "Please provide azure_account_name and azure_account_key."
            )

        try:
            from azure.storage.blob import (
                ContainerSasPermissions,
                generate_container_sas,
            )
        except ImportError as e:
            raise ImportError(
                "azure-storage-blob is required for Azure signed URLs. "
                "Install with: pip install azure-storage-blob"
            ) from e

        container_name = (
            self.azure_container_name if self.storage_type == "azure" else self.bucket
        )

        # Define permissions (Read + List)
        permissions = ContainerSasPermissions(read=True, list=True)

        # Set the duration
        start_time = datetime.datetime.now(datetime.timezone.utc)
        expiry_time = start_time + datetime.timedelta(seconds=expires_in)

        # Generate the SAS token
        sas_token = generate_container_sas(
            account_name=self.azure_account_name,
            account_key=self.azure_account_key,
            container_name=container_name,
            permission=permissions,
            expiry=expiry_time,
            start=start_time,
        )

        # Construct the full URL with the prefix
        # Remove leading slash from remote_key if present
        prefix = remote_key.lstrip("/")
        base_url = f"https://{self.azure_account_name}.blob.core.windows.net/{container_name}/{prefix}"
        signed_url = f"{base_url}?{sas_token}"

        logger.debug(
            f"Generated Azure SAS URL for {remote_key}, expires in {expires_in}s"
        )
        return signed_url

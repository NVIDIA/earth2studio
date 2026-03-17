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
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from earth2studio.serve.server.object_storage import (
    MSCObjectStorage,
    ObjectStorage,
    ObjectStorageError,
    UploadResult,
)


class TestUploadResult:
    """Tests for UploadResult dataclass."""

    def test_repr_success(self):
        """__repr__ shows SUCCESS when success is True."""
        r = UploadResult(
            success=True,
            files_uploaded=10,
            total_bytes=1000,
            destination="s3://bucket/prefix",
            errors=[],
        )
        assert "SUCCESS" in repr(r)
        assert "files=10" in repr(r)
        assert "bytes=1000" in repr(r)

    def test_repr_failed(self):
        """__repr__ shows FAILED when success is False."""
        r = UploadResult(
            success=False,
            files_uploaded=0,
            total_bytes=0,
            destination="s3://bucket/prefix",
            errors=["error1"],
        )
        assert "FAILED" in repr(r)


class TestObjectStorageError:
    """Tests for ObjectStorageError."""

    def test_is_exception(self):
        """ObjectStorageError is a subclass of Exception."""
        assert issubclass(ObjectStorageError, Exception)

    def test_raise_and_catch(self):
        """Can raise and catch ObjectStorageError with message."""
        with pytest.raises(ObjectStorageError, match="test message"):
            raise ObjectStorageError("test message")


class TestObjectStorage:
    """Tests for ObjectStorage abstract base class."""

    def test_cannot_instantiate_directly(self):
        """ObjectStorage cannot be instantiated (abstract)."""
        with pytest.raises(TypeError):
            ObjectStorage()


class TestMSCObjectStorage:
    """Tests for MSCObjectStorage (with mocked multistorageclient)."""

    @pytest.fixture
    def mock_msc(self):
        """Mock multistorageclient module so MSCObjectStorage can be instantiated."""
        mock_module = MagicMock()
        mock_module.StorageClientConfig.from_dict.side_effect = [
            MagicMock(),
            MagicMock(),
        ]
        mock_module.StorageClient.return_value = MagicMock()
        with patch.dict(sys.modules, {"multistorageclient": mock_module}):
            yield mock_module

    def test_init_requires_multistorageclient(self):
        """MSCObjectStorage raises ImportError if multistorageclient is not available."""
        import builtins

        real_import = builtins.__import__

        def fail_msc(name, *args, **kwargs):
            if name == "multistorageclient":
                raise ImportError("No module named 'multistorageclient'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fail_msc):
            with pytest.raises(ImportError, match="multi-storage-client"):
                MSCObjectStorage(bucket="my-bucket")

    def test_upload_directory_raises_when_dir_not_found(self, mock_msc):
        """upload_directory raises FileNotFoundError when path does not exist."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")

        with pytest.raises(FileNotFoundError, match="not found"):
            storage.upload_directory(
                local_directory="/nonexistent/path",
                remote_prefix="prefix",
            )

    def test_upload_directory_raises_when_path_not_directory(self, mock_msc):
        """upload_directory raises ValueError when path is not a directory."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            file_path = f.name
        try:
            storage = MSCObjectStorage(bucket="b", region="us-east-1")

            with pytest.raises(ValueError, match="not a directory"):
                storage.upload_directory(
                    local_directory=file_path,
                    remote_prefix="prefix",
                )
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_upload_directory_success_returns_upload_result(self, mock_msc):
        """upload_directory returns UploadResult with success when sync_from succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "f1.txt").write_text("hello")
            storage = MSCObjectStorage(bucket="b", region="us-east-1")
            storage._storage_client.sync_from.return_value = None

            result = storage.upload_directory(
                local_directory=tmpdir,
                remote_prefix="prefix",
            )

            assert isinstance(result, UploadResult)
            assert result.success is True
            assert result.files_uploaded == 1
            assert result.total_bytes == 5
            assert "s3://b/prefix" in result.destination
            assert result.errors == []

    def test_upload_directory_failure_appends_errors(self, mock_msc):
        """upload_directory returns success=False and appends error when sync_from raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "f1.txt").write_text("x")
            storage = MSCObjectStorage(bucket="b", region="us-east-1")
            storage._storage_client.sync_from.side_effect = Exception("sync failed")

            result = storage.upload_directory(
                local_directory=tmpdir,
                remote_prefix="prefix",
            )

            assert result.success is False
            assert result.files_uploaded == 0
            assert len(result.errors) == 1
            assert "sync failed" in result.errors[0]

    def test_upload_file_returns_false_when_file_not_found(self, mock_msc):
        """upload_file returns False when local file does not exist."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")

        assert (
            storage.upload_file(
                local_file="/nonexistent/file.txt",
                remote_key="key.txt",
            )
            is False
        )

    def test_upload_file_returns_true_on_success(self, mock_msc):
        """upload_file returns True when upload succeeds."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"data")
            path = f.name
        try:
            storage = MSCObjectStorage(bucket="b", region="us-east-1")

            result = storage.upload_file(
                local_file=path,
                remote_key="key.txt",
            )
            assert result is True
            storage._storage_client.upload_file.assert_called_once()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_upload_file_returns_false_on_exception(self, mock_msc):
        """upload_file returns False when upload_file raises."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            storage = MSCObjectStorage(bucket="b", region="us-east-1")
            storage._storage_client.upload_file.side_effect = Exception("upload failed")

            result = storage.upload_file(
                local_file=path,
                remote_key="key.txt",
            )
            assert result is False
        finally:
            Path(path).unlink(missing_ok=True)

    def test_file_exists_returns_true_when_info_succeeds(self, mock_msc):
        """file_exists returns True when info() does not raise."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage._storage_client.info.return_value = None

        assert storage.file_exists("my/key") is True

    def test_file_exists_returns_false_when_file_not_found(self, mock_msc):
        """file_exists returns False when info() raises FileNotFoundError."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage._storage_client.info.side_effect = FileNotFoundError()

        assert storage.file_exists("my/key") is False

    def test_delete_file_returns_true_on_success(self, mock_msc):
        """delete_file returns True when delete succeeds."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")

        assert storage.delete_file("my/key") is True
        storage._storage_client.delete.assert_called_once()

    def test_delete_file_returns_false_when_file_not_found(self, mock_msc):
        """delete_file returns False when delete raises FileNotFoundError."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage._storage_client.delete.side_effect = FileNotFoundError()

        assert storage.delete_file("my/key") is False

    def test_url_safe_b64(self, mock_msc):
        """_url_safe_b64 produces URL-safe base64 (no +, /, =)."""
        _ = MSCObjectStorage(
            bucket="b", region="us-east-1"
        )  # need instance so mock_msc context is active
        raw = b"hello"
        result = MSCObjectStorage._url_safe_b64(raw)
        # Standard b64 could have +, /, =
        assert "+" not in result
        assert "/" not in result
        assert "=" not in result
        # Decode and check round-trip (with URL-safe mapping)
        decoded = base64.urlsafe_b64decode(
            result.replace("-", "+").replace("_", "=").replace("~", "/")
        )
        assert decoded == raw

    def test_generate_signed_url_raises_when_cloudfront_not_configured(self, mock_msc):
        """generate_signed_url raises ObjectStorageError when CloudFront not configured."""
        storage = MSCObjectStorage(
            bucket="b",
            region="us-east-1",
            cloudfront_domain=None,
            cloudfront_key_pair_id=None,
            cloudfront_private_key=None,
        )

        with pytest.raises(ObjectStorageError, match="CloudFront configuration"):
            storage.generate_signed_url("key.txt")

    def test_generate_signed_url_returns_url_when_configured(self, mock_msc):
        """generate_signed_url returns URL string when CloudFront is configured."""
        storage = MSCObjectStorage(
            bucket="b",
            region="us-east-1",
            cloudfront_domain="d123.cloudfront.net",
            cloudfront_key_pair_id="KP123",
            cloudfront_private_key="-----BEGIN RSA PRIVATE KEY-----\nMIIEow==\n-----END RSA PRIVATE KEY-----",
        )
        storage._rsa_signer = MagicMock(return_value=b"signature")

        url = storage.generate_signed_url("path/to/file", expires_in=3600)

        assert isinstance(url, str)
        assert "d123.cloudfront.net" in url
        assert "Policy=" in url
        assert "Signature=" in url
        assert "Key-Pair-Id=KP123" in url


class TestMSCObjectStorageS3Additional:
    """Additional tests to cover S3 init branches and other uncovered S3 paths."""

    @pytest.fixture
    def mock_msc(self):
        mock_module = MagicMock()
        mock_module.StorageClientConfig.from_dict.side_effect = [
            MagicMock(),
            MagicMock(),
        ]
        mock_module.StorageClient.return_value = MagicMock()
        with patch.dict(sys.modules, {"multistorageclient": mock_module}):
            yield mock_module

    def test_init_s3_transfer_acceleration(self, mock_msc):
        """S3 init with use_transfer_acceleration sets accelerate endpoint URL."""
        storage = MSCObjectStorage(bucket="my-bucket", use_transfer_acceleration=True)
        assert storage.endpoint_url == "https://my-bucket.s3-accelerate.amazonaws.com"

    def test_init_s3_with_credentials(self, mock_msc):
        """S3 init with credentials sets AWS environment variables."""
        import os

        MSCObjectStorage(
            bucket="b",
            access_key_id="AKID",
            secret_access_key="SECRET",  # noqa: S106
            session_token="TOKEN",  # noqa: S106
        )
        assert os.environ.get("AWS_ACCESS_KEY_ID") == "AKID"
        assert os.environ.get("AWS_SECRET_ACCESS_KEY") == "SECRET"
        assert os.environ.get("AWS_SESSION_TOKEN") == "TOKEN"

    def test_init_s3_with_endpoint_url(self, mock_msc):
        """S3 init with endpoint_url stores it and includes it in provider options."""
        storage = MSCObjectStorage(bucket="b", endpoint_url="http://minio:9000")
        assert storage.endpoint_url == "http://minio:9000"

    def test_init_s3_with_rust_client(self, mock_msc):
        """S3 init with use_rust_client=True adds rust_client section to config."""
        storage = MSCObjectStorage(bucket="b", use_rust_client=True)
        assert storage.use_rust_client is True

    def test_init_unsupported_storage_type(self, mock_msc):
        """__init__ raises ValueError for unsupported storage_type."""
        with pytest.raises(ValueError, match="Unsupported storage_type"):
            MSCObjectStorage(bucket="b", storage_type="gcs")

    def test_upload_directory_non_recursive(self, mock_msc):
        """upload_directory with recursive=False only counts top-level files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "top.txt").write_text("hello")
            subdir = Path(tmpdir) / "sub"
            subdir.mkdir()
            (subdir / "deep.txt").write_text("world")

            storage = MSCObjectStorage(bucket="b", region="us-east-1")
            storage._storage_client.sync_from.return_value = None

            result = storage.upload_directory(
                local_directory=tmpdir,
                remote_prefix="prefix",
                recursive=False,
            )
            assert result.success is True
            assert result.files_uploaded == 1  # only top-level file

    def test_upload_file_path_is_directory(self, mock_msc):
        """upload_file returns False when local_file path is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = MSCObjectStorage(bucket="b", region="us-east-1")
            result = storage.upload_file(local_file=tmpdir, remote_key="key.txt")
            assert result is False

    def test_delete_file_generic_exception(self, mock_msc):
        """delete_file returns False on an unexpected (non-FileNotFoundError) exception."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage._storage_client.delete.side_effect = RuntimeError("unexpected")
        assert storage.delete_file("my/key") is False

    def test_rsa_signer_no_key_raises(self, mock_msc):
        """_rsa_signer raises ObjectStorageError when cloudfront_private_key is None."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage.cloudfront_private_key = None
        with pytest.raises(ObjectStorageError, match="No CloudFront private key"):
            storage._rsa_signer(b"message")

    def test_rsa_signer_with_mocked_key(self, mock_msc):
        """_rsa_signer signs message using the configured private key."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage.cloudfront_private_key = (
            "-----BEGIN RSA PRIVATE KEY-----\nfake\n-----END RSA PRIVATE KEY-----"
        )

        mock_private_key = MagicMock()
        mock_private_key.sign.return_value = b"fake_signature"
        mock_serialization = MagicMock()
        mock_serialization.load_pem_private_key.return_value = mock_private_key

        crypto_mocks = {
            "cryptography": MagicMock(),
            "cryptography.hazmat": MagicMock(),
            "cryptography.hazmat.primitives": MagicMock(),
            "cryptography.hazmat.primitives.hashes": MagicMock(),
            "cryptography.hazmat.primitives.asymmetric": MagicMock(),
            "cryptography.hazmat.primitives.asymmetric.padding": MagicMock(),
            "cryptography.hazmat.primitives.serialization": mock_serialization,
        }
        with patch.dict(sys.modules, crypto_mocks):
            result = storage._rsa_signer(b"message")

        assert result == b"fake_signature"

    def test_generate_signed_url_unsupported_type(self, mock_msc):
        """generate_signed_url raises ObjectStorageError for unsupported storage_type."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage.storage_type = "unsupported"
        with pytest.raises(ObjectStorageError, match="Unsupported storage_type"):
            storage.generate_signed_url("key.txt")


class TestMSCObjectStorageAzure:
    """Tests for MSCObjectStorage with Azure storage type."""

    @pytest.fixture
    def mock_msc(self):
        mock_module = MagicMock()
        mock_module.StorageClientConfig.from_dict.side_effect = [
            MagicMock(),
            MagicMock(),
        ]
        mock_module.StorageClient.return_value = MagicMock()
        with patch.dict(sys.modules, {"multistorageclient": mock_module}):
            yield mock_module

    def _make_azure_storage(self, mock_msc, **kwargs):
        """Helper to reset mock side_effect for multiple instantiations."""
        mock_msc.StorageClientConfig.from_dict.side_effect = [
            MagicMock(),
            MagicMock(),
        ]
        return MSCObjectStorage(**kwargs)

    def test_init_azure_managed_identity_with_account_name(self, mock_msc):
        """Azure init with managed identity uses DefaultAzureCredentials."""
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_account_name="myaccount",
        )
        assert storage.use_managed_identity is True
        assert storage.azure_account_name == "myaccount"
        assert storage.storage_type == "azure"

    def test_init_azure_managed_identity_with_endpoint_url(self, mock_msc):
        """Azure init with managed identity and explicit endpoint_url succeeds."""
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            endpoint_url="https://myaccount.blob.core.windows.net",
        )
        assert storage.use_managed_identity is True

    def test_init_azure_no_account_name_no_endpoint_raises(self, mock_msc):
        """Azure managed identity raises ObjectStorageError when neither account name nor endpoint_url is given."""
        with pytest.raises(
            ObjectStorageError, match="Azure endpoint_url cannot be determined"
        ):
            MSCObjectStorage(
                bucket="mycontainer",
                storage_type="azure",
                # no azure_account_name, no endpoint_url, no connection_string
            )

    def test_init_azure_connection_string_extracts_account_name(self, mock_msc):
        """Azure init with connection string extracts AccountName from it."""
        conn_str = "DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=key123;EndpointSuffix=core.windows.net"
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_connection_string=conn_str,
        )
        assert storage.use_managed_identity is False
        assert storage.azure_account_name == "myaccount"

    def test_init_azure_connection_string_explicit_account_name(self, mock_msc):
        """Azure init with connection string uses explicitly provided azure_account_name."""
        conn_str = "DefaultEndpointsProtocol=https;AccountKey=somekey"
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_connection_string=conn_str,
            azure_account_name="explicitaccount",
        )
        assert storage.azure_account_name == "explicitaccount"

    def test_init_azure_connection_string_no_account_name_raises(self, mock_msc):
        """Azure init raises ObjectStorageError when connection string has no AccountName and none provided."""
        conn_str = "DefaultEndpointsProtocol=https;AccountKey=somekey"
        with pytest.raises(ObjectStorageError, match="Could not extract account name"):
            MSCObjectStorage(
                bucket="mycontainer",
                storage_type="azure",
                azure_connection_string=conn_str,
            )

    def test_init_azure_connection_string_with_blob_endpoint(self, mock_msc):
        """Azure init uses BlobEndpoint directly from connection string."""
        conn_str = "BlobEndpoint=https://myaccount.blob.core.windows.net;AccountName=myaccount;AccountKey=key"
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_connection_string=conn_str,
        )
        assert storage.azure_account_name == "myaccount"

    def test_init_azure_connection_string_with_endpoint_suffix(self, mock_msc):
        """Azure init constructs endpoint URL using EndpointSuffix from connection string."""
        conn_str = (
            "AccountName=myaccount;AccountKey=key;EndpointSuffix=core.chinacloudapi.cn"
        )
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_connection_string=conn_str,
        )
        assert storage.azure_account_name == "myaccount"

    def test_upload_directory_azure_destination(self, mock_msc):
        """upload_directory for azure uses azure:// in the destination."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "f.txt").write_text("hi")
            storage = MSCObjectStorage(
                bucket="mycontainer",
                storage_type="azure",
                azure_account_name="acct",
                endpoint_url="https://acct.blob.core.windows.net",
            )
            storage._storage_client.sync_from.return_value = None

            result = storage.upload_directory(
                local_directory=tmpdir,
                remote_prefix="prefix",
            )
            assert result.success is True
            assert "azure://" in result.destination
            assert "mycontainer" in result.destination

    def test_generate_signed_url_azure_delegates_to_sas(self, mock_msc):
        """generate_signed_url for azure delegates to _generate_azure_sas_url."""
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_account_name="acct",
            endpoint_url="https://acct.blob.core.windows.net",
        )
        storage._generate_azure_sas_url = MagicMock(return_value="https://sas_url")

        result = storage.generate_signed_url("key.txt")

        assert result == "https://sas_url"
        storage._generate_azure_sas_url.assert_called_once_with("key.txt", 86400)

    def test_generate_azure_sas_url_missing_credentials_raises(self, mock_msc):
        """_generate_azure_sas_url raises ObjectStorageError when account key is missing."""
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_account_name="acct",
            endpoint_url="https://acct.blob.core.windows.net",
            # azure_account_key not provided → None
        )
        with pytest.raises(
            ObjectStorageError, match="Azure account name and account key"
        ):
            storage._generate_azure_sas_url("key.txt", 3600)

    def test_generate_azure_sas_url_import_error(self, mock_msc):
        """_generate_azure_sas_url raises ImportError when azure-storage-blob is not installed."""
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_account_name="acct",
            azure_account_key="key123",
            endpoint_url="https://acct.blob.core.windows.net",
        )
        with patch.dict(
            sys.modules,
            {
                "azure": MagicMock(),
                "azure.storage": MagicMock(),
                "azure.storage.blob": None,
            },
        ):
            with pytest.raises(ImportError, match="azure-storage-blob"):
                storage._generate_azure_sas_url("key.txt", 3600)

    def test_generate_azure_sas_url_success(self, mock_msc):
        """_generate_azure_sas_url returns a properly constructed SAS URL."""
        storage = MSCObjectStorage(
            bucket="mycontainer",
            storage_type="azure",
            azure_account_name="myaccount",
            azure_account_key="mykey",
            endpoint_url="https://myaccount.blob.core.windows.net",
        )

        mock_azure_blob = MagicMock()
        mock_azure_blob.ContainerSasPermissions.return_value = MagicMock()
        mock_azure_blob.generate_container_sas.return_value = "sv=2020&sig=abc"

        with patch.dict(
            sys.modules,
            {
                "azure": MagicMock(),
                "azure.storage": MagicMock(),
                "azure.storage.blob": mock_azure_blob,
            },
        ):
            url = storage._generate_azure_sas_url("path/to/file", 3600)

        assert "myaccount" in url
        assert "mycontainer" in url
        assert "path/to/file" in url
        assert "sv=2020" in url

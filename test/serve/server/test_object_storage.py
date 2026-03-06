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

"""Unit tests for the serve/server object_storage module."""

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
            storage._s3_client.sync_from.return_value = None

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
            storage._s3_client.sync_from.side_effect = Exception("sync failed")

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
            storage._s3_client.upload_file.assert_called_once()
        finally:
            Path(path).unlink(missing_ok=True)

    def test_upload_file_returns_false_on_exception(self, mock_msc):
        """upload_file returns False when upload_file raises."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = f.name
        try:
            storage = MSCObjectStorage(bucket="b", region="us-east-1")
            storage._s3_client.upload_file.side_effect = Exception("upload failed")

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
        storage._s3_client.info.return_value = None

        assert storage.file_exists("my/key") is True

    def test_file_exists_returns_false_when_file_not_found(self, mock_msc):
        """file_exists returns False when info() raises FileNotFoundError."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage._s3_client.info.side_effect = FileNotFoundError()

        assert storage.file_exists("my/key") is False

    def test_delete_file_returns_true_on_success(self, mock_msc):
        """delete_file returns True when delete succeeds."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")

        assert storage.delete_file("my/key") is True
        storage._s3_client.delete.assert_called_once()

    def test_delete_file_returns_false_when_file_not_found(self, mock_msc):
        """delete_file returns False when delete raises FileNotFoundError."""
        storage = MSCObjectStorage(bucket="b", region="us-east-1")
        storage._s3_client.delete.side_effect = FileNotFoundError()

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

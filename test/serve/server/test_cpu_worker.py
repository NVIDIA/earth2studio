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
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from unittest.mock import Mock, patch

import pytest
import redis


# Create mock config classes before importing cpu_worker
@dataclass
class MockRedisConfig:
    """Mock Redis configuration"""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    socket_connect_timeout: int = 5
    socket_timeout: int = 5
    decode_responses: bool = True
    retention_ttl: int = 604800  # Redis key retention TTL in seconds (7 days)


@dataclass
class MockQueueConfig:
    """Mock queue configuration"""

    name: str = "inference"
    result_zip_queue_name: str = "result_zip"
    object_storage_queue_name: str = "object_storage"
    finalize_metadata_queue_name: str = "finalize_metadata"
    max_size: int = 10
    default_timeout: str = "1h"
    job_timeout: str = "2h"


# Secure temp dir for mock path defaults (S108)
_secure_test_dir = tempfile.mkdtemp(prefix="e2s_testing_")


@dataclass
class MockPathsConfig:
    """Mock paths configuration"""

    default_output_dir: str = field(default_factory=lambda: _secure_test_dir)
    results_zip_dir: str = field(default_factory=lambda: _secure_test_dir)
    output_format: Literal["zarr", "netcdf4"] = "zarr"
    result_zip_enabled: bool = False


@dataclass
class MockLoggingConfig:
    """Mock logging configuration"""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class MockServerConfig:
    """Mock server configuration"""

    host: str = "0.0.0.0"  # noqa: S104
    port: int = 8000
    workers: int = 1
    title: str = "REST API"
    description: str = "REST API for running workflows"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


@dataclass
class MockCORSConfig:
    """Mock CORS configuration"""

    allow_origins: list = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: list = field(default_factory=lambda: ["*"])
    allow_headers: list = field(default_factory=lambda: ["*"])


@dataclass
class MockObjectStorageConfig:
    """Mock object storage configuration"""

    enabled: bool = False
    bucket: str | None = None
    region: str = "us-east-1"
    prefix: str = "outputs"
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    cloudfront_domain: str | None = None
    cloudfront_key_pair_id: str | None = None
    cloudfront_private_key_path: str | None = None
    signed_url_expires_in: int = 3600


@dataclass
class MockAppConfig:
    """Mock root configuration"""

    redis: MockRedisConfig = field(default_factory=MockRedisConfig)
    queue: MockQueueConfig = field(default_factory=MockQueueConfig)
    paths: MockPathsConfig = field(default_factory=MockPathsConfig)
    logging: MockLoggingConfig = field(default_factory=MockLoggingConfig)
    server: MockServerConfig = field(default_factory=MockServerConfig)
    cors: MockCORSConfig = field(default_factory=MockCORSConfig)
    object_storage: MockObjectStorageConfig = field(
        default_factory=MockObjectStorageConfig
    )


# Create a mock config module
mock_config_module = Mock()
mock_config_module.AppConfig = MockAppConfig
mock_config_module.get_config = Mock(return_value=MockAppConfig())
mock_config_module.get_config_manager = Mock()

pytest.importorskip("earth2studio.serve.server")

# Inject the mock before importing cpu_worker
sys.modules["earth2studio.serve.server.config"] = mock_config_module

# Now import the module under test
from earth2studio.serve.server.cpu_worker import (  # noqa: E402  # noqa: E402
    FileManifestEntry,
    ResultMetadata,
    build_file_manifest,
    create_results_zip,
    fail_workflow,
    process_finalize_metadata,
    process_object_storage_upload,
    process_result_zip,
)
from earth2studio.serve.server.workflow import WorkflowResult  # noqa: E402


class TestCreateResultsZip:
    """Test suite for create_results_zip function"""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client"""
        return Mock(spec=redis.Redis)

    @pytest.fixture
    def inference_request(self):
        """Sample inference request data"""
        return {
            "id": "test_req_123",
            "type": "deterministic",
            "status": "completed",
            "completion_time": "2024-01-01T12:00:00Z",
            "execution_time_seconds": 45.5,
            "created_at": "2024-01-01T11:00:00Z",
            "peak_memory_usage": "2.5GB",
            "device": "cuda:0",
            "request": {
                "workflow_type": "deterministic",
                "time": "2024-01-01T00:00:00Z",
                "nsteps": 10,
                "prognostic": {"model_type": "fcn"},
                "data": {"source_type": "gfs"},
                "io": {"backend_type": "zarr"},
            },
        }

    def test_create_results_zip_with_single_file(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test creating zip with a single output file"""
        # Create a temporary output file
        output_file = tmp_path / "output.nc"
        output_file.write_text("test data content")
        file_size = output_file.stat().st_size

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip
        zip_filename = create_results_zip(
            request_id="test_req_123",
            output_path=output_file,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        # Verify zip file was created - filename is now just request_id
        assert zip_filename is not None
        assert zip_filename == "test_req_123"

        zip_path = results_zip_dir / zip_filename
        assert zip_path.exists()

        # Verify Redis was updated (4 times: metadata, results_zip_dir, zip, output_path)
        assert mock_redis.setex.call_count == 4
        calls = mock_redis.setex.call_args_list

        # First call: metadata (stored in Redis for finalize worker)
        assert calls[0][0][0] == "inference_request:test_req_123:pending_metadata"

        # Second call: results_zip_dir path
        assert calls[1][0][0] == "inference_request:test_req_123:results_zip_dir"
        assert calls[1][0][2] == str(results_zip_dir)

        # Third call: zip filename
        assert calls[2][0][0] == "inference_request_zips:test_req_123:zip_file"
        assert calls[2][0][2] == zip_filename

        # Fourth call: output path
        assert calls[3][0][0] == "inference_request:test_req_123:output_path"
        assert calls[3][0][1] == 86400  # 24 hours
        assert calls[3][0][2] == str(output_file)

        # Verify zip contents - metadata.json is NOT in the zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_list = zf.namelist()
            assert "metadata.json" not in file_list  # metadata is now separate
            assert "output.nc" in file_list

        # Verify metadata is stored in Redis (not as file - file is created by finalize worker)
        # Get the metadata from the first setex call
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)

        # Verify metadata structure
        assert metadata["request_id"] == "test_req_123"
        assert metadata["status"] == "completed"
        assert metadata["workflow_type"] == "deterministic"
        assert metadata["execution_time_seconds"] == 45.5
        assert metadata["device"] == "cuda:0"
        assert "zip_created_at" in metadata

        # Verify parameters field
        assert "parameters" in metadata
        assert metadata["parameters"] == inference_request["request"]
        assert metadata["parameters"]["workflow_type"] == "deterministic"
        assert metadata["parameters"]["nsteps"] == 10

        # Verify output_files manifest includes both the file AND the zip
        assert "output_files" in metadata
        assert isinstance(metadata["output_files"], list)
        assert len(metadata["output_files"]) == 2  # output.nc + zip file entry

        # Convert to dict for easier checking
        files_dict = {f["path"]: f["size"] for f in metadata["output_files"]}

        # Check the output file entry
        assert "output.nc" in files_dict
        assert files_dict["output.nc"] == file_size

        # Check the zip file entry (path = request_id)
        assert "test_req_123" in files_dict
        assert files_dict["test_req_123"] == zip_path.stat().st_size

    def test_create_results_zip_with_directory(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test creating zip with a directory containing multiple files"""
        # Create a temporary output directory with files
        output_dir = tmp_path / "output_data"
        output_dir.mkdir()

        file1 = output_dir / "data1.nc"
        file1.write_text("data 1 content")
        file1_size = file1.stat().st_size

        file2 = output_dir / "data2.nc"
        file2.write_text("data 2 content with more text")
        file2_size = file2.stat().st_size

        # Create subdirectory
        subdir = output_dir / "plots"
        subdir.mkdir()
        file3 = subdir / "plot.png"
        file3.write_bytes(b"fake png data")
        file3_size = file3.stat().st_size

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip
        zip_filename = create_results_zip(
            request_id="test_req_456",
            output_path=output_dir,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        # Verify zip file was created - filename is now just request_id
        assert zip_filename is not None
        assert zip_filename == "test_req_456"
        zip_path = results_zip_dir / zip_filename
        assert zip_path.exists()

        # Verify zip contents - metadata.json is NOT in the zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_list = zf.namelist()
            assert "metadata.json" not in file_list

            # Verify files exist in zip
            assert "output_data/data1.nc" in file_list
            assert "output_data/data2.nc" in file_list
            assert "output_data/plots/plot.png" in file_list

        # Verify metadata is stored in Redis (not as file)
        assert mock_redis.setex.call_count == 4
        calls = mock_redis.setex.call_args_list
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)

        # Verify output_files manifest (3 files + 1 zip entry)
        assert "output_files" in metadata
        output_files = metadata["output_files"]
        assert len(output_files) == 4  # 3 files + zip entry

        # Convert to dict for easier checking
        files_dict = {f["path"]: f["size"] for f in output_files}

        # Verify all files are in manifest with correct sizes
        assert "output_data/data1.nc" in files_dict
        assert files_dict["output_data/data1.nc"] == file1_size

        assert "output_data/data2.nc" in files_dict
        assert files_dict["output_data/data2.nc"] == file2_size

        assert "output_data/plots/plot.png" in files_dict
        assert files_dict["output_data/plots/plot.png"] == file3_size

        # Verify zip file entry (path = request_id)
        assert "test_req_456" in files_dict
        assert files_dict["test_req_456"] == zip_path.stat().st_size

    def test_create_results_zip_with_nonexistent_output(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test creating zip when output path doesn't exist"""
        # Use nonexistent path
        output_path = tmp_path / "nonexistent"

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip
        zip_filename = create_results_zip(
            request_id="test_req_789",
            output_path=output_path,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        # Should still create zip - filename is now just request_id
        assert zip_filename is not None
        assert zip_filename == "test_req_789"
        zip_path = results_zip_dir / zip_filename
        assert zip_path.exists()

        # Verify zip contents - should be empty (no output files)
        with zipfile.ZipFile(zip_path, "r") as zf:
            file_list = zf.namelist()
            assert len(file_list) == 0  # No files in zip

        # Verify metadata is stored in Redis (not as file)
        assert mock_redis.setex.call_count == 4
        calls = mock_redis.setex.call_args_list
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)

        # output_files should only contain the zip entry
        assert "output_files" in metadata
        assert len(metadata["output_files"]) == 1  # Only zip entry
        assert metadata["output_files"][0]["path"] == "test_req_789"

    def test_create_results_zip_parameters_validation(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test that parameters are correctly stored in metadata"""
        # Create a temporary output file
        output_file = tmp_path / "output.nc"
        output_file.write_text("test")

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Modify inference request to have specific parameters
        inference_request["request"] = {
            "workflow_type": "ensemble",
            "time": "2024-01-15T00:00:00Z",
            "nsteps": 20,
            "nensemble": 10,
            "prognostic": {"model_type": "graphcast"},
            "data": {"source_type": "era5"},
            "io": {"backend_type": "netcdf"},
            "perturbation": {"method": "spherical_gaussian"},
        }

        # Call create_results_zip
        result = create_results_zip(
            request_id="test_req_params",
            output_path=output_file,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )
        assert result is not None

        # Verify metadata is stored in Redis (not as file)
        assert mock_redis.setex.call_count == 4
        calls = mock_redis.setex.call_args_list
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)

        # Verify parameters field exists and matches
        assert "parameters" in metadata
        params = metadata["parameters"]

        assert params["workflow_type"] == "ensemble"
        assert params["time"] == "2024-01-15T00:00:00Z"
        assert params["nsteps"] == 20
        assert params["nensemble"] == 10
        assert params["prognostic"]["model_type"] == "graphcast"
        assert params["data"]["source_type"] == "era5"
        assert params["io"]["backend_type"] == "netcdf"
        assert params["perturbation"]["method"] == "spherical_gaussian"

    def test_create_results_zip_file_manifest_structure(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test that file manifest has correct structure"""
        # Create output directory with multiple files
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        for i in range(5):
            file = output_dir / f"file_{i}.dat"
            file.write_text(f"content {i}" * 100)

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip
        zip_filename = create_results_zip(
            request_id="test_manifest",
            output_path=output_dir,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        # Verify metadata is stored in Redis
        assert mock_redis.setex.call_count == 4
        calls = mock_redis.setex.call_args_list
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)

        manifest = metadata["output_files"]

        # Verify manifest structure (5 files + 1 zip entry)
        assert isinstance(manifest, list)
        assert len(manifest) == 6  # 5 files + zip entry

        # Find the zip entry (path = request_id)
        zip_entry = None
        file_entries = []
        for entry in manifest:
            if entry["path"] == "test_manifest":
                zip_entry = entry
            else:
                file_entries.append(entry)

        assert zip_entry is not None
        assert len(file_entries) == 5

        # Verify file entries structure
        zip_path = results_zip_dir / zip_filename
        with zipfile.ZipFile(zip_path, "r") as zf:
            for entry in file_entries:
                assert isinstance(entry, dict)
                assert "path" in entry
                assert "size" in entry
                assert isinstance(entry["path"], str)
                assert isinstance(entry["size"], int)
                assert entry["size"] > 0

                # Verify path starts with output directory name
                assert entry["path"].startswith("outputs/")

                # Verify size matches actual file in zip
                actual_size = zf.getinfo(entry["path"]).file_size
                assert entry["size"] == actual_size

    def test_create_results_zip_missing_request_parameters(self, mock_redis, tmp_path):
        """Test handling when request doesn't have parameters field"""
        # Create inference request without 'request' field
        inference_request = {
            "id": "test_req_no_params",
            "type": "deterministic",
            "status": "completed",
        }

        # Create output file
        output_file = tmp_path / "output.nc"
        output_file.write_text("test")

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip
        zip_filename = create_results_zip(
            request_id="test_req_no_params",
            output_path=output_file,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        # Verify zip was created
        assert zip_filename is not None

        # Verify metadata stored in Redis - parameters should be None
        assert mock_redis.setex.call_count == 4
        calls = mock_redis.setex.call_args_list
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)

        # Parameters should be None or not present
        assert metadata.get("parameters") is None

    def test_create_results_zip_redis_failure(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test handling when Redis operation fails"""
        # Make Redis setex raise an exception
        mock_redis.setex.side_effect = Exception("Redis connection failed")

        # Create output file
        output_file = tmp_path / "output.nc"
        output_file.write_text("test")

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip - should handle exception gracefully
        zip_filename = create_results_zip(
            request_id="test_redis_fail",
            output_path=output_file,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        # Should return None since storing in Redis failed
        assert zip_filename is None

    def test_create_results_zip_exception_handling(self, mock_redis, tmp_path):
        """Test general exception handling"""
        # Use invalid inference request that will cause issues
        invalid_request = None

        # Create output file
        output_file = tmp_path / "output.nc"
        output_file.write_text("test")

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip with invalid data
        zip_filename = create_results_zip(
            request_id="test_exception",
            output_path=output_file,
            inference_request=invalid_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        # Should return None on exception
        assert zip_filename is None

    def test_create_results_zip_metadata_completeness(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test that all expected metadata fields are present"""
        # Create output file
        output_file = tmp_path / "output.nc"
        output_file.write_text("test data")

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip
        zip_filename = create_results_zip(
            request_id="test_metadata",
            output_path=output_file,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        assert zip_filename is not None

        # Verify metadata stored in Redis
        assert mock_redis.setex.call_count == 4
        calls = mock_redis.setex.call_args_list
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)

        # Check all expected fields
        required_fields = [
            "request_id",
            "status",
            "completion_time",
            "execution_time_seconds",
            "workflow_type",
            "created_at",
            "peak_memory_usage",
            "device",
            "zip_created_at",
            "parameters",
            "output_files",
        ]

        for f in required_fields:
            assert f in metadata, f"Missing required field: {f}"

        # Verify zip_created_at is a valid timestamp
        zip_created_at = metadata["zip_created_at"]
        assert zip_created_at.endswith("Z")
        # Should be parseable as ISO format
        datetime.fromisoformat(zip_created_at.replace("Z", "+00:00"))

    def test_create_results_zip_manifest_includes_zip_entry(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test that the manifest includes an entry for the zip file itself"""
        # Create output file
        output_file = tmp_path / "output.nc"
        output_file.write_text("test data content")

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        request_id = "test_zip_entry"

        # Call create_results_zip
        zip_filename = create_results_zip(
            request_id=request_id,
            output_path=output_file,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        assert zip_filename is not None

        zip_path = results_zip_dir / zip_filename
        zip_size = zip_path.stat().st_size

        # Verify metadata stored in Redis
        calls = mock_redis.setex.call_args_list
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)

        # Find the zip entry in manifest
        zip_entry = None
        for entry in metadata["output_files"]:
            if entry["path"] == request_id:
                zip_entry = entry
                break

        # Verify zip entry exists and has correct size
        assert zip_entry is not None, "Zip file entry not found in manifest"
        assert zip_entry["path"] == request_id
        assert zip_entry["size"] == zip_size

    def test_create_results_zip_output_path_stored_in_redis(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test that output path is stored in Redis"""
        # Create output file
        output_file = tmp_path / "output.nc"
        output_file.write_text("test data")

        # Create results directory
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        # Call create_results_zip
        zip_filename = create_results_zip(
            request_id="test_output_path",
            output_path=output_file,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        assert zip_filename is not None

        # Verify Redis setex was called 4 times (metadata, results_zip_dir, zip, output_path)
        assert mock_redis.setex.call_count == 4

        # Check the output path call (4th call)
        calls = mock_redis.setex.call_args_list
        output_path_call = calls[3]

        assert (
            output_path_call[0][0] == "inference_request:test_output_path:output_path"
        )
        assert output_path_call[0][1] == 86400  # 24 hours
        assert output_path_call[0][2] == str(output_file)

    def test_create_results_zip_manifest_only_no_zip(
        self, mock_redis, inference_request, tmp_path
    ):
        """Test create_results_zip with create_zip=False: no zip file, manifest only."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "a.nc").write_text("data")
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        result = create_results_zip(
            request_id="req_manifest_only",
            output_path=output_dir,
            inference_request=inference_request,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
            create_zip=False,
        )

        assert result is None  # No zip filename when create_zip=False
        assert not (results_zip_dir / "req_manifest_only").exists()
        # Metadata should still be stored (manifest only)
        assert mock_redis.setex.call_count >= 3
        calls = mock_redis.setex.call_args_list
        metadata_json = calls[0][0][2]
        metadata = json.loads(metadata_json)
        assert len(metadata["output_files"]) == 1  # Just the one file, no zip entry

    def test_create_results_zip_with_workflow_result(self, mock_redis, tmp_path):
        """Test create_results_zip with WorkflowResult (custom workflow) path."""
        output_file = tmp_path / "out.nc"
        output_file.write_text("data")
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        workflow_result = WorkflowResult(
            workflow_name="my_workflow",
            execution_id="exec_1",
            status="completed",
            start_time="2024-01-01T10:00:00Z",
            end_time="2024-01-01T11:00:00Z",
            execution_time_seconds=60.0,
            metadata={
                "peak_memory_usage": "1GB",
                "device": "cuda:0",
                "parameters": {"nsteps": 5},
            },
        )

        zip_filename = create_results_zip(
            request_id="my_workflow:exec_1",
            output_path=output_file,
            inference_request=workflow_result,
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        assert zip_filename is not None
        metadata_json = mock_redis.setex.call_args_list[0][0][2]
        metadata = json.loads(metadata_json)
        assert metadata.get("workflow_name") == "my_workflow"
        # from_workflow_result uses workflow_result.execution_id for request_id when set
        assert metadata.get("request_id") == "exec_1"
        assert metadata.get("peak_memory_usage") == "1GB"
        assert metadata.get("parameters", {}).get("nsteps") == 5

    def test_create_results_zip_invalid_type_returns_none(self, mock_redis, tmp_path):
        """Test that invalid inference_request type is handled (exception caught, returns None)."""
        output_file = tmp_path / "out.nc"
        output_file.write_text("data")
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        result = create_results_zip(
            request_id="req_invalid",
            output_path=output_file,
            inference_request=[],  # list is not dict or WorkflowResult
            results_zip_dir=results_zip_dir,
            redis_client=mock_redis,
        )

        assert result is None


class TestFailWorkflow:
    """Tests for fail_workflow."""

    def test_fail_workflow_returns_error_dict(self):
        """fail_workflow returns success=False and error message."""
        out = fail_workflow("w", "e1", "Something failed")
        assert out["success"] is False
        assert out["error"] == "Something failed"

    def test_fail_workflow_updates_redis_when_workflow_registered(self):
        """When workflow is in registry, _update_execution_data is called."""
        mock_workflow_class = Mock()
        mock_workflow_class._update_execution_data = Mock()

        with patch(
            "earth2studio.serve.server.cpu_worker.workflow_registry"
        ) as mock_registry:
            mock_registry.get_workflow_class.return_value = mock_workflow_class
            with patch("earth2studio.serve.server.cpu_worker.redis_client"):
                out = fail_workflow("my_wf", "exec_1", "Failed")

        assert out["success"] is False
        mock_workflow_class._update_execution_data.assert_called_once()
        call_args = mock_workflow_class._update_execution_data.call_args
        assert call_args[0][1] == "my_wf"
        assert call_args[0][2] == "exec_1"
        assert call_args[0][3]["status"] == "failed"
        assert call_args[0][3]["error_message"] == "Failed"

    def test_fail_workflow_exception_still_returns_dict(self):
        """When _update_execution_data raises, we still return error dict."""
        mock_workflow_class = Mock()
        mock_workflow_class._update_execution_data.side_effect = Exception("Redis down")

        with patch(
            "earth2studio.serve.server.cpu_worker.workflow_registry"
        ) as mock_registry:
            mock_registry.get_workflow_class.return_value = mock_workflow_class
            out = fail_workflow("w", "e1", "Err")

        assert out["success"] is False
        assert out["error"] == "Err"


class TestBuildFileManifest:
    """Tests for build_file_manifest."""

    def test_build_file_manifest_single_file(self, tmp_path):
        """Single file returns one entry."""
        f = tmp_path / "a.nc"
        f.write_text("x")
        entries = build_file_manifest(f)
        assert len(entries) == 1
        assert entries[0].path == "a.nc"
        assert entries[0].size == f.stat().st_size

    def test_build_file_manifest_directory_with_subdirs(self, tmp_path):
        """Directory with subdirs is scanned recursively."""
        d = tmp_path / "out"
        d.mkdir()
        (d / "a.nc").write_text("a")
        sub = d / "sub"
        sub.mkdir()
        (sub / "b.nc").write_text("bb")
        entries = build_file_manifest(d)
        assert len(entries) == 2
        paths = {e.path for e in entries}
        assert "out/a.nc" in paths
        assert "out/sub/b.nc" in paths

    def test_build_file_manifest_with_arc_prefix(self, tmp_path):
        """arc_prefix is applied to paths."""
        f = tmp_path / "f.nc"
        f.write_text("x")
        entries = build_file_manifest(f, arc_prefix="prefix")
        assert len(entries) == 1
        assert entries[0].path == "prefix/f.nc"

    def test_build_file_manifest_nonexistent_returns_empty(self, tmp_path):
        """Nonexistent path returns empty list."""
        entries = build_file_manifest(tmp_path / "nonexistent")
        assert entries == []


class TestResultMetadata:
    """Tests for ResultMetadata factory methods and to_dict."""

    def test_from_workflow_result(self):
        """ResultMetadata.from_workflow_result sets workflow_name and metadata."""
        wr = WorkflowResult(
            workflow_name="custom_wf",
            execution_id="exec_1",
            status="completed",
            start_time="2024-01-01T10:00:00Z",
            end_time="2024-01-01T11:00:00Z",
            execution_time_seconds=30.0,
            metadata={"peak_memory_usage": "2GB", "parameters": {"n": 1}},
        )
        manifest = [FileManifestEntry(path="out.nc", size=100)]
        meta = ResultMetadata.from_workflow_result(
            wr, "exec_1", manifest, "2024-01-01T12:00:00Z"
        )
        assert meta.request_id == "exec_1"
        assert meta.workflow_name == "custom_wf"
        assert meta.status == "completed"
        assert meta.parameters == {"n": 1}
        assert meta.output_files == manifest

    def test_from_legacy_dict(self):
        """ResultMetadata.from_legacy_dict sets workflow_type from dict."""
        req = {
            "status": "completed",
            "completion_time": "2024-01-01T11:00:00Z",
            "execution_time_seconds": 10.0,
            "type": "deterministic",
            "created_at": "2024-01-01T10:00:00Z",
            "request": {"nsteps": 5},
        }
        manifest = [FileManifestEntry(path="out.zarr", size=200)]
        meta = ResultMetadata.from_legacy_dict(
            req, "req_123", manifest, "2024-01-01T12:00:00Z"
        )
        assert meta.request_id == "req_123"
        assert meta.workflow_type == "deterministic"
        assert meta.parameters == {"nsteps": 5}

    def test_to_dict_includes_workflow_name_and_type(self):
        """to_dict includes workflow_name or workflow_type when set."""
        meta = ResultMetadata(
            request_id="r1",
            status="completed",
            completion_time=None,
            execution_time_seconds=None,
            workflow_name="my_wf",
            workflow_type=None,
            output_files=[],
        )
        d = meta.to_dict()
        assert d["workflow_name"] == "my_wf"
        assert "workflow_type" not in d

        meta2 = ResultMetadata(
            request_id="r2",
            status="completed",
            completion_time=None,
            execution_time_seconds=None,
            workflow_type="ensemble",
            output_files=[],
        )
        d2 = meta2.to_dict()
        assert d2["workflow_type"] == "ensemble"
        assert "workflow_name" not in d2

    def test_to_dict_includes_signed_url_and_remote_path(self):
        """to_dict includes signed_url and remote_path when set."""
        meta = ResultMetadata(
            request_id="r1",
            status="completed",
            completion_time=None,
            execution_time_seconds=None,
            signed_url="https://s3.example.com/signed",
            remote_path="outputs/r1/out.nc",
            output_files=[],
        )
        d = meta.to_dict()
        assert d["signed_url"] == "https://s3.example.com/signed"
        assert d["remote_path"] == "outputs/r1/out.nc"


class TestProcessResultZip:
    """Tests for process_result_zip RQ worker function."""

    def test_process_result_zip_success_returns_zip_filename(self, tmp_path):
        """process_result_zip with valid workflow and create_zip returns zip filename."""
        output_file = tmp_path / "out.nc"
        output_file.write_text("data")
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        mock_workflow_class = Mock()
        mock_workflow_class._get_execution_data.return_value = {
            "id": "wf:exec_1",
            "type": "deterministic",
            "status": "completed",
        }
        mock_workflow_class._update_execution_data = Mock()

        with patch(
            "earth2studio.serve.server.cpu_worker.workflow_registry"
        ) as mock_registry:
            with patch(
                "earth2studio.serve.server.cpu_worker.redis_client"
            ) as mock_redis:
                with patch(
                    "earth2studio.serve.server.cpu_worker.queue_next_stage"
                ) as mock_queue_next:
                    mock_registry.get_workflow_class.return_value = mock_workflow_class
                    mock_redis.setex = Mock()
                    mock_queue_next.return_value = "job_123"

                    result = process_result_zip(
                        workflow_name="test_wf",
                        execution_id="exec_1",
                        output_path_str=str(output_file),
                        results_zip_dir_str=str(results_zip_dir),
                        create_zip=True,
                    )

                    assert result is not None
                    assert result == "test_wf:exec_1"  # zip filename is request_id
                    mock_queue_next.assert_called_once_with(
                        redis_client=mock_redis,
                        current_stage="result_zip",
                        workflow_name="test_wf",
                        execution_id="exec_1",
                        output_path_str=str(output_file),
                        results_zip_dir_str=str(results_zip_dir),
                    )

    def test_process_result_zip_workflow_not_found_raises(self, tmp_path):
        """process_result_zip when workflow not in registry raises after fail_workflow."""
        with patch(
            "earth2studio.serve.server.cpu_worker.workflow_registry"
        ) as mock_registry:
            with patch(
                "earth2studio.serve.server.cpu_worker.fail_workflow"
            ) as mock_fail:
                mock_registry.get_workflow_class.return_value = None

                with pytest.raises(ValueError, match="not found in registry"):
                    process_result_zip(
                        workflow_name="nonexistent",
                        execution_id="e1",
                        output_path_str=str(tmp_path / "out"),
                        results_zip_dir_str=str(tmp_path / "zip"),
                    )
                mock_fail.assert_called_once()

    def test_process_result_zip_queue_next_stage_returns_none(self, tmp_path):
        """When queue_next_stage returns None, fail_workflow is called and function returns None."""
        output_file = tmp_path / "out.nc"
        output_file.write_text("data")
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        mock_workflow_class = Mock()
        mock_workflow_class._get_execution_data.return_value = {
            "type": "deterministic",
            "status": "completed",
        }

        with patch(
            "earth2studio.serve.server.cpu_worker.workflow_registry"
        ) as mock_registry:
            with patch(
                "earth2studio.serve.server.cpu_worker.redis_client"
            ) as mock_redis:
                with patch(
                    "earth2studio.serve.server.cpu_worker.queue_next_stage"
                ) as mock_queue_next:
                    with patch(
                        "earth2studio.serve.server.cpu_worker.fail_workflow"
                    ) as mock_fail:
                        mock_registry.get_workflow_class.return_value = (
                            mock_workflow_class
                        )
                        mock_redis.setex = Mock()
                        mock_queue_next.return_value = None

                        result = process_result_zip(
                            workflow_name="test_wf",
                            execution_id="exec_1",
                            output_path_str=str(output_file),
                            results_zip_dir_str=str(results_zip_dir),
                            create_zip=True,
                        )

        assert result is None
        mock_fail.assert_called_once()
        assert "queue next pipeline stage" in mock_fail.call_args[0][2].lower()

    def test_process_result_zip_manifest_only_returns_manifest_only(self, tmp_path):
        """When create_zip=False and flow succeeds, returns 'manifest_only'."""
        output_file = tmp_path / "out.nc"
        output_file.write_text("data")
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()

        mock_workflow_class = Mock()
        mock_workflow_class._get_execution_data.return_value = {
            "type": "deterministic",
            "status": "completed",
        }

        with patch(
            "earth2studio.serve.server.cpu_worker.workflow_registry"
        ) as mock_registry:
            with patch(
                "earth2studio.serve.server.cpu_worker.redis_client"
            ) as mock_redis:
                with patch(
                    "earth2studio.serve.server.cpu_worker.queue_next_stage"
                ) as mock_queue_next:
                    mock_registry.get_workflow_class.return_value = mock_workflow_class
                    mock_redis.setex = Mock()
                    mock_queue_next.return_value = "job_123"

                    result = process_result_zip(
                        workflow_name="test_wf",
                        execution_id="exec_1",
                        output_path_str=str(output_file),
                        results_zip_dir_str=str(results_zip_dir),
                        create_zip=False,
                    )

        assert result == "manifest_only"

    def test_process_result_zip_create_fails_returns_none(self, tmp_path):
        """When create_results_zip returns None (create_zip=True), fail_workflow and return None."""
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()
        # Nonexistent output path so create_results_zip may still build manifest but we can mock return
        output_path_str = str(tmp_path / "nonexistent")

        mock_workflow_class = Mock()
        mock_workflow_class._get_execution_data.return_value = {
            "type": "deterministic",
            "status": "completed",
        }

        with patch(
            "earth2studio.serve.server.cpu_worker.workflow_registry"
        ) as mock_registry:
            with patch(
                "earth2studio.serve.server.cpu_worker.redis_client"
            ) as mock_redis:
                with patch("earth2studio.serve.server.cpu_worker.queue_next_stage"):
                    with patch(
                        "earth2studio.serve.server.cpu_worker.create_results_zip"
                    ) as mock_create:
                        with patch(
                            "earth2studio.serve.server.cpu_worker.fail_workflow"
                        ) as mock_fail:
                            mock_registry.get_workflow_class.return_value = (
                                mock_workflow_class
                            )
                            mock_redis.setex = Mock()
                            mock_create.return_value = None  # zip creation failed

                            result = process_result_zip(
                                workflow_name="test_wf",
                                execution_id="exec_1",
                                output_path_str=output_path_str,
                                results_zip_dir_str=str(results_zip_dir),
                                create_zip=True,
                            )

        assert result is None
        mock_fail.assert_called_once()
        assert "create results zip" in mock_fail.call_args[0][2].lower()


class TestProcessFinalizeMetadata:
    """Tests for process_finalize_metadata RQ worker function."""

    def test_process_finalize_metadata_success(self, tmp_path):
        """process_finalize_metadata writes metadata file and updates workflow status."""
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()
        request_id = "my_wf:exec_1"

        metadata_key = f"inference_request:{request_id}:pending_metadata"
        results_zip_dir_key = f"inference_request:{request_id}:results_zip_dir"
        storage_info_key = f"inference_request:{request_id}:storage_info"

        pending_metadata = {
            "request_id": request_id,
            "status": "completed",
            "output_files": [],
            "storage_type": "server",
        }
        mock_workflow_class = Mock()
        mock_workflow_class._update_execution_data = Mock()

        with patch("earth2studio.serve.server.cpu_worker.redis_client") as mock_redis:
            with patch(
                "earth2studio.serve.server.cpu_worker.workflow_registry"
            ) as mock_registry:
                mock_redis.get.side_effect = lambda k: {
                    metadata_key: json.dumps(pending_metadata),
                    results_zip_dir_key: str(results_zip_dir),
                    storage_info_key: None,
                }.get(k)
                mock_registry.get_workflow_class.return_value = mock_workflow_class

                result = process_finalize_metadata(
                    workflow_name="my_wf",
                    execution_id="exec_1",
                )

        assert result is not None
        assert result["success"] is True
        assert "metadata_path" in result
        metadata_path = tmp_path / "results" / f"metadata_{request_id}.json"
        assert metadata_path.exists()
        mock_workflow_class._update_execution_data.assert_called_once()
        mock_redis.delete.assert_any_call(metadata_key)
        mock_redis.delete.assert_any_call(results_zip_dir_key)

    def test_process_finalize_metadata_missing_pending_metadata_returns_failure(self):
        """When pending metadata or results_zip_dir is missing in Redis, returns fail_workflow dict."""
        with patch("earth2studio.serve.server.cpu_worker.redis_client") as mock_redis:
            with patch(
                "earth2studio.serve.server.cpu_worker.fail_workflow"
            ) as mock_fail:
                mock_redis.get.return_value = None
                mock_fail.return_value = {
                    "success": False,
                    "error": "Pending metadata not found",
                }

                result = process_finalize_metadata(
                    workflow_name="my_wf",
                    execution_id="exec_1",
                )

        assert result is not None and result.get("success") is False
        mock_fail.assert_called_once()
        assert "not found" in mock_fail.call_args[0][2].lower()

    def test_process_finalize_metadata_workflow_not_found_returns_failure(
        self, tmp_path
    ):
        """When workflow class is not in registry, returns fail_workflow dict."""
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()
        request_id = "my_wf:exec_1"
        metadata_key = f"inference_request:{request_id}:pending_metadata"
        results_zip_dir_key = f"inference_request:{request_id}:results_zip_dir"

        with patch("earth2studio.serve.server.cpu_worker.redis_client") as mock_redis:
            with patch(
                "earth2studio.serve.server.cpu_worker.workflow_registry"
            ) as mock_registry:
                with patch(
                    "earth2studio.serve.server.cpu_worker.fail_workflow"
                ) as mock_fail:
                    mock_redis.get.side_effect = lambda k: {
                        metadata_key: json.dumps({"request_id": request_id}),
                        results_zip_dir_key: str(results_zip_dir),
                    }.get(k)
                    mock_registry.get_workflow_class.return_value = None
                    mock_fail.return_value = {
                        "success": False,
                        "error": "Workflow class not found",
                    }

                    result = process_finalize_metadata(
                        workflow_name="my_wf",
                        execution_id="exec_1",
                    )

        assert result is not None and result.get("success") is False
        mock_fail.assert_called_once()
        assert "not found" in mock_fail.call_args[0][2].lower()

    def test_process_finalize_metadata_with_storage_info(self, tmp_path):
        """When storage_info is in Redis, it is applied to metadata and key is deleted."""
        results_zip_dir = tmp_path / "results"
        results_zip_dir.mkdir()
        request_id = "my_wf:exec_1"
        storage_info_key = f"inference_request:{request_id}:storage_info"
        pending_metadata = {"request_id": request_id, "status": "completed"}
        storage_info = {
            "storage_type": "s3",
            "remote_path": "s3://bucket/prefix",
            "signed_url": "https://signed.example.com",
        }

        mock_workflow_class = Mock()
        mock_workflow_class._update_execution_data = Mock()

        with patch("earth2studio.serve.server.cpu_worker.redis_client") as mock_redis:
            with patch(
                "earth2studio.serve.server.cpu_worker.workflow_registry"
            ) as mock_registry:
                mock_redis.get.side_effect = lambda k: {
                    f"inference_request:{request_id}:pending_metadata": json.dumps(
                        pending_metadata
                    ),
                    f"inference_request:{request_id}:results_zip_dir": str(
                        results_zip_dir
                    ),
                    storage_info_key: json.dumps(storage_info),
                }.get(k)
                mock_registry.get_workflow_class.return_value = mock_workflow_class

                result = process_finalize_metadata(
                    workflow_name="my_wf",
                    execution_id="exec_1",
                )

        assert result is not None
        assert result["storage_type"] == "s3"
        mock_redis.delete.assert_any_call(storage_info_key)
        metadata_path = tmp_path / "results" / f"metadata_{request_id}.json"
        data = json.loads(metadata_path.read_text())
        assert data.get("signed_url") == "https://signed.example.com"
        assert data.get("remote_path") == "s3://bucket/prefix"


class TestProcessObjectStorageUpload:
    """Tests for process_object_storage_upload RQ worker function."""

    def test_process_object_storage_upload_disabled_queues_finalize(self, tmp_path):
        """When object storage is disabled, skips upload and queues finalize_metadata."""
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "f.nc").write_text("data")

        with patch("earth2studio.serve.server.cpu_worker.redis_client") as mock_redis:
            with patch("earth2studio.serve.server.cpu_worker.config") as mock_config:
                with patch(
                    "earth2studio.serve.server.cpu_worker.queue_next_stage"
                ) as mock_queue_next:
                    mock_config.object_storage.enabled = False
                    mock_config.object_storage.bucket = None
                    mock_redis.setex = Mock()
                    mock_queue_next.return_value = "job_1"

                    result = process_object_storage_upload(
                        workflow_name="wf",
                        execution_id="exec_1",
                        output_path_str=str(output_dir),
                    )

        assert result is not None
        assert result["success"] is True
        assert result["storage_type"] == "server"
        mock_queue_next.assert_called_once_with(
            redis_client=mock_redis,
            current_stage="object_storage",
            workflow_name="wf",
            execution_id="exec_1",
            output_path_str=str(output_dir),
        )

    def test_process_object_storage_upload_output_path_missing_returns_failure(self):
        """When object storage enabled but output path does not exist, returns fail_workflow dict."""
        with patch("earth2studio.serve.server.cpu_worker.config") as mock_config:
            with patch(
                "earth2studio.serve.server.cpu_worker.fail_workflow"
            ) as mock_fail:
                mock_config.object_storage.enabled = True
                mock_config.object_storage.bucket = "my-bucket"
                mock_fail.return_value = {
                    "success": False,
                    "error": "Output path does not exist",
                }

                result = process_object_storage_upload(
                    workflow_name="wf",
                    execution_id="exec_1",
                    output_path_str="/nonexistent/path",
                )

        assert result is not None and result.get("success") is False
        mock_fail.assert_called_once()
        assert "does not exist" in mock_fail.call_args[0][2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

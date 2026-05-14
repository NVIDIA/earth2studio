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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException

from earth2studio.serve.server.utils import (
    create_file_stream,
    get_inference_request_metadata_key,
    get_inference_request_output_path_key,
    get_inference_request_zip_key,
    get_results_zip_dir_key,
    get_signed_url_key,
    parse_azure_blob_container_url,
    parse_range_header,
    queue_next_stage,
)


class TestParseRangeHeader:
    """Tests for parse_range_header."""

    def test_no_range_returns_full_file(self):
        start, end, length, status = parse_range_header(None, 100)
        assert (start, end, length, status) == (0, 99, 100, 200)

    def test_bytes_range_returns_206(self):
        start, end, length, status = parse_range_header("bytes=10-19", 100)
        assert start == 10 and end == 19 and length == 10 and status == 206

    def test_open_ended_range(self):
        start, end, length, status = parse_range_header("bytes=90-", 100)
        assert start == 90 and end == 99 and length == 10 and status == 206

    def test_suffix_range(self):
        start, end, length, status = parse_range_header("bytes=-5", 100)
        assert start == 95 and end == 99 and length == 5 and status == 206

    def test_invalid_range_raises_416(self):
        with pytest.raises((HTTPException, StarletteHTTPException)) as exc_info:
            parse_range_header("bytes=200-300", 100)
        assert exc_info.value.status_code == 416

    def test_non_bytes_range_unit_raises_416(self):
        with pytest.raises((HTTPException, StarletteHTTPException)) as exc_info:
            parse_range_header("items=0-1", 100)
        assert exc_info.value.status_code == 416

    def test_multiple_ranges_uses_first(self, caplog):
        import logging

        caplog.set_level(logging.WARNING)
        start, end, length, status = parse_range_header("bytes=0-1,10-15", 100)
        assert (start, end, length, status) == (0, 1, 2, 206)
        assert "Multiple ranges" in caplog.text

    def test_range_part_without_hyphen_raises_416(self):
        with pytest.raises((HTTPException, StarletteHTTPException)) as exc_info:
            parse_range_header("bytes=invalid", 100)
        assert exc_info.value.status_code == 416


class TestParseAzureBlobContainerUrl:
    """Tests for parse_azure_blob_container_url."""

    def test_valid_container_url(self):
        account, container = parse_azure_blob_container_url(
            "https://mystorage.blob.core.windows.net/mycontainer"
        )
        assert account == "mystorage"
        assert container == "mycontainer"

    def test_strips_whitespace(self):
        account, container = parse_azure_blob_container_url(
            "  https://acct.blob.core.windows.net/c1  "
        )
        assert account == "acct" and container == "c1"

    def test_extra_path_segments_uses_first_path_component_as_container(self):
        account, container = parse_azure_blob_container_url(
            "https://a.blob.core.windows.net/c1/blob/path"
        )
        assert account == "a" and container == "c1"

    def test_http_scheme_rejected(self):
        with pytest.raises(ValueError, match="https"):
            parse_azure_blob_container_url("http://a.blob.core.windows.net/c1")

    def test_non_azure_host_rejected(self):
        with pytest.raises(ValueError, match="blob.core.windows.net"):
            parse_azure_blob_container_url("https://example.com/c1")

    def test_missing_container_in_path_rejected(self):
        with pytest.raises(ValueError, match="container name"):
            parse_azure_blob_container_url("https://onlyacct.blob.core.windows.net/")


class TestCreateFileStream:
    """Tests for create_file_stream."""

    @pytest.mark.asyncio
    async def test_streams_full_file(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"abcdefghij")
        chunks = [part async for part in create_file_stream(p, 0, 10, "test file")]
        assert b"".join(chunks) == b"abcdefghij"

    @pytest.mark.asyncio
    async def test_skips_start_bytes_then_reads(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"0123456789")
        chunks = [part async for part in create_file_stream(p, 3, 4, "test file")]
        assert b"".join(chunks) == b"3456"

    @pytest.mark.asyncio
    async def test_stops_early_if_file_shorter_than_content_length(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"ab")
        chunks = [part async for part in create_file_stream(p, 0, 1000, "test file")]
        assert b"".join(chunks) == b"ab"

    @pytest.mark.asyncio
    async def test_skip_past_eof_yields_nothing(self, tmp_path):
        """Skip loop exits early when read returns empty before start offset."""
        p = tmp_path / "f.bin"
        p.write_bytes(b"ab")
        chunks = [part async for part in create_file_stream(p, 100, 10, "test file")]
        assert chunks == []

    @pytest.mark.asyncio
    async def test_open_error_propagates(self, tmp_path):
        missing = tmp_path / "nope.bin"
        with pytest.raises(OSError):
            async for _ in create_file_stream(missing, 0, 1, "missing"):
                pass

    @pytest.mark.asyncio
    async def test_read_error_logs_and_propagates(self, tmp_path, caplog):
        """Exception during read hits logger.exception and re-raises."""
        import logging

        p = tmp_path / "f.bin"
        p.write_bytes(b"x" * 100)
        mock_file = MagicMock()
        mock_file.read = AsyncMock(side_effect=RuntimeError("read failed"))
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_file)
        mock_cm.__aexit__ = AsyncMock(return_value=None)

        caplog.set_level(logging.ERROR, logger="earth2studio.serve.server.utils")
        with patch(
            "earth2studio.serve.server.utils.aiofiles.open", return_value=mock_cm
        ):
            with pytest.raises(RuntimeError, match="read failed"):
                async for _ in create_file_stream(p, 0, 10, "desc"):
                    pass
        assert "Error streaming desc" in caplog.text


class TestRedisKeyFunctions:
    """Tests for Redis key helper functions."""

    def test_get_inference_request_zip_key(self):
        """get_inference_request_zip_key returns expected key format."""
        assert get_inference_request_zip_key("req_1") == (
            "inference_request_zips:req_1:zip_file"
        )

    def test_get_inference_request_output_path_key(self):
        """get_inference_request_output_path_key returns expected key format."""
        assert get_inference_request_output_path_key("req_1") == (
            "inference_request:req_1:output_path"
        )

    def test_get_inference_request_metadata_key(self):
        """get_inference_request_metadata_key returns expected key format."""
        assert get_inference_request_metadata_key("req_1") == (
            "inference_request:req_1:pending_metadata"
        )

    def test_get_results_zip_dir_key(self):
        """get_results_zip_dir_key returns expected key format."""
        assert get_results_zip_dir_key("req_1") == (
            "inference_request:req_1:results_zip_dir"
        )

    def test_get_signed_url_key(self):
        """get_signed_url_key returns expected key format."""
        assert get_signed_url_key("req_1") == ("inference_request:req_1:signed_url")


class TestQueueNextStage:
    """Tests for queue_next_stage."""

    def test_unknown_stage_returns_none(self):
        """Unknown current_stage logs error and returns None."""
        mock_redis = MagicMock()
        result = queue_next_stage(
            redis_client=mock_redis,
            current_stage="unknown_stage",  # type: ignore[arg-type]
            workflow_name="wf",
            execution_id="exec_1",
            output_path_str="/path",
        )
        assert result is None

    def test_inference_stage_queues_result_zip(self):
        """current_stage=inference enqueues process_result_zip to result_zip queue."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.paths.result_zip_enabled = True
        mock_config.object_storage.enabled = False
        mock_config.queue.result_zip_queue_name = "result_zip"
        mock_config.queue.default_timeout = "1h"
        mock_config.queue.job_timeout = "2h"
        mock_job = MagicMock()
        mock_job.id = "job_123"
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_job

        with (
            patch(
                "earth2studio.serve.server.utils.get_config", return_value=mock_config
            ),
            patch("earth2studio.serve.server.utils.Queue", return_value=mock_queue),
        ):
            result = queue_next_stage(
                redis_client=mock_redis,
                current_stage="inference",
                workflow_name="wf",
                execution_id="exec_1",
                output_path_str="/out",
                results_zip_dir_str="/zip",
            )

        assert result == "job_123"
        mock_queue.enqueue.assert_called_once()
        call_kw = mock_queue.enqueue.call_args
        assert (
            call_kw[0][0] == "earth2studio.serve.server.cpu_worker.process_result_zip"
        )
        assert call_kw[0][1:5] == ("wf", "exec_1", "/out", "/zip")
        assert call_kw[0][5] is True  # create_zip

    def test_inference_stage_create_zip_false_when_object_storage_enabled(self):
        """When object_storage enabled, create_zip is False for inference stage."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.paths.result_zip_enabled = True
        mock_config.object_storage.enabled = True
        mock_config.queue.result_zip_queue_name = "result_zip"
        mock_config.queue.default_timeout = "1h"
        mock_config.queue.job_timeout = "2h"
        mock_job = MagicMock()
        mock_job.id = "job_123"
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_job

        with (
            patch(
                "earth2studio.serve.server.utils.get_config", return_value=mock_config
            ),
            patch("earth2studio.serve.server.utils.Queue", return_value=mock_queue),
        ):
            queue_next_stage(
                redis_client=mock_redis,
                current_stage="inference",
                workflow_name="wf",
                execution_id="exec_1",
                output_path_str="/out",
                results_zip_dir_str="/zip",
            )

        assert mock_queue.enqueue.call_args[0][5] is False  # create_zip

    def test_result_zip_stage_object_storage_enabled_queues_object_storage(self):
        """current_stage=result_zip with object_storage enabled enqueues process_object_storage_upload."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.object_storage.enabled = True
        mock_config.queue.object_storage_queue_name = "object_storage"
        mock_config.queue.default_timeout = "1h"
        mock_config.queue.job_timeout = "2h"
        mock_job = MagicMock()
        mock_job.id = "job_456"
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_job

        with (
            patch(
                "earth2studio.serve.server.utils.get_config", return_value=mock_config
            ),
            patch("earth2studio.serve.server.utils.Queue", return_value=mock_queue),
        ):
            result = queue_next_stage(
                redis_client=mock_redis,
                current_stage="result_zip",
                workflow_name="wf",
                execution_id="exec_1",
                output_path_str="/out",
            )

        assert result == "job_456"
        mock_queue.enqueue.assert_called_once()
        assert "process_object_storage_upload" in mock_queue.enqueue.call_args[0][0]
        assert mock_queue.enqueue.call_args[0][1:4] == ("wf", "exec_1", "/out")

    def test_result_zip_stage_object_storage_disabled_queues_finalize_metadata(self):
        """current_stage=result_zip with object_storage disabled enqueues process_finalize_metadata."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.object_storage.enabled = False
        mock_config.queue.finalize_metadata_queue_name = "finalize_metadata"
        mock_config.queue.default_timeout = "1h"
        mock_config.queue.job_timeout = "2h"
        mock_job = MagicMock()
        mock_job.id = "job_789"
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_job

        with (
            patch(
                "earth2studio.serve.server.utils.get_config", return_value=mock_config
            ),
            patch("earth2studio.serve.server.utils.Queue", return_value=mock_queue),
        ):
            result = queue_next_stage(
                redis_client=mock_redis,
                current_stage="result_zip",
                workflow_name="wf",
                execution_id="exec_1",
                output_path_str="/out",
            )

        assert result == "job_789"
        mock_queue.enqueue.assert_called_once()
        assert "process_finalize_metadata" in mock_queue.enqueue.call_args[0][0]
        assert mock_queue.enqueue.call_args[0][1:3] == ("wf", "exec_1")

    def test_object_storage_stage_queues_geocatalog(self):
        """current_stage=object_storage with Azure storage_type enqueues GeoCatalog ingestion."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.object_storage.storage_type = "azure"
        mock_config.queue.geocatalog_ingestion_queue_name = "geocatalog_ingestion"
        mock_config.queue.default_timeout = "1h"
        mock_config.queue.job_timeout = "2h"
        mock_job = MagicMock()
        mock_job.id = "job_geo"
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_job

        with (
            patch(
                "earth2studio.serve.server.utils.get_config", return_value=mock_config
            ),
            patch("earth2studio.serve.server.utils.Queue", return_value=mock_queue),
        ):
            result = queue_next_stage(
                redis_client=mock_redis,
                current_stage="object_storage",
                workflow_name="wf",
                execution_id="exec_1",
                output_path_str="/out",
            )

        assert result == "job_geo"
        mock_queue.enqueue.assert_called_once()
        assert (
            "azure_planetary_computer.geocatalog_ingestion.process_geocatalog_ingestion"
            in mock_queue.enqueue.call_args[0][0]
        )
        assert mock_queue.enqueue.call_args[0][1:3] == ("wf", "exec_1")

    def test_object_storage_stage_queues_finalize_metadata(self):
        """current_stage=object_storage enqueues finalize when storage_type is not Azure."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.object_storage.storage_type = "s3"
        mock_config.queue.finalize_metadata_queue_name = "finalize_metadata"
        mock_config.queue.default_timeout = "1h"
        mock_config.queue.job_timeout = "2h"
        mock_job = MagicMock()
        mock_job.id = "job_final"
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_job

        with (
            patch(
                "earth2studio.serve.server.utils.get_config", return_value=mock_config
            ),
            patch("earth2studio.serve.server.utils.Queue", return_value=mock_queue),
        ):
            result = queue_next_stage(
                redis_client=mock_redis,
                current_stage="object_storage",
                workflow_name="wf",
                execution_id="exec_1",
                output_path_str="/out",
            )

        assert result == "job_final"
        mock_queue.enqueue.assert_called_once()
        assert "process_finalize_metadata" in mock_queue.enqueue.call_args[0][0]

    def test_geocatalog_ingestion_stage_queues_finalize_metadata(self):
        """current_stage=geocatalog_ingestion enqueues process_finalize_metadata."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.queue.finalize_metadata_queue_name = "finalize_metadata"
        mock_config.queue.default_timeout = "1h"
        mock_config.queue.job_timeout = "2h"
        mock_job = MagicMock()
        mock_job.id = "job_finalize"
        mock_queue = MagicMock()
        mock_queue.enqueue.return_value = mock_job

        with (
            patch(
                "earth2studio.serve.server.utils.get_config", return_value=mock_config
            ),
            patch("earth2studio.serve.server.utils.Queue", return_value=mock_queue),
        ):
            result = queue_next_stage(
                redis_client=mock_redis,
                current_stage="geocatalog_ingestion",
                workflow_name="wf",
                execution_id="exec_1",
                output_path_str="/out",
            )

        assert result == "job_finalize"
        mock_queue.enqueue.assert_called_once()
        assert "process_finalize_metadata" in mock_queue.enqueue.call_args[0][0]
        assert mock_queue.enqueue.call_args[0][1:3] == ("wf", "exec_1")

    def test_enqueue_exception_returns_none(self):
        """When Queue.enqueue raises, queue_next_stage returns None."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.paths.result_zip_enabled = False
        mock_config.object_storage.enabled = False
        mock_config.queue.result_zip_queue_name = "result_zip"
        mock_config.queue.default_timeout = "1h"
        mock_config.queue.job_timeout = "2h"
        mock_queue = MagicMock()
        mock_queue.enqueue.side_effect = Exception("RQ error")

        with (
            patch(
                "earth2studio.serve.server.utils.get_config", return_value=mock_config
            ),
            patch("earth2studio.serve.server.utils.Queue", return_value=mock_queue),
        ):
            result = queue_next_stage(
                redis_client=mock_redis,
                current_stage="inference",
                workflow_name="wf",
                execution_id="exec_1",
                output_path_str="/out",
                results_zip_dir_str="/zip",
            )

        assert result is None

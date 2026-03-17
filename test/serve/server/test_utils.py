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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from earth2studio.serve.server.utils import (
    create_file_stream,
    get_inference_request_metadata_key,
    get_inference_request_output_path_key,
    get_inference_request_zip_key,
    get_results_zip_dir_key,
    get_signed_url_key,
    parse_range_header,
    queue_next_stage,
)


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

    def test_object_storage_stage_queues_finalize_metadata(self):
        """current_stage=object_storage enqueues process_finalize_metadata when geocatalog is not configured."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.object_storage.azure_geocatalog_url = None
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


class TestParseRangeHeader:
    """Tests for parse_range_header."""

    def test_no_range_header_returns_full_file(self):
        """None range_header returns full file range with status 200."""
        start, end, content_length, status_code = parse_range_header(None, 1000)
        assert start == 0
        assert end == 999
        assert content_length == 1000
        assert status_code == 200

    def test_explicit_range_returns_partial_content(self):
        """bytes=0-99 returns first 100 bytes with status 206."""
        start, end, content_length, status_code = parse_range_header("bytes=0-99", 1000)
        assert start == 0
        assert end == 99
        assert content_length == 100
        assert status_code == 206

    def test_open_ended_range(self):
        """bytes=500- returns from byte 500 to end of file."""
        start, end, content_length, status_code = parse_range_header("bytes=500-", 1000)
        assert start == 500
        assert end == 999
        assert content_length == 500
        assert status_code == 206

    def test_suffix_range(self):
        """bytes=-200 returns last 200 bytes of file."""
        start, end, content_length, status_code = parse_range_header("bytes=-200", 1000)
        assert start == 800
        assert end == 999
        assert content_length == 200
        assert status_code == 206

    def test_suffix_range_larger_than_file(self):
        """bytes=-2000 on a 1000-byte file returns entire file from byte 0."""
        start, end, content_length, status_code = parse_range_header(
            "bytes=-2000", 1000
        )
        assert start == 0
        assert end == 999
        assert content_length == 1000
        assert status_code == 206

    def test_multiple_ranges_uses_first(self):
        """Multiple ranges are accepted; only the first range is used."""
        start, end, content_length, status_code = parse_range_header(
            "bytes=0-99,200-299", 1000
        )
        assert start == 0
        assert end == 99
        assert content_length == 100
        assert status_code == 206

    def test_non_bytes_unit_raises_416(self):
        """Range header not starting with 'bytes=' raises HTTPException 416."""
        with pytest.raises(HTTPException) as exc_info:
            parse_range_header("items=0-99", 1000)
        assert exc_info.value.status_code == 416

    def test_missing_dash_raises_416(self):
        """Range spec without a dash raises HTTPException 416."""
        with pytest.raises(HTTPException) as exc_info:
            parse_range_header("bytes=100", 1000)
        assert exc_info.value.status_code == 416

    def test_start_beyond_file_size_raises_416(self):
        """start >= file_size raises HTTPException 416."""
        with pytest.raises(HTTPException) as exc_info:
            parse_range_header("bytes=1000-1099", 1000)
        assert exc_info.value.status_code == 416

    def test_end_beyond_file_size_clamped_to_last_byte(self):
        """end >= file_size is clamped to file_size-1 per RFC 9110 §14.1.2, returns 206."""
        start, end, content_length, status_code = parse_range_header(
            "bytes=0-1000", 1000
        )
        assert start == 0
        assert end == 999
        assert content_length == 1000
        assert status_code == 206

    def test_end_before_start_raises_416(self):
        """end < start raises HTTPException 416."""
        with pytest.raises(HTTPException) as exc_info:
            parse_range_header("bytes=200-100", 1000)
        assert exc_info.value.status_code == 416


class TestCreateFileStream:
    """Tests for create_file_stream."""

    @pytest.mark.asyncio
    async def test_streams_full_file(self):
        """Streams entire file content when start=0."""
        file_data = b"hello world"
        mock_file = AsyncMock()
        mock_file.read.side_effect = [file_data, b""]
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "earth2studio.serve.server.utils.aiofiles.open", return_value=mock_file
        ):
            chunks = [
                chunk
                async for chunk in create_file_stream(
                    Path("/fake/file.bin"), 0, len(file_data)
                )
            ]

        assert b"".join(chunks) == file_data
        mock_file.seek.assert_not_called()

    @pytest.mark.asyncio
    async def test_seeks_to_start_for_range_request(self):
        """Seeks to the start offset for range requests (start > 0)."""
        file_data = b"partial content"
        mock_file = AsyncMock()
        mock_file.read.side_effect = [file_data, b""]
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "earth2studio.serve.server.utils.aiofiles.open", return_value=mock_file
        ):
            chunks = [
                chunk
                async for chunk in create_file_stream(
                    Path("/fake/file.bin"), 512, len(file_data)
                )
            ]

        mock_file.seek.assert_called_once_with(512)
        assert b"".join(chunks) == file_data

    @pytest.mark.asyncio
    async def test_streams_in_multiple_chunks(self):
        """Yields multiple chunks when file exceeds chunk size."""
        chunk_size = 1048576  # 1MB
        chunk1 = b"A" * chunk_size
        chunk2 = b"B" * 512
        mock_file = AsyncMock()
        mock_file.read.side_effect = [chunk1, chunk2, b""]
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=False)

        content_length = chunk_size + 512
        with patch(
            "earth2studio.serve.server.utils.aiofiles.open", return_value=mock_file
        ):
            chunks = [
                chunk
                async for chunk in create_file_stream(
                    Path("/fake/file.bin"), 0, content_length
                )
            ]

        assert len(chunks) == 2
        assert chunks[0] == chunk1
        assert chunks[1] == chunk2

    @pytest.mark.asyncio
    async def test_stops_when_chunk_is_empty(self):
        """Stops streaming early if read returns empty bytes before content_length is reached."""
        mock_file = AsyncMock()
        mock_file.read.return_value = b""
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "earth2studio.serve.server.utils.aiofiles.open", return_value=mock_file
        ):
            chunks = [
                chunk
                async for chunk in create_file_stream(Path("/fake/file.bin"), 0, 1000)
            ]

        assert chunks == []

    @pytest.mark.asyncio
    async def test_reraises_exception_on_read_error(self):
        """Re-raises exceptions encountered during file streaming."""
        mock_file = AsyncMock()
        mock_file.read.side_effect = OSError("disk error")
        mock_file.__aenter__ = AsyncMock(return_value=mock_file)
        mock_file.__aexit__ = AsyncMock(return_value=False)

        with patch(
            "earth2studio.serve.server.utils.aiofiles.open", return_value=mock_file
        ):
            with pytest.raises(OSError, match="disk error"):
                async for _ in create_file_stream(Path("/fake/file.bin"), 0, 1000):
                    pass

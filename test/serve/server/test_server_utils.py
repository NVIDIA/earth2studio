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

from unittest.mock import MagicMock, patch

from earth2studio.serve.server.utils import (
    get_inference_request_metadata_key,
    get_inference_request_output_path_key,
    get_inference_request_zip_key,
    get_results_zip_dir_key,
    get_signed_url_key,
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

    def test_object_storage_stage_with_geocatalog_url_queues_geocatalog(self):
        """current_stage=object_storage with geocatalog URL enqueues process_geocatalog_ingestion."""
        mock_redis = MagicMock()
        mock_config = MagicMock()
        mock_config.object_storage.azure_geocatalog_url = (
            "https://geocatalog.example.com"
        )
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
        assert "process_geocatalog_ingestion" in mock_queue.enqueue.call_args[0][0]
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

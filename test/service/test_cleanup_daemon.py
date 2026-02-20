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
Unit tests for the cleanup daemon.

This module tests the cleanup daemon functionality with mocked Redis and filesystem operations.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from api_server.cleanup_daemon import (
    _delete_result_files,
    _process_expired_key,
    cleanup_expired_results,
)
from api_server.config import get_config
from api_server.workflow import WorkflowStatus


class TestDeleteResultFiles:
    """Test suite for _delete_result_files function"""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client"""
        return MagicMock()

    def test_delete_workflow_files_with_workflow_name(self, mock_redis):
        """Test deletion of workflow execution files including workflow-specific directories and metadata"""
        # Setup
        workflow_name = "custom_workflow"
        execution_id = "exec_456"
        combined_id = f"{workflow_name}:{execution_id}"
        zip_filename = f"workflow_{combined_id}.zip"

        mock_redis.get.return_value = zip_filename

        # Mock zip file
        mock_zip_file = MagicMock()
        mock_zip_file.exists.return_value = True
        mock_zip_file.unlink = MagicMock()

        # Mock metadata file for workflow
        mock_metadata_file = MagicMock()
        mock_metadata_file.is_file.return_value = True
        mock_metadata_file.is_dir.return_value = False
        mock_metadata_file.name = f"metadata_{combined_id}.json"
        mock_metadata_file.unlink = MagicMock()

        # Mock workflow directory
        mock_workflow_dir = MagicMock()
        mock_workflow_dir.exists.return_value = True

        mock_workflow_subdir = MagicMock()
        mock_workflow_subdir.is_dir.return_value = True
        mock_workflow_subdir.is_file.return_value = False
        mock_workflow_subdir.name = f"output_{execution_id}"

        mock_workflow_dir.iterdir.return_value = [mock_workflow_subdir]

        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []
        mock_output_dir.__truediv__ = lambda self, other: mock_workflow_dir

        mock_zip_dir = MagicMock()
        mock_zip_dir.__truediv__ = lambda self, other: mock_zip_file
        mock_zip_dir.iterdir.return_value = [mock_metadata_file]

        with (
            patch("api_server.cleanup_daemon.RESULTS_ZIP_DIR", mock_zip_dir),
            patch(
                "api_server.cleanup_daemon.DEFAULT_OUTPUT_DIR",
                mock_output_dir,
            ),
            patch("api_server.cleanup_daemon.shutil.rmtree") as mock_rmtree,
        ):
            # Execute
            _delete_result_files(
                mock_redis, combined_id, execution_id, workflow_name=workflow_name
            )

            # Verify zip file deleted
            mock_zip_file.unlink.assert_called_once()

            # Verify metadata file deleted
            mock_metadata_file.unlink.assert_called_once()

            # Verify workflow-specific directory deleted
            mock_rmtree.assert_called_once_with(mock_workflow_subdir)

    def test_delete_files_no_zip_in_redis(self, mock_redis):
        """Test deletion when no zip file is found in Redis"""
        # Setup
        result_id = "result_789"
        mock_redis.get.return_value = None

        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []

        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = []

        with (
            patch("api_server.cleanup_daemon.RESULTS_ZIP_DIR", mock_zip_dir),
            patch(
                "api_server.cleanup_daemon.DEFAULT_OUTPUT_DIR",
                mock_output_dir,
            ),
        ):
            # Execute - should not raise exception
            _delete_result_files(mock_redis, result_id, result_id)

            # Verify Redis was queried
            mock_redis.get.assert_called_once()


class TestProcessExpiredKey:
    """Test suite for _process_expired_key function"""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client"""
        return MagicMock()

    @pytest.fixture
    def mock_delete_func(self):
        """Create a mock delete function"""
        return MagicMock()

    def test_process_expired_key_success(self, mock_redis, mock_delete_func):
        """Test successful processing of an expired key"""
        # Setup
        key = "workflow_execution:wf1:exec_123"
        current_time = datetime.now(timezone.utc)
        old_time = current_time - timedelta(hours=25)  # 25 hours ago
        results_ttl_hours = 24

        request_data = {
            "status": WorkflowStatus.COMPLETED,
            "end_time": old_time.isoformat(),
            "execution_id": "exec_123",
        }

        mock_redis.get.return_value = json.dumps(request_data)
        config = get_config()

        # Execute
        result = _process_expired_key(
            redis_client=mock_redis,
            key=key,
            current_time=current_time,
            results_ttl_hours=results_ttl_hours,
            expected_status=WorkflowStatus.COMPLETED,
            get_end_time_field="end_time",
            delete_files_func=mock_delete_func,
            log_prefix="workflow",
        )

        # Verify
        assert result is True
        mock_delete_func.assert_called_once_with(mock_redis, key)
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == key
        assert args[1] == config.redis.retention_ttl
        updated_data = json.loads(args[2])
        assert updated_data["status"] == WorkflowStatus.EXPIRED

    def test_process_expired_key_not_expired(self, mock_redis, mock_delete_func):
        """Test processing of a key that is not yet expired"""
        # Setup
        key = "workflow_execution:wf1:exec_456"
        current_time = datetime.now(timezone.utc)
        recent_time = current_time - timedelta(hours=12)  # 12 hours ago
        results_ttl_hours = 24

        request_data = {
            "status": WorkflowStatus.COMPLETED,
            "end_time": recent_time.isoformat(),
            "execution_id": "exec_456",
        }

        mock_redis.get.return_value = json.dumps(request_data)

        # Execute
        result = _process_expired_key(
            redis_client=mock_redis,
            key=key,
            current_time=current_time,
            results_ttl_hours=results_ttl_hours,
            expected_status=WorkflowStatus.COMPLETED,
            get_end_time_field="end_time",
            delete_files_func=mock_delete_func,
            log_prefix="workflow",
        )

        # Verify - should not clean up
        assert result is False
        mock_delete_func.assert_not_called()
        mock_redis.setex.assert_not_called()


class TestCleanupExpiredResults:
    """Test suite for cleanup_expired_results function"""

    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client"""
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        """Mock the config"""
        with patch("api_server.cleanup_daemon.config") as mock_cfg:
            mock_cfg.server.results_ttl_hours = 24
            mock_cfg.redis.retention_ttl = 604800
            yield mock_cfg

    def test_cleanup_expired_results_workflow_only(self, mock_redis, mock_config):
        """Test cleanup with workflow executions only"""
        # Setup
        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=30)).isoformat()
        recent_time = (current_time - timedelta(hours=12)).isoformat()

        # Mock Redis keys - only workflow keys
        mock_redis.keys.return_value = [
            "workflow_execution:wf1:exec_old",
            "workflow_execution:wf2:exec_recent",
        ]

        # Mock Redis get responses
        def get_side_effect(key):
            if key == "workflow_execution:wf1:exec_old":
                return json.dumps(
                    {"status": WorkflowStatus.COMPLETED, "end_time": old_time}
                )
            elif key == "workflow_execution:wf2:exec_recent":
                return json.dumps(
                    {"status": WorkflowStatus.COMPLETED, "end_time": recent_time}
                )
            return None

        mock_redis.get.side_effect = get_side_effect

        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []
        mock_output_dir.__truediv__ = lambda self, other: MagicMock(
            exists=lambda: False
        )

        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = []

        with (
            patch("api_server.cleanup_daemon._delete_result_files"),
            patch(
                "api_server.cleanup_daemon.DEFAULT_OUTPUT_DIR",
                mock_output_dir,
            ),
            patch("api_server.cleanup_daemon.RESULTS_ZIP_DIR", mock_zip_dir),
        ):
            # Execute
            cleanup_expired_results(mock_redis)

            # Verify - should process 1 workflow (exec_old), skip exec_recent
            assert mock_redis.setex.call_count == 1

            # Verify the expired key was updated
            setex_calls = mock_redis.setex.call_args_list
            expired_keys = [call[0][0] for call in setex_calls]
            assert "workflow_execution:wf1:exec_old" in expired_keys

    def test_cleanup_expired_results_no_keys(self, mock_redis, mock_config):
        """Test cleanup when there are no keys in Redis"""
        # Setup
        mock_redis.keys.return_value = []

        # Execute
        cleanup_expired_results(mock_redis)

        # Verify - no cleanup operations
        mock_redis.setex.assert_not_called()

    def test_cleanup_expired_results_all_recent(self, mock_redis, mock_config):
        """Test cleanup when all workflow keys are recent (not expired)"""
        # Setup
        current_time = datetime.now(timezone.utc)
        recent_time = (current_time - timedelta(hours=5)).isoformat()

        mock_redis.keys.return_value = [
            "workflow_execution:wf1:exec_1",
            "workflow_execution:wf2:exec_2",
        ]

        mock_redis.get.return_value = json.dumps(
            {"status": WorkflowStatus.COMPLETED, "end_time": recent_time}
        )

        # Execute
        cleanup_expired_results(mock_redis)

        # Verify - no cleanup operations (not expired)
        mock_redis.setex.assert_not_called()

    def test_cleanup_expired_results_redis_get_failure(self, mock_redis, mock_config):
        """Test cleanup handles Redis get failures gracefully"""
        # Setup
        mock_redis.keys.return_value = [
            "workflow_execution:wf_fail:exec_1",
            "workflow_execution:wf_ok:exec_2",
        ]

        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=30)).isoformat()

        # First key causes exception, second succeeds
        def get_side_effect(key):
            if key == "workflow_execution:wf_fail:exec_1":
                raise Exception("Redis connection error")
            elif key == "workflow_execution:wf_ok:exec_2":
                return json.dumps(
                    {"status": WorkflowStatus.COMPLETED, "end_time": old_time}
                )
            return None

        mock_redis.get.side_effect = get_side_effect

        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []
        mock_output_dir.__truediv__ = lambda self, other: MagicMock(
            exists=lambda: False
        )

        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = []

        with (
            patch("api_server.cleanup_daemon._delete_result_files"),
            patch(
                "api_server.cleanup_daemon.DEFAULT_OUTPUT_DIR",
                mock_output_dir,
            ),
            patch("api_server.cleanup_daemon.RESULTS_ZIP_DIR", mock_zip_dir),
        ):
            # Execute - should not raise exception
            cleanup_expired_results(mock_redis)

            # Verify - exec_2 should still be processed despite exec_1 error
            assert mock_redis.setex.call_count == 1
            assert mock_redis.setex.call_args[0][0] == "workflow_execution:wf_ok:exec_2"

    def test_cleanup_expired_results_delete_files_failure(
        self, mock_redis, mock_config
    ):
        """Test cleanup handles file deletion failures gracefully and does not mark as expired"""
        # Setup
        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=30)).isoformat()

        mock_redis.keys.return_value = ["workflow_execution:wf_fail:exec_fail"]

        mock_redis.get.return_value = json.dumps(
            {"status": WorkflowStatus.COMPLETED, "end_time": old_time}
        )

        with patch("api_server.cleanup_daemon._delete_result_files") as mock_delete:
            mock_delete.side_effect = Exception("File deletion error")

            # Execute - should handle exception and continue
            cleanup_expired_results(mock_redis)

            # Verify delete was attempted
            mock_delete.assert_called_once()

            # Status should NOT be updated if file deletion fails
            # (we want to retry cleanup on next run)
            assert mock_redis.setex.call_count == 0

    def test_cleanup_expired_results_workflow_execution(self, mock_redis, mock_config):
        """Test cleanup of workflow execution keys"""
        # Setup
        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=30)).isoformat()

        mock_redis.keys.return_value = ["workflow_execution:my_workflow:exec_123"]

        mock_redis.get.return_value = json.dumps(
            {"status": WorkflowStatus.COMPLETED, "end_time": old_time}
        )

        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []
        mock_output_dir.__truediv__ = lambda self, other: MagicMock(
            exists=lambda: False
        )

        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = []

        with (
            patch("api_server.cleanup_daemon._delete_result_files") as mock_delete,
            patch(
                "api_server.cleanup_daemon.DEFAULT_OUTPUT_DIR",
                mock_output_dir,
            ),
            patch("api_server.cleanup_daemon.RESULTS_ZIP_DIR", mock_zip_dir),
        ):
            # Execute
            cleanup_expired_results(mock_redis)

            # Verify workflow was cleaned up
            assert mock_redis.setex.call_count == 1
            assert (
                mock_redis.setex.call_args[0][0]
                == "workflow_execution:my_workflow:exec_123"
            )

            # Verify delete was called with correct parameters
            mock_delete.assert_called_once()
            call_args = mock_delete.call_args
            assert call_args[1]["result_id"] == "my_workflow:exec_123"
            assert call_args[1]["search_id"] == "exec_123"
            assert call_args[1]["workflow_name"] == "my_workflow"

    def test_cleanup_expired_results_custom_ttl_config(self, mock_redis):
        """Test cleanup with custom TTL configuration"""
        # Setup custom config
        with patch("api_server.cleanup_daemon.config") as mock_cfg:
            mock_cfg.server.results_ttl_hours = 48  # Custom 48 hour TTL
            mock_cfg.redis.retention_ttl = 604800

            current_time = datetime.now(timezone.utc)
            # 30 hours ago - should NOT be expired with 48 hour TTL
            time_30h_ago = (current_time - timedelta(hours=30)).isoformat()
            # 50 hours ago - should be expired with 48 hour TTL
            time_50h_ago = (current_time - timedelta(hours=50)).isoformat()

            mock_redis.keys.return_value = [
                "workflow_execution:wf_30h:exec_30h",
                "workflow_execution:wf_50h:exec_50h",
            ]

            def get_side_effect(key):
                if key == "workflow_execution:wf_30h:exec_30h":
                    return json.dumps(
                        {"status": WorkflowStatus.COMPLETED, "end_time": time_30h_ago}
                    )
                elif key == "workflow_execution:wf_50h:exec_50h":
                    return json.dumps(
                        {"status": WorkflowStatus.COMPLETED, "end_time": time_50h_ago}
                    )
                return None

            mock_redis.get.side_effect = get_side_effect

            mock_output_dir = MagicMock()
            mock_output_dir.iterdir.return_value = []
            mock_output_dir.__truediv__ = lambda self, other: MagicMock(
                exists=lambda: False
            )

            mock_zip_dir = MagicMock()
            mock_zip_dir.iterdir.return_value = []

            with (
                patch("api_server.cleanup_daemon._delete_result_files"),
                patch(
                    "api_server.cleanup_daemon.DEFAULT_OUTPUT_DIR",
                    mock_output_dir,
                ),
                patch(
                    "api_server.cleanup_daemon.RESULTS_ZIP_DIR",
                    mock_zip_dir,
                ),
            ):
                # Execute
                cleanup_expired_results(mock_redis)

                # Verify - only exec_50h should be cleaned up
                assert mock_redis.setex.call_count == 1
                assert (
                    mock_redis.setex.call_args[0][0]
                    == "workflow_execution:wf_50h:exec_50h"
                )

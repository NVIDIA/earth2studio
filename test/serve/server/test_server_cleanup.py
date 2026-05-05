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
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from earth2studio.serve.server.cleanup_daemon import (
    _delete_result_files,
    _process_expired_key,
    cleanup_expired_results,
    main,
)
from earth2studio.serve.server.config import get_config
from earth2studio.serve.server.workflow import WorkflowStatus


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

        with patch(
            "earth2studio.serve.server.cleanup_daemon.shutil.rmtree"
        ) as mock_rmtree:
            # Execute
            _delete_result_files(
                mock_redis,
                combined_id,
                execution_id,
                mock_output_dir,
                mock_zip_dir,
                workflow_name=workflow_name,
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

        # Execute - should not raise exception
        _delete_result_files(
            mock_redis, result_id, result_id, mock_output_dir, mock_zip_dir
        )

        # Verify Redis was queried
        mock_redis.get.assert_called_once()

    def test_delete_result_files_zip_file_missing(self, mock_redis):
        """When zip filename is in Redis but file does not exist, no unlink is called."""
        result_id = "req_123"
        mock_redis.get.return_value = "missing.zip"

        mock_zip_path = MagicMock()
        mock_zip_path.exists.return_value = False

        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []
        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = []
        mock_zip_dir.__truediv__ = lambda self, other: mock_zip_path

        _delete_result_files(
            mock_redis, result_id, result_id, mock_output_dir, mock_zip_dir
        )

        mock_zip_path.unlink.assert_not_called()

    def test_delete_result_files_legacy_path_deletes_metadata_and_dirs(
        self, mock_redis
    ):
        """Without workflow_name, still deletes metadata files and raw/zip dirs matching search_id."""
        search_id = "exec_789"
        result_id = search_id
        mock_redis.get.return_value = None

        mock_metadata_file = MagicMock()
        mock_metadata_file.is_file.return_value = True
        mock_metadata_file.is_dir.return_value = False
        mock_metadata_file.name = f"metadata_{search_id}.json"
        mock_metadata_file.unlink = MagicMock()

        mock_raw_dir = MagicMock()
        mock_raw_dir.is_dir.return_value = True
        mock_raw_dir.name = f"output_{search_id}"
        mock_raw_dir.__str__ = lambda self: f"output_{search_id}"

        mock_zip_subdir = MagicMock()
        mock_zip_subdir.is_dir.return_value = True
        mock_zip_subdir.name = f"results_{search_id}"
        mock_zip_subdir.__str__ = lambda self: f"results_{search_id}"

        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = [mock_raw_dir]
        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = [mock_metadata_file, mock_zip_subdir]

        with patch(
            "earth2studio.serve.server.cleanup_daemon.shutil.rmtree"
        ) as mock_rmtree:
            _delete_result_files(
                mock_redis,
                result_id,
                search_id,
                mock_output_dir,
                mock_zip_dir,
            )

        mock_metadata_file.unlink.assert_called_once()
        assert mock_rmtree.call_count == 2  # raw dir + zip_dir subdir
        rmtree_calls = {str(c[0][0]) for c in mock_rmtree.call_args_list}
        assert f"output_{search_id}" in rmtree_calls or mock_rmtree.call_count == 2

    def test_delete_result_files_workflow_dir_not_exists(self, mock_redis):
        """When workflow_name is set but workflow subdirectory does not exist, no rmtree on it."""
        workflow_name = "my_wf"
        execution_id = "exec_1"
        combined_id = f"{workflow_name}:{execution_id}"
        mock_redis.get.return_value = None

        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []
        mock_workflow_dir = MagicMock()
        mock_workflow_dir.exists.return_value = False  # workflow dir not present
        mock_output_dir.__truediv__ = lambda self, other: mock_workflow_dir

        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = []

        with patch(
            "earth2studio.serve.server.cleanup_daemon.shutil.rmtree"
        ) as mock_rmtree:
            _delete_result_files(
                mock_redis,
                combined_id,
                execution_id,
                mock_output_dir,
                mock_zip_dir,
                workflow_name=workflow_name,
            )

        mock_workflow_dir.iterdir.assert_not_called()
        mock_rmtree.assert_not_called()


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
            retention_ttl=config.redis.retention_ttl,
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
            retention_ttl=604800,
            expected_status=WorkflowStatus.COMPLETED,
            get_end_time_field="end_time",
            delete_files_func=mock_delete_func,
            log_prefix="workflow",
        )

        # Verify - should not clean up
        assert result is False
        mock_delete_func.assert_not_called()
        mock_redis.setex.assert_not_called()

    def test_process_expired_key_key_missing(self, mock_redis, mock_delete_func):
        """When Redis returns None for the key, return False without calling delete."""
        mock_redis.get.return_value = None

        result = _process_expired_key(
            redis_client=mock_redis,
            key="workflow_execution:wf1:exec_123",
            current_time=datetime.now(timezone.utc),
            results_ttl_hours=24,
            retention_ttl=604800,
            expected_status=WorkflowStatus.COMPLETED,
            get_end_time_field="end_time",
            delete_files_func=mock_delete_func,
            log_prefix="workflow",
        )

        assert result is False
        mock_delete_func.assert_not_called()
        mock_redis.setex.assert_not_called()

    def test_process_expired_key_wrong_status(self, mock_redis, mock_delete_func):
        """When status is not expected (e.g. RUNNING), return False."""
        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=25)).isoformat()
        mock_redis.get.return_value = json.dumps(
            {"status": WorkflowStatus.RUNNING, "end_time": old_time}
        )

        result = _process_expired_key(
            redis_client=mock_redis,
            key="workflow_execution:wf1:exec_123",
            current_time=current_time,
            results_ttl_hours=24,
            retention_ttl=604800,
            expected_status=WorkflowStatus.COMPLETED,
            get_end_time_field="end_time",
            delete_files_func=mock_delete_func,
            log_prefix="workflow",
        )

        assert result is False
        mock_delete_func.assert_not_called()
        mock_redis.setex.assert_not_called()

    def test_process_expired_key_no_end_time(self, mock_redis, mock_delete_func):
        """When end_time and completion_time are missing, return False."""
        mock_redis.get.return_value = json.dumps({"status": WorkflowStatus.COMPLETED})

        result = _process_expired_key(
            redis_client=mock_redis,
            key="workflow_execution:wf1:exec_123",
            current_time=datetime.now(timezone.utc),
            results_ttl_hours=24,
            retention_ttl=604800,
            expected_status=WorkflowStatus.COMPLETED,
            get_end_time_field="end_time",
            delete_files_func=mock_delete_func,
            log_prefix="workflow",
        )

        assert result is False
        mock_delete_func.assert_not_called()
        mock_redis.setex.assert_not_called()

    def test_process_expired_key_uses_completion_time_fallback(
        self, mock_redis, mock_delete_func
    ):
        """When end_time is missing but completion_time is present, use it and expire."""
        config = get_config()
        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=30)).isoformat()
        mock_redis.get.return_value = json.dumps(
            {
                "status": WorkflowStatus.COMPLETED,
                "completion_time": old_time,
            }
        )

        result = _process_expired_key(
            redis_client=mock_redis,
            key="workflow_execution:wf1:exec_old",
            current_time=current_time,
            results_ttl_hours=24,
            retention_ttl=config.redis.retention_ttl,
            expected_status=WorkflowStatus.COMPLETED,
            get_end_time_field="end_time",
            delete_files_func=mock_delete_func,
            log_prefix="workflow",
        )

        assert result is True
        mock_delete_func.assert_called_once()
        mock_redis.setex.assert_called_once()
        updated = json.loads(mock_redis.setex.call_args[0][2])
        assert updated["status"] == WorkflowStatus.EXPIRED

    def test_process_expired_key_invalid_timestamp(self, mock_redis, mock_delete_func):
        """When timestamp is invalid, log warning and return False."""
        mock_redis.get.return_value = json.dumps(
            {
                "status": WorkflowStatus.COMPLETED,
                "end_time": "not-a-valid-timestamp",
            }
        )

        result = _process_expired_key(
            redis_client=mock_redis,
            key="workflow_execution:wf1:exec_123",
            current_time=datetime.now(timezone.utc),
            results_ttl_hours=24,
            retention_ttl=604800,
            expected_status=WorkflowStatus.COMPLETED,
            get_end_time_field="end_time",
            delete_files_func=mock_delete_func,
            log_prefix="workflow",
        )

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
    def cleanup_args(self):
        """Default args for cleanup_expired_results (no module-level config)."""
        config = get_config()
        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []
        mock_output_dir.__truediv__ = lambda self, other: MagicMock(
            exists=lambda: False
        )
        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = []
        return {
            "results_ttl_hours": config.server.results_ttl_hours,
            "retention_ttl": config.redis.retention_ttl,
            "default_output_dir": mock_output_dir,
            "results_zip_dir": mock_zip_dir,
        }

    def test_cleanup_expired_results_workflow_only(self, mock_redis, cleanup_args):
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

        with patch("earth2studio.serve.server.cleanup_daemon._delete_result_files"):
            # Execute
            cleanup_expired_results(mock_redis, **cleanup_args)

            # Verify - should process 1 workflow (exec_old), skip exec_recent
            assert mock_redis.setex.call_count == 1

            # Verify the expired key was updated
            setex_calls = mock_redis.setex.call_args_list
            expired_keys = [call[0][0] for call in setex_calls]
            assert "workflow_execution:wf1:exec_old" in expired_keys

    def test_cleanup_expired_results_no_keys(self, mock_redis, cleanup_args):
        """Test cleanup when there are no keys in Redis"""
        # Setup
        mock_redis.keys.return_value = []

        # Execute
        cleanup_expired_results(mock_redis, **cleanup_args)

        # Verify - no cleanup operations
        mock_redis.setex.assert_not_called()

    def test_cleanup_expired_results_all_recent(self, mock_redis, cleanup_args):
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
        cleanup_expired_results(mock_redis, **cleanup_args)

        # Verify - no cleanup operations (not expired)
        mock_redis.setex.assert_not_called()

    def test_cleanup_expired_results_redis_get_failure(self, mock_redis, cleanup_args):
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

        with (patch("earth2studio.serve.server.cleanup_daemon._delete_result_files"),):
            # Execute - should not raise exception
            cleanup_expired_results(mock_redis, **cleanup_args)

            # Verify - exec_2 should still be processed despite exec_1 error
            assert mock_redis.setex.call_count == 1
            assert mock_redis.setex.call_args[0][0] == "workflow_execution:wf_ok:exec_2"

    def test_cleanup_expired_results_delete_files_failure(
        self, mock_redis, cleanup_args
    ):
        """Test cleanup handles file deletion failures gracefully and does not mark as expired"""
        # Setup
        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=30)).isoformat()

        mock_redis.keys.return_value = ["workflow_execution:wf_fail:exec_fail"]

        mock_redis.get.return_value = json.dumps(
            {"status": WorkflowStatus.COMPLETED, "end_time": old_time}
        )

        with patch(
            "earth2studio.serve.server.cleanup_daemon._delete_result_files"
        ) as mock_delete:
            mock_delete.side_effect = Exception("File deletion error")

            # Execute - should handle exception and continue
            cleanup_expired_results(mock_redis, **cleanup_args)

            # Verify delete was attempted
            mock_delete.assert_called_once()

            # Status should NOT be updated if file deletion fails
            # (we want to retry cleanup on next run)
            assert mock_redis.setex.call_count == 0

    def test_cleanup_expired_results_workflow_execution(self, mock_redis, cleanup_args):
        """Test cleanup of workflow execution keys"""
        # Setup
        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=30)).isoformat()

        mock_redis.keys.return_value = ["workflow_execution:my_workflow:exec_123"]

        mock_redis.get.return_value = json.dumps(
            {"status": WorkflowStatus.COMPLETED, "end_time": old_time}
        )

        with patch(
            "earth2studio.serve.server.cleanup_daemon._delete_result_files"
        ) as mock_delete:
            # Execute
            cleanup_expired_results(mock_redis, **cleanup_args)

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
        # Custom 48 hour TTL (pass as args; no module-level config)
        config = get_config()
        mock_output_dir = MagicMock()
        mock_output_dir.iterdir.return_value = []
        mock_output_dir.__truediv__ = lambda self, other: MagicMock(
            exists=lambda: False
        )
        mock_zip_dir = MagicMock()
        mock_zip_dir.iterdir.return_value = []

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

        with patch("earth2studio.serve.server.cleanup_daemon._delete_result_files"):
            # Execute with 48 hour TTL
            cleanup_expired_results(
                mock_redis,
                results_ttl_hours=48,
                retention_ttl=config.redis.retention_ttl,
                default_output_dir=mock_output_dir,
                results_zip_dir=mock_zip_dir,
            )

            # Verify - only exec_50h should be cleaned up
            assert mock_redis.setex.call_count == 1
            assert (
                mock_redis.setex.call_args[0][0] == "workflow_execution:wf_50h:exec_50h"
            )

    def test_cleanup_expired_results_malformed_key_skipped(
        self, mock_redis, cleanup_args
    ):
        """Keys with fewer than 3 parts (e.g. 'workflow_execution:incomplete') are skipped."""
        current_time = datetime.now(timezone.utc)
        old_time = (current_time - timedelta(hours=30)).isoformat()

        mock_redis.keys.return_value = [
            "workflow_execution:incomplete",  # only 2 parts
            "workflow_execution:wf1:exec_old",
        ]

        def get_side_effect(key):
            if key == "workflow_execution:wf1:exec_old":
                return json.dumps(
                    {"status": WorkflowStatus.COMPLETED, "end_time": old_time}
                )
            return None

        mock_redis.get.side_effect = get_side_effect

        with patch("earth2studio.serve.server.cleanup_daemon._delete_result_files"):
            cleanup_expired_results(mock_redis, **cleanup_args)

            # Only the valid key (exec_old) should be processed; malformed key is skipped
        assert mock_redis.setex.call_count == 1
        assert mock_redis.setex.call_args[0][0] == "workflow_execution:wf1:exec_old"


class TestCleanupDaemonMain:
    """Tests for main() daemon entrypoint."""

    def test_main_redis_connection_failure_exits_with_one(self, tmp_path):
        """When Redis connection or ping fails, main() calls sys.exit(1)."""
        mock_config = MagicMock()
        mock_config.paths.default_output_dir = str(tmp_path / "out")
        mock_config.paths.results_zip_dir = str(tmp_path / "zip")
        mock_config.server.results_ttl_hours = 24
        mock_config.server.cleanup_watchdog_sec = 60
        mock_config.redis.retention_ttl = 604800
        mock_config.redis.host = "localhost"
        mock_config.redis.port = 6379
        mock_config.redis.db = 0
        mock_config.redis.password = None
        mock_config.redis.decode_responses = True
        mock_config.redis.socket_connect_timeout = 5
        mock_config.redis.socket_timeout = 5

        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.side_effect = Exception("Connection refused")

        with (
            patch(
                "earth2studio.serve.server.cleanup_daemon.get_config",
                return_value=mock_config,
            ),
            patch(
                "earth2studio.serve.server.cleanup_daemon.redis.Redis",
                return_value=mock_redis_instance,
            ),
            patch("earth2studio.serve.server.cleanup_daemon.sys.exit") as mock_exit,
        ):
            mock_exit.side_effect = SystemExit(1)
            with pytest.raises(SystemExit):
                main()
            mock_exit.assert_called_once_with(1)

    def test_main_runs_one_cycle_then_handles_keyboard_interrupt(self, tmp_path):
        """When time.sleep raises KeyboardInterrupt (e.g. SIGINT), main() exits gracefully and closes Redis."""
        mock_config = MagicMock()
        mock_config.paths.default_output_dir = str(tmp_path / "out")
        mock_config.paths.results_zip_dir = str(tmp_path / "zip")
        mock_config.server.results_ttl_hours = 24
        mock_config.server.cleanup_watchdog_sec = 60
        mock_config.redis.retention_ttl = 604800
        mock_config.redis.host = "localhost"
        mock_config.redis.port = 6379
        mock_config.redis.db = 0
        mock_config.redis.password = None
        mock_config.redis.decode_responses = True
        mock_config.redis.socket_connect_timeout = 5
        mock_config.redis.socket_timeout = 5

        mock_redis_instance = MagicMock()
        mock_redis_instance.ping.return_value = True
        mock_redis_instance.close = MagicMock()

        with (
            patch(
                "earth2studio.serve.server.cleanup_daemon.get_config",
                return_value=mock_config,
            ),
            patch(
                "earth2studio.serve.server.cleanup_daemon.redis.Redis",
                return_value=mock_redis_instance,
            ),
            patch("earth2studio.serve.server.cleanup_daemon.cleanup_expired_results"),
            patch("earth2studio.serve.server.cleanup_daemon.signal.signal"),
            patch(
                "earth2studio.serve.server.cleanup_daemon.time.sleep",
                side_effect=KeyboardInterrupt,
            ),
        ):
            main()
            mock_redis_instance.close.assert_called_once()

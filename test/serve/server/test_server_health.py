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

from earth2studio.serve.server.health import (
    HealthResult,
    ServiceStatus,
    check_all_services,
)


class TestCheckAllServices:
    """Tests for the in-process health check module."""

    @patch("earth2studio.serve.server.health._pgrep")
    def test_all_services_healthy(self, mock_pgrep):
        mock_pgrep.return_value = [1234]
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.llen.return_value = 0

        result = check_all_services(redis_client=mock_redis)

        assert result.healthy
        assert result.redis.running
        assert result.api_workers.running
        assert result.cleanup_daemon.running
        assert all(s.running for s in result.rq_workers.values())

    @patch("earth2studio.serve.server.health._pgrep")
    def test_redis_down(self, mock_pgrep):
        mock_pgrep.return_value = []
        result = check_all_services(redis_client=None)

        assert not result.healthy
        assert not result.redis.running

    @patch("earth2studio.serve.server.health._pgrep")
    def test_redis_process_running_but_connection_failed(self, mock_pgrep):
        mock_pgrep.return_value = [1234]
        mock_redis = MagicMock()
        mock_redis.ping.side_effect = Exception("Connection refused")

        result = check_all_services(redis_client=mock_redis)

        assert not result.redis.running
        assert result.redis.details["connection"] == "failed"

    @patch("earth2studio.serve.server.health._pgrep")
    def test_partial_rq_workers_down(self, mock_pgrep):
        def pgrep_side_effect(pattern, *, exact=False):
            if "inference" in pattern:
                return []
            return [1234]

        mock_pgrep.side_effect = pgrep_side_effect
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.llen.return_value = 0

        result = check_all_services(redis_client=mock_redis)

        assert not result.healthy
        assert not result.rq_workers["inference"].running
        assert result.rq_workers["result_zip"].running

    @patch("earth2studio.serve.server.health._pgrep")
    def test_no_redis_client_provided(self, mock_pgrep):
        mock_pgrep.return_value = [1234]
        result = check_all_services(redis_client=None)

        assert result.healthy
        assert result.redis.details["connection"] == "skipped"

    @patch("earth2studio.serve.server.health._pgrep")
    def test_queue_depth_reported(self, mock_pgrep):
        mock_pgrep.return_value = [1234]
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_redis.llen.return_value = 42

        result = check_all_services(redis_client=mock_redis)

        assert result.rq_workers["inference"].details["queue_depth"] == 42


class TestServiceStatus:
    """Tests for the ServiceStatus dataclass."""

    def test_defaults(self):
        s = ServiceStatus()
        assert not s.running
        assert s.pids == []
        assert s.details == {}


class TestHealthResult:
    """Tests for the HealthResult dataclass."""

    def test_healthy_property_true(self):
        r = HealthResult(
            redis=ServiceStatus(running=True),
            api_workers=ServiceStatus(running=True),
            rq_workers={"q": ServiceStatus(running=True)},
            cleanup_daemon=ServiceStatus(running=True),
        )
        assert r.healthy

    def test_healthy_property_false_when_redis_down(self):
        r = HealthResult(
            redis=ServiceStatus(running=False),
            api_workers=ServiceStatus(running=True),
            rq_workers={"q": ServiceStatus(running=True)},
            cleanup_daemon=ServiceStatus(running=True),
        )
        assert not r.healthy

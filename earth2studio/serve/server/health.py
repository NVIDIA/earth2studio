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

"""In-process health check for all earth2studio serve services."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

RQ_QUEUES = [
    "inference",
    "result_zip",
    "object_storage",
    "geocatalog_ingestion",
    "finalize_metadata",
]


@dataclass
class ServiceStatus:
    """Status of an individual service (process presence and connectivity)."""

    running: bool = False
    pids: list[int] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthResult:
    """Aggregated health status across all monitored services."""

    redis: ServiceStatus = field(default_factory=ServiceStatus)
    api_workers: ServiceStatus = field(default_factory=ServiceStatus)
    rq_workers: dict[str, ServiceStatus] = field(default_factory=dict)
    cleanup_daemon: ServiceStatus = field(default_factory=ServiceStatus)

    @property
    def healthy(self) -> bool:
        rq_ok = all(s.running for s in self.rq_workers.values())
        return (
            self.redis.running
            and self.api_workers.running
            and rq_ok
            and self.cleanup_daemon.running
        )


def _pgrep(pattern: str, *, exact: bool = False) -> list[int]:
    """Return PIDs matching the pattern via pgrep."""
    cmd = ["pgrep", "-x" if exact else "-f", pattern]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=5  # noqa: S603
        )
        if result.returncode == 0 and result.stdout.strip():
            return [
                int(pid) for pid in result.stdout.strip().split("\n") if pid.strip()
            ]
    except (subprocess.TimeoutExpired, ValueError, OSError):
        pass
    return []


def _check_redis(redis_client: Any | None) -> ServiceStatus:
    status = ServiceStatus()
    pids = _pgrep("redis-server", exact=True)
    status.pids = pids
    if pids:
        status.running = True
    if redis_client is not None:
        try:
            redis_client.ping()
            status.details["connection"] = "ok"
        except Exception:
            status.details["connection"] = "failed"
            status.running = False
    else:
        status.details["connection"] = "skipped"
    return status


def _check_api_workers() -> ServiceStatus:
    status = ServiceStatus()
    pids = _pgrep("uvicorn.*earth2studio.serve.server.main:app")
    status.pids = pids
    status.running = len(pids) > 0
    status.details["worker_count"] = len(pids)
    return status


def _check_rq_workers(redis_client: Any | None) -> dict[str, ServiceStatus]:
    results: dict[str, ServiceStatus] = {}
    for queue_name in RQ_QUEUES:
        status = ServiceStatus()
        pids = _pgrep(f"rq.*worker.*{queue_name}")
        status.pids = pids
        status.running = len(pids) > 0
        status.details["worker_count"] = len(pids)
        if redis_client is not None and status.running:
            try:
                depth = redis_client.llen(f"rq:queue:{queue_name}")
                status.details["queue_depth"] = depth
            except Exception:  # noqa: S110
                pass
        results[queue_name] = status
    return results


def _check_cleanup_daemon() -> ServiceStatus:
    status = ServiceStatus()
    pids = _pgrep("python.*cleanup_daemon")
    status.pids = pids
    status.running = len(pids) > 0
    return status


def check_all_services(redis_client: Any | None = None) -> HealthResult:
    """Run all health checks and return a structured result."""
    return HealthResult(
        redis=_check_redis(redis_client),
        api_workers=_check_api_workers(),
        rq_workers=_check_rq_workers(redis_client),
        cleanup_daemon=_check_cleanup_daemon(),
    )


def _print_status(result: HealthResult) -> None:
    """Log human-readable status output (mirrors original status.sh format)."""
    logger.info("Services Status")
    logger.info("============================")

    logger.info("Redis:")
    if result.redis.running:
        logger.info(f"  Status: Running (PIDs: {result.redis.pids})")
        conn = result.redis.details.get("connection", "skipped")
        if conn == "skipped":
            logger.info("  Connection: not verified (no client available)")
        else:
            logger.info(f"  Connection: {conn.upper()}")
    else:
        conn = result.redis.details.get("connection")
        if conn == "failed":
            logger.warning("  Status: Process running but connection failed")
        else:
            logger.warning("  Status: Not running")

    logger.info("API Workers:")
    if result.api_workers.running:
        count = result.api_workers.details.get("worker_count", 0)
        logger.info(f"  Status: Running ({count} workers)")
        logger.info(f"  PIDs: {result.api_workers.pids}")
    else:
        logger.warning("  Status: Not running")

    logger.info("RQ Workers:")
    for queue_name, status in result.rq_workers.items():
        logger.info(f"  {queue_name}:")
        if status.running:
            count = status.details.get("worker_count", 0)
            logger.info(f"    Workers: {count} (PIDs: {status.pids})")
            if "queue_depth" in status.details:
                logger.info(
                    f"    Queue depth: {status.details['queue_depth']} jobs pending"
                )
        else:
            logger.warning("    Workers: Not running")

    logger.info("Cleanup Daemon:")
    if result.cleanup_daemon.running:
        logger.info(f"  Status: Running (PIDs: {result.cleanup_daemon.pids})")
    else:
        logger.warning("  Status: Not running")

    logger.info("Summary:")
    logger.info("========")

    def _icon(ok: bool) -> str:
        return "+" if ok else "x"

    logger.info(f"  [{_icon(result.redis.running)}] Redis")
    logger.info(f"  [{_icon(result.api_workers.running)}] API Workers")
    rq_ok = all(s.running for s in result.rq_workers.values())
    logger.info(f"  [{_icon(rq_ok)}] RQ Workers")
    logger.info(f"  [{_icon(result.cleanup_daemon.running)}] Cleanup Daemon")

    if result.healthy:
        logger.info("Overall Status: All services running")
    else:
        logger.warning("Overall Status: One or more services failed")


if __name__ == "__main__":
    import sys

    redis_client = None
    try:
        from earth2studio.serve.server.redis_factory import create_sync_redis_client

        redis_client = create_sync_redis_client()
    except Exception:  # noqa: S110
        pass
    result = check_all_services(redis_client=redis_client)
    _print_status(result)
    sys.exit(0 if result.healthy else 1)

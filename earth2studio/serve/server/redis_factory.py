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

"""Centralized Redis client factory for worker processes.

Worker processes (RQ) run in separate OS processes and cannot access FastAPI's
app.state. This module provides a single factory function so all Redis client
construction is defined in one place.
"""

from __future__ import annotations

from earth2studio.utils.imports import OptionalDependencyError

try:
    import redis  # type: ignore[import-untyped]
    import redis.asyncio as redis_async  # type: ignore[import-untyped]
except ImportError as e:
    raise OptionalDependencyError(
        "serve", "earth2studio.serve.server.redis_factory", e, e.__traceback__
    )

from earth2studio.serve.server.config import get_config


def create_sync_redis_client() -> redis.Redis:
    """Create a synchronous Redis client from the current server configuration."""
    config = get_config()
    return redis.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        password=config.redis.password,
        decode_responses=config.redis.decode_responses,
        socket_connect_timeout=config.redis.socket_connect_timeout,
        socket_timeout=config.redis.socket_timeout,
    )


def create_async_redis_client() -> redis_async.Redis:
    """Create an async Redis client from the current server configuration."""
    config = get_config()
    return redis_async.Redis(
        host=config.redis.host,
        port=config.redis.port,
        db=config.redis.db,
        password=config.redis.password,
        decode_responses=config.redis.decode_responses,
        socket_connect_timeout=config.redis.socket_connect_timeout,
        socket_timeout=config.redis.socket_timeout,
    )


_worker_redis_instance: redis.Redis | None = None


def get_worker_redis_client() -> redis.Redis:
    """Singleton Redis client for RQ worker processes.

    Returns the same client instance on every call within a single OS process.
    Used by both GPU and CPU worker modules to avoid duplicating the singleton
    pattern.
    """
    global _worker_redis_instance
    if _worker_redis_instance is None:
        _worker_redis_instance = create_sync_redis_client()
    return _worker_redis_instance

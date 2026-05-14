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

"""FastAPI dependency injection helpers for Redis and RQ resources."""

from __future__ import annotations

from typing import Annotated

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import redis as redis_sync  # type: ignore[import-untyped]
    import redis.asyncio as redis_async  # type: ignore[import-untyped]
    from fastapi import Depends, HTTPException, Request
    from rq import Queue
except ImportError:
    OptionalDependencyFailure("serve")
    redis_sync = None
    redis_async = None
    Queue = None


@check_optional_dependencies()
def _get_async_redis(request: Request) -> redis_async.Redis:
    """Retrieve the async Redis client from app.state."""
    client: redis_async.Redis | None = getattr(request.app.state, "redis_client", None)
    if client is None:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
    return client


@check_optional_dependencies()
def _get_sync_redis(request: Request) -> redis_sync.Redis:
    """Retrieve the synchronous Redis client from app.state."""
    client: redis_sync.Redis | None = getattr(
        request.app.state, "redis_sync_client", None
    )
    if client is None:
        raise HTTPException(status_code=503, detail="Redis connection not initialized")
    return client


@check_optional_dependencies()
def _get_inference_queue(request: Request) -> Queue:
    """Retrieve the RQ inference queue from app.state."""
    queue: Queue | None = getattr(request.app.state, "inference_queue", None)
    if queue is None:
        raise HTTPException(status_code=503, detail="Inference queue not initialized")
    return queue


AsyncRedis = Annotated[redis_async.Redis, Depends(_get_async_redis)]
SyncRedis = Annotated[redis_sync.Redis, Depends(_get_sync_redis)]
InferenceQueue = Annotated[Queue, Depends(_get_inference_queue)]

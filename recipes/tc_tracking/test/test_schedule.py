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

"""Retries of unstable batches across ranks, ``src.ensemble.schedule``.

Every rank is a thread with its own scheduler; all share one in-process store,
which stands in for the TCP store served by the root process. No model is run:
a rank only records the attempts its scheduler hands out.
"""

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future

import numpy as np
import torch.distributed as dist
from src.ensemble.schedule import RetryQueue, RetryScheduler

from earth2studio.run import WorkItem, work_items

# seconds after which a rank counts as stuck
TIMEOUT = 10

# attempts run by a rank, in order
Attempts = list[tuple[WorkItem, int]]


def _batches(count: int) -> list[WorkItem]:
    """Batches of one member each."""
    times = [np.datetime64("2024-09-24T00:00:00")]
    return work_items(times, np.timedelta64(6, "h"), members=count, member_batch=1)


def _run_rank(
    scheduler: RetryScheduler,
    share: list[WorkItem],
    unstable: Callable[[WorkItem, int], bool],
    during: Callable[[WorkItem, int], None] = lambda item, attempt: None,
) -> Attempts:
    """Run a rank's batches like the pipeline and the runner, one attempt each."""
    ran = []
    for item in scheduler.schedule(share):
        attempt = scheduler.attempt(item)
        during(item, attempt)
        ran.append((item, attempt))
        seed = None if unstable(item, attempt) else 0
        scheduler.batch_finished(item, seed, attempt + 1)
    return ran


def _start(target: Callable, *args: object) -> Future:
    """Run ``target`` in a thread that cannot keep a failed test from ending."""
    future: Future = Future()

    def work() -> None:
        try:
            future.set_result(target(*args))
        except BaseException as error:  # noqa: BLE001 - raised by the test
            future.set_exception(error)

    threading.Thread(target=work, daemon=True).start()
    return future


def test_retries_run_on_free_ranks_until_stable_or_given_up() -> None:
    """Rank 0 holds both batches; one recovers in its retry, one never does."""
    recovers, fails = _batches(2)
    retried = threading.Event()

    def during(item: WorkItem, attempt: int) -> None:
        if (item, attempt) == (recovers, 1):
            retried.set()
        if (item, attempt) == (fails, 0):
            # rank 0 stays busy until a free rank has taken its first retry
            assert retried.wait(TIMEOUT), "the retry was not taken by a free rank"

    store = dist.HashStore()
    ranks = [
        _start(
            _run_rank,
            RetryScheduler(RetryQueue(store, world_size=3), max_retries=1),
            share,
            lambda item, attempt: item == fails or attempt == 0,
            during,
        )
        for share in ([recovers, fails], [], [])
    ]
    ran = [rank.result(TIMEOUT) for rank in ranks]

    assert ran[0][:2] == [(recovers, 0), (fails, 0)]
    assert (recovers, 1) in ran[1] + ran[2]
    # every batch got both attempts, and none a third
    assert sorted(sum(ran, []), key=lambda run: (run[0].member_ids, run[1])) == [
        (recovers, 0),
        (recovers, 1),
        (fails, 0),
        (fails, 1),
    ]


def test_abort_releases_a_waiting_rank() -> None:
    """Rank 1 waits for retries when rank 0 fails before finishing its batches."""
    store = dist.HashStore()
    rank0, rank1 = (
        RetryScheduler(RetryQueue(store, world_size=2), max_retries=1) for _ in range(2)
    )
    waiting = _start(_run_rank, rank1, [], lambda item, attempt: False)
    deadline = time.monotonic() + TIMEOUT
    while store.add("tickets", 0) == 0:  # until rank 1 has drawn a ticket
        assert time.monotonic() < deadline, "rank 1 does not wait for a retry"
        time.sleep(0.01)

    rank0.abort()
    assert waiting.result(TIMEOUT) == []


def test_resume_continues_with_the_next_attempt() -> None:
    """Batches of an earlier run continue at their attempt, or are skipped."""
    done_twice, exhausted = _batches(2)
    scheduler = RetryScheduler(
        RetryQueue(dist.HashStore(), world_size=1),
        max_retries=2,
        first_attempt=lambda item: 2 if item == done_twice else 3,
    )
    ran = _run_rank(scheduler, [done_twice, exhausted], lambda item, attempt: True)
    assert ran == [(done_twice, 2)]

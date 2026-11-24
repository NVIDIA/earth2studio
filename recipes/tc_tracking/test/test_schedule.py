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
which stands in for the TCP store of ``torch.distributed``. No model is run:
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
    during: Callable[[WorkItem, int], None],
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


def _run_ranks(
    shares: list[list[WorkItem]],
    max_retries: int,
    unstable: Callable[[WorkItem, int], bool],
    during: Callable[[WorkItem, int], None] = lambda item, attempt: None,
    store: dist.Store | None = None,
) -> list[Attempts]:
    """Run every rank's share in its own thread, returning each rank's attempts."""
    store = store or dist.HashStore()
    ranks = [
        _start(
            _run_rank,
            RetryScheduler(RetryQueue(store, len(shares)), max_retries),
            share,
            unstable,
            during,
        )
        for share in shares
    ]
    return [rank.result(TIMEOUT) for rank in ranks]


def _until_a_rank_waits(store: dist.Store) -> None:
    """Block until a rank waits for a retry, i.e. has drawn a ticket."""
    deadline = time.monotonic() + TIMEOUT
    while store.add("tickets", 0) == 0:
        assert time.monotonic() < deadline, "no rank waits for a retry"
        time.sleep(0.01)


def _members_and_attempts(ran: list[Attempts]) -> list[tuple[int, int]]:
    return sorted(
        (item.member_ids[0], attempt) for rank in ran for item, attempt in rank
    )


def test_uneven_shares_are_retried_once_each() -> None:
    """Every batch of the only rank with batches needs one retry."""
    batches = _batches(4)
    ran = _run_ranks(
        [batches, [], []], max_retries=2, unstable=lambda item, attempt: attempt == 0
    )
    assert _members_and_attempts(ran) == [(m, a) for m in range(4) for a in (0, 1)]
    assert ran[0][:4] == [(batch, 0) for batch in batches]


def test_batches_unstable_in_every_attempt_are_given_up() -> None:
    batches = _batches(3)
    ran = _run_ranks(
        [batches[:2], batches[2:]], max_retries=2, unstable=lambda item, attempt: True
    )
    assert _members_and_attempts(ran) == [(m, a) for m in range(3) for a in range(3)]


def test_free_rank_takes_the_retry_of_a_busy_rank() -> None:
    first, second = _batches(2)
    retried = threading.Event()

    def during(item: WorkItem, attempt: int) -> None:
        if (item, attempt) == (first, 1):
            retried.set()
        if item == second:
            # rank 0 stays busy until the retry of its first batch has started
            assert retried.wait(TIMEOUT), "the retry was not taken by the free rank"

    ran = _run_ranks(
        [[first, second], []],
        max_retries=1,
        unstable=lambda item, attempt: (item, attempt) == (first, 0),
        during=during,
    )
    assert ran == [[(first, 0), (second, 0)], [(first, 1)]]


def test_waiting_ranks_are_released_once_all_batches_are_done() -> None:
    store = dist.HashStore()
    (batch,) = _batches(1)
    ran = _run_ranks(
        [[batch], []],
        max_retries=1,
        unstable=lambda item, attempt: False,
        # rank 0 finishes its batch only once rank 1 waits for retries
        during=lambda item, attempt: _until_a_rank_waits(store),
        store=store,
    )
    assert ran == [[(batch, 0)], []]


def test_abort_releases_waiting_ranks() -> None:
    store = dist.HashStore()
    waiting = _start(
        _run_rank,
        RetryScheduler(RetryQueue(store, world_size=2), max_retries=1),
        [],
        lambda item, attempt: True,
        lambda item, attempt: None,
    )
    _until_a_rank_waits(store)
    # the other rank fails before it has finished its batches
    RetryScheduler(RetryQueue(store, world_size=2), max_retries=1).abort()
    assert waiting.result(TIMEOUT) == []


def test_resume_continues_with_the_next_attempt() -> None:
    """Batches of an earlier run continue at their attempt, or are skipped."""
    done_twice, exhausted = _batches(2)
    scheduler = RetryScheduler(
        RetryQueue(dist.HashStore(), world_size=1),
        max_retries=2,
        first_attempt=lambda item: 2 if item == done_twice else 3,
    )
    ran = _run_rank(
        scheduler,
        [done_twice, exhausted],
        lambda item, attempt: True,
        lambda item, attempt: None,
    )
    assert ran == [(done_twice, 2)]

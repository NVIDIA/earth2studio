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

"""Retries of unstable batches, run by whichever rank is free.

Instability is rare and uneven, so retrying a batch on the rank where it became
unstable would leave the other ranks idle. Instead, every rank first runs its
share of the batches. An unstable batch is put into a :class:`RetryQueue`
shared by all ranks as soon as its attempt is written, and every rank that has
finished its share takes retries from it until no rank can produce any more.

- :class:`RetryQueue` passes retries between ranks through a key-value store
  served by the root process.
- :class:`RetryScheduler` decides which batches a rank runs, in which attempt.
- :class:`RetryPipeline` is an Earth2Studio :class:`~earth2studio.run.Pipeline`
  running the batches of the scheduler.
"""

import pickle
import socket
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import torch.distributed as dist
from loguru import logger

from earth2studio.run import Pipeline, WorkItem
from earth2studio.run.work import broadcast, rank_info

# a rank waits for a retry for as long as other ranks are busy
_NO_TIMEOUT = timedelta(days=365)


@dataclass(frozen=True)
class Retry:
    """An attempt of a batch, to be run by any rank.

    Parameters
    ----------
    item : WorkItem
        The batch
    attempt : int
        Number of the attempt
    """

    item: WorkItem
    attempt: int


class RetryQueue:
    """Queue of retries shared by all ranks.

    The queue lives in a key-value store reachable from every rank, usually a
    TCP store served by the root process. Its requests are point to point, not
    collectives, so any rank can use it at any time.

    Every rank first runs its own batches, putting unstable ones into the
    queue, and reports with :meth:`finish_initial` once it is done. It then
    takes retries until :meth:`take` returns ``None``, which happens once every
    rank has finished its own batches and no retry is queued or running, so
    none can follow. A retry is taken by exactly one rank: a rank waiting for
    work draws a ticket and waits, inside the store, for the retry of that
    number. Each retry therefore costs a few requests to the store, however
    many ranks there are.

    Parameters
    ----------
    store : dist.Store
        Store shared by all ranks, see :meth:`shared`
    world_size : int
        Number of ranks
    """

    def __init__(self, store: dist.Store, world_size: int) -> None:
        self.store = store
        self.world_size = world_size

    @classmethod
    def shared(cls) -> "RetryQueue":
        """Queue in a store of its own, shared by all ranks. Collective.

        Rank 0 serves the store and tells the other ranks where to reach it, so
        every rank has to call this at the same point of the run. Without a
        process group, the queue is local to the process.

        Returns
        -------
        RetryQueue
            The queue
        """
        rank, world_size = rank_info()
        if world_size == 1:
            return cls(dist.HashStore(), world_size)
        server = None
        if rank == 0:
            # port 0: any free port
            server = dist.TCPStore(
                socket.gethostname(), 0, is_master=True, wait_for_workers=False
            )
        host, port = broadcast(None if server is None else (server.host, server.port))
        store = server or dist.TCPStore(host, port, is_master=False)
        return cls(store, world_size)

    def put(self, retry: Retry) -> None:
        """Queue a retry for any rank to take.

        Only a rank that has not yet reported :meth:`finish_initial`, or is
        running a retry it took, may put one.

        Parameters
        ----------
        retry : Retry
            The retry
        """
        # counted before it can be taken, so the queue never looks finished
        # while the retry waits
        self.store.add("outstanding", 1)
        index = self.store.add("queued", 1) - 1
        self.store.set(f"retry/{index}", pickle.dumps((retry, time.time())))

    def finish_initial(self) -> None:
        """Report that this rank has run its own batches."""
        finished = self.store.add("finished", 1)
        if finished == self.world_size and self._count("outstanding") == 0:
            self._close()

    def take(self) -> Retry | None:
        """Wait for a retry to run.

        Every retry taken has to be reported with :meth:`task_done` once its
        outputs are written.

        Returns
        -------
        Retry | None
            The retry, or None once there is no work left for any rank
        """
        ticket = self.store.add("tickets", 1) - 1
        key = f"retry/{ticket}"
        if self.store.check(["aborted"]):
            return None
        # closed before this ticket was drawn, so no retry will come for it
        if self.store.check(["closed"]) and not self.store.check([key]):
            return None
        # ends with a retry or, once the queue is closed, a release entry
        self.store.wait([key], _NO_TIMEOUT)
        entry = pickle.loads(self.store.get(key))  # noqa: S301 - own store
        if entry is None:
            return None
        retry, queued_at = entry
        rank, _ = rank_info()
        logger.info(
            f"rank {rank}: took the retry of "
            f"{np.datetime_as_string(retry.item.time, unit='s')} members "
            f"{list(retry.item.member_ids)}, {time.time() - queued_at:.2f} s "
            "after it was queued"
        )
        return retry

    def task_done(self) -> None:
        """Report that a retry taken has been run and its outputs are written."""
        outstanding = self.store.add("outstanding", -1)
        if outstanding == 0 and self._count("finished") == self.world_size:
            self._close()

    def abort(self) -> None:
        """Stop handing out retries, e.g. because this rank failed.

        Ranks waiting for a retry are released at once, so they can reach the
        end of the run, where the failure is raised on every rank.
        """
        self.store.set("aborted", b"1")
        self._close()

    def _count(self, key: str) -> int:
        return self.store.add(key, 0)

    def _close(self) -> None:
        """Release every rank waiting for a retry; any rank may do it, repeatedly."""
        # set before the tickets are counted: a rank drawing a ticket later
        # sees the flag, an earlier one gets a release entry below
        self.store.set("closed", b"1")
        queued, tickets = self._count("queued"), self._count("tickets")
        for ticket in range(queued, tickets):
            self.store.set(f"retry/{ticket}", pickle.dumps(None))


class RetryScheduler:
    """Choose the batches a rank runs: its own, then retries of any rank's.

    An unstable attempt with attempts left is queued for a retry as soon as
    the pipeline has written its outputs, so that the retry, possibly on
    another rank, cannot race with its writes. As a
    :class:`~src.ensemble.runner.BatchObserver` of the runner, the scheduler
    learns whether an attempt was stable, and the runner asks it for the
    attempt of each batch with :meth:`attempt`.

    Parameters
    ----------
    queue : RetryQueue
        Queue shared by all ranks
    max_retries : int
        Number of the last attempt of a batch
    first_attempt : Callable[[WorkItem], int], optional
        Attempt to start a batch with, e.g. after attempts of an earlier run,
        by default the first, 0
    """

    def __init__(
        self,
        queue: RetryQueue,
        max_retries: int,
        first_attempt: Callable[[WorkItem], int] | None = None,
    ) -> None:
        self.queue = queue
        self.max_retries = max_retries
        self.first_attempt = first_attempt or (lambda item: 0)
        # attempt of the batch the pipeline is running
        self._attempt = 0
        self._unstable = False

    def schedule(self, items: Iterable[WorkItem]) -> Iterator[WorkItem]:
        """Yield this rank's batches, then retries until none are left.

        The pipeline asks for the next batch only once the previous one is
        written and marked, which is when its retry is queued.

        Parameters
        ----------
        items : Iterable[WorkItem]
            This rank's share of the batches

        Yields
        ------
        WorkItem
            Batch to run next
        """
        for item in items:
            attempt = self.first_attempt(item)
            if attempt > self.max_retries:
                logger.warning(
                    f"{np.datetime_as_string(item.time, unit='s')} members "
                    f"{list(item.member_ids)}: all {attempt} attempts were "
                    "unstable in earlier runs; raise max_retries to try again"
                )
                continue
            yield from self._run(item, attempt)

        self.queue.finish_initial()
        while (retry := self.queue.take()) is not None:
            yield from self._run(retry.item, retry.attempt)
            self.queue.task_done()

    def attempt(self, item: WorkItem) -> int:
        """Attempt of the batch the pipeline is running.

        A rank runs one batch at a time, so this is the attempt of the batch
        handed out last, whatever ``item`` is.

        Parameters
        ----------
        item : WorkItem
            The batch, not used; part of the interface of the runner

        Returns
        -------
        int
            Number of the attempt
        """
        return self._attempt

    def abort(self) -> None:
        """Stop the retries on all ranks, because this rank failed."""
        self.queue.abort()

    def batch_finished(self, item: WorkItem, seed: int | None, attempts: int) -> None:
        """Note whether the attempt was unstable.

        Parameters
        ----------
        item : WorkItem
            The batch
        seed : int | None
            Seed of the attempt, None if it was unstable
        attempts : int
            Number of attempts made
        """
        self._unstable = seed is None

    def _run(self, item: WorkItem, attempt: int) -> Iterator[WorkItem]:
        """Yield a batch for one attempt, then queue a retry if it was unstable."""
        self._attempt = attempt
        self._unstable = False
        yield item
        if self._unstable and attempt < self.max_retries:
            self.queue.put(Retry(item, attempt + 1))


class RetryPipeline(Pipeline):
    """Pipeline whose ranks run the retries of unstable batches of any rank.

    Each rank runs the batches its :class:`RetryScheduler` yields: its own
    share, then retries until none are left. If a batch fails with an error,
    the retries are stopped, so that no rank waits for the failed one.

    Parameters
    ----------
    scheduler : RetryScheduler
        Scheduler of the batches, also an observer of the runner
    *args : Any
        Arguments of :class:`~earth2studio.run.Pipeline`
    **kwargs : Any
        Keyword arguments of :class:`~earth2studio.run.Pipeline`

    Raises
    ------
    ValueError
        If the pipeline has a ``sync`` hook, whose lockstep rounds need every
        rank's batches up front
    """

    def __init__(self, scheduler: RetryScheduler, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.sync is not None:
            raise ValueError("retries on any rank cannot be combined with a sync hook")
        self.scheduler = scheduler

    def run_items(self, items: Iterable[WorkItem], rank: int) -> None:
        """Run this rank's batches, then retries of any rank's batches.

        Parameters
        ----------
        items : Iterable[WorkItem]
            This rank's share of the batches
        rank : int
            This process's rank
        """
        try:
            super().run_items(self.scheduler.schedule(items), rank)
        except BaseException:
            self.scheduler.abort()
            raise

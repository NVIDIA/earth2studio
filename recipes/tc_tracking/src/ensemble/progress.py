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

"""Resume of an ensemble run, including batches that failed."""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from earth2studio.run import ProgressStore, WorkItem
from earth2studio.run.work import MANIFEST_NAME

# files of the batches each rank recorded as failed, one per rank
_FAILED_LOG = "failed-{rank:04d}.log"
_FAILED_LOGS = "failed-*.log"

# a batch: IC in seconds since the epoch and its members
_Batch = tuple[int, tuple[int, ...]]


def _batch(item: WorkItem) -> _Batch:
    time = int(item.time.astype("datetime64[s]").astype("int64"))
    return time, tuple(int(member) for member in item.member_ids)


class BatchProgress(ProgressStore):
    """Progress store that keeps failed batches open for a resumed run.

    A batch whose attempt was unstable is recorded as failed instead of done,
    with the number of attempts made, until a later attempt is stable. A
    resumed run therefore runs it again, continuing with its next attempt; raise
    ``max_retries`` to give a batch more once it used all of them. As a
    :class:`~src.ensemble.runner.BatchObserver` of the runner, the store learns
    which attempts failed.

    Parameters
    ----------
    path : str | os.PathLike
        Directory of the progress records, shared by all ranks
    tags : Mapping[str, Any], optional
        Settings a resumed run must not change, by default None
    overwrite : bool, optional
        Start from scratch, deleting earlier records, by default False
    """

    def __init__(
        self,
        path: str | os.PathLike,
        tags: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        super().__init__(path, tags, overwrite)
        self._failed_log = self.path / _FAILED_LOG.format(rank=0)
        # attempts made for each batch recorded as failed
        self._attempts: dict[_Batch, int] = {}
        # batches of this run that failed, until the pipeline marks them done
        self._failing: dict[WorkItem, int] = {}

    @staticmethod
    def stored_seed(path: str | os.PathLike) -> int | None:
        """Base seed of an earlier run, None if there is none.

        Parameters
        ----------
        path : str | os.PathLike
            Directory of the progress records

        Returns
        -------
        int | None
            The seed recorded in the manifest
        """
        manifest = Path(path) / MANIFEST_NAME
        if not manifest.exists():
            return None
        return json.loads(manifest.read_text()).get("seed")

    def open(
        self, manifest: Mapping[str, Any], rank: int = 0, attempt: int | None = None
    ) -> None:
        """Load the progress records, including the failed batches.

        Parameters
        ----------
        manifest : Mapping[str, Any]
            Description of the run, see :class:`ProgressStore`
        rank : int, optional
            This process's rank, by default 0
        attempt : int | None, optional
            Number of the run, see :class:`ProgressStore`, by default None
        """
        super().open(manifest, rank, attempt)
        self._failed_log = self.path / _FAILED_LOG.format(rank=rank)
        self._attempts = {}
        for log in self.path.glob(_FAILED_LOGS):
            for line in log.read_text().splitlines():
                record = json.loads(line)
                time = int(np.datetime64(record["time"], "s").astype("int64"))
                batch = (time, tuple(record["members"]))
                self._attempts[batch] = max(
                    self._attempts.get(batch, 0), record["attempts"]
                )

    def attempts_made(self, item: WorkItem) -> int:
        """Attempts made for a batch in earlier runs, 0 unless it failed.

        Parameters
        ----------
        item : WorkItem
            The batch

        Returns
        -------
        int
            Number of attempts made
        """
        return self._attempts.get(_batch(item), 0)

    def attempt_started(self, item: WorkItem, attempt: int, seed: int) -> None:
        """Nothing to record before a batch is done.

        Parameters
        ----------
        item : WorkItem
            The batch
        attempt : int
            Number of the attempt
        seed : int
            Seed of the attempt
        """

    def batch_finished(self, item: WorkItem, seed: int | None, attempts: int) -> None:
        """Note a failed attempt, recorded once the pipeline marks it done.

        Parameters
        ----------
        item : WorkItem
            The batch
        seed : int | None
            Seed of the attempt, None if it was unstable
        attempts : int
            Number of attempts made
        """
        if seed is None:
            self._failing[item] = attempts

    def mark_done(self, item: WorkItem) -> None:
        """Record a batch as done, or as failed if all its attempts were unstable.

        Parameters
        ----------
        item : WorkItem
            The batch, whose outputs are written
        """
        attempts = self._failing.pop(item, None)
        if attempts is None:
            super().mark_done(item)
            return
        record = {
            "time": str(item.time.astype("datetime64[s]")),
            "members": [int(member) for member in item.member_ids],
            "attempts": attempts,
        }
        with open(self._failed_log, "a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")
            log.flush()
            os.fsync(log.fileno())
        self._attempts[_batch(item)] = attempts

    def clear(self) -> None:
        """Delete all records, including those of failed batches."""
        super().clear()
        for log in self.path.glob(_FAILED_LOGS):
            log.unlink()

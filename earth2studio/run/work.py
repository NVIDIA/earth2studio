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

"""Work construction, distribution, seeding, progress, and rank helpers.

Implements the work layer of ``dev/spec/EXECUTION_CONTRACT_SPEC.md``: defining
the unit of work, building :class:`WorkItem` lists, splitting them across ranks,
deriving per-member seeds, recording finished items for resume, and the thin
``torch.distributed`` helpers everything above uses. The helpers are no-ops
without a process group; this module never initializes one.

This module is a leaf: it imports nothing from the rest of the run package, so
runners, output supervision, and pipelines can all depend on it.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO, Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from loguru import logger

T = TypeVar("T")

MANIFEST_NAME = "manifest.json"
"""File name of the progress manifest inside a :class:`ProgressStore`."""

COMPACT_LOG_NAME = "done.log"
"""File name of the compacted progress log inside a :class:`ProgressStore`."""

_LOG_PREFIX = "done-"
_LOG_SUFFIX = ".log"

# A progress unit: (time seconds, final lead time seconds, member id or None).
_Unit = tuple[int, int, int | None]


@dataclass(frozen=True)
class WorkItem:
    """One reference time, final lead time, and optional ensemble member group."""

    time: np.datetime64
    final_lead_time: np.timedelta64
    member_ids: tuple[int, ...] = ()


def work_items(
    times: Sequence[np.datetime64],
    final_lead_time: np.timedelta64,
    members: int | Sequence[int] | None = None,
    member_batch: int = 1,
) -> list[WorkItem]:
    """Build a time-major list of work items.

    Each time's members are split into consecutive groups of ``member_batch``,
    so no group spans two times. ``members=None`` gives deterministic items,
    one per time, with no member IDs. Times and ``final_lead_time`` are normalized to
    second resolution, so items compare and hash by instant.

    Parameters
    ----------
    times : Sequence[np.datetime64]
        Reference times, one or more. Repeats are rejected, since they would
        repeat a ``(time, final_lead_time, member)`` unit.
    final_lead_time : np.timedelta64
        Last lead time to produce for every item; ``0`` runs one step.
    members : int | Sequence[int] | None, optional
        Ensemble members: a count, giving ``range(members)``, or explicit
        member IDs. ``None`` gives deterministic items, by default None
    member_batch : int, optional
        Members per item, by default 1

    Returns
    -------
    list[WorkItem]
        Items ordered time-major, then by member group.

    Raises
    ------
    ValueError
        If ``member_batch`` is below 1, ``final_lead_time`` is negative, ``members``
        is empty, negative, or repeats a member ID, ``times`` repeats a
        time, or the resulting items mix deterministic and ensemble members.
    """
    if member_batch < 1:
        raise ValueError(f"member_batch must be >= 1, got {member_batch}")

    final_lead = np.timedelta64(final_lead_time, "s")
    if final_lead < np.timedelta64(0, "s"):
        raise ValueError(f"final_lead_time must be non-negative, got {final_lead_time}")

    time_list: list[np.datetime64] = [time.astype("datetime64[s]") for time in times]
    seen: set[np.datetime64] = set()
    for time in time_list:
        if time in seen:
            raise ValueError(
                f"Repeated time {time} in work_items; every (time, member) "
                "must appear once."
            )
        seen.add(time)

    member_ids = _resolve_members(members)

    items: list[WorkItem] = []
    for time in time_list:
        if member_ids is None:
            items.append(WorkItem(time=time, final_lead_time=final_lead))
            continue
        for start in range(0, len(member_ids), member_batch):
            group = tuple(member_ids[start : start + member_batch])
            items.append(
                WorkItem(time=time, final_lead_time=final_lead, member_ids=group)
            )

    _validate_items(items)
    logger.debug(
        f"Built {len(items)} work items from {len(time_list)} times and "
        f"{0 if member_ids is None else len(member_ids)} members"
    )
    return items


def _resolve_members(members: int | Sequence[int] | None) -> list[int] | None:
    if members is None:
        return None
    if isinstance(members, (int, np.integer)):
        count = int(members)
        if count < 1:
            raise ValueError(f"members must be >= 1 when an int, got {count}")
        return list(range(count))
    ids = [int(member) for member in members]
    if not ids:
        raise ValueError("members must not be empty; use None for deterministic items")
    if any(member < 0 for member in ids):
        raise ValueError(f"member IDs must be non-negative, got {ids}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"member IDs must be unique, got {ids}")
    return ids


def _validate_items(items: Sequence[WorkItem]) -> None:
    """Check that items write disjoint outputs and are uniformly ensembled.

    Every item writes lead times from zero at its ``(time, member)`` positions,
    so two items sharing one would write the same outputs even with different
    final lead times.
    """
    has_members = {bool(item.member_ids) for item in items}
    if len(has_members) > 1:
        raise ValueError(
            "Work items must either all carry members or none; got a mix of "
            "deterministic and ensemble items."
        )
    seen: set[tuple[int, int | None]] = set()
    for item in items:
        for time, _, member in _unit_keys(item):
            if (time, member) in seen:
                label = np.datetime64(time, "s")
                raise ValueError(
                    f"Repeated (time, member) ({label}, {member}) in the item "
                    "list; items write lead times from zero, so each time and "
                    "member must appear in one item only, whatever its final lead time."
                )
            seen.add((time, member))


def member_seed(base: int, time: np.datetime64, member_id: int) -> int:
    """Derive a per-member seed from the base seed, time, and member ID.

    FNV-1a 64 over ``struct.pack(">qqq", base, seconds, member_id)``, with
    ``time`` in integer seconds since the epoch, modulo ``2**63``. This is the
    eval recipe's derivation byte-for-byte, so eval results reproduce. Depends
    only on its arguments, never on batching or world size.

    Parameters
    ----------
    base : int
        Base seed, typically a runner's ``seed``.
    time : np.datetime64
        The work item's reference time.
    member_id : int
        Ensemble member index; deterministic items seed as member ``0``.

    Returns
    -------
    int
        A seed in ``[0, 2**63)``.
    """
    seconds = int(time.astype("datetime64[s]").astype("int64"))
    digest = 0xCBF29CE484222325
    for byte in struct.pack(">qqq", int(base), seconds, int(member_id)):
        digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest % 2**63


def distribute_work(items: Sequence[T], rank: int, world_size: int) -> list[T]:
    """Split items into one contiguous, balanced slice per rank.

    The first ``len(items) % world_size`` ranks get one extra item; ranks with
    nothing left get ``[]`` and are expected to keep joining collectives. An
    item is never split, so a member group always runs on one rank.

    Parameters
    ----------
    items : Sequence[T]
        Full list of items, identical on every rank.
    rank : int
        This process's rank.
    world_size : int
        Number of ranks.

    Returns
    -------
    list[T]
        This rank's contiguous slice, possibly empty.

    Raises
    ------
    ValueError
        If ``world_size`` is below 1 or ``rank`` is outside
        ``[0, world_size)``.
    """
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    total = len(items)
    if world_size == 1:
        return list(items)

    base, remainder = divmod(total, world_size)
    if rank < remainder:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = remainder * (base + 1) + (rank - remainder) * base
        end = start + base

    subset = list(items[start:end])
    if subset:
        logger.debug(f"Rank {rank}: assigned {len(subset)}/{total} work items")
    else:
        logger.debug(f"Rank {rank}: no work items assigned; will idle")
    return subset


def _unit_keys(item: WorkItem) -> list[_Unit]:
    """Return the ``(time, final_lead_time, member)`` units an item covers."""
    time = int(item.time.astype("datetime64[s]").astype("int64"))
    final_lead_time = int(item.final_lead_time.astype("timedelta64[s]").astype("int64"))
    if not item.member_ids:
        return [(time, final_lead_time, None)]
    return [(time, final_lead_time, int(member)) for member in item.member_ids]


class ProgressStore:
    """Record finished work items so a run can resume.

    Progress is recorded as ``(time, final_lead_time, member_id)`` units, with no
    member for deterministic items, so resume is free to change item order,
    rank count and member grouping; a partly done member group reruns in full.
    Each rank appends to its own log under ``path``, named by run attempt and
    rank, so no file ever has two writers. ``path`` must be shared by all ranks.

    The store is rank-local and runs no collectives. One rank calls
    :meth:`setup`; every rank then calls :meth:`open` with the attempt it
    returned::

        attempt = broadcast(store.setup(manifest) if rank == 0 else None)
        store.open(manifest, rank, attempt)

    Parameters
    ----------
    path : str | os.PathLike
        Directory holding the manifest and the progress logs. Created on
        demand.
    tags : Mapping[str, Any] | None, optional
        Caller metadata recorded in the manifest, such as model name and
        checkpoint version, by default None
    overwrite : bool, optional
        Clear existing logs and the manifest in :meth:`setup`, by default False
    """

    def __init__(
        self,
        path: str | os.PathLike,
        tags: Mapping[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.path = Path(path)
        self.tags: dict[str, Any] = dict(tags or {})
        self.overwrite = overwrite
        self._done: set[_Unit] | None = None
        self._attempt: int | None = None
        self._rank = 0
        self._handle: IO[str] | None = None

    # -- public API --------------------------------------------------------

    def setup(self, manifest: Mapping[str, Any]) -> int:
        """Prepare the directory for a run, on one rank only, before :meth:`open`.

        Clears the store when ``overwrite`` is set, compacts existing logs into
        one ``done`` log so the file count stays bounded, and writes the
        manifest if none exists.

        Parameters
        ----------
        manifest : Mapping[str, Any]
            JSON-serializable run description. The store's ``tags`` are merged
            in under ``"tags"``.

        Returns
        -------
        int
            The run attempt every rank passes to :meth:`open`.
        """
        # Compaction may delete the log behind a handle from an earlier run.
        self._close_log()
        if self.overwrite:
            self.clear()
        self.path.mkdir(parents=True, exist_ok=True)
        self._compact()
        manifest_path = self.path / MANIFEST_NAME
        if not manifest_path.exists():
            payload = self._manifest_payload(manifest)
            self._write_atomic(
                manifest_path, json.dumps(payload, sort_keys=True, indent=2) + "\n"
            )
        return self._next_attempt()

    def open(
        self, manifest: Mapping[str, Any], rank: int = 0, attempt: int | None = None
    ) -> None:
        """Check the manifest and load the recorded progress, after :meth:`setup`.

        Parameters
        ----------
        manifest : Mapping[str, Any]
            The run description given to :meth:`setup`, identical on every
            rank.
        rank : int, optional
            This process's rank, which names its log, by default 0
        attempt : int | None, optional
            The run attempt :meth:`setup` returned, which names this run's
            logs. ``None`` derives it from the logs present, which is only
            safe in a single process, by default None

        Raises
        ------
        ValueError
            If the stored manifest differs from ``manifest``; the message
            names the differing keys.
        """
        # Later records go to this attempt's log, never an earlier handle.
        self._close_log()
        self._check_manifest(self._manifest_payload(manifest))
        self._rank = rank
        self._attempt = self._next_attempt() if attempt is None else attempt
        self._done = self._read_logs()
        logger.debug(
            f"Progress store '{self.path}' opened with {len(self._done)} "
            f"recorded units (attempt {self._attempt}, rank {rank})"
        )

    def remaining(self, items: Sequence[WorkItem]) -> list[WorkItem]:
        """Return the items that are not yet done.

        Parameters
        ----------
        items : Sequence[WorkItem]
            Full item list.

        Returns
        -------
        list[WorkItem]
            The items with at least one unrecorded unit, in input order.
        """
        left = [item for item in items if not self.is_done(item)]
        skipped = len(items) - len(left)
        if skipped:
            logger.info(f"Resume: skipping {skipped}/{len(items)} completed work items")
        return left

    def is_done(self, item: WorkItem) -> bool:
        """Whether every unit of an item is recorded.

        Parameters
        ----------
        item : WorkItem
            Item to check.

        Returns
        -------
        bool
            ``True`` when all of the item's units are recorded.
        """
        done = self._loaded()
        return all(unit in done for unit in _unit_keys(item))

    def mark_done(self, item: WorkItem) -> None:
        """Record an item as finished.

        Appends one line to this rank's log and fsyncs it, so no record
        outlives the data it describes.

        Parameters
        ----------
        item : WorkItem
            The finished item.
        """
        record = {
            "time": str(item.time.astype("datetime64[s]")),
            "final_lead_time": int(
                item.final_lead_time.astype("timedelta64[s]").astype("int64")
            ),
            "members": [int(member) for member in item.member_ids],
        }
        handle = self._open_log()
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        self._loaded().update(_unit_keys(item))

    def clear(self) -> None:
        """Delete the recorded progress and manifest."""
        self._close_log()
        if self.path.exists():
            for file in self._log_files():
                file.unlink(missing_ok=True)
            (self.path / MANIFEST_NAME).unlink(missing_ok=True)
        self._done = set()
        logger.debug(f"Cleared progress store '{self.path}'")

    def close(self) -> None:
        """Close this rank's log file; further writes reopen it."""
        self._close_log()

    def __enter__(self) -> ProgressStore:
        """Return the store, for ``with`` blocks that close the log."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close this rank's log file."""
        self._close_log()

    # -- manifest ----------------------------------------------------------

    def _manifest_payload(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(manifest)
        if self.tags:
            payload["tags"] = {**dict(payload.get("tags") or {}), **self.tags}
        # Round-trip so tuples, numpy scalars and the stored file compare alike.
        return json.loads(json.dumps(payload, sort_keys=True, default=str))

    def _check_manifest(self, payload: Mapping[str, Any]) -> None:
        manifest_path = self.path / MANIFEST_NAME
        if not manifest_path.exists():
            return
        try:
            stored = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Progress manifest '{manifest_path}' is not valid JSON; delete "
                "the progress store or run with overwrite=True."
            ) from err
        differing = sorted(
            key
            for key in set(stored) | set(payload)
            if stored.get(key) != payload.get(key)
        )
        if differing:
            raise ValueError(
                f"Progress manifest '{manifest_path}' does not match this run; "
                f"differing keys: {', '.join(differing)}. Resume needs the same "
                "streams, seed and tags; otherwise use a new progress path or "
                "overwrite=True."
            )

    # -- logs --------------------------------------------------------------

    def _log_files(self) -> list[Path]:
        if not self.path.is_dir():
            return []
        compact = self.path / COMPACT_LOG_NAME
        files = sorted(self.path.glob(f"{_LOG_PREFIX}*{_LOG_SUFFIX}"))
        if compact.exists():
            files.insert(0, compact)
        return files

    def _next_attempt(self) -> int:
        attempts = []
        for file in self.path.glob(f"{_LOG_PREFIX}*{_LOG_SUFFIX}"):
            parts = file.stem[len(_LOG_PREFIX) :].split("-")
            if parts and parts[0].isdigit():
                attempts.append(int(parts[0]))
        return max(attempts, default=-1) + 1

    def _log_path(self) -> Path:
        if self._attempt is None:
            self.path.mkdir(parents=True, exist_ok=True)
            self._attempt = self._next_attempt()
        name = f"{_LOG_PREFIX}{self._attempt:04d}-{self._rank:04d}{_LOG_SUFFIX}"
        return self.path / name

    def _open_log(self) -> IO[str]:
        if self._handle is None:
            path = self._log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(path, "a", encoding="utf-8")
        return self._handle

    def _close_log(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def _loaded(self) -> set[_Unit]:
        if self._done is None:
            self._done = self._read_logs()
        return self._done

    def _read_logs(self) -> set[_Unit]:
        done: set[_Unit] = set()
        for file in self._log_files():
            done.update(_units_in(_lines(file), file))
        return done

    def _compact(self) -> None:
        """Merge every log into one ``done`` log: temp file, rename, delete."""
        files = self._log_files()
        if len(files) <= 1:
            return
        lines: list[str] = []
        for file in files:
            lines.extend(_lines(file))
        target = self.path / COMPACT_LOG_NAME
        self._write_atomic(target, "".join(line + "\n" for line in lines))
        for file in files:
            if file != target:
                file.unlink(missing_ok=True)
        logger.debug(
            f"Compacted {len(files)} progress logs into '{target}' "
            f"({len(lines)} records)"
        )

    @staticmethod
    def _write_atomic(path: Path, text: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


def _lines(path: Path) -> list[str]:
    """Return a log's complete lines; a torn last line is dropped."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    if not text:
        return []
    parts = text.split("\n")
    if parts[-1] != "":
        logger.debug(f"Ignoring torn last line of progress log '{path}'")
    parts.pop()
    return [part for part in parts if part.strip()]


def _units_in(lines: Iterable[str], path: Path) -> set[_Unit]:
    """Parse progress units from log lines, skipping unreadable ones."""
    units: set[_Unit] = set()
    for line in lines:
        try:
            record = json.loads(line)
            time = int(
                np.datetime64(record["time"]).astype("datetime64[s]").astype("int64")
            )
            final_lead_time = int(record["final_lead_time"])
            members = [int(member) for member in record["members"]]
        except (ValueError, TypeError, KeyError):
            logger.warning(f"Ignoring unreadable progress record in '{path}': {line!r}")
            continue
        if not members:
            units.add((time, final_lead_time, None))
        else:
            units.update((time, final_lead_time, member) for member in members)
    return units


# ---------------------------------------------------------------------------
# Rank helpers
#
# These call torch.distributed directly and no-op without a process group.
# Core never initializes one: launch with torchrun and call
# init_process_group, or physicsnemo's DistributedManager.initialize().
# ---------------------------------------------------------------------------


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _collective_device() -> torch.device:
    """Return the device collectives must use: CUDA under NCCL, else CPU."""
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            return torch.device(f"cuda:{int(local_rank)}")
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def rank_info() -> tuple[int, int]:
    """Return ``(rank, world_size)``, or ``(0, 1)`` without a process group.

    Returns
    -------
    tuple[int, int]
        This process's rank and the number of ranks.
    """
    if not _distributed():
        return (0, 1)
    return (dist.get_rank(), dist.get_world_size())


def rank0_first(fn: Callable[[], T]) -> T:
    """Run ``fn`` on rank 0, barrier, then run it on the other ranks.

    Lets rank 0 create filesystem objects, such as stores and caches, before
    the other ranks touch them.

    Parameters
    ----------
    fn : Callable[[], T]
        Zero-argument callable run on every rank, rank 0 first.

    Returns
    -------
    T
        This rank's result of ``fn``.
    """
    if not _distributed():
        return fn()
    if dist.get_rank() == 0:
        result = fn()
        barrier()  # rank 0 is done; release the others
        barrier()  # wait for the others to finish
        return result
    barrier()  # wait for rank 0
    result = fn()
    barrier()
    return result


def broadcast(obj: T | None) -> T:
    """Return rank 0's object on every rank.

    Parameters
    ----------
    obj : T | None
        The object on rank 0; ignored, and typically ``None``, elsewhere. It
        must be picklable.

    Returns
    -------
    T
        Rank 0's object.
    """
    if not _distributed():
        return obj  # type: ignore[return-value]
    payload: list[Any] = [obj if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0, device=_collective_device())
    return payload[0]  # type: ignore[no-any-return]


def barrier() -> None:
    """Synchronize every rank."""
    if not _distributed():
        return
    dist.barrier()


def raise_if_any_failed(error: Exception | None) -> None:
    """Raise on every rank if any rank failed. Collective.

    A rank with its own ``error`` re-raises it; the others raise
    :class:`RuntimeError`, so no rank runs on, and none hangs at a later
    collective.

    Parameters
    ----------
    error : Exception | None
        This rank's failure, or ``None``.

    Raises
    ------
    Exception
        This rank's ``error``, when it has one.
    RuntimeError
        When another rank failed and this one did not.
    """
    any_failed = error is not None
    if _distributed():
        flag = torch.tensor(
            [1 if error is not None else 0],
            dtype=torch.int32,
            device=_collective_device(),
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        any_failed = bool(flag.item() > 0)
    if error is not None:
        raise error
    if any_failed:
        rank, _ = rank_info()
        raise RuntimeError(
            f"Another rank failed; rank {rank} is stopping. See that rank's "
            "traceback for the original error."
        )

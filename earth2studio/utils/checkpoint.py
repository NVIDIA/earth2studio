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

from __future__ import annotations

import json
import os
import shutil
import uuid
import warnings
from collections.abc import Mapping
from contextvars import ContextVar, Token
from dataclasses import MISSING, dataclass, fields, is_dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import numpy as np
import torch

T = TypeVar("T")
CheckpointStatePolicy = Literal["minimal", "state", "full"]

_CHECKPOINT_VERSION = 1
_ACTIVE_SESSION: ContextVar[CheckpointSession | None] = ContextVar(
    "earth2studio_checkpoint_session", default=None
)
_CURRENT_CHECKPOINT: ContextVar[Checkpoint | None] = ContextVar(
    "earth2studio_checkpoint", default=None
)
_PENDING_STATES: ContextVar[tuple[PendingCheckpointState, ...]] = ContextVar(
    "earth2studio_checkpoint_pending_states", default=()
)


class CheckpointError(RuntimeError):
    """Base error for checkpoint failures."""


class CheckpointStateCollision(CheckpointError):
    """Raised when a state dataclass type is bound more than once."""


class CheckpointSerializationError(CheckpointError):
    """Raised when checkpoint state cannot be serialized without pickle."""


class CheckpointStateSchemaError(CheckpointError):
    """Raised when a saved dataclass payload does not match the current schema."""


@dataclass(frozen=True)
class PendingCheckpointState:
    """Dataclass state bound before a checkpoint session is active."""

    checkpoint: Checkpoint
    state_id: str
    state: Any
    reusable: bool = False


@dataclass(frozen=True)
class CheckpointEntry:
    """A committed checkpoint catalog row."""

    commit_id: str
    labels: dict[str, Any]
    lead_time: Any | None
    write_count: int
    saved_at: str
    rank: int
    world_size: int
    state_ids: tuple[str, ...]
    artifacts: tuple[str, ...]


class CheckpointState(Generic[T]):
    """Bound checkpoint state proxy returned by :func:`bind_checkpoint_state`.

    The proxy forwards normal attribute access to the wrapped dataclass while
    exposing checkpoint metadata through ``checkpoint_*`` properties.
    """

    __slots__ = ("_state", "_checkpoint", "_session", "_state_loaded")

    def __init__(
        self,
        state: T,
        checkpoint: Checkpoint | None = None,
        session: CheckpointSession | None = None,
        state_loaded: bool = False,
    ) -> None:
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_checkpoint", checkpoint)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_state_loaded", state_loaded)

    @property
    def checkpoint_dataclass(self) -> T:
        """Wrapped dataclass instance serialized by the checkpoint."""
        return self._state

    @property
    def checkpoint_enabled(self) -> bool:
        """Whether this state is associated with a checkpoint."""
        return self._checkpoint is not None

    @property
    def checkpoint_state_policy(self) -> CheckpointStatePolicy:
        """Checkpoint state policy requested by the user."""
        if self._checkpoint is None:
            return "minimal"
        return self._checkpoint.state_policy

    @property
    def checkpoint_device(self) -> torch.device:
        """Device used for live checkpoint state tensors."""
        if self._checkpoint is None:
            return torch.device("cpu")
        return self._checkpoint.device

    @property
    def device(self) -> torch.device:
        """Alias for the checkpoint tensor state device."""
        return self.checkpoint_device

    @property
    def checkpoint_flush_interval(self) -> int | None:
        """Flush interval configured on the associated checkpoint."""
        if self._checkpoint is None:
            return None
        return self._checkpoint.flush_interval

    @property
    def checkpoint_write_count(self) -> int:
        """Number of write boundaries recorded in the active session."""
        if self._session is None:
            return 0
        return self._session.write_count

    @property
    def checkpoint_is_flush_due(self) -> bool:
        """Whether the next checkpoint write is expected to flush to disk."""
        interval = self.checkpoint_flush_interval
        if self._session is None or interval is None:
            return False
        return (self._session.write_count + 1) % interval == 0

    @property
    def checkpoint_selected(self) -> bool:
        """Whether this state is bound to an existing checkpoint row."""
        return self._session is not None and self._session.exists

    @property
    def checkpoint_state_loaded(self) -> bool:
        """Whether this dataclass was hydrated from the selected checkpoint row."""
        return self._state_loaded

    @property
    def checkpoint_lead_time(self) -> Any | None:
        """Selected checkpoint lead time, if one exists."""
        if self._session is None:
            return None
        return self._session.lead_time

    @property
    def checkpoint_labels(self) -> Mapping[str, Any]:
        """Labels for the active checkpoint session or pending checkpoint."""
        if self._session is not None:
            return self._session.labels
        return {}

    def _bind_checkpoint(
        self,
        checkpoint: Checkpoint | None,
        session: CheckpointSession | None = None,
        state_loaded: bool = False,
    ) -> None:
        object.__setattr__(self, "_checkpoint", checkpoint)
        object.__setattr__(self, "_session", session)
        object.__setattr__(self, "_state_loaded", state_loaded)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        if name == "device" or name.startswith("checkpoint_"):
            raise AttributeError(f"{name!r} is checkpoint metadata and is read-only.")
        setattr(self._state, name, value)

    def __repr__(self) -> str:
        return repr(self._state)


class NullCheckpointSession:
    """No-op checkpoint session used when checkpointing is disabled."""

    exists = False
    lead_time = None
    device = torch.device("cpu")
    checkpoint_device = torch.device("cpu")

    @property
    def labels(self) -> Mapping[str, Any]:
        """Labels for the no-op session."""
        return {}

    @property
    def write_count(self) -> int:
        """Number of checkpoint writes accepted by the no-op session."""
        return 0

    @property
    def is_active(self) -> bool:
        """Whether this checkpoint session is active in the current context."""
        return False

    def write(
        self,
        lead_time: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> None:
        """Accept a checkpoint boundary without committing anything."""
        return None

    def flush(
        self,
        lead_time: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> None:
        """Accept a flush request without committing anything."""
        return None

    def __enter__(self) -> NullCheckpointSession:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def __bool__(self) -> bool:
        return False


NO_CHECKPOINT = NullCheckpointSession()


def default_checkpoint_path(name: str) -> Path:
    """Return the default path for a named checkpoint store."""
    base = Path(
        os.environ.get("EARTH2STUDIO_CACHE", Path.home() / ".cache" / "earth2studio")
    )
    return base / "checkpoints" / name


def bind_checkpoint_state(state: T) -> CheckpointState[T]:
    """Bind a dataclass instance to checkpoint state metadata.

    The returned proxy forwards normal dataclass field access and exposes
    checkpoint metadata through ``checkpoint_*`` properties. When no checkpoint
    session is active, state is buffered for the most recently instantiated
    :class:`Checkpoint` in this context.
    """
    if isinstance(state, CheckpointState):
        bound_state = state
    else:
        if not is_dataclass(state) or isinstance(state, type):
            raise TypeError("bind_checkpoint_state requires a dataclass instance.")
        bound_state = CheckpointState(state)

    session = _ACTIVE_SESSION.get()
    if session is not None:
        return session.bind(bound_state)

    checkpoint = _CURRENT_CHECKPOINT.get()
    if checkpoint is not None:
        bound_state._bind_checkpoint(checkpoint)
        return _buffer_pending_state(checkpoint, bound_state)
    return bound_state


class Checkpoint:
    """Catalog of restart checkpoints for a named inference run.

    Checkpoints store small restart metadata, optional artifacts, and dataclass state
    bound by components through :func:`bind_checkpoint_state`.
    """

    def __init__(
        self,
        name: str,
        path: str | Path | None = None,
        mode: Literal["overwrite", "append"] = "overwrite",
        flush_interval: int | None = 1,
        history_size: int | None = None,
        state_policy: CheckpointStatePolicy = "full",
        rank: int | None = None,
        world_size: int | None = None,
        device: str | torch.device = torch.device("cpu"),
    ) -> None:
        if mode not in ("overwrite", "append"):
            raise ValueError("mode must be 'overwrite' or 'append'.")
        if flush_interval is not None and flush_interval < 1:
            raise ValueError("flush_interval must be a positive integer or None.")
        if history_size is not None and history_size < 1:
            raise ValueError("history_size must be a positive integer or None.")
        state_policy = _normalize_state_policy(state_policy)

        detected_rank, detected_world_size = _detect_distributed_rank()
        self.name = name
        self.path = Path(path) if path is not None else default_checkpoint_path(name)
        self.mode = mode
        self.flush_interval = flush_interval
        self.history_size = 1 if mode == "overwrite" else history_size
        self.state_policy = state_policy
        self.device = torch.device(device)
        self.rank = detected_rank if rank is None else rank
        self.world_size = detected_world_size if world_size is None else world_size
        self._catalog: list[CheckpointEntry] | None = None
        self._context_sessions: list[CheckpointSession] = []
        _CURRENT_CHECKPOINT.set(self)

    @property
    def rank_path(self) -> Path:
        """Directory for the current process rank."""
        return self.path / f"rank_{self.rank:06d}"

    @property
    def catalog(self) -> tuple[CheckpointEntry, ...]:
        """Committed checkpoint entries for the current rank."""
        self.refresh()
        return tuple(self._catalog or [])

    @property
    def active(self) -> CheckpointSession | None:
        """Active checkpoint selected from this catalog, if one is in scope."""
        selected = _ACTIVE_SESSION.get()
        if selected is not None and selected.catalog is self:
            return selected
        return None

    def refresh(self) -> None:
        """Refresh the checkpoint catalog from disk."""
        self._catalog = _read_catalog(self.rank_path)

    def select(self, row: int | None = None, **labels: Any) -> CheckpointSession:
        """Select a checkpoint row or label set.

        A positional integer selects a catalog row, with negative indexing supported.
        Keyword labels select the latest matching row and also define labels for
        future writes. A keyword value of ``-1`` selects the latest saved value for
        that label after all other labels are applied.
        """
        self.refresh()
        entries = list(self._catalog or [])
        encoded_labels = _encode_labels(labels, entries)

        if encoded_labels:
            entries = [
                entry
                for entry in entries
                if all(
                    entry.labels.get(key) == value
                    for key, value in encoded_labels.items()
                )
            ]

        selected_entry: CheckpointEntry | None = None
        if row is not None:
            if not entries:
                raise IndexError("No checkpoint entries match this selection.")
            selected_entry = entries[row]
            if not encoded_labels:
                encoded_labels = selected_entry.labels.copy()
        elif entries:
            selected_entry = entries[-1]

        return CheckpointSession(self, encoded_labels, selected_entry)

    def __enter__(self) -> CheckpointSession:
        active = self.active
        session = active if active is not None else self.select()
        self._context_sessions.append(session)
        return session.__enter__()

    def __exit__(self, *args: Any) -> None:
        if self._context_sessions:
            self._context_sessions.pop().__exit__(*args)

    def __repr__(self) -> str:
        entries = self.catalog
        lines = [
            f'Checkpoint("{self.name}")',
            f"path: {self.path}",
            f"mode: {self.mode}",
            f"state_policy: {self.state_policy}",
            f"rank: {self.rank}/{self.world_size}",
        ]
        if not entries:
            return "\n".join(lines + ["catalog: empty"])

        label_names = sorted({key for entry in entries for key in entry.labels})
        columns = ["id", *label_names, "lead_time", "write_count", "saved_at"]
        rows = [
            [
                str(index),
                *[_display_value(entry.labels.get(name)) for name in label_names],
                _display_value(entry.lead_time),
                str(entry.write_count),
                entry.saved_at,
            ]
            for index, entry in enumerate(entries)
        ]
        widths = [
            max(len(columns[index]), *(len(row[index]) for row in rows))
            for index in range(len(columns))
        ]
        header = "  ".join(
            column.ljust(width) for column, width in zip(columns, widths)
        )
        body = [
            "  ".join(value.ljust(width) for value, width in zip(row, widths))
            for row in rows
        ]
        return "\n".join(lines + ["", header, *body])

    def _commit(
        self,
        session: CheckpointSession,
        lead_time: Any | None,
        artifacts: Mapping[str, Any] | None,
    ) -> CheckpointEntry:
        self.rank_path.mkdir(parents=True, exist_ok=True)
        commits_path = self.rank_path / "commits"
        commits_path.mkdir(parents=True, exist_ok=True)

        commit_id = f"commit_{session.write_count:08d}_{uuid.uuid4().hex[:12]}"
        tmp_path = self.rank_path / f".tmp_{commit_id}"
        commit_path = commits_path / commit_id
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True)

        manifest = {
            "version": _CHECKPOINT_VERSION,
            "checkpoint": self.name,
            "commit_id": commit_id,
            "mode": self.mode,
            "rank": self.rank,
            "world_size": self.world_size,
            "labels": session.labels,
            "lead_time": _encode_json_value(_normalize_lead_time(lead_time)),
            "write_count": session.write_count,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "states": {},
            "artifacts": {},
        }

        states_path = tmp_path / "states"
        for state_id, state in session.bound_states.items():
            dataclass_state = _unwrap_checkpoint_state(state)
            state_path = states_path / _safe_dir_name(state_id)
            _write_dataclass_state(dataclass_state, state_id, state_path)
            manifest["states"][state_id] = {
                "path": str(state_path.relative_to(tmp_path)),
                "schema_hash": _schema_hash(dataclass_state),
            }

        if artifacts:
            artifacts_path = tmp_path / "artifacts"
            artifacts_path.mkdir(parents=True, exist_ok=True)
            for name, value in artifacts.items():
                if not isinstance(name, str):
                    raise CheckpointSerializationError(
                        "checkpoint artifact names must be strings."
                    )
                manifest["artifacts"][name] = _dump_value(
                    value, artifacts_path, (_safe_dir_name(name),)
                )

        _write_json(tmp_path / "manifest.json", manifest)
        tmp_path.rename(commit_path)

        entry = _entry_from_manifest(manifest)
        self._update_catalog(entry)
        session._entry = entry
        session._loaded_states = _load_state_index(commit_path, manifest)
        return entry

    def _update_catalog(self, entry: CheckpointEntry) -> None:
        entries = (
            list(self._catalog)
            if self._catalog is not None
            else _read_catalog(self.rank_path)
        )
        entries = [item for item in entries if item.commit_id != entry.commit_id]
        if self.mode == "overwrite":
            entries = [item for item in entries if item.labels != entry.labels]
        entries.append(entry)
        entries = _apply_history_size(entries, entry.labels, self.history_size)
        _write_catalog(self.rank_path, entries)
        self._catalog = entries
        _prune_commits(self.rank_path, {item.commit_id for item in entries})


class CheckpointSession:
    """Active checkpoint row or future label set."""

    def __init__(
        self,
        catalog: Checkpoint,
        labels: dict[str, Any],
        entry: CheckpointEntry | None,
    ) -> None:
        self.catalog = catalog
        self.labels = labels
        self._entry = entry
        self.bound_states: dict[str, Any] = {}
        self._reusable_state_ids: set[str] = set()
        self.write_count = entry.write_count if entry is not None else 0
        self._pending_lead_time: Any | None = None
        self._pending_artifacts: Mapping[str, Any] | None = None
        self._pending_dirty = False
        self._tokens: list[Token[CheckpointSession | None]] = []
        self._pending_adopted = False
        self._loaded_states = self._load_selected_states()

    @property
    def exists(self) -> bool:
        """Whether this session resolves to an existing checkpoint row."""
        return self._entry is not None

    @property
    def is_active(self) -> bool:
        """Whether this checkpoint session is active in the current context."""
        return _ACTIVE_SESSION.get() is self

    @property
    def commit_id(self) -> str | None:
        """Selected commit identifier, if one exists."""
        return None if self._entry is None else self._entry.commit_id

    @property
    def lead_time(self) -> Any | None:
        """Lead time recorded for this session, if present."""
        return None if self._entry is None else self._entry.lead_time

    @property
    def device(self) -> torch.device:
        """Device used for live checkpoint state tensors."""
        return self.catalog.device

    @property
    def artifacts(self) -> dict[str, Any]:
        """Load artifacts recorded for the selected checkpoint row."""
        if self._entry is None:
            return {}
        manifest = self._read_manifest()
        artifacts_path = self._commit_path / "artifacts"
        return {
            name: _load_value(payload, artifacts_path)
            for name, payload in manifest.get("artifacts", {}).items()
        }

    def artifact(self, name: str) -> Any:
        """Load one artifact by name."""
        artifacts = self.artifacts
        if name not in artifacts:
            raise KeyError(f"Artifact {name!r} not found in checkpoint session.")
        return artifacts[name]

    def bind(self, state: T | CheckpointState[T]) -> CheckpointState[T]:
        """Bind and hydrate a dataclass state object."""
        bound_state = _as_checkpoint_state(state)
        dataclass_state = bound_state.checkpoint_dataclass
        state_id = _state_id(dataclass_state)
        existing = self.bound_states.get(state_id)
        if existing is not None:
            if _unwrap_checkpoint_state(existing) is dataclass_state:
                return existing
            if state_id not in self._reusable_state_ids:
                raise CheckpointStateCollision(
                    f"{state_id} was registered more than once in this checkpoint session."
                )
            self._reusable_state_ids.remove(state_id)

        loaded_state = self._loaded_states.get(state_id)
        if loaded_state is not None:
            _populate_dataclass_state(dataclass_state, state_id, loaded_state)

        bound_state._bind_checkpoint(self.catalog, self, loaded_state is not None)
        self.bound_states[state_id] = bound_state
        return bound_state

    def _adopt_pending_states(self) -> None:
        pending = _PENDING_STATES.get()
        if not pending:
            return

        adopted = [item for item in pending if item.checkpoint is self.catalog]
        if not adopted:
            return

        if self.exists and any(not item.reusable for item in adopted):
            warnings.warn(
                "Checkpoint state was bound before an existing checkpoint session "
                "was active. Saved dataclass state is being hydrated late; "
                "constructor side effects that depended on that state will not be "
                "replayed. Construct restartable components inside "
                "`with checkpoint.select(...):` when hydration must happen during "
                "initialization.",
                UserWarning,
                stacklevel=3,
            )

        for item in adopted:
            self.bind(item.state)
            if item.reusable:
                self._reusable_state_ids.add(item.state_id)

        _PENDING_STATES.set(
            tuple(item for item in pending if item.checkpoint is not self.catalog)
        )

    def write(
        self,
        lead_time: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> CheckpointEntry | None:
        """Record a safe checkpoint boundary and flush if due."""
        self.write_count += 1
        self._pending_lead_time = _normalize_lead_time(lead_time)
        self._pending_artifacts = artifacts
        self._pending_dirty = True
        interval = self.catalog.flush_interval
        if interval is not None and self.write_count % interval == 0:
            return self.flush()
        return None

    def flush(
        self,
        lead_time: Any | None = None,
        artifacts: Mapping[str, Any] | None = None,
    ) -> CheckpointEntry | None:
        """Force an atomic checkpoint commit for the current session."""
        has_updates = lead_time is not None or artifacts is not None
        commit_lead_time = (
            self._pending_lead_time
            if lead_time is None
            else _normalize_lead_time(lead_time)
        )
        commit_artifacts = self._pending_artifacts if artifacts is None else artifacts
        if not self._pending_dirty and not has_updates:
            return None

        entry = self.catalog._commit(self, commit_lead_time, commit_artifacts)
        self._pending_lead_time = commit_lead_time
        self._pending_artifacts = commit_artifacts
        self._pending_dirty = False
        return entry

    def __enter__(self) -> CheckpointSession:
        if not self._pending_adopted:
            self._adopt_pending_states()
            self._pending_adopted = True
        self._tokens.append(_ACTIVE_SESSION.set(self))
        return self

    def __exit__(self, *args: Any) -> None:
        if self._tokens:
            _ACTIVE_SESSION.reset(self._tokens.pop())
        for state in self.bound_states.values():
            state._bind_checkpoint(self.catalog)
            _buffer_pending_state(self.catalog, state, reusable=True)

    def __bool__(self) -> bool:
        return self.exists

    @property
    def _commit_path(self) -> Path:
        if self._entry is None:
            raise CheckpointError("This checkpoint session does not exist.")
        return self.catalog.rank_path / "commits" / self._entry.commit_id

    def _read_manifest(self) -> dict[str, Any]:
        return _read_json(self._commit_path / "manifest.json")

    def _load_selected_states(self) -> dict[str, LoadedState]:
        if self._entry is None:
            return {}
        manifest = self._read_manifest()
        return _load_state_index(self._commit_path, manifest)


@dataclass(frozen=True)
class LoadedState:
    """Serialized dataclass payload waiting to be bound."""

    path: Path
    manifest: dict[str, Any]


def _buffer_pending_state(
    checkpoint: Checkpoint, state: CheckpointState[T], reusable: bool = False
) -> CheckpointState[T]:
    state_id = _state_id(state.checkpoint_dataclass)
    pending = _PENDING_STATES.get()
    for index, item in enumerate(pending):
        if item.checkpoint is not checkpoint or item.state_id != state_id:
            continue
        if _unwrap_checkpoint_state(item.state) is state.checkpoint_dataclass:
            return item.state
        if item.reusable:
            updated = list(pending)
            updated[index] = PendingCheckpointState(
                checkpoint, state_id, state, reusable=reusable
            )
            _PENDING_STATES.set(tuple(updated))
            return state
    _PENDING_STATES.set(
        (*pending, PendingCheckpointState(checkpoint, state_id, state, reusable))
    )
    return state


def _as_checkpoint_state(state: T | CheckpointState[T]) -> CheckpointState[T]:
    if isinstance(state, CheckpointState):
        return state
    if not is_dataclass(state) or isinstance(state, type):
        raise TypeError("checkpoint state requires a dataclass instance.")
    return CheckpointState(state)


def _unwrap_checkpoint_state(state: Any) -> Any:
    if isinstance(state, CheckpointState):
        return state.checkpoint_dataclass
    return state


def _detect_distributed_rank() -> tuple[int, int]:
    try:
        from physicsnemo.distributed import DistributedManager

        manager = DistributedManager()
        rank = getattr(manager, "rank", None)
        world_size = getattr(manager, "world_size", None)
        if rank is not None and world_size is not None:
            return int(rank), int(world_size)
    except ImportError:
        pass
    except (AttributeError, RuntimeError, ValueError, TypeError):
        pass

    for rank_name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        rank = os.environ.get(rank_name)
        if rank is not None:
            world_size = os.environ.get("WORLD_SIZE", "1")
            return int(rank), int(world_size)
    return 0, 1


def _encode_labels(
    labels: Mapping[str, Any], entries: list[CheckpointEntry]
) -> dict[str, Any]:
    encoded = {key: _encode_json_value(value) for key, value in labels.items()}
    latest_keys = [key for key, value in labels.items() if _is_latest_selector(value)]
    if not latest_keys:
        return encoded

    filtered = entries
    for key, value in encoded.items():
        if key not in latest_keys:
            filtered = [entry for entry in filtered if entry.labels.get(key) == value]
    if not filtered:
        return encoded
    latest = filtered[-1]
    for key in latest_keys:
        if key in latest.labels:
            encoded[key] = latest.labels[key]
    return encoded


def _read_catalog(rank_path: Path) -> list[CheckpointEntry]:
    catalog_path = rank_path / "catalog.json"
    if catalog_path.exists():
        try:
            payload = _read_json(catalog_path)
            return [_entry_from_catalog(item) for item in payload.get("entries", [])]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    return _scan_catalog(rank_path)


def _scan_catalog(rank_path: Path) -> list[CheckpointEntry]:
    commits_path = rank_path / "commits"
    if not commits_path.exists():
        return []
    entries: list[CheckpointEntry] = []
    for manifest_path in sorted(commits_path.glob("*/manifest.json")):
        try:
            entries.append(_entry_from_manifest(_read_json(manifest_path)))
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            continue
    return sorted(entries, key=lambda entry: entry.saved_at)


def _write_catalog(rank_path: Path, entries: list[CheckpointEntry]) -> None:
    payload = {
        "version": _CHECKPOINT_VERSION,
        "entries": [_entry_to_catalog(entry) for entry in entries],
    }
    _write_json(rank_path / "catalog.json", payload)


def _apply_history_size(
    entries: list[CheckpointEntry], labels: dict[str, Any], history_size: int | None
) -> list[CheckpointEntry]:
    if history_size is None:
        return entries
    matching = [entry for entry in entries if entry.labels == labels]
    remove = {entry.commit_id for entry in matching[:-history_size]}
    return [entry for entry in entries if entry.commit_id not in remove]


def _prune_commits(rank_path: Path, keep: set[str]) -> None:
    commits_path = rank_path / "commits"
    if not commits_path.exists():
        return
    for commit_path in commits_path.iterdir():
        if commit_path.is_dir() and commit_path.name not in keep:
            shutil.rmtree(commit_path)
    for tmp_path in rank_path.glob(".tmp_*"):
        if tmp_path.is_dir():
            shutil.rmtree(tmp_path)


def _entry_from_manifest(manifest: Mapping[str, Any]) -> CheckpointEntry:
    return CheckpointEntry(
        commit_id=str(manifest["commit_id"]),
        labels=dict(manifest.get("labels", {})),
        lead_time=_decode_json_value(manifest.get("lead_time")),
        write_count=int(manifest.get("write_count", 0)),
        saved_at=str(manifest["saved_at"]),
        rank=int(manifest.get("rank", 0)),
        world_size=int(manifest.get("world_size", 1)),
        state_ids=tuple(manifest.get("states", {}).keys()),
        artifacts=tuple(manifest.get("artifacts", {}).keys()),
    )


def _entry_to_catalog(entry: CheckpointEntry) -> dict[str, Any]:
    return {
        "commit_id": entry.commit_id,
        "labels": entry.labels,
        "lead_time": _encode_json_value(entry.lead_time),
        "write_count": entry.write_count,
        "saved_at": entry.saved_at,
        "rank": entry.rank,
        "world_size": entry.world_size,
        "state_ids": list(entry.state_ids),
        "artifacts": list(entry.artifacts),
    }


def _entry_from_catalog(payload: Mapping[str, Any]) -> CheckpointEntry:
    return CheckpointEntry(
        commit_id=str(payload["commit_id"]),
        labels=dict(payload.get("labels", {})),
        lead_time=_decode_json_value(payload.get("lead_time")),
        write_count=int(payload.get("write_count", 0)),
        saved_at=str(payload["saved_at"]),
        rank=int(payload.get("rank", 0)),
        world_size=int(payload.get("world_size", 1)),
        state_ids=tuple(payload.get("state_ids", ())),
        artifacts=tuple(payload.get("artifacts", ())),
    )


def _write_dataclass_state(state: Any, state_id: str, state_path: Path) -> None:
    state = _unwrap_checkpoint_state(state)
    state_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "state_id": state_id,
        "schema_hash": _schema_hash(state),
        "fields": {
            field.name: _dump_value(
                getattr(state, field.name), state_path, (field.name,)
            )
            for field in fields(state)
        },
    }
    _write_json(state_path / "metadata.json", manifest)


def _load_state_index(
    commit_path: Path, manifest: Mapping[str, Any]
) -> dict[str, LoadedState]:
    loaded: dict[str, LoadedState] = {}
    for state_id, payload in manifest.get("states", {}).items():
        state_path = commit_path / payload["path"]
        loaded[state_id] = LoadedState(
            path=state_path,
            manifest=_read_json(state_path / "metadata.json"),
        )
    return loaded


def _populate_dataclass_state(
    state: Any, state_id: str, loaded_state: LoadedState
) -> None:
    state = _unwrap_checkpoint_state(state)
    if loaded_state.manifest.get("state_id") != state_id:
        raise CheckpointStateSchemaError(
            f"Saved checkpoint state {loaded_state.manifest.get('state_id')} does not match {state_id}."
        )
    expected_hash = _schema_hash(state)
    if loaded_state.manifest.get("schema_hash") != expected_hash:
        raise CheckpointStateSchemaError(
            f"Saved checkpoint state {state_id} does not match the current dataclass schema."
        )

    current_fields = {field.name: field for field in fields(state)}
    saved_fields = loaded_state.manifest.get("fields", {})
    if set(saved_fields) != set(current_fields):
        raise CheckpointStateSchemaError(
            f"Saved checkpoint state {state_id} fields do not match the current dataclass fields."
        )
    for name, payload in saved_fields.items():
        current_value = getattr(state, name)
        setattr(state, name, _load_value(payload, loaded_state.path, current_value))


def _normalize_state_policy(
    policy: str,
) -> CheckpointStatePolicy:
    if policy not in ("minimal", "state", "full"):
        raise ValueError("state_policy must be 'minimal', 'state', or 'full'.")
    return policy


def _state_id(state: Any) -> str:
    state = _unwrap_checkpoint_state(state)
    if not is_dataclass(state) or isinstance(state, type):
        raise TypeError("Checkpoint state must be a dataclass instance.")
    cls = type(state)
    return f"{cls.__module__}.{cls.__qualname__}"


def _schema_hash(state: Any) -> str:
    state = _unwrap_checkpoint_state(state)
    schema = "|".join(
        f"{field.name}:{field.type!r}:{_field_default_id(field)}"
        for field in fields(state)
    )
    return sha256(schema.encode("utf-8")).hexdigest()


def _field_default_id(field: Any) -> str:
    if field.default is not MISSING:
        default_type = type(field.default)
        return f"default:{default_type.__module__}.{default_type.__qualname__}"
    if field.default_factory is not MISSING:
        factory = field.default_factory
        module = getattr(factory, "__module__", type(factory).__module__)
        qualname = getattr(factory, "__qualname__", type(factory).__qualname__)
        return f"factory:{module}.{qualname}"
    return "required"


def _dump_value(
    value: Any, base_path: Path, rel_parts: tuple[str, ...]
) -> dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int) and not isinstance(value, bool):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, datetime):
        return {"kind": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, timedelta):
        return {
            "kind": "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, np.datetime64):
        return _encode_np_datetime(value)
    if isinstance(value, np.timedelta64):
        return _encode_np_timedelta(value)
    if isinstance(value, torch.device):
        return {"kind": "torch_device", "value": str(value)}
    if isinstance(value, torch.dtype):
        return {"kind": "torch_dtype", "value": str(value)}
    if isinstance(value, np.dtype):
        return {"kind": "np_dtype", "value": str(value)}
    if isinstance(value, np.generic):
        return _dump_value(value.item(), base_path, rel_parts)
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return _dump_array(array, "tensor", base_path, rel_parts)
    if isinstance(value, np.ndarray):
        return _dump_array(value, "ndarray", base_path, rel_parts)
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [
                _dump_value(item, base_path, (*rel_parts, str(index)))
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [
                _dump_value(item, base_path, (*rel_parts, str(index)))
                for index, item in enumerate(value)
            ],
        }
    if isinstance(value, dict):
        payload = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointSerializationError(
                    "checkpoint dictionaries must use string keys."
                )
            payload[key] = _dump_value(
                item, base_path, (*rel_parts, _safe_dir_name(key))
            )
        return {"kind": "dict", "items": payload}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "kind": "dataclass",
            "state_id": _state_id(value),
            "schema_hash": _schema_hash(value),
            "fields": {
                field.name: _dump_value(
                    getattr(value, field.name), base_path, (*rel_parts, field.name)
                )
                for field in fields(value)
            },
        }
    raise CheckpointSerializationError(
        f"Unsupported checkpoint value {type(value).__module__}.{type(value).__qualname__}."
    )


def _dump_array(
    array: np.ndarray,
    kind: Literal["tensor", "ndarray"],
    base_path: Path,
    rel_parts: tuple[str, ...],
) -> dict[str, Any]:
    if array.dtype == object:
        raise CheckpointSerializationError(
            "object dtype arrays cannot be checkpointed."
        )
    rel_path = Path(*rel_parts).with_suffix(".npy")
    full_path = base_path / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(full_path, array, allow_pickle=False)
    return {
        "kind": kind,
        "path": str(rel_path),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
    }


def _load_value(
    payload: Mapping[str, Any], base_path: Path, current_value: Any = None
) -> Any:
    kind = payload["kind"]
    if kind == "none":
        return None
    if kind in ("bool", "int", "float", "str"):
        return payload["value"]
    if kind == "datetime":
        return datetime.fromisoformat(payload["value"])
    if kind == "date":
        return date.fromisoformat(payload["value"])
    if kind == "timedelta":
        return timedelta(
            days=payload["days"],
            seconds=payload["seconds"],
            microseconds=payload["microseconds"],
        )
    if kind == "np_datetime64":
        return np.datetime64(payload["value"], payload["unit"])
    if kind == "np_timedelta64":
        return np.timedelta64(payload["value"], payload["unit"])
    if kind == "torch_device":
        return torch.device(payload["value"])
    if kind == "torch_dtype":
        return getattr(torch, payload["value"].split(".")[-1])
    if kind == "np_dtype":
        return np.dtype(payload["value"])
    if kind in ("tensor", "ndarray"):
        array = np.load(base_path / payload["path"], allow_pickle=False)
        if (
            list(array.shape) != payload["shape"]
            or str(array.dtype) != payload["dtype"]
        ):
            raise CheckpointSerializationError(
                "checkpoint array metadata does not match stored data."
            )
        if kind == "tensor":
            return torch.from_numpy(array)
        return array
    if kind == "list":
        return [_load_value(item, base_path) for item in payload["items"]]
    if kind == "tuple":
        return tuple(_load_value(item, base_path) for item in payload["items"])
    if kind == "dict":
        return {
            key: _load_value(item, base_path) for key, item in payload["items"].items()
        }
    if kind == "dataclass":
        if is_dataclass(current_value) and not isinstance(current_value, type):
            expected_hash = _schema_hash(current_value)
            if payload.get("schema_hash") != expected_hash:
                raise CheckpointStateSchemaError(
                    f"Saved nested state {payload.get('state_id')} does not match the current dataclass schema."
                )
            for field in fields(current_value):
                setattr(
                    current_value,
                    field.name,
                    _load_value(
                        payload["fields"][field.name],
                        base_path,
                        getattr(current_value, field.name),
                    ),
                )
            return current_value
        return {
            key: _load_value(item, base_path) for key, item in payload["fields"].items()
        }
    raise CheckpointSerializationError(f"Unsupported checkpoint payload kind {kind!r}.")


def _encode_json_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, datetime):
        return {"kind": "datetime", "value": value.isoformat()}
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if isinstance(value, timedelta):
        return {
            "kind": "timedelta",
            "days": value.days,
            "seconds": value.seconds,
            "microseconds": value.microseconds,
        }
    if isinstance(value, np.datetime64):
        return _encode_np_datetime(value)
    if isinstance(value, np.timedelta64):
        return _encode_np_timedelta(value)
    if isinstance(value, torch.device):
        return {"kind": "torch_device", "value": str(value)}
    if isinstance(value, torch.dtype):
        return {"kind": "torch_dtype", "value": str(value)}
    if isinstance(value, np.dtype):
        return {"kind": "np_dtype", "value": str(value)}
    if isinstance(value, np.generic):
        return _encode_json_value(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            raise CheckpointSerializationError(
                "object dtype arrays cannot be used as checkpoint labels."
            )
        return {
            "kind": "ndarray_label",
            "dtype": str(value.dtype),
            "values": [_encode_json_value(item) for item in value.reshape(-1)],
            "shape": list(value.shape),
        }
    if isinstance(value, list | tuple):
        return [_encode_json_value(item) for item in value]
    if isinstance(value, dict):
        encoded = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CheckpointSerializationError(
                    "checkpoint label dictionaries must use string keys."
                )
            encoded[key] = _encode_json_value(item)
        return encoded
    raise CheckpointSerializationError(
        f"Unsupported checkpoint metadata value {type(value).__module__}.{type(value).__qualname__}."
    )


def _decode_json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_json_value(item) for item in value]
    if not isinstance(value, dict) or "kind" not in value:
        return value
    kind = value["kind"]
    if kind == "datetime":
        return datetime.fromisoformat(value["value"])
    if kind == "date":
        return date.fromisoformat(value["value"])
    if kind == "timedelta":
        return timedelta(
            days=value["days"],
            seconds=value["seconds"],
            microseconds=value["microseconds"],
        )
    if kind == "np_datetime64":
        return np.datetime64(value["value"], value["unit"])
    if kind == "np_timedelta64":
        return np.timedelta64(value["value"], value["unit"])
    if kind == "torch_device":
        return torch.device(value["value"])
    if kind == "torch_dtype":
        return getattr(torch, value["value"].split(".")[-1])
    if kind == "np_dtype":
        return np.dtype(value["value"])
    if kind == "ndarray_label":
        decoded = [_decode_json_value(item) for item in value["values"]]
        return np.asarray(decoded, dtype=np.dtype(value["dtype"])).reshape(
            value["shape"]
        )
    return value


def _encode_np_datetime(value: np.datetime64) -> dict[str, Any]:
    unit, _ = np.datetime_data(value.dtype)
    return {"kind": "np_datetime64", "value": str(value), "unit": unit}


def _encode_np_timedelta(value: np.timedelta64) -> dict[str, Any]:
    unit, _ = np.datetime_data(value.dtype)
    return {
        "kind": "np_timedelta64",
        "value": int(value.astype(f"timedelta64[{unit}]").astype("int64")),
        "unit": unit,
    }


def _normalize_lead_time(value: Any | None) -> Any | None:
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0]
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.detach().cpu().reshape(-1)[0].item()
    return value


def _is_latest_selector(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == -1


def _safe_dir_name(name: str) -> str:
    return sha256(name.encode("utf-8")).hexdigest()[:24]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _display_value(value: Any) -> str:
    value = _decode_json_value(value)
    if isinstance(value, np.ndarray):
        return np.array2string(value, separator=", ")
    return "" if value is None else str(value)

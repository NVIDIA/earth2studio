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

import concurrent.futures
from typing import Any

import numpy as np
import torch
import zarr
from loguru import logger
from zarr.core.array import Array as ZarrArray
from zarr.core.array import CompressorsLike

from earth2studio.io.zarr import ZarrBackend
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    import icechunk
except ImportError:
    OptionalDependencyFailure("data")
    icechunk = None  # type: ignore[assignment]


@check_optional_dependencies()
class IceChunkBackend(ZarrBackend):
    """A backend that writes to an `Icechunk <https://icechunk.io/>`_ repository.

    Icechunk is a transactional storage engine for Zarr that adds version control
    (commits, branches, tags) on top of a regular object store. This backend
    behaves identically to :class:`earth2studio.io.ZarrBackend`
    for ``add_array`` / ``write`` / ``read``, using an Icechunk writable session's
    store in place of a plain Zarr store. Since Icechunk snapshots are immutable,
    call :func:`commit` to persist accumulated writes; uncommitted writes are still
    visible to this backend but are lost if the process exits before committing
    (a warning is logged if the backend is destroyed with pending writes).

    By default, ``write`` is non-blocking: the actual store write for a call runs in
    a background thread so the inference loop can move on to the next step while the
    previous step's write is still in flight. ``read``, ``__getitem__`` and
    ``commit`` all flush pending writes first, so they always observe the latest
    data; pass ``blocking=True`` to write synchronously instead, e.g. if a `commit`
    on every step makes the flush unconditional anyway.

    Parameters
    ----------
    storage : icechunk.Storage | str, optional
        Icechunk storage backend to open/create the repository with. If a string is
        provided, it is treated as a path for
        `icechunk.local_filesystem_storage`. If None, an in-memory Icechunk
        repository is created, by default None
    branch : str, optional
        Branch to open a writable session on. Created (from the tip of "main") if it
        does not already exist, by default "main"
    repo_kwargs : dict[str, Any], optional
        Key word arguments passed to `icechunk.Repository.open_or_create`,
        by default {}
    chunks : dict[str, int], optional
        An ordered dict of chunks to use with the data passed through data/coords, by
        default {}
    backend_kwargs : dict[str, Any], optional
        Key word arguments for zarr.Group root object, by default {"overwrite": False}
    zarr_codecs: CompressorsLike, optional
        Compression codec to use when creating any new arrays. Only effects Zarr 3.0.
        If None, will use no compressor, by default None
    blocking : bool, optional
        If False (default), ``write`` submits the store write to a background thread
        and returns immediately instead of waiting for it to complete. If True,
        ``write`` blocks until the store write finishes, by default False
    pool_size : int, optional
        Number of background threads used for non-blocking writes, ignored if
        ``blocking`` is True, by default 8

    Note
    ----
    For more information about Icechunk see: https://icechunk.io/en/latest/

    Warning
    -------
    In non-blocking mode, writing overlapping regions of the same array back to
    back before a flush point (``read``/``__getitem__``/``commit``) races: the
    background threads may apply in either order. This is not a concern for the
    typical inference-loop pattern of writing disjoint time/lead_time slices.
    """

    def __init__(
        self,
        storage: "icechunk.Storage | str | None" = None,
        branch: str = "main",
        repo_kwargs: dict[str, Any] = {},
        chunks: dict[str, int] = {  # to avoid writing in the same chunk by default
            "ensemble": 1,  # dimensions not present in data are ignored
            "time": 1,
            "lead_time": 1,
            "variable": 1,
        },
        backend_kwargs: dict[str, Any] = {"overwrite": False},
        zarr_codecs: CompressorsLike = None,
        blocking: bool = False,
        pool_size: int = 8,
    ) -> None:

        if storage is None:
            storage = icechunk.in_memory_storage()
        elif isinstance(storage, str):
            storage = icechunk.local_filesystem_storage(storage)

        self.repo = icechunk.Repository.open_or_create(storage, **repo_kwargs)
        if branch not in self.repo.list_branches():
            self.repo.create_branch(branch, self.repo.lookup_branch("main"))
        self.branch = branch

        self.session = self.repo.writable_session(self.branch)
        self.store = self.session.store
        self.root = zarr.group(self.store, **backend_kwargs)
        self.zarr_codecs = zarr_codecs
        self._blocking = blocking
        self._executor = (
            None
            if blocking
            else concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)
        )
        self._pending: list[concurrent.futures.Future] = []
        self._read_store_state(chunks)

    def _write_array(self, name: str, selection: tuple, data: np.ndarray) -> None:
        executor = self._executor
        if executor is None:
            super()._write_array(name, selection, data)
            return
        self._pending.append(
            executor.submit(super()._write_array, name, selection, data)
        )

    def flush(self) -> None:
        """Wait for all in-flight non-blocking writes to complete.

        Re-raises the first exception encountered by a background write, if any.
        No-op in blocking mode or if there are no in-flight writes.
        """
        pending, self._pending = self._pending, []
        for future in pending:
            future.result()

    def __getitem__(self, item: str) -> "ZarrArray":
        """Gets item in Zarr Group, flushing pending non-blocking writes first.

        Parameters
        ----------
        item : str
        """
        self.flush()
        return super().__getitem__(item)

    def read(
        self, coords: CoordSystem, array_name: str, device: torch.device = "cpu"
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Read data from the current zarr group, flushing pending non-blocking
        writes first.

        Parameters
        ----------
        coords : OrderedDict
            Coordinates of the data to be read.
        array_name : str | list[str]
            Name(s) of the array(s) to read from.
        device : torch.device
            device to place the read data from, by default 'cpu'
        """
        self.flush()
        return super().read(coords, array_name, device)

    def __del__(self) -> None:
        # Icechunk only persists writes on commit; warn instead of silently
        # dropping data when a backend with pending writes is garbage collected.
        # The len(root) check skips the fresh-repo case where the only pending
        # change is the root group metadata written during construction.
        try:
            self.flush()
            if self._executor is not None:
                self._executor.shutdown(wait=True)
            if self.session.has_uncommitted_changes and len(self.root) > 0:
                logger.warning(
                    "IceChunkBackend deleted with uncommitted changes; these writes "
                    "were not persisted to branch '{}'. Call commit() to persist.",
                    self.branch,
                )
        except Exception:  # noqa: S110 interpreter may be shutting down
            pass

    def commit(self, message: str, **kwargs: Any) -> str:
        """Commit all writes since the last commit to the Icechunk repository,
        creating a new immutable snapshot. A new writable session is opened
        immediately afterwards so this backend remains usable.

        Parameters
        ----------
        message : str
            Commit message describing the changes being persisted.
        kwargs : Any
            Additional key word arguments passed to `icechunk.Session.commit`,
            such as `allow_empty` or `rebase_with`.

        Returns
        -------
        str
            The ID of the newly created snapshot.

        Raises
        ------
        icechunk.IcechunkError
            If no writes were made since the last commit (pass
            ``allow_empty=True`` to commit anyway).
        icechunk.ConflictError
            If another writer committed to the branch first (pass
            ``rebase_with`` to resolve).
        """
        self.flush()
        snapshot_id = self.session.commit(message, **kwargs)
        logger.debug(
            "Committed snapshot {} to Icechunk branch '{}'", snapshot_id, self.branch
        )

        # overwrite must stay False here regardless of the backend_kwargs used at
        # construction, otherwise every commit would wipe the arrays just written
        self.session = self.repo.writable_session(self.branch)
        self.store = self.session.store
        self.root = zarr.group(self.store, overwrite=False)

        return snapshot_id

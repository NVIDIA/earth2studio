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

from typing import Any

import zarr
from loguru import logger
from zarr.core.array import CompressorsLike

from earth2studio.io.zarr import ZarrBackend
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

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
    visible to this backend but are lost if the process exits before committing.

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

    Note
    ----
    For more information about Icechunk see: https://icechunk.io/en/latest/
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
        self._read_store_state(chunks)

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
            ``allow_empty=True`` to commit anyway), or if another writer
            committed to the branch first (pass ``rebase_with`` to resolve).
        """
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

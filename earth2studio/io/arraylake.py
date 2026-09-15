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

import os
from typing import Any

from zarr.core.array import CompressorsLike

from earth2studio.io.icechunk import IceChunkBackend
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import arraylake
except ImportError:
    OptionalDependencyFailure("data")
    arraylake = None  # type: ignore[assignment]

_API_KEY_ENV_VAR = "EARTHMOVER_API_KEY"  # noqa: S105


@check_optional_dependencies()
class ArraylakeBackend(IceChunkBackend):
    """A backend that writes to an [Arraylake](https://docs.earthmover.io/) repository.

    Arraylake is Earthmover's hosted Icechunk service: an Arraylake repository is a
    regular `icechunk.Repository` stored in cloud object storage, with Arraylake
    providing the catalog and vending the bucket credentials. This backend resolves
    the repository through an `arraylake.Client` and then behaves exactly like
    :class:`earth2studio.io.IceChunkBackend`, including its non-blocking ``write``
    and its ``commit`` / ``flush`` semantics.

    As with Icechunk, writes are only persisted by :func:`commit`; uncommitted writes
    are visible to this backend but are lost if the process exits before committing.

    Parameters
    ----------
    repo : str
        Name of the Arraylake repository in ``"org/repo"`` form, e.g.
        ``"my-org/forecasts"``
    branch : str, optional
        Branch to open a writable session on. Created (from the tip of "main") if it
        does not already exist, by default "main"
    client : arraylake.Client, optional
        Authenticated Arraylake client to use. If None, one is created from
        ``token`` / the environment, by default None
    token : str, optional
        Arraylake API token. If None, falls back to the ``EARTHMOVER_API_KEY``
        environment variable and then to an unauthenticated `arraylake.Client()`,
        which picks up ``ARRAYLAKE_TOKEN`` or a cached ``al auth login`` session,
        by default None
    repo_kwargs : dict[str, Any], optional
        Key word arguments passed to `arraylake.Client.get_or_create_repo`, such as
        `bucket_config_nickname` or `config`, by default {}
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
    For more information about Arraylake's Icechunk repositories see:
    https://docs.earthmover.io/guide/icechunk
    """

    def __init__(
        self,
        repo: str,
        branch: str = "main",
        client: "arraylake.Client | None" = None,
        token: str | None = None,
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

        self.repo_name = repo
        if client is None:
            client = self._make_client(token)

        repository = client.get_or_create_repo(repo, **repo_kwargs)

        # Arraylake hands back a genuine icechunk.Repository, so the Icechunk backend
        # can take it over as is; repo_kwargs is consumed here, not forwarded.
        super().__init__(
            storage=repository,
            branch=branch,
            chunks=chunks,
            backend_kwargs=backend_kwargs,
            zarr_codecs=zarr_codecs,
            blocking=blocking,
            pool_size=pool_size,
        )

    @staticmethod
    def _make_client(token: str | None) -> "arraylake.Client":
        """Create an Arraylake client from an explicit token or the environment.

        Parameters
        ----------
        token : str | None
            Arraylake API token. If None, the ``EARTHMOVER_API_KEY`` environment
            variable is used.

        Returns
        -------
        arraylake.Client
            Authenticated client. With no token at all, a bare `arraylake.Client()`
            is returned, which resolves ``ARRAYLAKE_TOKEN`` or a cached
            ``al auth login`` session itself.
        """
        token = token or os.environ.get(_API_KEY_ENV_VAR)
        if token:
            return arraylake.Client(token=token)
        return arraylake.Client()

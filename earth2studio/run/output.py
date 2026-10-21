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

"""Output-store lifecycle for :class:`earth2studio.run.pipeline.Pipeline`.

:class:`OutputManager` is internal: it owns one IO backend per stream and
drives their lifecycle (``prepare``, ``write``, ``flush``, ``close``) as
``dev/spec/EXECUTION_CONTRACT_SPEC.md`` specifies. It is rank-local and runs no
collectives; ``Pipeline`` orders the ranks around it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr
from loguru import logger
from numpy.typing import ArrayLike

from earth2studio.io.base import IOBackend
from earth2studio.io.utils import output_template
from earth2studio.run.work import WorkItem, _validate_items
from earth2studio.utils.type import CoordinateSystem

# Dimensions work is distributed over: an item writes one `time` and its
# `member_ids` along `ensemble`. Every other dimension is written whole by a
# single item, so no two ranks ever share one of its blocks.
_DISTRIBUTED_DIMS = ("time", "ensemble")


class OutputManager:
    """Per-stream output stores.

    One IO backend is held per stream name. Streams without a backend are not
    written, so a caller may hand over a partial mapping.

    Parameters
    ----------
    backends : Mapping[str, IOBackend]
        Stream name to the backend its outputs are written to.
    """

    def __init__(self, backends: Mapping[str, IOBackend]) -> None:
        self.backends: dict[str, IOBackend] = dict(backends)
        self._closed = False

    def prepare(
        self, schemas: Mapping[str, CoordinateSystem], items: Sequence[WorkItem]
    ) -> None:
        """Create or validate every store, spanning all work items.

        Each store's template comes from its stream's planning signature, the
        sorted unique item times and, for ensembles, the sorted union of
        ``member_ids`` over every item. The stores therefore span the whole
        campaign, so a resume at any world size reuses them (``X8``). Adding
        arrays that already exist only validates them (``I4``), so ranks after
        the first one to prepare change nothing (``X13``).

        Parameters
        ----------
        schemas : Mapping[str, CoordinateSystem]
            Planning signature per stream, as a runner's ``output_coords``
            gives them: ``lead_time``, ``variable`` and the grid's dimensions
            are fixed, ``time`` and ``ensemble`` are not.
        items : Sequence[WorkItem]
            Every item of the run, including the items already done.

        Raises
        ------
        ValueError
            If a backend names a stream without a schema, or ``items`` is empty
            or repeats a ``(time, member)``.
        """
        unknown = sorted(set(self.backends).difference(schemas))
        if unknown:
            raise ValueError(f"Output backends name unknown streams: {unknown}")
        if self.backends and not items:
            raise ValueError("Output stores cannot be prepared without work items")
        _validate_items(items)

        times = np.unique(
            np.array([item.time for item in items], dtype="datetime64[ns]")
        )
        members = np.array(
            sorted({int(member) for item in items for member in item.member_ids}),
            dtype=np.int64,
        )
        leading: dict[str, ArrayLike] = {}
        if members.size:
            leading["ensemble"] = members
        leading["time"] = times

        for stream, backend in self.backends.items():
            logger.debug(f"Adding arrays for stream '{stream}'")
            backend.add_array(output_template(schemas[stream], leading))

    def check_distributed(self) -> None:
        """Reject stores that ranks would share, after :meth:`prepare` (``X14``).

        Ranks write different ``time`` and ``ensemble`` positions, so a store
        is safe only with blocks of 1 along both. Backends report their stored
        layout through an optional ``write_blocks`` property; those without one
        cannot be checked and warn once per type (``X15``).

        Raises
        ------
        ValueError
            If a store has a block wider than 1 along ``time`` or ``ensemble``.
        """
        unchecked: set[str] = set()
        for stream, backend in self.backends.items():
            blocks = getattr(backend, "write_blocks", None)
            if blocks is None:
                unchecked.add(type(backend).__name__)
                continue
            wide = {
                dim: blocks[dim] for dim in _DISTRIBUTED_DIMS if blocks.get(dim, 1) > 1
            }
            if wide:
                raise ValueError(
                    f"Store of stream '{stream}' has blocks {wide}. Ranks write "
                    f"different {list(_DISTRIBUTED_DIMS)} positions, so blocks "
                    "along them must be 1, or ranks would overwrite each other's "
                    "data. Recreate the store chunked at 1 along them."
                )
        for name in sorted(unchecked):
            logger.warning(
                f"{name} reports no write_blocks, so it cannot be checked for "
                "distributed writes: a process-local backend holds only its own "
                "rank's items, and shared storage may be corrupted."
            )

    def write(self, stream: str, x: xr.DataArray) -> None:
        """Write one field, skipping streams without a backend.

        Parameters
        ----------
        stream : str
            Stream the field belongs to.
        x : xr.DataArray
            Field to write, located in the store by its coordinate labels.
        """
        backend = self.backends.get(stream)
        if backend is None:
            return
        backend.write(x)

    def flush(self) -> None:
        """Flush every backend, making earlier writes visible to readers (``I9``)."""
        for backend in self.backends.values():
            backend.flush()

    def close(self) -> Exception | None:
        """Close every backend, even after a failure. Idempotent.

        Closed backends are kept, so they stay readable for callers that
        return them (``I11``). A failure is returned rather than raised, so the
        caller can report it through the same failure-flag reduction as a run
        failure instead of leaving the other ranks waiting.

        Returns
        -------
        Exception | None
            The first error a backend's ``close`` raised, or ``None``. Always
            ``None`` on a repeated call.
        """
        if self._closed:
            return None
        self._closed = True
        errors: list[Exception] = []
        for stream, backend in self.backends.items():
            try:
                backend.close()
            except Exception as error:  # noqa: BLE001 - every backend must close
                logger.error(f"Closing the store of stream '{stream}' failed: {error}")
                errors.append(error)
        return errors[0] if errors else None

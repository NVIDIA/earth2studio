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

"""Pipeline: supervise work items across ranks, resume, scoring, and IO.

:class:`Pipeline` drives any :class:`~earth2studio.run.runner.Runner`: it
distributes work items over ranks, filters and regrids each yielded stream,
hands it to an optional :class:`Scorer`, writes it to that stream's store, and
records finished items so a run can resume. See
``dev/spec/EXECUTION_CONTRACT_SPEC.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, TypeVar

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.grids import infer_grid, same_geometry
from earth2studio.grids.base import E2S_GRID_ID, Regridder
from earth2studio.io.base import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.run._fields import _map_field
from earth2studio.run.output import OutputManager
from earth2studio.run.runner import PrognosticRunner, Runner, Source
from earth2studio.run.work import (
    ProgressStore,
    WorkItem,
    _unit_keys,
    _validate_items,
    broadcast,
    distribute_work,
    raise_if_any_failed,
    rank_info,
)
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, coord_array
from earth2studio.utils.type import CoordinateSystem

T = TypeVar("T")

MANIFEST_VERSION = 1
"""Version of the progress manifest :class:`Pipeline` writes."""


class Scorer(Protocol):
    """Online scoring hook, called with every field before it is written.

    Metrics, verification data and any cross-rank communication stay in the
    scorer. It persists durable state in :meth:`finish_item`, which runs before
    the item is marked done, so a resumed run never skips an unscored item.

    Hooks run only on the rank that owns the item, and ranks own different
    numbers of items, so a hook must never run a collective over the pipeline's
    ranks: unmatched calls hang or fail. Reduce across ranks in a ``sync``
    hook passed to :class:`Pipeline`, which every rank joins once per round, or
    after :meth:`Pipeline.run` returns.
    """

    def begin_item(self, item: WorkItem) -> None:
        """Start scoring ``item``."""
        ...

    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None:
        """Score one field: filtered and regridded, before ``ensemble`` is added."""
        ...

    def finish_item(self, item: WorkItem) -> None:
        """Finish scoring ``item`` and persist its results."""
        ...


class Pipeline:
    """Run work items through a runner on every rank, writing one store per stream.

    ``Pipeline`` owns everything above a single work item: validating the item
    list, planning stores from every item, splitting items over ranks, filtering
    and regridding outputs, scoring, writing, and resume. The runner owns what
    happens inside an item. Launch distributed runs with ``torchrun`` and
    initialize a process group first; without one, ``run`` is single-process.

    Parameters
    ----------
    runner : Runner
        Executes one work item at a time, yielding outputs keyed by stream.
    output : Mapping[str, IOBackend] | None, optional
        Backend per stream. Streams left out are not written. ``None`` writes
        nothing, for scorer-only runs, by default None
    scorer : Scorer | None, optional
        Online scoring hook, by default None. At least one of ``output`` and
        ``scorer`` is required; both may be given.
    output_variables : Mapping[str, Sequence[str]] | None, optional
        Variables kept per stream, in this order. Streams left out keep every
        variable, by default None
    regridder : Mapping[str, Regridder] | None, optional
        Regridder per stream, applied after variable filtering. Its
        ``source_grid`` must match the stream's grid; the stream is stored on
        its ``target_grid``, by default None
    progress : ProgressStore | None, optional
        Records finished items, so a later run skips them, by default None
    device : torch.device | str | None, optional
        Device the runner runs on, by default ``cuda:{LOCAL_RANK}`` when CUDA is
        available, otherwise CPU
    sync : Callable[[WorkItem | None], None] | None, optional
        Per-round hook every rank joins, by default None. When set, ranks run
        their items in lockstep rounds: each runs one item, then every rank
        calls ``sync`` with that item, or ``None`` once it has none left. It may
        run collectives over all ranks, such as reducing a scorer's
        accumulators. Items are marked done only after their round's ``sync``.

    Raises
    ------
    ValueError
        If neither ``output`` nor ``scorer`` is given, or ``output_variables``
        names no variables for a stream.

    Examples
    --------
    >>> items = work_items(times, horizon=np.timedelta64(240, "h"))
    >>> output = {"forecast": ZarrBackend("out/forecast.zarr")}
    >>> Pipeline.from_model(model, GFS(), output=output).run(items)
    """

    def __init__(
        self,
        runner: Runner,
        output: Mapping[str, IOBackend] | None = None,
        scorer: Scorer | None = None,
        output_variables: Mapping[str, Sequence[str]] | None = None,
        regridder: Mapping[str, Regridder] | None = None,
        progress: ProgressStore | None = None,
        device: torch.device | str | None = None,
        sync: Callable[[WorkItem | None], None] | None = None,
    ) -> None:
        if output is None and scorer is None:
            raise ValueError("Pipeline needs an output mapping, a scorer, or both")
        self.runner = runner
        self.output = None if output is None else dict(output)
        self.scorer = scorer
        self.output_variables: dict[str, tuple[str, ...]] = {}
        for stream, variables in (output_variables or {}).items():
            names = (variables,) if isinstance(variables, str) else tuple(variables)
            if not names:
                raise ValueError(f"output_variables names no variables for {stream!r}")
            self.output_variables[stream] = names
        self.regridder = dict(regridder or {})
        self.progress = progress
        self.device = torch.device(device) if device is not None else _default_device()
        self.sync = sync
        # Per-run state, set by run().
        self._outputs: OutputManager | None = None
        self._ensemble = False

    @classmethod
    def from_model(
        cls,
        prognostic: PrognosticModel,
        source: Source | None = None,
        forcing: tuple[Source, ...] | None = None,
        diagnostics: Mapping[str, DiagnosticModel] | None = None,
        seed: int = 0,
        **kwargs: Any,
    ) -> Pipeline:
        """Build a pipeline over a :class:`~earth2studio.run.runner.PrognosticRunner`.

        Parameters
        ----------
        prognostic : PrognosticModel
            Model to roll out; its stream is ``forecast``.
        source : DataSource | ForecastSource, optional
            Source of initial conditions, by default the model's recommendation
        forcing : tuple[DataSource | ForecastSource, ...], optional
            One source per forcing slot, by default the model's recommendations
        diagnostics : Mapping[str, DiagnosticModel], optional
            Diagnostics applied every step, each yielding the stream it is keyed
            by, by default none
        seed : int, optional
            Base seed for per-member seeding of stochastic models, by default 0
        **kwargs : Any
            Further :class:`Pipeline` arguments, such as ``output``.

        Returns
        -------
        Pipeline
            A pipeline driving the new runner.
        """
        runner = PrognosticRunner(prognostic, source, forcing, diagnostics, seed)
        return cls(runner, **kwargs)

    # -- run ---------------------------------------------------------------

    def run(self, items: Sequence[WorkItem]) -> None:
        """Run every item not yet done, each on one rank. Collective.

        Every rank must call ``run`` with identical items. Stores are planned
        from all items, so a resume at any world size reuses them. An item is
        marked done only after its outputs flush, its scorer finishes and, with
        a ``sync`` hook, its round's ``sync`` returns; an interrupted item reruns
        in full. A failure on any rank raises on every
        rank once all ranks have closed their outputs.

        Parameters
        ----------
        items : Sequence[WorkItem]
            Full item list, including items already done, usually from
            :func:`~earth2studio.run.work.work_items`.

        Raises
        ------
        ValueError
            If the items are inconsistent across ranks, repeat a
            ``(time, member)``, mix deterministic and ensemble items, carry
            several members for a runner without member batching, or have a
            horizon the runner cannot reach; if a per-stream setting names an
            unknown stream or variable, or a regridder does not match its stream's
            grid; or if the progress manifest does not match this run.
        RuntimeError
            On ranks that did not fail themselves, when another rank failed.
        """
        rank, world_size = rank_info()
        items = self._check_identical(items, world_size)
        if not items:
            logger.warning("Pipeline.run was given no work items; nothing to do")
            return
        # Local setup can fail on one rank only, e.g. out of memory moving the
        # model, so every rank checks in before the next collective.
        schemas = _all_agree(lambda: self._setup(items))
        assert schemas is not None  # noqa: S101 - _all_agree raised otherwise
        self._ensemble = bool(items[0].member_ids)

        outputs = OutputManager(self.output or {})
        self._outputs = outputs
        try:
            self._open_progress(schemas, rank)
            # Rank 0 creates every store before the others open them (X13).
            _all_agree(lambda: outputs.prepare(schemas, items), rank == 0)
            _all_agree(lambda: outputs.prepare(schemas, items), rank != 0)
            if world_size > 1:
                _all_agree(outputs.check_distributed)
            remaining = self._remaining(items, rank)
        except BaseException:
            # Every rank raises here together; close errors would only mask it.
            outputs.close()
            _close_progress(self.progress)
            raise

        mine = distribute_work(remaining, rank, world_size)
        error: Exception | None = None
        try:
            if self.sync is None:
                for index, item in enumerate(mine):
                    self._run_item(item)
                    self._mark_done(item)
                    _log_finished(rank, index, mine)
            else:
                # Ranks with fewer items idle through the last round.
                rounds = -(-len(remaining) // world_size)
                for index in range(rounds):
                    self._run_round(rank, index, mine)
        except Exception as err:
            logger.error(f"Rank {rank}: work item failed: {err}")
            error = err
        finally:
            # Close failures count as this rank's failure, so they reach the
            # collective below instead of leaving the other ranks waiting.
            close_error = outputs.close()
            close_error = _close_progress(self.progress) or close_error
        raise_if_any_failed(error or close_error)

    def _run_item(self, item: WorkItem) -> None:
        outputs = self._outputs
        assert outputs is not None  # noqa: S101 - set by run()
        if self.scorer is not None:
            self.scorer.begin_item(item)
        for step in self.runner.run_item(item):
            for stream, x in step.items():
                self._emit(item, stream, x)
        outputs.flush()
        if self.scorer is not None:
            self.scorer.finish_item(item)

    def _run_round(self, rank: int, index: int, mine: list[WorkItem]) -> None:
        """Run this rank's item of one round, then join every rank in ``sync``.

        Collective. Each phase reduces a failure flag, so a failure on any rank
        stops every rank in the same round instead of mismatching collectives.
        Items are marked done only once ``sync`` succeeded on every rank, since
        one rank's ``sync`` may fail after another's returned.
        """
        sync = self.sync
        assert sync is not None  # noqa: S101 - only called with a sync hook
        item = mine[index] if index < len(mine) else None
        if item is not None:
            _all_agree(lambda: self._run_item(item))
        else:
            _all_agree(lambda: None)

        _all_agree(lambda: sync(item))
        if item is not None:
            _all_agree(lambda: self._mark_done(item))
            _log_finished(rank, index, mine)
        else:
            _all_agree(lambda: None)

    def _mark_done(self, item: WorkItem) -> None:
        if self.progress is not None:
            self.progress.mark_done(item)

    def _emit(self, item: WorkItem, stream: str, x: xr.DataArray) -> None:
        """Filter, regrid, score, add ``ensemble`` if stored, then write."""
        outputs = self._outputs
        assert outputs is not None  # noqa: S101 - set by run()
        written = stream in outputs.backends
        if not written and self.scorer is None:
            return
        variables = self.output_variables.get(stream)
        if variables is not None:
            x = _select_variables(x, variables)
        regridder = self.regridder.get(stream)
        if regridder is not None:
            x = regridder(x)
        if self.scorer is not None:
            self.scorer.update(item, stream, x)
        if not written:
            return
        if self._ensemble and "ensemble" not in x.dims:
            if len(item.member_ids) != 1:
                raise ValueError(
                    f"Stream {stream!r} output lacks an 'ensemble' dimension for "
                    f"an item with members {item.member_ids}"
                )
            x = x.expand_dims(ensemble=list(item.member_ids), axis=0)
        outputs.write(stream, x)

    # -- validation and planning ------------------------------------------

    def _check_identical(
        self, items: Sequence[WorkItem], world_size: int
    ) -> list[WorkItem]:
        """Check every rank passed the same items (``X7``). Collective."""
        items = list(items)
        if world_size > 1:
            digest = _items_digest(items)
            reference = broadcast(digest)

            def check() -> None:
                if digest != reference:
                    raise ValueError(
                        "Every rank must pass identical work items to Pipeline.run; "
                        "this rank's list differs from rank 0's"
                    )

            _all_agree(check)
        return items

    def _setup(self, items: list[WorkItem]) -> dict[str, CoordinateSystem]:
        """Validate items, place the runner and plan stored schemas. Rank-local."""
        self._validate(items)
        self.runner.to(self.device)
        self.regridder = {k: r.to(self.device) for k, r in self.regridder.items()}
        horizon = max(item.horizon for item in items)
        return self._stored_schemas(self.runner.output_coords(horizon))

    def _validate(self, items: list[WorkItem]) -> None:
        """Check items before any store or progress is touched."""
        _validate_items(items)
        if not getattr(self.runner, "supports_member_batching", False):
            batched = [item for item in items if len(item.member_ids) > 1]
            if batched:
                raise ValueError(
                    f"{type(self.runner).__name__} does not batch ensemble members, "
                    f"but {len(batched)} items carry several; build items with "
                    "member_batch=1"
                )
        # Every horizon must be reachable, not only the longest.
        for horizon in {item.horizon for item in items}:
            self.runner.output_coords(horizon)

    def _stored_schemas(
        self, signatures: Mapping[str, CoordinateSystem]
    ) -> dict[str, CoordinateSystem]:
        """Apply per-stream variable filters and regridders to stream signatures."""
        for setting, keys in (
            ("output", self.output or {}),
            ("output_variables", self.output_variables),
            ("regridder", self.regridder),
        ):
            unknown = sorted(set(keys).difference(signatures))
            if unknown:
                raise ValueError(
                    f"{setting} names unknown streams {unknown}; the runner "
                    f"yields {sorted(signatures)}"
                )
        if self.output is not None:
            for stream in sorted(set(signatures).difference(self.output)):
                logger.info(f"Stream {stream!r} has no output backend; not writing it")

        schemas: dict[str, CoordinateSystem] = {}
        for stream, signature in signatures.items():
            variables = self.output_variables.get(stream)
            if variables is not None:
                if "variable" not in signature.dims:
                    raise ValueError(f"Stream {stream!r} has no variable dimension")
                known = set(np.asarray(signature["variable"].values).astype(str))
                missing = [name for name in variables if name not in known]
                if missing:
                    raise ValueError(
                        f"output_variables for stream {stream!r} names unknown "
                        f"variables {missing}; it has {sorted(known)}"
                    )
                signature = _select_variables(signature, variables)
            regridder = self.regridder.get(stream)
            if regridder is not None:
                signature = _regridded_signature(stream, signature, regridder)
            schemas[stream] = signature
        return schemas

    # -- progress ------------------------------------------------------------

    def _manifest(self, schemas: Mapping[str, CoordinateSystem]) -> dict[str, Any]:
        written = set(self.output or {})
        return {
            "version": MANIFEST_VERSION,
            "streams": {
                stream: _stream_manifest(schema, stream in written)
                for stream, schema in sorted(schemas.items())
            },
            "seed": getattr(self.runner, "seed", None),
            "tags": {},
        }

    def _open_progress(
        self, schemas: Mapping[str, CoordinateSystem], rank: int
    ) -> None:
        progress = self.progress
        if progress is None:
            return
        manifest = self._manifest(schemas)
        attempt = broadcast(_all_agree(lambda: progress.setup(manifest), rank == 0))
        _all_agree(lambda: progress.open(manifest, rank, attempt))

    def _remaining(self, items: list[WorkItem], rank: int) -> list[WorkItem]:
        """Rank 0 lists the items left to run and broadcasts them (``X18``)."""
        progress = self.progress
        if progress is None:
            return items
        return broadcast(_all_agree(lambda: progress.remaining(items), rank == 0))


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    return torch.device("cpu")


def _all_agree(fn: Callable[[], T], run: bool = True) -> T | None:
    """Run ``fn`` where ``run`` is set, then raise on every rank if any failed.

    Collective. Keeps a failure on one rank from leaving the others waiting at
    a later collective. Returns ``fn``'s result, or ``None`` where it did not run.
    """
    result: T | None = None
    error: Exception | None = None
    if run:
        try:
            result = fn()
        except Exception as err:
            error = err
    raise_if_any_failed(error)
    return result


def _close_progress(progress: ProgressStore | None) -> Exception | None:
    """Close a progress store, returning its error rather than raising it."""
    if progress is None:
        return None
    try:
        progress.close()
    except Exception as error:  # noqa: BLE001 - reported collectively
        logger.error(f"Closing progress store '{progress.path}' failed: {error}")
        return error
    return None


def _items_digest(items: Sequence[WorkItem]) -> str:
    """Fingerprint an ordered item list, to check every rank passes the same one."""
    payload = [_unit_keys(item) for item in items]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def _log_finished(rank: int, index: int, mine: list[WorkItem]) -> None:
    item = mine[index]
    members = f", members {list(item.member_ids)}" if item.member_ids else ""
    logger.info(
        f"Rank {rank}: finished work item {index + 1}/{len(mine)} "
        f"(time {item.time}, horizon {item.horizon}{members})"
    )


def _select_variables(x: xr.DataArray, variables: Sequence[str]) -> xr.DataArray:
    """Keep ``variables`` in order; works on fields and signatures alike."""
    return _map_field(x, OrderedDict(variable=np.asarray(variables)))


def _regridded_signature(
    stream: str, signature: CoordinateSystem, regridder: Regridder
) -> CoordinateSystem:
    """Plan a stream's schema on its regridder's target grid (``X11``, ``R4``).

    The regridder is applied once to a single spatial field, so the planned
    schema carries exactly the coordinates and grid attributes its writes will.
    """
    try:
        grid = infer_grid(signature)
    except ValueError as error:
        raise ValueError(
            f"Cannot determine the grid of stream {stream!r} to check its regridder"
        ) from error
    if not same_geometry(regridder.source_grid, grid):
        raise ValueError(
            f"Regridder for stream {stream!r} has a source_grid that does not "
            "match the stream's grid"
        )

    spatial = tuple(regridder.source_grid.dims)
    probe = xr.DataArray(
        np.zeros(tuple(signature.sizes[dim] for dim in spatial), dtype=signature.dtype),
        dims=spatial,
        coords={
            name: coord.variable
            for name, coord in signature.coords.items()
            if coord.dims and set(coord.dims).issubset(spatial)
        },
        name=signature.name,
        attrs=dict(signature.attrs),
    )
    regridded = regridder(probe)

    leading = tuple(dim for dim in signature.dims if dim not in spatial)
    coordinates = {
        name: coord.variable
        for name, coord in signature.coords.items()
        if not set(coord.dims).intersection(spatial)
    }
    coordinates.update(
        {name: coord.variable for name, coord in regridded.coords.items()}
    )
    sizes = {dim: signature.sizes[dim] for dim in leading}
    sizes.update({str(dim): size for dim, size in regridded.sizes.items()})
    return coord_array(
        (*leading, *regridded.dims),
        coordinates,
        sizes=sizes,
        dynamic=tuple(signature.attrs.get(E2S_DYNAMIC_DIMS, ())),
        dtype=regridded.dtype,
        name=signature.name,
        attrs=regridded.attrs,
    )


def _stream_manifest(schema: CoordinateSystem, written: bool) -> dict[str, Any]:
    """Describe a stored schema for the progress manifest.

    Records dimensions, non-spatial labels, grid identity and dtype; item times
    and members are left out, since a resume may change them.
    """
    spatial = set(schema.attrs.get("dims", ()))
    dynamic = set(schema.attrs.get(E2S_DYNAMIC_DIMS, ()))
    labels = {
        str(dim): [str(value) for value in np.asarray(schema[dim].values)]
        for dim in schema.dims
        if dim not in spatial and dim not in dynamic and dim in schema.coords
    }
    grid = schema.attrs.get(E2S_GRID_ID)
    if grid is None:
        try:
            grid = infer_grid(schema).fingerprint()
        except ValueError:
            grid = None
    return {
        "dims": [str(dim) for dim in schema.dims],
        "labels": labels,
        "grid": grid,
        "dtype": str(schema.dtype),
        "written": written,
    }

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

"""insitubatch-backed initial-condition / verification feed for Earth2Studio.

Earth2Studio's standard path is ``DataSource -> xr.DataArray -> fetch_data ->
prep_data_array -> (torch.Tensor, coords)``; xarray is load-bearing down to
``prep_data_array``, and ``fetch_data`` issues one read per ``(time, lead_time)`` and keeps
no memory across calls. A source's own cache (on by default) turns a repeated read into a
local disk hit, but it holds *compressed* bytes keyed by chunk, so a chunk covering several
time steps is decoded again for every step asked of it. For an IO-bound hindcast / scoring
campaign the ``(init, lead)`` grid maps many requested slices onto the **same** stored chunk
(consecutive init times share valid times; a fat time-chunk holds several steps), so that
grid costs a decode per requested slice rather than per chunk.

This module skips xarray: it reads the analysis store with **insitubatch**
(:class:`~insitubatch.source.InSituDataset` -- an async read plan that fetches ahead within a
fixed memory budget and decodes each shared chunk once) and converts each numpy ``Batch`` to
the exact ``(x, coords)`` tuple ``fetch_data(..., legacy=True)`` returns, so it is a drop-in
for the initial-condition feed of ``earth2studio.run`` workflows.

The lead axis is unified: pass ``lead_times`` covering a model's input history (``<= 0``,
e.g. ``[-6h, 0]`` for a 2-step history model) and/or verification leads (``> 0``, for
scoring), each realised as a sample-axis ``shift`` view of one stored array -- no reshard.
See :func:`batch_to_xcoords` for the tensor contract and :class:`InSituForecastFeed` for
the prefetched iterator.
"""

from collections import OrderedDict
from collections.abc import Iterator

import cftime
import numpy as np
import torch
import zarr
from zarr.abc.store import Store

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem, VariableArray

try:
    from insitubatch import (
        Batch,
        InSituDataset,
        open_geometries,
        split_by_chunk,
        to_torch,
        valid_anchor_range,
    )
except ImportError:
    OptionalDependencyFailure("insitu")
    Batch = None
    InSituDataset = None
    open_geometries = None
    split_by_chunk = None
    to_torch = None
    valid_anchor_range = None


def decode_cf_time(
    values: np.ndarray, units: str, calendar: str = "standard"
) -> np.ndarray:
    """Decode a CF ``"<unit> since <reference>"`` integer time coordinate to datetime64[ns].

    Handles the common reanalysis encoding (e.g. WB2/ARCO ERA5 store ``time`` as
    ``"hours since 1959-01-01"``). Uses ``cftime`` (an Earth2Studio dependency) so odd
    reference dates and units are handled the same way the rest of the ecosystem decodes
    them. A non-standard calendar (``360_day`` / ``noleap``) yields ``cftime`` objects that
    do not fit E2S's ``datetime64[ns]`` coordinate contract, so the cast raises here -- the
    right place to fail for an out-of-contract analysis store.
    """
    dates = cftime.num2date(
        values, units, calendar=calendar, only_use_cftime_datetimes=False
    )
    return np.asarray(dates, dtype="datetime64[ns]")


def batch_to_xcoords(
    batch: Batch,
    *,
    labels: list[list[str]],
    variables: VariableArray,
    lead_time: np.ndarray,
    time: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    transpose_inner: bool = False,
    device: torch.device | str = "cpu",
    levels: list[int | None] | None = None,
) -> tuple[torch.Tensor, CoordSystem]:
    """Convert one insitubatch ``Batch`` to the ``fetch_data(legacy=True)`` contract.

    ``labels`` is a ``[lead][variable]`` grid of the batch keys (each a sample-axis
    ``shift`` view of one stored array); the returned tensor has the model input layout
    ``(time, lead_time, variable, lat, lon)`` and ``coords`` is the matching ``OrderedDict``
    in that exact key order -- ``time`` (datetime64[ns]), ``lead_time`` (timedelta64[ns]),
    ``variable`` (str), ``lat``/``lon`` (float32) -- so it drops straight into
    ``prognostic.create_iterator`` after ``map_coords``. ``transpose_inner`` swaps the two
    field axes when the store lays fields out ``(lon, lat)`` but the contract wants
    ``(lat, lon)``. ``levels`` gives a per-variable index into a level-dimensioned array's
    second axis (``None`` for an array that already holds one field per sample).
    """
    tensors = to_torch(batch)  # {label: (n_time, *inner)} via zero-copy DLPack

    def field(li: int, vi: int) -> torch.Tensor:
        """The ``(n_time, *field)`` slice for one (lead, variable), level-selected."""
        t = tensors[labels[li][vi]]
        j = None if levels is None else levels[vi]
        # A level-dimensioned array arrives whole -- (n_time, n_level, *field) -- because the
        # stored chunk holds every level anyway; selecting here costs no extra read, and lets
        # several channels share one decode of one array.
        return t if j is None else t[:, j]

    # (n_time, var, *field) per lead -> stack the lead axis -> (n_time, lead, var, *field).
    per_lead = [
        torch.stack([field(li, vi) for vi in range(len(variables))], dim=1)
        for li in range(len(lead_time))
    ]
    x = torch.stack(per_lead, dim=1)
    if transpose_inner:
        x = x.transpose(-1, -2)
    x = x.contiguous().to(device)
    coords: CoordSystem = OrderedDict(
        [
            ("time", np.asarray(time, dtype="datetime64[ns]")),
            ("lead_time", np.asarray(lead_time, dtype="timedelta64[ns]")),
            ("variable", np.asarray(variables)),
            ("lat", np.asarray(lat, dtype=np.float32)),
            ("lon", np.asarray(lon, dtype=np.float32)),
        ]
    )
    return x, coords


def _level_indices(
    store: Store,
    arrays: list[str],
    wanted: list[int | None],
    level_name: str,
) -> list[int | None]:
    """Resolve each channel's requested level to an index into its array's level axis.

    A ``var_map`` value of ``"geopotential::500"`` asks for level 500 of a
    ``(sample, level, *field)`` array; the index comes from the store's own level
    coordinate, so the caller names the level in physical units rather than by position.
    Returns ``None`` for a channel whose array already holds one field per sample.
    """
    if all(w is None for w in wanted):
        return [None] * len(wanted)
    group = zarr.open_group(store=store, mode="r")
    if level_name not in list(group.array_keys()):
        raise ValueError(
            f"a var_map entry requested a level, but the store has no {level_name!r} "
            f"coordinate to resolve it against; pass level_name= if it is named differently"
        )
    coord = np.asarray(group[level_name][:])
    out: list[int | None] = []
    for array, want in zip(arrays, wanted, strict=True):
        if want is None:
            out.append(None)
            continue
        hit = np.flatnonzero(coord == want)
        if hit.size == 0:
            raise ValueError(
                f"level {want} requested for array {array!r} is not in the store's "
                f"{level_name!r} coordinate {coord.tolist()}"
            )
        out.append(int(hit[0]))
    return out


@check_optional_dependencies()
class InSituForecastFeed:
    """Iterate ``(x, coords)`` batches over a hindcast window, prefetched and de-duplicated.

    Reads the analysis ``store`` with insitubatch over a contiguous ``sample_range`` of the
    time axis, yielding ``batch_size`` consecutive init times per step with async read-ahead
    bounded by ``max_inflight``. ``lead_times`` populates the ``lead_time`` axis: pass a
    model's ``input_coords()["lead_time"]`` (values ``<= 0``) for a multi-step history
    window, verification leads (``> 0``) for scoring, or their union. Each lead is a
    sample-axis ``shift`` view of one stored array, so the ``(init, lead)`` grid decodes each
    shared chunk exactly once (the win over per-``(time, lead)`` ``fetch_data``).

    Init times whose leads would read past either end of the store are rejected with a
    ``ValueError`` (rather than silently dropped); with ``sample_range`` unset the feed spans
    exactly the in-bounds init window for the requested leads.

    ``variables`` are the ids to expose on the ``variable`` coordinate; ``var_map`` maps them
    to store array names when they differ (e.g. ``t2m -> 2m_temperature``). A value may name a
    level as ``"array::level"`` (e.g. ``z500 -> "geopotential::500"``) to select one level of a
    ``(sample, level, *field)`` array -- the spelling :class:`WB2Lexicon` already uses, so its
    vocabulary can be passed through unchanged. Channels sharing an array share one read: a
    stored chunk holds every level anyway, so U-CAST's 83 channels cost the 11 arrays that
    hold them, not 83. Build ``store`` with :func:`insitubatch.obstore_store` /
    :func:`insitubatch.fsspec_store` (e.g. anon public buckets). ``self.dataset`` exposes the
    underlying :class:`InSituDataset` for its ``cache_hits`` / ``cache_misses`` /
    ``resident_peak`` counters.

    Setting ``cache_dir`` turns on a **cross-run persistent cache**: the decoded chunks a run
    touches are written there (decode-once, no reshard) and a later run over the same store
    reads them from local disk as ``cache_hits`` instead of re-fetching the cloud. Because a
    reanalysis store is static, this is a drop-in replacement for a pre-download step when the
    *same* ground truth is scored repeatedly (many models, one fixed verification set). The
    path is the cache identity -- use a fresh ``cache_dir`` when the store changes.

    **One process per ``cache_dir``.** insitubatch takes an exclusive advisory lock on the
    directory for the feed's lifetime, so a second process pointed at the same path fails at
    construction naming the holder's PID and host, rather than the two of them corrupting each
    other's chunk files. Give each concurrently running job its own ``cache_dir`` (or run them
    in sequence); the lock is released by the kernel when a process dies, including under
    ``SIGKILL`` and spot preemption, so there is no stale lock to clean up. Adding a variable
    to a later run is *not* a cache reset -- an array with no entries is cold, not stale, and
    arrays a run does not read keep their files -- so several variable subsets can share one
    directory sequentially.

    ``readonly_cache=True`` is the many-scorers half of that workflow: one job warms the
    cache, then any number of scoring jobs open it **read-only**. They take the directory
    lock *shared*, so they coexist with each other (a writer still does not), they write
    nothing, and a cache **miss raises** instead of quietly reaching for the cloud. That is
    what makes it a contract -- *this cache is complete for what I am about to read* -- and
    it is what you want when a campaign's cost model assumes no egress: a warming run whose
    split or variable set was narrower than the scoring run's is then an error at the chunk
    it first needs, not a surprise bill.

    The store's time axis must be **uniformly spaced**: leads are mapped to sample-axis steps
    through a single ``dt``, so an irregular axis would silently score against the wrong valid
    times. It is validated at construction.
    """

    def __init__(
        self,
        store: Store,
        variables: VariableArray,
        *,
        var_map: dict[str, str] | None = None,
        lead_times: np.ndarray | None = None,
        time_name: str = "time",
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        level_name: str = "level",
        sample_range: tuple[int, int] | None = None,
        batch_size: int = 8,
        max_inflight: int | None = None,
        cache_dir: str | None = None,
        readonly_cache: bool = False,
        transpose_inner: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.store = store
        self.variables = [str(v) for v in variables]
        self.device = device
        self.transpose_inner = transpose_inner
        if readonly_cache and cache_dir is None:
            raise ValueError(
                "readonly_cache=True needs a cache_dir -- it is an assertion that a warmed "
                "cache is complete for this run, and there is no cache to read without one"
            )
        vmap = var_map or {v: v for v in self.variables}
        if missing := [v for v in self.variables if v not in vmap]:
            raise ValueError(
                f"var_map has no entry for {missing}; it must name a store array for every "
                f"id in `variables`. Given: {sorted(vmap)}. Omit var_map entirely when the "
                "ids already are the store's array names."
            )

        group = zarr.open_group(store=store, mode="r")
        time_arr = np.asarray(group[time_name][:])
        attrs = dict(group[time_name].attrs)
        units = attrs.get("units")
        self.time = (
            decode_cf_time(time_arr, units, attrs.get("calendar", "standard"))
            if units
            else time_arr.astype("datetime64[ns]")
        )
        self.lat = np.asarray(group[lat_name][:]).astype(np.float32)
        self.lon = np.asarray(group[lon_name][:]).astype(np.float32)

        # Sample-axis step of the store (dt); every lead must be an integer multiple of it.
        # A lead is mapped to sample-axis steps through this one dt, so a store whose time
        # axis is irregular (a gap, a resolution change, a concatenation seam) would map
        # leads onto the wrong valid times -- silently, and wrongly only for the inits after
        # the seam. Validate the whole axis rather than trusting its first two entries. The
        # length check is the same guard: a 1-step axis has no dt to read.
        if self.time.size < 2:
            raise ValueError(
                f"the store's `{time_name}` axis has {self.time.size} step(s); a lead axis "
                "needs at least two to establish the sample-axis step"
            )
        steps_between = np.diff(self.time)
        dt = steps_between[0]
        if not np.all(steps_between == dt):
            bad = int(np.argmax(steps_between != dt))
            raise ValueError(
                f"the store's `{time_name}` axis is not uniformly spaced: step {bad} is "
                f"{steps_between[bad]} against {dt} at the start "
                f"({self.time[bad]} -> {self.time[bad + 1]}). Leads are mapped to sample-axis "
                "steps through a single dt, so an irregular axis would silently read the "
                "wrong valid times; slice the store to a uniform window and pass "
                "`sample_range` instead."
            )
        leads = (
            np.array([np.timedelta64(0, "ns")])
            if lead_times is None
            else np.asarray(lead_times)
        )
        self.lead_time = leads.astype("timedelta64[ns]")
        steps = self.lead_time / dt
        if not np.all(steps == np.round(steps)):
            raise ValueError(
                f"every lead_time must be an integer multiple of the store step {dt}; got {leads}"
            )
        self.lead_steps = np.round(steps).astype(np.int64)

        # ``var_map`` values may name a level: "geopotential::500" selects level 500 of a
        # (sample, level, *field) array, matching the spelling in WB2Lexicon. Several channels
        # may name one array (z50..z1000 are 13 channels of `geopotential`); they share one
        # geometry, so the array is read and decoded once for all of them.
        specs = [vmap[v].split("::", 1) for v in self.variables]
        arrays = [spec[0] for spec in specs]
        # A trailing "::" with no level is how WB2Lexicon spells a surface variable, so an
        # empty level reads as "no level axis" rather than as a parse error -- which lets a
        # caller hand this feed `WB2Lexicon.VOCAB` entries unchanged.
        wanted = [
            int(spec[1]) if len(spec) == 2 and spec[1] != "" else None for spec in specs
        ]
        # One array cannot be both level-selected and taken whole: the two shapes differ, and
        # the channels share a read, so catch it here rather than in a stack() shape error.
        for a in set(arrays):
            uses = {
                w is None for w, arr in zip(wanted, arrays, strict=True) if arr == a
            }
            if len(uses) > 1:
                raise ValueError(
                    f"array {a!r} is mapped both with and without a level; every channel "
                    f"reading one array must select a level, or none of them may"
                )
        opened = open_geometries(store, variables=sorted(set(arrays)))
        self.levels = _level_indices(store, arrays, wanted, level_name)

        # Each lead shifts the read to anchor + step, so init times near a store edge whose
        # shifted read would leave [0, n_samples) are unusable. valid_anchor_range gives the
        # in-bounds init window for these leads; the engine would silently drop out-of-range
        # anchors, so validate here instead of scoring a shorter window than requested.
        n_samples = opened[arrays[0]].n_samples
        lo, hi = valid_anchor_range(self.lead_steps.tolist(), n_samples)
        if lo >= hi:
            raise ValueError(
                f"lead steps {self.lead_steps.tolist()} span more than the store's "
                f"{n_samples} samples; no init time can satisfy every lead"
            )
        if sample_range is None:
            sample_range = (lo, hi)  # every init whose leads all fall within the store
        elif sample_range[0] < lo or sample_range[1] > hi:
            raise ValueError(
                f"sample_range {sample_range} with lead steps {self.lead_steps.tolist()} "
                f"reads outside the store [0, {n_samples}); the valid init range for these "
                f"leads is [{lo}, {hi})"
            )

        # One shifted geometry per (lead, array); the label grid indexes them for the stacker.
        # Keying by array rather than by channel is what makes 83 channels over 11 arrays cost
        # 11 reads per lead rather than 83.
        geometries: dict[str, object] = {}
        self.labels: list[list[str]] = []
        for li, k in enumerate(self.lead_steps):
            row = []
            for a in arrays:
                label = f"{a}#{li}"
                geometries[label] = opened[a].shift(int(k))
                row.append(label)
            self.labels.append(row)

        manifest = split_by_chunk(
            opened[arrays[0]], fractions=(1.0, 0.0, 0.0), sample_range=sample_range
        )
        self.dataset = InSituDataset(
            store,
            manifest,
            geometries=geometries,  # type: ignore[arg-type]
            batch_size=batch_size,
            shuffle=False,
            cache_dir=cache_dir,
            # cache_dir set => cross-run persistent cache (not just an in-run spill tier)
            persist=cache_dir is not None,
            readonly_cache=readonly_cache,
            max_inflight=max_inflight,
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        self.dataset.set_epoch(0)
        for batch in self.dataset.all:
            yield batch_to_xcoords(
                batch,
                labels=self.labels,
                variables=self.variables,
                lead_time=self.lead_time,
                time=self.time[batch.sample_indices],
                lat=self.lat,
                lon=self.lon,
                transpose_inner=self.transpose_inner,
                device=self.device,
                levels=self.levels,
            )

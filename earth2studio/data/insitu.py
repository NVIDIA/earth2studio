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
``prep_data_array``, and ``fetch_data`` gathers one read per ``(time, lead_time)`` with
no de-duplication. For an IO-bound hindcast / scoring campaign the ``(init, lead)`` grid
maps many requested slices onto the **same** stored chunk (consecutive init times share
valid times; a fat time-chunk holds several steps), so that path re-reads and re-decodes
the same bytes over and over.

This module skips xarray: it reads the analysis store with **insitubatch**
(:class:`~insitubatch.source.InSituDataset` -- bounded-fan-out async prefetch, a read plan
that decodes each shared chunk once) and converts each numpy ``Batch`` to the exact
``(x, coords)`` tuple ``fetch_data(..., legacy=True)`` returns, so it is a drop-in for the
initial-condition feed of ``earth2studio.run`` workflows.

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
except ImportError as exc:  # pragma: no cover - optional integration
    raise ImportError(
        "earth2studio.data.insitu needs insitubatch; install it with: pip install insitubatch"
    ) from exc


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
) -> tuple[torch.Tensor, CoordSystem]:
    """Convert one insitubatch ``Batch`` to the ``fetch_data(legacy=True)`` contract.

    ``labels`` is a ``[lead][variable]`` grid of the batch keys (each a sample-axis
    ``shift`` view of one stored array); the returned tensor has the model input layout
    ``(time, lead_time, variable, lat, lon)`` and ``coords`` is the matching ``OrderedDict``
    in that exact key order -- ``time`` (datetime64[ns]), ``lead_time`` (timedelta64[ns]),
    ``variable`` (str), ``lat``/``lon`` (float32) -- so it drops straight into
    ``prognostic.create_iterator`` after ``map_coords``. ``transpose_inner`` swaps the two
    field axes when the store lays fields out ``(lon, lat)`` but the contract wants
    ``(lat, lon)``.
    """
    tensors = to_torch(batch)  # {label: (n_time, *inner)} via zero-copy DLPack
    # (n_time, var, *inner) per lead -> stack the lead axis -> (n_time, lead, var, *inner).
    per_lead = [
        torch.stack([tensors[labels[li][vi]] for vi in range(len(variables))], dim=1)
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
    to store array names when they differ (e.g. ``t2m -> 2m_temperature``). Build ``store``
    with :func:`insitubatch.obstore_store` / :func:`insitubatch.fsspec_store` (e.g. anon
    public buckets). ``self.dataset`` exposes the underlying :class:`InSituDataset` for its
    ``cache_hits`` / ``cache_misses`` / ``resident_peak`` counters.

    Setting ``cache_dir`` turns on a **cross-run persistent cache**: the decoded chunks a run
    touches are written there (decode-once, no reshard) and a later run over the same store
    reads them from local disk as ``cache_hits`` instead of re-fetching the cloud. Because a
    reanalysis store is static, this is a drop-in replacement for a pre-download step when the
    *same* ground truth is scored repeatedly (many models, one fixed verification set). The
    path is the cache identity -- use a fresh ``cache_dir`` when the store or variables change.
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
        sample_range: tuple[int, int] | None = None,
        batch_size: int = 8,
        max_inflight: int | None = None,
        cache_dir: str | None = None,
        transpose_inner: bool = False,
        device: torch.device | str = "cpu",
    ) -> None:
        self.store = store
        self.variables = [str(v) for v in variables]
        self.device = device
        self.transpose_inner = transpose_inner
        vmap = var_map or {v: v for v in self.variables}

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
        dt = self.time[1] - self.time[0]
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

        arrays = [vmap[v] for v in self.variables]
        opened = open_geometries(store, variables=arrays)

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

        # One shifted geometry per (lead, variable); label grid indexes them for the stacker.
        geometries: dict[str, object] = {}
        self.labels: list[list[str]] = []
        for li, k in enumerate(self.lead_steps):
            row = []
            for v in self.variables:
                label = f"{v}#{li}"
                geometries[label] = opened[vmap[v]].shift(int(k))
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
            )

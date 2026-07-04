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
``prep_data_array``. For an IO-bound hindcast / scoring campaign over many init times, that
per-``(time, variable)`` fetch re-reads overlapping chunks and its ``gather`` is unbounded.

This module skips xarray entirely: it reads the analysis store with **insitubatch**
(`InSituDataset` -- bounded-fan-out async prefetch, a read plan that de-duplicates chunks
shared across init times and lead offsets) and converts each numpy ``Batch`` to the exact
``(x, coords)`` tuple ``fetch_data(..., legacy=True)`` returns, so it is a drop-in for the
initial-condition feed of ``earth2studio.run`` workflows. See :func:`batch_to_xcoords` for
the contract and :class:`InSituForecastFeed` for the prefetched iterator.
"""

from collections import OrderedDict

import numpy as np
import torch
import zarr

from earth2studio.utils.type import CoordSystem, TimeArray, VariableArray

try:
    from insitubatch import obstore_store, open_geometries, split_by_chunk
    from insitubatch.frameworks import to_torch
    from insitubatch.source import InSituDataset
    from insitubatch.types import Batch
except ImportError as exc:  # pragma: no cover - optional integration
    raise ImportError(
        "earth2studio.data.insitu needs insitubatch; install it with: pip install insitubatch"
    ) from exc


def batch_to_xcoords(
    batch: Batch,
    *,
    variables: VariableArray,
    time: TimeArray,
    lat: np.ndarray,
    lon: np.ndarray,
    device: torch.device | str = "cpu",
    lead_time: np.ndarray | None = None,
) -> tuple[torch.Tensor, CoordSystem]:
    """Convert one insitubatch ``Batch`` to the ``fetch_data(legacy=True)`` contract.

    The batch carries one ``(n_time, lat, lon)`` array per variable, keyed by the
    Earth2Studio variable id used to build the dataset. The returned tensor has the model's
    input layout ``(time, lead_time, variable, lat, lon)`` and ``coords`` is the matching
    ``OrderedDict`` in that exact key order -- ``time`` (datetime64[ns]), ``lead_time``
    (timedelta64[ns]), ``variable`` (str), ``lat``/``lon`` (float32) -- so it drops straight
    into ``prognostic.create_iterator`` after ``map_coords``.

    ``lead_time`` defaults to a single 0 h step (an initial condition); pass the model's
    ``input_coords()["lead_time"]`` for a multi-step history window.
    """
    variables = np.asarray(variables)
    lead = np.array([np.timedelta64(0, "ns")]) if lead_time is None else np.asarray(lead_time)
    tensors = to_torch(batch)  # {var: (n_time, lat, lon)} via zero-copy DLPack
    # (n_time, variable, lat, lon) -> insert the size-1 lead axis -> (n_time, lead, var, lat, lon)
    x = torch.stack([tensors[str(v)] for v in variables], dim=1).unsqueeze(1).to(device)
    coords: CoordSystem = OrderedDict(
        [
            ("time", np.asarray(time, dtype="datetime64[ns]")),
            ("lead_time", lead.astype("timedelta64[ns]")),
            ("variable", variables),
            ("lat", np.asarray(lat, dtype=np.float32)),
            ("lon", np.asarray(lon, dtype=np.float32)),
        ]
    )
    return x, coords


class InSituForecastFeed:
    """Iterate initial-condition ``(x, coords)`` batches for a hindcast window, prefetched.

    Reads the analysis ``store`` with insitubatch over a contiguous ``sample_range`` of the
    time axis, batching ``batch_size`` consecutive init times per step with async read-ahead
    bounded by ``max_inflight``. Each batch is converted to the ``fetch_data`` contract and
    fed to ``prognostic.create_iterator``; while the model rolls out one batch, the loader
    prefetches the next batch's ICs. ``variables`` are Earth2Studio ids; ``var_map`` maps them
    to store array names when they differ (e.g. WB2 ``t2m -> 2m_temperature``).
    """

    def __init__(
        self,
        store_url: str,
        variables: VariableArray,
        *,
        var_map: dict[str, str] | None = None,
        time_name: str = "time",
        lat_name: str = "latitude",
        lon_name: str = "longitude",
        sample_range: tuple[int, int] | None = None,
        batch_size: int = 8,
        max_inflight: int | None = None,
        cache_dir: str | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.variables = [str(v) for v in variables]
        self.device = device
        vmap = var_map or {v: v for v in self.variables}
        store = obstore_store(store_url)

        group = zarr.open_group(store=store, mode="r")
        self.time = np.asarray(group[time_name][:]).astype("datetime64[ns]")
        self.lat = np.asarray(group[lat_name][:]).astype(np.float32)
        self.lon = np.asarray(group[lon_name][:]).astype(np.float32)

        arrays = [vmap[v] for v in self.variables]
        opened = open_geometries(store, variables=arrays)
        geometries = {v: opened[vmap[v]] for v in self.variables}
        manifest = split_by_chunk(
            opened[arrays[0]], fractions=(1.0, 0.0, 0.0), sample_range=sample_range
        )
        self.dataset = InSituDataset(
            store,
            manifest,
            geometries=geometries,
            batch_size=batch_size,
            shuffle=False,
            cache_dir=cache_dir,
            max_inflight=max_inflight,
        )

    def __iter__(self):  # type: ignore[no-untyped-def]
        self.dataset.set_epoch(0)
        for batch in self.dataset.train:
            time = self.time[batch.sample_indices]
            yield batch_to_xcoords(
                batch,
                variables=self.variables,
                time=time,
                lat=self.lat,
                lon=self.lon,
                device=self.device,
            )

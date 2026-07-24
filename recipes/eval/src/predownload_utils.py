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

"""Shared helpers used by pipelines to declare predownload requirements.

These were previously private module-level functions in ``predownload.py``.
Moving them into a neutral utility module lets concrete ``Pipeline``
subclasses reuse them to build :class:`~src.pipelines.PredownloadStore`
entries without a circular import back into the entry-point script.
``predownload.py`` re-exports these under their original names for
backward compatibility with existing tests and any callers that imported
the private helpers.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.data import fetch_data
from earth2studio.utils.coords import CoordSystem

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from .pipelines.base import PredownloadStore


class _RegriddedDataSource:
    """Wraps a DataSource, applying xarray.interp to regrid output to target lat/lon.

    Used when the data source's native grid doesn't match the zarr store schema
    (e.g. ARCO at 721×1440 downloaded into a store sized for a 240×121 model).
    Only lat/lon dimensions are interpolated; all other dims pass through unchanged.
    """

    def __init__(
        self,
        source: object,
        target_lat: np.ndarray,
        target_lon: np.ndarray,
    ) -> None:
        self._source = source
        self._lat = xr.DataArray(np.asarray(target_lat), dims="lat")
        self._lon = xr.DataArray(np.asarray(target_lon), dims="lon")

    def __call__(self, time: object, variable: object) -> xr.DataArray:
        da = self._source(time, variable)  # type: ignore[operator]
        if "lat" in da.dims and "lon" in da.dims:
            da = da.interp(lat=self._lat, lon=self._lon, method="linear")
        return da

    async def fetch(self, time: object, variable: object) -> xr.DataArray:
        return self(time, variable)


def _align_source_to_spatial_ref(
    source: object,
    spatial_ref: CoordSystem,
    probe_time: np.datetime64,
    probe_variable: str,
) -> object:
    """Return *source* (possibly wrapped) so its output matches *spatial_ref*'s lat/lon.

    Probes *source* for one field to discover its native spatial grid.  When
    the native lat array differs from *spatial_ref*'s lat, wraps with
    :class:`_RegriddedDataSource` so every subsequent fetch is bilinearly
    interpolated to the target grid before being written to zarr.  When the
    grids already match, returns *source* unchanged (no overhead at fetch time).
    """
    tgt_lat = spatial_ref.get("lat")
    tgt_lon = spatial_ref.get("lon")
    if tgt_lat is None or tgt_lon is None:
        return source

    _, probe = fetch_data(
        source=source,  # type: ignore[arg-type]
        time=[probe_time],
        variable=[probe_variable],
        lead_time=np.array([np.timedelta64(0, "ns")]),
        device=torch.device("cpu"),
    )

    src_lat = probe.get("lat")
    if src_lat is None:
        return source

    tgt_lat = np.asarray(tgt_lat)
    if src_lat.shape == tgt_lat.shape and np.allclose(src_lat, tgt_lat, atol=1e-5):
        return source  # grids already match

    src_shape = (src_lat.shape[0], probe.get("lon", np.array([])).shape[0])
    tgt_shape = (tgt_lat.shape[0], np.asarray(tgt_lon).shape[0])
    logger.info(
        f"Predownload: source grid {src_shape} ≠ target {tgt_shape}; "
        "wrapping with lat/lon interpolation."
    )
    return _RegriddedDataSource(source, tgt_lat, np.asarray(tgt_lon))


def infer_step_hours(model: object) -> int:
    """Infer the model's output timestep in hours from its coordinate methods.

    Computed as the difference between the first output lead time and the last
    input lead time, which equals the model's intrinsic step size.
    """
    ic_coords = model.input_coords()  # type: ignore[attr-defined]
    out_coords = model.output_coords(ic_coords)  # type: ignore[attr-defined]
    delta = out_coords["lead_time"][0] - ic_coords["lead_time"][-1]
    return int(delta / np.timedelta64(1, "h"))


def compute_verification_times(
    ic_times: list[np.datetime64],
    nsteps: int,
    step_hours: int,
) -> list[np.datetime64]:
    """Collect all unique model-output valid times across every IC window."""
    step = np.timedelta64(step_hours, "h")
    times: set[np.datetime64] = set()
    for t in ic_times:
        for k in range(nsteps + 1):
            times.add(t + k * step)
    return sorted(times)


def union_variables(*var_lists: list[str]) -> list[str]:
    """Return the ordered union of multiple variable lists."""
    seen: set[str] = set()
    result: list[str] = []
    for vl in var_lists:
        for v in vl:
            if v not in seen:
                result.append(v)
                seen.add(v)
    return result


def single_source_stores_disabled(cfg: DictConfig) -> bool:
    """Return True when no single-source predownload stores are needed.

    Lets pipelines short-circuit before doing expensive model inspection:
    if both IC and verification are BYO (or verification is disabled),
    :func:`declare_single_source_stores` will always return an empty list.
    """
    ic_byo = cfg.get("ic_source") is not None
    verif_byo = cfg.get("verification_source") is not None
    pd_cfg = cfg.get("predownload", {})
    verif_cfg = pd_cfg.get("verification", {}) if pd_cfg else {}
    verif_enabled = verif_cfg.get("enabled", False)
    return ic_byo and (verif_byo or not verif_enabled)


def declare_single_source_stores(
    cfg: DictConfig,
    *,
    ic_variables: list[str],
    ic_times: list[np.datetime64],
    verif_variables: list[str],
    verif_times: list[np.datetime64],
    spatial_ref: CoordSystem,
    always_separate_verification: bool = False,
) -> list[PredownloadStore]:
    """Declare IC and verification predownload stores for single-source pipelines.

    Implements the shared IC/verification store resolution used by
    :class:`~src.pipelines.forecast.ForecastPipeline`,
    :class:`~src.pipelines.forecast.DiagnosticPipeline`, and
    :class:`~src.pipelines.dlesym.DLESyMPipeline`.  Returns 0, 1, or 2
    :class:`~src.pipelines.base.PredownloadStore` entries based on:

    * ``cfg.ic_source`` BYO override (skip the IC / ``data.zarr`` store)
    * ``cfg.verification_source`` BYO override (skip the verification store)
    * ``cfg.predownload.verification.enabled`` (gate verification entirely)
    * ``cfg.predownload.verification.source`` — when absent and both stores
      are enabled, the verification slice is *merged* into ``data.zarr``
      (variables and times are unioned).  When present, verification goes
      into its own ``verification.zarr`` instantiated from that source.
    * *always_separate_verification* — when ``True``, never merge: used by
      pipelines whose IC and verification variable sets rarely overlap.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config — must provide ``data_source``, optional
        ``ic_source``, ``verification_source``, and ``predownload`` blocks.
    ic_variables, verif_variables : list[str]
        Variable names required for IC fetching and verification, respectively.
    ic_times, verif_times : list[np.datetime64]
        Times to fetch for IC and verification, respectively.
    spatial_ref : CoordSystem
        Spatial reference (output coords) shared by both stores.
    always_separate_verification : bool
        Force ``verification`` into its own store even when verification
        would otherwise be merged into ``data.zarr``.

    Returns
    -------
    list[PredownloadStore]
    """
    # Lazy import to avoid a cycle (pipelines/base.py imports nothing here,
    # but some pipeline modules that use this helper may chain through
    # predownload.py).
    from .pipelines.base import PredownloadStore

    ic_byo = cfg.get("ic_source") is not None
    verif_byo = cfg.get("verification_source") is not None
    pd_cfg = cfg.get("predownload", {})
    verif_cfg = pd_cfg.get("verification", {}) if pd_cfg else {}
    verif_enabled = verif_cfg.get("enabled", False)
    verif_has_separate_source = verif_cfg.get("source") is not None

    do_verif = verif_enabled and not verif_byo
    do_data_store = not ic_byo
    do_merged = (
        do_data_store
        and do_verif
        and not verif_has_separate_source
        and not always_separate_verification
    )
    do_verif_store = do_verif and not do_merged

    if not do_data_store and not do_verif_store:
        return []

    stores: list[PredownloadStore] = []
    data_source = None

    if do_data_store:
        data_source = hydra.utils.instantiate(cfg.data_source)
        if do_merged:
            times = sorted(set(ic_times) | set(verif_times))
            variables = union_variables(ic_variables, verif_variables)
        else:
            times = list(ic_times)
            variables = list(ic_variables)
        if times and variables:
            data_source = _align_source_to_spatial_ref(
                data_source, spatial_ref, times[0], variables[0]
            )
        stores.append(
            PredownloadStore(
                name="data",
                source=data_source,
                times=times,
                variables=variables,
                spatial_ref=spatial_ref,
                role="ic",
            )
        )

    if do_verif_store:
        verif_source_cfg = verif_cfg.get("source")
        if verif_source_cfg is not None:
            verif_source = hydra.utils.instantiate(verif_source_cfg)
            if verif_times and verif_variables:
                verif_source = _align_source_to_spatial_ref(
                    verif_source, spatial_ref, verif_times[0], verif_variables[0]
                )
        elif data_source is not None:
            # Reuse the already-aligned IC source (already wrapped if needed).
            verif_source = data_source
        else:
            verif_source = hydra.utils.instantiate(cfg.data_source)
            if verif_times and verif_variables:
                verif_source = _align_source_to_spatial_ref(
                    verif_source, spatial_ref, verif_times[0], verif_variables[0]
                )
        stores.append(
            PredownloadStore(
                name="verification",
                source=verif_source,
                times=list(verif_times),
                variables=list(verif_variables),
                spatial_ref=spatial_ref,
                role="verification",
            )
        )

    return stores


def declare_verification_only_store(
    cfg: DictConfig,
    *,
    verif_variables: list[str],
    verif_times: list[np.datetime64],
    spatial_ref: CoordSystem,
) -> list[PredownloadStore]:
    """Declare a standalone ``verification.zarr`` store (no IC store).

    Used by pipelines whose inputs don't come from a gridded ``DataSource``
    at all (e.g. data-assimilation pipelines whose inputs are observation
    DataFrames) but that still need gridded verification data for scoring.
    Honors the same gates as :func:`declare_single_source_stores`:
    ``predownload.verification.enabled``, the ``verification_source`` BYO
    override, and the optional ``predownload.verification.source`` (falling
    back to ``cfg.data_source``).

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    verif_variables : list[str]
        Variable names required for verification.
    verif_times : list[np.datetime64]
        Valid times to fetch.
    spatial_ref : CoordSystem
        Spatial coordinate system of the stored data.  Must match the
        grid the verification source returns — the store schema is
        validated against fetched data at download time.

    Returns
    -------
    list[PredownloadStore]
        Zero or one entries.
    """
    from .pipelines.base import PredownloadStore

    verif_byo = cfg.get("verification_source") is not None
    pd_cfg = cfg.get("predownload", {})
    verif_cfg = pd_cfg.get("verification", {}) if pd_cfg else {}
    verif_enabled = verif_cfg.get("enabled", False)

    if not verif_enabled or verif_byo:
        return []

    verif_source_cfg = verif_cfg.get("source")
    if verif_source_cfg is not None:
        verif_source = hydra.utils.instantiate(verif_source_cfg)
    else:
        verif_source = hydra.utils.instantiate(cfg.data_source)

    if verif_times and verif_variables:
        verif_source = _align_source_to_spatial_ref(
            verif_source, spatial_ref, verif_times[0], verif_variables[0]
        )

    return [
        PredownloadStore(
            name="verification",
            source=verif_source,
            times=list(verif_times),
            variables=list(verif_variables),
            spatial_ref=spatial_ref,
            role="verification",
        )
    ]


def squeeze_lead_time(
    x: torch.Tensor, coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Remove the lead_time dimension from a (tensor, coords) pair.

    ``fetch_data()`` always returns a ``lead_time`` dimension even when it
    contains a single ``[0ns]`` entry.  The predownload stores have no
    ``lead_time`` dimension, so we squeeze it out before writing.
    """
    lt_idx = list(coords.keys()).index("lead_time")
    x = x.squeeze(lt_idx)
    coords = OrderedDict((k, v) for k, v in coords.items() if k != "lead_time")
    return x, coords

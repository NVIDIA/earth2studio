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

"""Coupled GOES/MRMS nowcasting pipeline using StormScope models."""

from __future__ import annotations

import inspect
import os
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.utils.coords import CoordSystem, cat_coords

from ..data import (
    CadenceRoundedSource,
    CompositeSource,
    PredownloadedSource,
    ValidTimeForecastAdapter,
    resolve_ic_source,
)
from ..regrid import BilinearRegridder, NearestNeighborRegridder, RegriddedSource
from ..work import WorkItem
from .base import Pipeline, PredownloadStore


class StormScopePipeline(Pipeline):
    """Coupled GOES/MRMS nowcasting pipeline using StormScope models.

    Runs two prognostic models together (default variant ``3km_10min``):

    * :class:`earth2studio.models.px.StormScopeGOES` — forecasts the GOES
      ABI satellite channels.  The ``3km_10min`` variant is *pure obs*
      (no external conditioning); legacy 60-min variants condition on
      synoptic-scale GFS data.
    * :class:`earth2studio.models.px.StormScopeMRMS` — forecasts MRMS
      reflectivity (``refc``, ``refc_base``) plus a gridded GLM lightning
      channel (``glm_density``), conditioned on GOES (either observations
      at IC time or the GOES model's own predictions during rollout).

    Both models share the HRRR Lambert-conformal grid (``y``, ``x``)
    and the pipeline yields a combined ``(variables × y × x)`` tensor
    per forecast step — GOES channels, MRMS radar, and GLM stacked along
    the ``variable`` axis so the output zarr is a single unified store.

    Config shape
    ------------
    Expected structure under ``cfg.model`` (see
    ``cfg/model/stormscope_goes_mrms.yaml``)::

        goes:
            architecture: earth2studio.models.px.StormScopeGOES
            load_args:
                model_name: 3km_10min
                conditioning_data_source: null      # pure-obs
            ic_source: {_target_: earth2studio.data.GOES, satellite: goes16, scan_mode: C}
            ic_grid: {_target_: src.grids.goes_grid, satellite: goes16, scan_mode: C}
        mrms:
            architecture: earth2studio.models.px.StormScopeMRMS
            load_args:
                model_name: 3km_10min
                conditioning_data_source: {_target_: earth2studio.data.GOES, ...}
                glm_data_source: {_target_: earth2studio.data.GOESGLMGrid, satellite: east}
            ic_source: {_target_: earth2studio.data.MRMS}
            ic_grid: {_target_: src.grids.mrms_grid}
            glm_grid: {_target_: src.grids.glm_grid, satellite: east}
            conditioning_grid: {_target_: src.grids.goes_grid, ...}
        max_dist_km: 30.0     # optional; passed to build_*_interpolator

    The pipeline caches the two IC sources during :meth:`setup` and
    uses them in :meth:`run_item`, ignoring the ``data_source`` argument
    that ``main.py`` wires in (which is a single-source abstraction).

    Ensembles
    ---------
    StormScope exposes ensemble diversity through the diffusion sampler's
    stochasticity.  We honor :attr:`cfg.ensemble_size` the same way
    other pipelines do — one :class:`WorkItem` per member, seeded
    deterministically.  The model's ``batch`` dimension is used
    internally (required by ``call_with_conditioning``) but is squeezed
    out of the yielded tensor so :meth:`Pipeline._inject_ensemble` can
    prepend a proper ``ensemble`` axis at write time.

    Predownload
    -----------
    :meth:`predownload_stores` declares one zarr per IC source
    (``data_goes.zarr``, ``data_mrms.zarr``) plus, for GLM-bearing
    variants, a ``data_glm.zarr`` for the ``glm_density`` state channel.
    Each is wrapped in a :class:`~src.regrid.RegriddedSource` that
    resamples the raw source onto the model's HRRR sub-region at write
    time — **nearest-neighbor** for radar/satellite, **bilinear** for the
    sparse GLM count field (matching
    :meth:`StormScopeMRMS.build_glm_interpolator`).  The resulting stores
    have ``(time, y, x)`` dims whose ``y``/``x`` values match the model's
    native grid exactly — so at inference time
    :meth:`StormScopeBase.prep_input` detects the match and skips its
    live interpolation.

    :meth:`setup` auto-detects the predownloaded stores under
    ``<output.path>/data_{goes,mrms,glm}.zarr`` and uses
    :class:`~src.data.PredownloadedSource` in their place.  The MRMS state
    spans two grids (radar + GLM), so its IC is reassembled from
    ``data_mrms.zarr`` + ``data_glm.zarr`` via a
    :class:`~src.data.CompositeSource`.  Users can skip predownload
    entirely with per-model BYO overrides (see :class:`StormScopePipeline`
    docstring) or by pointing to raw live sources and running with the
    model's live interpolators.
    """

    needs_data_source = False
    """StormScope resolves its own IC sources per-model from
    ``cfg.model.{goes,mrms}.ic_source`` — ``main.py`` should not
    instantiate a primary data source on our behalf."""

    _run_item_includes_batch_dim = True
    """StormScope's diffusion sampler output retains a size-1 ``batch``
    dim; :meth:`Pipeline.run` squeezes it before filtering and ensemble
    injection so :meth:`run_item` can yield raw model outputs."""

    model_goes: Any
    model_mrms: Any
    nsteps: int
    _goes_ic_source: Any
    _mrms_ic_source: Any
    _goes_ic_coords: CoordSystem
    _mrms_ic_coords: CoordSystem

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        if "model" not in cfg or "goes" not in cfg.model or "mrms" not in cfg.model:
            raise ValueError(
                "StormScopePipeline requires cfg.model.{goes, mrms} sub-blocks. "
                "See cfg/model/stormscope_goes_mrms.yaml for a template."
            )

        self.nsteps = cfg.nsteps
        max_dist_km = cfg.model.get("max_dist_km", None)

        # All ranks participate in model loading for barrier correctness.
        self.model_goes = _load_stormscope_model(cfg.model.goes).to(device)
        self.model_goes.eval()
        self.model_mrms = _load_stormscope_model(cfg.model.mrms).to(device)
        self.model_mrms.eval()

        # Build input + conditioning interpolators for each model.  The grid
        # resolvers return (lats, lons) via Hydra-instantiable helpers —
        # see src/grids.py.
        goes_in_lat, goes_in_lon = hydra.utils.instantiate(cfg.model.goes.ic_grid)
        self.model_goes.build_input_interpolator(
            goes_in_lat, goes_in_lon, max_dist_km=max_dist_km
        )
        mrms_in_lat, mrms_in_lon = hydra.utils.instantiate(cfg.model.mrms.ic_grid)
        self.model_mrms.build_input_interpolator(
            mrms_in_lat, mrms_in_lon, max_dist_km=max_dist_km
        )

        if cfg.model.goes.get("conditioning_grid") is not None:
            cgoes_lat, cgoes_lon = hydra.utils.instantiate(
                cfg.model.goes.conditioning_grid
            )
            self.model_goes.build_conditioning_interpolator(
                cgoes_lat, cgoes_lon, max_dist_km=max_dist_km
            )
        if cfg.model.mrms.get("conditioning_grid") is not None:
            cmrms_lat, cmrms_lon = hydra.utils.instantiate(
                cfg.model.mrms.conditioning_grid
            )
            self.model_mrms.build_conditioning_interpolator(
                cmrms_lat, cmrms_lon, max_dist_km=max_dist_km
            )

        # IC sources: prefer predownloaded, regridded zarrs when present
        # (<output.path>/data_{goes,mrms}.zarr).  Their y/x coords already
        # match self.model_{goes,mrms}.y/x so the models' prep_input skips
        # re-interpolation.  Fall back to the live ic_source otherwise.
        self._goes_ic_source = resolve_ic_source(
            cfg,
            store_name="data_goes.zarr",
            live_source=cfg.model.goes.ic_source,
        )
        self._mrms_ic_source = self._resolve_mrms_ic_source(cfg)

        # Conditioning source: swap in the predownloaded, regridded zarr
        # (<output.path>/cond_goes.zarr) when present so the GOES model's
        # internal fetch_conditioning reads locally instead of hitting a
        # live forecast source per step.  The conditioning zarr's y/x
        # match self.model_goes.y/x so prep_input's conditioning regrid
        # is also skipped.  MRMS is intentionally excluded — its
        # conditioning is supplied externally via call_with_conditioning.
        self._maybe_swap_conditioning(cfg, "goes", self.model_goes)

        # Cache input coords for efficiency in run_item.
        self._goes_ic_coords = self.model_goes.input_coords()
        self._mrms_ic_coords = self.model_mrms.input_coords()

        # Spatial reference = HRRR grid, with the unified output variable list
        # so that build_output_coords filters correctly against yielded tensors.
        output_vars = _concat_var_lists(
            list(self._goes_ic_coords["variable"]),
            list(self._mrms_ic_coords["variable"]),
        )
        self._spatial_ref = OrderedDict(
            [
                ("variable", np.array(output_vars)),
                ("y", _as_np(self.model_goes.y)),
                ("x", _as_np(self.model_goes.x)),
            ]
        )

    # ------------------------------------------------------------------
    # Output schema
    # ------------------------------------------------------------------

    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        # Derive a step-size from each model's input→output stride.  Require
        # both models to agree so the combined zarr has a single lead_time
        # axis with consistent stride.
        goes_step = _infer_step_delta(self.model_goes)
        mrms_step = _infer_step_delta(self.model_mrms)
        if goes_step != mrms_step:
            raise ValueError(
                f"StormScope GOES and MRMS models must share a forecast stride; "
                f"got {goes_step} and {mrms_step}."
            )

        total: CoordSystem = OrderedDict()
        if ensemble_size > 1:
            total["ensemble"] = np.arange(ensemble_size)
        total["time"] = times
        total["lead_time"] = np.array(
            [goes_step * (i + 1) for i in range(self.nsteps)]
        ).astype("timedelta64[ns]")

        spatial_ref = self.effective_spatial_ref()
        for d in ("y", "x"):
            total[d] = spatial_ref[d]
        return total

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        # ``data_source`` is ignored — StormScope uses the two IC sources
        # cached during setup.

        y, y_coords = self._fetch_ic(
            self._goes_ic_source, self._goes_ic_coords, item, device
        )
        y_m, y_m_coords = self._fetch_ic(
            self._mrms_ic_source, self._mrms_ic_coords, item, device
        )

        # Seed torch's global RNG so the diffusion sampler's per-member
        # noise is deterministic across runs.  StormScope models don't
        # expose a set_rng hook; torch.manual_seed is the supported path.
        torch.manual_seed(item.seed)

        # Rank only gates tqdm output below.  Default to 0 when the
        # DistributedManager isn't initialized (unit tests, or single-process
        # runs) rather than forcing initialization here — which requires an
        # indexed accelerator on some backends and fails on CPU.
        rank = DistributedManager().rank if DistributedManager.is_initialized() else 0

        for step_idx in tqdm(
            range(self.nsteps),
            desc=f"IC {item.time}",
            position=1,
            leave=False,
            disable=rank != 0,
        ):
            pred_goes, pred_goes_coords = self.model_goes(y, y_coords)
            pred_mrms, pred_mrms_coords = self.model_mrms.call_with_conditioning(
                y_m,
                y_m_coords,
                conditioning=y,
                conditioning_coords=y_coords,
            )

            # Concat predictions along the variable axis.  Both tensors
            # are on the HRRR grid after prep_input, so spatial dims
            # line up; cat_coords handles coord-system merging.
            # The model-internal batch dim (size 1) is squeezed out by
            # Pipeline.run because _run_item_includes_batch_dim is True.
            combined, combined_coords = cat_coords(
                (pred_goes, pred_mrms),
                (pred_goes_coords, pred_mrms_coords),
                "variable",
            )

            yield combined, combined_coords

            # Prepare next-step inputs.  next_input handles sliding
            # window (10min variants) or passes pred through (60min).
            y, y_coords = self.model_goes.next_input(
                pred_goes, pred_goes_coords, y, y_coords
            )
            y_m, y_m_coords = self.model_mrms.next_input(
                pred_mrms, pred_mrms_coords, y_m, y_m_coords
            )

    # ------------------------------------------------------------------
    # Predownload
    # ------------------------------------------------------------------

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare the regridded IC + verification stores for StormScope.

        One store per IC source — ``data_goes`` (satellite) and
        ``data_mrms`` (radar) — plus, for GLM-bearing variants, a
        ``data_glm`` store for the ``glm_density`` state channel.  Radar
        and satellite are resampled onto the model's HRRR sub-region via a
        :class:`~src.regrid.NearestNeighborRegridder`; the sparse GLM count
        field uses a :class:`~src.regrid.BilinearRegridder` (see
        :meth:`_build_glm_store`).  In every case the inference-time
        ``prep_input`` grid-match check passes and the live interpolator
        never runs.

        The stored times are the union of:

        * ``{ic + input_lead}`` for every IC-input lead time (typically
          just ``ic + 0h`` for single-step variants; the full sliding
          window for 10-min variants).
        * ``{ic + k·stride}`` for every forecast step ``k = 1..nsteps``.
          For nowcasting these are real observations at future times —
          the same data :mod:`src.scoring` will load as verification.

        Storing both classes in the same zarr keeps the predownload /
        scoring paths aligned (both read from ``data_{goes,mrms}.zarr``)
        without needing a separate ``verification.zarr``.

        BYO overrides are honored by skipping the corresponding store:
        pass ``cfg.model.{goes,mrms}.ic_byo=true`` (or set a non-default
        ``_target_`` on the ``ic_source`` block that already points
        at a local store) to take full control of that source.

        Additionally, for the GOES model the conditioning data source
        (typically ``GFS_FX``) is predownloaded into ``cond_goes.zarr``
        — the model's ``fetch_conditioning`` path is invoked once per
        forecast step, so a local cache meaningfully reduces S3 traffic
        for multi-IC campaigns.  The MRMS model's conditioning is
        provided externally in this pipeline (GOES state → MRMS via
        ``call_with_conditioning``), so its internal conditioning
        source is bypassed and not predownloaded.
        """
        # CPU-load the models so we can read their cropped HRRR y/x/lat/lon
        # buffers without materializing weights to a device.  The load is
        # the same `_load_stormscope_model` call that ``setup()`` uses, so
        # no new weights are downloaded at predownload time.
        from ..work import build_work_items

        stores: list[PredownloadStore] = []
        all_items = build_work_items(cfg)
        unique_ic_times = sorted({i.time for i in all_items})

        for side, model_cfg_key in (("goes", "goes"), ("mrms", "mrms")):
            model_cfg = cfg.model[model_cfg_key]
            if model_cfg.get("ic_byo", False):
                logger.info(f"StormScope {side}: ic_byo=true — skipping predownload")
                continue

            model = _load_stormscope_model(model_cfg)
            ic_coords = model.input_coords()
            stride = _infer_step_delta(model)

            max_dist_km = cfg.model.get("max_dist_km", None)
            src_lat, src_lon = hydra.utils.instantiate(model_cfg.ic_grid)
            regridder = _nn_regridder(
                model, src_lat, src_lon, max_dist_km, default_km=12.0
            )

            raw_source = hydra.utils.instantiate(model_cfg.ic_source)
            wrapped = RegriddedSource(raw_source, regridder)

            # Offsets covering both the IC input window and every forecast
            # valid time.  Cast to ns for datetime64 arithmetic.
            input_offsets = [np.timedelta64(lt, "ns") for lt in ic_coords["lead_time"]]
            forecast_offsets = [stride * (k + 1) for k in range(cfg.nsteps)]
            fetch_times = sorted(
                {
                    t + off
                    for t in unique_ic_times
                    for off in input_offsets + forecast_offsets
                }
            )

            spatial_ref: CoordSystem = OrderedDict(
                [
                    ("y", _as_np(model.y)),
                    ("x", _as_np(model.x)),
                ]
            )

            # GLM-bearing MRMS variants (``3km_10min``) carry a ``glm_density``
            # state channel sourced from a different grid than radar.  Split it
            # into its own bilinear-regridded ``data_glm.zarr`` store; the radar
            # store (``ic_source``) then holds only the radar channels.  At
            # inference the two are recombined via a CompositeSource.
            glm_vars = {str(v) for v in getattr(model, "glm_variables", [])}
            radar_vars = [
                str(v) for v in ic_coords["variable"] if str(v) not in glm_vars
            ]
            stores.append(
                PredownloadStore(
                    name=f"data_{side}",
                    source=wrapped,
                    times=fetch_times,
                    variables=radar_vars,
                    spatial_ref=spatial_ref,
                    role="ic",
                )
            )

            if glm_vars:
                glm_store = self._build_glm_store(
                    model=model,
                    model_cfg=model_cfg,
                    fetch_times=fetch_times,
                    spatial_ref=spatial_ref,
                )
                if glm_store is not None:
                    stores.append(glm_store)

            # Conditioning predownload — only for the GOES model.  In our
            # coupling loop MRMS is always called via call_with_conditioning,
            # so its internal conditioning_data_source is bypassed.  Users
            # can opt out with cfg.model.<side>.cond_byo=true.
            if side != "goes" or model_cfg.get("cond_byo", False):
                continue

            cond_store = self._build_conditioning_store(
                model=model,
                model_cfg=model_cfg,
                side=side,
                unique_ic_times=unique_ic_times,
                input_offsets=input_offsets,
                forecast_offsets=forecast_offsets,
                spatial_ref=spatial_ref,
                max_dist_km=max_dist_km,
            )
            if cond_store is not None:
                stores.append(cond_store)

        return stores

    def _build_conditioning_store(
        self,
        *,
        model: Any,
        model_cfg: DictConfig,
        side: str,
        unique_ic_times: list[np.datetime64],
        input_offsets: list[np.timedelta64],
        forecast_offsets: list[np.timedelta64],
        spatial_ref: CoordSystem,
        max_dist_km: float | None,
    ) -> PredownloadStore | None:
        """Build a predownload store for a StormScope model's conditioning source.

        Returns ``None`` when the model doesn't advertise conditioning
        variables or when the campaign config lacks a ``conditioning_grid``
        resolver (can't regrid without knowing the source grid).

        Supports both :class:`~earth2studio.data.base.ForecastSource`
        conditioning (e.g. ``GFS_FX``) — via
        :class:`~src.data.ValidTimeForecastAdapter` — and plain
        :class:`~earth2studio.data.base.DataSource` conditioning
        (e.g. ``ARCO`` ERA5) — forwarded directly.  Source type is
        detected by inspecting ``__call__``'s signature for a
        ``lead_time`` parameter, matching
        :func:`earth2studio.data.fetch_data`.

        The optional ``cfg.model.<side>.conditioning_cadence`` rounds
        each ``(ic_time + offset)`` to the coarser native resolution of
        the source (e.g. ``"1h"`` for hourly GFS / ERA5) and dedupes,
        cutting redundant 10-minute fetches from campaigns that use a
        sub-hour-stride model.
        """
        # conditioning_variables may be a numpy array (truth-testing an array
        # is ambiguous) or None — normalize before iterating.  Pure-obs
        # variants (e.g. GOES 3km_10min) expose an empty array.
        cond_vars_raw = model.conditioning_variables
        cond_vars: list[str] = (
            [] if cond_vars_raw is None else [str(v) for v in cond_vars_raw]
        )
        if not cond_vars:
            return None
        if model_cfg.get("conditioning_grid") is None:
            logger.info(
                f"StormScope {side}: conditioning_grid not configured — "
                "skipping conditioning predownload."
            )
            return None
        cond_source_cfg = model_cfg.get("load_args", {}).get("conditioning_data_source")
        if cond_source_cfg is None:
            logger.info(
                f"StormScope {side}: no conditioning_data_source in load_args — "
                "skipping conditioning predownload."
            )
            return None

        cond_src_lat, cond_src_lon = hydra.utils.instantiate(
            model_cfg.conditioning_grid
        )
        cond_regridder = NearestNeighborRegridder(
            source_lats=cond_src_lat,
            source_lons=cond_src_lon,
            target_lats=model.latitudes,
            target_lons=model.longitudes,
            target_y=_as_np(model.y),
            target_x=_as_np(model.x),
            max_dist_km=(float(max_dist_km) if max_dist_km is not None else 26.0),
        )

        # Cadence rounding: optional per-model knob that collapses
        # sub-hour-stride requests down to the source's native
        # resolution.  Example: a 10-min StormScope variant pulling
        # hourly GFS conditioning — 6 requests per hour collapse to 1.
        cadence = model_cfg.get("conditioning_cadence")
        cadence_ns: np.timedelta64 | None = None
        if cadence is not None:
            cadence_ns = (
                pd.Timedelta(cadence).to_timedelta64().astype("timedelta64[ns]")
            )
            logger.info(
                f"StormScope {side}: conditioning_cadence={cadence} — "
                f"rounding fetch times to the nearest {cadence} boundary."
            )

        def _round_to_cadence(t: np.datetime64) -> np.datetime64:
            if cadence_ns is None:
                return t
            c = int(cadence_ns.astype("int64"))
            t_int = int(np.datetime64(t, "ns").astype("int64"))
            return np.datetime64(((t_int + c // 2) // c) * c, "ns")

        # Lookup covers every valid time the GOES model will request at
        # inference — rounded to cadence when configured.  First-IC-wins
        # when two ICs share a (rounded) valid time.
        lookup: dict[np.datetime64, tuple[np.datetime64, np.timedelta64]] = {}
        for ic_time in unique_ic_times:
            ic_ns = np.datetime64(ic_time, "ns")
            for off in list(input_offsets) + list(forecast_offsets):
                vt_raw = ic_ns + np.timedelta64(off, "ns")
                vt = _round_to_cadence(vt_raw)
                if vt not in lookup:
                    lookup[vt] = (ic_ns, vt - ic_ns)

        # Detect source type — ForecastSource needs the (init, lead)
        # adapter; DataSource goes straight in.
        raw_source = hydra.utils.instantiate(cond_source_cfg)
        sig = inspect.signature(raw_source.__call__)
        if "lead_time" in sig.parameters:
            source_for_predownload: Any = ValidTimeForecastAdapter(raw_source, lookup)
            logger.info(
                f"StormScope {side}: wrapping ForecastSource conditioning "
                "with ValidTimeForecastAdapter for predownload."
            )
        else:
            source_for_predownload = raw_source
            logger.info(
                f"StormScope {side}: conditioning source is a DataSource — "
                "no valid-time adapter needed."
            )

        wrapped = RegriddedSource(source_for_predownload, cond_regridder)

        return PredownloadStore(
            name=f"cond_{side}",
            source=wrapped,
            times=sorted(lookup),
            variables=cond_vars,
            spatial_ref=spatial_ref,
            role="conditioning",
        )

    def _build_glm_store(
        self,
        *,
        model: Any,
        model_cfg: DictConfig,
        fetch_times: list[np.datetime64],
        spatial_ref: CoordSystem,
    ) -> PredownloadStore | None:
        """Build the ``data_glm.zarr`` predownload store for a GLM-bearing model.

        The gridded GLM source (:class:`~earth2studio.data.GOESGLMGrid`) lives
        on a 0.1-degree lat/lon grid, so it is resampled onto the model's HRRR
        sub-region with a :class:`~src.regrid.BilinearRegridder` — the same
        bilinear map the model applies internally
        (:meth:`StormScopeMRMS.build_glm_interpolator`) — rather than the
        nearest-neighbor path used for radar/satellite channels.  Stored counts
        are physical (the model applies ``log1p`` at inference).

        The store covers the same ``fetch_times`` as the radar store — the IC
        input window plus every forecast valid time — so it doubles as GLM
        verification for scoring ``glm_density``.

        Returns ``None`` when the campaign config provides neither a
        ``glm_data_source`` (under ``load_args``) nor a ``glm_grid`` resolver;
        without those the GLM channel can't be sourced or regridded.
        """
        glm_vars = [str(v) for v in getattr(model, "glm_variables", [])]
        if not glm_vars:
            return None

        glm_source_cfg = model_cfg.get("load_args", {}).get("glm_data_source")
        if glm_source_cfg is None:
            logger.warning(
                "StormScope mrms: model has GLM state channels but no "
                "load_args.glm_data_source is configured — skipping GLM "
                "predownload.  GLM will be fetched live at inference."
            )
            return None
        if model_cfg.get("glm_grid") is None:
            logger.warning(
                "StormScope mrms: glm_grid resolver not configured — cannot "
                "regrid GLM at predownload.  GLM will be fetched live at "
                "inference."
            )
            return None

        glm_src_lat, glm_src_lon = hydra.utils.instantiate(model_cfg.glm_grid)
        glm_regridder = _bilinear_regridder(model, glm_src_lat, glm_src_lon)
        raw_glm_source = hydra.utils.instantiate(glm_source_cfg)
        wrapped_glm = RegriddedSource(raw_glm_source, glm_regridder)

        return PredownloadStore(
            name="data_glm",
            source=wrapped_glm,
            times=fetch_times,
            variables=glm_vars,
            spatial_ref=spatial_ref,
            role="ic",
        )

    # Verification sourcing is handled by the base class: its
    # default `Pipeline.verification_source` picks up
    # ``data_{goes,mrms}.zarr`` via the ``data_*.zarr`` glob and wraps
    # them in a :class:`~src.data.CompositeSource` for scoring.

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _maybe_swap_conditioning(
        self,
        cfg: DictConfig,
        side: str,
        model: Any,
    ) -> None:
        """Swap *model*'s conditioning source for a predownloaded zarr.

        Looks for ``<cfg.output.path>/cond_<side>.zarr`` and assigns a
        :class:`~src.data.PredownloadedSource` reading from it to the
        model's ``conditioning_data_source`` attribute.  When the cache
        is absent this is a no-op — the model retains whatever
        forecast source the campaign config installed at load time
        (typically ``GFS_FX``).

        The predownloaded zarr is valid-time-keyed with ``(y, x)``
        coords matching ``model.y`` / ``model.x``; at inference time
        :meth:`StormScopeBase.fetch_conditioning` converts the model's
        ``(time, lead_time)`` request into a valid-time lookup via
        :func:`earth2studio.data.fetch_data` and
        :meth:`StormScopeBase.prep_input` skips the conditioning
        interpolator on the grid match.
        """
        cache_path = os.path.join(cfg.output.path, f"cond_{side}.zarr")
        if not os.path.exists(cache_path):
            logger.info(
                f"StormScope {side}: no conditioning cache at {cache_path} — "
                f"model will fetch conditioning from its configured source."
            )
            return

        source: Any = PredownloadedSource(cache_path)

        # If the predownload was cadence-deduplicated, the cached zarr only
        # holds values on the coarser cadence.  Wrap with CadenceRoundedSource
        # so the model's finer-grained queries route to the nearest stored
        # value while still receiving DataArrays labeled at the requested
        # valid times (preserving fetch_data's lead_time reconstruction).
        cadence = cfg.model[side].get("conditioning_cadence")
        if cadence is not None:
            source = CadenceRoundedSource(source, cadence)
            logger.info(
                f"StormScope {side}: using predownloaded conditioning store "
                f"{cache_path} with cadence={cadence} query rounding."
            )
        else:
            logger.info(
                f"StormScope {side}: using predownloaded conditioning store "
                f"{cache_path}"
            )
        model.conditioning_data_source = source

    def _resolve_mrms_ic_source(self, cfg: DictConfig) -> DataSource:
        """Resolve the MRMS initial-condition source.

        For GLM-bearing variants (``3km_10min``), the MRMS state spans two
        native grids — radar (``refc`` / ``refc_base`` from
        :class:`~earth2studio.data.MRMS`) and lightning (``glm_density`` from
        :class:`~earth2studio.data.GOESGLMGrid`) — so no single source can
        supply it.  We build a :class:`~src.data.CompositeSource` that
        dispatches radar variables to one component and GLM to the other,
        with each component resolved independently:

        * a predownloaded, already-regridded zarr (``data_mrms.zarr`` /
          ``data_glm.zarr``) when present — the cheap inference path; or
        * a live source wrapped in the matching regridder (nearest-neighbor
          for radar, **bilinear** for GLM, matching training) so both land
          on the model ``y`` / ``x`` grid.

        Either way the assembled state is already on the model grid, so
        :meth:`StormScopeBase.prep_input` skips re-interpolation and the
        ``glm_density`` channel then evolves autoregressively through
        :meth:`next_input` (``call_with_conditioning`` never re-fetches it).

        Variants without GLM channels keep the original single-source
        resolution (predownloaded ``data_mrms.zarr`` → live ``ic_source``).
        """
        if int(getattr(self.model_mrms, "n_glm_channels", 0)) <= 0:
            return resolve_ic_source(
                cfg,
                store_name="data_mrms.zarr",
                live_source=cfg.model.mrms.ic_source,
            )

        model_cfg = cfg.model.mrms
        max_dist_km = cfg.model.get("max_dist_km", None)
        glm_vars = {str(v) for v in self.model_mrms.glm_variables}

        # Resolve config nodes lazily (``.get``) — a campaign that relies on the
        # predownloaded stores may omit the live ic_source / glm_data_source /
        # grid blocks entirely; they're only needed for the live fallback.
        radar_src = self._resolve_component_source(
            cfg,
            store_name="data_mrms.zarr",
            raw_source_cfg=model_cfg.get("ic_source"),
            grid_cfg=model_cfg.get("ic_grid"),
            regridder_kind="nearest",
            max_dist_km=max_dist_km,
        )
        glm_src = self._resolve_component_source(
            cfg,
            store_name="data_glm.zarr",
            raw_source_cfg=model_cfg.get("load_args", {}).get("glm_data_source"),
            grid_cfg=model_cfg.get("glm_grid"),
            regridder_kind="bilinear",
            max_dist_km=max_dist_km,
        )

        sources: dict[str, DataSource] = {"data_mrms": radar_src, "data_glm": glm_src}
        variable_index = {
            str(v): ("data_glm" if str(v) in glm_vars else "data_mrms")
            for v in self.model_mrms.variables
        }
        logger.info(
            "StormScope mrms: assembling GLM-bearing IC state from radar + GLM "
            "component sources (composite)."
        )
        return CompositeSource(sources, variable_index)

    def _resolve_component_source(
        self,
        cfg: DictConfig,
        *,
        store_name: str,
        raw_source_cfg: Any,
        grid_cfg: Any,
        regridder_kind: str,
        max_dist_km: float | None,
    ) -> DataSource:
        """Resolve one component of the composite MRMS IC source.

        Returns a :class:`~src.data.PredownloadedSource` when the regridded
        zarr ``<output.path>/<store_name>`` exists; otherwise instantiates
        the live source and wraps it in the appropriate regridder so its
        output lands on the model ``y`` / ``x`` grid.
        """
        cache_path = os.path.join(cfg.output.path, store_name)
        if os.path.exists(cache_path):
            logger.info(f"StormScope mrms: using predownloaded {store_name}")
            return PredownloadedSource(cache_path)

        if raw_source_cfg is None or grid_cfg is None:
            raise ValueError(
                f"StormScope mrms: no predownloaded '{store_name}' and the live "
                "fallback is unconfigured (missing source and/or grid block). "
                "Either run predownload.py first, or provide the live source and "
                "its grid resolver in cfg.model.mrms."
            )

        logger.info(
            f"StormScope mrms: no {store_name} cache — wrapping live source "
            f"in a {regridder_kind} regridder onto the model grid."
        )
        raw_source = hydra.utils.instantiate(raw_source_cfg)
        src_lat, src_lon = hydra.utils.instantiate(grid_cfg)
        if regridder_kind == "bilinear":
            regridder: Any = _bilinear_regridder(self.model_mrms, src_lat, src_lon)
        else:
            regridder = _nn_regridder(
                self.model_mrms, src_lat, src_lon, max_dist_km, default_km=12.0
            )
        return RegriddedSource(raw_source, regridder)

    def _fetch_ic(
        self,
        source: DataSource,
        ic_coords: CoordSystem,
        item: WorkItem,
        device: torch.device,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Fetch IC data for one of the two StormScope models.

        The returned ``(x, coords)`` has a ``batch`` dim prepended —
        required by :meth:`StormScopeMRMS.call_with_conditioning`.
        """
        x, coords = fetch_data(
            source=source,
            time=[item.time],
            variable=ic_coords["variable"],
            lead_time=ic_coords["lead_time"],
            device=device,
        )
        x = x.unsqueeze(0)
        coords = OrderedDict([("batch", np.arange(1))] + list(coords.items()))
        return x, coords


# ------------------------------------------------------------------
# StormScope helpers (module-level for reuse in tests and config)
# ------------------------------------------------------------------


def _load_stormscope_model(model_cfg: DictConfig) -> Any:
    """Load a StormScope model (GOES or MRMS) from its config sub-block.

    Uses the same pattern as ``load_prognostic`` in ``src.models``:
    resolve the class via ``architecture``, optionally load a package
    from ``package_path``, then call ``cls.load_model(package, **load_args)``
    — the StormScope API requires ``conditioning_data_source`` and
    ``model_name`` under ``load_args``.
    """
    cls = hydra.utils.get_class(model_cfg.architecture)

    if model_cfg.get("package_path"):
        from earth2studio.models.auto import Package

        pkg = Package(model_cfg.package_path)
    else:
        from ..distributed import run_on_rank0_first

        pkg = run_on_rank0_first(cls.load_default_package)

    load_args = dict(model_cfg.get("load_args", {}))
    # Resolve nested _target_ entries (e.g. conditioning_data_source).
    resolved: dict[str, Any] = {}
    for k, v in load_args.items():
        if isinstance(v, (DictConfig, dict)) and "_target_" in v:
            resolved[k] = hydra.utils.instantiate(v)
        else:
            resolved[k] = v

    model = cls.load_model(package=pkg, **resolved)
    logger.success(
        f"Loaded StormScope model: {cls.__name__} "
        f"(model_name={resolved.get('model_name', 'default')})"
    )
    return model


def _nn_regridder(
    model: Any,
    src_lat: Any,
    src_lon: Any,
    max_dist_km: float | None,
    *,
    default_km: float,
) -> NearestNeighborRegridder:
    """Build a nearest-neighbor regridder from a source grid onto ``model``'s
    HRRR sub-region.  Used for radar / satellite channels."""
    return NearestNeighborRegridder(
        source_lats=src_lat,
        source_lons=src_lon,
        target_lats=model.latitudes,
        target_lons=model.longitudes,
        target_y=_as_np(model.y),
        target_x=_as_np(model.x),
        max_dist_km=(float(max_dist_km) if max_dist_km is not None else default_km),
    )


def _bilinear_regridder(model: Any, src_lat: Any, src_lon: Any) -> BilinearRegridder:
    """Build a bilinear regridder from a source grid onto ``model``'s HRRR
    sub-region.  Used for the sparse ``glm_density`` lightning field, matching
    the model's internal :meth:`StormScopeMRMS.build_glm_interpolator`."""
    return BilinearRegridder(
        source_lats=src_lat,
        source_lons=src_lon,
        target_lats=model.latitudes,
        target_lons=model.longitudes,
        target_y=_as_np(model.y),
        target_x=_as_np(model.x),
    )


def _infer_step_delta(model: Any) -> np.timedelta64:
    """Return the forecast stride (single model step) as a timedelta64[ns].

    Computed as ``output_coords.lead_time[-1] - input_coords.lead_time[-1]``,
    which gives the per-step advance for both sliding-window (10-min)
    and single-step (60-min) StormScope variants.
    """
    ic = model.input_coords()
    out = model.output_coords(ic)
    delta = out["lead_time"][-1] - ic["lead_time"][-1]
    return np.timedelta64(delta, "ns")


def _concat_var_lists(a: list[str], b: list[str]) -> list[str]:
    """Append *b*'s entries to *a*, skipping duplicates.  Preserves order."""
    seen = set(a)
    out = list(a)
    for v in b:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _as_np(arr: Any) -> np.ndarray:
    """Coerce a torch tensor / array-like into a numpy array."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

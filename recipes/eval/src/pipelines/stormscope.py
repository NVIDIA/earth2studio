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
    PredownloadedSource,
    ValidTimeForecastAdapter,
    resolve_ic_source,
)
from ..regrid import NearestNeighborRegridder, RegriddedSource
from ..work import WorkItem
from .base import Pipeline, PredownloadStore


class StormScopePipeline(Pipeline):
    """Coupled GOES/MRMS nowcasting pipeline using StormScope models.

    Runs two prognostic models together:

    * :class:`earth2studio.models.px.StormScopeGOES` — forecasts GOES
      satellite channels.  Conditioned on synoptic-scale GFS data (60-min
      variants) or nothing (10-min variants).
    * :class:`earth2studio.models.px.StormScopeMRMS` — forecasts MRMS
      composite reflectivity (``refc``), conditioned on GOES (either
      observations at IC time or the GOES model's own predictions
      during rollout).

    Both models share the HRRR Lambert-conformal grid (``y``, ``x``)
    and the pipeline yields a combined ``(variables × y × x)`` tensor
    per forecast step — GOES channels and MRMS refc stacked along the
    ``variable`` axis so the output zarr is a single unified store.

    Config shape
    ------------
    Expected structure under ``cfg.model``::

        goes:
            architecture: earth2studio.models.px.StormScopeGOES
            load_args:
                model_name: 6km_60min_natten_cos_zenith_input_eoe_v2
                conditioning_data_source: {_target_: earth2studio.data.GFS_FX}
            ic_source: {_target_: earth2studio.data.GOES, satellite: goes16, scan_mode: C}
            ic_grid: {_target_: src.grids.goes_grid, satellite: goes16, scan_mode: C}
            conditioning_grid: {_target_: src.grids.gfs_grid}
        mrms:
            architecture: earth2studio.models.px.StormScopeMRMS
            load_args:
                model_name: 6km_60min_natten_cos_zenith_input_mrms_eoe
                conditioning_data_source: {_target_: earth2studio.data.GOES, ...}
            ic_source: {_target_: earth2studio.data.MRMS}
            ic_grid: {_target_: src.grids.mrms_grid}
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
    (``data_goes.zarr``, ``data_mrms.zarr``), each wrapped in a
    :class:`~src.regrid.RegriddedSource` that resamples the raw source
    onto the model's HRRR sub-region at write time.  The resulting
    store has ``(time, y, x)`` dims whose ``y``/``x`` values match the
    model's native grid exactly — so at inference time
    :meth:`StormScopeBase.prep_input` detects the match and skips its
    live interpolation.

    :meth:`setup` auto-detects the predownloaded stores under
    ``<output.path>/data_{goes,mrms}.zarr`` and uses
    :class:`~src.data.PredownloadedSource` in their place.  Users can
    skip predownload entirely with per-model BYO overrides (see
    :class:`StormScopePipeline` docstring) or by pointing to raw live
    sources and running with the model's live interpolators.
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
        self._mrms_ic_source = resolve_ic_source(
            cfg,
            store_name="data_mrms.zarr",
            live_source=cfg.model.mrms.ic_source,
        )

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

        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        rank = DistributedManager().rank

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
        """Declare one regridded IC + verification store per StormScope model.

        Each store holds data pre-resampled onto the model's HRRR
        sub-region via a :class:`~src.regrid.NearestNeighborRegridder`,
        so the inference-time ``prep_input`` grid-match check passes
        and the live interpolator never runs.

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
            regridder = NearestNeighborRegridder(
                source_lats=src_lat,
                source_lons=src_lon,
                target_lats=model.latitudes,
                target_lons=model.longitudes,
                target_y=_as_np(model.y),
                target_x=_as_np(model.x),
                max_dist_km=(float(max_dist_km) if max_dist_km is not None else 12.0),
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

            stores.append(
                PredownloadStore(
                    name=f"data_{side}",
                    source=wrapped,
                    times=fetch_times,
                    variables=[str(v) for v in ic_coords["variable"]],
                    spatial_ref=spatial_ref,
                    role="ic",
                )
            )

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
        cond_vars: list[str] = [str(v) for v in (model.conditioning_variables or [])]
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

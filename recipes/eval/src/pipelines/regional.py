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

"""Pipelines for limited-area models on a window of a larger grid.

Two pipelines and the source adapters they share:

* :class:`RegionalForecastPipeline` runs a regional prognostic model
  (StormCast, StormCast-CONUS): it crops the initial-condition and
  verification sources to the model window with :class:`SubgridSource`,
  and, for models that condition on coarse global fields, either
  predownloads those fields into ``conditioning.zarr`` or streams them.
* :class:`RegionalDiagnosticPipeline` runs a regional downscaling model
  (CorrDiff COSMO): it applies a ``crop`` box through the model's
  ``set_domain``, windows a global input source onto the model's input
  grid with the longitude convention aligned, and stores a curvilinear
  output on index dimensions.

Both take the same ``crop`` block (see :func:`apply_crop`).  A streaming
StormCast campaign looks like::

    pipeline:
      _target_: src.pipelines.regional.RegionalForecastPipeline
      conditioning_source: {_target_: earth2studio.data.ARCO_ERA5, cache: false}
      conditioning_mode: live
    ic_source: {_target_: earth2studio.data.HRRR, cache: false}
    verification_source: {_target_: earth2studio.data.HRRR, cache: false}
    require_predownload: false
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig

from earth2studio.utils.coords import map_coords
from earth2studio.utils.type import CoordSystem

from ..regions import NON_SPATIAL
from .diagnostic import DiagnosticPipeline
from .forecast import ForecastPipeline


class SubgridSource:
    """DataSource wrapper that crops a source to a store's spatial grid.

    Predownload writes exactly the grid a store declares, and the recipe
    only re-aligns sources on latitude/longitude grids.  A limited-area
    source that returns its whole grid, such as HRRR (1059 x 1799) feeding
    StormCast's 512 x 640 window, needs its output cropped at fetch time.
    Selection is by nearest coordinate value along every spatial dimension
    of ``spatial_ref``.  The wrapper also drops auxiliary coordinates that
    index no dimension (HRRR's 2-D ``lat``/``lon``), since the stores carry
    the projection coordinates only.

    Parameters
    ----------
    source : DataSource
        Source to wrap; called as ``source(time, variable)``.
    spatial_ref : CoordSystem
        Coordinate system whose spatial axes define the crop.
    """

    def __init__(self, source: Any, spatial_ref: CoordSystem) -> None:
        self._source = source
        self._sel = {
            d: np.asarray(v)
            for d, v in spatial_ref.items()
            if d not in NON_SPATIAL and np.ndim(v) == 1 and np.size(v) > 0
        }

    def __call__(self, time: Any, variable: Any) -> xr.DataArray:
        """Fetch from the wrapped source and crop to the target grid.

        Parameters
        ----------
        time : Any
            Timestamps, forwarded to the wrapped source.
        variable : Any
            Variable names, forwarded to the wrapped source.

        Returns
        -------
        xr.DataArray
            The source's data on the target grid, dimension coordinates only.
        """
        da = self._source(time, variable)
        if "lon" in da.dims and "lon" in self._sel:
            da = align_longitudes(da, self._sel["lon"])
        sel = {d: v for d, v in self._sel.items() if d in da.dims}
        if sel:
            da = da.sel(sel, method="nearest")
        if "lead_time" in da.dims and da.sizes["lead_time"] == 1:
            # Some sources tag analyses with a zero lead; stores and the
            # online scorer expect (time, variable, <spatial...>).
            da = da.isel(lead_time=0, drop=True)
        return da.reset_coords(drop=True)


def align_longitudes(da: xr.DataArray, target_lon: np.ndarray) -> xr.DataArray:
    """Express a source's ``lon`` coordinate in the target grid's convention.

    Global sources serve longitudes on [0, 360); some regional models
    describe their input grid on [-180, 180) (CorrDiff COSMO's ERA5 window,
    for instance).  Nearest-neighbour selection needs both sides on the same
    convention, so the source is re-labelled and re-sorted when the
    conventions differ.  A no-op when they already agree.

    Parameters
    ----------
    da : xr.DataArray
        Source data with a 1-D ``lon`` dimension coordinate.
    target_lon : np.ndarray
        Longitudes of the grid that receives the selection.

    Returns
    -------
    xr.DataArray
    """
    lon = np.asarray(da["lon"].values)
    target = np.asarray(target_lon)
    if target.min() < 0 and lon.max() > 180:
        da = da.assign_coords(lon=((lon + 180.0) % 360.0) - 180.0).sortby("lon")
    elif target.max() > 180 and lon.min() < 0:
        da = da.assign_coords(lon=lon % 360.0).sortby("lon")
    return da


def apply_crop(model: Any, crop: dict | None) -> Any:
    """Restrict a loaded model to a lat/lon box through its ``set_domain``.

    One ``crop:`` block in a campaign covers both ways Earth2Studio models
    expose cropped inference: models that choose their window after loading
    (``set_domain``, CorrDiff COSMO) take it here; models that take window
    limits in their constructor (StormCast-CONUS's ``hrrr_lat_lim`` and
    ``hrrr_lon_lim``) receive them through ``model.load_args`` instead, and
    a ``crop:`` block on such a model raises with that hint.

    Parameters
    ----------
    model : Any
        Loaded model.
    crop : dict | None
        ``{lat_min, lat_max, lon_min, lon_max}`` plus any extra keyword
        arguments for ``set_domain`` (for example ``margin_deg``); ``None``
        leaves the model unchanged.

    Returns
    -------
    Any
        The cropped model (a new instance for ``set_domain`` models).
    """
    if not crop:
        return model
    if not hasattr(model, "set_domain"):
        raise TypeError(
            f"{type(model).__name__} has no set_domain(); pass its own window "
            "limits through model.load_args instead of a crop block."
        )
    box = dict(crop)
    bbox = [box.pop(k) for k in ("lat_min", "lat_max", "lon_min", "lon_max")]
    return model.set_domain(*bbox, **box)


def _index_dims(coords: CoordSystem) -> tuple[CoordSystem, dict[str, np.ndarray]]:
    """Replace 2-D ``lat``/``lon`` dimension coordinates by index dims.

    Curvilinear models (rotated-pole COSMO) report their grid as ``lat`` and
    ``lon`` dimensions carrying 2-D arrays, which no store or scorer can
    index.  The convention the recipe understands is the HRRR one:
    projection (index) dimensions ``y``/``x`` with the 2-D latitude and
    longitude kept aside as auxiliary fields.

    Parameters
    ----------
    coords : CoordSystem
        Coordinate system from a model.

    Returns
    -------
    tuple[CoordSystem, dict[str, np.ndarray]]
        The coordinate system with ``y``/``x`` index dims, and the 2-D
        ``lat``/``lon`` arrays (empty when the grid was already 1-D).
    """
    lat = np.asarray(coords.get("lat", np.empty(0)))
    lon = np.asarray(coords.get("lon", np.empty(0)))
    if lat.ndim != 2:
        return coords, {}
    out: CoordSystem = OrderedDict()
    for k, v in coords.items():
        if k == "lat":
            out["y"] = np.arange(lat.shape[0])
        elif k == "lon":
            out["x"] = np.arange(lat.shape[1])
        else:
            out[k] = v
    return out, {"lat": lat, "lon": lon}


def _conditioning_window(
    lat: np.ndarray, lon: np.ndarray, margin_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Latitudes and longitudes of the 0.25 degree ERA5 grid around a domain.

    Parameters
    ----------
    lat : np.ndarray
        Latitudes of the model grid (any shape), in degrees.
    lon : np.ndarray
        Longitudes of the model grid (any shape), in degrees on [0, 360).
    margin_deg : float
        Margin added on every side, so the model's interpolation never
        reaches past the stored window.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ERA5 latitudes (descending) and longitudes inside the window.
    """
    lat_lo, lat_hi = float(np.min(lat)) - margin_deg, float(np.max(lat)) + margin_deg
    lon_lo, lon_hi = float(np.min(lon)) - margin_deg, float(np.max(lon)) + margin_deg
    return (
        ERA5_LAT[(ERA5_LAT >= lat_lo) & (ERA5_LAT <= lat_hi)],
        ERA5_LON[(ERA5_LON >= lon_lo) & (ERA5_LON <= lon_hi)],
    )


# How a model's coarse conditioning fields reach it: fetched into a local
# store before the run, or streamed from the source during inference.
# The 0.25 degree ERA5 grid (721 x 1440, north to south) that global
# conditioning sources such as ARCO ERA5 serve.
ERA5_LAT = np.linspace(90.0, -90.0, 721)
ERA5_LON = np.arange(0.0, 360.0, 0.25)

CONDITIONING_MODES = ("predownload", "live")


class RegionalForecastPipeline(ForecastPipeline):
    """Forecast pipeline for limited-area models on a window of a larger grid.

    Two things separate regional models (StormCast is the configured example).
    First, their initial conditions and truth come from a source that returns
    its whole grid while the model runs on a window of it, so the pipeline
    crops with :class:`SubgridSource`: every predownloaded store,
    and a live ``verification_source`` if the campaign brings one instead of
    predownloading truth.  Second, some of them condition each step on
    coarse global fields that the model fetches through a data-source
    attribute; given a ``conditioning_source``, the pipeline either
    predownloads those fields into ``conditioning.zarr`` on a lat/lon window
    around the domain and attaches that store as a local
    :class:`src.data.PredownloadedSource`, or, in ``live`` mode, attaches
    the source itself and streams them during inference.

    ``conditioning_mode`` covers the conditioning fields only. The
    initial-condition and verification stores follow the recipe's own
    ``predownload`` settings, so an event campaign can predownload them as
    usual, or bring its own ``ic_source`` and ``verification_source`` to
    stream them. Combining all three streaming options gives a campaign
    that saves only its scores::

        pipeline:
          _target_: src.pipelines.regional.RegionalForecastPipeline
          conditioning_source:
            _target_: earth2studio.data.ARCO_ERA5
            cache: false
          conditioning_mode: live
        ic_source: {_target_: earth2studio.data.HRRR, cache: false}
        verification_source: {_target_: earth2studio.data.HRRR, cache: false}
        require_predownload: false
        model:
          architecture: earth2studio.models.px.StormCast
          load_args:
            conditioning_data_source: null

    Every process fetches through its own temporary data cache (see
    :meth:`_isolate_data_cache`), so ranks pulling the same fields at once
    never race on a source's temporary files, and nothing persists.

    Parameters
    ----------
    conditioning_source : DictConfig | dict | DataSource | None, optional
        Hydra spec (or, under Hydra's recursive instantiation, the live
        instance) of the global source for the conditioning variables.
        ``None`` (the default) for models that need no conditioning; the
        pipeline then only crops.
    conditioning_mode : str, optional
        ``"predownload"`` (the default) fetches the conditioning fields
        into ``conditioning.zarr`` during predownload; ``"live"`` streams
        them from the source during inference.
    conditioning_margin_deg : float, optional
        Degrees of margin kept around the model domain in the predownloaded
        conditioning store, by default 5.0.
    conditioning_attr : str, optional
        Model attribute that receives the conditioning source, by default
        ``"conditioning_data_source"``.
    conditioning_variables : list[str] | str, optional
        The conditioning variable names, or the name of the model attribute
        that lists them, by default ``"conditioning_variables"``.
    domain_attrs : tuple[str, str], optional
        Model attributes holding the domain's latitudes and longitudes (any
        shape, degrees, longitude on [0, 360)), used to size the
        conditioning window, by default ``("lat", "lon")``.
    crop_to_model_grid : bool, optional
        Crop the predownload stores and a live verification source to the
        model grid, by default True.  A no-op for sources that already
        return the model grid.
    isolate_data_cache : bool, optional
        Give each process its own temporary data cache, by default True.
        Set to False to keep a shared, persistent cache for sources
        configured with ``cache: true``.
    crop : DictConfig | dict | None, optional
        ``{lat_min, lat_max, lon_min, lon_max}`` applied through the model's
        ``set_domain`` after loading (see :func:`apply_crop`).  Models that
        take window limits in their constructor use ``model.load_args``
        instead.  By default None.
    """

    def __init__(
        self,
        conditioning_source: object | None = None,
        conditioning_mode: str = "predownload",
        conditioning_margin_deg: float = 5.0,
        conditioning_attr: str = "conditioning_data_source",
        conditioning_variables: list[str] | str = "conditioning_variables",
        domain_attrs: tuple[str, str] = ("lat", "lon"),
        crop_to_model_grid: bool = True,
        isolate_data_cache: bool = True,
        crop: DictConfig | dict | None = None,
    ) -> None:
        super().__init__()
        self._crop_box = dict(crop) if crop else None
        if conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(
                f"conditioning_mode must be one of {CONDITIONING_MODES}; "
                f"got {conditioning_mode!r}."
            )
        self._cond_source_cfg = conditioning_source
        self._cond_mode = conditioning_mode
        self._margin = float(conditioning_margin_deg)
        self._cond_attr = str(conditioning_attr)
        self._cond_variables = conditioning_variables
        self._domain_attrs = tuple(domain_attrs)
        self._crop = bool(crop_to_model_grid)
        self._isolate = bool(isolate_data_cache)
        self._data_cache: str | None = None
        self._cond_source: Any = None

    def _isolate_data_cache(self) -> None:
        """Give this process its own temporary data-source cache.

        Sources such as HRRR download into a temporary directory under the
        data cache root and delete it after every fetch.  When ranks fetch
        at once through one shared root, one rank's cleanup removes files
        another rank is still reading.  A per-process root avoids that, and
        nothing persists: sources with ``cache: false`` leave it empty, and
        the directory goes at exit.  Model packages use the model cache,
        which this leaves alone.
        """
        if not self._isolate or self._data_cache is not None:
            return
        self._data_cache = tempfile.mkdtemp(prefix="e2s-data-cache-")
        os.environ["EARTH2STUDIO_DATA_CACHE"] = self._data_cache
        atexit.register(shutil.rmtree, self._data_cache, ignore_errors=True)

    def _conditioning_source(self) -> Any:
        """The conditioning source instance, instantiating a Hydra spec once."""
        if self._cond_source is None:
            import hydra

            source = self._cond_source_cfg
            if isinstance(source, (dict, DictConfig)):
                source = hydra.utils.instantiate(source)
            self._cond_source = source
        return self._cond_source

    def _conditioning_variables(self, model: Any) -> list[str]:
        """The conditioning variable names, from config or from the model."""
        spec = self._cond_variables
        if isinstance(spec, str):
            if not hasattr(model, spec):
                raise AttributeError(
                    f"{type(model).__name__} has no attribute '{spec}' listing its "
                    "conditioning variables; pass conditioning_variables explicitly."
                )
            spec = getattr(model, spec)
        return [str(v) for v in spec]

    def _domain(self, model: Any) -> tuple[np.ndarray, np.ndarray]:
        """The model domain's latitudes and longitudes."""
        lat_attr, lon_attr = self._domain_attrs
        if not (hasattr(model, lat_attr) and hasattr(model, lon_attr)):
            raise AttributeError(
                f"{type(model).__name__} has no '{lat_attr}'/'{lon_attr}' attributes "
                "describing its domain; set domain_attrs on the pipeline."
            )
        return np.asarray(getattr(model, lat_attr)), np.asarray(
            getattr(model, lon_attr)
        )

    def _model_spatial_ref(self, cfg: DictConfig) -> CoordSystem:
        """The model's output grid: from :meth:`setup`, or a CPU inspection."""
        ref = getattr(self, "_spatial_ref", None)
        if ref is None:
            from src.models import load_prognostic

            model = apply_crop(
                load_prognostic(cfg, self._model_node(cfg)), self._crop_box
            )
            ref = model.output_coords(model.input_coords())
        return ref

    def predownload_stores(self, cfg: DictConfig) -> list:
        """Cropped IC and verification stores, plus the conditioning store
        when the conditioning mode is ``predownload``.

        Parameters
        ----------
        cfg : DictConfig
            Campaign configuration.

        Returns
        -------
        list
            Stores to predownload; empty for a fully streaming campaign.
        """
        from src.models import load_prognostic
        from src.pipelines.base import PredownloadStore
        from src.predownload_utils import (
            compute_verification_times,
            declare_single_source_stores,
            infer_step_hours,
            single_source_stores_disabled,
        )
        from src.work import build_work_items

        self._isolate_data_cache()
        want_conditioning = (
            self._cond_source_cfg is not None and self._cond_mode == "predownload"
        )
        if single_source_stores_disabled(cfg) and not want_conditioning:
            return []

        # Inspect the model once (CPU) for its window, step and variables.
        model = apply_crop(load_prognostic(cfg, self._model_node(cfg)), self._crop_box)
        ic_coords = model.input_coords()
        spatial_ref = model.output_coords(ic_coords)
        step_hours = infer_step_hours(model)
        unique_ic_times = sorted({i.time for i in build_work_items(cfg)})

        stores: list = []
        if not single_source_stores_disabled(cfg):
            ic_fetch_times = sorted(
                {t + lt for t in unique_ic_times for lt in ic_coords["lead_time"]}
            )
            verif_times = compute_verification_times(
                unique_ic_times, cfg.nsteps, step_hours
            )
            stores = declare_single_source_stores(
                cfg,
                ic_variables=list(ic_coords["variable"]),
                ic_times=ic_fetch_times,
                verif_variables=list(cfg.output.variables),
                verif_times=verif_times,
                spatial_ref=spatial_ref,
            )
            if self._crop:
                # The source returns its whole grid; the stores hold the window.
                stores = [
                    replace(
                        store, source=SubgridSource(store.source, store.spatial_ref)
                    )
                    for store in stores
                ]

        if not want_conditioning:
            return stores

        # Conditioning at the input time of every step: IC + k * step for
        # k below nsteps.  Stored on a lat/lon window around the domain;
        # the model interpolates onto its own grid.
        lat, lon = self._domain(model)
        window = OrderedDict(
            zip(("lat", "lon"), _conditioning_window(lat, lon, self._margin))
        )
        stores.append(
            PredownloadStore(
                name="conditioning",
                source=SubgridSource(self._conditioning_source(), window),
                times=compute_verification_times(
                    unique_ic_times, max(int(cfg.nsteps) - 1, 0), step_hours
                ),
                variables=self._conditioning_variables(model),
                spatial_ref=window,
                role="conditioning",
            )
        )
        return stores

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        """Load the model and give it its conditioning source.

        Parameters
        ----------
        cfg : DictConfig
            Campaign configuration.
        device : torch.device
            Target device for inference.

        Raises
        ------
        FileNotFoundError
            In ``predownload`` mode without ``conditioning.zarr``, that is,
            predownload has not run for this campaign.
        """
        from src.data import PredownloadedSource

        self._isolate_data_cache()
        super().setup(cfg, device)
        if self._crop_box:
            cropped: Any = apply_crop(getattr(self, "prognostic"), self._crop_box)
            cropped = cropped.to(device)
            self.prognostic = cropped
            self._prognostic_ic = cropped.input_coords()
            self._spatial_ref = cropped.output_coords(self._prognostic_ic)
        if self._cond_source_cfg is None:
            return
        if self._cond_mode == "live":
            source = self._conditioning_source()
        else:
            path = os.path.join(cfg.output.path, "conditioning.zarr")
            if not os.path.isdir(path):
                raise FileNotFoundError(
                    f"The model needs the conditioning store at {path}; run "
                    "predownload.py for this campaign first."
                )
            source = PredownloadedSource(path)
        setattr(self.prognostic, self._cond_attr, source)

    def verification_source(self, cfg: DictConfig) -> Any:
        """The verification source, cropped to the model window when live.

        A predownloaded store already holds the window.  A live
        ``verification_source`` returns its whole grid, so it goes through
        :class:`SubgridSource`; the online scorer then reads exactly the
        scored grid.

        Parameters
        ----------
        cfg : DictConfig
            Campaign configuration.

        Returns
        -------
        DataSource
            The source scoring reads truth from.
        """
        source = super().verification_source(cfg)
        if self._crop and cfg.get("verification_source") is not None:
            return SubgridSource(source, self._model_spatial_ref(cfg))
        return source


class RegionalDiagnosticPipeline(DiagnosticPipeline):
    """Diagnostic pipeline for limited-area downscaling models.

    Extends the recipe's :class:`~src.pipelines.diagnostic.DiagnosticPipeline`
    with the three things a regional downscaler needs (CorrDiff COSMO is the
    configured example):

    * ``crop``: a lat/lon box applied through the model's ``set_domain``
      after loading, the same block :class:`RegionalForecastPipeline` takes.
    * Input windowing: the pipeline crops a global source to the model's
      own input grid at fetch time and aligns the longitude convention, so
      ARCO ERA5 on [0, 360) can feed an input window described on
      [-180, 180).
    * Curvilinear output: a model whose output grid is 2-D ``lat``/``lon``
      (a rotated-pole grid) lands on index dimensions ``y``/``x`` in the
      store, with the 2-D latitudes and longitudes written alongside as the
      coordinates ``lat[y, x]`` and ``lon[y, x]``.

    The output carries the diagnostic fields only; the raw input lives on a
    different grid and stays out.  Scoring needs truth on the model's output
    grid, which no bundled source provides yet, so campaigns keep the raw
    forecast store on disk and score offline.

    Parameters
    ----------
    crop : DictConfig | dict | None, optional
        ``{lat_min, lat_max, lon_min, lon_max}`` for ``set_domain``, plus
        any extra keyword arguments it takes, by default None.
    """

    def __init__(self, crop: DictConfig | dict | None = None) -> None:
        super().__init__()
        self._crop_box = dict(crop) if crop else None
        self._grid_latlon: dict[str, np.ndarray] = {}
        self._input_sources: dict[int, Any] = {}
        self._ensemble_axis = False

    def _load(self, cfg: DictConfig) -> list[Any]:
        from src.models import load_diagnostics

        return [apply_crop(dx, self._crop_box) for dx in load_diagnostics(cfg)]

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        """Load and crop the diagnostics; derive the index-dimension grid."""
        # The store carries an ensemble axis only for ensemble_size > 1
        # (see src.output.build_diagnostic_coords).
        self._ensemble_axis = int(cfg.get("ensemble_size", 1)) > 1
        self.diagnostics = [dx.to(device) for dx in self._load(cfg)]
        if not self.diagnostics:
            raise ValueError(
                "Diagnostic pipeline requires at least one entry in 'diagnostics'."
            )
        self._dx_input_coords = {id(dx): dx.input_coords() for dx in self.diagnostics}
        all_vars: list[str] = []
        for dx in self.diagnostics:
            for v in self._dx_input_coords[id(dx)]["variable"]:
                if str(v) not in all_vars:
                    all_vars.append(str(v))
        self._all_input_vars = all_vars
        dx0 = self.diagnostics[0]
        native = OrderedDict(
            (d, v)
            for d, v in dx0.output_coords(self._dx_input_coords[id(dx0)]).items()
            if d != "sample"
        )
        self._spatial_ref, self._grid_latlon = _index_dims(native)
        self._zero_lead = np.array([np.timedelta64(0, "ns")])

    def grid_coords(self) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
        """The 2-D latitude and longitude of a curvilinear output grid, stored
        beside the variables as ``lat[y, x]`` and ``lon[y, x]``."""
        return {k: (("y", "x"), v) for k, v in self._grid_latlon.items()}

    def _windowed(self, data_source: Any, dx: Any) -> Any:
        """The source cropped to *dx*'s input grid (cached per source)."""
        key = id(data_source)
        if key not in self._input_sources:
            self._input_sources[key] = SubgridSource(
                data_source, self._dx_input_coords[id(dx)]
            )
        return self._input_sources[key]

    def _run_diagnostics(
        self, x: torch.Tensor, coords: CoordSystem, member_ids: np.ndarray
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run the diagnostics on the model's own layout and return the
        outputs in the store's layout.

        Downscalers such as CorrDiff take ``[batch, time, variable, lat,
        lon]`` and drive their solar channel from the ``time`` axis, so this
        method drops the fetched ``lead_time`` singleton and adds a batch axis
        before the call; afterwards the output goes back to ``(ensemble,
        time, lead_time, variable, y, x)`` with 2-D lat/lon mapped to index
        dims.
        """
        from .diagnostic import _rename_sample_axis

        out: tuple[torch.Tensor, CoordSystem] | None = None
        for dx in self.diagnostics:
            x_in, c_in = map_coords(x, coords, self._dx_input_coords[id(dx)])
            keys = list(c_in)
            if "lead_time" in keys and x_in.shape[keys.index("lead_time")] == 1:
                x_in = x_in.squeeze(keys.index("lead_time"))
                c_in = OrderedDict((k, v) for k, v in c_in.items() if k != "lead_time")
            if "batch" not in c_in or np.size(c_in["batch"]) == 0:
                x_in = x_in.unsqueeze(0)
                c_in = OrderedDict(
                    [("batch", np.array([0]))]
                    + [(k, v) for k, v in c_in.items() if k != "batch"]
                )
            y, y_coords = dx(x_in, c_in)
            y, y_coords = _rename_sample_axis(y, y_coords, member_ids)
            y_coords, _ = _index_dims(y_coords)
            keys = list(y_coords)
            if "batch" in keys:
                y = y.squeeze(keys.index("batch"))
                y_coords = OrderedDict(
                    (k, v) for k, v in y_coords.items() if k != "batch"
                )
            keys = list(y_coords)
            if "ensemble" in keys and not self._ensemble_axis:
                # A generative model's single draw in a deterministic-size
                # campaign: the store has no ensemble axis to put it on.
                y = y.squeeze(keys.index("ensemble"))
                y_coords = OrderedDict(
                    (k, v) for k, v in y_coords.items() if k != "ensemble"
                )
            # Store layout: (ensemble, time, lead_time, variable, <spatial>).
            order = [k for k in ("ensemble", "time") if k in y_coords] + [
                k for k in y_coords if k not in ("ensemble", "time")
            ]
            y = y.permute(*[list(y_coords).index(k) for k in order])
            y_coords = OrderedDict((k, y_coords[k]) for k in order)
            t_idx = list(y_coords).index("time")
            y = y.unsqueeze(t_idx + 1)
            items = list(y_coords.items())
            items.insert(t_idx + 1, ("lead_time", self._zero_lead))
            y_coords = OrderedDict(items)
            if out is None:
                out = (y, y_coords)
            else:
                v_idx = list(y_coords).index("variable")
                merged = OrderedDict(
                    (k, np.concatenate([out[1][k], v]) if k == "variable" else v)
                    for k, v in y_coords.items()
                )
                out = (torch.cat([out[0], y], dim=v_idx), merged)
        if out is None:
            raise RuntimeError("no diagnostics produced output")
        return out

    def run_item(
        self, item: Any, data_source: Any, device: torch.device
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Run one work item, fetching from the windowed source."""
        source = self._windowed(data_source, self.diagnostics[0])
        yield from super().run_item(item, source, device)

    def run_item_batched(
        self, items: list[Any], data_source: Any, device: torch.device
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Run one initial condition's members, fetching from the windowed source."""
        source = self._windowed(data_source, self.diagnostics[0])
        yield from super().run_item_batched(items, source, device)

    def predownload_stores(self, cfg: DictConfig) -> list:
        """Declare the input store on the cropped model's input grid."""
        from src.pipelines.base import PredownloadStore
        from src.predownload_utils import (
            declare_single_source_stores,
            single_source_stores_disabled,
        )
        from src.work import build_work_items

        if single_source_stores_disabled(cfg):
            return []
        diagnostics = self._load(cfg)
        dx0 = diagnostics[0]
        ic = dx0.input_coords()
        times = sorted({i.time for i in build_work_items(cfg)})
        spatial_ref, _ = _index_dims(
            OrderedDict(
                (d, v) for d, v in dx0.output_coords(ic).items() if d != "sample"
            )
        )
        stores: list[PredownloadStore] = declare_single_source_stores(
            cfg,
            ic_variables=[str(v) for v in ic["variable"]],
            ic_times=times,
            verif_variables=list(cfg.output.variables),
            verif_times=times,
            spatial_ref=spatial_ref,
            always_separate_verification=True,
        )
        # The input store lives on the model's input window, not its output
        # grid: crop the global source onto it.
        return [
            (
                replace(st, source=SubgridSource(st.source, ic), spatial_ref=ic)
                if getattr(st, "role", "") != "verification"
                else st
            )
            for st in stores
        ]

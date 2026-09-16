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

"""Eval-recipe pipeline variants used by the scorecard campaigns.

The scorecard verifies every model against one shared ERA5 store on the
0.25 degree grid (721 x 1440). Models on a different native grid (Aurora runs
on 720 x 1440) cannot use the recipe's stock
:class:`src.pipelines.forecast.ForecastPipeline` unchanged: their output must
be placed onto the shared grid before writing.

:class:`RegriddedForecastPipeline` handles that and is a no-op for models
already on the target grid, so a campaign can always point at it::

    pipeline:
      _target_: scorecard.utils.pipelines.RegriddedForecastPipeline
      target_lat: {start: 90.0, stop: -90.0, num: 721}
      target_lon: {start: 0.0, stop: 359.75, num: 1440}

The regrid uses ``Pipeline._output_regridder``, the recipe's documented
extension point, so no recipe code is modified. For Aurora the "regrid" is a
pure row gather: its 720 latitudes are exactly the target's first 720, and the
appended -90 row carries zero latitude weight in scoring.

:class:`RegionalForecastPipeline` drives limited-area models such as
StormCast: it crops the source to the model window and predownloads any
global conditioning fields, so an event campaign runs offline like every
other campaign.
"""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from collections import OrderedDict
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig
from src.pipelines.forecast import ForecastPipeline
from src.regions import NON_SPATIAL
from src.regrid import Regridder

from earth2studio.utils.type import CoordSystem


def _axis(spec: DictConfig | dict) -> np.ndarray:
    """Build a 1D coordinate vector from a ``{start, stop, num}`` config block.

    Parameters
    ----------
    spec : DictConfig | dict
        Mapping with ``start``, ``stop`` and ``num`` entries, as written in
        the campaign config.

    Returns
    -------
    np.ndarray
        ``num`` evenly spaced float32 values from ``start`` to ``stop``.
    """
    return np.linspace(
        float(spec["start"]), float(spec["stop"]), int(spec["num"])
    ).astype(np.float32)


class SeparableNearestRegridder(Regridder):
    """Nearest-neighbour regridder between two regular lat/lon grids.

    Both grids are separable (a latitude vector by a longitude vector), so the
    nearest-neighbour map reduces to one precomputed index vector per axis and
    applying it is a pure ``index_select`` gather. Longitudes are matched on
    the circle, so a target at 359.9 matches a source at 0.0. Where the axes
    coincide -- Aurora's 720 latitudes are exactly the target's first 720 --
    the gather is the identity and values pass through bit for bit.

    This exists because the recipe's own regridders do not fit the scorecard
    case: ``NearestNeighborRegridder`` requires the optional ``earth2grid``
    CUDA extension, and ``BilinearRegridder`` fills target points outside the
    source grid (the -90 pole row) with a constant instead of a real value.

    Parameters
    ----------
    source_lats : np.ndarray
        Latitudes of the model's native grid.
    source_lons : np.ndarray
        Longitudes of the model's native grid.
    target_lats : np.ndarray
        Latitudes of the grid to write to.
    target_lons : np.ndarray
        Longitudes of the grid to write to.
    """

    def __init__(
        self,
        source_lats: np.ndarray,
        source_lons: np.ndarray,
        target_lats: np.ndarray,
        target_lons: np.ndarray,
    ) -> None:
        self._target_lat = np.asarray(target_lats, dtype=np.float32)
        self._target_lon = np.asarray(target_lons, dtype=np.float32)
        lat_idx = np.abs(
            self._target_lat[:, None]
            - np.asarray(source_lats, dtype=np.float32)[None, :]
        ).argmin(axis=1)
        # Longitude is periodic: compare on the circle so a target at 359.9
        # can match a source at 0.0 rather than snapping to the far end.
        dlon = (
            self._target_lon[:, None]
            - np.asarray(source_lons, dtype=np.float32)[None, :]
        ) % 360.0
        lon_idx = np.minimum(dlon, 360.0 - dlon).argmin(axis=1)
        self._lat_idx = torch.as_tensor(lat_idx, dtype=torch.long)
        self._lon_idx = torch.as_tensor(lon_idx, dtype=torch.long)

    def to(self, device: str | torch.device) -> SeparableNearestRegridder:
        """Move the gather indices to a device.

        Parameters
        ----------
        device : str | torch.device
            Target device.

        Returns
        -------
        SeparableNearestRegridder
            This regridder, for chaining.
        """
        self._lat_idx = self._lat_idx.to(device)
        self._lon_idx = self._lon_idx.to(device)
        return self

    def target_coords(self) -> CoordSystem:
        """Return the spatial coordinates of the target grid.

        Returns
        -------
        CoordSystem
            Ordered dict with ``lat`` and ``lon`` arrays of the target grid.
        """
        coords: CoordSystem = OrderedDict()
        coords["lat"] = self._target_lat
        coords["lon"] = self._target_lon
        return coords

    def apply(self, x: torch.Tensor, *, spatial_dims: tuple[str, ...]) -> torch.Tensor:
        """Gather the two trailing spatial dimensions onto the target grid.

        Parameters
        ----------
        x : torch.Tensor
            Tensor whose last two dimensions are latitude and longitude.
        spatial_dims : tuple[str, str]
            Names of the two trailing spatial dimensions.

        Returns
        -------
        torch.Tensor
            ``x`` with its trailing dimensions gathered onto the target grid.
        """
        if len(spatial_dims) != 2:
            raise ValueError(
                f"expects exactly two trailing spatial dims, got {spatial_dims}"
            )
        x = torch.index_select(x, -2, self._lat_idx.to(x.device))
        return torch.index_select(x, -1, self._lon_idx.to(x.device))


class RegriddedForecastPipeline(ForecastPipeline):
    """Forecast pipeline that writes model output on a configurable grid.

    Use this for any model whose native grid differs from the shared
    verification grid. It builds a :class:`SeparableNearestRegridder` from the
    model's native grid to the configured target grid and installs it via the
    recipe's ``Pipeline._output_regridder`` extension point. When the model is
    already on the target grid the regridder is skipped entirely, so the same
    pipeline is safe to configure for every model.

    Parameters
    ----------
    target_lat : DictConfig | dict
        Target latitudes as a ``{start, stop, num}`` block, e.g.
        ``{start: 90.0, stop: -90.0, num: 721}`` for ERA5.
    target_lon : DictConfig | dict
        Target longitudes as a ``{start, stop, num}`` block, e.g.
        ``{start: 0.0, stop: 359.75, num: 1440}`` for ERA5.
    share_verification : bool, optional
        If True, do not declare a verification store during predownload and
        use the shared on-disk store instead, by default False.
    """

    def __init__(
        self,
        target_lat: DictConfig | dict,
        target_lon: DictConfig | dict,
        share_verification: bool = False,
    ) -> None:
        super().__init__()
        self._target_lat = _axis(target_lat)
        self._target_lon = _axis(target_lon)
        self._share_verification = bool(share_verification)

    def predownload_stores(self, cfg: DictConfig) -> list:
        """Return the stores to predownload, without shared verification.

        The parent builds every store on the model's NATIVE grid, so an
        off-grid model would declare a 720-row verification store while the
        shared store on disk has 721 rows, and opening it fails the coordinate
        handshake. With ``share_verification: true`` the verification store is
        simply not declared here: the shared store is already fully populated,
        and scoring resolves it from disk rather than from this list. The
        initial-condition store is untouched and still fetched on the model's
        native grid, including history frames.

        Parameters
        ----------
        cfg : DictConfig
            Campaign configuration.

        Returns
        -------
        list
            Stores to predownload.
        """
        stores = super().predownload_stores(cfg)
        if not self._share_verification:
            return stores
        return [s for s in stores if getattr(s, "role", "") != "verification"]

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        """Load the model and install the output regridder if needed.

        Parameters
        ----------
        cfg : DictConfig
            Campaign configuration.
        device : torch.device
            Device to run inference on.
        """
        # Loads the model and sets self._spatial_ref to its native grid.
        super().setup(cfg, device)

        src_lat = np.asarray(self._spatial_ref["lat"], dtype=np.float32)
        src_lon = np.asarray(self._spatial_ref["lon"], dtype=np.float32)

        # No-op when the model is already on the target grid.
        if (
            src_lat.shape == self._target_lat.shape
            and src_lon.shape == self._target_lon.shape
        ):
            if np.allclose(src_lat, self._target_lat) and np.allclose(
                src_lon, self._target_lon
            ):
                return

        self._output_regridder = SeparableNearestRegridder(
            source_lats=src_lat,
            source_lons=src_lon,
            target_lats=self._target_lat,
            target_lons=self._target_lon,
        )


class ClimatologyPipeline(ForecastPipeline):
    """Forecast pipeline for the climatology baseline.

    Additions over the plain forecast pipeline, so that
    :class:`scorecard.utils.baselines.ClimatologyForecast` never touches a
    remote source inside the inference loop:

    * ``predownload_stores`` declares one extra store —
      ``climatology.zarr`` — holding the climatological field at every
      valid time of the campaign, fetched from
      ``cfg.pipeline.climatology_source`` (e.g.
      ``earth2studio.data.WB2Climatology``).
    * ``setup`` attaches that store to the model as a local
      :class:`src.data.PredownloadedSource`.

    Parameters
    ----------
    climatology_source : DictConfig | dict | DataSource
        Hydra spec (or, under Hydra's recursive instantiation, the live
        instance) of the DataSource supplying climatology on the
        verification grid.
    """

    def __init__(self, climatology_source: object) -> None:
        super().__init__()
        self._clim_source_cfg = climatology_source

    def predownload_stores(self, cfg: DictConfig) -> list:
        """Parent stores plus the campaign-valid-time climatology store."""
        import hydra
        from src.data import PredownloadedSource  # noqa: F401 (doc cross-ref)
        from src.pipelines.base import PredownloadStore
        from src.predownload_utils import compute_verification_times
        from src.work import build_work_items

        stores = super().predownload_stores(cfg)

        unique_ic_times = sorted({i.time for i in build_work_items(cfg)})
        valid_times = compute_verification_times(unique_ic_times, cfg.nsteps, 6)
        # Hydra instantiates the pipeline recursively, so under normal use
        # climatology_source arrives as a live DataSource; only a plain
        # dict/DictConfig still needs instantiating.
        source = self._clim_source_cfg
        if isinstance(source, (dict, DictConfig)):
            source = hydra.utils.instantiate(source)
        from .baselines import ERA5_LAT, ERA5_LON

        stores.append(
            PredownloadStore(
                name="climatology",
                source=source,
                times=list(valid_times),
                variables=list(cfg.output.variables),
                spatial_ref=OrderedDict({"lat": ERA5_LAT, "lon": ERA5_LON}),
                role="conditioning",
            )
        )
        return stores

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        """Load the baseline and point it at the local climatology store."""
        import os

        from src.data import PredownloadedSource

        super().setup(cfg, device)
        self.prognostic.set_source(
            PredownloadedSource(os.path.join(cfg.output.path, "climatology.zarr"))
        )


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
        sel = {d: v for d, v in self._sel.items() if d in da.dims}
        if sel:
            da = da.sel(sel, method="nearest")
        if "lead_time" in da.dims and da.sizes["lead_time"] == 1:
            # Some sources tag analyses with a zero lead; stores and the
            # online scorer expect (time, variable, <spatial...>).
            da = da.isel(lead_time=0, drop=True)
        return da.reset_coords(drop=True)


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
    from .baselines import ERA5_LAT, ERA5_LON

    lat_lo, lat_hi = float(np.min(lat)) - margin_deg, float(np.max(lat)) + margin_deg
    lon_lo, lon_hi = float(np.min(lon)) - margin_deg, float(np.max(lon)) + margin_deg
    return (
        ERA5_LAT[(ERA5_LAT >= lat_lo) & (ERA5_LAT <= lat_hi)],
        ERA5_LON[(ERA5_LON >= lon_lo) & (ERA5_LON <= lon_hi)],
    )


# How a model's coarse conditioning fields reach it: fetched into a local
# store before the run, or streamed from the source during inference.
CONDITIONING_MODES = ("predownload", "live")


class RegionalForecastPipeline(ForecastPipeline):
    """Forecast pipeline for limited-area models on a window of a larger grid.

    Two things separate regional models (StormCast is the configured example)
    Their initial conditions and truth come from a source that returns its
    whole grid while the model runs on a window of it, so the pipeline crops
    every store, and a live ``verification_source``, with :class:`SubgridSource`
    (a live ``ic_source`` needs no help: the recipe selects the window itself).
    And some of them condition each step on coarse global fields that the
    model fetches through a data-source attribute; given a
    ``conditioning_source``, the pipeline either predownloads those fields
    into ``conditioning.zarr`` on a lat/lon window around the domain and
    attaches that store as a local :class:`src.data.PredownloadedSource`,
    or, in ``live`` mode, attaches the source itself so the run writes
    nothing.
    A streaming campaign that saves only its scores looks like::

        pipeline:
          _target_: scorecard.utils.pipelines.RegionalForecastPipeline
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
    ) -> None:
        super().__init__()
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

            model = load_prognostic(cfg, self._model_node(cfg))
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
        model = load_prognostic(cfg, self._model_node(cfg))
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

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
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch
from omegaconf import DictConfig
from src.pipelines.forecast import ForecastPipeline
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

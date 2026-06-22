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

from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.dlesym import DLESyM, DLESyMLatLon
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from omegaconf import OmegaConf
    from physicsnemo import Module
except ImportError:
    OptionalDependencyFailure("dlesym")
    Module = None
    OmegaConf = None


# Climatology file names + the (mean, std) data-variable names inside each
# netCDF. Kept here so the docstring, loader, and error messages agree.
_TTR_CLIM_FILE = "era5_ttr_doy_stats_hpx64.nc"
_OLR_CLIM_FILE = "isccp_olr_doy_stats_hpx64.nc"
_TTR_CLIM_VARS = ("ttr1h_mean", "ttr1h_std")
_OLR_CLIM_VARS = ("olr_mean", "olr_std")


def apply_ttr_to_olr(
    x: torch.Tensor,
    coords: CoordSystem,
    ttr_idx: int,
    ttr_clim_mean: torch.Tensor,
    ttr_clim_std: torch.Tensor,
    olr_clim_mean: torch.Tensor,
    olr_clim_std: torch.Tensor,
    olr_floor: float = 0.0,
) -> torch.Tensor:
    """Convert ERA5 TTR to ISCCP-distributed OLR via per-doy moment matching.

    Mirrors the upstream pipeline transform at
    nathanielcresswellclay/dlesym_pipeline @ aimip:
    ``processing/utils/transform_ttr1h_olr.py``. The transform is per
    ``(face, height, width, day-of-year)`` so it preserves spatial structure
    while matching the per-pixel marginal distribution to ISCCP OLR.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with the TTR channel at axis ``-4``; shape
        ``(B, T, LT, V, F, H, W)``.
    coords : CoordSystem
        Coordinates carrying ``time`` (datetime64) and ``lead_time``
        (timedelta64) used to derive day-of-year per (T, LT) pair.
    ttr_idx : int
        Position of the TTR channel along the variable axis.
    ttr_clim_mean, ttr_clim_std : torch.Tensor
        ERA5 TTR per-doy climatology, shape ``(D, F, H, W)``.
    olr_clim_mean, olr_clim_std : torch.Tensor
        ISCCP OLR per-doy climatology, same shape.
    olr_floor : float, optional
        Lower bound applied to the transformed OLR (the upstream pipeline
        uses a small positive quantile floor). Defaults to 0.0.

    Returns
    -------
    torch.Tensor
        ``x`` with the TTR channel replaced by ISCCP-distributed OLR.
    """
    ttr = x[:, :, :, ttr_idx, :, :, :]  # (B, T, LT, F, H, W)

    times = np.asarray(coords["time"], dtype="datetime64[ns]")
    leads = np.asarray(coords["lead_time"], dtype="timedelta64[ns]")
    valid_times = times[:, None] + leads[None, :]  # (T, LT)
    doy = (
        valid_times.astype("datetime64[D]") - valid_times.astype("datetime64[Y]")
    ).astype(int)
    doy = np.clip(doy, 0, ttr_clim_mean.shape[0] - 1)

    doy_t = torch.from_numpy(doy.flatten()).long().to(ttr_clim_mean.device)

    def _gather(buf: torch.Tensor) -> torch.Tensor:
        return buf.index_select(0, doy_t).view(*doy.shape, *buf.shape[1:])

    ttr_mu = _gather(ttr_clim_mean).unsqueeze(0).to(x.device)
    ttr_sd = _gather(ttr_clim_std).unsqueeze(0).to(x.device)
    olr_mu = _gather(olr_clim_mean).unsqueeze(0).to(x.device)
    olr_sd = _gather(olr_clim_std).unsqueeze(0).to(x.device)

    ttr_scaled = ((ttr - ttr_mu) / ttr_sd) * -1.0
    olr = ttr_scaled * olr_sd + olr_mu
    olr = olr.clamp_min(olr_floor).to(x.dtype)

    x_out = x.clone()
    x_out[:, :, :, ttr_idx, :, :, :] = olr
    return x_out


def _load_clim_nc(
    path: Path, var_names: tuple[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    """Load a per-doy climatology netCDF as two ``(D, F, H, W)`` arrays.

    The bundled climatology files store ``(mean, std)`` as separate data
    variables indexed by ``(dayofyear, face, height, width)`` with
    ``dayofyear`` 1-based. We sort by ``dayofyear`` so index 0 corresponds to
    day-of-year 1, then strip the xarray metadata.
    """
    mean_name, std_name = var_names
    with xr.open_dataset(path) as ds:
        ds = ds.sortby("dayofyear")
        mean = np.asarray(ds[mean_name].values, dtype="float32")
        std = np.asarray(ds[std_name].values, dtype="float32")
    return mean, std


@check_optional_dependencies()
class DLESyMv0_ISCCP_ERA5(DLESyM):
    """DLESyMv0_ISCCP_ERA5 prognostic model for climate-timescale rollouts.

    This model packages the atmosphere and ocean checkpoints distributed by
    `AtmosSci-DLESM/DLESyM <https://github.com/AtmosSci-DLESM/DLESyM>`_ (the
    University of Washington group, Cresswell-Clay et al. 2024). It is
    designed and validated for multi-decadal to millennial climate
    integration (100–1000 year rollouts), as demonstrated in the original
    paper. This distinguishes it from
    :class:`~earth2studio.models.px.DLESyM` (``DLESyM-V1-ERA5``), which was
    optimised for subseasonal-to-seasonal (S2S) ensemble forecasting over
    lead times of days to weeks.

    The architecture is similar to
    :class:`~earth2studio.models.px.DLESyM` -- both use
    ``physicsnemo.models.dlwp_healpix`` on a HEALPix ``nside=64``
    (≈ 1°) grid with coupled atmosphere/ocean rollout -- but the upstream
    checkpoints carry a different variable set (9 atmospheric variables
    including outgoing longwave radiation trained on ISCCP-distributed OLR,
    and a 3-variable ocean coupling) rather than ERA5 TTR.

    When ``use_ttr=True`` (default), the wrapper accepts ERA5 ``ttr`` and
    applies the upstream team's per-day-of-year moment-matching transform to
    convert it to ISCCP-distributed OLR before the forward pass. The
    transform requires bundled climatology netCDFs
    (``era5_ttr_doy_stats_hpx64.nc`` with ``ttr1h_mean`` / ``ttr1h_std`` and
    ``isccp_olr_doy_stats_hpx64.nc`` with ``olr_mean`` / ``olr_std``) inside
    the model package, each indexed by ``(dayofyear, face, height, width)``.
    When ``use_ttr=False``, supply pre-transformed OLR (``rlut``) directly.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`DLESyM`.
    use_ttr : bool, optional
        If True, declare ``ttr`` as the radiative input variable and apply
        the TTR -> OLR transform internally. If False, declare ``rlut``
        directly. Defaults to True.
    ttr_clim_mean : np.ndarray | None, optional
        Per-day-of-year mean of ERA5 TTR, shape ``(D, F, H, W)`` with ``D``
        typically 366 (one entry per DOY) and ``F, H, W`` matching the
        HEALPix grid. Required when ``use_ttr=True``.
    ttr_clim_std : np.ndarray | None, optional
        Per-day-of-year std of ERA5 TTR (same shape as ``ttr_clim_mean``).
    olr_clim_mean : np.ndarray | None, optional
        Per-day-of-year mean of ISCCP OLR (same shape).
    olr_clim_std : np.ndarray | None, optional
        Per-day-of-year std of ISCCP OLR (same shape).
    olr_floor : float, optional
        Lower bound applied to the transformed OLR (the upstream pipeline
        uses a small positive quantile floor). Defaults to 0.0.
    **kwargs
        Keyword arguments forwarded to :class:`DLESyM`.

    Note
    ----
    See :class:`~earth2studio.models.px.dlesym.DLESyM` for details on the
    coupled rollout, ``retrieve_valid_atmos_outputs`` /
    ``retrieve_valid_ocean_outputs``, and the HEALPix grid layout.

    Example
    -------
    .. code-block:: python

        pkg = DLESyMv0_ISCCP_ERA5.load_default_package()
        model = DLESyMv0_ISCCP_ERA5.load_model(pkg, use_ttr=True)
        for step, (x, coords) in enumerate(model.create_iterator(x0, coords0)):
            ...

    Badges
    ------
    region:global class:cm product:wind product:temp product:atmos product:ocean year:2024
    gpu:40gb
    """

    def __init__(
        self,
        *args: Any,
        use_ttr: bool = True,
        ttr_clim_mean: np.ndarray | None = None,
        ttr_clim_std: np.ndarray | None = None,
        olr_clim_mean: np.ndarray | None = None,
        olr_clim_std: np.ndarray | None = None,
        olr_floor: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.use_ttr = use_ttr
        self.olr_floor = float(olr_floor)

        # The upstream model variable space uses Earth2Studio's ``rlut`` (the
        # user-space swap to ``ttr`` happens in ``input_coords`` when
        # ``use_ttr=True``). Validate up-front so a misconfigured config.yaml
        # fails at load time rather than during forward.
        if "rlut" not in list(self.atmos_variables):
            raise ValueError(
                "DLESyMv0_ISCCP_ERA5 expects 'rlut' to be present in atmos_variables. "
                f"Got {list(self.atmos_variables)}. Check the package's config.yaml."
            )

        if use_ttr:
            for name, arr in (
                ("ttr_clim_mean", ttr_clim_mean),
                ("ttr_clim_std", ttr_clim_std),
                ("olr_clim_mean", olr_clim_mean),
                ("olr_clim_std", olr_clim_std),
            ):
                if arr is None:
                    raise ValueError(
                        "use_ttr=True requires all four climatology arrays "
                        "(ttr_clim_mean, ttr_clim_std, olr_clim_mean, olr_clim_std)."
                    )
                if arr.ndim != 4 or arr.shape[1:] != (12, self.nside, self.nside):
                    raise ValueError(
                        f"{name} has shape {arr.shape}; expected "
                        f"(D, 12, {self.nside}, {self.nside}) where D is days-per-year."
                    )
            self.register_buffer(
                "ttr_clim_mean", torch.from_numpy(np.asarray(ttr_clim_mean)).float()
            )
            self.register_buffer(
                "ttr_clim_std", torch.from_numpy(np.asarray(ttr_clim_std)).float()
            )
            self.register_buffer(
                "olr_clim_mean", torch.from_numpy(np.asarray(olr_clim_mean)).float()
            )
            self.register_buffer(
                "olr_clim_std", torch.from_numpy(np.asarray(olr_clim_std)).float()
            )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

        When ``use_ttr=True``, ``rlut`` in the model's atmos variable list is
        replaced by ``ttr`` so that any ERA5-compatible
        :class:`~earth2studio.data.DataSource` resolves the input cleanly.
        """
        coords = super().input_coords()
        # ``use_ttr`` is set after the parent __init__ chain runs, and the
        # parent's __init__ calls ``self.input_coords()`` while computing
        # variable indices. Default to False until the subclass finishes init.
        if getattr(self, "use_ttr", False):
            variables = list(coords["variable"])
            variables[variables.index("rlut")] = "ttr"
            coords["variable"] = np.array(variables)
        return coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLESyMv0_ISCCP_ERA5 model package on HuggingFace."""
        return Package(
            "hf://nvidia/dlesym-v0-isccp-era5@924b2d62644ef61289dd960e018f60d6e067bfca",
            cache_options={
                "cache_storage": Package.default_cache("dlesym_v0_isccp_era5"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        use_ttr: bool = True,
        atmos_model_idx: int = 0,
        ocean_model_idx: int = 0,
    ) -> PrognosticModel:
        """Load the DLESyMv0_ISCCP_ERA5 prognostic from a package.

        Parameters
        ----------
        package : Package
            Model package containing ``config.yaml``, ``atmos_model_*.mdlus``,
            ``ocean_model_*.mdlus``, the HEALPix lat/lon and constant fields,
            and (when ``use_ttr=True``) ``era5_ttr_doy_stats_hpx64.nc`` and
            ``isccp_olr_doy_stats_hpx64.nc``.
        use_ttr : bool, optional
            See :class:`DLESyMv0_ISCCP_ERA5`. Defaults to True.
        atmos_model_idx : int, optional
            Index into ``cfg.models.atmos_model_checkpoints``. Defaults to 0.
        ocean_model_idx : int, optional
            Index into ``cfg.models.ocean_model_checkpoints``. Defaults to 0.

        Returns
        -------
        PrognosticModel
            Loaded ``DLESyMv0_ISCCP_ERA5`` instance.
        """
        cfg_file = Path(package.resolve("config.yaml"))
        cfg = OmegaConf.load(cfg_file)
        nside = cfg.data.nside

        atmos_ckpt = package.resolve(
            cfg.models.atmos_model_checkpoints[atmos_model_idx]
        )
        ocean_ckpt = package.resolve(
            cfg.models.ocean_model_checkpoints[ocean_model_idx]
        )
        atmos_model = Module.from_checkpoint(atmos_ckpt)
        ocean_model = Module.from_checkpoint(ocean_ckpt)
        atmos_model.output_time_dim = len(cfg.io.atmos_output_times)
        ocean_model.output_time_dim = len(cfg.io.ocean_output_times)

        # The recurrent atmos model (HEALPixRecUNet) warms up its hidden state on
        # ``presteps`` extra input windows, so it consumes
        # ``(presteps + 1) * input_time_dim`` history timesteps. A config that
        # lists too few ``atmos_input_times`` otherwise fails deep inside the
        # first convolution with an opaque channel-count mismatch.
        for name, model, times in (
            ("atmos", atmos_model, cfg.io.atmos_input_times),
            ("ocean", ocean_model, cfg.io.ocean_input_times),
        ):
            expected = (getattr(model, "presteps", 0) + 1) * model.input_time_dim
            if len(times) != expected:
                raise ValueError(
                    f"config.yaml lists {len(times)} {name}_input_times "
                    f"({list(times)}), but the {name} model has "
                    f"presteps={getattr(model, 'presteps', 0)} and "
                    f"input_time_dim={model.input_time_dim}, which requires "
                    f"{expected} history timesteps. Rebuild the package with "
                    "tools/convert_dlesym_upstream.py."
                )

        ctr = np.array(
            [cfg.data.scaling[var]["mean"] for var in cfg.io.atmos_variables]
            + [cfg.data.scaling[var]["mean"] for var in cfg.io.ocean_variables]
        )
        scl = np.array(
            [cfg.data.scaling[var]["std"] for var in cfg.io.atmos_variables]
            + [cfg.data.scaling[var]["std"] for var in cfg.io.ocean_variables]
        )
        center = ctr[None, None, None, :, None, None, None]
        scale = scl[None, None, None, :, None, None, None]

        hpx_lat = np.load(package.resolve("hpx_lat.npy"))
        hpx_lon = np.load(package.resolve("hpx_lon.npy"))
        atmos_constants = np.stack(
            [np.load(package.resolve(f"{c}.npy")) for c in cfg.io.atmos_constants],
            axis=1,
        )
        ocean_constants = np.stack(
            [np.load(package.resolve(f"{c}.npy")) for c in cfg.io.ocean_constants],
            axis=1,
        )

        ttr_clim_mean = ttr_clim_std = None
        olr_clim_mean = olr_clim_std = None
        if use_ttr:
            try:
                ttr_path = Path(package.resolve(_TTR_CLIM_FILE))
                olr_path = Path(package.resolve(_OLR_CLIM_FILE))
            except Exception as e:
                raise FileNotFoundError(
                    f"use_ttr=True requires climatology files {_TTR_CLIM_FILE} "
                    f"and {_OLR_CLIM_FILE} in the model package. They were not "
                    "found. Either rebuild the package with the climatology "
                    "netCDFs included, or pass `use_ttr=False` to disable the "
                    "TTR->OLR transform (then supply ISCCP-distributed `rlut` "
                    "directly to the model)."
                ) from e
            ttr_clim_mean, ttr_clim_std = _load_clim_nc(ttr_path, _TTR_CLIM_VARS)
            olr_clim_mean, olr_clim_std = _load_clim_nc(olr_path, _OLR_CLIM_VARS)

        return cls(
            atmos_model,
            ocean_model,
            center=center,
            scale=scale,
            atmos_constants=atmos_constants,
            ocean_constants=ocean_constants,
            hpx_lat=hpx_lat,
            hpx_lon=hpx_lon,
            nside=nside,
            atmos_input_times=np.array(
                cfg.io.atmos_input_times, dtype="timedelta64[h]"
            ),
            ocean_input_times=np.array(
                cfg.io.ocean_input_times, dtype="timedelta64[h]"
            ),
            atmos_output_times=np.array(
                cfg.io.atmos_output_times, dtype="timedelta64[h]"
            ),
            ocean_output_times=np.array(
                cfg.io.ocean_output_times, dtype="timedelta64[h]"
            ),
            atmos_variables=list(cfg.io.atmos_variables),
            ocean_variables=list(cfg.io.ocean_variables),
            atmos_coupling_variables=list(cfg.io.atmos_coupling_variables),
            ocean_coupling_variables=list(cfg.io.ocean_coupling_variables),
            use_ttr=use_ttr,
            ttr_clim_mean=ttr_clim_mean,
            ttr_clim_std=ttr_clim_std,
            olr_clim_mean=olr_clim_mean,
            olr_clim_std=olr_clim_std,
        )

    def _apply_ttr_to_olr(self, x: torch.Tensor, coords: CoordSystem) -> torch.Tensor:
        """Replace the TTR channel in ``x`` with ISCCP-distributed OLR.

        Delegates to :func:`apply_ttr_to_olr` after locating the TTR channel
        in ``coords["variable"]``. No-op if ``coords`` already advertises
        ``rlut`` (the model output space) at the radiation slot.
        """
        variables = list(coords["variable"])
        if "ttr" not in variables:
            return x  # already in OLR space (model output or rerun)
        return apply_ttr_to_olr(
            x,
            coords,
            ttr_idx=variables.index("ttr"),
            ttr_clim_mean=self.ttr_clim_mean,
            ttr_clim_std=self.ttr_clim_std,
            olr_clim_mean=self.olr_clim_mean,
            olr_clim_std=self.olr_clim_std,
            olr_floor=self.olr_floor,
        )

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs upstream DLESyM forward 1 coupled step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, T, LT, V, F, H, W)``.
        coords : CoordSystem
            Input coordinate system (with ``ttr`` in ``variable`` when
            ``use_ttr=True``).

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and output coordinates (in model variable space:
            ``rlut`` rather than ``ttr``).
        """
        output_coords = self.output_coords(coords)

        if self.use_ttr:
            x = self._apply_ttr_to_olr(x, coords)
            coords = coords.copy()
            variables = list(coords["variable"])
            if "ttr" in variables:
                variables[variables.index("ttr")] = "rlut"
                coords["variable"] = np.array(variables)

        return self._forward(x, coords), output_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        coords = coords.copy()
        # Saved for output_coords validation after each forward step: the
        # parent's output_coords validates `coords["variable"]` against
        # `self.input_coords()` (which advertises ``ttr`` in user-space).
        base_vars = coords["variable"]

        if self.use_ttr:
            x = self._apply_ttr_to_olr(x, coords)
            variables = list(coords["variable"])
            if "ttr" in variables:
                variables[variables.index("ttr")] = "rlut"
                coords["variable"] = np.array(variables)

        yield x, coords

        while True:
            x, coords = self.front_hook(x, coords)

            x = self._forward(x, coords)

            # output_coords expects the user-space variable list (``ttr``) for
            # validation; restore it from base_vars.
            base_coords = coords.copy()
            base_coords["variable"] = base_vars
            coords = self.output_coords(base_coords)

            x, coords = self.rear_hook(x, coords)

            yield x, coords.copy()

            x, coords = self._next_step_inputs(x, coords)

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Create a time-integration iterator (yields initial condition first).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            ``(x, coords)`` at each step. The first yield is the initial
            condition in model variable space (``rlut``, post-transform).
        """
        yield from self._default_generator(x, coords)


@check_optional_dependencies()
class DLESyMv0_ISCCP_ERA5LatLon(DLESyMv0_ISCCP_ERA5, DLESyMLatLon):
    """Lat/lon convenience wrapper for :class:`DLESyMv0_ISCCP_ERA5`.

    Combines the DLESyMv0_ISCCP_ERA5 climate checkpoints (see
    :class:`DLESyMv0_ISCCP_ERA5`) with the lat/lon regridding interface of
    :class:`~earth2studio.models.px.dlesym.DLESyMLatLon`. Inputs are accepted
    on the equiangular lat/lon grid (so any ERA5-compatible
    :class:`~earth2studio.data.DataSource` works directly), regridded to
    HEALPix ``nside=64`` internally, and the outputs are regridded back to
    lat/lon before being returned. This is the recommended entry point for
    most users of the upstream checkpoints, including climate-timescale
    rollouts.

    Like :class:`DLESyMv0_ISCCP_ERA5`, when ``use_ttr=True`` (default) the
    wrapper advertises ERA5 ``ttr`` as the radiative input variable and applies
    the per-day-of-year moment-matching TTR -> OLR transform internally. The
    transform is applied *after* the initial condition is regridded to HEALPix;
    subsequent rollout steps reuse the model's own OLR output, so it is only
    applied once. Derived variables (``ws10m`` from ``u10m``/``v10m`` and
    ``tau300-700`` from ``z300``/``z700``) and SST NaN-interpolation are
    handled identically to :class:`~earth2studio.models.px.dlesym.DLESyMLatLon`.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`DLESyMv0_ISCCP_ERA5`.
    **kwargs
        Keyword arguments forwarded to :class:`DLESyMv0_ISCCP_ERA5` (including
        ``use_ttr`` and the climatology arrays).

    Note
    ----
    See :class:`DLESyMv0_ISCCP_ERA5` and
    :class:`~earth2studio.models.px.dlesym.DLESyMLatLon` for details. Model
    hooks applied during iteration operate on the HEALPix grid, as with
    :class:`~earth2studio.models.px.dlesym.DLESyMLatLon`.

    Example
    -------
    .. code-block:: python

        pkg = DLESyMv0_ISCCP_ERA5LatLon.load_default_package()
        model = DLESyMv0_ISCCP_ERA5LatLon.load_model(pkg, use_ttr=True)

        # x, coords come straight from an ERA5 data source on the lat/lon grid
        x, coords = fetch_data(...)
        y, y_coords = model(x, coords)

        atmos, atmos_coords = model.retrieve_valid_atmos_outputs(y, y_coords)
        ocean, ocean_coords = model.retrieve_valid_ocean_outputs(y, y_coords)

    Badges
    ------
    region:global class:cm product:wind product:temp product:atmos product:ocean year:2024
    gpu:40gb
    """

    def _ttr_to_olr_hpx(self, x: torch.Tensor, coords_hpx: CoordSystem) -> torch.Tensor:
        """Apply the TTR -> OLR transform on a HEALPix tensor.

        The radiation channel has already been renamed ``ttr`` -> ``rlut`` in
        ``coords_hpx`` (so derived-variable prep treats it as a passthrough) but
        still carries raw ERA5 TTR values; locate it by the ``rlut`` name.
        """
        return apply_ttr_to_olr(
            x,
            coords_hpx,
            ttr_idx=list(coords_hpx["variable"]).index("rlut"),
            ttr_clim_mean=self.ttr_clim_mean,
            ttr_clim_std=self.ttr_clim_std,
            olr_clim_mean=self.olr_clim_mean,
            olr_clim_std=self.olr_clim_std,
            olr_floor=self.olr_floor,
        )

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs upstream DLESyM forward 1 coupled step, regridding to/from HEALPix.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor on the lat/lon grid, with ``ttr`` in ``variable`` when
            ``use_ttr=True``.
        coords : CoordSystem
            Input coordinate system (lat/lon).

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinates on the lat/lon grid (in model variable
            space: ``rlut`` rather than ``ttr``).
        """
        # Validate + build output coords against the user-space (``ttr``) input.
        output_coords = self.output_coords(coords)

        coords = coords.copy()
        if self.use_ttr:
            # Rename ``ttr`` -> ``rlut`` so derived-variable prep passes the
            # radiation channel through unchanged; the values are still raw ERA5
            # TTR and get transformed once we are on the HEALPix grid.
            variables = list(coords["variable"])
            if "ttr" in variables:
                variables[variables.index("ttr")] = "rlut"
                coords["variable"] = np.array(variables)

        x, coords = self._prepare_derived_variables(x, coords)

        x = self.to_hpx(x)
        coords_hpx = self.coords_to_hpx(coords)
        if self.use_ttr:
            x = self._ttr_to_olr_hpx(x, coords_hpx)

        x = self._forward(x, coords_hpx)
        x = self.to_ll(x)
        return x, output_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        coords = coords.copy()
        # Preserve the user-space variable list (``ttr``) for output_coords
        # validation after each forward step.
        base_vars = coords["variable"]

        if self.use_ttr:
            variables = list(coords["variable"])
            if "ttr" in variables:
                variables[variables.index("ttr")] = "rlut"
                coords["variable"] = np.array(variables)

        x, coords = self._prepare_derived_variables(x, coords)

        # Regrid to HEALPix and apply the TTR -> OLR transform once, on the
        # initial condition. Subsequent rollout steps reuse the model's own OLR
        # (``rlut``) output, so the transform is not reapplied. We keep ``x`` on
        # the HEALPix grid for the rollout but yield the initial condition back
        # on the lat/lon grid, in model variable space (post-transform) to match
        # :class:`DLESyMv0_ISCCP_ERA5`.
        x = self.to_hpx(x)
        if self.use_ttr:
            x = self._ttr_to_olr_hpx(x, self.coords_to_hpx(coords))

        yield self.to_ll(x), coords

        while True:
            # Front hook (operates on the HEALPix grid)
            x, coords = self.front_hook(x, coords)

            x = self._forward(x, self.coords_to_hpx(coords))

            # output_coords expects the user-space variable list (``ttr``) for
            # validation; restore it from base_vars.
            base_coords = coords.copy()
            base_coords["variable"] = base_vars
            coords = self.output_coords(base_coords)

            # Rear hook
            x, coords = self.rear_hook(x, coords)

            yield self.to_ll(x), coords.copy()

            x, coords = self._next_step_inputs(x, coords)

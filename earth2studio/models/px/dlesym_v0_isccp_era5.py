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

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.models.auto import Package
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.dlesym import (
    DLESyM,
    DLESyMAtmos,
    DLESyMOcean,
    _load_package_config,
    _PackageConfig,
)
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem, CoordSystem

try:
    import omegaconf  # noqa: F401
    import physicsnemo  # noqa: F401
except ImportError:
    OptionalDependencyFailure("dlesym")


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

    times = np.asarray(coords["time"], dtype="datetime64[s]")
    leads = np.asarray(coords["lead_time"], dtype="timedelta64[s]")
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

    ttr_scaled = ((ttr - ttr_mu) / ttr_sd.clamp(min=1e-8)) * -1.0
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


def _load_climatology(package: Package, use_ttr: bool) -> dict[str, np.ndarray]:
    """Climatology arguments of the TTR -> OLR transform, empty without it."""
    if not use_ttr:
        return {}
    ttr_mean, ttr_std = _load_clim_nc(
        Path(package.resolve(_TTR_CLIM_FILE)), _TTR_CLIM_VARS
    )
    olr_mean, olr_std = _load_clim_nc(
        Path(package.resolve(_OLR_CLIM_FILE)), _OLR_CLIM_VARS
    )
    return {
        "ttr_clim_mean": ttr_mean,
        "ttr_clim_std": ttr_std,
        "olr_clim_mean": olr_mean,
        "olr_clim_std": olr_std,
    }


def _check_history(cfg: _PackageConfig, component: str, core: torch.nn.Module) -> None:
    """Check the config lists the history a core consumes.

    The recurrent atmos model (HEALPixRecUNet) warms up its hidden state on
    ``presteps`` extra input windows, so it consumes
    ``(presteps + 1) * input_time_dim`` history timesteps. A config that lists
    too few input times otherwise fails deep inside the first convolution with
    an opaque channel-count mismatch.
    """
    times = cfg.input_times[component]
    presteps = getattr(core, "presteps", 0)
    expected = (presteps + 1) * core.input_time_dim
    if len(times) != expected:
        raise ValueError(
            f"config.yaml lists {len(times)} {component}_input_times "
            f"({times.astype(int).tolist()}), but the {component} model has "
            f"presteps={presteps} and input_time_dim={core.input_time_dim}, which "
            f"requires {expected} history timesteps. Rebuild the package with "
            "tools/convert_dlesym_upstream.py."
        )


class _TTRInput:
    """ERA5 ``ttr`` accepted as input and converted to ISCCP-distributed OLR.

    Shared by the fused model and its atmosphere component: with ``use_ttr`` the
    public input signature advertises ``ttr`` in place of ``rlut``, and the
    initial history is converted once, before the first forward pass.
    """

    use_ttr: bool
    nside: int
    olr_floor: float
    center: torch.Tensor
    ttr_clim_mean: torch.Tensor
    ttr_clim_std: torch.Tensor
    olr_clim_mean: torch.Tensor
    olr_clim_std: torch.Tensor
    register_buffer: Callable[..., None]

    def _register_climatology(
        self,
        ttr_clim_mean: np.ndarray | None,
        ttr_clim_std: np.ndarray | None,
        olr_clim_mean: np.ndarray | None,
        olr_clim_std: np.ndarray | None,
        olr_floor: float,
    ) -> None:
        self.olr_floor = float(olr_floor)
        if not self.use_ttr:
            return
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
            self.register_buffer(name, torch.from_numpy(np.asarray(arr)).float())

    def _public_input_coords(self, coords: CoordinateSystem) -> CoordinateSystem:
        """Advertise ``ttr`` in place of ``rlut`` when ``use_ttr`` is set."""
        if not self.use_ttr:
            return coords
        variables = list(coords["variable"].values)
        variables[variables.index("rlut")] = "ttr"
        return coord_array_like(coords, {"variable": np.array(variables)})

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

    def _initial_state(self, x: xr.DataArray) -> xr.DataArray:
        if not self.use_ttr:
            return x
        tensor, coords = x.e2s.to_torch()
        shape = tensor.shape
        tensor = tensor.to(self.center.device).reshape(-1, *shape[-6:])
        tensor = self._apply_ttr_to_olr(tensor, coords).reshape(shape)
        variables = list(x["variable"].values)
        variables[variables.index("ttr")] = "rlut"
        result = from_torch(
            tensor, coord_array_like(x, {"variable": np.array(variables)})
        )
        result.encoding = x.encoding.copy()
        return result


@check_optional_dependencies()
class DLESyMv0_ISCCP_ERA5(_TTRInput, DLESyM):
    """DLESyMv0_ISCCP_ERA5 prognostic model for climate-timescale rollouts.

    This model packages the atmosphere and ocean checkpoints distributed by
    [AtmosSci-DLESM/DLESyM](https://github.com/AtmosSci-DLESM/DLESyM) (the
    University of Washington group, Cresswell-Clay et al. 2024). It is
    designed and validated for multi-decadal to millennial climate
    integration (100–1000 year rollouts), as demonstrated in the original
    paper. This distinguishes it from
    [`DLESyM`][earth2studio.models.px.DLESyM] (``DLESyM-V1-ERA5``), which was
    optimised for subseasonal-to-seasonal (S2S) ensemble forecasting over
    lead times of days to weeks.

    The architecture is similar to
    [`DLESyM`][earth2studio.models.px.DLESyM] -- both use
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
    For more information see:

    - https://github.com/AtmosSci-DLESM/DLESyM
    - https://arxiv.org/abs/2409.16247 (Cresswell-Clay et al. 2024)
    - https://huggingface.co/nvidia/dlesym-v0-isccp-era5

    See [`DLESyM`][earth2studio.models.px.DLESyM] for details on the
    coupled rollout, ``retrieve_valid_atmos_outputs`` /
    ``retrieve_valid_ocean_outputs``, and the HEALPix grid layout.

    Example
    -------
    ```python
    pkg = DLESyMv0_ISCCP_ERA5.load_default_package()
    model = DLESyMv0_ISCCP_ERA5.load_model(pkg, use_ttr=True)
    for step, x in enumerate(model.create_iterator(x0)):
        ...

    ```
    Badges
    ------
    region:global class:climate product:wind product:temp product:atmos product:ocean year:2024
    gpu:40gb
    provider:nvidia backend:pytorch
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
        # The upstream model variable space uses Earth2Studio's ``rlut`` (the
        # user-space swap to ``ttr`` happens in ``input_coords`` when
        # ``use_ttr=True``). Validate before super().__init__(): the parent's
        # __init__ builds `ocean_coupling_var_idx` by looking up "rlut" (among
        # others) in atmos_variables, so a missing "rlut" would otherwise
        # surface as an opaque `list.index` ValueError from deep inside the
        # parent instead of this clear message.
        atmos_variables = kwargs.get("atmos_variables")
        if atmos_variables is not None and "rlut" not in list(atmos_variables):
            raise ValueError(
                "DLESyMv0_ISCCP_ERA5 expects 'rlut' to be present in atmos_variables. "
                f"Got {list(atmos_variables)}. Check the package's config.yaml."
            )

        # Set before super().__init__() because the parent __init__ calls
        # self.input_coords(), which (via Python dispatch) hits this subclass's
        # override and reads self.use_ttr.
        self.use_ttr = use_ttr
        super().__init__(*args, **kwargs)

        self._register_climatology(
            ttr_clim_mean, ttr_clim_std, olr_clim_mean, olr_clim_std, olr_floor
        )

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature.
        """
        return self._public_input_coords(super().input_coords())

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
        cfg = _load_package_config(package)
        kwargs = cfg.fused_kwargs(atmos_model_idx, ocean_model_idx)
        _check_history(cfg, "atmos", kwargs["atmos_model"])
        _check_history(cfg, "ocean", kwargs["ocean_model"])
        return cls(**kwargs, use_ttr=use_ttr, **_load_climatology(package, use_ttr))


@check_optional_dependencies()
class DLESyMv0_ISCCP_ERA5Atmos(_TTRInput, DLESyMAtmos):
    """Atmosphere component of
    [`DLESyMv0_ISCCP_ERA5`][earth2studio.models.px.DLESyMv0_ISCCP_ERA5], as a
    standalone prognostic model.

    Couples with
    [`DLESyMv0_ISCCP_ERA5Ocean`][earth2studio.models.px.DLESyMv0_ISCCP_ERA5Ocean]
    under :class:`~earth2studio.coupling.CoupledRunner` the way
    [`DLESyMAtmos`][earth2studio.models.px.DLESyMAtmos] couples with its ocean.
    Like the fused model, when ``use_ttr=True`` (default) it accepts ERA5
    ``ttr`` in its initial history and converts it to ISCCP-distributed OLR
    before the first forward pass; outputs report ``rlut``.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`DLESyMAtmos`.
    use_ttr : bool, optional
        If True, declare ``ttr`` as the radiative input variable and apply
        the TTR -> OLR transform internally. If False, declare ``rlut``
        directly. Defaults to True.
    ttr_clim_mean : np.ndarray | None, optional
        Per-day-of-year mean of ERA5 TTR, shape ``(D, F, H, W)``. Required when
        ``use_ttr=True``.
    ttr_clim_std : np.ndarray | None, optional
        Per-day-of-year std of ERA5 TTR (same shape as ``ttr_clim_mean``).
    olr_clim_mean : np.ndarray | None, optional
        Per-day-of-year mean of ISCCP OLR (same shape).
    olr_clim_std : np.ndarray | None, optional
        Per-day-of-year std of ISCCP OLR (same shape).
    olr_floor : float, optional
        Lower bound applied to the transformed OLR. Defaults to 0.0.
    **kwargs
        Keyword arguments forwarded to :class:`DLESyMAtmos`.

    Note
    ----
    See [`DLESyMv0_ISCCP_ERA5`][earth2studio.models.px.DLESyMv0_ISCCP_ERA5] for
    more information about the model.

    Badges
    ------
    region:global class:climate product:wind product:temp product:atmos year:2024
    gpu:40gb
    provider:nvidia backend:pytorch
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
        self.use_ttr = use_ttr
        super().__init__(*args, **kwargs)
        if "rlut" not in self.variables:
            raise ValueError(
                "DLESyMv0_ISCCP_ERA5Atmos expects 'rlut' to be present in variables. "
                f"Got {self.variables}. Check the package's config.yaml."
            )
        self._register_climatology(
            ttr_clim_mean, ttr_clim_std, olr_clim_mean, olr_clim_std, olr_floor
        )

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the atmosphere.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature.
        """
        return self._public_input_coords(super().input_coords())

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLESyMv0_ISCCP_ERA5 model package on HuggingFace."""
        return DLESyMv0_ISCCP_ERA5.load_default_package()

    @classmethod
    @check_optional_dependencies()
    def load_model(  # type: ignore[override]
        cls,
        package: Package,
        use_ttr: bool = True,
        atmos_model_idx: int = 0,
    ) -> PrognosticModel:
        """Load the atmosphere component from a package.

        Parameters
        ----------
        package : Package
            Model package, the one :class:`DLESyMv0_ISCCP_ERA5` loads.
        use_ttr : bool, optional
            See :class:`DLESyMv0_ISCCP_ERA5Atmos`. Defaults to True.
        atmos_model_idx : int, optional
            Index into ``cfg.models.atmos_model_checkpoints``. Defaults to 0.

        Returns
        -------
        PrognosticModel
            Loaded ``DLESyMv0_ISCCP_ERA5Atmos`` instance.
        """
        cfg = _load_package_config(package)
        kwargs = cfg.atmos_kwargs(atmos_model_idx)
        _check_history(cfg, "atmos", kwargs["core"])
        return cls(**kwargs, use_ttr=use_ttr, **_load_climatology(package, use_ttr))


@check_optional_dependencies()
class DLESyMv0_ISCCP_ERA5Ocean(DLESyMOcean):
    """Ocean component of
    [`DLESyMv0_ISCCP_ERA5`][earth2studio.models.px.DLESyMv0_ISCCP_ERA5], as a
    standalone prognostic model.

    Identical to [`DLESyMOcean`][earth2studio.models.px.DLESyMOcean] but loaded
    from the DLESyMv0_ISCCP_ERA5 package, whose ocean is also forced with the
    window mean of the atmosphere's ``rlut``.

    Parameters
    ----------
    *args
        Positional arguments forwarded to :class:`DLESyMOcean`.
    **kwargs
        Keyword arguments forwarded to :class:`DLESyMOcean`.

    Note
    ----
    See [`DLESyMv0_ISCCP_ERA5`][earth2studio.models.px.DLESyMv0_ISCCP_ERA5] for
    more information about the model.

    Badges
    ------
    region:global class:climate product:ocean year:2024
    gpu:40gb
    provider:nvidia backend:pytorch
    """

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLESyMv0_ISCCP_ERA5 model package on HuggingFace."""
        return DLESyMv0_ISCCP_ERA5.load_default_package()

    @classmethod
    @check_optional_dependencies()
    def load_model(  # type: ignore[override]
        cls, package: Package, ocean_model_idx: int = 0
    ) -> PrognosticModel:
        """Load the ocean component from a package.

        Parameters
        ----------
        package : Package
            Model package, the one :class:`DLESyMv0_ISCCP_ERA5` loads.
        ocean_model_idx : int, optional
            Index into ``cfg.models.ocean_model_checkpoints``. Defaults to 0.

        Returns
        -------
        PrognosticModel
            Loaded ``DLESyMv0_ISCCP_ERA5Ocean`` instance.
        """
        cfg = _load_package_config(package)
        kwargs = cfg.ocean_kwargs(ocean_model_idx)
        _check_history(cfg, "ocean", kwargs["core"])
        return cls(**kwargs)

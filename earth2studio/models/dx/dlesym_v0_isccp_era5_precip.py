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

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.dlesym_v0_isccp_era5 import (
    _OLR_CLIM_FILE,
    _OLR_CLIM_VARS,
    _TTR_CLIM_FILE,
    _TTR_CLIM_VARS,
    _load_clim_nc,
    apply_ttr_to_olr,
)
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from omegaconf import OmegaConf
    from physicsnemo import Module
    from physicsnemo.utils.insolation import insolation
except ImportError:
    OptionalDependencyFailure("dlesym")
    Module = None
    OmegaConf = None
    insolation = None


@check_optional_dependencies()
class DLESyMv0_ISCCP_ERA5Precip(torch.nn.Module, AutoModelMixin):
    """Precipitation diagnostic for the DLESyMv0_ISCCP_ERA5 climate model.

    A ``HEALPixUNet`` diagnostic from the AtmosSci-DLESM/DLESyM repository
    that predicts 6-hourly accumulated precipitation (``tp06``) from the full
    coupled atmosphere/ocean state. It is designed to be chained off
    :class:`~earth2studio.models.px.DLESyMv0_ISCCP_ERA5` (or its lat/lon
    variant).

    The model takes 2 consecutive history timesteps of 10 variables (the 9
    atmospheric variables plus ``sst``) on a HEALPix ``nside=64`` grid and
    predicts ``tp06`` at the last input timestep (i.e., precipitation
    accumulated over the 6 h prior to ``t=0``).

    When ``use_ttr=True`` (default), the wrapper accepts ERA5 ``ttr`` in
    place of ``rlut`` and applies the same per-doy moment-matching TTR -> OLR
    transform as
    :class:`~earth2studio.models.px.dlesym_v0_isccp_era5.DLESyMv0_ISCCP_ERA5`
    before the forward pass. This allows the diagnostic to run standalone
    from an ERA5 initial condition. When ``use_ttr=False``, supply
    pre-transformed ``rlut`` directly -- use this when chaining off
    :class:`~earth2studio.models.px.DLESyMv0_ISCCP_ERA5` output, which is
    already in OLR space.

    Parameters
    ----------
    core_model : torch.nn.Module
        The wrapped ``physicsnemo.models.dlwp_healpix.HEALPixUNet``.
    hpx_lat : np.ndarray
        HEALPix latitude grid, shape ``(12, nside, nside)``.
    hpx_lon : np.ndarray
        HEALPix longitude grid, shape ``(12, nside, nside)``.
    nside : int
        HEALPix nside.
    center : np.ndarray
        Per-variable input means, shape ``(1, 1, 1, V, 1, 1, 1)`` ordered by
        ``variables``.
    scale : np.ndarray
        Per-variable input stds, same shape as ``center``.
    constants : np.ndarray
        Constant fields (e.g. land-sea mask, topography), shape
        ``(12, n_constants, nside, nside)``.
    input_times : np.ndarray
        Input lead times, shape ``(input_time_dim,)``. e.g.
        ``np.array([-6, 0], dtype='timedelta64[h]')``.
    variables : list[str]
        Input variable names in channel order. Must contain ``rlut`` (the
        model variable-space name); ``input_coords`` swaps it to ``ttr``
        when ``use_ttr=True``.
    output_variable : str
        Output variable name, typically ``"tp06"``.
    log_epsilon : float | None
        If non-None, denormalize as ``exp(out + log(eps)) - eps`` to invert
        the upstream log-transform applied to precipitation during training.
        Set to ``None`` to disable.
    use_ttr : bool, optional
        Accept ERA5 ``ttr`` instead of ``rlut`` and run the TTR -> OLR
        transform internally. Defaults to True.
    ttr_clim_mean, ttr_clim_std, olr_clim_mean, olr_clim_std : np.ndarray | None
        Per-doy climatology arrays of shape ``(D, F, H, W)``. Required when
        ``use_ttr=True``.
    olr_floor : float, optional
        Lower bound applied to the transformed OLR. Defaults to 0.0.

    Note
    ----
    For more information see:

    - https://github.com/AtmosSci-DLESM/DLESyM
    - https://arxiv.org/abs/2409.16247 (the published paper backing this
      checkpoint set)

    Badges
    ------
    region:global class:cm product:precip year:2024 gpu:40gb
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        hpx_lat: np.ndarray,
        hpx_lon: np.ndarray,
        nside: int,
        center: np.ndarray,
        scale: np.ndarray,
        constants: np.ndarray,
        input_times: np.ndarray,
        variables: list[str],
        output_variable: str = "tp06",
        log_epsilon: float | None = 1e-8,
        use_ttr: bool = True,
        ttr_clim_mean: np.ndarray | None = None,
        ttr_clim_std: np.ndarray | None = None,
        olr_clim_mean: np.ndarray | None = None,
        olr_clim_std: np.ndarray | None = None,
        olr_floor: float = 0.0,
    ):
        super().__init__()
        self.core_model = core_model.eval()

        self.register_buffer("center", torch.from_numpy(center).to(dtype=torch.float32))
        self.register_buffer("scale", torch.from_numpy(scale).to(dtype=torch.float32))
        self.register_buffer(
            "constants", torch.from_numpy(constants).to(dtype=torch.float32)
        )

        self.hpx_lat = hpx_lat
        self.hpx_lon = hpx_lon
        self.nside = nside
        self.variables = list(variables)
        self.output_variable = output_variable
        self.log_epsilon = log_epsilon

        if not np.issubdtype(input_times.dtype, np.timedelta64):
            raise ValueError(
                f"input_times must be timedelta64, got {input_times.dtype}"
            )
        self.input_times = input_times

        if "rlut" not in self.variables:
            raise ValueError(
                "DLESyMv0_ISCCP_ERA5Precip expects 'rlut' to be present in variables. "
                f"Got {self.variables}. Check the package's config.yaml."
            )

        self.use_ttr = use_ttr
        self.olr_floor = float(olr_floor)

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
        """Input coordinate system of diagnostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary with 2 history timesteps and the
            full coupled-state variable list on the HEALPix grid. When
            ``use_ttr=True``, the radiation channel is advertised as
            ``ttr`` (the wrapper converts to ``rlut`` internally).
        """
        variables = list(self.variables)
        if getattr(self, "use_ttr", False) and "rlut" in variables:
            variables[variables.index("rlut")] = "ttr"
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.input_times,
                "variable": np.array(variables),
                "face": np.arange(12),
                "height": np.arange(self.nside),
                "width": np.arange(self.nside),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform.

        Returns
        -------
        CoordSystem
            Output coords with ``lead_time = [0]`` and ``variable = [tp06]``.
        """
        target_input_coords = self.input_coords()
        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        for i, key in enumerate(target_input_coords):
            if key not in ["batch", "time"]:
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        out_coords = input_coords.copy()
        out_coords["lead_time"] = np.array(
            [input_coords["lead_time"][-1]], dtype=input_coords["lead_time"].dtype
        )
        out_coords["variable"] = np.array([self.output_variable])
        return out_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLESyMv0_ISCCP_ERA5 precip package on HuggingFace."""
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
        model_idx: int = 0,
        use_ttr: bool = True,
    ) -> DiagnosticModel:
        """Load the DLESyMv0_ISCCP_ERA5 precip diagnostic from a package.

        Parameters
        ----------
        package : Package
            Model package containing ``config.yaml``, the precip ``.mdlus``
            checkpoint, the HEALPix lat/lon, the constant fields, and (when
            ``use_ttr=True``) the TTR/OLR climatology netCDFs
            (``era5_ttr_doy_stats_hpx64.nc`` and ``isccp_olr_doy_stats_hpx64.nc``).
        model_idx : int, optional
            Index into ``cfg.models.precip_model_checkpoints``. Defaults to 0.
        use_ttr : bool, optional
            See :class:`DLESyMv0_ISCCP_ERA5Precip`. Defaults to True.

        Returns
        -------
        DiagnosticModel
            Loaded ``DLESyMv0_ISCCP_ERA5Precip`` instance.
        """
        cfg_file = Path(package.resolve("config.yaml"))
        cfg = OmegaConf.load(cfg_file)
        nside = int(cfg.data.nside)

        if "precip_model_checkpoints" not in cfg.models:
            raise KeyError(
                "Package config.yaml has no precip_model_checkpoints entry. "
                "Rebuild the package with the precip model included "
                "(`tools/convert_dlesym_upstream.py` without --skip-precip)."
            )
        ckpt_path = package.resolve(cfg.models.precip_model_checkpoints[model_idx])
        core_model = Module.from_checkpoint(ckpt_path)
        core_model.output_time_dim = 1  # diagnostic always emits a single step

        variables = list(cfg.io.precip_variables)

        # The precip model was trained with its own normalization statistics,
        # which differ from the atmos/ocean stats in ``data.scaling``. Prefer the
        # precip-specific ``data.precip_scaling`` when the package provides it;
        # using the shared (atmos) stats mis-normalizes the inputs and yields
        # non-physical precip extremes. Fall back to ``data.scaling`` for older
        # packages built before ``precip_scaling`` was emitted.
        scaling = cfg.data.get("precip_scaling", None) or cfg.data.scaling
        ctr = np.array([scaling[v]["mean"] for v in variables])
        scl = np.array([scaling[v]["std"] for v in variables])
        center = ctr[None, None, None, :, None, None, None]
        scale = scl[None, None, None, :, None, None, None]

        hpx_lat = np.load(package.resolve("hpx_lat.npy"))
        hpx_lon = np.load(package.resolve("hpx_lon.npy"))
        constants = np.stack(
            [np.load(package.resolve(f"{c}.npy")) for c in cfg.io.precip_constants],
            axis=1,
        )

        input_times = np.array(cfg.io.precip_input_times, dtype="timedelta64[h]")

        output_variable = (
            list(cfg.io.precip_output_variables)[0]
            if cfg.io.get("precip_output_variables")
            else "tp06"
        )

        log_epsilon = cfg.io.get("precip_log_epsilon", None)

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
            core_model=core_model,
            hpx_lat=hpx_lat,
            hpx_lon=hpx_lon,
            nside=nside,
            center=center,
            scale=scale,
            constants=constants,
            input_times=input_times,
            variables=variables,
            output_variable=output_variable,
            log_epsilon=log_epsilon,
            use_ttr=use_ttr,
            ttr_clim_mean=ttr_clim_mean,
            ttr_clim_std=ttr_clim_std,
            olr_clim_mean=olr_clim_mean,
            olr_clim_std=olr_clim_std,
        )

    def _make_insolation(
        self, anchor_times: np.ndarray, timedeltas: np.ndarray
    ) -> torch.Tensor:
        """Build the decoder insolation tensor.

        Mirrors :meth:`~earth2studio.models.px.dlesym.DLESyM._make_insolation_tensor`:
        outputs shape ``(B, F, LT, 1, H, W)`` aligned to the core model's
        decoder input.
        """
        times_flat = np.array(
            [[a + t for t in timedeltas] for a in anchor_times]
        ).flatten()
        sol = insolation(times_flat, self.hpx_lat, self.hpx_lon)
        _, f, h, w = sol.shape
        sol = torch.from_numpy(sol).view(len(anchor_times), len(timedeltas), 1, f, h, w)
        return sol.permute(0, 3, 1, 2, 4, 5)

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run the precip diagnostic forward.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, LT, V, F, H, W)`` with ``LT = input_time_dim``
            history timesteps and ``V = len(variables)`` channels on HEALPix.
        coords : CoordSystem
            Input coordinates.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor of shape ``(B, T, 1, 1, F, H, W)`` and the
            corresponding output coords with ``variable = [tp06]``.
        """
        output_coords = self.output_coords(coords)

        if self.use_ttr:
            variables = list(coords["variable"])
            if "ttr" in variables:
                x = apply_ttr_to_olr(
                    x,
                    coords,
                    ttr_idx=variables.index("ttr"),
                    ttr_clim_mean=self.ttr_clim_mean,
                    ttr_clim_std=self.ttr_clim_std,
                    olr_clim_mean=self.olr_clim_mean,
                    olr_clim_std=self.olr_clim_std,
                    olr_floor=self.olr_floor,
                )

        x_norm = (x - self.center) / self.scale

        # Flatten batch+time, reshape from (B, T, LT, V, F, H, W) to
        # (B*T, F, LT, V, H, W) which is what HEALPixUNet.forward expects.
        if x.shape[0] > 1:
            stacked_times = np.concatenate([coords["time"]] * x.shape[0], axis=0)
        else:
            stacked_times = coords["time"]

        state = x_norm.reshape(-1, *x_norm.shape[2:]).permute(0, 3, 1, 2, 4, 5)

        anchor_times = stacked_times + coords["lead_time"][-1]
        sol = self._make_insolation(anchor_times, self.input_times)
        sol = sol.to(x.device, x.dtype)

        inputs = [state, sol, self.constants]
        out = self.core_model(inputs)
        # core_model returns (B*T, F, output_time_dim=1, 1, H, W).

        # Reshape back to (B, T, 1, 1, F, H, W).
        out = out.permute(0, 2, 3, 1, 4, 5).reshape(
            len(coords["batch"]), len(coords["time"]), 1, 1, 12, self.nside, self.nside
        )

        # Invert upstream log-precipitation transform.
        if self.log_epsilon is not None:
            eps = self.log_epsilon
            out = torch.exp(out + np.log(eps)) - eps

        return out, output_coords

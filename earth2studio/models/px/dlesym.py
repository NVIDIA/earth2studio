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
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    import earth2grid
    from omegaconf import OmegaConf
    from physicsnemo import Module
    from physicsnemo.utils.insolation import insolation
except ImportError:
    OptionalDependencyFailure("dlesym")
    Module = None
    insolation = None
    OmegaConf = None
    earth2grid = None

_ATMOS_VARIABLES = [
    "z500",
    "tau300-700",
    "z1000",
    "t2m",
    "tcwv",
    "t850",
    "z250",
    "ws10m",
]
_OCEAN_VARIABLES = ["sst"]
_ATMOS_COUPLING_VARIABLES = ["sst"]
_OCEAN_COUPLING_VARIABLES = ["z1000", "ws10m"]

_ATMOS_INPUT_TIMES = np.array([-18, -12, -6, 0], dtype="timedelta64[h]")
_OCEAN_INPUT_TIMES = np.array([-48, 0], dtype="timedelta64[h]")

_ATMOS_OUTPUT_TIMES = np.array(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],
    dtype="timedelta64[h]",
)
_OCEAN_OUTPUT_TIMES = np.array([48, 96], dtype="timedelta64[h]")


@check_optional_dependencies()
class DLESyM(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """DLESyM-V1-ERA5 prognostic model. This is an ensemble forecast model for
    global earth system modeling. This model includes an atmosphere and ocean
    component, using atmospheric variables as well as the sea-surface temperature
    on a HEALPix nside=64 (approximately 1 degree) resolution grid. The model
    architecture is a U-Net with padding operations modified to support using
    the HEALPix grid. Because the atmosphere and ocean models are predicted at
    different times, not all entries in the output tensor are valid. As a result,
    we provide convenience methods for retrieving the valid atmospheric and
    oceanic outputs.

    Parameters
    ----------
    atmos_model : torch.nn.Module
        Atmosphere model
    ocean_model : torch.nn.Module
        Ocean model
    hpx_lat : np.ndarray
        HEALPix latitude coordinates, shape (12, nside, nside)
    hpx_lon : np.ndarray
        HEALPix longitude coordinates, shape (12, nside, nside)
    nside : int
        HEALPix nside
    center : np.ndarray
        Means of the full output variable set (prognostics + diagnostics,
        in the same order as `output_coords`'s `variable` axis), shape
        (1, 1, 1, num_output_variables, 1, 1, 1)
    scale : np.ndarray
        Standard deviations of the full output variable set, same shape
        and ordering as `center`
    atmos_constants : np.ndarray
        Constants for the atmosphere model, shape (12, num_atmos_constants, nside, nside)
    ocean_constants : np.ndarray
        Constants for the ocean model, shape (12, num_ocean_constants, nside, nside)
    atmos_input_times : np.ndarray
        Atmospheric input times, shape (num_atmos_input_times,)
    ocean_input_times : np.ndarray
        Ocean input times, shape (num_ocean_input_times,)
    atmos_output_times : np.ndarray
        Atmospheric output times, shape (num_atmos_output_times,)
    atmos_variables : list[str]
        Atmospheric variables
    ocean_variables : list[str]
        Ocean variables
    atmos_coupling_variables : list[str]
        Atmospheric coupling variables
    ocean_coupling_variables : list[str]
        Ocean coupling variables
    atmos_diagnostic_variables : list[str], optional
        Atmospheric diagnostic output variables. These are produced by the
        atmos model but, unlike `atmos_variables`, are not fed back in as
        input to the next autoregressive step, by default []
    ocean_diagnostic_variables : list[str], optional
        Ocean diagnostic output variables, analogous to
        `atmos_diagnostic_variables`, by default []
    use_cln : bool, optional
        Whether the atmos/ocean models use conditional layer norm, which
        requires sampling and passing noise to the model to produce
        ensemble variability from a single set of weights, by default False
    condition_shape : int, optional
        Dimension of the conditional layer norm noise vector. Required if
        `use_cln` is True, by default None

    Note
    ----
    For more information about this model see:

    - https://arxiv.org/abs/2409.16247
    - https://arxiv.org/abs/2311.06253v2
    - https://huggingface.co/nvidia/dlesym-v1-era5

    For more information about the HEALPix grid see:

    - https://github.com/NVlabs/earth2grid

    Example
    -------
    ```python
    pkg = DLESyM.load_default_package()
    model = DLESyM.load_model(pkg)

    # Create iterator
    iterator = model.create_iterator(x, coords)

    for step, (x, coords) in enumerate(iterator):
        if step > 0:
            # Valid atmos and ocean predictions with their respective coordinates extracted below
            atmos_outputs, atmos_coords = model.retrieve_valid_atmos_outputs(x, coords)
            ocean_outputs, ocean_coords = model.retrieve_valid_ocean_outputs(x, coords)
            ...

    ```
    Badges
    ------
    region:global class:subseasonal-seasonal product:wind product:temp product:atmos product:ocean year:2025
    gpu:40gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        atmos_model: torch.nn.Module,
        ocean_model: torch.nn.Module,
        hpx_lat: np.ndarray,
        hpx_lon: np.ndarray,
        nside: int,
        center: np.ndarray,
        scale: np.ndarray,
        atmos_constants: np.ndarray,
        ocean_constants: np.ndarray,
        atmos_input_times: np.ndarray,
        ocean_input_times: np.ndarray,
        atmos_output_times: np.ndarray,
        ocean_output_times: np.ndarray,
        atmos_variables: list[str],
        ocean_variables: list[str],
        atmos_coupling_variables: list[str],
        ocean_coupling_variables: list[str],
        atmos_diagnostic_variables: list[str] | None = None,
        ocean_diagnostic_variables: list[str] | None = None,
        use_cln: bool = False,
        condition_shape: int | None = None,
    ):

        super().__init__()
        self.atmos_model = atmos_model.eval()
        self.ocean_model = ocean_model.eval()

        if use_cln and condition_shape is None:
            raise ValueError("'condition_shape' is required when 'use_cln' is True")
        self.use_cln = use_cln
        self.condition_shape = condition_shape
        self._cln_generator: torch.Generator | None = None

        self.register_buffer("center", torch.from_numpy(center).to(dtype=torch.float32))
        self.register_buffer("scale", torch.from_numpy(scale).to(dtype=torch.float32))
        self.register_buffer(
            "atmos_constants", torch.from_numpy(atmos_constants).to(dtype=torch.float32)
        )
        self.register_buffer(
            "ocean_constants", torch.from_numpy(ocean_constants).to(dtype=torch.float32)
        )

        self.hpx_lat = hpx_lat
        self.hpx_lon = hpx_lon
        self.nside = nside
        self.atmos_variables = atmos_variables
        self.ocean_variables = ocean_variables
        self.atmos_coupling_variables = atmos_coupling_variables
        self.ocean_coupling_variables = ocean_coupling_variables
        self.atmos_diagnostic_variables = atmos_diagnostic_variables or []
        self.ocean_diagnostic_variables = ocean_diagnostic_variables or []

        # Validate the input and output times
        for name, times in zip(
            [
                "atmos_input_times",
                "ocean_input_times",
                "atmos_output_times",
                "ocean_output_times",
            ],
            [
                atmos_input_times,
                ocean_input_times,
                atmos_output_times,
                ocean_output_times,
            ],
        ):
            if not np.issubdtype(times.dtype, np.timedelta64):
                raise ValueError(
                    f"Input and output times must be of type 'timedelta64', got {times.dtype} for {name}"
                )
        if ocean_input_times[0] > atmos_input_times[0]:
            raise ValueError(
                f"Ocean input time must be equal to or before atmos input time, got {ocean_input_times[0]} and {atmos_input_times[0]}"
            )
        if len(atmos_input_times) < len(ocean_input_times):
            raise ValueError(
                f"Atmos input times must be equal to or longer than ocean input times, got {len(atmos_input_times)} and {len(ocean_input_times)}"
            )
        if len(atmos_output_times) < len(ocean_output_times):
            raise ValueError(
                f"Atmos output times must be equal to or longer than ocean output times, got {len(atmos_output_times)} and {len(ocean_output_times)}"
            )

        self.atmos_input_times = atmos_input_times
        self.ocean_input_times = ocean_input_times
        # Full input times are a merge of the atmos and ocean input times:
        #  - Need to go as far back as -48h for the first ocean input time
        #  - Need to use the same temporal resolution (6h) as the atmos input times
        self.full_input_times = np.arange(
            self.ocean_input_times[0],
            self.atmos_input_times[-1] + 1,
            atmos_input_times[1] - atmos_input_times[0],
        )
        self.atmos_output_times = atmos_output_times
        self.ocean_output_times = ocean_output_times

        # Setup the insolation and coupling times from the input and output times
        self.atmos_sol_times = np.concatenate(
            [self.atmos_input_times, self.atmos_output_times]
        )
        self.ocean_sol_times = np.concatenate(
            [self.ocean_input_times, self.ocean_output_times]
        )
        n_atmos_steps = 1 + max(
            self.atmos_model.output_time_dim // self.atmos_model.input_time_dim, 1
        )
        self.atmos_coupling_times = np.array(
            [self.atmos_input_times[-1]] * n_atmos_steps
        )
        self.ocean_coupling_times = self.atmos_output_times

        # Setup the lead time indices for [atmos, ocean] [input, coupled input, output]
        in_coords = self.input_coords()
        out_coords = self.output_coords(in_coords)
        self.atmos_input_lt_idx = [
            list(in_coords["lead_time"]).index(t) for t in self.atmos_input_times
        ]
        self.ocean_input_lt_idx = [
            list(in_coords["lead_time"]).index(t) for t in self.ocean_input_times
        ]
        self.atmos_coupled_input_lt_idx = [
            list(in_coords["lead_time"]).index(t) for t in self.atmos_coupling_times
        ]
        self.ocean_coupled_input_lt_idx = [
            list(out_coords["lead_time"]).index(t) for t in self.ocean_coupling_times
        ]
        self.atmos_output_lt_idx = [
            list(out_coords["lead_time"]).index(t) for t in self.atmos_output_times
        ]
        self.register_buffer(
            "ocean_output_lt_idx",
            torch.tensor(
                [
                    list(out_coords["lead_time"]).index(t)
                    for t in self.ocean_output_times
                ],
                dtype=torch.long,
            ),
        )

        # Setup the variable indices for [atmos, ocean]
        # Input-side indices select prognostic variables out of `x`, which
        # always carries the canonical `atmos_variables + ocean_variables`
        # layout (diagnostics are dropped before being fed back in -- see
        # `_next_step_inputs`). This must NOT be sourced from
        # `self.input_coords()`: for `DLESyMLatLon`, that method reports a
        # different, lat/lon-native variable set (e.g. `u10m`/`v10m`
        # instead of `ws10m`) that only exists before
        # `_prepare_derived_variables` converts it to the canonical set --
        # `prepare_input_data` itself only ever sees the canonical layout.
        # `prognostic_variables` is `atmos_variables + ocean_variables` by
        # construction, so each component's own indices within it are just
        # the contiguous range it occupies.
        prognostic_variables = self.atmos_variables + self.ocean_variables
        self.atmos_var_idx = list(range(len(self.atmos_variables)))
        self.ocean_var_idx = list(
            range(len(self.atmos_variables), len(prognostic_variables))
        )
        self.atmos_coupling_var_idx = [
            prognostic_variables.index(var) for var in self.atmos_coupling_variables
        ]
        # Ocean coupling is built from the atmos model's own raw output
        # tensor, whose channel layout is
        # [atmos_variables, atmos_diagnostic_variables] regardless of
        # where those variables land in the merged `out_coords`.
        self.ocean_coupling_var_idx = [
            list(self.atmos_variables + self.atmos_diagnostic_variables).index(var)
            for var in self.ocean_coupling_variables
        ]

        # Output-side indices scatter each component's raw output tensor
        # (channel layout [prognostics, diagnostics]) into the merged
        # output tensor, whose variable axis follows `out_coords["variable"]`.
        # Registered as buffers (rather than plain lists) so they follow
        # the module to whatever device it's moved to.
        self.register_buffer(
            "atmos_output_var_idx",
            torch.tensor(
                [
                    list(out_coords["variable"]).index(var)
                    for var in self.atmos_variables + self.atmos_diagnostic_variables
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "ocean_output_var_idx",
            torch.tensor(
                [
                    list(out_coords["variable"]).index(var)
                    for var in self.ocean_variables + self.ocean_diagnostic_variables
                ],
                dtype=torch.long,
            ),
        )

        # Diagnostic-only output variables are dropped before being fed
        # back in as the next step's input.
        self._has_diagnostic_variables = bool(self.atmos_diagnostic_variables) or bool(
            self.ocean_diagnostic_variables
        )
        self.register_buffer(
            "_prognostic_out_idx",
            torch.tensor(
                [
                    list(out_coords["variable"]).index(var)
                    for var in self.atmos_variables + self.ocean_variables
                ],
                dtype=torch.long,
            ),
        )

        # `self.center`/`self.scale` cover the full output variable set
        # (prognostics + diagnostics, in `out_coords["variable"]` order) --
        # correct for `_denormalize_output`, but `_normalize_input` needs
        # the prognostic-only subset in `input_coords`'s order to match
        # the input tensor's (smaller) variable axis.
        self.register_buffer(
            "input_center", self.center.index_select(3, self._prognostic_out_idx)
        )
        self.register_buffer(
            "input_scale", self.scale.index_select(3, self._prognostic_out_idx)
        )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.full_input_times,
                "variable": np.array(self.atmos_variables + self.ocean_variables),
                "face": np.arange(12),
                "height": np.arange(self.nside),
                "width": np.arange(self.nside),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """

        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": self.atmos_output_times,  # atmos model has the finer temporal resolution over output lead times
                "variable": np.array(
                    self.atmos_variables
                    + self.atmos_diagnostic_variables
                    + self.ocean_variables
                    + self.ocean_diagnostic_variables
                ),
                "face": np.arange(12),
                "height": np.arange(self.nside),
                "width": np.arange(self.nside),
            }
        )

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )

        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key not in ["batch", "time"]:
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            input_coords["lead_time"][-1] + output_coords["lead_time"]
        )

        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLESyM model package on NGC

        The package's top-level ``config.yaml`` lists the available
        checkpoint versions; see :func:`load_model` for how to select
        between them.
        """
        # TODO: bump commit hash once the restructured (multi-version)
        # package layout is pushed to the HF repo.
        package = Package(
            "hf://nvidia/dlesym-v1-era5@9dbcdb83706702ac3b7d93f5dad5e535abc2fb72",
            cache_options={
                "cache_storage": Package.default_cache("dlesym"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        atmos_model_idx: int = 0,
        ocean_model_idx: int = 0,
        version: Literal["v1.0", "v1.1"] = "v1.1",
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        version : {"v1.0", "v1.1"}, optional
            Checkpoint version to load; see each version's entry in the
            package's `config.yaml` for details. `v1.1` is the checkpoint
            submitted to the ECMWF AI Weather Quest competition
            (https://aiweatherquest.ecmwf.int/). `v1.0` is the previous
            checkpoint, kept for reproducibility; loading it logs a
            deprecation warning, by default "v1.1"
        atmos_model_idx : int, optional
            Index of atmos model weights to load. Only meaningful for
            checkpoint versions that ship multiple atmos checkpoints
            (used to build ensembles without conditional layer norm), by
            default 0
        ocean_model_idx : int, optional
            Index of ocean model weights to load. Only meaningful for
            checkpoint versions that ship multiple ocean checkpoints, by
            default 0

        Returns
        -------
        PrognosticModel
            Prognostic model
        """

        top_cfg = OmegaConf.load(Path(package.resolve("config.yaml")))

        if version not in top_cfg.versions:
            raise ValueError(
                f"Unknown DLESyM checkpoint version '{version}', available "
                f"versions: {list(top_cfg.versions.keys())}"
            )

        version_cfg = top_cfg.versions[version]
        if version_cfg.get("deprecated", False):
            logger.warning(
                version_cfg.get(
                    "deprecation_message",
                    f"DLESyM checkpoint version '{version}' is deprecated.",
                )
            )

        version_path = version_cfg.path
        cfg = OmegaConf.load(Path(package.resolve(f"{version_path}/config.yaml")))
        nside = cfg.data.nside

        atmos_ckpts = cfg.models.atmos_model_checkpoints
        ocean_ckpts = cfg.models.ocean_model_checkpoints
        if atmos_model_idx != 0 and len(atmos_ckpts) == 1:
            raise ValueError(
                f"DLESyM checkpoint version '{version}' only ships a single "
                "atmos checkpoint; 'atmos_model_idx' must be 0."
            )
        if ocean_model_idx != 0 and len(ocean_ckpts) == 1:
            raise ValueError(
                f"DLESyM checkpoint version '{version}' only ships a single "
                "ocean checkpoint; 'ocean_model_idx' must be 0."
            )

        atmos_model_ckpt = package.resolve(
            f"{version_path}/atmos/{atmos_ckpts[atmos_model_idx]}"
        )
        ocean_model_ckpt = package.resolve(
            f"{version_path}/ocean/{ocean_ckpts[ocean_model_idx]}"
        )

        atmos_model = Module.from_checkpoint(atmos_model_ckpt)
        ocean_model = Module.from_checkpoint(ocean_model_ckpt)
        atmos_model.output_time_dim = len(cfg.io.atmos_output_times)
        ocean_model.output_time_dim = len(cfg.io.ocean_output_times)

        atmos_diagnostic_variables = list(cfg.io.get("atmos_diagnostic_variables", []))
        ocean_diagnostic_variables = list(cfg.io.get("ocean_diagnostic_variables", []))

        # Normalization constants, covering the full output variable set
        # (prognostics + diagnostics) in the same order as
        # `output_coords`'s `variable` axis.
        output_variables = (
            list(cfg.io.atmos_variables)
            + atmos_diagnostic_variables
            + list(cfg.io.ocean_variables)
            + ocean_diagnostic_variables
        )
        ctr = np.array([cfg.data.scaling[var]["mean"] for var in output_variables])
        scl = np.array([cfg.data.scaling[var]["std"] for var in output_variables])
        center = ctr[None, None, None, :, None, None, None]
        scale = scl[None, None, None, :, None, None, None]

        # Constant fields (shared across checkpoint versions, package root)
        hpx_lat = np.load(package.resolve("hpx_lat.npy"))
        hpx_lon = np.load(package.resolve("hpx_lon.npy"))
        atmos_constants = np.stack(
            [
                np.load(package.resolve(f"{const}.npy"))
                for const in cfg.io.atmos_constants
            ],
            axis=1,
        )
        ocean_constants = np.stack(
            [
                np.load(package.resolve(f"{const}.npy"))
                for const in cfg.io.ocean_constants
            ],
            axis=1,
        )

        cln_cfg = cfg.get("cln", {})
        use_cln = bool(cln_cfg.get("enabled", False))
        condition_shape = cln_cfg.get("condition_shape", None) if use_cln else None

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
            atmos_variables=cfg.io.atmos_variables,
            ocean_variables=cfg.io.ocean_variables,
            atmos_coupling_variables=cfg.io.atmos_coupling_variables,
            ocean_coupling_variables=cfg.io.ocean_coupling_variables,
            atmos_diagnostic_variables=atmos_diagnostic_variables,
            ocean_diagnostic_variables=ocean_diagnostic_variables,
            use_cln=use_cln,
            condition_shape=condition_shape,
        )

    def prepare_input_data(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Prepare input data for the atmos and ocean models.
        From the data in `x`, we will build a list of tensors for each model.
        Assumes `x` is a tensor of shape (batch, time, lead_time, variable, face, height, width).
        The list order is [state, insolation, constants, coupled_inputs].
        Models expect input data in the following shapes:
            state: (batch, face, lead_time, variable, height, width)
            insolation: (batch, face, lead_time, variable(=1), height, width)
            constants: (face, variable, height, width)
            coupled_inputs: (lead_time, batch, variable, face, height, width)

        Parameters
        ----------
        x : torch.Tensor
            Input data
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        tuple[List[torch.Tensor], List[torch.Tensor]]
            Prepared input data for atmos and ocean models, respectively
        """

        if x.ndim != 7:
            raise ValueError(
                f"DLESyM input data must be of shape (batch, time, lead_time, variable, face, height, width), got {x.shape} for coords {coords.keys()}"
            )

        # Flatten the batch and time dimensions
        # Before flattening, check if we have multiple batch elements
        if x.shape[0] > 1:
            # Roll the times for each batch element into one array for computing insolation
            stacked_times = np.concatenate([coords["time"]] * x.shape[0], axis=0)
        else:
            stacked_times = coords["time"]

        x = x.reshape(-1, *x.shape[2:])

        # Atmos inputs: state, insolation, constants, coupled inputs
        atmos_state = x[:, self.atmos_input_lt_idx][
            ..., self.atmos_var_idx, :, :, :
        ].permute(0, 3, 1, 2, 4, 5)
        atmos_insolation = self._make_insolation_tensor(
            anchor_times=stacked_times,
            timedeltas=self.atmos_sol_times + coords["lead_time"][-1],
        )
        atmos_coupling = self._make_atmos_coupling(x, coords)
        atmos_inputs = list(
            map(
                lambda y: y.to(x.device, x.dtype),
                [atmos_state, atmos_insolation, self.atmos_constants, atmos_coupling],
            )
        )

        # Ocean inputs: state, insolation, constants, coupled inputs
        # Coupling is not set as the ocean coupling comes after the atmos model forward pass
        ocean_state = x[:, self.ocean_input_lt_idx][
            ..., self.ocean_var_idx, :, :, :
        ].permute(0, 3, 1, 2, 4, 5)
        ocean_insolation = self._make_insolation_tensor(
            anchor_times=stacked_times,
            timedeltas=self.ocean_sol_times + coords["lead_time"][-1],
        )
        ocean_inputs = list(
            map(
                lambda y: y.to(x.device, x.dtype),
                [ocean_state, ocean_insolation, self.ocean_constants],
            )
        )

        return atmos_inputs, ocean_inputs

    def prepare_output_data(
        self,
        atmos_outputs: torch.Tensor,
        ocean_outputs: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:
        """Prepare output data for the atmos and ocean models.
        From the data in `atmos_outputs` and `ocean_outputs`, we will build a tensor of shape (batch, time, lead_time, variable, face, height, width).
        Assumes the atmosphere has the finest temporal resolution over output lead times.
        Expects input data to be of shape (batch * time, face, lead_time, variable, height, width).

        Parameters
        ----------
        atmos_outputs : torch.Tensor
            Atmos outputs
        ocean_outputs : torch.Tensor
            Ocean outputs
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        output_data : torch.Tensor
            Output data
        """

        # Variable dim covers the full output variable set (prognostics +
        # diagnostics for each component), which is larger than
        # `coords["variable"]` (the input, prognostic-only variable set)
        # whenever diagnostic output variables are configured.
        n_output_vars = len(self.atmos_output_var_idx) + len(self.ocean_output_var_idx)
        output_data = torch.empty(
            (
                len(coords["batch"]),
                len(coords["time"]),
                len(self.atmos_output_times),
                n_output_vars,
                len(coords["face"]),
                len(coords["height"]),
                len(coords["width"]),
            ),
            device=atmos_outputs.device,
        )

        # Reorder the face dim to after variable dim, restore the batch and time dimensions
        atmos_outputs = atmos_outputs.permute(0, 2, 3, 1, 4, 5)
        ocean_outputs = ocean_outputs.permute(0, 2, 3, 1, 4, 5)
        atmos_outputs = atmos_outputs.reshape(
            len(coords["batch"]), len(coords["time"]), *atmos_outputs.shape[1:]
        )
        ocean_outputs = ocean_outputs.reshape(
            len(coords["batch"]), len(coords["time"]), *ocean_outputs.shape[1:]
        )

        output_data[:, :, :, self.atmos_output_var_idx, :, :, :] = atmos_outputs
        for src_idx, dst_idx in enumerate(self.ocean_output_var_idx):
            output_data[:, :, self.ocean_output_lt_idx, dst_idx, :, :, :] = (
                ocean_outputs[:, :, :, src_idx, :, :, :]
            )

        return output_data

    @check_optional_dependencies()
    def _make_insolation_tensor(
        self, anchor_times: np.ndarray, timedeltas: np.ndarray
    ) -> torch.Tensor:
        """Make insolation tensor from anchor times and timedeltas

        Parameters
        ----------
        anchor_times : np.ndarray
            Anchor times
        timedeltas : np.ndarray
            Timedeltas

        Returns
        -------
        torch.Tensor
            Insolation tensor of shape ("batch" (anchor_times), face, lead_time, variable, height, width)
        """

        # Create insolation tensor, shape (len(anchor_times) * len(timedeltas), face, height, width)
        times_flattened = np.array(
            [[a + t for t in timedeltas] for a in anchor_times]
        ).flatten()
        sol = insolation(
            times_flattened,
            self.hpx_lat,
            self.hpx_lon,
        )
        t, f, h, w = sol.shape
        sol = torch.from_numpy(sol).view(len(anchor_times), len(timedeltas), 1, f, h, w)
        sol = sol.permute(0, 3, 1, 2, 4, 5)
        return sol

    def _make_atmos_coupling(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> torch.Tensor:
        """Make atmos coupling tensor from input data
        Assumes `x` is a tensor of shape (batch, lead_time, variable, face, height, width).
        Also assumes `x` has been properly normalized.

        Parameters
        ----------
        x : torch.Tensor
            Input data
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        torch.Tensor
            Atmos coupling tensor: shape (lead_time, batch, variable, face, height, width)
        """

        atmos_coupling = x[:, self.atmos_coupled_input_lt_idx][
            ..., self.atmos_coupling_var_idx, :, :, :
        ].permute(1, 0, 2, 3, 4, 5)

        return atmos_coupling

    def _make_ocean_coupling(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> torch.Tensor:
        """Make ocean coupling tensor from atmos outputs
        Assumes `x` is a tensor of shape (batch, face, lead_time, variable, height, width).
        Also assumes `x` has been properly normalized.

        Parameters
        ----------
        x : torch.Tensor
            Input data
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        torch.Tensor
            Ocean coupling tensor: shape (lead_time, batch, variable, face, height, width)
        """

        # Subselect the ocean coupling variables -- we use all atmos lead times here so we can average over them
        ocean_coupling = x[:, :, :, self.ocean_coupling_var_idx, :, :]

        # Slice along the lead_time dim, average, and stack along the variable dim
        # (different time-averaged quantities are defined as different variables)
        slices = ocean_coupling.chunk(len(self.ocean_output_times), dim=2)
        ocean_coupling = torch.concat(
            [s.mean(dim=2, keepdim=True) for s in slices], dim=3
        )
        ocean_coupling = ocean_coupling.permute(2, 0, 3, 1, 4, 5)
        return ocean_coupling

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input data"""

        return (x - self.input_center) / self.input_scale

    def _denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output data"""

        return x * self.scale + self.center

    def retrieve_valid_ocean_outputs(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Retrieve the valid ocean model outputs from an output data tensor.
        Because we use a dense grid of output times for the coupled model, some of the output times
        for the ocean model may not be valid because it takes a coarser time-step.
        This function will retrieve the valid outputs and return them in a tensor of shape (batch, time, lead_time, variable, face, height, width).

        Parameters
        ----------
        x : torch.Tensor
            Output data tensor
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        torch.Tensor
            Ocean outputs
        CoordSystem
            Output coordinates
        """

        self._validate_output_coords(coords)

        var_dim = list(coords.keys()).index("variable")
        lead_dim = list(coords.keys()).index("lead_time")
        out_coords = coords.copy()
        out_coords["variable"] = np.array(
            self.ocean_variables + self.ocean_diagnostic_variables
        )
        out_coords["lead_time"] = np.array(
            [t for t in coords["lead_time"] if t % self.ocean_output_times[0] == 0]
        )

        ocean_outputs = x.index_select(dim=var_dim, index=self.ocean_output_var_idx)
        ocean_outputs = ocean_outputs.index_select(
            dim=lead_dim, index=self.ocean_output_lt_idx
        )
        return ocean_outputs, out_coords

    def retrieve_valid_atmos_outputs(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Retrieve the valid atmospheric model outputs from an output data tensor.
        This function will retrieve the valid outputs and return them in a tensor of shape (batch, time, lead_time, variable, face, height, width).

        Parameters
        ----------
        x : torch.Tensor
            Output data tensor
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        torch.Tensor
            Atmos outputs
        CoordSystem
            Output coordinates
        """

        self._validate_output_coords(coords)

        var_dim = list(coords.keys()).index("variable")

        out_coords = coords.copy()
        out_coords["variable"] = np.array(
            self.atmos_variables + self.atmos_diagnostic_variables
        )

        atmos_outputs = x.index_select(dim=var_dim, index=self.atmos_output_var_idx)

        return atmos_outputs, out_coords

    def _validate_output_coords(self, coords: CoordSystem) -> None:
        """Validate the coordinates passed to the output subselection methods

        Parameters
        ----------
        coords : CoordSystem
            Output coordinates to be validated

        Raises
        ------
        ValueError
            If the coordinates are invalid (missing or incorrect length lead_time dim)
        """
        if "lead_time" not in coords:
            raise ValueError("Lead time is required in the output coordinates")
        if len(coords["lead_time"]) != len(self.atmos_output_times):
            raise ValueError(
                f"Lead time dimension length mismatch between model and coords: expected {len(self.atmos_output_times)}, got {len(coords['lead_time'])}"
            )

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the generator used to sample conditional layer norm noise.

        Matches the `set_rng(seed, reset)` convention used by other
        stochastic prognostic models in earth2studio (e.g. FCN3), so
        callers (ensemble drivers, eval recipes) can reseed this model the
        same way as any other stochastic model via
        `hasattr(model, "set_rng")`. Has no effect if `use_cln` is False.

        Parameters
        ----------
        seed : int
            Seed value
        reset : bool, optional
            Reinitialize the generator from `seed`. If False and the
            generator is already initialized, this is a no-op, by default
            True
        """
        if reset or self._cln_generator is None:
            self._cln_generator = torch.Generator().manual_seed(seed)

    def _sample_conditions_cln(
        self,
        batch_size: int,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Draw a fresh conditional layer norm noise tensor for each
        internal integration step of a component model.

        Draws are pulled from `self._cln_generator` when set via
        `set_rng`, otherwise fall back to the global RNG. Nothing is
        cached: each call advances the generator's state, so a full
        autoregressive rollout for a given seed produces a unique,
        reproducible noise trajectory across all steps rather than a
        single fixed draw held constant for the whole forecast.

        Parameters
        ----------
        batch_size : int
            Batch size, matching the target model's flattened
            (batch * time) input dimension
        n_steps : int
            Number of noise tensors to draw, matching the target model's
            `integration_steps`
        device : torch.device
            Target device
        dtype : torch.dtype
            Target dtype

        Returns
        -------
        list[torch.Tensor]
            List of length `n_steps`, each of shape
            (batch_size, condition_shape)
        """
        return [
            torch.randn(
                batch_size, self.condition_shape, generator=self._cln_generator
            ).to(device=device, dtype=dtype)
            for _ in range(n_steps)
        ]

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:

        x = self._normalize_input(x)

        # Forward pass of atmos model first
        atmos_inputs, ocean_inputs = self.prepare_input_data(x, coords)
        if self.use_cln:
            atmos_conditions = self._sample_conditions_cln(
                batch_size=atmos_inputs[0].shape[0],
                n_steps=self.atmos_model.integration_steps,
                device=x.device,
                dtype=x.dtype,
            )
            atmos_outputs = self.atmos_model(
                atmos_inputs, conditions_cln=atmos_conditions
            )
        else:
            atmos_outputs = self.atmos_model(atmos_inputs)

        # Use atmos outputs as coupling for ocean, run forward pass of ocean model
        ocean_inputs.append(self._make_ocean_coupling(atmos_outputs, coords))
        if self.use_cln:
            ocean_conditions = self._sample_conditions_cln(
                batch_size=ocean_inputs[0].shape[0],
                n_steps=self.ocean_model.integration_steps,
                device=x.device,
                dtype=x.dtype,
            )
            ocean_outputs = self.ocean_model(
                ocean_inputs, conditions_cln=ocean_conditions
            )
        else:
            ocean_outputs = self.ocean_model(ocean_inputs)

        output_data = self.prepare_output_data(atmos_outputs, ocean_outputs, coords)
        output_data = self._denormalize_output(output_data)

        return output_data

    def _next_step_inputs(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Get the inputs for the next step of the prognostic model,
        to be used with the model iterator.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
        """

        next_coords = coords.copy()
        next_coords["lead_time"] = coords["lead_time"][-len(self.full_input_times) :]

        next_x = x[:, :, -len(self.full_input_times) :, ...]

        if self._has_diagnostic_variables:
            # Diagnostic-only variables are real outputs but are not part
            # of the model's input schema -- drop them before this tensor
            # is fed back in as the next step's input.
            var_dim = list(coords.keys()).index("variable")
            next_x = next_x.index_select(dim=var_dim, index=self._prognostic_out_idx)
            next_coords["variable"] = np.array(
                self.atmos_variables + self.ocean_variables
            )

        return next_x, next_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs coupled DLESyM model forward 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system for the prediction
        """

        output_coords = self.output_coords(coords)

        return self._forward(x, coords), output_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        coords = coords.copy()

        yield x, coords

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            x = self._forward(x, coords)
            coords = self.output_coords(coords)

            # Rear hook
            x, coords = self.rear_hook(x, coords)

            yield x, coords.copy()

            x, coords = self._next_step_inputs(x, coords)

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system


        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)


class DLESyMLatLon(DLESyM):
    """DLESyM prognostic model supporting lat/lon input and output coordinates.
    This model still uses the HEALPix grid internally, but the first input is regridded
    from lat/lon and the outputs are regridded back to lat/lon upon returning from the model.
    Regridding is done using the `earth2grid` package. For convenience, we expose
    regridding methods that are accessible as `.to_hpx` and `.to_ll`.

    Note
    ----
    See :class:`DLESyM` for more information about the prognostic model. Due to the internal
    regridding, model hooks applied during iteration will need to operate on the HEALPix grid.

    Example
    -------
    ```python
    pkg = DLESyMLatLon.load_default_package()
    model = DLESyMLatLon.load_model(pkg)

    # x and coords are data defined on appropriate lat/lon grid
    x, coords = fetch_data(...)

    # Run model
    x, coords = model(x, coords)

    # Lat-lon outputs
    atmos_outputs, atmos_coords = model.retrieve_valid_atmos_outputs(x, coords)
    ocean_outputs, ocean_coords = model.retrieve_valid_ocean_outputs(x, coords)

    # HEALPix outputs
    atmos_outputs_hpx, atmos_coords_hpx = model.to_hpx(atmos_outputs), model.coords_to_hpx(atmos_coords)
    ocean_outputs_hpx, ocean_coords_hpx = model.to_hpx(ocean_outputs), model.coords_to_hpx(ocean_coords)

    ```
    Args
    ----
    *args
        Arguments for :class:`DLESyM`
    **kwargs
        Keyword arguments for :class:`DLESyM`

    Badges
    ------
    region:global class:subseasonal-seasonal product:wind product:temp product:atmos product:ocean year:2025
    gpu:40gb
    provider:nvidia backend:pytorch
    """

    @check_optional_dependencies()
    def __init__(
        self,
        atmos_model: torch.nn.Module,
        ocean_model: torch.nn.Module,
        hpx_lat: np.ndarray,
        hpx_lon: np.ndarray,
        nside: int,
        center: np.ndarray,
        scale: np.ndarray,
        atmos_constants: np.ndarray,
        ocean_constants: np.ndarray,
        atmos_input_times: np.ndarray,
        ocean_input_times: np.ndarray,
        atmos_output_times: np.ndarray,
        ocean_output_times: np.ndarray,
        atmos_variables: list[str],
        ocean_variables: list[str],
        atmos_coupling_variables: list[str],
        ocean_coupling_variables: list[str],
        atmos_diagnostic_variables: list[str] | None = None,
        ocean_diagnostic_variables: list[str] | None = None,
        use_cln: bool = False,
        condition_shape: int | None = None,
    ):

        self.lat = np.linspace(90, -90, 721, endpoint=True)
        self.lon = np.linspace(0, 360, 1440, endpoint=False)

        super().__init__(
            atmos_model=atmos_model,
            ocean_model=ocean_model,
            hpx_lat=hpx_lat,
            hpx_lon=hpx_lon,
            nside=nside,
            center=center,
            scale=scale,
            atmos_constants=atmos_constants,
            ocean_constants=ocean_constants,
            atmos_input_times=atmos_input_times,
            ocean_input_times=ocean_input_times,
            atmos_output_times=atmos_output_times,
            ocean_output_times=ocean_output_times,
            atmos_variables=atmos_variables,
            ocean_variables=ocean_variables,
            atmos_coupling_variables=atmos_coupling_variables,
            ocean_coupling_variables=ocean_coupling_variables,
            atmos_diagnostic_variables=atmos_diagnostic_variables,
            ocean_diagnostic_variables=ocean_diagnostic_variables,
            use_cln=use_cln,
            condition_shape=condition_shape,
        )

        self.hpx_grid = earth2grid.healpix.Grid(
            level=int(np.log2(self.nside)),
            pixel_order=earth2grid.healpix.HEALPIX_PAD_XY,
        )
        self.ll_grid = earth2grid.latlon.equiangular_lat_lon_grid(721, 1440)
        self.regrid_to_hpx = earth2grid.get_regridder(self.ll_grid, self.hpx_grid).to(
            torch.float32
        )
        self.regrid_to_ll = earth2grid.get_regridder(self.hpx_grid, self.ll_grid).to(
            torch.float32
        )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        coords = super().input_coords()
        coords = self.coords_to_ll(coords)

        # Modify to use the base variables instead of the derived variables
        input_variables = [
            v for v in list(coords["variable"]) if v not in ["tau300-700", "ws10m"]
        ]
        input_variables.extend(["u10m", "v10m", "z300", "z700"])
        coords["variable"] = np.array(input_variables)
        return coords

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system

        Returns
        -------
        CoordSystem
            Output coordinate system
        """
        coords = super().output_coords(input_coords)
        coords = self.coords_to_ll(coords)
        return coords

    def to_hpx(self, x: torch.Tensor) -> torch.Tensor:
        """Regrid input data to HEALPix grid. Last 2 dimensions are assumed to be
        (lat, lon)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Regridded tensor, shape (..., 12, nside, nside)
        """
        x = x.to(torch.float32)
        leading_dims = x.shape[:-2]
        x = x.reshape(-1, *x.shape[-2:])
        x = self.regrid_to_hpx(x).reshape(*leading_dims, -1)
        x = x.reshape(*leading_dims, 12, self.nside, self.nside)
        return x

    def to_ll(self, x: torch.Tensor) -> torch.Tensor:
        """Regrid output data from HEALPix grid. Last 3 dimensions are assumed to be
        (face, height, width)

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Regridded tensor, shape (..., 721, 1440)
        """
        x = x.to(torch.float32)
        leading_dims = x.shape[:-3]
        x = x.reshape(-1, np.prod(x.shape[-3:]))
        x = self.regrid_to_ll(x).reshape(*leading_dims, len(self.lat), len(self.lon))
        return x

    def coords_to_hpx(self, coords: CoordSystem) -> CoordSystem:
        """Convenience method to pop out lat/lon dimensions from coords and replace with HEALPix"""
        hpx_coords = coords.copy()
        hpx_coords.pop("lat")
        hpx_coords.pop("lon")
        hpx_coords.update(
            {
                "face": np.arange(12),
                "height": np.arange(self.nside),
                "width": np.arange(self.nside),
            }
        )
        for dim in ["face", "height", "width"]:
            hpx_coords.move_to_end(dim)
        return hpx_coords

    def coords_to_ll(self, coords: CoordSystem) -> CoordSystem:
        """Convenience method to pop out HEALPix dimensions from coords and replace with lat/lon"""
        ll_coords = coords.copy()
        ll_coords.pop("face")
        ll_coords.pop("height")
        ll_coords.pop("width")
        ll_coords.update({"lat": self.lat, "lon": self.lon})
        for dim in ["lat", "lon"]:
            ll_coords.move_to_end(dim)
        return ll_coords

    # Trailing window, in hours, over which the `ttr-3h` prognostic input
    # variable is accumulated. See `ttr_3h_query_times`/`compute_ttr_3h`
    # below for why it needs its own fetch.
    TTR_3H_WINDOW_HOURS = 3

    def ttr_3h_query_times(self, lead_time: np.ndarray | None = None) -> np.ndarray:
        """Hourly lead times (relative to the same reference time as
        `lead_time`) of raw `ttr` samples needed to compute `ttr-3h` via
        `compute_ttr_3h`.

        Fetch raw `ttr` (not `ttr-3h`) at exactly these lead times from a
        data source -- independently of, and without expanding, the
        model's main `input_coords()["lead_time"]` fetch.

        Parameters
        ----------
        lead_time : np.ndarray, optional
            Lead times `ttr-3h` will be computed at, by default
            `self.atmos_input_times`

        Returns
        -------
        np.ndarray
            Sorted, de-duplicated hourly lead time offsets (`timedelta64[h]`)
        """
        if lead_time is None:
            lead_time = self.atmos_input_times
        offsets = np.concatenate(
            [
                lt + np.arange(-self.TTR_3H_WINDOW_HOURS, 1, dtype="timedelta64[h]")
                for lt in lead_time
            ]
        )
        return np.unique(offsets)

    def compute_ttr_3h(
        self,
        raw_ttr: torch.Tensor,
        raw_coords: CoordSystem,
        target_lead_time: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Reduce raw hourly `ttr` samples into the `ttr-3h` accumulated
        input variable at each of `target_lead_time`.

        ERA5's `ttr` is accumulated since the most recent 00/12 UTC
        forecast-cycle start and is exactly 0 at each cycle boundary, so
        the accumulation over the trailing `TTR_3H_WINDOW_HOURS` window
        ending at `t` is `raw_ttr[t] - raw_ttr[t - TTR_3H_WINDOW_HOURS]`.

        Warning
        -------
        This has not been verified against the AIWQ training pipeline's
        own `ttr-3h` computation -- confirm the two agree (e.g. for a
        known date) before trusting model output that depends on it.

        Parameters
        ----------
        raw_ttr : torch.Tensor
            Raw `ttr` tensor, with a `lead_time` axis covering at least
            `ttr_3h_query_times(target_lead_time)`
        raw_coords : CoordSystem
            Coordinates for `raw_ttr`
        target_lead_time : np.ndarray, optional
            Lead times to compute `ttr-3h` at, by default
            `self.atmos_input_times`

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            `ttr-3h` tensor and coordinates, with `lead_time` set to
            `target_lead_time`
        """
        if target_lead_time is None:
            target_lead_time = self.atmos_input_times

        lead_dim = list(raw_coords.keys()).index("lead_time")
        lt_list = list(raw_coords["lead_time"])

        end_idx = [lt_list.index(t) for t in target_lead_time]
        start_idx = [
            lt_list.index(t - np.timedelta64(self.TTR_3H_WINDOW_HOURS, "h"))
            for t in target_lead_time
        ]

        end_vals = raw_ttr.index_select(
            dim=lead_dim, index=torch.tensor(end_idx, device=raw_ttr.device)
        )
        start_vals = raw_ttr.index_select(
            dim=lead_dim, index=torch.tensor(start_idx, device=raw_ttr.device)
        )
        ttr_3h = end_vals - start_vals

        out_coords = raw_coords.copy()
        out_coords["lead_time"] = np.array(target_lead_time)
        return ttr_3h, out_coords

    def _nan_interpolate_sst(
        self, sst: torch.Tensor, coords: CoordSystem
    ) -> torch.Tensor:
        """Custom interpolation to fill NaNs over landmasses in SST data."""

        da_sst = xr.DataArray(sst.cpu().numpy(), dims=coords.keys())
        da_interp = da_sst.interpolate_na(
            dim="lon", method="linear", use_coordinate=False
        )

        # Second pass: roll, interpolate along longitude, and unroll
        roll_amount_lon = int(len(da_interp.lon) / 2)
        da_double_interp = (
            da_interp.roll(lon=roll_amount_lon, roll_coords=False)
            .interpolate_na(dim="lon", method="linear", use_coordinate=False)
            .roll(lon=len(da_interp.lon) - roll_amount_lon, roll_coords=False)
        )

        # Third pass do a similar roll along latitude
        roll_amount_lat = int(len(da_double_interp.lat) / 2)
        da_triple_interp = (
            da_double_interp.roll(lat=roll_amount_lat, roll_coords=False)
            .interpolate_na(dim="lat", method="linear", use_coordinate=False)
            .roll(lat=len(da_double_interp.lat) - roll_amount_lat, roll_coords=False)
        )

        return torch.from_numpy(da_triple_interp.values).to(sst.device)

    def _prepare_derived_variables(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepare derived variables for the DLESyM model.

        This method handles the preparation of derived variables from the input tensor
        and coordinates. It ensures that the derived variables are correctly computed,
        and performs NaN-interpolation on the SST data.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system for the derived variables
        """

        prep_coords = coords.copy()

        # Fetch the base variables
        base_vars = list(prep_coords["variable"])
        src_vars = {
            v: x[..., base_vars.index(v) : base_vars.index(v) + 1, :, :]
            for v in base_vars
        }

        # Compute the derived variables
        out_vars = {
            "ws10m": torch.sqrt(src_vars["u10m"] ** 2 + src_vars["v10m"] ** 2),
            "tau300-700": src_vars["z300"] - src_vars["z700"],
        }
        out_vars.update(src_vars)

        # Fill SST nans by custom interpolation
        out_vars["sst"] = self._nan_interpolate_sst(out_vars["sst"], coords)

        # Update the tensor with the derived variables and return
        prep_coords["variable"] = np.array(self.atmos_variables + self.ocean_variables)
        x_out = torch.empty(
            *[v.shape[0] for v in prep_coords.values()], device=x.device
        )
        for i, v in enumerate(prep_coords["variable"]):
            x_out[..., i : i + 1, :, :] = out_vars[v]

        return x_out, prep_coords

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs coupled DLESyM model forward 1 step, regridding to/from HEALPix grid

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system for the prediction
        """
        output_coords = self.output_coords(coords)

        x, coords = self._prepare_derived_variables(x, coords)

        x = self.to_hpx(x)
        x = self._forward(x, self.coords_to_hpx(coords))
        x = self.to_ll(x)
        return x, output_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        coords = coords.copy()

        base_vars = coords["variable"]

        x, coords = self._prepare_derived_variables(x, coords)

        yield x, coords

        x = self.to_hpx(x)

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            x = self._forward(x, self.coords_to_hpx(coords))

            # Output coords expects the input variable set to include base variables,
            # but will return the ouptut variables with the derived variables
            base_coords = coords.copy()
            base_coords["variable"] = base_vars
            coords = self.output_coords(base_coords)

            # Rear hook
            x, coords = self.rear_hook(x, coords)

            yield self.to_ll(x), coords.copy()

            x, coords = self._next_step_inputs(x, coords)

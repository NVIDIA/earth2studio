# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
from typing import Any, Dict, List, Tuple
from datetime import datetime, timezone
import numpy as np
import torch
import torch.nn as nn
import json

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    handshake_size,
)
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem
from earth2studio.data import HRRR, GFS_FX
from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.data.utils import fetch_data


try:
    # Optional dependency: physicsnemo and its sampler are required to run inference.
    from physicsnemo.models import Module
    from physicsnemo.utils.generative import deterministic_sampler  # noqa: F401
    from physicsnemo.utils.zenith_angle import cos_zenith_angle
except ImportError:
    OptionalDependencyFailure("obscast")
    Module = None  # type: ignore[assignment]
    deterministic_sampler = None  # type: ignore[assignment]
    cos_zenith_angle = None  # type: ignore[assignment]

from earth2studio.models.nn.scv2_util import EDMPrecond, DropInDiT, edm_sampler # TODO remove when upstreamed
def model_wrap(model: Module) -> nn.Module:
    """Wrap a physicsnemo Module so it is compatible with the preconditioning and sampler used by ObsCast.
    TODO: Remove once core EDMPrecond architecture is fully upstreamed
    """

    return EDMPrecond(
        model=DropInDiT(
            pnm=model,
        ),
    )

@check_optional_dependencies()
class ObsCastBase(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """ObsCast diffusion-only prognostic base model with staged denoising.
    Variants should subclass to define dataset/resolution specifics (e.g., grids,
    variables, and data loading).

    This base class supports an "ensemble of experts" approach where the diffusion
    sampler is invoked in multiple stages, each using a different set of model
    weights applicable to a specific sigma range. Stages are defined by `model_spec`.

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications. Each entry must contain:
        - 'model': physicsnemo.Module
        - 'sigma_min': float
        - 'sigma_max': float
        The sigma interval determines which expert is applied during denoising.
    means : torch.Tensor
        Per-variable mean used to normalize inputs. Expected shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std used to normalize inputs. Expected shape [1, C, 1, 1].
    variables : np.ndarray
        Names of prognostic variables corresponding to the high-resolution state.
    latitudes : torch.Tensor
        Latitudes of the grid, expected shape [H, W]
    longitudes : torch.Tensor
        Longitudes of the grid, expected shape [H, W]
    conditioning_means : torch.Tensor | None, optional
        Means to normalize any external conditioning data, by default None.
    conditioning_stds : torch.Tensor | None, optional
        Stds to normalize any external conditioning data, by default None.
    conditioning_variables : np.ndarray | None, optional
        Names of external conditioning variables, by default None.
    conditioning_data_source : Any | None, optional
        Data source for external conditioning. Subclasses should define how this is
        used; base class does not fetch conditioning by default. Defaults to None.
    sampler_args : dict[str, float | int], optional
        Default sampler arguments passed to the diffusion sampler for every stage,
        by default {}.
    time_step_size : np.timedelta64, optional
        Time step size for the model, by default 1 hour.
    y_coords : np.ndarray | None, optional
        Y coordinates of the grid, expected shape [H, W]. Defaults to None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    x_coords : np.ndarray | None, optional
        X coordinates of the grid, expected shape [H, W]. Defaults to None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    invalid_mask: torch.Tensor | None, optional
        Mask specifying invalid gridpoints, expected shape [H, W] and dtype bool with
        True values indicating invalid gridpoints. Defaults to None, in which case the
        case the model does not mask invalid gridpoints.
    """
    _CENTRAL_LAT_CONSTANT = 38.5
    _LAT_SCALE = 15
    _CENTRAL_LON_CONSTANT = 262.5
    _LON_SCALE = 36

    def __init__(
        self,
        model_spec: List[Dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        variables: np.ndarray,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_variables: np.ndarray | None = None,
        conditioning_data_source: Any | None = None,
        sampler_args: Dict[str, float | int] | None = None,
        time_step_size: np.timedelta64 = np.timedelta64(1, "h"),
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        invalid_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        # Validate and store staged models
        if not isinstance(model_spec, list) or len(model_spec) == 0:
            raise ValueError("model_spec must be a non-empty list of stage dicts.")

        self.time_step_size = time_step_size
        self.register_buffer("latitudes", latitudes)
        self.register_buffer("longitudes", longitudes)
        self._lat_cpu_copy = self.latitudes.cpu().numpy()
        self._lon_cpu_copy = self.longitudes.cpu().numpy()
        
        if invalid_mask is not None:
            self.register_buffer("invalid_mask", invalid_mask)
        else:
            self.register_buffer("invalid_mask", torch.zeros_like(latitudes, dtype=torch.bool))

        if y_coords is not None and x_coords is not None:
            self.y = y_coords
            self.x = x_coords
        else:
            self.y = np.arange(latitudes.shape[0])
            self.x = np.arange(longitudes.shape[1])
        
        for i, spec in enumerate(model_spec):
            if not isinstance(spec, dict):
                raise TypeError(f"model_spec[{i}] must be a dict.")
            for key in ("model", "sigma_min", "sigma_max"):
                if key not in spec:
                    raise KeyError(f"model_spec[{i}] missing required key '{key}'.")
        
        # Sort stages by descending sigma_max to ensure large->small sampling schedule
        self.model_spec = sorted(
            model_spec, key=lambda s: float(s["sigma_max"]), reverse=True
        )
        # Store in ModuleList so `.to(device)` works
        self.stage_models = nn.ModuleList([spec["model"] for spec in self.model_spec])

        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.variables = variables

        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)
        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)

        self.conditioning_variables = conditioning_variables
        self.conditioning_data_source = conditioning_data_source
        self.sampler_args = sampler_args or {}
        self.variables = variables

    @classmethod
    def load_default_package(cls) -> Package:
        """Load a default local package for ObsCast models."""
        package = Package(
            "/lustre/fsw/portfolios/coreai/users/pharrington/model_pkg/scv2",
            cache_options={
                "cache_storage": Package.default_cache("obscast"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    def load_model(
        cls,
        package: Package,
        *args: Any,
        **kwargs: Any,
    ) -> DiagnosticModel:
        """Load model from package. Must be implemented by subclasses."""
        raise NotImplementedError("ObsCastBase.load_model must be implemented by a subclass.")


    def input_coords(self) -> CoordSystem:
        """Input coordinate system. Subclasses should override for specific variants."""
        raise NotImplementedError(
            "ObsCastBase.input_coords must be implemented by a subclass."
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of prognostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output coordinates.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary for the model output.
        """
        raise NotImplementedError(
            "ObsCastBase.output_coords must be implemented by a subclass."
        )

    def fetch_conditioning(
        self, coords: CoordSystem, device: torch.device
    ) -> Tuple[torch.Tensor | None, CoordSystem | None]:
        """Fetch external conditioning data. Subclasses should override.

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system.
        device : torch.device
            Device on which the conditioning tensor should reside.

        Returns
        -------
        tuple[torch.Tensor | None, CoordSystem | None]
            Conditioning tensor aligned with `coords`, or (None, None) if unused.
        """
        raise NotImplementedError(
            "ObsCastBase.fetch_conditioning must be implemented by a subclass."
        )

    def normalize_conditioning(self, conditioning: torch.Tensor | None) -> torch.Tensor | None:
        """Normalize external conditioning with stored stats if available."""
        if conditioning is None:
            return None
        x = conditioning
        if "conditioning_means" in self._buffers:
            x = x - self.conditioning_means
        if "conditioning_stds" in self._buffers:
            x = x / self.conditioning_stds
        return x

    def build_condition(
        self,
        x_norm: torch.Tensor,
        coords: CoordSystem,
        conditioning_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Construct the low-resolution conditioning input to the diffusion model.
        Subclasses can override to customize the channel layout.

        Parameters
        ----------
        x_norm : torch.Tensor
            Normalized input high-resolution state.
        coords : CoordSystem
            Input coordinate system to provide time and lead time for cos zenith angle calculation
        conditioning_norm : torch.Tensor | None, optional
            Optional normalized external conditioning, by default None.

        Returns
        -------
        torch.Tensor
            Conditioning tensor to pass as `img_lr` to the diffusion sampler.
        """
        parts = [x_norm]
        if conditioning_norm is not None:
            parts.append(conditioning_norm)
        if self.latitudes is not None and self.longitudes is not None:
            normed_lat = (self.latitudes - self._CENTRAL_LAT_CONSTANT) / self._LAT_SCALE
            normed_lon = (self.longitudes + 180.0) % 360.0 - 180.0 # trained models expect [-180, 180] longitudes
            normed_lon = (normed_lon - self._CENTRAL_LON_CONSTANT) / self._LON_SCALE
            latlon_input = torch.cat((normed_lat[None, None, :, :], normed_lon[None, None, :, :]), dim=1)
            parts.append(latlon_input)
        
        input_cz_times = coords["time"] + coords["lead_time"][-1]
        target_cz_times = input_cz_times + self.time_step_size
        if isinstance(input_cz_times, np.ndarray):
            input_cz_times = np.array([datetime.fromisoformat(str(t)[:19]).replace(tzinfo=timezone.utc) for t in input_cz_times])
            target_cz_times = np.array([datetime.fromisoformat(str(t)[:19]).replace(tzinfo=timezone.utc) for t in target_cz_times])
        
        cz_0 = cos_zenith_angle(input_cz_times, self._lon_cpu_copy, self._lat_cpu_copy)
        cz_1 = cos_zenith_angle(target_cz_times, self._lon_cpu_copy, self._lat_cpu_copy)
        parts.extend(
            [
                torch.from_numpy(cz_0).to(x_norm.device)[None, None, :, :],
                torch.from_numpy(cz_1).to(x_norm.device)[None, None, :, :],
            ]
        )

        return torch.cat(parts, dim=1)

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run a single prognostic step using staged diffusion denoising.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for a single step with shape [B, T=1, L=1, C, H, W] or
            collapsed batch-time-lead dims depending on the caller's batching.
        coords : CoordSystem
            Coordinates describing `x`. Used by subclasses if needed.
        conditioning : torch.Tensor | None, optional
            External conditioning aligned to `x` if used, by default None.

        Returns
        -------
        torch.Tensor
            Output tensor for the same step with denoised prognostic variables.
        """
        # Scale input
        x_norm = (x - self.means) / self.stds
        x_norm = torch.where(self.invalid_mask[None, None, :, :], torch.zeros_like(x_norm), x_norm)
        
        # Optional conditioning
        conditioning_norm = self.normalize_conditioning(conditioning)

        # Base conditioning to diffusion model
        condition = self.build_condition(x_norm, coords=coords, conditioning_norm=conditioning_norm)

        # Single noise initialization shared across stages
        latents = torch.randn_like(x_norm)

        # Aggregate corrections over sigma bands
        total_correction = torch.zeros_like(x_norm)
        for i, stage in enumerate(self.model_spec):
            stage_args = dict(self.sampler_args)
            stage_args.update(
                {
                    "sigma_min": float(stage["sigma_min"]),
                    "sigma_max": float(stage["sigma_max"]),
                    "num_steps": 100,
                    "S_churn": 10,
                }
            )

            ## FORCED INPUTS
            srcpath = "/lustre/fsw/portfolios/coreai/users/pharrington/dev/scv2/stormcast-v2/"
            # Make side-by-side imshow plots per channel
            for ch in range(condition.shape[1]):
                import matplotlib.pyplot as plt
                ref = torch.from_numpy(np.load(srcpath + "inp_.npy")).to(x_norm.device)[:1]
                print(condition.shape, ref.shape)
                plt.figure(figsize=(20,10))
                plt.subplot(1, 3, 1)
                plt.imshow(ref[0, ch, :, :].cpu().numpy(), cmap="viridis")
                plt.colorbar(orientation="horizontal")
                plt.title(f"REF CH {ch}")
                plt.subplot(1, 3, 2)
                plt.imshow(condition[0, ch, :, :].cpu().numpy(), cmap="viridis")
                plt.colorbar(orientation="horizontal")
                plt.title(f"ACTUAL CH {ch}")
                diff = condition[0, ch, :, :].cpu().numpy() - ref[0, ch, :, :].cpu().numpy()
                cmax = np.nanmax(np.abs(diff))
                plt.subplot(1, 3, 3)
                plt.imshow(diff, cmap="bwr", vmin=-cmax, vmax=cmax)
                plt.colorbar(orientation="horizontal")
                plt.title(f"DIFF CH {ch}")
                plt.savefig(f"xxx_condition_ch{ch}.png", dpi=300)
                plt.close()

            # latents = torch.from_numpy(np.load(srcpath + "latents.npy")).to(x_norm.device)[:1]
            # condition = torch.from_numpy(np.load(srcpath + "inp_.npy")).to(x_norm.device)[:1]
            # pred = torch.from_numpy(np.load(srcpath + "pred.npy")).to(x_norm.device)[:1]
            edm_out = edm_sampler(  # type: ignore[misc]
                self.stage_models[i],
                latents=latents,
                condition=condition,
                **stage_args,
            )
            total_correction = total_correction + edm_out

        out = total_correction
        out = out * self.stds + self.means
        return out

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs the prognostic model one step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system.
        """

        # Handle invalid gridpoints: initially zero-fill any NaNs so they don't cause problems
        # Data will be zero-filled again after normalization in the forward method.
        # TODO better broadcasting logic here
        x = torch.where(self.invalid_mask[None, None, None, :, :], 0., x)

        if torch.isnan(x).any():
            raise ValueError("NaNs found in input data which were not masked by the model's invalid_mask")

        # Allow subclasses to provide external conditioning; optional for base.
        conditioning, conditioning_coords = self.fetch_conditioning(coords, device=x.device)
        # If provided, expand to batch to align with `x` (following StormCast pattern).
        if conditioning is not None:
            # Broadcast to batch dimension if needed. Expect [B, T, L, C, H, W].
            if conditioning.dim() == x.dim() - 1:
                conditioning = conditioning.repeat(x.shape[0], 1, 1, 1, 1, 1)

        output_coords = self.output_coords(coords)

        for i, _ in enumerate(coords.get("batch", np.empty(0))):
            for j, _ in enumerate(coords.get("time", np.empty(0))):
                for k, _ in enumerate(coords.get("lead_time", np.empty(0))):
                    cond_slice = None
                    if conditioning is not None:
                        cond_slice = conditioning[i, j, k : k + 1]
                    x[i, j, k : k + 1] = self._forward(
                        x[i, j, k : k + 1],
                        coords,
                        conditioning=cond_slice,
                    )

        return x, output_coords

    @batch_func()
    def _default_generator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()
        self.output_coords(coords)
        yield x, coords
        while True:
            x, coords = self.front_hook(x, coords)
            x, coords = self.__call__(x, coords)
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates an iterator to perform time-integration of the prognostic model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model containing the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)


class ObsCastGOES(ObsCastBase):
    """ObsCast variant for GOES inputs on the HRRR grid.
    Single input lead time and predicts one step 1 hour ahead.

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications; see `ObsCastBase`.
    means : torch.Tensor
        Per-variable mean for normalization, shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std for normalization, shape [1, C, 1, 1].
    hrrr_lat_lim : tuple[int, int], optional
        HRRR grid latitude limits, by default (273, 785).
    hrrr_lon_lim : tuple[int, int], optional
        HRRR grid longitude limits, by default (579, 1219).
    variables : np.ndarray, optional
        GOES input variables, by default:
        ["abi01c","abi02c","abi03c","abi07c","abi08c","abi09c","abi10c"].
    conditioning_variables : np.ndarray, optional
        Auxiliary conditioning variables, by default ["z500"].
    conditioning_means : torch.Tensor | None, optional
        Means for any external conditioning, by default None.
    conditioning_stds : torch.Tensor | None, optional
        Stds for any external conditioning, by default None.
    conditioning_data_source : Any | None, optional
        Data source for external conditioning, by default None.
    sampler_args : dict[str, float | int], optional
        Default sampler args used for each diffusion stage, by default {}.
    latitudes : torch.Tensor | None, optional
        Latitudes of the grid, by default None.
    longitudes : torch.Tensor | None, optional
        Longitudes of the grid, by default None.
    invalid_mask: torch.Tensor | None, optional
        Mask specifying invalid gridpoints, expected shape [H, W] and dtype bool with
        True values indicating invalid gridpoints. Defaults to None, in which case the
        case the model does not mask invalid gridpoints.
    """

    def __init__(
        self,
        model_spec: List[Dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        variables: np.ndarray = np.array(
            ["abi01c", "abi02c", "abi03c", "abi07c", "abi08c", "abi09c", "abi10c", "abi13c"]
        ),
        conditioning_variables: np.ndarray = np.array(["z500"]),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_data_source: Any | None = None,
        sampler_args: Dict[str, float | int] | None = None,
        time_step_size: np.timedelta64 = np.timedelta64(1, "h"),
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        invalid_mask: torch.Tensor | None = None,
    ):

        super().__init__(
            model_spec=model_spec,
            means=means,
            stds=stds,
            variables=variables,
            latitudes=latitudes,
            longitudes=longitudes,
            conditioning_means=conditioning_means,
            conditioning_variables=conditioning_variables,
            conditioning_stds=conditioning_stds,
            conditioning_data_source=conditioning_data_source,
            sampler_args=sampler_args or {},
            time_step_size=time_step_size,
            y_coords=y_coords,
            x_coords=x_coords,
            invalid_mask=invalid_mask,
        )
        self.means = self.means[:, -len(self.variables):, :, :]
        self.stds = self.stds[:, -len(self.variables):, :, :]

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
                "y": self.y,
                "x": self.x,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output coordinates.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(1, "h")]),
                "variable": np.array(self.variables),
                "y": self.y,
                "x": self.x,
            }
        )
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "x", 5)
        handshake_dim(input_coords, "y", 4)
        handshake_dim(input_coords, "variable", 3)
        handshake_size(input_coords, "y", self.latitudes.shape[0])
        handshake_size(input_coords, "x", self.latitudes.shape[1])
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = output_coords["lead_time"] + input_coords["lead_time"]
        return output_coords

    def fetch_conditioning(
        self, coords: CoordSystem, device: torch.device
    ) -> Tuple[torch.Tensor | None, CoordSystem | None]:
        """Fetch external conditioning data.

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system.
        device : torch.device
            Device on which the conditioning tensor should reside.
        """

        if self.conditioning_data_source is None:
            raise RuntimeError(
                "ObsCastGOES has been called without initializing the model's conditioning_data_source"
            )

        conditioning, conditioning_coords = fetch_data(
            self.conditioning_data_source,
            time=coords["time"],
            variable=self.conditioning_variables,
            lead_time=coords["lead_time"],
            device=device,
            interp_to=coords | {"_lat": self._lat_cpu_copy, "_lon": self._lon_cpu_copy},
            interp_method="linear",
        )
        return conditioning, conditioning_coords

    @classmethod
    def load_model(
        cls,
        package: Package,
        model_name: str = "6km_60min_natten_cos_zenith_input_eoe",
        conditioning_data_source: DataSource | ForecastSource = GFS_FX(),
        image_size: tuple[int, int] = (1024, 1792),
        spatial_downsample: int = 2,
        *args: Any,
        **kwargs: Any,
    ) -> DiagnosticModel:
        """Load model from package. Must be implemented by subclasses."""
        
        with open(package.resolve("registry.json"), "r") as f:
            registry = json.load(f)
            pkg = registry[model_name]

        model_spec = []
        for m in pkg["checkpoints"]:
            model = Module.from_checkpoint(package.resolve(m["path"]))
            model_spec.append(
                {
                    "model": model_wrap(model),
                    "sigma_min": float(m["sigma_min"]),
                    "sigma_max": float(m["sigma_max"]),
                }
            )

        # Normalization constants
        means = torch.from_numpy(np.load(package.resolve("goes_means.npy")))[None, :, None, None]
        stds = torch.from_numpy(np.load(package.resolve("goes_stds.npy")))[None, :, None, None]
        conditioning_means = torch.from_numpy(np.expand_dims(np.load(package.resolve("era5_means.npy")), 0))[None, :, None, None]
        conditioning_stds = torch.from_numpy(np.expand_dims(np.load(package.resolve("era5_stds.npy")), 0))[None, :, None, None]
        
        # Grid coordinates: crop a subregion from the HRRR grid
        latitudes = torch.from_numpy(np.load(package.resolve("lat.npy")))
        longitudes = (torch.from_numpy(np.load(package.resolve("lon.npy"))) + 360.0) % 360.0 # TODO align on [0, 360] range
        invalid = torch.from_numpy(np.load(package.resolve("invalid_gridpoints2.npy"))).to(dtype=torch.bool)
        hrrr_y, hrrr_x = HRRR.HRRR_Y, HRRR.HRRR_X
        full_y, full_x = latitudes.shape[0], longitudes.shape[1]
        anchor_y = int((full_y - image_size[0]) / 2)
        anchor_x = int((full_x - image_size[1]) / 2)        
        latitudes = latitudes[anchor_y:anchor_y+image_size[0], anchor_x:anchor_x+image_size[1]]
        longitudes = longitudes[anchor_y:anchor_y+image_size[0], anchor_x:anchor_x+image_size[1]]
        invalid = invalid[anchor_y:anchor_y+image_size[0], anchor_x:anchor_x+image_size[1]]
        y = hrrr_y[anchor_y:anchor_y+image_size[0]]
        x = hrrr_x[anchor_x:anchor_x+image_size[1]]

        # Spatial downsample
        y = y[::spatial_downsample]
        x = x[::spatial_downsample]
        latitudes = latitudes[::spatial_downsample, ::spatial_downsample]
        longitudes = longitudes[::spatial_downsample, ::spatial_downsample]
        invalid = invalid[::spatial_downsample, ::spatial_downsample]

        srcpath = "/lustre/fsw/portfolios/coreai/users/pharrington/dev/scv2/stormcast-v2/"
        demo_invalid = np.load(srcpath + "inp_.npy")[0, 0] == 0.
        np.save("invalid_gridpoints2.npy", demo_invalid.astype(int))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,10))
        plt.subplot(1, 2, 1)
        plt.imshow(demo_invalid, cmap="viridis")
        plt.colorbar(orientation="horizontal")
        plt.title("DEMO INVALID")
        plt.subplot(1, 2, 2)
        plt.imshow(invalid.numpy().astype(int), cmap="viridis")
        plt.colorbar(orientation="horizontal")
        plt.title("LOADED INVALID")
        plt.savefig("xxx_demo_invalid.png", dpi=300)
        plt.close()


        return cls(
            model_spec=model_spec,
            means=means,
            stds=stds,
            latitudes=latitudes,
            longitudes=longitudes,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            conditioning_data_source=conditioning_data_source,
            y_coords=y,
            x_coords=x,
            invalid_mask=invalid,
        )
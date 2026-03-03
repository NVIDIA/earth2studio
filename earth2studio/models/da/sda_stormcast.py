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

import warnings
from collections import OrderedDict
from itertools import product

import numpy as np
import torch
import xarray as xr
import zarr

from earth2studio.data import GFS_FX, HRRR, DataSource, ForecastSource, fetch_data
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
    handshake_size,
)
from earth2studio.utils.coords import map_coords
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from omegaconf import OmegaConf
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.preconditioners.legacy import EDMPrecond
    from physicsnemo.diffusion.samplers.legacy_deterministic_sampler import (
        deterministic_sampler,
    )
    from physicsnemo.models.diffusion_unets import StormCastUNet
except ImportError:
    OptionalDependencyFailure("stormcast")
    StormCastUNet = None
    EDMPreconditioner = None
    OmegaConf = None
    deterministic_sampler = None


# Variables used in StormCastV1 paper
VARIABLES = (
    ["u10m", "v10m", "t2m", "msl"]
    + [
        var + str(level)
        for var, level in product(
            ["u", "v", "t", "q", "Z", "p"],
            map(
                lambda x: str(x) + "hl",
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 20, 25, 30],
            ),
        )
        if not ((var == "p") and (int(level.replace("hl", "")) > 20))
    ]
    + [
        "refc",
    ]
)

CONDITIONING_VARIABLES = ["u10m", "v10m", "t2m", "tcwv", "sp", "msl"] + [
    var + str(level)
    for var, level in product(["u", "v", "z", "t", "q"], [1000, 850, 500, 250])
]

INVARIANTS = ["lsm", "orography"]


@check_optional_dependencies()
class StormCast(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """StormCast generative convection-allowing model for regional forecasts consists of
    two core models: a regression and diffusion model. Model time step size is 1 hour,
    taking as input:

    - High-resolution (3km) HRRR state over the central United States (99 vars)
    - High-resolution land-sea mask and orography invariants
    - Coarse resolution (25km) global state (26 vars)

    The high-resolution grid is the HRRR Lambert conformal projection
    Coarse-resolution inputs are regridded to the HRRR grid internally.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2408.10958
    - https://huggingface.co/nvidia/stormcast-v1-era5-hrrr

    Parameters
    ----------
    regression_model : torch.nn.Module
        Deterministic model used to make an initial prediction
    diffusion_model : torch.nn.Module
        Generative model correcting the deterministic prediciton
    means : torch.Tensor
        Mean value of each input high-resolution variable
    stds : torch.Tensor
        Standard deviation of each input high-resolution variable
    invariants : torch.Tensor
        Static invariant  quantities
    hrrr_lat_lim : tuple[int, int], optional
        HRRR grid latitude limits, defaults to be the StormCastV1 region in central
        United States, by default (273, 785)
    hrrr_lon_lim : tuple[int, int], optional
        HRRR grid longitude limits, defaults to be the StormCastV1 region in central
        United States,, by default (579, 1219)
    variables : np.array, optional
        High-resolution variables, by default np.array(VARIABLES)
    conditioning_means : torch.Tensor | None, optional
        Means to normalize conditioning data, by default None
    conditioning_stds : torch.Tensor | None, optional
        Standard deviations to normalize conditioning data, by default None
    conditioning_variables : np.array, optional
        Global variables for conditioning, by default np.array(CONDITIONING_VARIABLES)
    conditioning_data_source : DataSource | ForecastSource | None, optional
        Data Source to use for global conditioning. Required for running in iterator mode, by default None
    sampler_args : dict[str, float  |  int], optional
        Arguments to pass to the diffusion sampler, by default {}
    """

    def __init__(
        self,
        regression_model: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        means: torch.Tensor,
        stds: torch.Tensor,
        invariants: torch.Tensor,
        hrrr_lat_lim: tuple[int, int] = (273, 785),
        hrrr_lon_lim: tuple[int, int] = (579, 1219),
        variables: np.array = np.array(VARIABLES),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_variables: np.array = np.array(CONDITIONING_VARIABLES),
        conditioning_data_source: DataSource | ForecastSource | None = None,
        sampler_args: dict[str, float | int] = {},
    ):
        super().__init__()
        self.regression_model = regression_model
        self.diffusion_model = diffusion_model
        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("invariants", invariants)
        self.sampler_args = sampler_args

        hrrr_lat, hrrr_lon = HRRR.grid()
        self.lat = hrrr_lat[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]
        self.lon = hrrr_lon[
            hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]
        ]

        self.hrrr_x = HRRR.HRRR_X[hrrr_lon_lim[0] : hrrr_lon_lim[1]]
        self.hrrr_y = HRRR.HRRR_Y[hrrr_lat_lim[0] : hrrr_lat_lim[1]]

        self.variables = variables

        self.conditioning_variables = conditioning_variables
        self.conditioning_data_source = conditioning_data_source
        if conditioning_data_source is None:
            warnings.warn(
                "No conditioning data source was provided to StormCast, "
                + "set the conditioning_data_source attribute of the model "
                + "before running inference."
            )

        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)

        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system"""
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self.variables),
                "hrrr_y": self.hrrr_y,
                "hrrr_x": self.hrrr_x,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of diagnostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """

        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(1, "h")]),
                "variable": np.array(self.variables),
                "hrrr_y": self.hrrr_y,
                "hrrr_x": self.hrrr_x,
            }
        )

        target_input_coords = self.input_coords()

        handshake_dim(input_coords, "hrrr_x", 5)
        handshake_dim(input_coords, "hrrr_y", 4)
        handshake_dim(input_coords, "variable", 3)
        # Index coords are arbitrary as long its on the HRRR grid, so just check size
        handshake_size(input_coords, "hrrr_y", self.lat.shape[0])
        handshake_size(input_coords, "hrrr_x", self.lat.shape[1])
        handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            output_coords["lead_time"] + input_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "hf://nvidia/stormcast-v1-era5-hrrr@6c89a0877a0d6b231033d3b0d8b9828a6f833ed8",
            cache_options={
                "cache_storage": Package.default_cache("stormcast"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        conditioning_data_source: DataSource | ForecastSource = GFS_FX(verbose=False),
    ) -> DiagnosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        conditioning_data_source : DataSource | ForecastSource, optional
            Data source to use for global conditioning, by default GFS_FX

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        try:
            OmegaConf.register_new_resolver("eval", eval)
        except ValueError:
            # Likely already registered so skip
            pass

        # load model registry:
        config = OmegaConf.load(package.resolve("model.yaml"))

        # TODO: remove strict=False once checkpoints/imports updated to new diffusion API
        regression = StormCastUNet.from_checkpoint(
            package.resolve("StormCastUNet.0.0.mdlus"),
            strict=False,
        )
        diffusion = EDMPrecond.from_checkpoint(
            package.resolve("EDMPrecond.0.0.mdlus"),
            strict=False,
        )

        # Load metadata: means, stds, grid
        store = zarr.storage.ZipStore(package.resolve("metadata.zarr.zip"), mode="r")
        metadata = xr.open_zarr(store, zarr_format=2)

        variables = metadata["variable"].values
        conditioning_variables = metadata["conditioning_variable"].values

        # Expand dims and tensorify normalization buffers
        means = torch.from_numpy(metadata["means"].values[None, :, None, None])
        stds = torch.from_numpy(metadata["stds"].values[None, :, None, None])
        conditioning_means = torch.from_numpy(
            metadata["conditioning_means"].values[None, :, None, None]
        )
        conditioning_stds = torch.from_numpy(
            metadata["conditioning_stds"].values[None, :, None, None]
        )

        # Load invariants
        invariants = metadata["invariants"].sel(invariant=config.data.invariants).values
        invariants = torch.from_numpy(invariants).repeat(1, 1, 1, 1)

        # EDM sampler arguments
        if config.sampler_args is not None:
            sampler_args = config.sampler_args
        else:
            sampler_args = {}

        return cls(
            regression,
            diffusion,
            means,
            stds,
            invariants,
            variables=variables,
            conditioning_means=conditioning_means,
            conditioning_stds=conditioning_stds,
            conditioning_data_source=conditioning_data_source,
            conditioning_variables=conditioning_variables,
            sampler_args=sampler_args,
        )

    @torch.inference_mode()
    def _forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:

        # Scale data
        if "conditioning_means" in self._buffers:
            conditioning = conditioning - self.conditioning_means
        if "conditioning_stds" in self._buffers:
            conditioning = conditioning / self.conditioning_stds

        x = (x - self.means) / self.stds

        # Run regression model
        invariant_tensor = self.invariants.repeat(x.shape[0], 1, 1, 1)
        concats = torch.cat((x, conditioning, invariant_tensor), dim=1)

        out = self.regression_model(concats)

        # Concat for diffusion conditioning
        condition = torch.cat((x, out, invariant_tensor), dim=1)
        latents = torch.randn_like(x)
        latents = self.sampler_args["sigma_max"] * latents.to(dtype=torch.float64)

        # Could also do:
        # tN = torch.Tensor([self.sampler_args['sigma_max']]).to(x.device).repeat(x.shape[0])
        # latents = scheduler.init_latents(x.shape[1:], tN, device=x.device, dtype=torch.float64)

        from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        from physicsnemo.diffusion.samplers import sample

        class _CondtionalDiffusionWrapper(torch.nn.Module):
            def __init__(self, model: torch.nn.Module, img_lr: torch.Tensor):
                super().__init__()
                self.model = model
                self.img_lr = img_lr

            def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                return self.model(x, t, condition=self.img_lr)

        scheduler = EDMNoiseScheduler(
            sigma_min=self.sampler_args["sigma_min"],
            sigma_max=self.sampler_args["sigma_max"],
            rho=self.sampler_args["rho"],
        )
        denoiser = scheduler.get_denoiser(
            x0_predictor=_CondtionalDiffusionWrapper(self.diffusion_model, condition)
        )

        edm_out = sample(
            denoiser,
            latents.to(dtype=torch.float64),
            noise_scheduler=scheduler,
            num_steps=self.sampler_args["num_steps"],
            solver="edm_stochastic_heun",
            solver_options={
                "S_churn": self.sampler_args["S_churn"],
                "S_min": self.sampler_args["S_min"],
                "S_max": self.sampler_args["S_max"],
                "S_noise": self.sampler_args["S_noise"],
            },
        )

        # Run diffusion model
        # edm_out = deterministic_sampler(
        #     self.diffusion_model,
        #     latents=latents,
        #     img_lr=condition,
        #     **self.sampler_args
        # )
        out += edm_out

        out = out * self.stds + self.means

        return out

    @torch.inference_mode()
    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system

        Raises
        ------
        RuntimeError
            If conditioning data source is not initialized
        """

        if self.conditioning_data_source is None:
            raise RuntimeError(
                "StormCast has been called without initializing the model's conditioning_data_source"
            )

        # TODO: Eventually pull out interpolation into model and remove it from fetch
        # data potentially
        conditioning, conditioning_coords = fetch_data(
            self.conditioning_data_source,
            time=coords["time"],
            variable=self.conditioning_variables,
            lead_time=coords["lead_time"],
            device=x.device,
            interp_to=coords | {"_lat": self.lat, "_lon": self.lon},
            interp_method="linear",
        )
        # ensure data dimensions in the expected order
        conditioning_coords_ordered = OrderedDict(
            {
                k: conditioning_coords[k]
                for k in ["time", "lead_time", "variable", "lat", "lon"]
            }
        )
        conditioning, conditioning_coords = map_coords(
            conditioning, conditioning_coords, conditioning_coords_ordered
        )

        # Add a batch dim
        conditioning = conditioning.repeat(x.shape[0], 1, 1, 1, 1, 1)
        conditioning_coords.update({"batch": np.empty(0)})
        conditioning_coords.move_to_end("batch", last=False)

        # Handshake conditioning coords
        # TODO: ugh the interp... have to deal with this for now, no solution
        # handshake_coords(conditioning_coords, coords, "hrrr_x")
        # handshake_coords(conditioning_coords, coords, "hrrr_y")
        handshake_coords(conditioning_coords, coords, "lead_time")
        handshake_coords(conditioning_coords, coords, "time")

        output_coords = self.output_coords(coords)

        for i, _ in enumerate(coords["batch"]):
            for j, _ in enumerate(coords["time"]):
                for k, _ in enumerate(coords["lead_time"]):
                    x[i, j, k : k + 1] = self._forward(
                        x[i, j, k : k + 1], conditioning[i, j, k : k + 1]
                    )

        return x, output_coords


if __name__ == "__main__":

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    package = StormCast.load_default_package()
    model = StormCast.load_model(package)
    model = model.to("cuda")

    data = HRRR(verbose=False)
    x, coords = fetch_data(
        data,
        np.array(["2024-01-01"], dtype=np.datetime64),
        model.input_coords()["variable"],
        device="cuda",
    )
    del coords["lat"]
    del coords["lon"]

    x, coords = map_coords(x, coords, model.input_coords())

    out, out_coords = model(x, coords)

    # Load stormcast_original.pt
    torch.save(out, "stormcast.pt")
    original = torch.load("stormcast_original.pt", map_location=out.device)

    # Assume the dimensionality/order is the same as out
    diff = out - original

    print("Difference between out and stormcast_original.pt:")
    print("Max absolute difference:", diff.abs().max().item())
    print("Mean absolute difference:", diff.abs().mean().item())
    print("Shape of diff:", diff.shape)

    import matplotlib.pyplot as plt

    # Plot the first variable, first batch, first lead_time, first time
    # Infer axes: usually channels, y, x
    # out shape: (batch, time, lead_time, variable, y, x)
    var_axis = 3
    y_axis = 4
    x_axis = 5

    plt.figure(figsize=(8, 6))
    img = out[0, 0, 0].cpu().numpy()  # Shape: (y, x)
    plt.imshow(img, cmap="viridis", aspect="auto", vmin=-10, vmax=12.5)
    plt.title(f"Forecast: variable idx 0 (shape {img.shape})")
    plt.colorbar(label="Value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("stormcast.jpg")

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

import zipfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.nn.climatenet_conv import CGNetModule
from earth2studio.utils import (
    handshake_coords,
    handshake_dim,
)
from earth2studio.utils.type import CoordSystem

import yaml
from climate_learn.models.hub import Res_Slim_ViT
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from climate_learn.utils.fused_attn import FusedAttn

VARIABLES = [
    "t2",
    #"t2m",
    "t200",
    "t500",
    "t850",
    #"u10",
    "u10m",
    "u200",
    "u500",
    "u850",
    #"v10",
    "v10m",
    "v200",
    "v500",
    "v850",
    "q200",
    "q500",
    "q850",
    "swvl1",
#    "lsm",
]

ORBIT_VARIABLE_MAPPING = [
    "2m_temperature",
    "temperature_200",
    "temperature_500",
    "temperature_850",
    "10m_u_component_of_wind",
    "u_component_of_wind_200",
    "u_component_of_wind_500",
    "u_component_of_wind_850",
    "10m_v_component_of_wind",
    "v_component_of_wind_200",
    "v_component_of_wind_500",
    "v_component_of_wind_850",
    "specific_humidity_200",
    "specific_humidity_500",
    "specific_humidity_850",
    "volumetric_soil_water_layer_1",
    "total_precipitation_24hr",
    "2m_temperature_max",
    "2m_temperature_min",
]

STATIC_VARIABLES = [
    "land_sea_mask",
    "orography",
    "lattitude",
    "landcover",
]

OUT_VARIABLES = [
    "total_precipitation_24hr",
]

IN_HEIGHT = 720
IN_WIDTH = 1440


class OrbitGlobalPrecip9_5M(torch.nn.Module, AutoModelMixin):
    """Climate Net diagnostic model, built into Earth2Studio. This model can be used to
    create prediction labels for tropical cyclones and atmospheric rivers from a set of
    three atmospheric variables on a quater degree resolution equirectangular grid. It
    produces three non-standard output channels climnet_bg, climnet_tc and climnet_ar
    representing background label, tropical cyclone and atmospheric river labels.

    Note
    ----
    This model and checkpoint are from Prabhat et al. 2021. For more information see the
    following references:

    - https://doi.org/10.5194/gmd-14-107-2021
    - https://github.com/andregraubner/ClimateNet

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model
    center : torch.Tensor
        Model center normalization tensor of size [20,1,1]
    scale : torch.Tensor
        Model scale normalization tensor of size [20,1,1]
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        land_sea_mask,
        orography,
        lattitude,
        landcover,
        #center: torch.Tensor,
        #scale: torch.Tensor,
    ):
        super().__init__()

        self.model = core_model

        self.in_variables = ORBIT_VARIABLE_MAPPING + STATIC_VARIABLES
        self.out_variables = OUT_VARIABLES
        #self.out_variables = []

        self.land_sea_mask = land_sea_mask
        self.orography = orography
        self.lattitude = lattitude
        self.landcover = landcover

        #input_variables = VARIABLES
        #input_height = 720
        #input_width = 1440
        #self.register_buffer("input_variables", input_variables)
        #self.register_buffer("input_height", input_height)
        #self.register_buffer("input_width", input_width)

        #self.core_model = core_model
        #self.register_buffer("center", center)
        #self.register_buffer("scale", scale)

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of diagnostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(-90, 90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
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
                "variable": np.array(OUT_VARIABLES),
                "lat": np.linspace(-90, 90, 2880, endpoint=False), #720*4
                "lon": np.linspace(0, 360, 5760, endpoint=False), #1440*4
            }
        )
        # Validate input coordinates
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)
        #handshake_coords(input_coords, target_input_coords, "lon")
        #handshake_coords(input_coords, target_input_coords, "lat")
        #handshake_coords(input_coords, target_input_coords, "variable")

        output_coords["batch"] = input_coords["batch"]
        output_coords["variable"] = np.array(OUT_VARIABLES)
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "hf://jychoi-hpc/ORBIT-2",
            cache_options={
                "cache_storage": Package.default_cache("orbit-2"),
                "same_names": True,
            },
        )
        return package

    #@classmethod
    #def load_default_package(cls) -> Package:
    #    """Default pre-trained climatenet model package from Nvidia model registry"""
    #    return Package(
    #        "ngc://models/nvidia/modulus/modulus_diagnostics@v0.1",
    #        cache_options={
    #            "cache_storage": Package.default_cache("climatenet"),
    #            "same_names": True,
    #        },
    #    )

    #def load_model(cls, package: Package, model_folder: str, model_size: str, model_variable: str, in_channels: list, in_shape: tuple) -> DiagnosticModel:
    @classmethod
    def load_model(cls, package: Package, model_type: str, model_size: str, model_variable: str) -> DiagnosticModel:
        """Load diagnostic from package"""
        # Load YAML configuration
        with open(package.resolve(model_type+"-finetune"+"/"+model_type+"_"+model_size+"_"+model_variable+".yaml"), "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            
        #tensor_par_size = conf["parallelism"]["tensor_par"]

        #try:
        #    do_tiling = conf["tiling"]["do_tiling"]
        #    if do_tiling:
        #        div = conf["tiling"]["div"]
        #        overlap = conf["tiling"]["overlap"]
        #    else:
        #        div = 1
        #        overlap = 0
        #except Exception:
        #    do_tiling = False
        #    div = 1
        #    overlap = 0

        #var_weights = conf["data"]["var_weights"]

        default_vars = conf["data"]["default_vars"]
        spatial_resolution = conf["data"]["spatial_resolution"]

        superres_mag = conf["model"]["superres_mag"]
        cnn_ratio = conf["model"]["cnn_ratio"]
        patch_size = conf["model"]["patch_size"]
        embed_dim = conf["model"]["embed_dim"]
        depth = conf["model"]["depth"]
        decoder_depth = conf["model"]["decoder_depth"]
        num_heads = conf["model"]["num_heads"]
        mlp_ratio = conf["model"]["mlp_ratio"]
        drop_path = conf["model"]["drop_path"]
        drop_rate = conf["model"]["drop_rate"]

        #data_par_size = fsdp_size * simple_ddp_size

        #in_channels, in_height, in_width = in_shape[1:]
        #in_channels = len(VARIABLES + STATIC_VARIABLES)
        in_channels = len(ORBIT_VARIABLE_MAPPING + STATIC_VARIABLES)
        in_height = IN_HEIGHT
        in_width = IN_WIDTH
        if model_type in ["global"]:
            out_channels = 1

        #Check in_variables and out_variables vs dict_in_variables and dict_out_variables and map to finetune model variable names



        model = Res_Slim_ViT(
            default_vars,
            (in_height, in_width),
            in_channels,
            out_channels,
            superres_mag = superres_mag,
            history=1,
            patch_size= patch_size,
            cnn_ratio = cnn_ratio,
            learn_pos_emb=True,
            embed_dim=embed_dim,
            depth=depth,
            decoder_depth=decoder_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            drop_rate=drop_rate,
            #tensor_par_size = tensor_par_size,
            #tensor_par_group = tensor_par_group,
            FusedAttn_option = FusedAttn.DEFAULT, 
        )

        #NEED THIS?
        #model = model.to(device)
        map_location = "cpu"
        checkpoint = torch.load(package.resolve(model_type+"-finetune"+"/"+model_type+"_"+model_size+"_"+model_variable+".ckpt"), map_location=map_location)

        pretrain_model = checkpoint["model_state_dict"]
        del checkpoint

        state_dict = model.state_dict()

        for k in list(
            pretrain_model.keys()
        ):  # in pre-train model weights, but not fine-tuning model
            if k not in state_dict.keys():
                print(f"Removing key {k} from pretrained checkpoint: no exist")
                del pretrain_model[k]
            elif (
                pretrain_model[k].shape != state_dict[k].shape
            ):  # if pre-train and fine-tune model weights dimension doesn't match
                if k == "pos_embed":
                    print("interpolate positional embedding", flush=True)
                    interpolate_pos_embed(model, pretrain_model, new_size=model.img_size)
                else:
                    print(
                        f"Removing key {k} from pretrained checkpoint: no matching shape",
                        pretrain_model[k].shape,
                        state_dict[k].shape,
                    )
                    del pretrain_model[k]

        #model.load_state_dict(checkpoint["model_state_dict"])
        model.load_state_dict(pretrain_model)

        #land_sea_mask = np.load(package.resolve("Static_variables/land_sea_mask_0.25deg.npy"))
        #land_sea_mask = np.load("/ocean/projects/ees250003p/ilyngaas/E2S_Static/land_sea_mask_0.25deg.npy")
        land_sea_mask = np.load("/lustre/orion/stf006/proj-shared/irl1/E2S_Static/land_sea_mask_0.25deg.npy")
        #orography = np.load(package.resolve("Static_variables/orography_0.25deg.npy"))
        #orography = np.load("/ocean/projects/ees250003p/ilyngaas/E2S_Static/orography_0.25deg.npy")
        orography = np.load("/lustre/orion/stf006/proj-shared/irl1/E2S_Static/orography_0.25deg.npy")
        #lattitude = np.load(package.resolve("Static_variables/lattitude_0.25deg.npy"))
        #lattitude = np.load("/ocean/projects/ees250003p/ilyngaas/E2S_Static/lattitude_0.25deg.npy")
        lattitude = np.load("/lustre/orion/stf006/proj-shared/irl1/E2S_Static/lattitude_0.25deg.npy")
        #landcover = np.load(package.resolve("Static_variables/landcover_0.25deg.npy"))
        #landcover = np.load("/ocean/projects/ees250003p/ilyngaas/E2S_Static/landcover_0.25deg.npy")
        landcover = np.load("/lustre/orion/stf006/proj-shared/irl1/E2S_Static/landcover_0.25deg.npy")

        return cls(model, land_sea_mask, orography, lattitude, landcover)


    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:

        #Remove 90 degree latitude from data
        x = x[:,:,1:,:].to(torch.float32)

        #Flip latitude (89.75, -90) -> (-90, 89.75)
        x = torch.flip(x, dims=(2,))

        #Add static Variables to input tensor
        land_sea_mask = torch.from_numpy(self.land_sea_mask).to(x.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        orography = torch.from_numpy(self.orography).to(x.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        lattitude = torch.from_numpy(self.lattitude).to(x.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        landcover = torch.from_numpy(self.landcover).to(x.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        x = torch.cat((x, land_sea_mask, orography, lattitude, landcover),dim=1)

        return x

    @staticmethod
    def clip_replace_constant(y, out_variables):

        prcp_index = out_variables.index("total_precipitation_24hr")
        for i in range(y.shape[1]):
            if i == prcp_index:
                torch.clamp_(y[:, prcp_index, :, :], min=0.0)

        return y


    @torch.inference_mode()
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """

        x = self.preprocess_input(x)
        yhat = self.model.forward(x, self.in_variables, self.out_variables)
        yhat = self.clip_replace_constant(yhat, self.out_variables)
        return yhat

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""

        output_coords = self.output_coords(coords)

        out = torch.zeros(
            [len(v) for v in output_coords.values()],
            device=x.device,
            dtype=torch.float32,
        )

        #for i in range(out.shape[0]):
        #    out[i] = self._forward(x[i])
        out = self._forward(x)

        return out, output_coords

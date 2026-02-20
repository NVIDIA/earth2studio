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
from climate_learn.utils.fused_attn import FusedAttn
from climate_learn.data.precipmodule import LogTransform
from climate_learn.data.processing.era5_constants import PRECIP_VARIABLES
from climate_learn.utils.visualize import TileProcessor, TileCoordinates
from torchvision.transforms import transforms
import xesmf as xe
import xarray as xr
from typing import Tuple, List, Optional, Dict, Any

VARIABLES = [
    "t2m",
    "t200",
    "t500",
    "t850",
    "u10m",
    "u200",
    "u500",
    "u850",
    "v10m",
    "v200",
    "v500",
    "v850",
    "q200",
    "q500",
    "q850",
    "swvl1",
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
OUT_HEIGHT = 2880
OUT_WIDTH = 5760


class OrbitGlobalPrecip9_5M(torch.nn.Module, AutoModelMixin):
    """ORBIT-2 precipitation downscaling model, built into Earth2Studio.

    Note
    ----
    This model and checkpoint are from Wang et al. 2025. For more information see the
    following references:

    - https://dl.acm.org/doi/10.1145/3712285.3771989
    - https://dl.acm.org/doi/10.1145/3712285.3771989

    Parameters
    ----------
    core_model : torch.nn.Module
        Core pytorch model
    land_sea_mask : np.array
        Static Variable Used in Model
    orography : np.array
        Static Variable Used in Model
    lattitude : np.array
        Static Variable Used in Model
    landcover : np.array
        Static Variable Used in Model
    normalize_mean_lowres : Float
        Mean for data normalization
    normalize_std_lowres : Float
        Standard Deviation for data normalization
    normalize_mean_highres : Float
        Mean value for data normalization
    normalize_std_highres : Float
        Standard Deviation for data normalization
    do_tiling : Bool
        Boolean to indicate whether tiled inference is performed
    div : Int
        If performing tiling, number of tiles to divide input into
    overlap: Int
        If performing tiling, number of overlap pixels to use during tiled inference
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        land_sea_mask,
        orography,
        lattitude,
        landcover,
        normalize_mean_lowres,
        normalize_std_lowres,
        normalize_mean_highres,
        normalize_std_highres,
        do_tiling,
        div,
        overlap
    ):
        super().__init__()

        self.model = core_model

        self.in_variables = ORBIT_VARIABLE_MAPPING + STATIC_VARIABLES
        self.out_variables = OUT_VARIABLES

        self.land_sea_mask = land_sea_mask
        self.orography = orography
        self.lattitude = lattitude
        self.landcover = landcover
        self.normalize_mean_lowres = normalize_mean_lowres
        self.normalize_std_lowres = normalize_std_lowres
        self.normalize_mean_highres = normalize_mean_highres
        self.normalize_std_highres = normalize_std_highres
        self.do_tiling = do_tiling
        self.div = div
        self.overlap = overlap

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
                "lat": np.linspace(90, -90, 721),
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
                "lat": np.linspace(-90, 90, OUT_HEIGHT, endpoint=False), #720*4
                "lon": np.linspace(0, 360, OUT_WIDTH, endpoint=False), #1440*4
            }
        )
        # Validate input coordinates
        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lon", 3)
        handshake_dim(input_coords, "lat", 2)
        handshake_dim(input_coords, "variable", 1)

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

    @classmethod
    def load_model(cls, package: Package, model_type: str, model_size: str, model_variable: str) -> DiagnosticModel:
        """Load diagnostic from package"""
        # Load YAML configuration
        with open(package.resolve(model_type+"-finetune"+"/"+model_type+"_"+model_size+"_"+model_variable+".yaml"), "r") as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            
        try:
            do_tiling = conf["tiling"]["do_tiling"]
            if do_tiling:
                div = conf["tiling"]["div"]
                overlap = conf["tiling"]["overlap"]
            else:
                div = 1
                overlap = 0
        except Exception:
            do_tiling = False
            div = 1
            overlap = 0

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

        in_channels = len(ORBIT_VARIABLE_MAPPING + STATIC_VARIABLES)
        if do_tiling:
            assert overlap % 2 == 0, "Only handling even overlapping for now"
            top = bottom = overlap // 2
            left = right = overlap // 2 * 2
            in_height = int(IN_HEIGHT/div + top + bottom)
            in_width = int(IN_WIDTH/div + left + right)
        else:
            in_height = IN_HEIGHT
            in_width = IN_WIDTH

        out_channels = 1

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
            FusedAttn_option = FusedAttn.DEFAULT, 
        )
        model.data_config(
            spatial_resolution['ERA5_2'],
            (in_height, in_width),
            in_channels,
            out_channels,
        )

        map_location = "cpu"
        checkpoint = torch.load(package.resolve(model_type+"-finetune"+"/"+model_type+"_"+model_size+"_"+model_variable+".ckpt"), map_location=map_location)

        pretrain_model = checkpoint["model_state_dict"]
        del checkpoint

        state_dict = model.state_dict()

        model.load_state_dict(pretrain_model)

        land_sea_mask = np.load(package.resolve("static_variables/land_sea_mask_0.25deg.npy"))
        orography = np.load(package.resolve("static_variables/orography_0.25deg.npy"))
        lattitude = np.load(package.resolve("static_variables/lattitude_0.25deg.npy"))
        landcover = np.load(package.resolve("static_variables/landcover_0.25deg.npy"))

        normalize_mean_lowres = np.load(package.resolve("mean_std/era5/0.25_deg/normalize_mean.npz"))
        normalize_std_lowres = np.load(package.resolve("mean_std/era5/0.25_deg/normalize_std.npz"))

        normalize_mean_highres = np.load(package.resolve("mean_std/era5/0.25_deg/normalize_mean.npz"))
        normalize_std_highres = np.load(package.resolve("mean_std/era5/0.25_deg/normalize_std.npz"))

        return cls(model, land_sea_mask, orography, lattitude, landcover, normalize_mean_lowres, normalize_std_lowres, normalize_mean_highres, normalize_std_highres, do_tiling, div, overlap)

    def get_normalize_lowres(self):
        """Get Normalization Transformations for Low Resolution Data"""
        normalize_mean = dict(self.normalize_mean_lowres)
        normalize_std = dict(self.normalize_std_lowres)
        normed = OrderedDict()
        for var in self.in_variables:
            if var in PRECIP_VARIABLES:
                normed[var] = LogTransform(m2mm=True, LOG1P=True, thres_mm_per_day=0.25)
            else:
                normed[var] = transforms.Normalize(
                    normalize_mean[var][0], normalize_std[var][0]
                )
        return normed

    def get_normalize_highres(self):
        """Get Normalization Transformations for High Resolution Data"""
        normalize_mean = dict(self.normalize_mean_highres)
        normalize_std = dict(self.normalize_std_highres)
        normed = OrderedDict()
        for var in self.out_variables:
            if var in PRECIP_VARIABLES:
                normed[var] = LogTransform(m2mm=True, LOG1P=True, thres_mm_per_day=0.25)
            else:
                normed[var] = transforms.Normalize(
                    normalize_mean[var][0], normalize_std[var][0]
                )
        return normed

    def get_denormalize(self):
        """Get DeNormalization Transformations for High Resolution Data"""
        norm = self.get_normalize_highres()
        if isinstance(norm, dict):
            mean_norm = torch.tensor([norm[k].mean if k not in PRECIP_VARIABLES else 0. for k in norm.keys()])
            std_norm = torch.tensor([norm[k].std if k not in PRECIP_VARIABLES else 1. for k in norm.keys()])
        else:
            mean_norm = norm.mean
            std_norm = norm.std
        std_denorm = 1 / std_norm
        mean_denorm = -mean_norm * std_denorm
        denormed = transforms.Normalize(mean_denorm, std_denorm)
        return denormed


    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess Earth2Studio data to match ORBIT-2 DATA"""
        device = x.device

        #Regrid Data
        grid_out = xr.Dataset(
            {
                "lat": np.linspace(-90, 90, num=721, endpoint=True),
                "lon": np.arange(0, 360, 0.25),
            }
        )

        x = x.detach().cpu().numpy()
        x = xr.DataArray(
            x,
            dims=("time", "variables", "lat", "lon"),
            coords={
                "time": [1],
                "variables": ORBIT_VARIABLE_MAPPING,
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False)
            },
            name="dummy_data"
        )
        regridder = xe.Regridder(
            x, grid_out, "bilinear", periodic=True, reuse_weights=False
        )
        x = regridder(x, keep_attrs=True).astype("float32")

        #Remove 90 degree latitude from data to make (720, 1440)
        x = x[:,:,1:,:]
        x = torch.from_numpy(x.values).to(device)

        #Flip latitude (89.75, -90) -> (-90, 89.75)
        x = torch.flip(x, dims=(2,))

        #Add static Variables to input tensor
        land_sea_mask = torch.from_numpy(self.land_sea_mask).to(x.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        orography = torch.from_numpy(self.orography).to(x.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        lattitude = torch.from_numpy(self.lattitude).to(x.device).to(torch.float32).unsqueeze(0).unsqueeze(0)
        landcover = torch.from_numpy(self.landcover).to(x.device).to(torch.float32).unsqueeze(0).unsqueeze(0)

        x = torch.cat((x, land_sea_mask, orography, lattitude, landcover),dim=1)

        #Normalize Data
        norm_transforms = self.get_normalize_lowres()
        i = 0
        for k in norm_transforms.keys():
            x[:,i] = norm_transforms[k](x[:,i])
            i = i + 1

        return x

    @staticmethod
    def clip_replace_constant(y, out_variables):
        """Postprocess Precipitation Data to get rid of unphysical values"""

        prcp_index = out_variables.index("total_precipitation_24hr")
        for i in range(y.shape[1]):
            if i == prcp_index:
                torch.clamp_(y[:, prcp_index, :, :], min=0.0)

        return y

    @staticmethod
    def adjust_coords_for_flip(
        coords: TileCoordinates, processor: "TileProcessor"
    ) -> TileCoordinates:
        """Adjust coordinates after vertical image flip."""
        # Adjust input tile extraction coordinates after flip
        # Calculate flipped y-coordinates for tile extraction
        tile_height_with_overlap = processor.yinp // processor.div + (
            processor.top + processor.bottom
        )
        yi2tp = tile_height_with_overlap - coords.yi1t
        yi1tp = tile_height_with_overlap - coords.yi2t
        coords.yi1t = yi1tp
        coords.yi2t = yi2tp

        # Calculate flipped y-coordinates for result placement
        yi2rp = processor.yinp - coords.yi1r
        yi1rp = processor.yinp - coords.yi2r
        coords.yi1r = yi1rp
        coords.yi2r = yi2rp

        # Adjust output tile extraction coordinates after flip
        output_tile_height_with_overlap = (
            processor.yout // processor.div
            + (processor.top + processor.bottom) * processor.vmul
        )
        yo2tp = output_tile_height_with_overlap - coords.yo1t
        yo1tp = output_tile_height_with_overlap - coords.yo2t
        coords.yo1t = yo1tp
        coords.yo2t = yo2tp

        yo2rp = processor.yout - coords.yo1r
        yo1rp = processor.yout - coords.yo2r
        coords.yo1r = yo1rp
        coords.yo2r = yo2rp

        return coords

    @staticmethod
    def stitch_tiles(
        tiles: List, tile_coords: List, processor: TileProcessor 
    ) -> torch.Tensor:
        """Stitch tiles together into complete images.

        Reconstructs full image from processed tiles, handling overlap regions.
        """
        preds = torch.zeros((1,1,processor.yout, processor.xout), dtype=torch.float32, device=tiles[0].device)

        # Place each tile in the correct position
        for i in range(len(tiles)):
            coords = tile_coords[i]

            # Place prediction
            preds[:, :, coords.yo1r : coords.yo2r, coords.xo1r : coords.xo2r] = tiles[i][:, :, coords.yo1t : coords.yo2t, coords.xo1t : coords.xo2t]

        return preds



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
        if self.do_tiling:
            x = self.preprocess_input(x)
            processor = TileProcessor(self.div, self.overlap, (IN_HEIGHT, IN_WIDTH), (OUT_HEIGHT, OUT_WIDTH), 1)
            tiles = []
            tile_coords = []
            for vindex in range(self.div):  # Vertical tile index
                for hindex in range(self.div):  # Horizontal tile index
                    # Get tile coordinates with overlap handling
                    coords = processor.get_tile_coordinates(hindex, vindex)

                    # Extract tile data from full tensors
                    # x_tile: [batch, channels, height, width] for input
                    x_tile = x[:, :, coords.yi1 : coords.yi2, coords.xi1 : coords.xi2]
                    yhat = self.model.forward(x_tile, self.in_variables, self.out_variables)
                    yhat = self.clip_replace_constant(yhat, self.out_variables)
                    denorm_transforms = self.get_denormalize()
                    yhat[:,:] = denorm_transforms(yhat[:,:])
                    yhat = torch.flip(yhat, dims=(2,))
                    tile_coords.append(self.adjust_coords_for_flip(coords, processor))
                    tiles.append(yhat)
            yhat = self.stitch_tiles(tiles, tile_coords, processor)
            yhat = torch.flip(yhat, dims=(2,))
        else:
            x = self.preprocess_input(x)
            yhat = self.model.forward(x, self.in_variables, self.out_variables)
            yhat = self.clip_replace_constant(yhat, self.out_variables)
            #Flip Lattitude to have correct orientation
            yhat = torch.flip(yhat, dims=(2,))
            denorm_transforms = self.get_denormalize()
            yhat[:,:] = denorm_transforms(yhat[:,:])
        return yhat

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Forward pass of diagnostic"""

        output_coords = self.output_coords(coords)

        with torch.no_grad():
            out = self._forward(x)

        return out, output_coords


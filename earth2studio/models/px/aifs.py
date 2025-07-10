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
import json
import zipfile
from collections import OrderedDict
from collections.abc import Generator, Iterator

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import check_extra_imports
from earth2studio.utils.type import CoordSystem

VARIABLES = [
    "q50",
    "q100",
    "q150",
    "q200",
    "q250",
    "q300",
    "q400",
    "q500",
    "q600",
    "q700",
    "q850",
    "q925",
    "q1000",
    "t50",
    "t100",
    "t150",
    "t200",
    "t250",
    "t300",
    "t400",
    "t500",
    "t600",
    "t700",
    "t850",
    "t925",
    "t1000",
    "u50",
    "u100",
    "u150",
    "u200",
    "u250",
    "u300",
    "u400",
    "u500",
    "u600",
    "u700",
    "u850",
    "u925",
    "u1000",
    "v50",
    "v100",
    "v150",
    "v200",
    "v250",
    "v300",
    "v400",
    "v500",
    "v600",
    "v700",
    "v850",
    "v925",
    "v1000",
    "w50",
    "w100",
    "w150",
    "w200",
    "w250",
    "w300",
    "w400",
    "w500",
    "w600",
    "w700",
    "w850",
    "w925",
    "w1000",
    "z50",
    "z100",
    "z150",
    "z200",
    "z250",
    "z300",
    "z400",
    "z500",
    "z600",
    "z700",
    "z850",
    "z925",
    "z1000",
    "u10m",
    "v10m",
    "d2m",
    "t2m",
    "lsm",
    "msl",
    "sdor",
    "skt",
    "slor",
    "sp",
    "tcw",
    "zsl",
    "stl1",
    "stl2",
    "swvl1",
    "swvl2",
]


@check_extra_imports(
    "aifs",
    [
        "anemoi.inference",
        "anemoi.models",
        "earthkit.regrid",
        "ecmwf.opendata",
        "flash_attn",
    ],
)
class AIFS(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Artificial Intelligence Forecasting System (AIFS), a data driven forecast model
    developed by the European Centre for Medium-Range Weather Forecasts (ECMWF). AIFS is
    based on a graph neural network (GNN) encoder and decoder, and a sliding window
    transformer processor, and is trained on ECMWF's ERA5 re-analysis and ECMWF's
    operational numerical weather prediction (NWP) analyses.
    Consists of a single model with a time-step size of 6 hours.

    Note
    ----
    This model uses the checkpoints provided by ECMWF.
    For additional information see the following resources:

    - https://arxiv.org/abs/2406.01465
    - https://huggingface.co/ecmwf/aifs-single-1.0
    - https://github.com/ecmwf/anemoi-core

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    variables : np.array, optional
        Variables associated with model, by default 73 variable model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        interpolation_matrix: torch.Tensor,
        inverse_interpolation_matrix: torch.Tensor,
        variables: np.array = np.array(VARIABLES),
    ) -> None:
        super().__init__()
        self.model = model
        self.variables = variables
        self.register_buffer("latitudes", latitudes)
        self.register_buffer("longitudes", longitudes)
        self.register_buffer("interpolation_matrix", interpolation_matrix)
        self.register_buffer(
            "inverse_interpolation_matrix", inverse_interpolation_matrix
        )

        # Mask indices
        self.prognostic_output_mask = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            96,
            97,
            99,
            100,
        ]
        self.prognostic_input_mask = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            83,
            85,
            87,
            88,
            99,
            100,
            101,
            102,
        ]

    def __str__(self) -> str:
        return "aifs-single-1.0"

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
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(self.variables),
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model
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
                "lead_time": np.array([np.timedelta64(6, "h")]),
                "variable": np.array(self.variables),
                "lat": np.linspace(90.0, -90.0, 721),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        if input_coords is None:
            return output_coords

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
        """Load prognostic package"""
        package = Package(
            "hf://ecmwf/aifs-single-1.0",
            cache_options={
                "cache_storage": Package.default_cache("aifs-single-1.0"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_extra_imports(
        "aifs",
        [
            "anemoi.inference",
            "anemoi.models",
            "earthkit.regrid",
            "ecmwf.opendata",
            "flash_attn",
        ],
    )
    def load_model(cls, package: Package) -> PrognosticModel:
        """Load prognostic from package"""

        # Load model
        model_path = package.resolve("aifs-single-mse-1.0.ckpt")
        model = torch.load(
            model_path, weights_only=False, map_location=torch.ones(1).device
        )
        model.eval()

        # Define the path to the metadata file
        metadata_path = "inference-last/anemoi-metadata/ai-models.json"

        # Extract metadata and supporting arrays from the zip file
        with zipfile.ZipFile(model_path, "r") as zipf:  # NOTE: this is totally baffling
            # Load metadata
            metadata = json.load(zipf.open(metadata_path))

            # Load supporting arrays
            supporting_arrays = {}
            for key, entry in metadata.get("supporting_arrays_paths", {}).items():
                supporting_arrays[key] = np.frombuffer(
                    zipf.read(entry["path"]),
                    dtype=entry["dtype"],
                ).reshape(entry["shape"])

        # Load interpolation matrix
        # TODO: Maybe change this to allow for multiple packages?
        interpolation_package = Package(
            "https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear",
            cache_options={
                "cache_storage": Package.default_cache(
                    "aifs-single-1.0_interpolation_matrix"
                ),
                "same_names": True,
            },
        )
        interpolation_matrix_path = interpolation_package.resolve(
            "9533e90f8433424400ab53c7fafc87ba1a04453093311c0b5bd0b35fedc1fb83.npz"
        )
        interpolation_matrix = np.load(interpolation_matrix_path)
        torch_interpolation_matrix = torch.sparse_csr_tensor(
            crow_indices=torch.from_numpy(interpolation_matrix["indptr"]),
            col_indices=torch.from_numpy(interpolation_matrix["indices"]),
            values=torch.from_numpy(interpolation_matrix["data"]),
            size=(interpolation_matrix["shape"][0], interpolation_matrix["shape"][1]),
            dtype=torch.float64,
        )
        inverse_interpolation_package = Package(
            "https://get.ecmwf.int/repository/earthkit/regrid/db/1/mir_16_linear/",
            cache_options={
                "cache_storage": Package.default_cache(
                    "aifs-single-1.0_inverse_interpolation_matrix"
                ),
                "same_names": True,
            },
        )
        inverse_interpolation_matrix_path = inverse_interpolation_package.resolve(
            "7f0be51c7c1f522592c7639e0d3f95bcbff8a044292aa281c1e73b842736d9bf.npz"
        )
        inverse_interpolation_matrix = np.load(inverse_interpolation_matrix_path)
        torch_inverse_interpolation_matrix = torch.sparse_csr_tensor(
            crow_indices=torch.from_numpy(inverse_interpolation_matrix["indptr"]),
            col_indices=torch.from_numpy(inverse_interpolation_matrix["indices"]),
            values=torch.from_numpy(inverse_interpolation_matrix["data"]),
            size=(
                inverse_interpolation_matrix["shape"][0],
                inverse_interpolation_matrix["shape"][1],
            ),
            dtype=torch.float64,
        )

        return cls(
            model,
            latitudes=torch.Tensor(supporting_arrays["latitudes"]).reshape(1, 1, -1, 1),
            longitudes=torch.Tensor(supporting_arrays["longitudes"]).reshape(
                1, 1, -1, 1
            ),
            interpolation_matrix=torch_interpolation_matrix,
            inverse_interpolation_matrix=torch_inverse_interpolation_matrix,
            variables=VARIABLES,
        )

    def get_cos_sin_julian_day(
        self,
        time_array: np.datetime64,
        longitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cosine and sine of Julian day"""
        days = (
            time_array.astype("datetime64[D]") - time_array.astype("datetime64[Y]")
        ).astype(np.float32)
        hours = (
            time_array.astype("datetime64[h]") - time_array.astype("datetime64[D]")
        ).astype(np.float32)
        julian_days = days + (hours / 24.0)
        normalized = 2 * np.pi * (julian_days / 365.25)
        cos_julian_day = torch.full_like(
            longitudes, np.cos(normalized), dtype=torch.float32
        )
        sin_julian_day = torch.full_like(
            longitudes, np.sin(normalized), dtype=torch.float32
        )
        return cos_julian_day, sin_julian_day

    def get_cos_sin_local_time(
        self,
        time_array: np.datetime64,
        longitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cosine and sine of local time"""
        hours = (
            time_array.astype("datetime64[h]") - time_array.astype("datetime64[D]")
        ).astype(np.float32)
        normalized_time = 2 * np.pi * (hours / 24.0)
        normalized_longitudes = 2 * np.pi * (longitudes / 360.0)
        tau = normalized_time + normalized_longitudes
        cos_local_time = torch.cos(tau)
        sin_local_time = torch.sin(tau)
        return cos_local_time, sin_local_time

    def get_cosine_zenith_fields(
        self,
        date: np.datetime64,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
    ) -> torch.Tensor:
        """Get cosine zenith fields for input time array"""

        # Get Julian day
        days = (date.astype("datetime64[D]") - date.astype("datetime64[Y]")).astype(
            np.float32
        )
        hours = (date.astype("datetime64[h]") - date.astype("datetime64[D]")).astype(
            np.float32
        )
        seconds = (date.astype("datetime64[s]") - date.astype("datetime64[h]")).astype(
            np.float32
        )
        julian_day = days + seconds / 86400.0

        # Convert angle to tensor
        angle = torch.tensor(
            julian_day / 365.25 * torch.pi * 2, device=latitudes.device
        )

        # declination in [degrees]
        declination = (
            0.396372
            - 22.91327 * torch.cos(angle)
            + 4.025430 * torch.sin(angle)
            - 0.387205 * torch.cos(2 * angle)
            + 0.051967 * torch.sin(2 * angle)
            - 0.154527 * torch.cos(3 * angle)
            + 0.084798 * torch.sin(3 * angle)
        )

        # time correction in [h.degrees]
        time_correction = (
            0.004297
            + 0.107029 * torch.cos(angle)
            - 1.837877 * torch.sin(angle)
            - 0.837378 * torch.cos(2 * angle)
            - 2.340475 * torch.sin(2 * angle)
        )

        # Convert to radians
        declination = torch.deg2rad(declination)
        latitudes = torch.deg2rad(latitudes)

        # Calculate sine and cosine of declination and latitude
        sindec_sinlat = torch.sin(declination) * torch.sin(latitudes)
        cosdec_coslat = torch.cos(declination) * torch.cos(latitudes)

        # Solar hour angle
        solar_angle = torch.deg2rad((hours - 12) * 15 + longitudes + time_correction)
        zenith_angle = sindec_sinlat + cosdec_coslat * torch.cos(solar_angle)

        # Clip negative values
        return torch.clamp(zenith_angle, min=0.0)

    def _prepare_input(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepare input tensor and coordinates for the AIFS model."""

        # Interpolate the input tensor to the model grid
        shape = x.shape
        x = x.flatten(start_dim=4)
        x = x.flatten(end_dim=3)
        x = torch.swapaxes(x, 0, -1)
        x = x.to(dtype=torch.float64)
        x = self.interpolation_matrix @ x
        x = x.to(dtype=torch.float32)
        x = torch.swapaxes(x, 0, -1)
        x = x.reshape([shape[0] * shape[1], shape[2], shape[3], -1])
        x = torch.swapaxes(x, 2, 3)

        # Get cos, sin of latitude and longitude
        # (cos_latitude, sin_latitude, cos_longitude, sin_longitude)
        cos_latitude = torch.cos(torch.deg2rad(self.latitudes))
        sin_latitude = torch.sin(torch.deg2rad(self.latitudes))
        cos_longitude = torch.cos(torch.deg2rad(self.longitudes))
        sin_longitude = torch.sin(torch.deg2rad(self.longitudes))
        cos_latitude = torch.cat([cos_latitude, cos_latitude], dim=1)
        cos_longitude = torch.cat([cos_longitude, cos_longitude], dim=1)
        sin_latitude = torch.cat([sin_latitude, sin_latitude], dim=1)
        sin_longitude = torch.cat([sin_longitude, sin_longitude], dim=1)

        # Get cos, sin of Julian day
        cos_julian_day_0, sin_julian_day_0 = self.get_cos_sin_julian_day(
            coords["time"][0] - np.timedelta64(6, "h"), self.longitudes
        )
        cos_julian_day_1, sin_julian_day_1 = self.get_cos_sin_julian_day(
            coords["time"][0], self.longitudes
        )
        cos_julian_day = torch.cat([cos_julian_day_0, cos_julian_day_1], dim=1)
        sin_julian_day = torch.cat([sin_julian_day_0, sin_julian_day_1], dim=1)

        # Get cos, sin local time
        cos_local_time_0, sin_local_time_0 = self.get_cos_sin_local_time(
            coords["time"][0] - np.timedelta64(6, "h"), self.longitudes
        )
        cos_local_time_1, sin_local_time_1 = self.get_cos_sin_local_time(
            coords["time"][0], self.longitudes
        )
        cos_local_time = torch.cat([cos_local_time_0, cos_local_time_1], dim=1)
        sin_local_time = torch.cat([sin_local_time_0, sin_local_time_1], dim=1)

        # Get cosine zenith angle
        # Add insolation / cosine zenith angle
        cos_zenith_angle_0 = self.get_cosine_zenith_fields(
            coords["time"][0] - np.timedelta64(6, "h"), self.latitudes, self.longitudes
        )
        cos_zenith_angle_1 = self.get_cosine_zenith_fields(
            coords["time"][0], self.latitudes, self.longitudes
        )
        cos_zenith_angle = torch.cat([cos_zenith_angle_0, cos_zenith_angle_1], dim=1)

        # Combine inputs
        x = torch.cat(
            [
                x[:, :, :, :90],
                cos_latitude.repeat(shape[0] * shape[1], 1, 1, 1),
                cos_longitude.repeat(shape[0] * shape[1], 1, 1, 1),
                sin_latitude.repeat(shape[0] * shape[1], 1, 1, 1),
                sin_longitude.repeat(shape[0] * shape[1], 1, 1, 1),
                cos_julian_day.repeat(shape[0] * shape[1], 1, 1, 1),
                cos_local_time.repeat(shape[0] * shape[1], 1, 1, 1),
                sin_julian_day.repeat(shape[0] * shape[1], 1, 1, 1),
                sin_local_time.repeat(shape[0] * shape[1], 1, 1, 1),
                cos_zenith_angle.repeat(shape[0] * shape[1], 1, 1, 1),
                x[:, :, :, 90:],
            ],
            dim=3,
        )

        return x

    def _copy_output_to_input(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:
        """Copy output tensor to input tensor following the AIFS model."""

        # Copy prognostic field output to the input tensor
        x = x.roll(-1, dims=1)
        x[:, -1, :, self.prognostic_input_mask] = y[..., self.prognostic_output_mask]

        # Get cos, sin of Julian day
        cos_julian_day, sin_julian_day = self.get_cos_sin_julian_day(
            coords["time"][0] + coords["lead_time"][0], self.longitudes
        )

        # Get cos, sin of local time
        cos_local_time, sin_local_time = self.get_cos_sin_local_time(
            coords["time"][0] + coords["lead_time"][0], self.longitudes
        )

        # Get cosine zenith angle
        cos_zenith_angle = self.get_cosine_zenith_fields(
            coords["time"][0] + coords["lead_time"][0], self.latitudes, self.longitudes
        )

        # Add terms to x
        x[:, 1:2, :, 94:95] = cos_julian_day
        x[:, 1:2, :, 95:96] = cos_local_time
        x[:, 1:2, :, 96:97] = sin_julian_day
        x[:, 1:2, :, 97:98] = sin_local_time
        x[:, 1:2, :, 98:99] = cos_zenith_angle

        return x

    def _prepare_output(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepare input tensor and coordinates for the AIFS model."""

        # Interpolate the model grid to the lat lon grid
        shape = x.shape
        x = x[:, 1:2]
        x = x.flatten(end_dim=1)
        x = torch.swapaxes(x, 0, 1)
        x = x.flatten(start_dim=1)
        x = x.to(dtype=torch.float64)
        x = self.inverse_interpolation_matrix @ x
        x = x.to(dtype=torch.float32)
        x = torch.reshape(x, [x.shape[0], shape[0], shape[-1]])
        x = torch.swapaxes(x, 0, 1)
        x = torch.swapaxes(x, 1, 2)
        x = torch.reshape(
            x,
            [
                coords["batch"].shape[0],
                coords["time"].shape[0],
                coords["lead_time"].shape[0],
                coords["variable"].shape[0] + 9,
                coords["lat"].shape[0],
                coords["lon"].shape[0],
            ],
        )
        x = torch.cat([x[:, :, :, :90], x[:, :, :, 99:]], dim=3)
        return x

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)
        with torch.autocast(device_type=str(x.device), dtype=torch.float16):
            y = self.model.predict_step(x)
        return y, output_coords

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 6 hours in the future
        """
        _ = self.output_coords(coords)  # NOTE: Quick fix for exception handling
        x = self._prepare_input(x, coords)
        y, coords = self._forward(x, coords)
        x = self._copy_output_to_input(x, y, coords)
        x = self._prepare_output(x, coords)
        return x, coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:

        coords = coords.copy()
        self.output_coords(coords)
        coords_out = coords.copy()
        coords_out["lead_time"] = coords["lead_time"][1:]
        yield x[:, :, 1:], coords_out

        # Prepare input tensor
        x = self._prepare_input(x, coords)

        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward is identity operator
            y, coords_out = self._forward(x, coords)

            # Copy output to input
            x = self._copy_output_to_input(x, y, coords_out)

            # Prepare output tensor
            output_tensor = self._prepare_output(x, coords_out)

            # Rear hook
            output_tensor, coords_out = self.rear_hook(output_tensor, coords_out)

            # Yield output tensor
            yield output_tensor, coords_out

            # Update coordinates
            coords["lead_time"] += coords_out["lead_time"][0] - coords["lead_time"][-1]

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

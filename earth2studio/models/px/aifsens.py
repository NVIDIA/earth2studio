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
from earth2studio.models.px.aifs import AIFS
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    import anemoi.models  # noqa: F401
    import earthkit.regrid  # noqa: F401
    import ecmwf.opendata  # noqa: F401
    import flash_attn  # noqa: F401
except ImportError:
    OptionalDependencyFailure("aifs")

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
    "z",
    "cp06",
    "tp06",
    "cos_latitude",
    "cos_longitude",
    "sin_latitude",
    "sin_longitude",
    "cos_julian_day",
    "cos_local_time",
    "sin_julian_day",
    "sin_local_time",
    "insolation",
    "stl1",
    "stl2",
    "ssrd",
    "strd",
    "sf",
    "tcc",
    "mcc",
    "hcc",
    "lcc",
    "u100m",
    "v100m",
    "ro",
]  # from config.json >> dataset.variables


class AIFSENS(AIFS):

    def __str__(self) -> str:
        return "aifs-ens-1.0"

    @property
    def input_variables(self) -> list[str]:
        indices = torch.cat(
            [
                self.model.data_indices.data.input.prognostic,
                self.model.data_indices.data.input.forcing,
            ]
        )

        # Sort the concatenated tensor
        indices = indices.sort().values

        # Create the range of values to remove
        to_remove = torch.arange(92, 101)  # generated forcings

        # Keep only elements NOT in to_remove
        mask = ~torch.isin(indices, to_remove)

        selected = [VARIABLES[i] for i in indices[mask].tolist()]
        return selected

    @property
    def output_variables(self) -> list[str]:
        # Input constants + prognostic and diagnostic - generated forcings
        indices = torch.cat(
            [
                self.model.data_indices.data.input.forcing,
                self.model.data_indices.data.output.full,
            ]
        )

        # Sort the concatenated tensor
        indices = torch.unique(indices.sort().values)

        # Create the range of values to remove
        to_remove = torch.arange(92, 101)  # generated forcings

        # Keep only elements NOT in to_remove
        mask = ~torch.isin(indices, to_remove)

        selected = [VARIABLES[i] for i in indices[mask].tolist()]
        return selected

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "hf://ecmwf/aifs-ens-1.0",
            cache_options={
                "cache_storage": Package.default_cache("aifs-ens-1.0"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(cls, package: Package) -> PrognosticModel:
        """Load prognostic from package"""

        # Load model
        model_path = package.resolve("aifs-ens-crps-1.0.ckpt")
        model = torch.load(
            model_path, weights_only=False, map_location=torch.ones(1).device
        )
        model.eval()

        # Define the path to the metadata file
        metadata_path = "inference-anemoi-by_epoch-epoch_001-step_000040_tp_fix_0.05/anemoi-metadata/ai-models.json"

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
        )

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        output_coords = self.output_coords(coords)
        with torch.autocast(device_type=str(x.device), dtype=torch.float16):
            y = self.model.predict_step(x, fcstep=1)
            out = torch.empty(
                (x.shape[0], x.shape[1], x.shape[2], len(VARIABLES)),
                device=x.device,
            )
            out[..., 0, :, self.model.data_indices.data.input.full] = x[
                :,
                1,
            ]
            out[..., 1, :, self.model.data_indices.data.output.full] = y
            out[..., 1, :, self.model.data_indices.data.input.forcing] = x[
                :, 1, :, self.model.data_indices.model.input.forcing
            ]

        return out, output_coords

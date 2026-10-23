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

import hydra
from omegaconf import DictConfig
from src.diagnostic_distributed import PrecipDiagnostic, diagnostic_distributed

from earth2studio.data import GFS
from earth2studio.io import ZarrBackend
from earth2studio.models.px import FCN, PrognosticModel
from earth2studio.utils.distributed import DistributedInference, DistributedManager


def diagnostic(
    time: str = "2023-06-01T00:00:00",
    nsteps: int = 12,
    output_path: str = "diagnostic_distributed.zarr",
    model_cls: type[PrognosticModel] = FCN,
) -> None:
    """Distributed diagnostic model recipe."""
    dist = DistributedManager()
    if dist.world_size < 2:
        raise ValueError("This recipe requires at least 2 processes")

    if dist.rank == 0:  # rank 0 will run the prognostic model and handle IO
        model = model_cls.load_model(model_cls.load_default_package())

        # create diagnostic models on the other available ranks
        dist_diagnostic = DistributedInference(PrecipDiagnostic)

        # initialize data source and IO backend
        data = GFS()
        io = ZarrBackend(output_path)

        # run the inference
        diagnostic_distributed([time], nsteps, model, dist_diagnostic, data, io)


recipes = {
    "diagnostic": diagnostic,
}


@hydra.main(version_base=None, config_path="cfg")
def main(cfg: DictConfig) -> None:
    """Initialize DistributedInference, choose the recipe and run it."""
    DistributedInference.initialize()
    recipes[cfg.recipe](**cfg.parameters)
    DistributedInference.finalize()


if __name__ == "__main__":
    main()

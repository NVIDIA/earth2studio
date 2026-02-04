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

import hydra
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from src.hens_run import run_inference
from src.hens_utilities import initialise


@hydra.main(version_base=None, config_path="cfg", config_name="helene")
def main(cfg: DictConfig) -> None:
    """Main Hydra function with instantiation"""

    DistributedManager.initialize()

    (
        ensemble_configs,
        model_dict,
        dx_model_dict,
        cd_model_dict,
        cyclone_tracker,
        data_source,
        output_coords_dict,
        base_random_seed,
        writer_executor,
        writer_threads,
    ) = initialise(cfg)

    run_inference(
        cfg,
        ensemble_configs,
        model_dict,
        dx_model_dict,
        cd_model_dict,
        cyclone_tracker,
        data_source,
        output_coords_dict,
        base_random_seed,
        writer_executor,
        writer_threads,
    )


if __name__ == "__main__":
    main()

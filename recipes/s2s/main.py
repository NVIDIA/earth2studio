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
from src.s2s_run import run_inference
from src.s2s_utilities import configure_logging, initialize


@hydra.main(version_base=None, config_path="cfg", config_name="pnw_dlesym")
def main(cfg: DictConfig) -> None:
    """Main Hydra function with instantiation"""

    DistributedManager.initialize()

    # Configure logging
    configure_logging()

    (
        ensemble_configs,
        model_dict,
        dx_model_dict,
        data_source,
        output_coords_dict,
        base_random_seed,
    ) = initialize(cfg)

    run_inference(
        cfg,
        ensemble_configs,
        model_dict,
        dx_model_dict,
        data_source,
        output_coords_dict,
        base_random_seed,
    )


if __name__ == "__main__":
    main()

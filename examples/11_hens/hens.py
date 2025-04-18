# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from datetime import datetime

import hydra
import pandas as pd
from dotenv import load_dotenv
from ensemble_utilities import EnsembleBase
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from reproduce_utilities import create_base_seed_string
from utilities import (
    initialise,
    initialise_output,
    initialise_perturbation,
    store_tracks,
    update_model_dict,
    write_to_disk,
)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Parallelised workflow for running an ensemble inference of a forecast model
    using multiple checkpoints, following the approach of Mahesh et al.
    https://arxiv.org/abs/2408.03100

    Parameters
    ----------
    cfg : DictConfig
        config.
    """

    DistributedManager.initialize()
    load_dotenv()

    (
        ensemble_configs,
        model_dict,
        dx_model_dict,
        cyclone_tracking,
        data,
        output_coords_dict,
        base_random_seed,
        all_tracks_dict,
        writer_executor,
        writer_threads,
    ) = initialise(cfg)

    # initialize output objects
    # all_tracks_dict, writer_executor, writer_threads = initialize_output_structures(cfg)

    # run forecasts
    then = datetime.now()

    for pkg, ic, ens_idx, batch_ids_produce in ensemble_configs:
        # TODO: add start time as optional call, so it works for run.ensemble and run_hens
        #       without having to pass start_time in initialisation
        base_seed_string = create_base_seed_string(pkg, ic, base_random_seed)
        # load new weights if necessary
        model_dict = update_model_dict(model_dict, pkg)

        io_dict = initialise_output(cfg, ic, model_dict, output_coords_dict)

        perturbation = initialise_perturbation(
            model=model_dict["model"], data=data, start_time=ic, cfg=cfg
        )

        run_hens = EnsembleBase(
            time=[ic],
            nsteps=cfg.nsteps,
            nensemble=cfg.nensemble,
            prognostic=model_dict["model"],
            data=data,
            io_dict=io_dict,
            perturbation=perturbation,
            output_coords_dict=output_coords_dict,
            dx_model_dict=dx_model_dict,
            cyclone_tracking=cyclone_tracking,
            batch_size=cfg.batch_size,
            ensemble_idx_base=ens_idx,
            batch_ids_produce=batch_ids_produce,
            base_seed_string=base_seed_string,
        )
        df_tracks_dict, io_dict = run_hens()
        for k, v in df_tracks_dict.items():
            v["ic"] = pd.to_datetime(ic)
            all_tracks_dict[k].append(v)

        # if in-memory flavour of io backend was chosen, write content to disk now
        if io_dict:
            writer_threads, writer_executor = write_to_disk(
                cfg,
                ic,
                model_dict,
                io_dict,
                writer_threads,
                writer_executor,
            )

    # Output run duration
    now = datetime.now()
    logger.info(
        f"Took {(now-then).total_seconds()}s for {len(ensemble_configs)} ics and "
        + f"{cfg.nsteps} steps rollout with ensemble size {cfg.nensemble}."
    )

    # Output summaries of cyclone tracks if required
    if "cyclone_tracking" in cfg:
        for area_name, all_tracks in all_tracks_dict.items():
            store_tracks(area_name, all_tracks, cfg)

    if writer_executor is not None:
        for thread in list(writer_threads):
            thread.result()
            writer_threads.remove(thread)
        writer_executor.shutdown()


if __name__ == "__main__":
    main()

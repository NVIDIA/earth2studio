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

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from omegaconf import DictConfig

from earth2studio.data.base import DataSource
from earth2studio.models.dx.base import DiagnosticModel
from src.hens_ensemble import EnsembleBase
from src.hens_perturbation import HENSPerturbation
from src.hens_utilities import (
    initialise_output,
    update_model_dict,
    write_to_disk,
)
from src.hens_utilities_reproduce import create_base_seed_string


def run_inference(
    cfg: DictConfig,
    ensemble_configs: list[Any],
    model_dict: dict,
    dx_model_dict: dict,
    cd_model_dict: dict,
    cyclone_tracker: DiagnosticModel | None,
    data_source: DataSource,
    output_coords_dict: dict,
    base_random_seed: str | int,
    writer_executor: ThreadPoolExecutor | None,
    writer_threads: list[Future],
) -> None:
    """HENS run inference function"""
    # We will bring all components together to execute the ensemble forecasting process:
    #
    # - iterate through each ensemble configuration, which contains the necessary parameters
    # for generating individual ensemble members.
    # - at each iteration, update the model dictionary whenever a new package is
    # encountered, ensuring the correct model weights are loaded.
    # - initialise the output object and set up the perturbation method, taking into account
    # any changes in the initial conditions.
    # - initialise the perturbation method with updated IC and checkpoint
    # - initialise the inference pipeline with updated IC and checkpoint
    # - run inference, which generates all ensemble members according to the specified
    # configuration.
    # - write the results to disk, ensuring that all forecast data and associated metadata
    # are properly stored for subsequent analysis.

    for pkg, ic, ens_idx, batch_ids_produce in ensemble_configs:
        # create seed base string required for reproducibility of individual batches
        base_seed_string = create_base_seed_string(pkg, ic, base_random_seed)

        # load new weights if necessary
        model_dict = update_model_dict(model_dict, pkg)

        # create new io object
        io_dict = initialise_output(cfg, ic, model_dict, output_coords_dict)

        # HENS perturbation. For this, we need to provide:
        # - a skill file, which contains the deterministic skill of the forecast model
        # (**Note**: provide links to download skill file, best in intro)
        # - the variable to perturb in the seeding step of the bred vector perturbation
        # - the number of integration steps for breeding the noise vector
        # - the noise amplification, by which the noise vector is scaled
        #
        # With this information, we can now assemble the HENS perturbation using
        # CorrelatedSphericalGaussian as seeding perturbation and HemisphericCentredBredVector
        # as bred vector perturbation. To see how it is aseembled form basic blocks
        # provided in Earth2Studio

        perturbation = HENSPerturbation(
            model=model_dict["model"],
            data_source=data_source,
            skill_path=cfg.perturbation.skill_path,
            noise_amplification=cfg.perturbation.noise_amplification,
            perturbed_var=cfg.perturbation.perturbed_var,
            integration_steps=cfg.perturbation.integration_steps,
        )

        # initialise inference pipeline with updated IC and checkpoint
        ensemble_runner = EnsembleBase(
            time=[ic],
            nsteps=cfg.nsteps,
            nensemble=cfg.nensemble,
            prognostic=model_dict["model"],
            data=data_source,
            io_dict=io_dict,
            perturbation=perturbation,
            output_coords_dict=output_coords_dict,
            dx_model_dict=dx_model_dict,
            cd_model_dict=cd_model_dict,
            cyclone_tracking=cyclone_tracker,
            batch_size=cfg.batch_size,
            ensemble_idx_base=ens_idx,
            batch_ids_produce=batch_ids_produce,
            base_seed_string=base_seed_string,
            pkg=pkg,
        )

        # run inference
        io_dict = ensemble_runner()

        # if in-memory flavour of io backend was chosen, write content to disk now
        if io_dict:
            writer_executor, writer_threads = write_to_disk(
                cfg,
                ic,
                model_dict,
                io_dict,
                writer_executor,
                writer_threads,
            )

    if writer_executor is not None:
        for thread in list(writer_threads):
            thread.result()
            writer_threads.remove(thread)
        writer_executor.shutdown()

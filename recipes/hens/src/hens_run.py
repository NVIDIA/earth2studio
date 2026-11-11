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
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from earth2studio.data.base import DataSource
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.run import ProgressStore

from .hens_ensemble import EnsembleBase
from .hens_perturbation import HENSPerturbation
from .hens_utilities import (
    initialise_output,
    update_model_dict,
    write_to_disk,
)
from .hens_utilities_reproduce import create_base_seed_string


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
    """Run this worker's checkpoint/IC pairs and drain asynchronous writers."""
    try:
        resume = cfg.get("file_output", {}).get("resume", False)
        if resume:
            output = cfg.file_output
            if (
                output.format._target_ != "earth2studio.io.ZarrBackend"
                or not output.format.get("_partial_", False)
                or output.format.get("backend_kwargs", {}).get("overwrite", False)
            ):
                raise ValueError(
                    "HENS resume requires a partial ZarrBackend with overwrite=False"
                )
        for pkg, ic, ens_idx, batch_ids_produce in ensemble_configs:
            base_seed_string = create_base_seed_string(pkg, ic, str(base_random_seed))
            model_dict = update_model_dict(model_dict, pkg)
            perturbation = HENSPerturbation(
                model=model_dict["model"],
                data_source=data_source,
                skill_path=cfg.perturbation.skill_path,
                noise_amplification=cfg.perturbation.noise_amplification,
                perturbed_var=cfg.perturbation.perturbed_var,
                integration_steps=cfg.perturbation.integration_steps,
            )
            io_dict = initialise_output(cfg, ic, model_dict, output_coords_dict)
            try:
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
                    progress=(
                        ProgressStore(
                            Path(cfg.file_output.path)
                            / "progress"
                            / f"{cfg.project}_{Path(pkg).name}_{ic}",
                            tags={
                                "package": pkg,
                                "config": OmegaConf.to_container(cfg, resolve=True),
                            },
                        )
                        if resume
                        else None
                    ),
                )
                ensemble_runner()
            finally:
                # Also cover construction/planning errors before Pipeline takes over.
                for io in io_dict.values():
                    io.close()
            if io_dict:
                writer_executor, writer_threads = write_to_disk(
                    cfg, ic, model_dict, io_dict, writer_executor, writer_threads
                )
        for thread in list(writer_threads):
            thread.result()
            writer_threads.remove(thread)
    finally:
        if writer_executor is not None:
            writer_executor.shutdown(wait=True)

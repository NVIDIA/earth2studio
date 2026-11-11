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

from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from earth2studio.data.base import DataSource
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.run import Pipeline, ProgressStore, work_items

from .hens_distributed import local_device
from .hens_ensemble import HENSRunner
from .hens_perturbation import HENSPerturbation
from .hens_utilities import (
    create_base_seed_string,
    initialise_output,
    update_model_dict,
    write_to_disk,
)


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
) -> None:
    """Run this worker's checkpoint/IC pairs with backend-owned output buffering."""
    if torch.distributed.is_initialized():
        raise RuntimeError("Launch HENS without initializing a Torch process group")
    if cfg.nensemble < 1 or cfg.batch_size < 1 or cfg.nsteps < 0:
        raise ValueError(
            "HENS requires positive ensemble/batch sizes and nonnegative nsteps"
        )
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
        with ExitStack() as outputs:
            # Cover planning failures too; close every store even if one fails.
            for io in io_dict.values():
                outputs.callback(io.close)
            runner = HENSRunner(
                nensemble=cfg.nensemble,
                prognostic=model_dict["model"],
                data=data_source,
                perturbation=perturbation,
                output_coords_dict={name: output_coords_dict[name] for name in io_dict},
                dx_model_dict=dx_model_dict,
                cd_model_dict=cd_model_dict,
                cyclone_tracking=cyclone_tracker,
                batch_size=min(cfg.batch_size, cfg.nensemble),
                base_seed_string=base_seed_string,
                pkg=pkg,
            )
            # Form complete batches first: filtering members would change pairings.
            items = work_items(
                [ic],
                cfg.nsteps * runner.step,
                members=range(ens_idx, ens_idx + cfg.nensemble),
                member_batch=runner.batch_size,
            )
            first_batch = ens_idx // cfg.nensemble * len(items)
            selected = set(batch_ids_produce)
            if not selected.issubset(range(first_batch, first_batch + len(items))):
                raise ValueError("Batch IDs do not belong to this checkpoint")
            Pipeline(
                runner,
                output=io_dict,
                device=local_device(),
                progress=(
                    ProgressStore(
                        Path(cfg.file_output.get("path", "outputs"))
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
            ).run([item for i, item in enumerate(items, first_batch) if i in selected])
        if io_dict:
            write_to_disk(cfg, ic, model_dict, io_dict)

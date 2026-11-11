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

"""Eval recipe inference entry point.

Builds the campaign's runner and work items from the Hydra config and hands
them to :class:`earth2studio.run.Pipeline`, which distributes items over
ranks, writes every stream into ``forecast.zarr``, scores online when asked,
and records finished items for resume.

Single process::

    python main.py campaign=dlwp_2024_monthly

Multi-GPU::

    python -m torch.distributed.run --nproc_per_node=8 --standalone main.py \\
        campaign=fcn3_2024_monthly
"""

from __future__ import annotations

from collections.abc import Sequence

import hydra
import torch
from loguru import logger
from omegaconf import DictConfig
from src.campaign import build_items, members_per_item, online_scoring
from src.logs import configure_logging, finalize_distributed, init_distributed
from src.output import (
    build_backend,
    build_regridders,
    parallel_coords,
    prepare_store,
    progress_path,
    retain_raw_output,
    sentinel_path,
    store_path,
    stream_variables,
)
from src.runners import build_runner

from earth2studio.run import Pipeline, ProgressStore, WorkItem
from earth2studio.run.work import rank0_first, rank_info


def fetches_nothing(runner: object, items: Sequence[WorkItem]) -> bool:
    """Whether ``runner`` declares no gridded or observation fetch for any item.

    A runner whose ``data_requests`` returns ``None`` (unknowable) counts as
    fetching.
    """
    if getattr(runner, "frame_requests", None) is not None:
        return False
    data_requests = getattr(runner, "data_requests", None)
    if data_requests is None:
        return False
    for item in items:
        requests = data_requests(item)
        if requests is None or len(requests):
            return False
    return True


def check_predownload(
    cfg: DictConfig, runner: object, items: Sequence[WorkItem]
) -> None:
    """Refuse to start without the predownload sentinel, when required.

    Runners that fetch nothing for their items skip the check (see
    :func:`fetches_nothing`), as do campaigns whose initial conditions and
    verification data are both bring-your-own.
    """
    if not cfg.get("require_predownload", True):
        return
    if fetches_nothing(runner, items):
        logger.info(f"{type(runner).__name__} fetches no data; no pre-download needed.")
        return
    fully_byo = (
        cfg.get("ic_source") is not None and cfg.get("verification_source") is not None
    )
    if fully_byo:
        logger.info("Initial conditions and verification are bring-your-own.")
        return
    path = sentinel_path(cfg)
    if not path.exists():
        raise RuntimeError(
            f"Pre-download sentinel not found at '{path}'.\n"
            "Run 'python predownload.py' with the same config before inference, "
            "or set require_predownload=false to fetch live."
        )
    logger.info(f"Pre-download sentinel found: {path}")


def run(cfg: DictConfig, device: torch.device | str = "cpu") -> None:
    """Run the campaign described by ``cfg`` on this rank.

    Parameters
    ----------
    cfg : DictConfig
        Full recipe config.
    device : torch.device | str, optional
        Device to run inference on, by default CPU.
    """
    rank, world_size = rank_info()
    online = online_scoring(cfg)
    resume = bool(cfg.get("resume", False))

    runner = build_runner(cfg)
    items = build_items(cfg, runner, member_batch=members_per_item(cfg))
    check_predownload(cfg, runner, items)
    final_lead_time = max(item.final_lead_time for item in items)
    schemas = runner.output_coords(final_lead_time)
    regridders = build_regridders(cfg, schemas)
    variables = stream_variables(schemas, list(cfg.output.variables))

    # Online scoring reduces each forecast to summary statistics in the
    # inference loop; it may also reorder the items so the ranks of an
    # ensemble group run one initial condition in lockstep.
    online_run = None
    scorer = sync = None
    if online:
        from src.online import OnlineRun, parse_online_settings

        settings = parse_online_settings(cfg)
        online_run = OnlineRun.build(
            cfg, settings, runner, items, schemas, regridders, variables, device
        )
        items = online_run.items
        scorer, sync = online_run.scorer, online_run.sync

    output = None
    if retain_raw_output(cfg):
        lead_time = schemas[next(iter(variables))].coords["lead_time"].values
        parallel = parallel_coords(items, lead_time)
        overwrite = bool(cfg.output.get("overwrite", False))
        prepare_store(store_path(cfg), resume, overwrite, rank)
        backend = rank0_first(
            lambda: build_backend(
                cfg, store_path(cfg), parallel, overwrite and not resume, rank
            )
        )
        output = {stream: backend for stream in variables}
    elif not online:
        raise ValueError("output.retain=none is only legal with scoring.mode=online")

    progress = ProgressStore(
        progress_path(cfg),
        # Online and offline runs share the progress store; the tag makes a
        # resume that switches mode fail instead of skipping finished items,
        # which would leave the new statistics store empty.
        tags={
            "project": str(cfg.project),
            "run_id": str(cfg.run_id),
            "scoring": "online" if online else "offline",
        },
        overwrite=not resume,
    )
    logger.info(
        f"Running {len(items)} work items on {world_size} rank(s); streams "
        f"{sorted(variables)} -> {store_path(cfg) if output else 'no raw store'}"
    )
    Pipeline(
        runner,
        output=output,
        scorer=scorer,
        output_variables=variables,
        regridder=regridders,
        progress=progress,
        device=device,
        sync=sync,
    ).run(items)
    if online_run is not None:
        online_run.finalize()


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig) -> None:
    """Eval recipe entry point: distributed model inference."""
    configure_logging()
    timeout = None
    if online_scoring(cfg):
        timeout = cfg.scoring.online.get("nccl_timeout_s", None)
    device = init_distributed(timeout)
    try:
        run(cfg, device)
    finally:
        finalize_distributed()
    logger.success("Eval recipe finished.")


if __name__ == "__main__":
    main()

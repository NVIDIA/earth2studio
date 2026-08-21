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

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from src.data import resolve_ic_source
from src.distributed import configure_logging
from src.online import (
    build_group_comm,
    build_online_scorer,
    build_statistics,
    check_verification_coverage,
    finalize_stats,
    online_enabled,
    open_stats_store,
    parse_online_settings,
    retain_raw_output,
)
from src.output import OutputManager, sentinel_path
from src.pipelines import build_pipeline
from src.pipelines.base import Pipeline
from src.work import (
    build_group_work_items,
    build_work_items,
    clear_online_progress,
    clear_progress,
    distribute_work,
    filter_completed_items,
    filter_online_completed,
)


@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig) -> None:
    """Eval recipe entry point — distributed model inference."""

    DistributedManager.initialize()
    configure_logging()
    dist = DistributedManager()
    device = dist.device

    # Instantiate the pipeline early (no weights loaded yet) so we can
    # consult its class-level flags before the pre-download sentinel check
    # and the primary data-source resolution.
    pipeline = build_pipeline(cfg)

    # --- Pre-download check -------------------------------------------------
    # The top-level sentinel check applies to single-source pipelines, where
    # `predownload.py` writes a sentinel after caching `cfg.data_source`.
    # Multi-source pipelines (needs_data_source=False) handle their own
    # source resolution + BYO via pipeline-specific config blocks, so the
    # top-level sentinel is not meaningful for them.
    if pipeline.needs_data_source:
        fully_byo = (
            cfg.get("ic_source") is not None
            and cfg.get("verification_source") is not None
        )
        if cfg.get("require_predownload", True) and not fully_byo:
            sp = sentinel_path(cfg)
            if not sp.exists():
                raise RuntimeError(
                    f"Pre-download sentinel not found at '{sp}'.\n"
                    "Run 'python predownload.py' with the same config before inference.\n"
                    "To skip this check, set require_predownload=false."
                )
            logger.info(f"Pre-download sentinel found: {sp}")
    else:
        logger.info(
            f"Pipeline '{type(pipeline).__name__}' resolves its own data "
            "sources; skipping top-level predownload sentinel check."
        )

    if online_enabled(cfg):
        _run_online(cfg, pipeline, dist, device)
    else:
        _run_offline(cfg, pipeline, dist, device)

    logger.success("Eval recipe finished.")


def _run_offline(
    cfg: DictConfig,
    pipeline: Pipeline,
    dist: DistributedManager,
    device: torch.device,
) -> None:
    """Store-then-score path — write every field to ``forecast.zarr``.

    Work items are ``(IC, member)`` pairs distributed freely across ranks;
    scoring happens afterwards in ``score.py``.
    """
    all_items = build_work_items(cfg)
    resume = cfg.get("resume", False)

    if resume:
        remaining_items = filter_completed_items(all_items, cfg)
        if not remaining_items:
            logger.success("All work items already completed — nothing to do.")
            if dist.distributed:
                torch.distributed.barrier()
            return
    else:
        remaining_items = all_items
        if cfg.output.get("overwrite", False):
            clear_progress(cfg)

    my_items = distribute_work(remaining_items, dist.rank, dist.world_size)

    pipeline.setup(cfg, device)

    # Use all_items for coord building so the zarr schema always covers the
    # full set of ICs, even when resuming a partial run.
    all_times = np.array(sorted({item.time for item in all_items}))
    output_variables = list(cfg.output.variables)
    total_coords = pipeline.build_total_coords(all_times, cfg.get("ensemble_size", 1))

    # Single-source pipelines: main.py resolves and passes the primary source.
    # Multi-source pipelines: pipeline.setup() has already cached its sources
    # internally, so main.py passes None and the pipeline ignores it.
    if pipeline.needs_data_source:
        data_source = resolve_ic_source(
            cfg,
            byo=cfg.get("ic_source"),
            live_source=cfg.data_source,
        )
    else:
        data_source = None

    with OutputManager(cfg) as output_mgr:
        output_mgr.validate_output_store(total_coords, output_variables)
        if my_items:
            pipeline.run(
                my_items, data_source, output_mgr, output_variables, device, cfg
            )
        else:
            logger.info(f"Rank {dist.rank}: no work items, waiting at barrier.")


def _run_online(
    cfg: DictConfig,
    pipeline: Pipeline,
    dist: DistributedManager,
    device: torch.device,
) -> None:
    """In-line scoring path — reduce to sufficient statistics as we go.

    Ranks are partitioned into ensemble groups; initial conditions are
    distributed across *groups* rather than across ranks, and every rank of
    a group runs the same IC with a different slice of the ensemble.  The
    durable artifact is ``stats.zarr``; ``forecast.zarr`` is written only
    when ``output.retain=all``.

    Every rank — including leftover ranks that join no group — must reach
    the store-creation and context-exit barriers, so the collective calls
    below are deliberately outside the `comm is not None` guards.
    """
    settings = parse_online_settings(cfg)
    ensemble_size = cfg.get("ensemble_size", 1)
    retain_raw = retain_raw_output(cfg)

    if not pipeline.supports_online_scoring:
        raise ValueError(
            f"Pipeline '{type(pipeline).__name__}' is not validated for online "
            "scoring (supports_online_scoring=False).  Use scoring.mode=offline, "
            "or set the flag on the pipeline once its yield sequence is "
            "confirmed identical across ensemble members."
        )
    if settings.members_per_rank > 1 and not pipeline.supports_member_batching():
        raise ValueError(
            f"scoring.online.members_per_rank={settings.members_per_rank} needs "
            f"a batched rollout, but '{type(pipeline).__name__}' does not "
            "implement run_item_batched.  Set members_per_rank=1 and launch at "
            f"least ensemble_size={ensemble_size} ranks."
        )

    # --- Ensemble groups ----------------------------------------------------
    # Collective over the world: every rank enters build_group_comm.
    comm = build_group_comm(settings, dist.rank, dist.world_size, ensemble_size)

    all_items = build_work_items(cfg)
    all_times = np.array(sorted({item.time for item in all_items}))

    resume = cfg.get("resume", False)
    if resume:
        remaining_times = filter_online_completed(list(all_times), cfg)
        if not remaining_times:
            logger.success("All ICs already scored online — nothing to do.")
            if dist.distributed:
                torch.distributed.barrier()
            if dist.rank == 0:
                finalize_stats(cfg)
            return
    else:
        remaining_times = list(all_times)
        if cfg.output.get("overwrite", False):
            clear_online_progress(cfg)

    my_times: list[np.datetime64] = []
    my_items = []
    if comm is not None and remaining_times:
        my_times = distribute_work(
            remaining_times, comm.group.group_id, comm.group.n_groups
        )
        my_items = build_group_work_items(my_times, comm.group, cfg)

    # --- Pipeline setup -----------------------------------------------------
    pipeline.setup(cfg, device)

    scoring_variables = list(cfg.scoring.variables)
    output_variables = (
        list(cfg.output.variables) if retain_raw else list(scoring_variables)
    )
    missing = [v for v in scoring_variables if v not in output_variables]
    if missing:
        raise ValueError(
            f"scoring.variables {missing} are not in output.variables — online "
            "scoring reads the same filtered chunks that are written, so every "
            "scored variable must also be an output variable."
        )

    total_coords = pipeline.build_total_coords(all_times, ensemble_size)
    lead_times = np.asarray(total_coords["lead_time"])
    spatial_coords = pipeline.effective_spatial_ref()

    # --- Verification -------------------------------------------------------
    # Verification is required predownloaded, so full coverage of every
    # (IC + lead) valid time is knowable before the first forecast runs.
    # That turns what would otherwise hang a whole ensemble group hours in
    # into a startup error.
    verif_source = pipeline.verification_source(cfg)
    if my_times:
        check_verification_coverage(
            verif_source, my_times, lead_times, scoring_variables
        )

    if pipeline.needs_data_source:
        data_source = resolve_ic_source(
            cfg,
            byo=cfg.get("ic_source"),
            live_source=cfg.data_source,
        )
    else:
        data_source = None

    # --- Stores -------------------------------------------------------------
    statistics = build_statistics(
        ensemble_size,
        settings.climatology is not None,
        settings,
        scoring_variables,
    )
    stats_mgr = open_stats_store(
        cfg, statistics, scoring_variables, all_times, lead_times, ensemble_size
    )
    raw_mgr = OutputManager(cfg) if retain_raw else None

    with ExitStack() as stack:
        stack.enter_context(stats_mgr)
        if raw_mgr is not None:
            stack.enter_context(raw_mgr)
            raw_mgr.validate_output_store(total_coords, output_variables)

        if my_items and comm is not None:
            scorer = build_online_scorer(
                cfg,
                settings,
                comm,
                verif_source,
                scoring_variables,
                lead_times,
                spatial_coords,
                stats_mgr,
                device,
                known_missing_leads=pipeline.known_missing_leads(),
            )
            pipeline.run(
                my_items,
                data_source,
                raw_mgr,
                output_variables,
                device,
                cfg,
                scorer=scorer,
                member_batch=settings.members_per_rank,
            )
        else:
            logger.info(f"Rank {dist.rank}: no ICs assigned, waiting at barrier.")

    # Cheap and idempotent — re-derives scores.zarr from the full store, so
    # it is correct after a resumed or multi-job campaign too.
    if dist.rank == 0:
        finalize_stats(cfg)


if __name__ == "__main__":
    main()

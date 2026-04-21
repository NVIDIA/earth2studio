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

"""Pre-download script for the eval recipe.

Fetches initial condition (and optionally verification) data for a validation
campaign and writes it into explicit zarr stores before the GPU inference job
runs.  Accepts the same Hydra config as ``main.py`` so the two scripts stay in
sync automatically.

IC variables, lead times, and model step size are inferred directly from the
model's ``input_coords()`` / ``output_coords()``, so no manual bookkeeping is
required when switching models.  Verification variables are taken from
``cfg.output.variables`` — only what will actually be scored is downloaded.

Stores
------
- ``data.zarr`` — primary data store with dimensions
  ``(time, variable, <spatial...>)``.  Contains all times and variables needed
  for model execution and, when verification uses the same source,
  verification data as well.
- ``verification.zarr`` (optional) — separate verification store, created only
  when ``predownload.verification.source`` is explicitly set to a different
  data source than ``cfg.data_source``.

Resume
------
Progress is tracked per-timestamp via marker files.  If interrupted (e.g. by
a SLURM time limit), re-running with the same config skips already-completed
timestamps automatically.  Set ``predownload.overwrite=true`` to recreate
stores from scratch.

Typical usage
-------------
Single-process::

    python predownload.py

Multi-process (CPU workers, e.g. on a login or pre-fetch node)::

    torchrun --nproc_per_node=8 --standalone predownload.py

Also pre-fetch verification data (merged into data.zarr when using the
same source)::

    python predownload.py predownload.verification.enabled=true

Override the IC time range::

    python predownload.py ic_block_start="2024-01-01" ic_block_end="2024-03-31" \\
        ic_block_step=24
"""

from __future__ import annotations

from collections import OrderedDict

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from src.distributed import configure_logging
from src.models import load_diagnostics, load_prognostic
from src.output import OutputManager, build_predownload_coords, sentinel_path
from src.work import (
    build_work_items,
    clear_predownload_progress,
    distribute_work,
    filter_predownload_completed,
    write_predownload_marker,
)

from earth2studio.data import DataSource, ForecastSource, fetch_data
from earth2studio.utils.coords import CoordSystem


def _infer_step_hours(model: object) -> int:
    """Infer the model's output timestep in hours from its coordinate methods.

    Computed as the difference between the first output lead time and the last
    input lead time, which equals the model's intrinsic step size.
    """
    ic_coords = model.input_coords()  # type: ignore[attr-defined]
    out_coords = model.output_coords(ic_coords)  # type: ignore[attr-defined]
    delta = out_coords["lead_time"][0] - ic_coords["lead_time"][-1]
    return int(delta / np.timedelta64(1, "h"))


def _compute_verification_times(
    ic_times: list[np.datetime64],
    nsteps: int,
    step_hours: int,
) -> list[np.datetime64]:
    """Collect all unique model-output valid times across every IC window."""
    step = np.timedelta64(step_hours, "h")
    times: set[np.datetime64] = set()
    for t in ic_times:
        for k in range(nsteps + 1):
            times.add(t + k * step)
    return sorted(times)


def _union_variables(*var_lists: list[str]) -> list[str]:
    """Return the ordered union of multiple variable lists."""
    seen: set[str] = set()
    result: list[str] = []
    for vl in var_lists:
        for v in vl:
            if v not in seen:
                result.append(v)
                seen.add(v)
    return result


def _squeeze_lead_time(
    x: torch.Tensor, coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Remove the lead_time dimension from a (tensor, coords) pair.

    ``fetch_data()`` always returns a ``lead_time`` dimension even when it
    contains a single ``[0ns]`` entry.  The predownload stores have no
    ``lead_time`` dimension, so we squeeze it out before writing.
    """
    lt_idx = list(coords.keys()).index("lead_time")
    x = x.squeeze(lt_idx)
    coords = OrderedDict((k, v) for k, v in coords.items() if k != "lead_time")
    return x, coords


def _download_to_store(
    source: DataSource | ForecastSource,
    times: list[np.datetime64],
    variables: list[str],
    output_mgr: OutputManager,
    cfg: DictConfig,
    store_name: str,
    dist: DistributedManager,
) -> None:
    """Fetch data for *times* from *source* and write to *output_mgr*.

    Handles resume filtering, work distribution, and per-timestamp progress
    markers.  All ranks must call this function (OutputManager barriers require
    it).
    """
    zero_lead = np.array([np.timedelta64(0, "ns")])

    remaining = filter_predownload_completed(times, cfg, store_name)
    my_times = distribute_work(remaining, dist.rank, dist.world_size)

    logger.info(
        f"Rank {dist.rank}: downloading {len(my_times)}/{len(remaining)} "
        f"remaining times ({len(times)} total) for {store_name} — "
        f"{len(variables)} variables"
    )

    for t in my_times:
        logger.info(f"Rank {dist.rank}: fetching {store_name} {t}")
        x, coords = fetch_data(
            source=source,
            time=[t],
            variable=variables,
            lead_time=zero_lead,
            device=torch.device("cpu"),
        )
        x, coords = _squeeze_lead_time(x, coords)
        output_mgr.write(x, coords)
        output_mgr.flush()
        write_predownload_marker(t, cfg, store_name)

    logger.success(f"Rank {dist.rank}: {store_name} download complete.")


def _download_store(
    cfg: DictConfig,
    dist: DistributedManager,
    store_name: str,
    source: DataSource | ForecastSource,
    times: list[np.datetime64],
    variables: list[str],
    spatial_ref: CoordSystem,
    overwrite: bool,
) -> None:
    """Build coords, open the store, validate, and download.

    ``store_name`` is the progress-marker key (e.g. ``"data"``); the on-disk
    zarr is ``f"{store_name}.zarr"``.
    """
    store_coords = build_predownload_coords(
        spatial_ref, np.array(times, dtype="datetime64[ns]")
    )
    with OutputManager(
        cfg,
        store_name=f"{store_name}.zarr",
        overwrite=overwrite,
        resume=not overwrite,
    ) as mgr:
        mgr.validate_output_store(store_coords, variables)
        _download_to_store(
            source=source,
            times=times,
            variables=variables,
            output_mgr=mgr,
            cfg=cfg,
            store_name=store_name,
            dist=dist,
        )


@hydra.main(version_base=None, config_path="cfg", config_name="predownload")
def main(cfg: DictConfig) -> None:
    """Pre-download data for the eval recipe into zarr stores."""

    DistributedManager.initialize()
    configure_logging()
    dist = DistributedManager()

    pd_cfg = cfg.predownload
    pd_overwrite = pd_cfg.get("overwrite", False)

    # Clear progress markers when overwriting (rank 0 only, before barriers).
    if pd_overwrite and dist.rank == 0:
        clear_predownload_progress(cfg)

    pipeline = cfg.get("pipeline", "forecast")

    if pipeline == "forecast":
        _predownload_forecast(cfg, dist, pd_cfg, pd_overwrite)
    elif pipeline == "diagnostic":
        _predownload_diagnostic(cfg, dist, pd_cfg, pd_overwrite)
    else:
        raise ValueError(
            f"Unknown pipeline '{pipeline}'. Expected 'forecast' or 'diagnostic'."
        )

    # --- Sentinel file ------------------------------------------------------
    if dist.distributed:
        torch.distributed.barrier()

    if dist.rank == 0:
        sp = sentinel_path(cfg)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(np.datetime64("now", "s").item().isoformat())
        logger.info(f"Sentinel written: {sp}")

    logger.success("Pre-download finished.")


def _predownload_forecast(
    cfg: DictConfig,
    dist: DistributedManager,
    pd_cfg: DictConfig,
    pd_overwrite: bool,
) -> None:
    """Pre-download data for the forecast pipeline."""
    ic_byo = cfg.get("ic_source") is not None
    verif_byo = cfg.get("verification_source") is not None

    verif_cfg = pd_cfg.get("verification", {})
    verif_enabled = verif_cfg.get("enabled", False)
    verif_has_separate_source = verif_cfg.get("source") is not None

    if ic_byo:
        logger.info("ic_source provided — skipping IC predownload.")
    if verif_byo and verif_enabled:
        logger.info("verification_source provided — skipping verification predownload.")

    do_verif = verif_enabled and not verif_byo
    do_data_store = not ic_byo
    # Merge is only possible when we download both sides from the same source.
    do_merged = do_data_store and do_verif and not verif_has_separate_source
    do_verif_store = do_verif and (ic_byo or verif_has_separate_source)

    if not do_data_store and not do_verif_store:
        logger.info("Nothing to predownload — sentinel will still be written.")
        return

    # Load model to infer IC requirements (CPU only, no inference).
    model = load_prognostic(cfg)
    ic_coords = model.input_coords()
    spatial_ref = model.output_coords(ic_coords)

    all_items = build_work_items(cfg)
    unique_ic_times: list[np.datetime64] = sorted({item.time for item in all_items})

    if do_data_store:
        ic_variables: list[str] = list(ic_coords["variable"])
        ic_lead_times: np.ndarray = ic_coords["lead_time"]
        # Adjusted valid times needed for IC loading.  For a model with
        # lead_times=[-6h, 0h] and IC 2024-01-01T00:00, this produces
        # 2023-12-31T18:00 and 2024-01-01T00:00.
        ic_fetch_times: list[np.datetime64] = sorted(
            {t + lt for t in unique_ic_times for lt in ic_lead_times}
        )
        logger.info(
            f"IC requirements: {len(ic_fetch_times)} unique valid times, "
            f"{len(ic_variables)} variables, "
            f"lead_times={[int(lt / np.timedelta64(1, 'h')) for lt in ic_lead_times]}h"
        )

    if do_verif:
        verif_variables: list[str] = list(cfg.output.variables)
        step_hours = _infer_step_hours(model)
        verif_times = _compute_verification_times(
            unique_ic_times, cfg.nsteps, step_hours
        )

    # --- Primary data store (data.zarr) -----------------------------------
    if do_data_store:
        if do_merged:
            all_times = sorted(set(ic_fetch_times) | set(verif_times))
            all_variables = _union_variables(ic_variables, verif_variables)
            logger.info(
                f"Merged store: {len(all_times)} unique times "
                f"({len(ic_fetch_times)} IC + {len(verif_times)} verif, "
                f"{len(all_times) - len(ic_fetch_times)} verif-only), "
                f"{len(all_variables)} variables"
            )
        else:
            all_times = ic_fetch_times
            all_variables = ic_variables

        data_source = hydra.utils.instantiate(cfg.data_source)
        _download_store(
            cfg=cfg,
            dist=dist,
            store_name="data",
            source=data_source,
            times=all_times,
            variables=all_variables,
            spatial_ref=spatial_ref,
            overwrite=pd_overwrite,
        )

    # --- Separate verification store --------------------------------------
    if do_verif_store:
        verif_source_cfg = verif_cfg.get("source") or cfg.data_source
        verif_source = hydra.utils.instantiate(verif_source_cfg)
        _download_store(
            cfg=cfg,
            dist=dist,
            store_name="verification",
            source=verif_source,
            times=verif_times,
            variables=verif_variables,
            spatial_ref=spatial_ref,
            overwrite=pd_overwrite,
        )


def _predownload_diagnostic(
    cfg: DictConfig,
    dist: DistributedManager,
    pd_cfg: DictConfig,
    pd_overwrite: bool,
) -> None:
    """Pre-download data for the diagnostic-only pipeline."""
    ic_byo = cfg.get("ic_source") is not None
    verif_byo = cfg.get("verification_source") is not None

    verif_cfg = pd_cfg.get("verification", {})
    verif_enabled = verif_cfg.get("enabled", False)

    if ic_byo:
        logger.info("ic_source provided — skipping IC predownload.")
    if verif_byo and verif_enabled:
        logger.info("verification_source provided — skipping verification predownload.")

    do_data_store = not ic_byo
    do_verif_store = verif_enabled and not verif_byo

    if not do_data_store and not do_verif_store:
        logger.info("Nothing to predownload — sentinel will still be written.")
        return

    diagnostics = load_diagnostics(cfg)
    if not diagnostics:
        raise ValueError(
            "Diagnostic pipeline requires at least one entry in 'diagnostics'."
        )

    input_variables = _union_variables(
        *([str(v) for v in dx.input_coords()["variable"]] for dx in diagnostics)
    )

    all_items = build_work_items(cfg)
    unique_times: list[np.datetime64] = sorted({item.time for item in all_items})

    # Spatial reference from first diagnostic model's output coords.
    dx0 = diagnostics[0]
    spatial_ref = dx0.output_coords(dx0.input_coords())

    # --- Primary data store (data.zarr) -----------------------------------
    data_source = None
    if do_data_store:
        data_source = hydra.utils.instantiate(cfg.data_source)
        _download_store(
            cfg=cfg,
            dist=dist,
            store_name="data",
            source=data_source,
            times=unique_times,
            variables=input_variables,
            spatial_ref=spatial_ref,
            overwrite=pd_overwrite,
        )

    # --- Verification store (always separate for diagnostic pipelines) ----
    # For diagnostics, verification times are the same as input times
    # (no forecast lead-time expansion).
    if do_verif_store:
        verif_variables: list[str] = list(cfg.output.variables)
        verif_source_cfg = verif_cfg.get("source")
        if verif_source_cfg is not None:
            verif_source = hydra.utils.instantiate(verif_source_cfg)
        elif data_source is not None:
            verif_source = data_source
        else:
            verif_source = hydra.utils.instantiate(cfg.data_source)
        _download_store(
            cfg=cfg,
            dist=dist,
            store_name="verification",
            source=verif_source,
            times=unique_times,
            variables=verif_variables,
            spatial_ref=spatial_ref,
            overwrite=pd_overwrite,
        )


if __name__ == "__main__":
    main()

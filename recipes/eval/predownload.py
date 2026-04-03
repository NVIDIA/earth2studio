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

Fetches and caches initial condition (and optionally verification) data for a
validation campaign before the GPU inference job runs.  Accepts the same Hydra
config as ``main.py`` so the two scripts stay in sync automatically.

IC variables, lead times, and model step size are inferred directly from the
model's ``input_coords()`` / ``output_coords()``, so no manual bookkeeping is
required when switching models.  Verification variables are taken from
``cfg.output.variables`` — only what will actually be scored is downloaded.

Typical usage
-------------
Single-process::

    python predownload.py

Multi-process (CPU workers, e.g. on a login or pre-fetch node)::

    torchrun --nproc_per_node=8 --standalone predownload.py

With a shared cache directory (set the same variable when launching main.py)::

    python predownload.py predownload.cache_dir=/lustre/shared/e2s_cache

Also pre-fetch ERA5 verification data for the full forecast window::

    python predownload.py predownload.verification.enabled=true

Override the IC time range just like the eval recipe (``ic_block_end`` is
inclusive on the step grid; same ``np.arange`` semantics as ``work.py``)::

    python predownload.py ic_block_start="2024-01-01" ic_block_end="2024-03-31" \\
        ic_block_step=24
"""

from __future__ import annotations

import os

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from src.distributed import configure_logging
from src.models import load_prognostic
from src.output import sentinel_path
from src.work import build_work_items, distribute_work

from earth2studio.data import fetch_data


def _infer_step_hours(model: object) -> int:
    """Infer the model's output timestep in hours from its coordinate methods.

    Computed as the difference between the first output lead time and the last
    input lead time, which equals the model's intrinsic step size.

    Parameters
    ----------
    model : PrognosticModel
        Loaded prognostic model.

    Returns
    -------
    int
        Step size in hours.
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
    """Collect all unique model-output times across every IC window.

    Parameters
    ----------
    ic_times : list[np.datetime64]
        All unique initial condition times.
    nsteps : int
        Number of model steps per forecast (from ``cfg.nsteps``).
    step_hours : int
        Model output timestep in hours.

    Returns
    -------
    list[np.datetime64]
        Sorted, deduplicated list of verification times to fetch.
    """
    step = np.timedelta64(step_hours, "h")
    times: set[np.datetime64] = set()
    for t in ic_times:
        for k in range(nsteps + 1):
            times.add(t + k * step)
    return sorted(times)


@hydra.main(version_base=None, config_path="cfg", config_name="predownload")
def main(cfg: DictConfig) -> None:
    """Pre-download IC and (optionally) verification data for the eval recipe."""

    DistributedManager.initialize()
    configure_logging()
    dist = DistributedManager()

    # Set cache directory BEFORE instantiating any data source so that
    # datasource_cache_root() picks up the override.
    pd_cfg = cfg.predownload
    cache_dir: str | None = pd_cfg.get("cache_dir")
    if cache_dir:
        os.environ["EARTH2STUDIO_DATA_CACHE"] = cache_dir
        logger.info(f"Cache directory overridden to: {cache_dir}")
        logger.info(
            "Set EARTH2STUDIO_DATA_CACHE to the same path when running main.py "
            "to load from this location."
        )

    # --- Load model to infer IC requirements --------------------------------
    # Model is loaded to CPU (no .to(device) call here); we only need the
    # coordinate metadata, not actual inference.
    model = load_prognostic(cfg)
    ic_coords = model.input_coords()
    ic_variables: list[str] = list(ic_coords["variable"])
    ic_lead_times: np.ndarray = ic_coords["lead_time"]

    # --- Build unique IC times and distribute across ranks ------------------
    all_items = build_work_items(cfg)
    # Ensemble members share the same IC data; deduplicate by time only.
    unique_ic_times: list[np.datetime64] = sorted({item.time for item in all_items})
    my_ic_times = distribute_work(unique_ic_times, dist.rank, dist.world_size)

    # --- Initial condition data ---------------------------------------------
    logger.info(
        f"Rank {dist.rank}: pre-downloading IC data for "
        f"{len(my_ic_times)}/{len(unique_ic_times)} times — "
        f"{len(ic_variables)} variables, "
        f"lead_times={[int(lt / np.timedelta64(1, 'h')) for lt in ic_lead_times]}h"
    )

    ic_source = hydra.utils.instantiate(cfg.data_source)

    for t in my_ic_times:
        logger.info(f"Rank {dist.rank}: fetching IC  {t}")
        fetch_data(
            source=ic_source,
            time=[t],
            variable=ic_variables,
            lead_time=ic_lead_times,
            device=torch.device("cpu"),
        )

    logger.success(f"Rank {dist.rank}: IC pre-download complete.")

    # --- Verification data (optional) ---------------------------------------
    verif_cfg = pd_cfg.get("verification", {})
    if verif_cfg.get("enabled", False):
        # Only fetch variables that will actually be written to the output
        # store — no point caching data that will never be scored.
        verif_variables: list[str] = list(cfg.output.variables)

        step_hours = _infer_step_hours(model)
        all_verif_times = _compute_verification_times(
            unique_ic_times, cfg.nsteps, step_hours
        )
        # Distribute verification times independently of IC distribution to
        # avoid duplicate downloads across ranks.
        my_verif_times = distribute_work(all_verif_times, dist.rank, dist.world_size)

        logger.info(
            f"Rank {dist.rank}: pre-downloading verification data for "
            f"{len(my_verif_times)}/{len(all_verif_times)} times — "
            f"{verif_variables} (step={step_hours}h)"
        )

        verif_source = hydra.utils.instantiate(verif_cfg.source)

        for t in my_verif_times:
            logger.info(f"Rank {dist.rank}: fetching verif {t}")
            verif_source(np.array([t], dtype="datetime64[ns]"), verif_variables)

        logger.success(f"Rank {dist.rank}: verification pre-download complete.")

    # --- Sentinel file ------------------------------------------------------
    # Barrier ensures every rank has finished before rank 0 stamps the file,
    # so a partial run can never leave a valid sentinel behind.
    if dist.distributed:
        torch.distributed.barrier()

    if dist.rank == 0:
        sp = sentinel_path(cfg)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(np.datetime64("now", "s").item().isoformat())
        logger.info(f"Sentinel written: {sp}")

    logger.success("Pre-download finished.")


if __name__ == "__main__":
    main()

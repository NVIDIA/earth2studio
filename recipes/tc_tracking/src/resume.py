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

"""Resume an interrupted or partly failed ensemble run.

A run records its settings in ``{project}_run.json`` and the outcome of every
finished batch in ``{project}_status/<batch>.json``: the seed that made it
stable, or ``-1`` with the last seed tried if it stayed unstable. A resumed
run skips the finished batches, reruns the missing ones and continues failed
ones with their next retry seed.
"""

import json
import os
import shutil
from typing import Any

import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.data.tc_hunt_file_output import batch_name
from src.tc_hunt_utils import set_initial_times

Run = tuple[np.datetime64, np.ndarray, tuple[int, ...]]

# settings that define the runs and their seeds, which a resumed run must share
_RUN_KEYS = (
    "model",
    "ics",
    "ensemble_size",
    "batch_size",
    "n_steps",
    "out_vars",
    "store_type",
)


def _project(cfg: DictConfig) -> str:
    return os.path.join(cfg.store_dir, cfg.project.removesuffix(".zarr"))


def run_info_path(cfg: DictConfig) -> str:
    """Path of the file recording the settings of a run."""
    return _project(cfg) + "_run.json"


def status_dir(cfg: DictConfig) -> str:
    """Directory of the status files of the finished batches."""
    return _project(cfg) + "_status"


def _run_info(cfg: DictConfig, seed: int) -> dict[str, Any]:
    out_vars = cfg.get("out_vars", None)
    return {
        "random_seed": seed,
        "model": cfg.model,
        "ics": [np.datetime_as_string(ic, unit="s") for ic in set_initial_times(cfg)],
        "ensemble_size": cfg.ensemble_size,
        "batch_size": cfg.batch_size,
        "n_steps": cfg.n_steps,
        "out_vars": None if out_vars is None else OmegaConf.to_object(out_vars),
        "store_type": cfg.store_type,
    }


def start_run(cfg: DictConfig, seed: int) -> None:
    """Record the settings of a new run, forgetting the batches of an old one.

    Called by one rank only.
    """
    os.makedirs(cfg.store_dir, exist_ok=True)
    if os.path.exists(status_dir(cfg)):
        shutil.rmtree(status_dir(cfg))
    with open(run_info_path(cfg), "w") as f:
        json.dump(_run_info(cfg, seed), f, indent=2)


def resumed_seed(cfg: DictConfig) -> int:
    """Base seed of the run to resume, checking that the settings match.

    Raises
    ------
    FileNotFoundError
        If there is no run to resume
    ValueError
        If the config differs from the run to resume in a setting that defines
        the runs and their seeds
    """
    path = run_info_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(f"resume: no run to resume, {path} does not exist")
    with open(path) as f:
        stored = json.load(f)

    seed = cfg.get("random_seed", None)
    current = _run_info(cfg, stored["random_seed"] if seed is None else int(seed))
    differ = [key for key in ("random_seed",) + _RUN_KEYS if stored[key] != current[key]]
    if differ:
        raise ValueError(
            f"resume: config differs from the run to resume ({path}) in "
            + ", ".join(f"{key}: {stored[key]} -> {current[key]}" for key in differ)
        )
    logger.info(f"resuming the run of {path}, random_seed {stored['random_seed']}")
    return int(stored["random_seed"])


def _status_path(cfg: DictConfig, ic: np.datetime64, mems: np.ndarray) -> str:
    return os.path.join(status_dir(cfg), batch_name(ic, mems) + ".json")


def write_status(
    cfg: DictConfig, ic: np.datetime64, mems: np.ndarray, seed: int, last_seed: int
) -> None:
    """Record a finished batch: its seed, or ``-1`` with the last seed tried."""
    os.makedirs(status_dir(cfg), exist_ok=True)
    with open(_status_path(cfg, ic, mems), "w") as f:
        json.dump({"random_seed": int(seed), "last_seed": int(last_seed)}, f)


def _tracks_complete(cfg: DictConfig, ic: np.datetime64, mems: np.ndarray, seed: int) -> bool:
    if "cyclone_tracking" not in cfg:
        return True
    track_dir = os.path.join(cfg.store_dir, "cyclone_tracks_te")
    ic_str = np.datetime_as_string(ic, unit="s")
    return all(
        os.path.exists(
            os.path.join(
                track_dir,
                f"tracks_{ic_str}_mem_{mem:04d}_seed_{seed}_bs_{cfg.batch_size}.csv",
            )
        )
        for mem in mems
    )


def pending_runs(cfg: DictConfig, runs: list[Run]) -> list[Run]:
    """The runs a resumed run still has to do.

    A stable batch with all its track files is done. A failed batch continues
    with the seeds after the last one tried, and is skipped if there are none
    left (raise ``max_retries`` to give it more). Any other batch reruns from
    its original seed.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config of the run
    runs : list[Run]
        All runs of the ensemble

    Returns
    -------
    list[Run]
        Runs to do, those of failed batches with only their remaining seeds
    """
    pending, done, still_failed = [], 0, 0
    for ic, mems, seeds in runs:
        path = _status_path(cfg, ic, mems)
        if not os.path.exists(path):
            pending.append((ic, mems, seeds))
            continue
        with open(path) as f:
            status = json.load(f)

        if status["random_seed"] != -1:
            if _tracks_complete(cfg, ic, mems, status["random_seed"]):
                done += 1
            else:
                # tracking was interrupted: rerun with the seed that was stable
                pending.append((ic, mems, seeds[seeds.index(status["random_seed"]) :]))
            continue

        tried = status["last_seed"]
        left = seeds[seeds.index(tried) + 1 :] if tried in seeds else ()
        if left:
            pending.append((ic, mems, left))
        else:
            still_failed += 1
            logger.warning(
                f"resume: members {mems.tolist()} of {np.datetime_as_string(ic, unit='s')} "
                "failed with all their seeds, raise stability_check.max_retries to "
                "try more"
            )

    logger.info(
        f"resume: {done} batches done, {len(pending)} to run, {still_failed} failed "
        "without seeds left"
    )
    return pending

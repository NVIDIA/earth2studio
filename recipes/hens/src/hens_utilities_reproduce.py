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

import hashlib
import secrets

import numpy as np
from omegaconf import DictConfig


def get_reproducibility_settings(cfg: DictConfig) -> tuple[str | int, list[int], bool]:
    """Retrieve reproducibility cfg elements or their default values

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object

    Returns
    -------
    base_random_seed: str|int
        a base random seed specfied in the config. If it is an integer it will be converted to a string later
    batch_ids_produce: list[int]
        a list of the batch ids that shall be produced in this run
    torch_use_deterministic_algorithms: bool
        variable that will be used for torch_use_deterministic_algorithms to control if torch is using deterministic algorithms to ensure reproducibility

    """
    try:
        batch_ids_produce = cfg["batch_ids_reproduce"]
    except KeyError:
        batch_ids_produce = list(
            range(
                0,
                int(
                    np.ceil(cfg.nensemble / cfg.batch_size)
                    * cfg.forecast_model.max_num_checkpoints
                ),
            )
        )
    try:
        base_random_seed = cfg["random_seed"]
    except KeyError:
        base_random_seed = secrets.randbelow(1_000_000)
    try:
        torch_use_deterministic_algorithms = cfg["torch_use_deterministic_algorithms"]
    except KeyError:
        torch_use_deterministic_algorithms = False

    return base_random_seed, batch_ids_produce, torch_use_deterministic_algorithms


def calculate_torch_seed(s: str) -> int:
    """Calculates torch seed based on a given string.
    String s is used as input to sha256 hash algorithm.
    Output is converted to integer by taking the maximum integer size of torch seed
    into account.

    Parameters
    ----------
    s : str
        seed string

    Returns
    -------
    torch: np.int64
        integer value that can be used as random seed in torch

    """
    torch_seed = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**64) - 1
    return torch_seed


def create_base_seed_string(pkg: str, ic: np.datetime64, base_random_seed: str) -> str:
    """Concatenates information of model package name, initial condition time and and
    base_random seed into one base seed string.

    Parameters
    ----------
    pkg : str
        Model package name
    ic : np.datetime64
        Initial condition time
    base_random_seed : str
        Base seed string

    Returns
    -------
    base_seed_string: str
        string that can be used as random seed

    """
    s0 = str(base_random_seed)
    s1 = "".join(
        e for e in pkg if e.isalnum()
    )  # remove all special characters from package name
    s2 = str(ic.astype("datetime64[s]"))
    base_seed_string = "_".join([s0, s1, s2])
    return base_seed_string


def calculate_all_torch_seeds(
    base_seed_string: str, batch_ids: list[int]
) -> tuple[np.array, np.array]:
    """
    calculates all torch random seeds that will be used based on the base_seed_string
    and the batch_ids

    Parameters
    ----------
    base_seed_string : str
        base seed
    batch_ids : list[int]
        list of batch_ids that will be calculated

    Returns
    -------
    full_seed_strings: np.array
        contains all seed strings that will be used to calculate torch seeds
    torch_seeds: np.array
        contains all torch random seeds that will be used

    """
    sall = np.char.add(
        np.array(base_seed_string + "_"), np.array([str(x) for x in batch_ids])
    )
    torch_seeds = np.zeros((len(sall), 1), dtype=np.uint64)
    full_seed_strings = np.empty(np.shape(torch_seeds), dtype=object)
    for i, s in enumerate(sall):
        full_seed_strings[i] = s
        torch_seeds[i] = calculate_torch_seed(s)
    return full_seed_strings, torch_seeds


def check_uniquness_of_torch_seeds(torch_seeds: np.array) -> bool:
    """Checks if all torch seeds are unique

    Parameters
    ----------
    torch_seeds : np.array
        Array of torch seeds

    Returns
    -------
    bool:
        True if no duplicates of torch seeds were found

    """
    num_runs = len(torch_seeds)
    num_unique_seeds = len(np.unique(torch_seeds))
    if num_unique_seeds == num_runs:
        all_unique = True
    else:
        all_unique = False
        raise ValueError(
            "Calculated torch seeds for every run must be unique! num_unique_seeds = %s, num_runs = %s"
            % (num_unique_seeds, num_runs)
        )
    return all_unique


def ensure_all_torch_seeds_are_unique(
    ensemble_configs: list[tuple], base_random_seed: str
) -> None:
    """Checks if all torch seeds based on ensemble_configs and base_random_seed are
    unique

    Parameters
    ----------
    ensemble_configs : list[tuple]
        List of ensemble config objects
    base_random_seed : str
        Base seed string

    Raises
    ------
    ValueError
        If the random seeds of all ensembles are not fully unique (duplicates)
    """
    torch_seeds_list = []
    full_seed_string_list = []
    for pkg, ic, _, batch_ids_produce in ensemble_configs:
        base_seed_string = create_base_seed_string(pkg, ic, base_random_seed)
        full_seed_strings, torch_seeds = calculate_all_torch_seeds(
            base_seed_string, batch_ids_produce
        )
        if check_uniquness_of_torch_seeds(torch_seeds):
            torch_seeds_list.append(torch_seeds)
            full_seed_string_list.append(full_seed_strings)
    if torch_seeds_list:
        check_uniquness_of_torch_seeds(np.concatenate(torch_seeds_list, axis=0))
    else:
        raise ValueError("Torch seeds could not be calculated.")

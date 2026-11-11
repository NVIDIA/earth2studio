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
from omegaconf import DictConfig, open_dict


def get_reproducibility_settings(
    cfg: DictConfig, num_packages: int
) -> tuple[str | int, list[int], bool]:
    """Resolve the seed and selected batches using the discovered packages."""
    if "random_seed" not in cfg:
        with open_dict(cfg):
            cfg.random_seed = secrets.randbelow(1_000_000)
    num_batches = -(-cfg.nensemble // cfg.batch_size) * num_packages
    return (
        cfg.random_seed,
        list(cfg.get("batch_ids_reproduce", range(num_batches))),
        cfg.get("torch_use_deterministic_algorithms", False),
    )


def calculate_torch_seed(s: str) -> int:
    """Preserve HENS' SHA256-based batch seed derivation."""
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**64) - 1


def create_base_seed_string(pkg: str, ic: np.datetime64, base_random_seed: str) -> str:
    """Combine seed, sanitized package path and second-resolution IC time."""
    package = "".join(char for char in pkg if char.isalnum())
    return f"{base_random_seed}_{package}_{ic.astype('datetime64[s]')}"


def ensure_all_torch_seeds_are_unique(
    ensemble_configs: list[tuple], base_random_seed: str
) -> None:
    """Reject repeated batch seeds across this worker's checkpoint/IC pairs."""
    seen: set[int] = set()
    for pkg, ic, _, batches in ensemble_configs:
        base = create_base_seed_string(pkg, ic, base_random_seed)
        for batch in batches:
            seed = calculate_torch_seed(f"{base}_{batch}")
            if seed in seen:
                raise ValueError("Calculated torch seeds for every run must be unique")
            seen.add(seed)

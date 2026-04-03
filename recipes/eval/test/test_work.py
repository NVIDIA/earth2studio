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

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf
from src.work import WorkItem, build_work_items, distribute_work

# ---------------------------------------------------------------------------
# build_work_items
# ---------------------------------------------------------------------------


class TestBuildWorkItems:
    def test_explicit_start_times(self):
        cfg = OmegaConf.create(
            {
                "start_times": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
                "ensemble_size": 1,
                "random_seed": 42,
            }
        )
        items = build_work_items(cfg)
        assert len(items) == 2
        assert all(isinstance(i, WorkItem) for i in items)
        assert items[0].ensemble_id == 0
        assert items[1].ensemble_id == 0

    def test_ic_block_range(self):
        cfg = OmegaConf.create(
            {
                "ic_block_start": "2024-01-01 00:00:00",
                "ic_block_end": "2024-01-03 00:00:00",
                "ic_block_step": 24,
                "ensemble_size": 1,
                "random_seed": 0,
            }
        )
        items = build_work_items(cfg)
        assert len(items) == 3
        times = [i.time for i in items]
        assert times[0] == np.datetime64("2024-01-01")
        assert times[-1] == np.datetime64("2024-01-03")

    def test_ensemble_multiplies_items(self):
        cfg = OmegaConf.create(
            {
                "start_times": ["2024-01-01 00:00:00"],
                "ensemble_size": 4,
                "random_seed": 42,
            }
        )
        items = build_work_items(cfg)
        assert len(items) == 4
        ensemble_ids = [i.ensemble_id for i in items]
        assert ensemble_ids == [0, 1, 2, 3]

    def test_both_sources_raises(self):
        cfg = OmegaConf.create(
            {
                "start_times": ["2024-01-01 00:00:00"],
                "ic_block_start": "2024-01-01 00:00:00",
                "ic_block_end": "2024-01-02 00:00:00",
                "ic_block_step": 24,
            }
        )
        with pytest.raises(ValueError, match="not both"):
            build_work_items(cfg)

    def test_no_sources_raises(self):
        cfg = OmegaConf.create({"ensemble_size": 1, "random_seed": 0})
        with pytest.raises(ValueError, match="must specify"):
            build_work_items(cfg)

    def test_deterministic_seeds(self):
        cfg = OmegaConf.create(
            {
                "start_times": ["2024-01-01 00:00:00"],
                "ensemble_size": 3,
                "random_seed": 42,
            }
        )
        items_a = build_work_items(cfg)
        items_b = build_work_items(cfg)
        assert [i.seed for i in items_a] == [i.seed for i in items_b]
        # Each ensemble member should get a distinct seed
        seeds = {i.seed for i in items_a}
        assert len(seeds) == 3


# ---------------------------------------------------------------------------
# distribute_work
# ---------------------------------------------------------------------------


class TestDistributeWork:
    def test_single_rank(self):
        items = list(range(5))
        assert distribute_work(items, rank=0, world_size=1) == items

    def test_even_split(self):
        items = list(range(6))
        r0 = distribute_work(items, rank=0, world_size=3)
        r1 = distribute_work(items, rank=1, world_size=3)
        r2 = distribute_work(items, rank=2, world_size=3)
        assert sorted(r0 + r1 + r2) == items
        assert len(r0) == len(r1) == len(r2) == 2

    def test_uneven_split_remainder_goes_to_early_ranks(self):
        items = list(range(7))
        r0 = distribute_work(items, rank=0, world_size=3)
        r1 = distribute_work(items, rank=1, world_size=3)
        r2 = distribute_work(items, rank=2, world_size=3)
        assert sorted(r0 + r1 + r2) == items
        # 7 / 3 = 2 remainder 1 → rank 0 gets 3, ranks 1-2 get 2
        assert len(r0) == 3
        assert len(r1) == 2
        assert len(r2) == 2

    def test_more_ranks_than_items(self):
        items = list(range(2))
        r0 = distribute_work(items, rank=0, world_size=4)
        r1 = distribute_work(items, rank=1, world_size=4)
        r2 = distribute_work(items, rank=2, world_size=4)
        r3 = distribute_work(items, rank=3, world_size=4)
        assert sorted(r0 + r1 + r2 + r3) == items
        assert r2 == []
        assert r3 == []

    def test_empty_items(self):
        assert distribute_work([], rank=0, world_size=2) == []

    @pytest.mark.parametrize("world_size", [1, 2, 3, 5, 8, 13])
    def test_no_item_lost(self, world_size):
        items = list(range(10))
        collected = []
        for rank in range(world_size):
            collected.extend(distribute_work(items, rank, world_size))
        assert sorted(collected) == items

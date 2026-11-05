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

"""Campaign planning: initial times, items and group ordering."""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf
from src.campaign import (
    build_items,
    ensemble_members,
    final_lead_time,
    initial_times,
    members_per_item,
    order_for_groups,
)

from earth2studio.run import work_items
from earth2studio.run.work import distribute_work


class _StepRunner:
    step = np.timedelta64(6, "h")


class _NoStepRunner:
    pass


class _NoBatchRunner(_StepRunner):
    supports_member_batching = False
    stochastic_models = ("prognostic",)


class _BatchRunner(_StepRunner):
    supports_member_batching = True


def test_initial_times_from_list() -> None:
    cfg = OmegaConf.create(
        {"start_times": ["2024-01-02 00:00:00", "2024-01-01 00:00:00"]}
    )
    times = initial_times(cfg)
    assert times == [
        np.datetime64("2024-01-01T00:00:00"),
        np.datetime64("2024-01-02T00:00:00"),
    ]


def test_initial_times_from_block_is_end_inclusive() -> None:
    cfg = OmegaConf.create(
        {
            "ic_block_start": "2024-01-01 00:00:00",
            "ic_block_end": "2024-01-02 00:00:00",
            "ic_block_step": 12,
        }
    )
    assert len(initial_times(cfg)) == 3


def test_initial_times_rejects_both_forms() -> None:
    cfg = OmegaConf.create(
        {"start_times": ["2024-01-01"], "ic_block_start": "2024-01-01"}
    )
    with pytest.raises(ValueError, match="not both"):
        initial_times(cfg)


def test_event_ics_join_the_campaign() -> None:
    cfg = OmegaConf.create(
        {
            "start_times": ["2024-01-01 00:00:00"],
            "scoring": {
                "events": {
                    "storm": {
                        "start": "2024-02-01 00:00:00",
                        "end": "2024-02-01 12:00:00",
                        "region": None,
                        "ics": {"step_hours": 12, "lookback_hours": 24},
                    }
                }
            },
        }
    )
    times = initial_times(cfg)
    assert times[0] == np.datetime64("2024-01-01T00:00:00")
    assert np.datetime64("2024-01-31T00:00:00") in times
    assert times[-1] == np.datetime64("2024-02-01T12:00:00")


def test_members_and_final_lead_time() -> None:
    assert ensemble_members(OmegaConf.create({"ensemble_size": 1})) is None
    assert ensemble_members(OmegaConf.create({"ensemble_size": 4})) == 4
    assert final_lead_time(
        OmegaConf.create({"nsteps": 4}), _StepRunner()
    ) == np.timedelta64(24, "h")
    assert final_lead_time(
        OmegaConf.create({"nsteps": 0}), _NoStepRunner()
    ) == np.timedelta64(0, "h")
    with pytest.raises(ValueError, match="no model step"):
        final_lead_time(OmegaConf.create({"nsteps": 2}), _NoStepRunner())


def test_members_per_item_follows_scoring_mode() -> None:
    offline = OmegaConf.create({"member_batch": 2, "scoring": {"mode": "offline"}})
    online = OmegaConf.create(
        {"scoring": {"mode": "online", "online": {"members_per_rank": 4}}}
    )
    assert members_per_item(offline) == 2
    assert members_per_item(online) == 4


def test_members_per_item_caps_runners_without_member_batching() -> None:
    offline = OmegaConf.create({"member_batch": 2, "scoring": {"mode": "offline"}})
    online = OmegaConf.create(
        {"scoring": {"mode": "online", "online": {"members_per_rank": 4}}}
    )
    assert members_per_item(offline, _BatchRunner()) == 2
    assert members_per_item(online, _BatchRunner()) == 4
    assert members_per_item(offline, _NoBatchRunner()) == 1
    with pytest.raises(ValueError, match="batch_stochastic_members"):
        members_per_item(online, _NoBatchRunner())
    single = OmegaConf.create({"member_batch": 1})
    assert members_per_item(single, _NoBatchRunner()) == 1


def test_build_items_matches_library_work_items() -> None:
    cfg = OmegaConf.create(
        {
            "start_times": ["2024-01-01", "2024-01-02"],
            "nsteps": 2,
            "ensemble_size": 4,
            "member_batch": 2,
        }
    )
    items = build_items(cfg, _StepRunner(), member_batch=members_per_item(cfg))
    expected = work_items(
        [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
        np.timedelta64(12, "h"),
        members=4,
        member_batch=2,
    )
    assert items == expected


@pytest.mark.parametrize(
    "n_times,world_size,group_size", [(5, 4, 2), (3, 6, 3), (7, 2, 2), (4, 4, 1)]
)
def test_order_for_groups_gives_lockstep_groups(
    n_times: int, world_size: int, group_size: int
) -> None:
    times = [
        np.datetime64("2024-01-01") + np.timedelta64(d, "D") for d in range(n_times)
    ]
    members = group_size * 2
    items = work_items(times, np.timedelta64(6, "h"), members=members, member_batch=2)
    ordered = order_for_groups(items, world_size, group_size)
    assert sorted(ordered, key=lambda i: (i.time, i.member_ids)) == sorted(
        items, key=lambda i: (i.time, i.member_ids)
    )
    slices = [distribute_work(ordered, rank, world_size) for rank in range(world_size)]
    for block in range(world_size // group_size):
        group = slices[block * group_size : (block + 1) * group_size]
        # Every rank of a group runs the same times in the same rounds.
        for rank_items in group[1:]:
            assert [i.time for i in rank_items] == [i.time for i in group[0]]
        # Each rank carries one fixed member group.
        for rank_items in group:
            assert len({i.member_ids for i in rank_items}) <= 1
        # Together a group covers every member of each of its times.
        for round_items in zip(*group):
            assert sorted(m for i in round_items for m in i.member_ids) == list(
                range(members)
            )


def test_order_for_groups_rejects_bad_shapes() -> None:
    items = work_items(
        [np.datetime64("2024-01-01")], np.timedelta64(6, "h"), members=4, member_batch=2
    )
    with pytest.raises(ValueError, match="multiple"):
        order_for_groups(items, 3, 2)
    with pytest.raises(ValueError, match="member groups"):
        order_for_groups(items, 4, 4)

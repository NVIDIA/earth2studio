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

"""Tests for ``earth2studio.run.work``: items, distribution, seed and progress."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from earth2studio.run.runner import WorkItem
from earth2studio.run.work import (
    COMPACT_LOG_NAME,
    MANIFEST_NAME,
    ProgressStore,
    barrier,
    distribute_work,
    member_seed,
    raise_if_any_failed,
    rank0_first,
    rank_info,
    work_items,
)

T0 = np.datetime64("2026-01-01T00")
T1 = np.datetime64("2026-01-01T06")
T2 = np.datetime64("2026-01-01T12")


# ---------------------------------------------------------------------------
# work_items
# ---------------------------------------------------------------------------


def test_work_items_deterministic_time_major():
    items = work_items([T0, T1], np.timedelta64(12, "h"))
    assert len(items) == 2
    assert all(not item.member_ids for item in items)
    assert items[0].time == T0
    assert items[1].time == T1
    assert items[0].horizon == np.timedelta64(12, "h")
    assert items[1].horizon == np.timedelta64(12, "h")


def test_work_items_member_batches_never_span_times():
    items = work_items(
        [T0, T1],
        np.timedelta64(6, "h"),
        members=4,
        member_batch=3,
    )
    assert items == [
        WorkItem(time=T0, horizon=np.timedelta64(6, "h"), member_ids=(0, 1, 2)),
        WorkItem(time=T0, horizon=np.timedelta64(6, "h"), member_ids=(3,)),
        WorkItem(time=T1, horizon=np.timedelta64(6, "h"), member_ids=(0, 1, 2)),
        WorkItem(time=T1, horizon=np.timedelta64(6, "h"), member_ids=(3,)),
    ]


def test_work_items_explicit_members():
    items = work_items(
        [T0], np.timedelta64(1, "h"), members=[10, 20, 30], member_batch=2
    )
    assert len(items) == 2
    assert items[0].member_ids == (10, 20)
    assert items[1].member_ids == (30,)


def test_work_items_repeated_time_raises():
    with pytest.raises(ValueError, match="Repeated time"):
        work_items([T0, T0], np.timedelta64(1, "h"))


def test_work_items_repeated_member_raises():
    with pytest.raises(ValueError, match="unique"):
        work_items([T0], np.timedelta64(1, "h"), members=[0, 0])


def test_work_items_invalid_member_batch():
    with pytest.raises(ValueError, match="member_batch"):
        work_items([T0], np.timedelta64(1, "h"), members=4, member_batch=0)


def test_work_items_negative_horizon():
    with pytest.raises(ValueError, match="non-negative"):
        work_items([T0], np.timedelta64(-1, "h"))


def test_work_items_empty_members_rejected():
    with pytest.raises(ValueError, match="empty"):
        work_items([T0], np.timedelta64(1, "h"), members=[])


def test_work_items_zero_members_rejected():
    with pytest.raises(ValueError, match=">= 1"):
        work_items([T0], np.timedelta64(1, "h"), members=0)


def test_work_items_negative_member_id_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        work_items([T0], np.timedelta64(1, "h"), members=[0, -1])


def test_work_items_hashes_independent_of_precision():
    a = work_items([np.datetime64("2026-01-01", "D")], np.timedelta64(1, "h"))
    b = work_items(
        [np.datetime64("2026-01-01T00:00:00.000000000")], np.timedelta64(60, "m")
    )
    assert a == b


# ---------------------------------------------------------------------------
# member_seed
# ---------------------------------------------------------------------------


def test_member_seed_matches_eval_recipe():
    """X6: member_seed must match the eval recipe byte-for-byte."""

    # Recreate the eval derivation in recipes/eval/src/work.py
    def eval_seed(base: int, time: np.datetime64, ensemble_id: int) -> int:
        import struct

        time_int = int(time.astype("datetime64[s]").astype("int64"))
        packed = struct.pack(">qqq", base, time_int, ensemble_id)
        h = 0xCBF29CE484222325
        for byte in packed:
            h ^= byte
            h = (h * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
        return h % (2**63)

    for base, member in [(0, 0), (42, 3), (7, 1), (12345, 99)]:
        assert member_seed(base, T0, member) == eval_seed(base, T0, member)
    # Golden values from recipes/eval/src/work.py::_deterministic_seed
    golden_time = np.datetime64("2024-01-01T00:00:00", "s")
    assert member_seed(0, golden_time, 0) == 8418075884780428886
    assert member_seed(42, golden_time, 3) == 44700045299834869
    assert (
        member_seed(7, np.datetime64("1970-01-01T00:00:00", "s"), 1)
        == 6480509491129526163
    )


# ---------------------------------------------------------------------------
# distribute_work
# ---------------------------------------------------------------------------


def test_distribute_work_balanced_split():
    items = list(range(10))
    rank0 = distribute_work(items, 0, 3)
    rank1 = distribute_work(items, 1, 3)
    rank2 = distribute_work(items, 2, 3)
    assert rank0 == [0, 1, 2, 3]
    assert rank1 == [4, 5, 6]
    assert rank2 == [7, 8, 9]
    assert sorted(rank0 + rank1 + rank2) == items


def test_distribute_work_idle_ranks_get_empty():
    items = ["a", "b"]
    assert distribute_work(items, 0, 4) == ["a"]
    assert distribute_work(items, 1, 4) == ["b"]
    assert distribute_work(items, 2, 4) == []
    assert distribute_work(items, 3, 4) == []


def test_distribute_work_single_process():
    items = [1, 2, 3]
    assert distribute_work(items, 0, 1) == items


def test_distribute_work_empty_items():
    assert distribute_work([], 0, 4) == []


def test_distribute_work_bad_rank():
    with pytest.raises(ValueError, match="rank"):
        distribute_work([1, 2], 3, 2)


def test_distribute_work_bad_world_size():
    with pytest.raises(ValueError, match="world_size"):
        distribute_work([1, 2], 0, 0)


# ---------------------------------------------------------------------------
# ProgressStore basics
# ---------------------------------------------------------------------------


@pytest.fixture
def progress_dir(tmp_path: Path) -> Path:
    return tmp_path / "progress"


def _items_deterministic() -> list[WorkItem]:
    return [
        WorkItem(time=T0, horizon=np.timedelta64(6, "h")),
        WorkItem(time=T1, horizon=np.timedelta64(6, "h")),
        WorkItem(time=T2, horizon=np.timedelta64(6, "h")),
    ]


def _items_ensemble() -> list[WorkItem]:
    return list(work_items([T0, T1], np.timedelta64(6, "h"), members=4, member_batch=2))


def test_progress_store_writes_and_is_done(progress_dir: Path):
    store = ProgressStore(progress_dir)
    store.open({"version": 1})
    items = _items_deterministic()
    assert not store.is_done(items[0])
    store.mark_done(items[0])
    assert store.is_done(items[0])
    assert not store.is_done(items[1])
    store.close()


def test_progress_store_remaining(progress_dir: Path):
    store = ProgressStore(progress_dir)
    store.open({"version": 1})
    items = _items_deterministic()
    store.mark_done(items[0])
    store.mark_done(items[2])
    assert store.remaining(items) == [items[1]]


def test_progress_store_partly_done_group_reruns(progress_dir: Path):
    store = ProgressStore(progress_dir)
    store.open({"version": 1})
    items = _items_ensemble()
    store.mark_done(items[0])  # (T0, members 0,1)
    assert store.remaining(items) == items[1:]


def test_progress_store_overwrite_clears_manifest_and_logs(progress_dir: Path):
    store1 = ProgressStore(progress_dir, tags={"model": "a"})
    store1.open({"version": 1})
    store1.mark_done(_items_deterministic()[0])
    store1.close()

    store2 = ProgressStore(progress_dir, overwrite=True)
    store2.open({"version": 1})
    assert store2.remaining(_items_deterministic()) == _items_deterministic()
    assert not (progress_dir / MANIFEST_NAME).exists() or (
        json.loads((progress_dir / MANIFEST_NAME).read_text())
        .get("tags", {})
        .get("model")
        is None
    )


def test_progress_store_manifest_mismatch(progress_dir: Path):
    store = ProgressStore(progress_dir)
    store.open({"version": 1})
    store.close()
    store2 = ProgressStore(progress_dir)
    with pytest.raises(ValueError, match="differing keys: seed"):
        store2.open({"version": 1, "seed": 7})


def test_progress_store_resume_with_changed_grouping(progress_dir: Path):
    store = ProgressStore(progress_dir)
    store.open({"version": 1})
    for item in work_items([T0], np.timedelta64(6, "h"), members=4, member_batch=2):
        store.mark_done(item)
    store.close()

    store2 = ProgressStore(progress_dir)
    store2.open({"version": 1})
    regrouped = work_items([T0], np.timedelta64(6, "h"), members=4, member_batch=1)
    assert store2.remaining(regrouped) == []


def test_progress_store_resume_with_changed_world_size(progress_dir: Path):
    store = ProgressStore(progress_dir)
    store.open({"version": 1})
    items = _items_deterministic()
    for item in items[::2]:
        store.mark_done(item)
    store.close()

    store2 = ProgressStore(progress_dir)
    store2.open({"version": 1})
    assert store2.remaining(items) == items[1::2]


def test_progress_store_torn_last_line_ignored(progress_dir: Path):
    store = ProgressStore(progress_dir)
    store.open({"version": 1})
    store.mark_done(_items_deterministic()[1])
    store.close()
    log = sorted(progress_dir.glob("done-*.log"))[0]

    def _record(time: np.datetime64) -> str:
        return json.dumps(
            {
                "time": str(np.datetime64(time, "s")),
                "horizon": int(
                    np.timedelta64(6, "h").astype("timedelta64[s]").astype("int64")
                ),
                "members": [],
            }
        )

    # A complete record, then a torn one that never finished its fsync
    log.write_text(_record(T0) + "\n" + _record(T2)[:10])

    store2 = ProgressStore(progress_dir)
    store2.open({"version": 1})
    items = _items_deterministic()
    assert store2.is_done(items[0])
    assert not store2.is_done(items[2])


def test_progress_store_compaction_bounds_file_count(progress_dir: Path):
    items = _items_deterministic()
    for attempt in range(3):
        store = ProgressStore(progress_dir)
        store.open({"version": 1})
        store.mark_done(items[attempt])
        store.close()
        # At most the manifest, the compacted log and this attempt's log
        assert len(list(progress_dir.iterdir())) <= 3

    assert (progress_dir / COMPACT_LOG_NAME).exists()
    assert len(sorted(progress_dir.glob("done-*.log"))) == 1

    final = ProgressStore(progress_dir)
    final.open({"version": 1})
    assert final.remaining(items) == []


def test_progress_store_manifest_includes_tags(progress_dir: Path):
    store = ProgressStore(progress_dir, tags={"model": "sfno", "step": 1})
    store.open({"version": 1})
    stored = json.loads((progress_dir / MANIFEST_NAME).read_text())
    assert stored["tags"]["model"] == "sfno"
    assert stored["tags"]["step"] == 1


# ---------------------------------------------------------------------------
# Rank helpers are no-ops without a process group; here we only exercise the
# single-process fallbacks. Multi-rank collectives are covered later tracks.
# ---------------------------------------------------------------------------


def test_rank_helpers_single_process():
    assert rank_info() == (0, 1)
    assert rank0_first(lambda: 123) == 123
    barrier()
    raise_if_any_failed(None)


def test_raise_if_any_failed_re_raises_local_error():
    with pytest.raises(ValueError, match="boom"):
        raise_if_any_failed(ValueError("boom"))

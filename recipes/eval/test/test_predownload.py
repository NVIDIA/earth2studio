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

from collections import OrderedDict

import numpy as np
import pytest
from omegaconf import OmegaConf
from predownload import _compute_verification_times, _infer_step_hours
from src.output import sentinel_path
from src.work import build_work_items

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(input_lead_times_h: list[int], output_lead_times_h: list[int]):
    """Return a minimal duck-typed model with controlled coordinate methods."""

    class _FakeModel:
        def input_coords(self) -> OrderedDict:
            return OrderedDict(
                {
                    "variable": np.array(["t2m", "z500"]),
                    "lead_time": np.array(
                        [np.timedelta64(h, "h") for h in input_lead_times_h]
                    ),
                }
            )

        def output_coords(self, input_coords: OrderedDict) -> OrderedDict:
            return OrderedDict(
                {
                    "variable": np.array(["t2m", "z500"]),
                    "lead_time": np.array(
                        [np.timedelta64(h, "h") for h in output_lead_times_h]
                    ),
                }
            )

    return _FakeModel()


# ---------------------------------------------------------------------------
# _compute_verification_times
# ---------------------------------------------------------------------------


class TestComputeVerificationTimes:
    def test_single_ic_correct_times(self):
        t0 = np.datetime64("2024-01-01T00")
        result = _compute_verification_times([t0], nsteps=2, step_hours=6)
        expected = [
            np.datetime64("2024-01-01T00"),
            np.datetime64("2024-01-01T06"),
            np.datetime64("2024-01-01T12"),
        ]
        assert result == expected

    def test_nsteps_zero_returns_ic_time_only(self):
        t0 = np.datetime64("2024-01-01T00")
        result = _compute_verification_times([t0], nsteps=0, step_hours=6)
        assert result == [t0]

    def test_overlapping_windows_deduplicated(self):
        # IC times 6 h apart; with nsteps=1 the first IC's second verification
        # time equals the second IC's analysis time — should appear once.
        t1 = np.datetime64("2024-01-01T00")
        t2 = np.datetime64("2024-01-01T06")
        result = _compute_verification_times([t1, t2], nsteps=1, step_hours=6)
        # t1 → [00, 06];  t2 → [06, 12];  union → [00, 06, 12]
        assert result == [
            np.datetime64("2024-01-01T00"),
            np.datetime64("2024-01-01T06"),
            np.datetime64("2024-01-01T12"),
        ]

    def test_result_is_sorted(self):
        # Supply ICs in reverse order to make sure output is always sorted.
        t1 = np.datetime64("2024-01-02T00")
        t2 = np.datetime64("2024-01-01T00")
        result = _compute_verification_times([t1, t2], nsteps=1, step_hours=24)
        assert result == sorted(result)

    def test_24h_step(self):
        t0 = np.datetime64("2024-01-01T00")
        result = _compute_verification_times([t0], nsteps=3, step_hours=24)
        assert result == [
            np.datetime64("2024-01-01"),
            np.datetime64("2024-01-02"),
            np.datetime64("2024-01-03"),
            np.datetime64("2024-01-04"),
        ]


# ---------------------------------------------------------------------------
# _infer_step_hours
# ---------------------------------------------------------------------------


class TestInferStepHours:
    def test_single_input_step_6h(self):
        # Single-step model: input lead_time=[0h], output lead_time=[6h]
        model = _make_model(input_lead_times_h=[0], output_lead_times_h=[6])
        assert _infer_step_hours(model) == 6

    def test_two_input_steps_dlwp_like(self):
        # DLWP-style: input lead_times=[-6h, 0h], output lead_times=[6h, 12h]
        # Step = output[0] - input[-1] = 6h - 0h = 6h
        model = _make_model(input_lead_times_h=[-6, 0], output_lead_times_h=[6, 12])
        assert _infer_step_hours(model) == 6

    def test_24h_step_model(self):
        model = _make_model(input_lead_times_h=[0], output_lead_times_h=[24])
        assert _infer_step_hours(model) == 24


# ---------------------------------------------------------------------------
# IC time deduplication (ensemble members must not cause duplicate fetches)
# ---------------------------------------------------------------------------


class TestIcTimeDeduplication:
    """Verify that the set-based deduplication used in predownload.main()
    collapses ensemble members back to unique IC times."""

    @pytest.fixture()
    def ensemble_cfg(self):
        return OmegaConf.create(
            {
                "start_times": ["2024-01-01 00:00:00", "2024-01-02 00:00:00"],
                "ensemble_size": 4,
                "random_seed": 42,
            }
        )

    def test_unique_times_equal_ic_count(self, ensemble_cfg):
        items = build_work_items(ensemble_cfg)
        assert len(items) == 8  # 2 ICs × 4 members

        unique_times = sorted({item.time for item in items})
        assert len(unique_times) == 2  # deduped back to just the IC times

    def test_unique_times_match_start_times(self, ensemble_cfg):
        items = build_work_items(ensemble_cfg)
        unique_times = sorted({item.time for item in items})
        assert unique_times[0] == np.datetime64("2024-01-01T00:00:00")
        assert unique_times[1] == np.datetime64("2024-01-02T00:00:00")


# ---------------------------------------------------------------------------
# Sentinel file
# ---------------------------------------------------------------------------


class TestSentinelPath:
    @pytest.fixture()
    def output_cfg(self, tmp_path):
        return OmegaConf.create({"output": {"path": str(tmp_path / "outputs")}})

    def test_path_is_inside_output_dir(self, output_cfg, tmp_path):
        sp = sentinel_path(output_cfg)
        assert sp.parent == tmp_path / "outputs"
        assert sp.name == ".predownload.done"

    def test_sentinel_absent_before_predownload(self, output_cfg):
        sp = sentinel_path(output_cfg)
        assert not sp.exists()

    def test_sentinel_present_after_write(self, output_cfg):
        sp = sentinel_path(output_cfg)
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text("2024-01-01T00:00:00")
        assert sp.exists()

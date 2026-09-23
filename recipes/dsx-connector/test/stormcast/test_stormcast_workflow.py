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


"""StormCast workflow lifecycle and failure handling."""

from __future__ import annotations

import sys
import threading
import types
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

if TYPE_CHECKING:
    from src.stormcast.workflow import _WorkflowSettings


def _install_stormcast_fakes(monkeypatch, deterministic) -> None:
    """Fake earth2studio + conditioning/coverage so StormCast's _run runs without the model stack."""
    run_mod = types.ModuleType("earth2studio.run")
    run_mod.deterministic = deterministic
    data_mod = types.ModuleType("earth2studio.data")
    data_mod.GFS = data_mod.HRRR = object
    time_mod = types.ModuleType("earth2studio.utils.time")
    time_mod.to_time_array = lambda lst: np.array([np.datetime64(lst[0])])
    utils = types.ModuleType("earth2studio.utils")
    utils.time = time_mod
    e2s = types.ModuleType("earth2studio")
    e2s.run, e2s.data, e2s.utils = run_mod, data_mod, utils
    torch_mod = types.ModuleType("torch")
    torch_mod.device = lambda *a, **k: "cpu"  # keep the test torch-free
    # conditioning is imported lazily inside the workflow functions (it imports torch), so a
    # sys.modules stub is picked up at call time.
    cond = types.ModuleType("src.stormcast.conditioning")
    cond._ConditioningDataNotReady = type("_ConditioningDataNotReady", (Exception,), {})
    cond.build_conditioning_model = lambda *a, **k: MagicMock()
    cond.run_conditioning = lambda *a, **k: MagicMock()
    for name, mod in [
        ("torch", torch_mod),
        ("earth2studio", e2s),
        ("earth2studio.run", run_mod),
        ("earth2studio.data", data_mod),
        ("earth2studio.utils", utils),
        ("earth2studio.utils.time", time_mod),
        ("src.stormcast.conditioning", cond),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    # coverage IS imported at the workflow module top (it is torch-free), so patch the names
    # bound on the workflow module rather than via sys.modules.
    import src.stormcast.workflow as wf

    monkeypatch.setattr(wf, "conditioning_covers", lambda *a, **k: True)
    monkeypatch.setattr(wf, "conditioning_hours", lambda n: 36)


def _settings(cache_retention_hours: float = 0) -> _WorkflowSettings:
    from src.stormcast.workflow import _WorkflowSettings

    return _WorkflowSettings(
        fixed_init=datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
        conditioning_output_path="conditioning.nc",
        conditioning_kind="sfno",
        model_device="cpu",
        conditioning_device="cpu",
        max_lookback=12,
        max_consecutive_failures=5,
        cache_retention_hours=cache_retention_hours,
        nsteps=6,
        poll_interval_s=600,
        heartbeat_s=90,
        one_shot=True,
    )


def _base_config() -> dict[str, object]:
    return {
        "sites": [],
        "conditioning": {},
        "model": {"device": "cpu"},
        "run": {
            "init_time": "latest",
            "nsteps": 6,
            "max_consecutive_failures": 5,
        },
        "bus": {},
    }


def test_parse_workflow_settings_normalizes_values() -> None:
    from src.stormcast.workflow import _parse_workflow_settings

    settings = _parse_workflow_settings(
        {
            "model": {"device": "cuda:0"},
            "conditioning": {
                "model": "fcn3",
                "device": "cuda:1",
                "max_lookback_hours": 18,
                "scratch_dir": "/workspace",
            },
            "run": {
                "init_time": "2026-01-02T13:00:00+01:00",
                "nsteps": 8,
                "poll_interval_seconds": 30,
                "max_consecutive_failures": 3,
                "cache_retention_hours": 48,
            },
            "bus": {"heartbeat_seconds": 60},
        },
        types.SimpleNamespace(once=True, dry_run=False),
    )

    assert settings.fixed_init == datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    assert settings.conditioning_output_path == "/workspace/dsx_conditioning.nc"
    assert settings.conditioning_kind == "fcn3"
    assert settings.model_device == "cuda:0"
    assert settings.conditioning_device == "cuda:1"
    assert settings.max_lookback == 18
    assert settings.max_consecutive_failures == 3
    assert settings.cache_retention_hours == 48
    assert settings.nsteps == 8
    assert settings.poll_interval_s == 30
    assert settings.heartbeat_s == 60
    assert settings.one_shot is True


@pytest.mark.parametrize(
    ("conditioning", "message"),
    [
        ([], "must be a mapping"),
        ({"model": "unknown"}, "conditioning.model"),
        ({"device": ""}, "conditioning.device"),
        ({"device": 1}, "conditioning.device"),
        ({"max_lookback_hours": -1}, "max_lookback_hours"),
        ({"max_lookback_hours": 1.5}, "max_lookback_hours"),
        ({"max_lookback_hours": True}, "max_lookback_hours"),
    ],
)
def test_run_rejects_invalid_conditioning_before_model_import(
    monkeypatch, conditioning: object, message: str
) -> None:
    import src.stormcast.workflow as workflow

    monkeypatch.setattr(workflow, "validate_and_normalize_sites", lambda sites: [])
    cfg = _base_config()
    cfg["conditioning"] = conditioning
    args = types.SimpleNamespace(once=True, dry_run=True)

    with pytest.raises(ValueError, match=message):
        workflow.run(cfg, args, threading.Event())


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("conditioning", "scratch_dir", "", "scratch_dir"),
        ("conditioning", "scratch_dir", 1, "scratch_dir"),
        ("run", "init_time", "not-a-time", "init_time"),
        ("run", "init_time", 1, "init_time"),
        ("run", "max_consecutive_failures", True, "max_consecutive_failures"),
        ("run", "max_consecutive_failures", 1.5, "max_consecutive_failures"),
        ("run", "nsteps", True, "nsteps"),
        ("run", "cache_retention_hours", float("inf"), "cache_retention_hours"),
        ("bus", "heartbeat_seconds", 0, "heartbeat_seconds"),
        ("bus", "heartbeat_seconds", 101, "heartbeat_seconds"),
        ("bus", "metadata_heartbeat_seconds", 90, "renamed to bus.heartbeat_seconds"),
    ],
)
def test_run_rejects_invalid_runtime_config_before_model_import(
    monkeypatch, section: str, key: str, value: object, message: str
) -> None:
    import src.stormcast.workflow as workflow

    monkeypatch.setattr(workflow, "validate_and_normalize_sites", lambda sites: [])
    cfg = _base_config()
    section_config = cfg[section]
    assert isinstance(section_config, dict)
    section_config[key] = value
    args = types.SimpleNamespace(once=True, dry_run=True)

    with pytest.raises(ValueError, match=message):
        workflow.run(cfg, args, threading.Event())


def test_conditioning_source_manager_closes_replaced_and_current_sources() -> None:
    from src.stormcast.workflow import _ConditioningSourceManager

    first = MagicMock()
    second = MagicMock()
    manager = _ConditioningSourceManager()

    manager.replace(first)
    manager.replace(second)
    first.da.close.assert_called_once()

    manager.close()
    second.da.close.assert_called_once()
    assert manager.current is None


def test_stormcast_one_shot_exhaustion_exits_nonzero(monkeypatch) -> None:
    def deterministic(*a, **k):
        raise FileNotFoundError("HRRR f00 not ready")

    _install_stormcast_fakes(monkeypatch, deterministic)
    from src.stormcast.workflow import _run

    with pytest.raises(SystemExit):
        _run(
            _settings(),
            MagicMock(),  # model
            MagicMock(),  # data
            MagicMock(),  # collector
            MagicMock(),  # coordinator
            MagicMock(),  # publisher
            None,  # transport (dry-run style)
            MagicMock(),  # conditioning_sources
            threading.Event(),
        )


def test_stormcast_one_shot_prunes_after_success(monkeypatch) -> None:
    _install_stormcast_fakes(monkeypatch, lambda *a, **k: None)
    import src.stormcast.workflow as workflow

    pruned = []
    monkeypatch.setattr(workflow, "prune_data_cache", pruned.append)
    publisher = MagicMock()
    publisher.publish_pending.return_value = True
    collector = MagicMock()
    collector.init_ms = 1

    workflow._run(
        _settings(cache_retention_hours=48),
        MagicMock(),
        MagicMock(),
        collector,
        MagicMock(),
        publisher,
        None,
        MagicMock(),
        threading.Event(),
    )

    assert pruned == [48]


def test_stormcast_staging_file_error_propagates(monkeypatch) -> None:
    def deterministic(*a, **k):
        return None  # the IC fetch succeeds

    _install_stormcast_fakes(monkeypatch, deterministic)
    from src.stormcast.workflow import _run

    coordinator = MagicMock()
    coordinator.stage_forecasts.side_effect = FileNotFoundError("schema missing")
    # It must propagate (narrow catch only wraps run.deterministic), not be swallowed + retried.
    with pytest.raises(FileNotFoundError):
        _run(
            _settings(),
            MagicMock(),  # model
            MagicMock(),  # data
            MagicMock(),  # collector
            coordinator,
            MagicMock(),  # publisher
            None,  # transport
            MagicMock(),  # conditioning_sources
            threading.Event(),
        )


def test_resolve_conditioning_tries_older_gfs_cycles(monkeypatch) -> None:
    _install_stormcast_fakes(monkeypatch, lambda *a, **k: None)
    import src.stormcast.workflow as workflow

    conditioning = sys.modules["src.stormcast.conditioning"]
    not_ready = conditioning._ConditioningDataNotReady
    target = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    attempted = []
    source = MagicMock()

    def run_conditioning(*args):
        cycle = args[2]
        attempted.append(cycle)
        if cycle == target:
            raise not_ready
        return source

    conditioning.run_conditioning = run_conditioning
    result = workflow._resolve_conditioning(
        MagicMock(),
        MagicMock(),
        target,
        None,
        6,
        "conditioning.nc",
        "cpu",
        12,
        "sfno",
    )

    assert result is not None
    resolved_source, resolved_cycle = result
    assert resolved_source is source
    assert resolved_cycle == target - timedelta(hours=6)
    assert attempted == [target, target - timedelta(hours=6)]


def test_resolve_conditioning_does_not_hide_other_file_errors(monkeypatch) -> None:
    _install_stormcast_fakes(monkeypatch, lambda *a, **k: None)
    import src.stormcast.workflow as workflow

    conditioning = sys.modules["src.stormcast.conditioning"]
    conditioning.run_conditioning = MagicMock(
        side_effect=FileNotFoundError("output path missing")
    )

    with pytest.raises(FileNotFoundError, match="output path missing"):
        workflow._resolve_conditioning(
            MagicMock(),
            MagicMock(),
            datetime(2026, 1, 2, 12, tzinfo=timezone.utc),
            None,
            6,
            "conditioning.nc",
            "cpu",
            12,
            "sfno",
        )

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

"""run_conditioning writes atomically: the previous cycle's file survives a not-ready cycle."""

from __future__ import annotations

import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _install_fake_deps(monkeypatch, *, raise_not_ready: bool) -> None:
    """Stub torch/xarray (imported at module top) and earth2studio.{run,data,io} (lazy)."""
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "xarray", types.ModuleType("xarray"))

    e2s = types.ModuleType("earth2studio")
    e2s_run = types.ModuleType("earth2studio.run")

    def _deterministic(*args, **kwargs):
        if raise_not_ready:
            raise FileNotFoundError("gfs analysis not uploaded yet")

    e2s_run.deterministic = _deterministic

    e2s_data = types.ModuleType("earth2studio.data")

    class _InferenceOutputSource:
        def __init__(self, path: str) -> None:
            self.path = path

    e2s_data.InferenceOutputSource = _InferenceOutputSource

    e2s_io = types.ModuleType("earth2studio.io")

    class _NetCDF4Backend:
        # Real backend creates and writes the file; the fake writes marker content so the
        # os.replace has something to move.
        def __init__(self, path: str) -> None:
            Path(path).write_text("new-conditioning")

        def close(self) -> None:
            pass

    e2s_io.NetCDF4Backend = _NetCDF4Backend

    monkeypatch.setitem(sys.modules, "earth2studio", e2s)
    monkeypatch.setitem(sys.modules, "earth2studio.run", e2s_run)
    monkeypatch.setitem(sys.modules, "earth2studio.data", e2s_data)
    monkeypatch.setitem(sys.modules, "earth2studio.io", e2s_io)
    # Force a fresh import so the stubbed torch/xarray are picked up.
    monkeypatch.delitem(sys.modules, "src.stormcast.conditioning", raising=False)


_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_run_conditioning_replaces_destination_on_success(
    monkeypatch, tmp_path
) -> None:
    _install_fake_deps(monkeypatch, raise_not_ready=False)
    from src.stormcast.conditioning import run_conditioning

    out = tmp_path / "dsx_conditioning.nc"
    out.write_text("PREVIOUS")
    source = run_conditioning(object(), object(), _START, 3, str(out), "cpu")

    assert out.read_text() == "new-conditioning"  # atomically replaced
    assert source.path == str(out)
    assert list(tmp_path.glob("*.tmp")) == []  # temp consumed by the replace


def test_run_conditioning_preserves_previous_file_when_not_ready(
    monkeypatch, tmp_path
) -> None:
    _install_fake_deps(monkeypatch, raise_not_ready=True)
    from src.stormcast.conditioning import _ConditioningDataNotReady, run_conditioning

    out = tmp_path / "dsx_conditioning.nc"
    out.write_text("PREVIOUS")  # the in-use previous-cycle file
    with pytest.raises(_ConditioningDataNotReady):
        run_conditioning(object(), object(), _START, 3, str(out), "cpu")

    assert (
        out.read_text() == "PREVIOUS"
    )  # untouched: never destroyed for a not-ready cycle
    assert list(tmp_path.glob("*.tmp")) == []  # temp cleaned up on failure

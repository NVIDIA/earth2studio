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

"""main.py dispatcher: workflow registry, default selection, unknown rejection, lazy import."""

from __future__ import annotations

import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock

import main
import pytest


def _write_cfg(tmp_path: Path, text: str) -> str:
    p = tmp_path / "cfg.yaml"
    p.write_text(text)
    return str(p)


def test_unknown_workflow_raises(tmp_path, monkeypatch) -> None:
    cfg = _write_cfg(tmp_path, "workflow: does-not-exist\n")
    import_module = MagicMock()
    register_signal = MagicMock()
    monkeypatch.setattr(main.importlib, "import_module", import_module)
    monkeypatch.setattr(main.signal, "signal", register_signal)
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", cfg])

    with pytest.raises(ValueError, match="unknown workflow"):
        main.main()

    import_module.assert_not_called()
    register_signal.assert_not_called()


@pytest.mark.parametrize(
    ("config_text", "module_name", "cli_flag", "expected_flags"),
    [
        pytest.param(
            "run: {init_time: latest}\n",
            "src.stormcast.workflow",
            "--once",
            (True, False),
            id="default-stormcast",
        ),
        pytest.param(
            "workflow: sfno\nrun: {init_time: latest}\n",
            "src.sfno.workflow",
            "--dry-run",
            (False, True),
            id="explicit-sfno",
        ),
    ],
)
def test_workflow_dispatches_lazily(
    tmp_path,
    monkeypatch,
    config_text: str,
    module_name: str,
    cli_flag: str,
    expected_flags: tuple[bool, bool],
) -> None:
    run = MagicMock()
    workflow_module = MagicMock(run=run)
    import_module = MagicMock(return_value=workflow_module)
    register_signal = MagicMock()
    monkeypatch.setattr(main.importlib, "import_module", import_module)
    monkeypatch.setattr(main.signal, "signal", register_signal)
    cfg_path = _write_cfg(tmp_path, config_text)
    monkeypatch.setattr(sys, "argv", ["main.py", "--config", cfg_path, cli_flag])

    main.main()

    import_module.assert_called_once_with(module_name)
    run.assert_called_once()
    cfg, args, stop = run.call_args.args
    assert run.call_args.kwargs == {}
    assert cfg["run"] == {"init_time": "latest"}
    assert args.config == cfg_path
    assert (args.once, args.dry_run) == expected_flags
    assert not stop.is_set()

    assert [item.args[0] for item in register_signal.call_args_list] == [
        signal.SIGTERM,
        signal.SIGINT,
    ]
    handlers = [item.args[1] for item in register_signal.call_args_list]
    assert handlers[0] is handlers[1]
    handlers[0](signal.SIGTERM, None)
    assert stop.is_set()

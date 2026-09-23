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

"""DSX connector entry point: pick a workflow from config and run it against the DSX bus.

Each workflow is a self-contained model pipeline (its own init trigger, model load, and forecast
loop) that reuses the shared DSX publishing layer. Select one with the ``workflow:`` config key:

    workflow: stormcast-conus   # hourly CONUS site weather (HRRR IC + AI conditioning)
    workflow: sfno              # global medium-range (14-day, 6-hourly) site weather from GFS

Run from the recipe dir (so `src` is importable), in the earth2studio venv:
    python main.py --dry-run     # validate + print one cycle, no bus (still loads the model)
    python main.py --once        # one cycle -> DSX bus, then exit
    python main.py               # persistent: forecast each cadence + heartbeat republishing
                                 # Metadata and the latest forecasts
"""

from __future__ import annotations

import argparse
import importlib
import signal
import sys
import threading
from pathlib import Path
from types import FrameType

import yaml  # type: ignore
from loguru import logger

# Log at INFO to stderr (loguru's default sink is DEBUG); the connector uses loguru throughout,
# matching earth2studio.
logger.remove()
logger.add(sys.stderr, level="INFO")

# Registered workflows: config ``workflow:`` value -> "src.<pkg>.workflow" module exposing run().
# Loaded lazily (each pulls in heavy, model-specific deps) so selecting one never imports another.
_WORKFLOWS = {
    "stormcast-conus": "src.stormcast.workflow",
    "sfno": "src.sfno.workflow",
}


def main() -> None:
    """Parse args, load config, and dispatch to the selected workflow's ``run(cfg, args, stop)``."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(Path(__file__).parent / "cfg" / "dsx-connector.stormcast.yaml"),
    )
    ap.add_argument("--once", action="store_true", help="publish one cycle then exit")
    ap.add_argument(
        "--dry-run", action="store_true", help="validate + print one cycle, no broker"
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    # Default to the original single-workflow behaviour so pre-workflow configs still run.
    name = cfg.get("workflow", "stormcast-conus")
    module = _WORKFLOWS.get(name)
    if module is None:
        raise ValueError(
            f"unknown workflow {name!r}; choose one of {sorted(_WORKFLOWS)}"
        )

    # Shared shutdown signal, installed in the main thread so SIGTERM (container stop) and SIGINT
    # (Ctrl-C) request a COOPERATIVE shutdown: the workflow checks it at each cycle boundary, lets
    # any in-progress confirmed publish complete, then closes the transport. It does NOT interrupt
    # an in-flight model load / rollout, nor force-publish queued-but-unsent bundles (a supervisor
    # restart re-runs the cycle), so shutdown can take until the current cycle finishes. Passed in.
    stop = threading.Event()

    def _on_signal(signum: int, _frame: FrameType | None) -> None:
        logger.info(
            "received signal {}; shutdown requested", signal.Signals(signum).name
        )
        stop.set()

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    importlib.import_module(module).run(cfg, args, stop)


if __name__ == "__main__":
    main()

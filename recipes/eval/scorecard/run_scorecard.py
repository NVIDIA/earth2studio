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

"""Run a scorecard campaign with the eval recipe.

Every stage launches one of the recipe's scripts (predownload.py, main.py,
score.py) under torchrun - no inference, scoring or metric code is
reimplemented here, and nothing is assumed beyond the campaign config.

    python run_scorecard.py fcn3_2025_scorecard
    python run_scorecard.py aurora_2025_scorecard score prune
    python run_scorecard.py fcn3_2025_scorecard infer -- nsteps=4 run_id=smoke

Stages (default: all four, in order):
  predownload   fetch initial conditions + verification from the data source
  infer         run the model, write forecast.zarr
  score         verify against ERA5, write scores.zarr (+ provenance attrs)
  prune         delete forecast.zarr once scores exist (~390 GB -> ~30 MB)

Anything after ``--`` is passed to Hydra verbatim (e.g. smoke overrides).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent  # scorecard root
RECIPE = HERE.parent
STAGES = ("predownload", "infer", "score", "prune")


def launch(entry: str, campaign: str, overrides: list[str], ngpu: int) -> None:
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={ngpu}", "--standalone",
        entry, "--config-dir", str(HERE / "cfg"), f"campaign={campaign}",
        *overrides,
    ]
    subprocess.run(cmd, check=True, cwd=RECIPE)


def stamp_provenance(store: Path) -> None:
    """Record on the artifact itself what produced it, so later exports are
    correct no matter when or where they run."""
    import torch
    import zarr

    prov = {
        "date_scored": dt.date.today().isoformat(),
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda or "n/a"),
        "repo_commit": subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip(),
    }
    if torch.cuda.is_available():
        prov["gpus"] = (
            f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"
            " (single node)"
        )
    zarr.open_group(str(store), mode="a").attrs["provenance"] = prov
    print(f"    stamped provenance onto {store}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("campaign", help="config under cfg/campaign/, e.g. fcn3_2025_scorecard")
    ap.add_argument("stages", nargs="*", help=f"subset of {'/'.join(STAGES)}")
    ap.add_argument("--ngpu", type=int, default=None, help="GPUs (default: all)")
    args, overrides = ap.parse_known_args()
    overrides = [o for o in overrides if o != "--"]
    stages = args.stages or list(STAGES)
    if bad := [s for s in stages if s not in STAGES]:
        ap.error(f"unknown stage(s) {bad}; choose from {STAGES}")

    if socket.gethostname().startswith("login"):
        raise SystemExit("!! REFUSING: login node, not a compute node")
    cfg_path = HERE / "cfg" / "campaign" / f"{args.campaign}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"!! no campaign config {cfg_path}")

    if args.ngpu is None:
        import torch

        args.ngpu = max(1, torch.cuda.device_count())

    # Everything below is read off the campaign config, with command-line
    # overrides taking precedence; the output dir is <project>_<run_id>.
    cfg = yaml.safe_load(cfg_path.read_text())
    ov = dict(o.split("=", 1) for o in overrides if "=" in o)
    project = ov.get("project", cfg["project"])
    run_id = ov.get("run_id", cfg["run_id"])
    out = Path(
        os.environ.setdefault(
            "SCORECARD_OUT", str(HERE / "models" / project / "outputs")
        )
    ) / f"{project}_{run_id}"
    os.environ.setdefault("EARTH2STUDIO_CACHE", "/opt/venv/.e2s_cache")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # scorecard.pipelines.* in the campaign configs resolves from the recipe root.
    os.environ["PYTHONPATH"] = (
        f"{RECIPE}{os.pathsep}{os.environ['PYTHONPATH']}"
        if os.environ.get("PYTHONPATH")
        else str(RECIPE)
    )

    print(f"=== campaign={args.campaign}  ngpu={args.ngpu}  stages={' '.join(stages)}")
    print(f"=== out={out}")

    for stage in stages:
        print(f"\n############### {stage} ###############")
        if stage == "predownload":
            launch("predownload.py", args.campaign, overrides, args.ngpu)
        elif stage == "infer":
            launch("main.py", args.campaign, overrides, args.ngpu)
        elif stage == "score":
            # scores.zarr is ~30 MB read whole, so chunk each array as one
            # blob (per-slice chunks once cost ~1.9 M inodes). Whole-array
            # chunks are what AsyncZarrBackend forbids on iteration axes,
            # hence io_backend=zarr for this store.
            chunks = [
                "output.io_backend=zarr",
                f"output.chunks.time={len(cfg['start_times'])}",
                f"output.chunks.lead_time={int(cfg['nsteps']) + 1}",
            ]
            if int(cfg.get("ensemble_size", 1)) > 1:
                chunks.append(f"+output.chunks.ensemble={cfg['ensemble_size']}")
            launch("score.py", args.campaign, chunks + overrides, args.ngpu)
            stamp_provenance(out / "scores.zarr")
        elif stage == "prune":
            # Check the artifact, not an exit code: an inference stage that
            # died mid-way can still exit 0, and deleting the forecast then
            # makes the run unrecoverable.
            if (out / "scores.zarr").is_dir():
                print(f"    scores present -- deleting {out / 'forecast.zarr'}")
                shutil.rmtree(out / "forecast.zarr", ignore_errors=True)
            else:
                raise SystemExit("!! no scores.zarr -- keeping forecast")

    print(f"\n=== done — {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

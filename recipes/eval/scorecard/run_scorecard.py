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
  infer         run the model.  Scorecard campaigns score ONLINE
                (scoring.mode: online): the inference loop reduces every
                forecast to summary statistics (stats.zarr), this stage
                derives scores.zarr from them, and the run writes no raw
                forecast store.
  score         online: re-derive scores.zarr from stats.zarr.
                offline: read forecast.zarr and score it.
  prune         offline only: delete forecast.zarr once scores exist.
                A no-op for online campaigns, which have nothing to prune.

Anything after ``--`` is passed to Hydra verbatim (e.g. smoke overrides).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent  # scorecard root
RECIPE = HERE.parent
STAGES = ("predownload", "infer", "score", "prune")


def launch(entry: str, campaign: str, overrides: list[str], ngpu: int) -> None:
    """Run one recipe entry point under torchrun with the campaign config.

    Defaults to a single-node --standalone launch. For other topologies set
    TORCHRUN_ARGS (replaces the default topology flags entirely), e.g.
    TORCHRUN_ARGS="--nnodes=4 --rdzv_backend=c10d --rdzv_endpoint=host:29500".
    Stages are resumable, so large campaigns can also be advanced
    incrementally across separate queue allocations.
    """
    topology = os.environ.get(
        "TORCHRUN_ARGS", f"--nproc_per_node={ngpu} --standalone"
    ).split()
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *topology,
        entry,
        "--config-dir",
        str(HERE / "cfg"),
        f"campaign={campaign}",
        *overrides,
    ]
    # cmd is built from sys.executable and our own arguments.
    subprocess.run(cmd, check=True, cwd=RECIPE)  # noqa: S603


def scores_complete(store: Path) -> bool:
    """True when the store exists and carries the provenance stamp that
    stamp_provenance() writes only after scoring finished cleanly."""
    try:
        return "provenance" in json.loads((store / "zarr.json").read_text()).get(
            "attributes", {}
        )
    except OSError:
        return False


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
        "repo_commit": subprocess.run(  # noqa: S603
            [shutil.which("git") or "git", "-C", str(HERE), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
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
    """Run the requested stages of one scorecard campaign."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "campaign", help="config under cfg/campaign/, e.g. fcn3_2025_scorecard"
    )
    ap.add_argument("stages", nargs="*", help=f"subset of {'/'.join(STAGES)}")
    ap.add_argument("--ngpu", type=int, default=None, help="GPUs (default: all)")
    args, extra = ap.parse_known_args()
    # Anything with '=' is a Hydra override, wherever argparse routed it
    # (after a bare `--` they arrive as positionals in `stages`).
    tokens = args.stages + [t for t in extra if t != "--"]
    overrides = [t for t in tokens if "=" in t]
    stages = [t for t in tokens if "=" not in t] or list(STAGES)
    if bad := [s for s in stages if s not in STAGES]:
        ap.error(f"unknown stage(s) {bad}; choose from {STAGES}")

    cfg_path = HERE / "cfg" / "campaign" / f"{args.campaign}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"!! no campaign config {cfg_path}")

    if args.ngpu is None:
        import torch

        args.ngpu = max(1, torch.cuda.device_count())

    # Everything below is read off the campaign config, with command-line
    # overrides taking precedence; the output dir is <project>_<run_id>.
    cfg = yaml.safe_load(cfg_path.read_text())
    # Hydra spellings like +key=value / ~key=value normalize to plain keys.
    ov = {
        k.lstrip("+~"): v for k, v in (o.split("=", 1) for o in overrides if "=" in o)
    }
    project = ov.get("project", cfg["project"])
    run_id = ov.get("run_id", cfg["run_id"])
    out = (
        Path(
            os.environ.setdefault(
                "SCORECARD_OUT", str(HERE / "models" / project / "outputs")
            )
        )
        / f"{project}_{run_id}"
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # scorecard.pipelines.* in the campaign configs resolves from the recipe root.
    os.environ["PYTHONPATH"] = (
        f"{RECIPE}{os.pathsep}{os.environ['PYTHONPATH']}"
        if os.environ.get("PYTHONPATH")
        else str(RECIPE)
    )

    print(f"=== campaign={args.campaign}  ngpu={args.ngpu}  stages={' '.join(stages)}")
    print(f"=== out={out}")

    mode = ov.get("scoring.mode", cfg.get("scoring", {}).get("mode", "offline"))
    online = str(mode).lower() == "online"

    for stage in stages:
        print(f"\n############### {stage} ###############")
        if stage == "predownload":
            launch("predownload.py", args.campaign, overrides, args.ngpu)
        elif stage == "infer":
            launch("main.py", args.campaign, overrides, args.ngpu)
            if online:
                # The online path derives scores.zarr at the end of
                # main.py; stamp it now (the derivation rebuilds the store,
                # so a stamp must always follow the pass that produced it).
                stamp_provenance(out / "scores.zarr")
        elif stage == "score":
            if online:
                # Cheap single-process re-derivation from stats.zarr.
                launch("score.py", args.campaign, overrides, 1)
            else:
                # time=1 chunks: ranks score disjoint ICs, so no
                # shared-chunk writes (races); whole-array elsewhere keeps
                # file counts small.
                chunks = [
                    "output.io_backend=zarr",
                    "output.chunks.time=1",
                    f"output.chunks.lead_time={int(cfg['nsteps']) + 1}",
                ]
                if int(cfg.get("ensemble_size", 1)) > 1:
                    chunks.append(f"+output.chunks.ensemble={cfg['ensemble_size']}")
                launch("score.py", args.campaign, chunks + overrides, args.ngpu)
            stamp_provenance(out / "scores.zarr")
        elif stage == "prune":
            if online and not (out / "forecast.zarr").exists():
                print("    online campaign, no raw store -- nothing to prune")
                continue
            # Gate on the provenance stamp, not on the store existing: a
            # scoring run killed mid-way leaves a partial scores.zarr on
            # disk, and deleting the forecast then makes the run
            # unrecoverable. The stamp is written only after score.py
            # exits cleanly, so it doubles as the completion marker.
            if scores_complete(out / "scores.zarr"):
                print(f"    scores complete -- deleting {out / 'forecast.zarr'}")
                shutil.rmtree(out / "forecast.zarr", ignore_errors=True)
            else:
                raise SystemExit(
                    "!! scores.zarr missing or unstamped (scoring "
                    "incomplete?) -- keeping forecast"
                )

    print(f"\n=== done — {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

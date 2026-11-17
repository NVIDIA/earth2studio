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
reimplemented here, and the campaign config is the only input.  The
campaigns select the recipe's runners (``runner: forecast``) and place
off-grid models onto the shared ERA5 grid with ``output.regrid``.  The
launcher composes the configuration those scripts compose, so the stores it
stamps and prunes are the ones they write.

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from omegaconf import DictConfig

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

    With SCORECARD_PIN_GPU=1 each rank starts through utils/rank_entry.sh,
    which makes only that rank's GPU visible before CUDA starts, and runs the
    entry point under this interpreter.
    """
    topology = os.environ.get(
        "TORCHRUN_ARGS", f"--nproc_per_node={ngpu} --standalone"
    ).split()
    pin = os.environ.get("SCORECARD_PIN_GPU") == "1"
    wrapper = ["--no-python", str(HERE / "utils" / "rank_entry.sh")] if pin else []
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        *topology,
        *wrapper,
        entry,
        "--config-dir",
        str(HERE / "cfg"),
        f"campaign={campaign}",
        *overrides,
    ]
    env = {**os.environ, "SCORECARD_PYTHON": sys.executable} if pin else None
    # cmd is built from sys.executable and our own arguments.
    subprocess.run(cmd, check=True, cwd=RECIPE, env=env)  # noqa: S603


def compose_campaign(campaign: str, overrides: Sequence[str]) -> DictConfig:
    """The configuration the recipe's scripts compose for this launch.

    :func:`launch` starts each script with ``--config-dir`` pointing at the
    scorecard's ``cfg``, next to the recipe's own ``cfg``.  This composes the
    same ``default`` config from the same two directories with the same
    overrides, so a value such as ``scoring.online.scores_store=null`` reads
    as Hydra reads it.  The two directories share no file, so the order
    Hydra searches them in does not change the result.

    Parameters
    ----------
    campaign : str
        Campaign config name under ``cfg/campaign``.
    overrides : Sequence[str]
        Hydra overrides passed to every script.

    Returns
    -------
    DictConfig
        The composed config, interpolations left to resolve on access.
    """
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(RECIPE / "cfg"), version_base=None):
        return compose(
            config_name="default",
            overrides=[
                f'hydra.searchpath=["file://{HERE / "cfg"}"]',
                f"campaign={campaign}",
                *overrides,
            ],
        )


def run_path(path: str) -> Path:
    """A path from the config as the scripts see it: relative to the recipe root."""
    return RECIPE / path


def score_stores(cfg: DictConfig) -> list[Path]:
    """Score stores the scoring pass of ``cfg`` writes, named by the recipe itself.

    Offline: :func:`src.scoring.score_store_path`.  Online: one store per
    reference, :func:`src.online.scores_store_names`.
    """
    from src.campaign import online_scoring
    from src.output import store_path

    if online_scoring(cfg):
        from src.online import scores_store_names

        return [run_path(store_path(cfg, name)) for name in scores_store_names(cfg)]
    from src.scoring import score_store_path

    return [run_path(score_store_path(cfg))]


def source_label(node: Any) -> str | None:
    """Display label of a source config node: the class behind it.

    A composite names every source it reads (``ARCO_ERA5+NCAR_ERA5``), and a
    wrapper taking a ``source`` (a regridding adapter) names the source it
    wraps.  A predownloaded store names nothing: it holds the fields of the
    source that filled it, which the caller falls back to.
    """
    if not isinstance(node, Mapping) or not node.get("_target_"):
        return None
    name = str(node["_target_"]).rsplit(".", 1)[-1]
    if name in ("PredownloadedSource", "PredownloadedForecastSource"):
        return None
    if name == "CompositeSource":
        labels = [source_label(s) for s in (node.get("sources") or {}).values()]
        return "+".join(dict.fromkeys(label for label in labels if label)) or name
    if isinstance(node.get("source"), Mapping):
        return source_label(node["source"]) or name
    return name


def source_labels(
    cfg: Mapping[str, Any], reference: str | None = None
) -> dict[str, Any]:
    """Initial-condition and verification source labels, and the weighting.

    The labels follow the recipe's resolution: a bring-your-own
    ``ic_source`` (or the analyses of a data assimilation model) over
    ``data_source`` for the initial conditions; a bring-your-own
    ``verification_source``, then ``predownload.verification.source``, then
    ``data_source`` for the truth.  A named online reference
    (``scoring.online.verification_sources``) labels the truth with its own
    source, and a ``null`` entry keeps the default.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        A composed campaign config as plain containers.
    reference : str | None, optional
        Online reference the scores compare with, by default the unnamed one.

    Returns
    -------
    dict[str, Any]
        ``ic_source`` and ``verification_source`` when known, and
        ``lat_weights``.
    """
    data = source_label(cfg.get("data_source"))
    da = ((cfg.get("model") or {}).get("da") or {}).get("architecture")
    ic = (str(da).rsplit(".", 1)[-1] if da else None) or source_label(
        cfg.get("ic_source")
    )
    ic = ic or data
    predownload = (cfg.get("predownload") or {}).get("verification") or {}
    verification = (
        source_label(cfg.get("verification_source"))
        or source_label(predownload.get("source"))
        or data
        or ic
    )
    if reference is not None:
        online = (cfg.get("scoring") or {}).get("online") or {}
        node = (online.get("verification_sources") or {}).get(reference)
        if node is not None:
            # The reference's own archive, unknown for a predownloaded store.
            verification = source_label(node)
    labels: dict[str, Any] = {}
    if ic:
        labels["ic_source"] = ic
    if verification:
        labels["verification_source"] = verification
    labels["lat_weights"] = bool((cfg.get("scoring") or {}).get("lat_weights", True))
    return labels


def labelled_score_stores(cfg: DictConfig) -> list[tuple[Path, dict[str, Any]]]:
    """Each score store of :func:`score_stores` with its reference's labels.

    Online scoring writes one store per reference, in
    :func:`src.online.verification_source_names` order; offline scoring
    writes one store against the default truth.
    """
    from omegaconf import OmegaConf
    from src.campaign import online_scoring

    references: list[str | None] = [None]
    if online_scoring(cfg):
        from src.online import verification_source_names

        references = verification_source_names(cfg)
    plain = cast(Mapping[str, Any], OmegaConf.to_container(cfg, resolve=False))
    labels = [source_labels(plain, reference) for reference in references]
    return list(zip(score_stores(cfg), labels, strict=True))


def scores_complete(store: Path) -> bool:
    """True when the store exists and carries the provenance stamp that
    stamp_provenance() writes only after scoring finished cleanly."""
    try:
        return "provenance" in json.loads((store / "zarr.json").read_text()).get(
            "attributes", {}
        )
    except OSError:
        return False


def stamp_provenance(store: Path, sources: Mapping[str, Any] | None = None) -> None:
    """Record on the artifact itself what produced it, so later exports are
    correct no matter when or where they run.  ``sources`` (see
    :func:`source_labels`) goes into its own ``sources`` attribute."""
    import torch
    import zarr

    if not (store / "zarr.json").is_file():
        raise SystemExit(f"!! {store} is missing after scoring; nothing to stamp")
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
    # r+ opens the store scoring wrote and never creates an empty one.
    group = zarr.open_group(str(store), mode="r+")
    group.attrs["provenance"] = prov
    if sources:
        group.attrs["sources"] = dict(sources)
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
    # (after a bare `--` they arrive as positionals in `stages`).  Hydra's
    # bare deletions and additions (`~scoring.events.x`, `+key`) count too.
    tokens = args.stages + [t for t in extra if t != "--"]

    def is_override(token: str) -> bool:
        return "=" in token or token.startswith(("~", "+"))

    overrides = [t for t in tokens if is_override(t)]
    stages = [t for t in tokens if not is_override(t)] or list(STAGES)
    if bad := [s for s in stages if s not in STAGES]:
        ap.error(f"unknown stage(s) {bad}; choose from {STAGES}")

    cfg_path = HERE / "cfg" / "campaign" / f"{args.campaign}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"!! no campaign config {cfg_path}")

    if args.ngpu is None:
        import torch

        args.ngpu = max(1, torch.cuda.device_count())

    # Everything below comes from the configuration the scripts compose
    # (recipe defaults, the campaign and the overrides), through the recipe's
    # own helpers, so the stamp and the prune gate follow the files the
    # scripts write.
    if str(RECIPE) not in sys.path:
        sys.path.insert(0, str(RECIPE))
    from src.campaign import online_scoring
    from src.output import store_path

    cfg = compose_campaign(args.campaign, overrides)
    # The campaigns write under $SCORECARD_OUT, by default the model's
    # models/<project>/outputs; set it before output.path reads it.
    os.environ.setdefault(
        "SCORECARD_OUT", str(HERE / "models" / str(cfg.project) / "outputs")
    )
    out = run_path(str(cfg.output.path))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # scorecard.utils.* targets in the campaign configs (baselines, source
    # adapters) resolve from the recipe root, next to the recipe's src package.
    os.environ["PYTHONPATH"] = (
        f"{RECIPE}{os.pathsep}{os.environ['PYTHONPATH']}"
        if os.environ.get("PYTHONPATH")
        else str(RECIPE)
    )

    print(f"=== campaign={args.campaign}  ngpu={args.ngpu}  stages={' '.join(stages)}")
    print(f"=== out={out}")

    online = online_scoring(cfg)
    # The score store(s) this configuration writes, each stamped with what
    # produced the initial conditions and the truth it scores against; the
    # stamp and the prune gate follow them rather than a fixed scores.zarr.
    labelled = labelled_score_stores(cfg)
    stores = [store for store, _ in labelled]
    forecast = run_path(store_path(cfg))

    for stage in stages:
        print(f"\n############### {stage} ###############")
        if stage == "predownload":
            launch("predownload.py", args.campaign, overrides, args.ngpu)
        elif stage == "infer":
            launch("main.py", args.campaign, overrides, args.ngpu)
            if online:
                # The online path derives the scores store at the end of
                # main.py; stamp it now (the derivation rebuilds the store,
                # so a stamp must always follow the pass that produced it).
                for store, sources in labelled:
                    stamp_provenance(store, sources)
        elif stage == "score":
            if online:
                # Cheap single-process re-derivation from stats.zarr.
                launch("score.py", args.campaign, overrides, 1)
            else:
                # score.py lays out the score store itself (one time per
                # chunk, ranks scoring disjoint initial conditions).
                launch("score.py", args.campaign, overrides, args.ngpu)
            for store, sources in labelled:
                stamp_provenance(store, sources)
        elif stage == "prune":
            if online and not forecast.exists():
                print("    online campaign, no raw store -- nothing to prune")
                continue
            # Gate on the provenance stamp, not on the store existing: a
            # scoring run killed mid-way leaves a partial score store on
            # disk, and deleting the forecast then makes the run
            # unrecoverable. The stamp is written only after score.py
            # exits cleanly, so it doubles as the completion marker.
            if all(scores_complete(store) for store in stores):
                print(f"    scores complete -- deleting {forecast}")
                shutil.rmtree(forecast, ignore_errors=True)
            else:
                raise SystemExit(
                    f"!! {', '.join(s.name for s in stores)} missing or unstamped "
                    "(scoring incomplete?) -- keeping forecast"
                )

    print(f"\n=== done — {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

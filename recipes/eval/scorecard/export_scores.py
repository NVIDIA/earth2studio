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

"""Export the scorecard JSON (eval_scores_<model>.json) from a scored run.

The JSON file is the hand-off between this recipe (which needs a GPU cluster and
the raw zarr stores) and the documentation (which only needs numbers): metric
curves per variable and lead time, units, variable grouping, and run
provenance. Values are aggregated over initial conditions with the recipe's
own ``src.report.aggregation`` -- nothing is reimplemented here.

    python export_scores.py fcn3 aurora          # -> exports/eval_scores_<model>.json
    python export_scores.py fcn3 --docs          # also copy into docs/_static/scorecard/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr

HERE = Path(__file__).resolve().parent  # scorecard root
sys.path.insert(0, str(HERE.parent))  # <recipe> -> src.report.*

from src.report.aggregation import aggregate_over_time  # noqa: E402

# The docs copy lives under _static so the interactive plot can fetch it
# directly in the browser -- one copy of the numbers, none embedded in HTML.
DOCS_STATIC = HERE.parent.parent.parent / "docs" / "_static" / "scorecard"
EXPORTS = HERE / "exports"

# metric key -> (display label, stored source metric; None = derived).
# Aggregation over ICs is delegated to src.report.aggregation, which knows
# rmse aggregates as sqrt(mean(x^2)) and ensemble_mean_mse as sqrt(mean(x)).
METRICS: dict[str, tuple[str, str | None]] = {
    "rmse": ("RMSE", "rmse"),
    "mae": ("MAE", "mae"),
    "lsd": ("Log spectral distance", "lsd"),
    "acc": ("ACC", "acc"),
    "ensemble_mean_mse": ("RMSE (ensemble mean)", "ensemble_mean_mse"),
    "crps": ("CRPS", "crps"),
    "ensemble_variance": ("Spread", "ensemble_variance"),
    "spread_skill": ("Spread / Skill", None),
}
# Deterministic metrics score ONE field: on an ensemble model take a single
# member, not the average of member scores (which describes a forecast nobody
# issued). The ensemble-as-a-whole is what ensemble_mean_mse / crps measure.
SINGLE_MEMBER = {"rmse", "mae", "acc", "lsd"}

_GROUPS = {
    "z": "Geopotential",
    "t": "Temperature",
    "u": "U wind",
    "v": "V wind",
    "q": "Specific humidity",
}
_UNITS_EXACT = {"msl": "Pa", "sp": "Pa", "tcwv": "kg m⁻²"}


def group_of(var: str) -> str:
    """Display group for a variable (Geopotential, Temperature, Surface...)."""
    m = re.fullmatch(r"([a-z]+)(\d+)", var)
    if m and m.group(1) in _GROUPS and var not in ("u10m", "v10m"):
        return _GROUPS[m.group(1)]
    return "Surface"


def unit_for(var: str) -> str:
    """Display unit for a variable, from its naming convention."""
    if var in _UNITS_EXACT:
        return _UNITS_EXACT[var]
    for prefix, unit in (
        ("z", "m² s⁻²"),
        ("u", "m s⁻¹"),
        ("v", "m s⁻¹"),
        ("q", "kg kg⁻¹"),
        ("t", "K"),
    ):
        if var.startswith(prefix):
            return unit
    return ""


def sort_key(v: str) -> tuple:
    """Surface first, then each group by ascending pressure level."""
    order = {"z": 2, "t": 3, "u": 4, "v": 5, "q": 6}
    m = re.fullmatch(r"([a-z]+)(\d+)", v)
    if m and m.group(1) in order and v not in ("u10m", "v10m"):
        return (order[m.group(1)], int(m.group(2)), v)
    return (1, 0, v)


def curve(ds: xr.Dataset, source: str, var: str, single: bool) -> np.ndarray | None:
    """Aggregated-over-ICs curve for one stored metric/variable."""
    key = f"{source}__{var}"
    if key not in ds:
        return None
    da = ds[key]
    if "ensemble" in da.dims:
        da = da.isel(ensemble=0) if single else da.mean("ensemble")
    da = da.transpose("time", "lead_time")  # never trust stored dim order
    out = np.asarray(aggregate_over_time(da, source).values, dtype=float)
    if source == "ensemble_variance":
        # Stored as variance; sqrt after the IC average so it carries the
        # variable's units and is comparable with RMSE.
        out = np.sqrt(np.clip(out, 0, None))
    return out


def r5(a: np.ndarray) -> list:
    """Round to 5 significant digits; non-finite values become None."""
    return [None if not np.isfinite(x) else float(f"{x:.5g}") for x in a.ravel()]


def provenance(run: Path) -> dict:
    """Provenance for the export.

    Preferred source is what run_scorecard.py stamped onto scores.zarr right
    after scoring -- that describes the environment which actually produced
    the numbers, correct no matter when or where the export happens. Stores
    that predate stamping fall back to capturing the EXPORT environment,
    marked provenance_source: export-time so the difference is visible.
    """
    import contextlib
    import datetime as dt
    import shutil
    import subprocess

    prov: dict = {
        "scores_written": dt.datetime.fromtimestamp(
            (run / "scores.zarr").stat().st_mtime
        ).strftime("%Y-%m-%d"),
        "exported": dt.date.today().isoformat(),
    }
    # Env probing is best effort throughout: the export must never die on it.
    stamped: dict = {}
    with contextlib.suppress(Exception):
        import zarr

        stamped = dict(
            zarr.open_group(str(run / "scores.zarr"), mode="r").attrs.get("provenance")
            or {}
        )
    if stamped:
        prov.update(stamped)
        prov["provenance_source"] = "run"
    else:
        prov["python"] = sys.version.split()[0]
        with contextlib.suppress(Exception):
            import torch

            prov["torch"] = torch.__version__
            prov["cuda"] = torch.version.cuda or "n/a"
            if torch.cuda.is_available():
                prov["gpus"] = (
                    f"{torch.cuda.device_count()} x "
                    f"{torch.cuda.get_device_name(0)} (single node)"
                )
        with contextlib.suppress(Exception):
            prov["repo_commit"] = subprocess.run(  # noqa: S603
                [shutil.which("git") or "git", "-C", str(HERE), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        prov["provenance_source"] = "export-time"
    # torch.__version__ is a str subclass; normalise everything to plain str.
    return {k: str(v) for k, v in prov.items()}


def export(model: str, run: Path) -> dict:
    """Assemble the full export document for one scored run."""
    ds = xr.open_zarr(run / "scores.zarr")
    stored = {k.split("__")[0] for k in ds.data_vars if "__" in k}
    variables = sorted(
        {k.split("__", 1)[1] for k in ds.data_vars if "__" in k}, key=sort_key
    )
    lead_h = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int).tolist()
    times = [str(t)[:16].replace("T", " ") for t in ds.time.values]
    n_ens = int(ds.sizes.get("ensemble", 1))

    metrics: dict = {}
    for key, (label, source) in METRICS.items():
        if key == "spread_skill":
            if not {"ensemble_mean_mse", "ensemble_variance"} <= stored:
                continue
        elif source is None or source not in stored:
            continue
        values = {}
        for var in variables:
            if key == "spread_skill":
                sp = curve(ds, "ensemble_variance", var, single=False)
                sk = curve(ds, "ensemble_mean_mse", var, single=False)
                if sp is None or sk is None:
                    continue
                c = np.divide(sp, sk, out=np.full_like(sp, np.nan), where=sk > 0)
            else:
                if source is None:  # unreachable; narrows the type for mypy
                    continue
                c = curve(ds, source, var, single=key in SINGLE_MEMBER)
                if c is None:
                    continue
            # Drop lead 0: it is the initial condition, identically zero.
            values[var] = r5(c[1:] if lead_h and lead_h[0] == 0 else c)
        if values:
            metrics[key] = {"label": label, "values": values}
            if key == "lsd":
                metrics[key]["unit"] = "dB"  # metric unit overrides var unit
    ds.close()

    out_leads = lead_h[1:] if lead_h and lead_h[0] == 0 else lead_h
    return {
        "model": model,
        "run": run.name,
        "kind": "prob" if n_ens > 1 else "det",
        "members": n_ens,
        "initial_conditions": times,
        "lead_hours": out_leads,
        "variables": variables,
        "units": {v: unit_for(v) for v in variables},
        "variable_groups": {v: group_of(v) for v in variables},
        "aggregation": (
            "mean over initial_conditions (sqrt-of-mean-square for rmse and "
            "ensemble_mean_mse, sqrt of mean variance for ensemble_variance); "
            "lead 0 is excluded -- it is the initial condition, identically 0"
        ),
        "provenance": provenance(run),
        "metrics": metrics,
    }


def main() -> int:
    """Export eval_scores_<model>.json for each requested model."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", help="model names, e.g. fcn3 aurora")
    ap.add_argument(
        "--run",
        metavar="PATH",
        help="run directory holding scores.zarr; default "
        "models/<model>/outputs/<model>_2025_scorecard",
    )
    ap.add_argument(
        "--docs",
        action="store_true",
        help=f"also copy each JSON to {DOCS_STATIC}/",
    )
    args = ap.parse_args()

    EXPORTS.mkdir(exist_ok=True)
    for model in args.models:
        run = (
            Path(args.run)
            if args.run
            else (HERE / "models" / model / "outputs" / f"{model}_2025_scorecard")
        )
        if not (run / "scores.zarr").exists():
            raise SystemExit(f"{model}: no scores.zarr under {run}")
        # Plain minified JSON: the docs plot fetches and JSON.parses it
        # directly, servers compress application/json natively, and no
        # whitespace is spent on indentation (pretty-print on demand). JSON
        # has no comments, so license and generator live as leading keys.
        doc = {
            "license": "SPDX-License-Identifier: Apache-2.0. Copyright (c) "
            "2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            "generated_by": "recipes/eval/scorecard/export_scores.py "
            "-- do not hand-edit",
        }
        doc.update(export(model, run))
        text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False) + "\n"
        out = EXPORTS / f"eval_scores_{model}.json"
        out.write_text(text)
        print(f"wrote {out}  ({out.stat().st_size / 1e3:.0f} kB)")
        if args.docs:
            DOCS_STATIC.mkdir(parents=True, exist_ok=True)
            dst = DOCS_STATIC / f"eval_scores_{model}.json"
            dst.write_text(text)
            print(f"  ->  {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

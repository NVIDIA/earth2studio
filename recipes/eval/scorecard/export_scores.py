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

The JSON files are the hand-off between this recipe (which needs a GPU cluster
and the zarr stores) and the documentation (which only needs numbers): metric
curves per variable and lead time, units, variable grouping, and run
provenance. Values are aggregated over initial conditions with the recipe's
own ``src.report.aggregation`` -- nothing is reimplemented here.

Runs scored with regional splits (``scoring.regions``) additionally get
one ``eval_scores_<model>_region_<name>.json`` per non-global region; runs
spanning more than one calendar month get ``eval_scores_<model>_monthly.json``
(individual months plus DJF/MAM/JJA/SON season blocks); runs with more than
one initialization hour get ``eval_scores_<model>_hourly.json`` (per synoptic
hour); and every run gets ``eval_scores_<model>_heatmap.json`` with per-IC
skill grids for every scored variable.  The docs plot fetches these lazily
when the reader first touches the matching control, so the first paint only
pays for the main file.

    python export_scores.py fcn3 aurora          # -> exports/eval_scores_<model>*.json
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

# metric key -> (display label, stored source candidates; empty = derived).
# The first stored name present wins: offline runs store `rmse` directly,
# online runs store per-member `mse` (aggregation then takes the sqrt).
# Aggregation over ICs is delegated to src.report.aggregation, which knows
# rmse aggregates as sqrt(mean(x^2)) and mse/ensemble_mean_mse as
# sqrt(mean(x)).
METRICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "rmse": ("RMSE", ("rmse", "mse")),
    "mae": ("MAE", ("mae",)),
    "lsd": ("Log spectral distance", ("lsd",)),
    "acc": ("ACC", ("acc",)),
    "ensemble_mean_mse": ("RMSE (ensemble mean)", ("ensemble_mean_mse",)),
    "crps": ("CRPS", ("crps",)),
    "ensemble_variance": ("Spread", ("ensemble_variance",)),
    "spread_skill": ("Spread / Skill", ()),
}
# Deterministic metrics score ONE field: on an ensemble model take a single
# member, not the average of member scores (which describes a forecast nobody
# issued). The ensemble-as-a-whole is what ensemble_mean_mse / crps measure.
SINGLE_MEMBER = {"rmse", "mae", "acc", "lsd"}

# Spectral metrics are global by construction: the online path stores them
# only in the whole-grid region, so the regional exports skip them.
GLOBAL_ONLY = {"lsd"}

MONTHS = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)  # fmt: skip

# Three-month season blocks, exported alongside the individual months
# (calendar-year grouping: DJF is Dec+Jan+Feb of the campaign year).
SEASONS: dict[str, tuple[int, ...]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}

# IC heatmap export: per-IC skill (no aggregation over initial conditions)
# for every scored variable — the docs plot renders it as an IC x lead
# grid.  Whole grid only, and single-member for the deterministic metric,
# matching the headline curves.  Lazily fetched, so the file size (a few
# MB minified at 72 variables) never touches the first page paint.
HEATMAP_METRICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "rmse": ("RMSE", ("rmse", "mse")),
    "crps": ("CRPS", ("crps",)),
}

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


def curve(
    ds: xr.Dataset,
    sources: tuple[str, ...],
    var: str,
    single: bool,
    region: str | None = None,
    months: tuple[int, ...] | None = None,
    hour: int | None = None,
) -> np.ndarray | None:
    """Aggregated-over-ICs curve for one stored metric/variable.

    Parameters
    ----------
    region : str | None
        Regional split to select when the store carries a ``region`` axis
        (online runs).  ``None`` selects the whole-grid split.
    months : tuple[int, ...] | None
        Restrict the IC average to these calendar months (1-12) — one
        month or a season block.
    hour : int | None
        Restrict the IC average to initial conditions at this synoptic
        hour (0/6/12/18).
    """
    source = next((s for s in sources if f"{s}__{var}" in ds), None)
    if source is None:
        return None
    da = ds[f"{source}__{var}"]
    if "region" in da.dims:
        labels = [str(r) for r in da.region.values]
        want = region or ("global" if "global" in labels else labels[0])
        if want not in labels:
            return None
        da = da.sel(region=want)
    elif region is not None:
        return None  # region requested but the store has no splits
    if "ensemble" in da.dims:
        da = da.isel(ensemble=0) if single else da.mean("ensemble")
    if months is not None:
        da = da.sel(time=da.time.dt.month.isin(list(months)))
        if da.sizes.get("time", 0) == 0:
            return None
    if hour is not None:
        da = da.sel(time=da.time.dt.hour == hour)
        if da.sizes.get("time", 0) == 0:
            return None
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


def data_sources(run: Path) -> dict:
    """Initial-condition and verification source class names, read from the
    campaign config that matches the run directory name (best effort)."""
    import contextlib

    out: dict = {}
    with contextlib.suppress(Exception):
        import yaml

        cfg = yaml.safe_load(
            (HERE / "cfg" / "campaign" / f"{run.name}.yaml").read_text()
        )
        ic = cfg.get("data_source", {}).get("_target_", "")
        verif = (
            cfg.get("predownload", {})
            .get("verification", {})
            .get("source", {})
            .get("_target_", "")
            or ic
        )
        if ic:
            out["ic_source"] = ic.rsplit(".", 1)[-1]
        if verif:
            out["verification_source"] = verif.rsplit(".", 1)[-1]
    return out


def build_metrics(
    ds: xr.Dataset,
    variables: list[str],
    lead_h: list[int],
    region: str | None = None,
    months: tuple[int, ...] | None = None,
    hour: int | None = None,
) -> dict:
    """The ``metrics`` block for one (region, months, hour) slice of a run."""
    stored = {k.split("__")[0] for k in ds.data_vars if "__" in k}

    metrics: dict = {}
    for key, (label, sources) in METRICS.items():
        if key == "spread_skill":
            if not {"ensemble_mean_mse", "ensemble_variance"} <= stored:
                continue
        elif not any(s in stored for s in sources):
            continue
        if region is not None and key in GLOBAL_ONLY:
            continue
        values = {}
        for var in variables:
            if key == "spread_skill":
                sp = curve(
                    ds,
                    ("ensemble_variance",),
                    var,
                    False,
                    region=region,
                    months=months,
                    hour=hour,
                )
                sk = curve(
                    ds,
                    ("ensemble_mean_mse",),
                    var,
                    False,
                    region=region,
                    months=months,
                    hour=hour,
                )
                if sp is None or sk is None:
                    continue
                c = np.divide(sp, sk, out=np.full_like(sp, np.nan), where=sk > 0)
            else:
                c = curve(
                    ds,
                    sources,
                    var,
                    single=key in SINGLE_MEMBER,
                    region=region,
                    months=months,
                    hour=hour,
                )
                if c is None:
                    continue
            # Drop lead 0: it is the initial condition, identically zero.
            vals = r5(c[1:] if lead_h and lead_h[0] == 0 else c)
            if any(v is not None for v in vals):
                values[var] = vals
        if values:
            metrics[key] = {"label": label, "values": values}
            if key == "lsd":
                metrics[key]["unit"] = "dB"  # metric unit overrides var unit
    return metrics


def build_heatmap(ds: xr.Dataset, variables: list[str], lead_h: list[int]) -> dict:
    """Per-IC skill grids for every scored variable (the docs' IC heatmap).

    One row per initial condition, one column per lead time — the raw
    per-IC scores, deliberately NOT aggregated, so a bad forecast bust or
    a seasonal stripe stays visible.  Whole grid, member 0 for the
    deterministic metric, lead 0 dropped like everywhere else.
    """
    drop0 = bool(lead_h and lead_h[0] == 0)
    metrics: dict = {}
    for key, (label, sources) in HEATMAP_METRICS.items():
        source = None
        values: dict = {}
        for var in variables:
            source = next((s for s in sources if f"{s}__{var}" in ds), None)
            if source is None:
                continue
            da = ds[f"{source}__{var}"]
            if "region" in da.dims:
                labels = [str(r) for r in da.region.values]
                da = da.sel(region="global" if "global" in labels else labels[0])
            if "ensemble" in da.dims:
                da = da.isel(ensemble=0)
            grid = np.asarray(da.transpose("time", "lead_time").values, dtype=float)
            if source in ("mse", "ensemble_mean_mse"):
                # Stored as per-IC MSE online; the heatmap shows RMSE.
                grid = np.sqrt(np.clip(grid, 0, None))
            values[var] = [r5(row[1:] if drop0 else row) for row in grid]
        if values:
            metrics[key] = {"label": label, "values": values}
    return metrics


def export(model: str, run: Path) -> tuple[dict, dict[str, dict]]:
    """Assemble the export documents for one scored run.

    Returns the main document plus sibling split documents keyed by file
    suffix (``region_<name>`` / ``monthly``) — the plot fetches those
    lazily, so the first paint never pays for the splits.
    """
    ds = xr.open_zarr(run / "scores.zarr")
    variables = sorted(
        {k.split("__", 1)[1] for k in ds.data_vars if "__" in k}, key=sort_key
    )
    lead_h = (ds.lead_time.values / np.timedelta64(1, "h")).astype(int).tolist()
    times = [str(t)[:16].replace("T", " ") for t in ds.time.values]
    n_ens = int(ds.sizes.get("ensemble", 1))
    regions = [str(r) for r in ds.region.values] if "region" in ds.dims else []
    months_present = sorted({int(m) for m in ds.time.dt.month.values})

    out_leads = lead_h[1:] if lead_h and lead_h[0] == 0 else lead_h
    doc = {
        "model": model,
        "run": run.name,
        "kind": "prob" if n_ens > 1 else "det",
        "members": n_ens,
        "initial_conditions": times,
        "lead_hours": out_leads,
        "variables": variables,
        **data_sources(run),
        "units": {v: unit_for(v) for v in variables},
        "variable_groups": {v: group_of(v) for v in variables},
        "aggregation": (
            "mean over initial_conditions (sqrt-of-mean-square for rmse and "
            "ensemble_mean_mse, sqrt of mean variance for ensemble_variance); "
            "lead 0 is excluded -- it is the initial condition, identically 0"
        ),
        "provenance": provenance(run),
        "metrics": build_metrics(ds, variables, lead_h),
    }

    # Split documents: one per non-global region (the global split IS the
    # main document) and one holding the monthly breakdown (whole grid).
    splits: dict[str, dict] = {}
    for region in regions:
        if region == "global":
            continue
        metrics = build_metrics(ds, variables, lead_h, region=region)
        if metrics:
            splits[f"region_{region}"] = {"region": region, "metrics": metrics}
    if len(months_present) > 1:
        # Season blocks first, then the individual months, in one file —
        # a season is just a three-month IC group through the same path.
        monthly: dict = {}
        for name, season_months in SEASONS.items():
            if not set(season_months) & set(months_present):
                continue
            metrics = build_metrics(ds, variables, lead_h, months=season_months)
            if metrics:
                monthly[name] = metrics
        for m in months_present:
            metrics = build_metrics(ds, variables, lead_h, months=(m,))
            if metrics:
                monthly[MONTHS[m - 1]] = metrics
        if monthly:
            splits["monthly"] = {"months": list(monthly), "metrics_by_month": monthly}

    # Time-of-day split: ICs grouped by synoptic initialization hour.
    hours_present = sorted({int(h) for h in ds.time.dt.hour.values})
    if len(hours_present) > 1:
        hourly: dict = {}
        for h in hours_present:
            metrics = build_metrics(ds, variables, lead_h, hour=h)
            if metrics:
                hourly[f"{h:02d}Z"] = metrics
        if hourly:
            splits["hourly"] = {"hours": list(hourly), "metrics_by_hour": hourly}

    heatmap = build_heatmap(ds, variables, lead_h)
    if heatmap:
        splits["heatmap"] = {"initial_conditions": times, "metrics": heatmap}

    if regions:
        doc["regions"] = regions
    if "monthly" in splits:
        doc["has_monthly"] = True
    if "hourly" in splits:
        doc["has_hourly"] = True
    if heatmap:
        doc["has_heatmap"] = True
    ds.close()
    return doc, splits


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
    ap.add_argument(
        "--main-only",
        action="store_true",
        help="skip the split files (regions/monthly/hourly/heatmap) — used "
        "for baseline runs, where the docs plot only reads the main curves",
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
        header = {
            "license": "SPDX-License-Identifier: Apache-2.0. Copyright (c) "
            "2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
            "generated_by": "recipes/eval/scorecard/export_scores.py "
            "-- do not hand-edit",
        }
        main_doc, splits = export(model, run)
        if args.main_only:
            # Baseline exports: the plot reads only the main curves,
            # and the flags below would otherwise advertise split files
            # that were never written.
            for key in ("regions", "has_monthly", "has_hourly", "has_heatmap"):
                main_doc.pop(key, None)
            splits = {}
        documents = {f"eval_scores_{model}.json": main_doc}
        # Split files sit beside the main export; the plot fetches them
        # lazily when the reader first selects a region or the monthly tab.
        for suffix, split_doc in splits.items():
            documents[f"eval_scores_{model}_{suffix}.json"] = split_doc
        for filename, body in documents.items():
            doc = dict(header)
            doc.update(body)
            text = json.dumps(doc, separators=(",", ":"), ensure_ascii=False) + "\n"
            out = EXPORTS / filename
            out.write_text(text)
            print(f"wrote {out}  ({out.stat().st_size / 1e3:.0f} kB)")
            if args.docs:
                DOCS_STATIC.mkdir(parents=True, exist_ok=True)
                dst = DOCS_STATIC / filename
                dst.write_text(text)
                print(f"  ->  {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

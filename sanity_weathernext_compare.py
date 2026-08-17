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
"""Compare existing GraphCast/GenCast wrappers across two checkouts.

By default this uses real ARCO inputs, checks shape/coordinate/finite output
compatibility, reports numerical drift metrics, and plots selected fields.

Example
-------
python sanity_weathernext_compare.py \
    --baseline-repo /tmp/earth2studio-main \
    --candidate-repo . \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

VARIANTS = {
    "graphcast-small": ("earth2studio.models.px", "GraphCastSmall", {}),
    "graphcast-operational": (
        "earth2studio.models.px",
        "GraphCastOperational",
        {},
    ),
    "gencast-mini": (
        "earth2studio.models.px",
        "GenCastMini",
        {"jit_compile": False, "seed": 0},
    ),
}

DEFAULT_PLOT_VARS = ("t2m", "u10m", "v10m", "msl", "z500")

RUNNER = r"""
import importlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from earth2studio.data import ARCO, Random, fetch_data

(
    variant,
    module_name,
    class_name,
    load_kwargs_json,
    device,
    seed,
    source_name,
    valid_time,
    plot_vars_json,
    fields_path,
) = sys.argv[1:]
np.random.seed(int(seed))
torch.manual_seed(int(seed))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(seed))

module = importlib.import_module(module_name)
model_cls = getattr(module, class_name)
package = model_cls.load_default_package()
model = model_cls.load_model(package, **json.loads(load_kwargs_json)).to(device)

# Model construction may consume RNG differently across upstream packages.
# Reset here so both checkouts receive the same synthetic input data.
np.random.seed(int(seed))
torch.manual_seed(int(seed))
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(int(seed))

coords_in = model.input_coords()
domain = OrderedDict(
    (key, value)
    for key, value in coords_in.items()
    if key not in {"batch", "time", "lead_time", "variable"}
)
lead_time = coords_in["lead_time"]
variable = coords_in["variable"]
time = np.array([np.datetime64(valid_time)])

if source_name == "arco":
    source = ARCO(cache=True, verbose=True)
    interp_domain = domain.copy()
    if "lat" in domain and "_lat" not in interp_domain:
        interp_domain["_lat"] = domain["lat"]
    if "lon" in domain and "_lon" not in interp_domain:
        interp_domain["_lon"] = domain["lon"]
    fetch_kwargs = {"interp_to": interp_domain}
elif source_name == "random":
    source = Random(domain)
    fetch_kwargs = {}
else:
    raise ValueError(f"Unknown source: {source_name}")

x, coords = fetch_data(
    source, time, variable, lead_time, device=device, **fetch_kwargs
)
if "_lat" in coords and "_lon" in coords:
    normalized_coords = OrderedDict()
    for key, value in coords.items():
        if key in {"lat", "lon"}:
            continue
        if key == "_lat":
            normalized_coords["lat"] = value
        elif key == "_lon":
            normalized_coords["lon"] = value
        else:
            normalized_coords[key] = value
    coords = normalized_coords
with torch.no_grad():
    out, out_coords = model(x, coords)
out = out.detach().cpu().float()
flat = out.reshape(-1)
finite_mask = torch.isfinite(flat)
valid = flat[finite_mask]
if valid.numel() == 0:
    valid = torch.zeros(1, dtype=flat.dtype)
probe_idx = np.linspace(0, flat.numel() - 1, min(2048, flat.numel()), dtype=np.int64)
probe_mask = finite_mask[torch.from_numpy(probe_idx)].numpy()
probe = flat[torch.from_numpy(probe_idx)].numpy()

def encode_coord(value):
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.datetime64) or np.issubdtype(array.dtype, np.timedelta64):
        return [str(v) for v in array.tolist()]
    return array.tolist()

def extract_field(var_name):
    variables = list(np.asarray(out_coords["variable"]).astype(str))
    if var_name not in variables:
        return None
    dims = list(out_coords.keys())
    index = []
    for dim in dims:
        if dim in {"lat", "lon"}:
            index.append(slice(None))
        elif dim == "variable":
            index.append(variables.index(var_name))
        else:
            index.append(0)
    return out.numpy()[tuple(index)]

plot_vars = json.loads(plot_vars_json)
field_data = {
    "lat": np.asarray(out_coords["lat"]),
    "lon": np.asarray(out_coords["lon"]),
}
saved_vars = []
for name in plot_vars:
    field = extract_field(name)
    if field is None:
        continue
    field_data[f"field_{name}"] = field
    saved_vars.append(name)
np.savez_compressed(fields_path, **field_data)

summary = {
    "variant": variant,
    "repo": str(Path.cwd()),
    "shape": list(out.shape),
    "coords": {key: encode_coord(value) for key, value in out_coords.items()},
    "finite": bool(finite_mask.all().item()),
    "finite_count": int(finite_mask.sum().item()),
    "numel": flat.numel(),
    "mean": float(valid.mean().item()),
    "std": float(valid.std(unbiased=False).item()),
    "sum": float(valid.sum().item()),
    "probe": probe.tolist(),
    "probe_finite": probe_mask.tolist(),
    "fields_path": fields_path,
    "saved_vars": saved_vars,
}
print(json.dumps(summary, sort_keys=True))
"""


def _run_variant(
    repo: Path,
    variant: str,
    device: str,
    seed: int,
    timeout: int,
    source: str,
    time: str,
    plot_vars: list[str],
    fields_path: Path,
) -> dict[str, Any]:
    module_name, class_name, load_kwargs = VARIANTS[variant]
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(RUNNER)
        runner = Path(f.name)
    try:
        cmd = [
            "uv",
            "run",
            "--project",
            str(repo),
            "--extra",
            "graphcast",
            "--extra",
            "gencast",
            "python",
            str(runner),
            variant,
            module_name,
            class_name,
            json.dumps(load_kwargs),
            device,
            str(seed),
            source,
            time,
            json.dumps(plot_vars),
            str(fields_path),
        ]
        env = os.environ.copy()
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        proc = subprocess.run(  # noqa: S603
            cmd,
            cwd=repo,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    finally:
        runner.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{repo} {variant} failed with exit {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def _compare(
    base: dict[str, Any],
    cand: dict[str, Any],
    rtol: float,
    atol: float,
    strict_values: bool,
) -> dict[str, float]:
    variant = base["variant"]
    if base["shape"] != cand["shape"]:
        raise AssertionError(
            f"{variant}: shape differs {base['shape']} != {cand['shape']}"
        )
    if base["coords"] != cand["coords"]:
        raise AssertionError(f"{variant}: output coordinates differ")
    if base["finite_count"] != cand["finite_count"] or base["numel"] != cand["numel"]:
        raise AssertionError(
            f"{variant}: finite counts differ "
            f"{base['finite_count']}/{base['numel']} != "
            f"{cand['finite_count']}/{cand['numel']}"
        )

    base_probe = np.asarray(base["probe"], dtype=np.float64)
    cand_probe = np.asarray(cand["probe"], dtype=np.float64)
    if base.get("probe_finite") != cand.get("probe_finite"):
        raise AssertionError(f"{variant}: probe finite masks differ")
    diff = cand_probe - base_probe
    max_abs = float(np.nanmax(np.abs(diff)))
    mean_abs = float(np.nanmean(np.abs(diff)))
    rel_l2 = float(
        np.linalg.norm(np.nan_to_num(diff))
        / max(np.linalg.norm(np.nan_to_num(base_probe)), np.finfo(np.float64).eps)
    )
    metrics = {
        "mean_delta": float(cand["mean"] - base["mean"]),
        "std_delta": float(cand["std"] - base["std"]),
        "sum_delta": float(cand["sum"] - base["sum"]),
        "probe_max_abs": max_abs,
        "probe_mean_abs": mean_abs,
        "probe_rel_l2": rel_l2,
    }

    if strict_values:
        for key in ("mean", "std", "sum"):
            if not math.isclose(base[key], cand[key], rel_tol=rtol, abs_tol=atol):
                raise AssertionError(
                    f"{variant}: {key} differs {base[key]} != {cand[key]}"
                )
        if not np.allclose(
            base_probe, cand_probe, rtol=rtol, atol=atol, equal_nan=True
        ):
            raise AssertionError(
                f"{variant}: probe values differ, max abs err {max_abs}"
            )

    return metrics


def _plot_fields(
    variant: str,
    base: dict[str, Any],
    cand: dict[str, Any],
    plot_vars: list[str],
    plot_dir: Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    base_npz = np.load(base["fields_path"])
    cand_npz = np.load(cand["fields_path"])
    available = [
        var
        for var in plot_vars
        if f"field_{var}" in base_npz.files and f"field_{var}" in cand_npz.files
    ]
    if not available:
        return None

    fig, axes = plt.subplots(
        len(available),
        3,
        figsize=(13, 3.2 * len(available)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, var in enumerate(available):
        base_field = base_npz[f"field_{var}"]
        cand_field = cand_npz[f"field_{var}"]
        diff = cand_field - base_field
        vmin = float(
            np.nanpercentile(
                np.concatenate([base_field.ravel(), cand_field.ravel()]), 2
            )
        )
        vmax = float(
            np.nanpercentile(
                np.concatenate([base_field.ravel(), cand_field.ravel()]), 98
            )
        )
        dmax = float(np.nanpercentile(np.abs(diff), 98))
        dmax = dmax if dmax > 0 else float(np.nanmax(np.abs(diff))) or 1.0
        panels = (
            (base_field, "baseline", {"vmin": vmin, "vmax": vmax, "cmap": "viridis"}),
            (cand_field, "candidate", {"vmin": vmin, "vmax": vmax, "cmap": "viridis"}),
            (
                diff,
                "candidate - baseline",
                {"vmin": -dmax, "vmax": dmax, "cmap": "coolwarm"},
            ),
        )
        for col, (field, title, kwargs) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(field, origin="upper", aspect="auto", **kwargs)
            ax.set_title(f"{var} {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, shrink=0.75)
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"{variant}_arco_compare.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-repo", type=Path, required=True)
    parser.add_argument("--candidate-repo", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--source", choices=("arco", "random"), default="arco")
    parser.add_argument("--time", default="2020-01-01T00:00")
    parser.add_argument("--plot-dir", type=Path, default=Path("weathernext_plots"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--plot-var",
        action="append",
        dest="plot_vars",
        help="Variable to plot. May be supplied more than once.",
    )
    parser.add_argument(
        "--strict-values",
        action="store_true",
        help="Fail if summary/probe values are not within tolerance.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(VARIANTS),
        help="Variant to compare. May be supplied more than once.",
    )
    args = parser.parse_args()

    variants = args.variant or list(VARIANTS)
    plot_vars = args.plot_vars or list(DEFAULT_PLOT_VARS)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for variant in variants:
            print(f"== {variant} ==")
            baseline = _run_variant(
                args.baseline_repo.resolve(),
                variant,
                args.device,
                args.seed,
                args.timeout,
                args.source,
                args.time,
                plot_vars,
                tmpdir / f"{variant}_baseline.npz",
            )
            candidate = _run_variant(
                args.candidate_repo.resolve(),
                variant,
                args.device,
                args.seed,
                args.timeout,
                args.source,
                args.time,
                plot_vars,
                tmpdir / f"{variant}_candidate.npz",
            )
            metrics = _compare(
                baseline, candidate, args.rtol, args.atol, args.strict_values
            )
            plot_path = None
            if not args.no_plots:
                plot_path = _plot_fields(
                    variant, baseline, candidate, plot_vars, args.plot_dir
                )
            print(
                f"ok shape={candidate['shape']} mean={candidate['mean']:.6g} "
                f"std={candidate['std']:.6g} "
                f"probe_rel_l2={metrics['probe_rel_l2']:.3e} "
                f"probe_max_abs={metrics['probe_max_abs']:.3e}"
            )
            if plot_path is not None:
                print(f"plot={plot_path}")


if __name__ == "__main__":
    main()

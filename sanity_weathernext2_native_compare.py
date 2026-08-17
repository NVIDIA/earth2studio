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
"""Compare Earth2Studio WeatherNext2CyclonesMini against vanilla WeatherNext2 inference.

This uses the public WeatherNext 2 sample dataset by default and compares the
Earth2Studio wrapper to an independently constructed WeatherNext FGN inference
function following ``docs/weathernext2/wn2_demo.ipynb`` from the WeatherNext repo.

Example
-------
uv run --extra weathernext python sanity_weathernext2_native_compare.py
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
from collections import OrderedDict
from pathlib import Path

import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
from weathernext.weathernext2 import fgn

from earth2studio.lexicon.wb2 import WB2Lexicon
from earth2studio.models.px import weathernext2_cyclones_mini as wn2_module
from earth2studio.models.px.weathernext2_cyclones_mini import (
    INPUT_VARIABLES,
    INV_VOCAB,
    OUTPUT_VARIABLES,
    PRESSURE_LEVELS,
    WN2_TARGET_VARIABLES,
    WeatherNext2CyclonesMini,
)
from earth2studio.utils.type import CoordSystem

DEFAULT_SAMPLE = (
    "dataset/source-hres_forecast_init-2024-10-07 00:00:00_"
    "res-1.0_levels-13_steps-01.nc"
)
DEFAULT_PLOT_VARS = ("t2m", "u10m", "v10m", "msl", "z500")


def _fill_missing_sample_inputs(dataset: xr.Dataset) -> xr.Dataset:
    """Fill demo-sample gaps needed by the public WeatherNext 2 config."""
    dataset = dataset.copy()
    missing = {
        "100m_u_component_of_wind": "10m_u_component_of_wind",
        "100m_v_component_of_wind": "10m_v_component_of_wind",
    }
    for target, source in missing.items():
        if target not in dataset and source in dataset:
            dataset[target] = dataset[source].copy(deep=True)
    return dataset


def _set_output_subset(variables: list[str]) -> None:
    """Restrict wrapper/native outputs for a memory-bounded sanity check."""
    global OUTPUT_VARIABLES, WN2_TARGET_VARIABLES

    OUTPUT_VARIABLES = variables
    WN2_TARGET_VARIABLES = tuple(
        dict.fromkeys(WB2Lexicon.VOCAB[var].split("::")[0] for var in variables)
    )
    wn2_module.OUTPUT_VARIABLES = OUTPUT_VARIABLES
    wn2_module.WN2_TARGET_VARIABLES = WN2_TARGET_VARIABLES


def _sample_to_e2s_tensor(
    dataset: xr.Dataset, model: WeatherNext2CyclonesMini, device: str
) -> tuple[torch.Tensor, CoordSystem]:
    """Convert WeatherNext sample inputs into Earth2Studio tensor/coords."""
    fields = []
    for variable in INPUT_VARIABLES:
        wb2_name, level = WB2Lexicon.VOCAB[variable].split("::")
        array = dataset[wb2_name].isel(batch=0, time=slice(0, 2))
        if level:
            array = array.sel(level=int(level))
        array = array.sel(lat=list(reversed(dataset.lat.values)))
        fields.append(array.values)

    x = np.stack(fields, axis=1)[np.newaxis]
    coords = OrderedDict(
        {
            "time": np.array([dataset.datetime.values[0, 1]]),
            "lead_time": model.input_coords()["lead_time"],
            "variable": np.array(INPUT_VARIABLES),
            "lat": np.asarray(list(reversed(dataset.lat.values))),
            "lon": dataset.lon.values,
        }
    )
    return torch.from_numpy(x.copy()).to(torch.float32).to(device), coords


def _native_run_forward(
    model: WeatherNext2CyclonesMini, model_name: str, jit_compile: bool
):
    """Build WeatherNext 2 inference function following the upstream demo."""
    config = copy.deepcopy(
        fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{model_name}")
    )
    task_config = dataclasses.replace(
        config.task, target_variables=WN2_TARGET_VARIABLES
    )
    noisy_function_kwargs = config.predictor_kwargs["noisy_function_kwargs"]
    noisy_function_kwargs["per_var_activation_fns"] = {
        key: value
        for key, value in noisy_function_kwargs.get(
            "per_var_activation_fns", {}
        ).items()
        if key in task_config.target_variables
    }
    transformer_kwargs = noisy_function_kwargs["mesh_model_ctor"].keywords[
        "transformer_kwargs"
    ]
    if jax.default_backend() == "gpu":
        transformer_kwargs["attention_type"] = "triblockdiag_mha"

    config_inference = fgn.PredictorConfig(
        task=task_config,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        predictor_wrappers=config.predictor_wrappers[:-2],
    )

    @hk.transform
    def run_forward(
        inputs: xr.Dataset, targets_template: xr.Dataset, forcings: xr.Dataset
    ) -> xr.Dataset:
        predictor = fgn.construct_predictor(config_inference)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def apply(
        rng,
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: xr.Dataset,
    ) -> xr.Dataset:
        return run_forward.apply(
            model.ckpt.params, rng, inputs, targets_template, forcings
        )

    if jit_compile:
        return jax.jit(apply), task_config
    return apply, task_config


def _load_checkpoint(package, model_name: str, split: str, model_index: int):
    """Load a WeatherNext2-family checkpoint from the public package."""
    if model_name == "WeatherNext2":
        params_name = f"params/WeatherNext2_<2025_model{model_index}.npz"
    else:
        params_name = f"params/{model_name}_<{split}.npz"
    with open(package.resolve(params_name), "rb") as f:
        return checkpoint.load(f, fgn.CheckPoint)


def _native_prediction(
    model: WeatherNext2CyclonesMini,
    model_name: str,
    dataset: xr.Dataset,
    seed: int,
    jit_compile: bool,
) -> torch.Tensor:
    """Run vanilla WeatherNext rollout on the sample batch."""
    run_forward, task_config = _native_run_forward(
        model, model_name=model_name, jit_compile=jit_compile
    )
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        dataset,
        target_lead_times=slice("6h", "6h"),
        **dataclasses.asdict(task_config),
    )
    _, rng = jax.random.split(jax.random.PRNGKey(seed))
    predictions = rollout.chunked_prediction(
        run_forward,
        rng=rng,
        inputs=inputs,
        targets_template=targets * np.nan,
        forcings=forcings,
    )
    return _prediction_to_e2s_tensor(predictions)


def _prediction_to_e2s_tensor(dataset: xr.Dataset) -> torch.Tensor:
    """Convert WeatherNext predictions to Earth2Studio output tensor order."""
    dataset = dataset.copy()
    for var in list(dataset.data_vars):
        if "level" in dataset[var].dims:
            for level in PRESSURE_LEVELS:
                dataset[f"{var}::{level}"] = dataset[var].sel(level=level)
            dataset = dataset.drop_vars(var)
        else:
            dataset = dataset.rename({var: f"{var}::"})

    if "level" in dataset.dims:
        dataset = dataset.drop_dims("level")
    if len(dataset.time) > 1:
        dataset = dataset.rename({"time": "lead_time"})
        dataset = dataset.expand_dims(dim="time")
    else:
        dataset = dataset.expand_dims(dim="lead_time")
    if "sample" in dataset.dims:
        dataset = dataset.isel(sample=0, drop=True)

    dataset = dataset.rename({key: INV_VOCAB[key] for key in dataset.data_vars})
    dataarray = (
        dataset[OUTPUT_VARIABLES]
        .to_dataarray()
        .T.transpose(..., "batch", "time", "lead_time", "variable", "lat", "lon")
    )
    return torch.from_numpy(dataarray.to_numpy().copy()).float().flip(-2)


def _align_native_shape(e2s: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    """Drop singleton native wrapper axes until shapes match."""
    while native.ndim > e2s.ndim:
        squeeze_dim = next(
            (i for i, size in enumerate(native.shape) if size == 1), None
        )
        if squeeze_dim is None:
            break
        native = native.squeeze(squeeze_dim)
    return native


def _print_delta(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> None:
    reference = reference.detach().cpu().float()
    candidate = candidate.detach().cpu().float()
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    ref = reference[finite]
    cand = candidate[finite]
    diff = cand - ref
    rel_l2 = torch.linalg.vector_norm(diff.reshape(-1)) / torch.linalg.vector_norm(
        ref.reshape(-1)
    )
    print(
        f"{name}: finite={int(finite.sum())}/{reference.numel()} "
        f"rel_l2={float(rel_l2):.6e} "
        f"mean_abs={float(diff.abs().mean()):.6e} "
        f"max_abs={float(diff.abs().max()):.6e}"
    )


def _field_index(coords: CoordSystem, variable: str) -> tuple:
    variables = list(np.asarray(coords["variable"]).astype(str))
    index = []
    for dim in coords:
        if dim in {"lat", "lon"}:
            index.append(slice(None))
        elif dim == "variable":
            index.append(variables.index(variable))
        else:
            index.append(0)
    return tuple(index)


def _field_metrics(
    e2s: torch.Tensor, native: torch.Tensor, coords: CoordSystem, variables: list[str]
) -> None:
    available = list(np.asarray(coords["variable"]).astype(str))
    for variable in variables:
        if variable not in available:
            print(f"{variable}: missing")
            continue
        idx = _field_index(coords, variable)
        e2s_field = e2s[idx].detach().cpu().float().numpy().astype(np.float64)
        native_field = native[idx].detach().cpu().float().numpy().astype(np.float64)
        diff = native_field - e2s_field
        rel_l2 = np.linalg.norm(np.nan_to_num(diff)) / max(
            np.linalg.norm(np.nan_to_num(e2s_field)), np.finfo(np.float64).eps
        )
        print(
            f"{variable}: rel_l2={rel_l2:.6e} "
            f"mean_abs={np.nanmean(np.abs(diff)):.6e} "
            f"max_abs={np.nanmax(np.abs(diff)):.6e}"
        )


def _plot_fields(
    e2s: torch.Tensor,
    native: torch.Tensor,
    coords: CoordSystem,
    variables: list[str],
    path: Path,
) -> None:
    available = [
        v for v in variables if v in set(np.asarray(coords["variable"]).astype(str))
    ]
    if not available:
        return
    fig, axes = plt.subplots(
        len(available), 3, figsize=(13, 3.2 * len(available)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for row, variable in enumerate(available):
        idx = _field_index(coords, variable)
        e2s_field = e2s[idx].detach().cpu().float().numpy()
        native_field = native[idx].detach().cpu().float().numpy()
        diff = native_field - e2s_field
        values = np.concatenate([e2s_field.ravel(), native_field.ravel()])
        vmin = float(np.nanpercentile(values, 2))
        vmax = float(np.nanpercentile(values, 98))
        dmax = float(np.nanpercentile(np.abs(diff), 98))
        dmax = dmax if dmax > 0 else float(np.nanmax(np.abs(diff))) or 1.0
        panels = (
            (
                e2s_field,
                "E2S WeatherNext2CyclonesMini",
                {"vmin": vmin, "vmax": vmax, "cmap": "viridis"},
            ),
            (
                native_field,
                "WeatherNext vanilla",
                {"vmin": vmin, "vmax": vmax, "cmap": "viridis"},
            ),
            (diff, "native - E2S", {"vmin": -dmax, "vmax": dmax, "cmap": "coolwarm"}),
        )
        for col, (field, title, kwargs) in enumerate(panels):
            ax = axes[row, col]
            image = ax.imshow(field, origin="upper", aspect="auto", **kwargs)
            ax.set_title(f"{variable} {title}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(image, ax=ax, shrink=0.75)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-name",
        default="WeatherNextCyclones_Mini",
        help="WeatherNext2-family config/checkpoint name to compare.",
    )
    parser.add_argument("--model-index", type=int, default=1)
    parser.add_argument("--split", default="2024")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-path", default=DEFAULT_SAMPLE)
    parser.add_argument("--plot-dir", type=Path, default=Path("weathernext_plots"))
    parser.add_argument("--plot-var", action="append", dest="plot_vars")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--all-output-vars",
        action="store_true",
        help="Compare all WeatherNext2 gridded outputs instead of the plotted subset.",
    )
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args()

    variables = args.plot_vars or list(DEFAULT_PLOT_VARS)
    if not args.all_output_vars:
        _set_output_subset(variables)

    package = WeatherNext2CyclonesMini.load_default_package()
    sample = _fill_missing_sample_inputs(
        xr.load_dataset(package.resolve(args.sample_path)).compute()
    )
    ckpt = _load_checkpoint(
        package,
        model_name=args.model_name,
        split=args.split,
        model_index=args.model_index,
    )
    model = WeatherNext2CyclonesMini(
        ckpt,
        sample["land_sea_mask"].values,
        sample["geopotential_at_surface"].values,
        seed=args.seed,
        jit_compile=not args.no_jit,
    ).to(args.device)
    model.run_forward, model.task_config = _native_run_forward(
        model, model_name=args.model_name, jit_compile=not args.no_jit
    )
    x, coords = _sample_to_e2s_tensor(sample, model, args.device)

    with jax.default_device(model.get_jax_device_from_tensor(x)):
        with torch.no_grad():
            e2s_out, out_coords = model(x, coords)
            e2s_repeat = model(x, coords)[0]
        native_out = _native_prediction(
            model,
            model_name=args.model_name,
            dataset=sample,
            seed=args.seed,
            jit_compile=not args.no_jit,
        ).to(args.device)
    native_out = _align_native_shape(e2s_out, native_out)

    if e2s_out.shape != native_out.shape:
        raise ValueError(
            f"Shape mismatch: E2S {tuple(e2s_out.shape)} != native {tuple(native_out.shape)}"
        )

    print(f"{args.model_name}: shape={list(e2s_out.shape)}")
    _print_delta("native-vs-e2s", e2s_out, native_out)
    _print_delta("e2s-repeat", e2s_out, e2s_repeat)
    _field_metrics(e2s_out, native_out, out_coords, variables)
    if not args.no_plot:
        plot_path = args.plot_dir / "weathernext2_native_compare.png"
        _plot_fields(e2s_out, native_out, out_coords, variables, plot_path)
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()

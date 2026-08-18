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
"""Compare native WeatherNext inference with Earth2Studio wrappers.

The script runs an Earth2Studio wrapper and a native WeatherNext rollout using
matching checkpoint parameters, prepared xarray inputs, RNG, target template, and
forcings. Use it to separate wrapper differences from WeatherNext/JAX runtime
repeatability.

Example
-------
uv run python sanity_weathernext_native_compare.py --variant graphcast-small
uv run python sanity_weathernext_native_compare.py --variant gencast-mini
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import functools
import io
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import haiku as hk
import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from weathernext.utils import (
    autoregressive,
    casting,
    data_utils,
    nan_cleaning,
    normalization,
    rollout,
)
from weathernext.weathernext1_gen import gencast
from weathernext.weathernext1_graph import graphcast

from earth2studio.data import ARCO, fetch_data
from earth2studio.models.px import GenCastMini, GraphCastOperational, GraphCastSmall
from earth2studio.utils.coords import map_coords

VARIANTS = {
    "graphcast-small": GraphCastSmall,
    "graphcast-operational": GraphCastOperational,
    "gencast-mini": GenCastMini,
}
DEFAULT_PLOT_VARS = ("t2m", "u10m", "v10m", "msl", "z500")


def _native_graphcast_run_forward(
    model: GraphCastSmall | GraphCastOperational,
) -> Callable:
    params = model.ckpt.params
    state: dict = {}
    model_config = model.ckpt.model_config
    task_config = model.ckpt.task_config

    def construct_wrapped_graphcast(
        model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig
    ) -> autoregressive.Predictor:
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=model.diffs_stddev_by_level,
            mean_by_level=model.mean_by_level,
            stddev_by_level=model.stddev_by_level,
        )
        return autoregressive.Predictor(predictor, gradient_checkpointing=True)

    @hk.transform_with_state
    def run_forward(
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig,
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: xr.Dataset,
    ) -> xr.Dataset:
        predictor = construct_wrapped_graphcast(model_config, task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    fn = functools.partial(
        run_forward.apply,
        model_config=model_config,
        task_config=task_config,
        params=params,
        state=state,
    )
    return lambda **kw: jax.jit(fn)(**kw)[0]


def _native_gencast_run_forward(model: GenCastMini, jit_compile: bool) -> Callable:
    params = model.ckpt.params
    state: dict = {}
    task_config = model.ckpt.task_config
    sampler_config = model.ckpt.sampler_config
    noise_config = model.ckpt.noise_config
    noise_encoder_config = model.ckpt.noise_encoder_config

    splash_spt_cfg = model.ckpt.denoiser_architecture_config.sparse_transformer_config
    tbd_spt_cfg = dataclasses.replace(
        splash_spt_cfg, attention_type="triblockdiag_mha", mask_type="full"
    )
    denoiser_architecture_config = dataclasses.replace(
        model.ckpt.denoiser_architecture_config,
        sparse_transformer_config=tbd_spt_cfg,
    )

    def construct_wrapped_gencast(
        task_config: graphcast.TaskConfig,
        denoiser_architecture_config: gencast.DenoiserArchitectureConfig,
        sampler_config: gencast.SamplerConfig,
        noise_config: gencast.NoiseConfig,
        noise_encoder_config: gencast.NoiseEncoderConfig,
    ) -> gencast.GenCast:
        predictor = gencast.GenCast(
            task_config=task_config,
            denoiser_architecture_config=denoiser_architecture_config,
            sampler_config=sampler_config,
            noise_config=noise_config,
            noise_encoder_config=noise_encoder_config,
        )
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=model.diffs_stddev_by_level,
            mean_by_level=model.mean_by_level,
            stddev_by_level=model.stddev_by_level,
        )
        return nan_cleaning.NaNCleaner(
            predictor=predictor,
            reintroduce_nans=True,
            fill_value=model.min_by_level,
            var_to_clean="sea_surface_temperature",
        )

    @hk.transform_with_state
    def run_forward(
        task_config: graphcast.TaskConfig,
        denoiser_architecture_config: gencast.DenoiserArchitectureConfig,
        sampler_config: gencast.SamplerConfig,
        noise_config: gencast.NoiseConfig,
        noise_encoder_config: gencast.NoiseEncoderConfig,
        inputs: xr.Dataset,
        targets_template: xr.Dataset,
        forcings: xr.Dataset,
    ) -> xr.Dataset:
        predictor = construct_wrapped_gencast(
            task_config,
            denoiser_architecture_config,
            sampler_config,
            noise_config,
            noise_encoder_config,
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    fn = functools.partial(
        run_forward.apply,
        task_config=task_config,
        denoiser_architecture_config=denoiser_architecture_config,
        sampler_config=sampler_config,
        noise_config=noise_config,
        noise_encoder_config=noise_encoder_config,
        params=params,
        state=state,
    )
    if jit_compile:
        fn = jax.jit(fn)
    return lambda **kw: fn(**kw)[0]


def _load_model(variant: str, jit_compile: bool) -> Any:
    cls = VARIANTS[variant]
    if cls is GenCastMini:
        return cls.load_model(
            cls.load_default_package(), jit_compile=jit_compile, seed=0
        )
    return cls.load_model(cls.load_default_package())


def _fetch_input(
    model: Any, device: str, valid_time: str
) -> tuple[torch.Tensor, OrderedDict]:
    coords_in = model.input_coords()
    domain = OrderedDict(
        (key, value)
        for key, value in coords_in.items()
        if key not in {"batch", "time", "lead_time", "variable"}
    )
    interp_domain = domain.copy()
    if "lat" in domain and "_lat" not in interp_domain:
        interp_domain["_lat"] = domain["lat"]
    if "lon" in domain and "_lon" not in interp_domain:
        interp_domain["_lon"] = domain["lon"]

    x, coords = fetch_data(
        ARCO(cache=True, verbose=True),
        np.array([np.datetime64(valid_time)]),
        coords_in["variable"],
        coords_in["lead_time"],
        device=device,
        interp_to=interp_domain,
    )
    if "_lat" in coords and "_lon" in coords:
        normalized = OrderedDict()
        for key, value in coords.items():
            if key in {"lat", "lon"}:
                continue
            if key == "_lat":
                normalized["lat"] = value
            elif key == "_lon":
                normalized["lon"] = value
            else:
                normalized[key] = value
        coords = normalized
    return x, coords


def _with_singleton_batch(
    x: torch.Tensor, coords: OrderedDict
) -> tuple[torch.Tensor, OrderedDict]:
    if "batch" in coords:
        return x, coords
    batched_coords = coords.copy()
    batched_coords.update({"batch": np.array([0])})
    batched_coords.move_to_end("batch", last=False)
    return x.unsqueeze(0), batched_coords


def _native_prediction(
    model: Any,
    x: torch.Tensor,
    coords: OrderedDict,
    reuse_wrapper_run_forward: bool,
    jit_compile: bool,
) -> torch.Tensor:
    x, coords = _with_singleton_batch(x, coords)
    x, coords = map_coords(x, coords, model.input_coords())
    step_hours = 12 if isinstance(model, GenCastMini) else 6
    data, target_lead_times = model.from_dataarray_to_dataset(
        xr.DataArray(x.cpu(), coords=coords), step_hours
    )
    inputs, targets, forcings = data_utils.extract_inputs_targets_forcings(
        data,
        target_lead_times=target_lead_times,
        **dataclasses.asdict(model.ckpt.task_config),
    )

    if reuse_wrapper_run_forward:
        run_forward = model.run_forward
    elif isinstance(model, GenCastMini):
        run_forward = _native_gencast_run_forward(model, jit_compile)
    else:
        run_forward = _native_graphcast_run_forward(model)

    if isinstance(model, GenCastMini):
        rng = (
            jax.random.PRNGKey(0)
            if model.seed is None
            else jax.random.PRNGKey(model.seed)
        )
        rng = jax.random.fold_in(rng, 0)
    else:
        rng = model.prng_key

    with contextlib.redirect_stdout(io.StringIO()):
        predictions = rollout.chunked_prediction(
            run_forward,
            rng=rng,
            inputs=inputs,
            targets_template=targets * np.nan,
            forcings=forcings,
        )
    return model.iterator_result_to_tensor(predictions).to(x.device)


def _align_native_shape(e2s: torch.Tensor, native: torch.Tensor) -> torch.Tensor:
    if native.ndim == e2s.ndim + 1 and native.shape[0] == 1:
        return native.squeeze(0)
    return native


def _field_index(coords: OrderedDict, variable: str) -> tuple:
    all_variables = list(np.asarray(coords["variable"]).astype(str))
    index = []
    for dim in coords:
        if dim in {"lat", "lon"}:
            index.append(slice(None))
        elif dim == "variable":
            index.append(all_variables.index(variable))
        else:
            index.append(0)
    return tuple(index)


def _field_metrics(
    e2s: torch.Tensor, native: torch.Tensor, coords: OrderedDict, variables: list[str]
) -> None:
    all_variables = list(np.asarray(coords["variable"]).astype(str))
    for variable in variables:
        if variable not in all_variables:
            print(f"{variable}: missing")
            continue
        index = _field_index(coords, variable)
        e2s_field = e2s[index].detach().cpu().float().numpy().astype(np.float64)
        native_field = native[index].detach().cpu().float().numpy().astype(np.float64)
        diff = native_field - e2s_field
        rel_l2 = np.linalg.norm(np.nan_to_num(diff)) / max(
            np.linalg.norm(np.nan_to_num(e2s_field)), np.finfo(np.float64).eps
        )
        print(
            f"{variable}: rel_l2={rel_l2:.6e} "
            f"mean_abs={np.nanmean(np.abs(diff)):.6e} "
            f"max_abs={np.nanmax(np.abs(diff)):.6e}"
        )


def _print_tensor_delta(
    name: str, reference: torch.Tensor, candidate: torch.Tensor
) -> None:
    reference = reference.detach().cpu().float()
    candidate = candidate.detach().cpu().float()
    finite = torch.isfinite(reference) & torch.isfinite(candidate)
    if not bool(finite.any()):
        print(f"{name}: no finite overlapping values")
        return
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


def _plot_fields(
    e2s: torch.Tensor,
    native: torch.Tensor,
    coords: OrderedDict,
    variables: list[str],
    path: Path,
) -> None:
    all_variables = list(np.asarray(coords["variable"]).astype(str))
    available = [variable for variable in variables if variable in all_variables]
    if not available:
        return
    fig, axes = plt.subplots(
        len(available), 3, figsize=(13, 3.2 * len(available)), constrained_layout=True
    )
    axes = np.atleast_2d(axes)
    for row, variable in enumerate(available):
        index = _field_index(coords, variable)
        e2s_field = e2s[index].detach().cpu().float().numpy()
        native_field = native[index].detach().cpu().float().numpy()
        diff = native_field - e2s_field
        field_values = np.concatenate([e2s_field.ravel(), native_field.ravel()])
        vmin = float(np.nanpercentile(field_values, 2))
        vmax = float(np.nanpercentile(field_values, 98))
        dmax = float(np.nanpercentile(np.abs(diff), 98))
        dmax = dmax if dmax > 0 else float(np.nanmax(np.abs(diff))) or 1.0
        panels = (
            (e2s_field, "E2S wrapper", {"vmin": vmin, "vmax": vmax, "cmap": "viridis"}),
            (
                native_field,
                "WeatherNext native",
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
    parser.add_argument(
        "--variant", choices=sorted(VARIANTS), default="graphcast-small"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time", default="2020-01-01T00:00")
    parser.add_argument("--plot-dir", type=Path, default=Path("weathernext_plots"))
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--plot-var", action="append", dest="plot_vars")
    parser.add_argument(
        "--reuse-wrapper-run-forward",
        action="store_true",
        help="Use model.run_forward while still executing native WeatherNext rollout.",
    )
    parser.add_argument(
        "--check-repeat",
        action="store_true",
        help="Also compare two repeated public E2S wrapper calls on the same input.",
    )
    parser.add_argument(
        "--jit-compile",
        action="store_true",
        help="JIT compile GenCast. GraphCast is always JIT compiled by WeatherNext.",
    )
    args = parser.parse_args()

    model = _load_model(args.variant, args.jit_compile).to(args.device)
    x, coords = _fetch_input(model, args.device, args.time)

    with jax.default_device(model.get_jax_device_from_tensor(x)):
        with torch.no_grad():
            e2s_out, out_coords = model(x, coords)
            e2s_repeat = model(x, coords)[0] if args.check_repeat else None
        native_out = _align_native_shape(
            e2s_out,
            _native_prediction(
                model,
                x,
                coords,
                args.reuse_wrapper_run_forward,
                args.jit_compile,
            ),
        )

    if e2s_out.shape != native_out.shape:
        raise ValueError(
            f"Shape mismatch: E2S {tuple(e2s_out.shape)} != native {tuple(native_out.shape)}"
        )

    print(f"{args.variant}: shape={list(e2s_out.shape)}")
    _print_tensor_delta("native-vs-e2s", e2s_out, native_out)
    if e2s_repeat is not None:
        _print_tensor_delta("e2s-repeat", e2s_out, e2s_repeat)

    variables = args.plot_vars or list(DEFAULT_PLOT_VARS)
    _field_metrics(e2s_out, native_out, out_coords, variables)
    if not args.no_plot:
        plot_path = args.plot_dir / f"{args.variant}_native_compare.png"
        _plot_fields(e2s_out, native_out, out_coords, variables, plot_path)
        print(f"plot={plot_path}")


if __name__ == "__main__":
    main()

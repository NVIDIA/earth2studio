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

import os
import shutil

import numpy as np
import xarray as xr
from omegaconf import DictConfig

from earth2studio.io import NetCDF4Backend, ZarrBackend
from earth2studio.io.utils import output_template
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import coord_array


def build_output_template(
    cfg: DictConfig, model: PrognosticModel, ics: np.ndarray, mems: np.ndarray
) -> xr.DataArray:
    """Signature of all fields written by the run.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object, providing ``ensemble_size``, ``n_steps`` and
        ``out_vars``.
    model : PrognosticModel
        Prognostic model whose output grid is stored.
    ics : np.ndarray
        Initial condition times (dtype ``datetime64``).
    mems : np.ndarray
        Ensemble member indices.

    Returns
    -------
    xr.DataArray
        Allocation-free template with dimensions ``(ensemble, time, lead_time,
        variable, lat, lon)``, where ``lead_time`` covers the initial condition
        and all ``n_steps`` forecast steps.

    Raises
    ------
    ValueError
        If an output variable is not both an input and an output of the model,
        which is required since the initial condition is stored as well.
    """
    out_vars = list(dict.fromkeys(cfg.out_vars))
    inputs = model.input_coords()
    outputs = model.output_coords(inputs)

    missing = set(out_vars) - (
        set(inputs.coords["variable"].values) & set(outputs.coords["variable"].values)
    )
    if missing:
        raise ValueError(f"out_vars {sorted(missing)} are not provided by the model")

    step = (
        outputs.coords["lead_time"].values[-1] - inputs.coords["lead_time"].values[-1]
    )
    return output_template(
        outputs,
        leading={"ensemble": np.asarray(mems), "time": np.asarray(ics)},
        coords={
            "lead_time": step * np.arange(cfg.n_steps + 1),
            "variable": np.array(out_vars),
        },
    )


def setup_output(
    cfg: DictConfig,
    model: PrognosticModel,
    ics: np.ndarray,
    add_arrays: bool,
) -> tuple[ZarrBackend | None, xr.DataArray | None]:
    """Set up the storage backend.

    Supports Zarr and NetCDF store types. For NetCDF the store is created
    later per initial condition (see :func:`initialise_netcdf_output`), so
    this function returns ``None`` as the store.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.
    model : PrognosticModel
        Prognostic model whose coordinate metadata is used.
    ics : np.ndarray
        Array of initial condition times (dtype ``datetime64``).
    add_arrays : bool
        If ``True``, allocate output arrays on the backend immediately.

    Returns
    -------
    tuple[ZarrBackend | None, xr.DataArray | None]
        The Zarr store (``None`` for NetCDF or no output) and the output
        template (``None`` if no fields are stored).

    Raises
    ------
    ValueError
        If ``cfg.store_type`` is not one of ``"zarr"``, ``"netcdf"``, or
        ``"none"``, or if ``out_vars`` is missing.
    """
    if cfg.store_type not in ("zarr", "netcdf", "none"):
        raise ValueError(f"Invalid store type: {cfg.store_type}")
    if cfg.store_type == "none":
        return None, None
    if "out_vars" not in cfg:
        raise ValueError("out_vars must be specified if store_type is not none.")

    template = build_output_template(cfg, model, ics, np.arange(cfg.ensemble_size))
    # TODO: let proc zero create dir and add barrier
    os.makedirs(cfg.store_dir, exist_ok=True)

    if cfg.store_type == "netcdf":
        return None, template

    file_name = os.path.join(cfg.store_dir, cfg.project)
    if not file_name.endswith(".zarr"):
        file_name += ".zarr"
    if os.path.exists(file_name) and add_arrays:
        shutil.rmtree(file_name)

    store = ZarrBackend(
        file_name=file_name,
        chunks={"ensemble": 1, "time": 1, "lead_time": 1},
        backend_kwargs={"overwrite": False},
    )
    if add_arrays:
        store.add_array(template)

    return store, template


def initialise_netcdf_output(
    cfg: DictConfig,
    model: PrognosticModel,
    ic: np.datetime64,
    runs: list[tuple[np.datetime64, np.ndarray, int]],
) -> NetCDF4Backend:
    """Create a NetCDF4 store for a single initial condition.

    The file name encodes the IC timestamp and ensemble member range.
    Random seeds are written into the store alongside the forecast arrays.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.
    model : PrognosticModel
        Prognostic model whose output grid is stored.
    ic : np.datetime64
        Initial condition time for this file.
    runs : list[tuple[np.datetime64, np.ndarray, int]]
        Full list of ``(ic_time, member_indices, seed)`` tuples; only
        entries matching *ic* are used.

    Returns
    -------
    NetCDF4Backend
        Initialised storage backend ready for writing.
    """
    batches = [(mems, seed) for iic, mems, seed in runs if iic == ic]
    mems = np.concatenate([mems for mems, _ in batches])
    seeds = np.concatenate([np.full(len(mems), seed) for mems, seed in batches])

    # setup filename
    file_name = os.path.join(cfg.store_dir, cfg.project).removesuffix(".nc")
    file_name = (
        file_name
        + f"_{np.datetime_as_string(ic, unit='s')}_mems{mems[0]:04d}-{mems[-1]:04d}.nc"
    )
    file_name = file_name.replace(":", ".")

    store = NetCDF4Backend(
        file_name=file_name, backend_kwargs={"mode": "w", "diskless": False}
    )

    # random seed of each member, an array named after itself rather than by a
    # variable label
    store.add_array(
        coord_array(
            ("ensemble",), {"ensemble": mems}, dtype=seeds.dtype, name="random_seed"
        )
    )
    store.write(
        xr.DataArray(
            seeds, dims=("ensemble",), coords={"ensemble": mems}, name="random_seed"
        )
    )

    store.add_array(build_output_template(cfg, model, np.array([ic]), mems))

    return store

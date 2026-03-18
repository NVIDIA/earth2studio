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

import copy
import os
import shutil
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from earth2studio.io import NetCDF4Backend, ZarrBackend
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import map_coords, split_coords


def initialise_output_coords(
    cfg: DictConfig,
    model: PrognosticModel,
    ics: np.ndarray,
    out_vars: list[str],
) -> OrderedDict:
    """construct output corrds for storing fields.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.
    model : PrognosticModel
        Prognostic model whose coordinate metadata is used.
    ics : np.ndarray
        Array of initial condition times (dtype ``datetime64``).
    out_vars : list[str]
        Variable names to include in the output.

    Returns
    -------
    OrderedDict
        Output coordinate mapping with keys *time*, *lead_time*,
        *variable*, and the spatial dimensions from the model.
    """
    out_coords = OrderedDict(
        {
            k: v
            for k, v in model.output_coords(model.input_coords()).items()
            if (k != "batch") and (v.shape != 0)
        }
    )

    out_coords["time"] = ics
    out_coords["lead_time"] = np.asarray(
        [out_coords["lead_time"] * i for i in range(cfg.n_steps + 1)]
    ).flatten()
    out_coords["variable"] = np.array(out_vars)

    out_coords.move_to_end("lead_time", last=False)
    out_coords.move_to_end("time", last=False)

    return out_coords


def add_arrays_to_store(
    store: ZarrBackend | NetCDF4Backend,
    out_coords: OrderedDict,
    mems: np.ndarray,
    add_arrays: bool,
    ic: np.datetime64 | None = None,
    array_kwargs: dict | None = None,
) -> None:
    """Allocate arrays on a storage backend.

    Parameters
    ----------
    store : ZarrBackend | NetCDF4Backend
        Storage backend.
    out_coords : OrderedDict
        Base output coordinate mapping (not modified).
    mems : np.ndarray
        Ensemble member indices.
    add_arrays : bool
        If ``True``, call ``store.add_array``; otherwise do nothing.
    ic : np.datetime64 | None, optional
        If provided, restrict the *time* coordinate to this single IC.
    array_kwargs : dict | None, optional
        Extra keyword arguments forwarded to ``store.add_array``.
    """
    if array_kwargs is None:
        array_kwargs = {}
    oco = copy.deepcopy(out_coords)
    oco["ensemble"] = mems
    oco.move_to_end("ensemble", last=False)
    out_vars = oco.pop("variable")

    if ic is not None:
        oco["time"] = np.array([ic])

    if add_arrays:
        store.add_array(coords=oco, array_name=out_vars, **array_kwargs)

    return


def setup_output(
    cfg: DictConfig,
    model: PrognosticModel,
    ics: np.ndarray,
    add_arrays: bool,
) -> tuple[ZarrBackend | None, OrderedDict]:
    """Set up the storage backend.

    Supports Zarr and NetCDF store types.  For NetCDF the store is created
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
    tuple[ZarrBackend | None, OrderedDict]
        The storage backend (or ``None`` for NetCDF / no-store) and the
        output coordinate mapping.

    Raises
    ------
    ValueError
        If ``cfg.store_type`` is not one of ``"zarr"``, ``"netcdf"``, or
        ``"none"``.
    """
    if "out_vars" in cfg:
        out_vars = list(dict.fromkeys(cfg.out_vars))
    else:
        out_vars = []
        if not cfg.store_type == "none":
            raise ValueError("out_vars must be specified if store_type is not none.")
    out_coords = initialise_output_coords(cfg, model, ics, out_vars)
    chunks = {
        "ensemble": 1,
        "time": 1,
        "lead_time": 1,
        "variable": 1,
    }
    array_kwargs: dict[str, Any] = {}

    # TODO: let proc zero create dir and add barrier
    os.makedirs(cfg.store_dir, exist_ok=True)
    file_name = os.path.join(cfg.store_dir, cfg.project)

    if cfg.store_type == "netcdf" or cfg.store_type == "none":
        store = None

    elif cfg.store_type == "zarr":
        # TODO: wrap in ordered_execution function
        if not file_name.endswith(".zarr"):
            file_name += ".zarr"

        if os.path.exists(file_name) and add_arrays:
            shutil.rmtree(file_name)

        store = ZarrBackend(
            file_name=file_name,
            chunks=chunks,
            backend_kwargs={"overwrite": False},
        )

        add_arrays_to_store(
            store=store,
            out_coords=out_coords,
            mems=np.asarray(list(range(cfg.ensemble_size))).flatten(),
            add_arrays=add_arrays,
            array_kwargs=array_kwargs,
        )

    else:
        raise ValueError(f"Invalid store type: {cfg.store_type}")

    return store, out_coords


def initialise_netcdf_output(
    cfg: DictConfig,
    out_coords: OrderedDict,
    ic: np.datetime64,
    ic_mems: list[tuple[np.datetime64, np.ndarray, int]],
) -> NetCDF4Backend:
    """Create a NetCDF4 store for a single initial condition.

    The file name encodes the IC timestamp and ensemble member range.
    Random seeds are written into the store alongside the forecast arrays.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.
    out_coords : OrderedDict
        Output coordinate mapping (not modified).
    ic : np.datetime64
        Initial condition time for this file.
    ic_mems : list[tuple[np.datetime64, np.ndarray, int]]
        Full list of ``(ic_time, member_indices, seed)`` tuples; only
        entries matching *ic* are used.

    Returns
    -------
    NetCDF4Backend
        Initialised storage backend ready for writing.
    """
    mems = np.concatenate([mem for iic, mem, _ in ic_mems if iic == ic]).flatten()
    seeds = np.concatenate(
        [
            np.array([seed] * len(mem), dtype=int)
            for iic, mem, seed in ic_mems
            if iic == ic
        ]
    ).flatten()

    # setup filename
    file_name = os.path.join(cfg.store_dir, cfg.project)
    if file_name.endswith(".nc"):
        file_name = file_name.replace(".nc", "")
    file_name = (
        file_name
        + f"_{np.datetime_as_string(ic, unit='s')}_mems{mems[0]:04d}-{mems[-1]:04d}.nc"
    )
    file_name = file_name.replace(":", ".")

    # create store
    chunks = {
        "ensemble": 1,
        "time": 1,
        "lead_time": 1,
        "variable": 1,
    }
    store = NetCDF4Backend(
        file_name=file_name,
        backend_kwargs={"mode": "w", "diskless": False, "chunks": chunks},
    )

    # add random seed to store
    store.add_array(coords={"ensemble": mems}, array_name="random_seed")
    store.write(torch.from_numpy(seeds), {"ensemble": mems}, "random_seed")

    # add arrays to the store
    add_arrays_to_store(
        store=store, out_coords=out_coords, mems=mems, ic=ic, add_arrays=True
    )

    return store


def write_to_store(
    store: ZarrBackend | NetCDF4Backend | None,
    xx: torch.Tensor,
    coords: OrderedDict,
    out_coords: OrderedDict,
) -> None:
    """Map coordinates and write a single time-step to the store.
    If store is ``None`` the call is a no-op.

    Parameters
    ----------
    store : ZarrBackend | NetCDF4Backend | None
        Storage backend, or ``None`` to skip writing.
    xx : torch.Tensor
        Data tensor for the current time-step.
    coords : OrderedDict
        Coordinate mapping for *xx*.
    out_coords : OrderedDict
        Target output coordinate mapping used for remapping.
    """
    if store is not None:
        xx_sub, coords_sub = map_coords(xx, coords, out_coords)
        store.write(*split_coords(xx_sub, coords_sub, dim="variable"))

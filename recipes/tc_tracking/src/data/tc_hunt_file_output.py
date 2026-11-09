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
import time

import numpy as np
import xarray as xr
import zarr
from loguru import logger
from omegaconf import DictConfig

from earth2studio.io import AsyncZarrBackend, NetCDF4Backend
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
    if isinstance(inputs, tuple):
        raise TypeError("models with several inputs are not supported")
    outputs = model.output_coords(inputs)
    if isinstance(outputs, tuple):
        raise TypeError("models with several outputs are not supported")

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


def combined_zarr_path(cfg: DictConfig) -> str:
    """Path of the Zarr store all batches are merged into."""
    return os.path.join(cfg.store_dir, cfg.project.removesuffix(".zarr") + ".zarr")


def batch_zarr_dir(cfg: DictConfig) -> str:
    """Directory of the per-batch Zarr stores."""
    return os.path.join(cfg.store_dir, cfg.project.removesuffix(".zarr") + "_batches")


def setup_output(
    cfg: DictConfig,
    model: PrognosticModel,
    ics: np.ndarray,
    remove_old: bool,
) -> xr.DataArray | None:
    """Check the output configuration and prepare the output directory.

    The stores themselves are created per batch, see :func:`open_batch_store`.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.
    model : PrognosticModel
        Prognostic model whose coordinate metadata is used.
    ics : np.ndarray
        Array of initial condition times (dtype ``datetime64``).
    remove_old : bool
        If ``True``, remove the Zarr outputs of an earlier run of the project.

    Returns
    -------
    xr.DataArray | None
        Output template of the whole run, ``None`` if no fields are stored.

    Raises
    ------
    ValueError
        If ``cfg.store_type`` is not one of ``"zarr"``, ``"netcdf"``, or
        ``"none"``, or if ``out_vars`` is missing.
    """
    if cfg.store_type not in ("zarr", "netcdf", "none"):
        raise ValueError(f"Invalid store type: {cfg.store_type}")
    if cfg.store_type == "none":
        return None
    if "out_vars" not in cfg:
        raise ValueError("out_vars must be specified if store_type is not none.")

    template = build_output_template(cfg, model, ics, np.arange(cfg.ensemble_size))
    os.makedirs(cfg.store_dir, exist_ok=True)

    if cfg.store_type == "zarr" and remove_old:
        for path in (combined_zarr_path(cfg), batch_zarr_dir(cfg)):
            if os.path.exists(path):
                shutil.rmtree(path)

    return template


def batch_name(ic: np.datetime64, mems: np.ndarray) -> str:
    """Name of a batch's output: IC and member range, e.g.
    ``2024-09-24T00.00.00_mems0000-0001``."""
    ic_str = np.datetime_as_string(ic, unit="s").replace(":", ".")
    return f"{ic_str}_mems{mems[0]:04d}-{mems[-1]:04d}"


def write_seed(store: NetCDF4Backend, mems: np.ndarray, seed: int) -> None:
    """Write the random seed of a batch, ``-1`` marks members without valid data.

    Parameters
    ----------
    store : NetCDF4Backend
        Store of the batch, with a ``random_seed`` array
    mems : np.ndarray
        Members of the batch
    seed : int
        Seed used for the batch, or ``-1`` if the batch failed
    """
    store.write(
        xr.DataArray(
            np.full(len(mems), seed, dtype=np.int64),
            dims=("ensemble",),
            coords={"ensemble": mems},
            name="random_seed",
        )
    )


def initialise_netcdf_output(
    cfg: DictConfig,
    model: PrognosticModel,
    ic: np.datetime64,
    mems: np.ndarray,
    seed: int,
) -> NetCDF4Backend:
    """Create the NetCDF4 store of one batch, overwriting an existing one.

    The file name encodes the IC and the batch's member range. The batch's
    random seed is written into the store alongside the forecast arrays.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.
    model : PrognosticModel
        Prognostic model whose output grid is stored.
    ic : np.datetime64
        Initial condition of the batch.
    mems : np.ndarray
        Members of the batch.
    seed : int
        Random seed of the batch.

    Returns
    -------
    NetCDF4Backend
        Initialised storage backend ready for writing.
    """
    file_name = os.path.join(cfg.store_dir, cfg.project).removesuffix(".nc")
    file_name = f"{file_name}_{batch_name(ic, mems)}.nc"

    store = NetCDF4Backend(
        file_name=file_name, backend_kwargs={"mode": "w", "diskless": False}
    )

    # random seed of each member, an array named after itself rather than by a
    # variable label
    store.add_array(
        coord_array(
            ("ensemble",), {"ensemble": mems}, dtype=np.int64, name="random_seed"
        )
    )
    write_seed(store, mems, seed)

    store.add_array(build_output_template(cfg, model, np.array([ic]), mems))

    return store


def batch_zarr_path(cfg: DictConfig, ic: np.datetime64, mems: np.ndarray) -> str:
    """Path of the Zarr store of one batch.

    Under ``<project>_batches/`` for an ensemble, merged at the end; for
    reproduced members ``<project>_<batch>.zarr``, kept as it is, like the
    NetCDF files.
    """
    name = batch_name(ic, mems) + ".zarr"
    if cfg.mode == "reproduce_members":
        project = cfg.project.removesuffix(".zarr")
        return os.path.join(cfg.store_dir, f"{project}_{name}")
    return os.path.join(batch_zarr_dir(cfg), name)


def open_batch_zarr(
    cfg: DictConfig, model: PrognosticModel, ic: np.datetime64, mems: np.ndarray
) -> AsyncZarrBackend:
    """Create the Zarr store of one batch, replacing an existing one.

    Writes run in background threads. Chunks hold one member, IC, lead time
    and variable. With ``zarr_shard_per_member`` (default ``True``), all lead
    times of a member, IC and variable form one shard, which is kept in host
    memory until the member is complete and then written at once. Without,
    every chunk is written as its own file as soon as it is computed.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.
    model : PrognosticModel
        Prognostic model whose output grid is stored.
    ic : np.datetime64
        Initial condition of the batch.
    mems : np.ndarray
        Members of the batch.

    Returns
    -------
    AsyncZarrBackend
        Store ready for writing; close it with :func:`close_batch_zarr`.
    """
    path = batch_zarr_path(cfg, ic, mems)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    template = build_output_template(cfg, model, np.array([ic]), mems)
    lead_times = template.coords["lead_time"].values
    shard = cfg.get("zarr_shard_per_member", True)
    store = AsyncZarrBackend(
        file_name=path,
        parallel_coords={
            "ensemble": np.asarray(mems),
            "time": np.array([ic]),
            "lead_time": lead_times,
        },
        blocking=False,
        shard_coords={"lead_time": len(lead_times)} if shard else {},
    )
    store.add_array(template)
    return store


def close_batch_zarr(
    store: AsyncZarrBackend,
    path: str,
    ic: np.datetime64,
    mems: np.ndarray,
    seed: int,
    n_seeds_tried: int,
) -> None:
    """Wait for the writes of a batch store, then record its seed.

    Parameters
    ----------
    store : AsyncZarrBackend
        Store of the batch
    path : str
        Path of the store
    ic : np.datetime64
        Initial condition of the batch
    mems : np.ndarray
        Members of the batch
    seed : int
        Seed used for the batch, ``-1`` if the batch failed
    n_seeds_tried : int
        Number of seeds the batch has been run with
    """
    then = time.time()
    store.close()
    logger.info(f"waited {time.time() - then:.1f}s for the writes of {path}")
    mark_batch_zarr(path, ic, mems, seed, n_seeds_tried)


def mark_batch_zarr(
    path: str, ic: np.datetime64, mems: np.ndarray, seed: int, n_seeds_tried: int
) -> None:
    """Record the seed of a finished batch store, which marks it as complete.

    A batch that failed with all its seeds gets ``random_seed = -1`` and keeps
    no fields, only these attributes.
    """
    if seed == -1:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
    group = zarr.open_group(path, mode="a")
    group.attrs.update(
        {
            "ic": np.datetime_as_string(ic, unit="s"),
            "members": [int(mem) for mem in mems],
            "random_seed": int(seed),
            "n_seeds_tried": n_seeds_tried,
            "complete": True,
        }
    )

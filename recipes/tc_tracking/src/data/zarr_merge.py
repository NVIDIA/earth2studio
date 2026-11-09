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

"""Merge the per-batch Zarr stores of a run into one store.

Every batch writes its own store (see :func:`open_batch_zarr`), so that any
rank can run any batch, also a retry. The batch stores and the combined store
share their layout: one array per variable, chunks (or shards) of a single
member and IC. A chunk or shard file of a batch therefore is a file of the
combined store too, only under a key with shifted member and IC indices. The
merge hard-links these files instead of copying the data.
"""

import glob
import json
import os
import shutil
from typing import Any

import numpy as np
import xarray as xr
import zarr
from loguru import logger
from omegaconf import DictConfig

from earth2studio.io import AsyncZarrBackend
from earth2studio.utils.coords import coord_array
from src.data.tc_hunt_file_output import batch_zarr_dir, combined_zarr_path
from src.tc_hunt_utils import set_initial_times

# array metadata that has to agree between batch and combined store
_LAYOUT_KEYS = ("data_type", "chunk_grid", "codecs", "fill_value", "dimension_names")


def _metadata(store_path: str, name: str) -> dict[str, Any]:
    with open(os.path.join(store_path, name, "zarr.json")) as f:
        return json.load(f)


def _data_arrays(store_path: str) -> list[str]:
    """Names of the field arrays, i.e. all arrays that are not coordinates."""
    group = zarr.open_group(store_path, mode="r")
    return sorted(
        name
        for name in group.array_keys()
        if name not in _metadata(store_path, name).get("dimension_names", [name])
    )


def _create_combined(
    path: str, batch: str, ics: np.ndarray, members: np.ndarray
) -> None:
    """Create the combined store with the layout of a batch store.

    The same backend and settings as for the batch stores make the codecs,
    chunks and shards identical, only the shape covers all members and ICs.
    """
    fields = _data_arrays(batch)
    ds = xr.open_zarr(batch, consolidated=False)
    dims = list(ds[fields[0]].dims)
    dims.insert(dims.index("lead_time") + 1, "variable")
    coords = {dim: ds.coords[dim].values for dim in dims if dim in ds.coords}
    coords.update(ensemble=members, time=ics, variable=np.array(fields))
    template = coord_array(dims, coords, dtype=ds[fields[0]].dtype)

    sharded = _metadata(batch, fields[0])["codecs"][0]["name"] == "sharding_indexed"
    lead_times = coords["lead_time"]
    store = AsyncZarrBackend(
        file_name=path,
        parallel_coords={"ensemble": members, "time": ics, "lead_time": lead_times},
        shard_coords={"lead_time": len(lead_times)} if sharded else {},
    )
    store.add_array(template)
    store.close()

    # seed of every member, -1 for members without valid data
    zarr.open_group(path, mode="a").create_array(
        "random_seed",
        shape=(len(members), len(ics)),
        chunks=(len(members), len(ics)),
        dtype=np.int64,
        fill_value=-1,
        dimension_names=["ensemble", "time"],
    )[...] = -1


def _link_batch(batch: str, path: str, ics: np.ndarray) -> tuple[np.ndarray, int]:
    """Hard-link the chunk files of a batch into the combined store.

    Returns
    -------
    tuple[np.ndarray, int]
        Indices of the batch's members and of its IC in the combined store
    """
    ds = xr.open_zarr(batch, consolidated=False)
    members = ds.coords["ensemble"].values
    ic_index = int(np.flatnonzero(ics == ds.coords["time"].values[0])[0])

    for name in _data_arrays(batch):
        meta = _metadata(batch, name)
        combined = _metadata(path, name)
        for key in _LAYOUT_KEYS:
            if meta.get(key) != combined.get(key):
                raise ValueError(
                    f"{batch}: array {name} differs from the combined store in {key}"
                )
        dims = meta["dimension_names"]
        member_axis, ic_axis = dims.index("ensemble"), dims.index("time")

        chunk_dir = os.path.join(batch, name, "c")
        for src in glob.glob(os.path.join(chunk_dir, "**", "*"), recursive=True):
            if os.path.isdir(src):
                continue
            parts = os.path.relpath(src, chunk_dir).split(os.sep)
            parts[member_axis] = str(int(members[int(parts[member_axis])]))
            parts[ic_axis] = str(ic_index)
            dst = os.path.join(path, name, "c", *parts)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                os.remove(dst)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    return members, ic_index


def merge_batch_stores(cfg: DictConfig) -> None:
    """Merge all complete batch stores of a run into the combined store.

    Creates the combined store if it does not exist, otherwise adds the
    batches to it, replacing the members they contain. Batch stores are
    removed once merged; incomplete ones (e.g. of an interrupted run) are kept
    and logged.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config of the run, providing ``store_dir``, ``project``,
        ``ensemble_size`` and the ICs
    """
    path = combined_zarr_path(cfg)
    ics = set_initial_times(cfg)
    members = np.arange(cfg.ensemble_size)

    complete: list[str] = []
    incomplete: list[str] = []
    for batch in sorted(glob.glob(os.path.join(batch_zarr_dir(cfg), "*.zarr"))):
        attrs = zarr.open_group(batch, mode="r").attrs
        (complete if attrs.get("complete", False) else incomplete).append(batch)
    for batch in incomplete:
        logger.warning(f"batch store {batch} is incomplete, not merged")

    with_fields = [
        batch
        for batch in complete
        if zarr.open_group(batch, mode="r").attrs["random_seed"] != -1
    ]
    if not os.path.exists(path):
        if not with_fields:
            logger.error(f"no batch with fields to merge into {path}")
            return
        _create_combined(path, with_fields[0], ics, members)

    seeds = zarr.open_array(os.path.join(path, "random_seed"), mode="r+")
    values = seeds[...]
    for batch in complete:
        attrs = zarr.open_group(batch, mode="r").attrs
        ic_index = int(np.flatnonzero(ics == np.datetime64(attrs["ic"]))[0])
        if attrs["random_seed"] == -1:
            values[attrs["members"], ic_index] = -1
        else:
            _link_batch(batch, path, ics)
            values[attrs["members"], ic_index] = attrs["random_seed"]
    seeds[...] = values

    zarr.consolidate_metadata(path)
    for batch in complete:
        shutil.rmtree(batch)
    if os.path.isdir(batch_zarr_dir(cfg)) and not os.listdir(batch_zarr_dir(cfg)):
        os.rmdir(batch_zarr_dir(cfg))

    missing = [
        f"{np.datetime_as_string(ic, unit='s')}: {np.flatnonzero(values[:, ii] == -1).tolist()}"
        for ii, ic in enumerate(ics)
        if (values[:, ii] == -1).any()
    ]
    logger.info(f"merged {len(complete)} batch stores into {path}")
    if missing:
        logger.warning(
            "members without valid data (failed or not run): " + "; ".join(missing)
        )

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

"""Test reproducibility metadata for every supported IO backend.

The recipe supports in-memory backends, exported to NetCDF at the end of a
checkpoint / IC pair, and store backends, which stream to disk during inference.
Both kinds are exercised with tiny synthetic data, so neither a GPU nor a model
checkpoint is needed.

``thread_io`` only changes the code path for the in-memory backends. For
ZarrBackend and NetCDF4Backend both settings run identical code; those cases guard
against the UnboundLocalError the threaded branch used to raise.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from netCDF4 import Dataset
from omegaconf import DictConfig, OmegaConf
from src.hens_utilities import initialize_output_structures, write_to_disk

from earth2studio.io import KVBackend, NetCDF4Backend, XarrayBackend, ZarrBackend

NENSEMBLE = 4
BATCH_SIZE = 2
IC = "2024-09-24T12:00:00"
PROJECT = "metadata_test"
PACKAGE = "sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16"
SEED = 377778

# batch_ids for ensemble members [0, 1, 2, 3] at nensemble=4, batch_size=2
EXPECTED_BATCH_IDS = [0, 0, 1, 1]

# ensemble, time, lead_time, lat, lon
SHAPE = (NENSEMBLE, 1, 2, 4, 4)

VARIABLES = ("t2m", "u10m")


def _coords() -> OrderedDict:
    """Output coordinates of a tiny two-variable, two-step forecast."""
    return OrderedDict(
        [
            ("ensemble", np.arange(NENSEMBLE)),
            ("time", np.array([np.datetime64(IC)])),
            ("lead_time", np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])),
            ("lat", np.linspace(-60, 75, 4)),
            ("lon", np.linspace(0, 270, 4)),
        ]
    )


def _expected_field(name: str) -> np.ndarray:
    """Deterministic per-variable test field.

    Every element is distinct and non-zero, so that a regression which replaces the
    forecast with zeros, truncates it, or swaps two variables is detectable by value
    and not only by name and shape.
    """
    offset = {"t2m": 1.0, "u10m": 10_000.0}[name]
    return (
        np.arange(int(np.prod(SHAPE)), dtype=np.float32).reshape(SHAPE) + offset
    ).astype(np.float32)


def _make_cfg(out_dir: Path, thread_io: bool) -> DictConfig:
    """Minimal config holding only the keys ``write_to_disk`` reads."""
    return OmegaConf.create(
        {
            "project": PROJECT,
            "nensemble": NENSEMBLE,
            "batch_size": BATCH_SIZE,
            # deliberately no `random_seed`: it is optional in the recipe and is
            # generated at runtime when absent, so the effective seed has to be
            # passed in explicitly.
            "file_output": {"path": str(out_dir), "thread_io": thread_io},
        }
    )


def _make_backend(
    backend: str, out_base: Path
) -> KVBackend | NetCDF4Backend | XarrayBackend | ZarrBackend:
    """Build one io backend and fill it the way the ensemble loop would."""
    if backend == "zarr":
        io: KVBackend | NetCDF4Backend | XarrayBackend | ZarrBackend = ZarrBackend(
            file_name=str(out_base) + ".zarr"
        )
    elif backend == "netcdf4":
        io = NetCDF4Backend(file_name=str(out_base) + ".nc")
    elif backend == "xarray":
        io = XarrayBackend()
    elif backend == "kv":
        io = KVBackend()
    else:
        raise ValueError(f"unknown backend {backend}")

    coords = _coords()
    io.add_array(coords, list(VARIABLES))
    for name in VARIABLES:
        io.write(torch.from_numpy(_expected_field(name)), coords, name)
    return io


def _run(tmp_path: Path, backend: str, thread_io: bool) -> tuple[dict, dict]:
    """Fill a backend, finalise it, and read the result back from disk.

    Reading back rather than inspecting the live object is the point: it is what
    proves the metadata was persisted.
    """
    out_base = tmp_path / "global" / f"{PROJECT}_{IC[:13]}_pkg_seed16"
    out_base.parent.mkdir(parents=True, exist_ok=True)
    io = _make_backend(backend, out_base)

    cfg = _make_cfg(tmp_path, thread_io)
    # Build the writer pool exactly as the recipe entry point does.
    _, writer_executor, writer_threads = initialize_output_structures(cfg)
    write_to_disk(
        cfg,
        IC,
        {"package": PACKAGE},
        {"global": io},
        writer_executor,
        writer_threads,
        SEED,
    )
    if writer_executor is not None:
        for thread in writer_threads:
            thread.result()
        writer_executor.shutdown()

    if backend == "zarr":
        with xr.open_zarr(str(out_base) + ".zarr") as ds:
            return dict(ds.attrs), {k: v.values for k, v in ds.data_vars.items()}
    if backend == "netcdf4":
        with Dataset(str(out_base) + ".nc", "r") as ds:
            attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
            return attrs, {k: v[:] for k, v in ds.variables.items() if k in VARIABLES}
    with xr.open_dataset(str(out_base) + ".nc") as ds:
        return dict(ds.attrs), {k: v.values for k, v in ds.data_vars.items()}


@pytest.mark.parametrize("thread_io", [False, True])
@pytest.mark.parametrize("backend", ["zarr", "netcdf4", "xarray", "kv"])
def test_write_to_disk_metadata_and_forecast(
    tmp_path: Path, backend: str, thread_io: bool
) -> None:
    attrs, data_vars = _run(tmp_path, backend, thread_io)

    assert attrs["model_package"] == PACKAGE
    assert attrs["batch_size"] == BATCH_SIZE
    assert attrs["nensemble"] == NENSEMBLE
    assert attrs["random_seed"] == str(SEED)
    assert str(attrs["torch_version"])
    assert np.asarray(attrs["batch_ids"]).tolist() == EXPECTED_BATCH_IDS

    # The forecast has to survive alongside the metadata. For NetCDF4Backend the
    # in-memory export path targets the very file the backend streamed into, so a
    # regression letting it fall through to to_netcdf would replace the forecast with
    # an empty dataset while every assertion above still passed.
    for name in VARIABLES:
        assert name in data_vars, f"forecast variable '{name}' is missing"
        got = np.asarray(data_vars[name], dtype=np.float32)
        want = _expected_field(name)
        assert got.shape == want.shape
        assert np.array_equal(got, want), f"{name} values differ from what was written"

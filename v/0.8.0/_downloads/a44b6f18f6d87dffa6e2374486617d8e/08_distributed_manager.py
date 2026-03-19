# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

# %%
"""
Distributed Manager Inference
=============================

Setting up distributed manager for parallel inference.

Many inference workflows are embarrassingly parallel and can be easily sharded across
multiple devices.
This example demonstrates how one can use the PhysicsNeMo distributed manager to distribute
inference across mutliple GPUs.
The `distributed manager <https://github.com/NVIDIA/physicsnemo/blob/main/physicsnemo/distributed/manager.py>`_
is a utility that provides a useful set of properties that pertain to a parallel
environment.

In this example you will learn:

- How to use the distributed manager to access parallel environment properties
- Parallelize deterministic inference across multiple initial date-times
- Limitations of parallel inference in Earth2Studio
- Post-processing strategies of parallel job outputs
"""

# %%
# Set Up
# ------
# Set up the distributed manager by initializing it. Out of the box, the distributed
# manager supports MPI, SLURM and PyTorch parallel environments which provide information
# regarding the parallel enviroment but environment variables.
#
# For example, this script could be ran using:
#
# .. code-block:: bash
#
#   # OpenMPI
#   mpirun -np 2 python 08_distributed_manager.py
#   # Torch run
#   torchrun --standalone --nnodes=1 --nproc-per-node=2 08_distributed_manager.py
#
# .. warning::
#
#   When running in parallel, make sure the .env file in the repository examples
#   folder is *not* present. The .env file is intended for the documentation build only.

# %%
import os

os.makedirs("outputs", exist_ok=True)
from dotenv import load_dotenv

load_dotenv()  # TODO: make common example prep function

import numpy as np
import torch
from loguru import logger
from physicsnemo.distributed import DistributedManager

DistributedManager.initialize()  # Only call this once in the entire script!
dist = DistributedManager()
assert (  # noqa: S101
    dist._distributed
), "Looks like torch distributed isn't set up. Check your env variables!"

logger.info(
    f"Inference runner {dist.rank} of {dist.world_size} with device {dist.device}"
)
# %%
# Next the needed components get initialized.
# Rigorous parallel support is not part of Earth2Studio's design goals, there are some
# spots where potential race conditions can occur. Thus some additional care should be
# taken to ensure safe parallel inference.

# %%
from earth2studio.data import WB2ERA5
from earth2studio.io import ZarrBackend
from earth2studio.models.px import DLWP

# Load model
package = DLWP.load_default_package()
if dist.rank == 0:
    model = DLWP.load_model(package)

torch.distributed.barrier()
if dist.rank != 0:
    model = DLWP.load_model(package)

# %%
# When loading models that are built into Earth2Studio, the model's checkpoint files
# will be downloaded into the machines cache. If each inference process has access to
# the same cache location, then only one should download the checkpoint triggered by
# :py:func:`load_model`.
#
# Here :py:class:`earth2studio.models.px.DLWP` checkpoint files are first downloaded by
# process 0 and then loaded by other processes.

# %%

# Create the data source
data = WB2ERA5()

# %%
# The remote date store will place cached data into separate caches for each process.
# This makes the download of initial state data safe during parallel inference but also
# means that multiple jobs will download the same date-time if needed.

# %%
chunks = {"time": 1, "lead_time": 1}
io = ZarrBackend(
    file_name=f"outputs/08_output_{dist.rank}.zarr",
    chunks=chunks,
    backend_kwargs={"overwrite": True},
)

# %%
# Earth2Studio does not provide distributed IO support. The recommendation is to always
# output data for each process to a separate file, then aggregate the data during post
# processing.
#
# Execute the Workflow
# --------------------
# Next we can run the workflow. This example will run inference for a random date across
# several years and just save total column water vapor. Shard the initial date-times
# across the each process. The distributed manager will provide the device ID for the
# process.

# %%
import earth2studio.run as run

times = np.array([f"200{i:d}-06-01T00:00:00" for i in range(0, 6)])
assert (  # noqa: S101
    len(times) > dist.world_size
), "Inference runs should be greater than processes"
time_shard = np.array_split(times, dist.world_size)[dist.rank]

nsteps = 20
output_coords = {"variable": np.array(["tcwv"])}
io = run.deterministic(
    time_shard, nsteps, model, data, io, output_coords=output_coords, device=dist.device
)

print(io.root.tree())
torch.distributed.barrier()

# %%
# Post Processing
# ---------------
# Finally, we can post process the results. Xarray provides a useful function for
# opening multiple files as a single dataset, :py:func:`xarray.open_mfdataset`. This
# allows outputs from all processes to get treated as a single data array.
#
# .. warning::
#
#   In this script process 0 is used to post process so the example is in one file.
#   It is best practice to perform post processing in a separate job / script entirely
#   to better utilize compute resources.

if dist.rank == 0:
    import matplotlib.pyplot as plt
    import xarray as xr

    from earth2studio.utils.time import timearray_to_datetime

    paths = [f"outputs/08_output_{i}.zarr" for i in range(dist.world_size)]
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="time", engine="zarr")
    print(ds)

    ncols = 3
    fig, ax = plt.subplots(2, ncols, figsize=(12, 6))

    time = timearray_to_datetime(ds.coords["time"].values.astype("datetime64[ns]"))
    for i in range(6):
        ax[i // ncols, i % ncols].imshow(
            ds["tcwv"].isel(time=i, lead_time=-1).values,
            cmap="gist_earth",
            vmin=0,
            vmax=100,
        )
        ax[i // ncols, i % ncols].set_title(time[i].strftime("%m/%d/%Y"))
    plt.suptitle(
        f'TCWV Forecast Lead Time - {ds.coords["lead_time"].values[-1].astype("timedelta64[ns]").astype("timedelta64[D]").astype(int)} days'
    )
    plt.savefig("outputs/08_tcwv_distributed_manager.jpg")

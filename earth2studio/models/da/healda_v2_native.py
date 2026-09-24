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
"""HealDA-v2 inference delegated to the healda package (single source of truth).

Unlike :py:class:`~earth2studio.models.da.healda_v2.HealDAv2`, which is a
self-contained port, this wrapper imports the ``healda`` package and reuses the
exact model construction, checkpoint loading, observation pipeline and decode
that training used. Earth2Studio contributes only the model API: the Hugging
Face package plumbing and an xarray interface.

The published `nvidia/healda-v2 <https://huggingface.co/nvidia/healda-v2>`_
checkpoint is the 18-level 0.25-degree lat-lon arm trained on the NNJA
observing system; its raw healda training state is loaded directly, so no
conversion step is needed.

Requirements (this wrapper does not run without them):

- the ``healda`` package, e.g.
  ``pip install git+ssh://git@gitlab-master.nvidia.com:12051/earth-2/healda.git``
- a ``.env`` in the working directory carrying the observation archive paths
  (``NNJA_ROOT``, ``ERA5_HOURLY_73CH``, ...), as for any healda run
- one GPU
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from collections import OrderedDict
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import healda.utils.distributed as healda_dist
    from healda.cli.train import LOOPS
    from healda.datasets.da import state_masks, state_transforms
    from healda.datasets.da.transform import collate as collate_v2
    from healda.training import distributed_checkpoint
except ImportError:
    OptionalDependencyFailure("da-healda-native")
    LOOPS = None

LATLON_NLAT, LATLON_NLON = 721, 1440

DEFAULT_LOOP = "v2-videoDA-nnja-nnjaConv-104ch-windDrop50-latlon025-18L-baseL7-aga-tp8-dp018-obsAll"


@check_optional_dependencies()
class HealDAv2Native(torch.nn.Module, AutoModelMixin):
    """HealDA-v2 data assimilation, run through the healda package itself.

    The analysis is produced exactly as in healda's own inference: the loop's
    dataset assembles the 8-frame observation window (loaders, QC,
    normalization, thinning and metadata identical to training), the network
    runs one deterministic bf16 forward, and the output is denormalized and
    mapped to physical space with the loop's own statistics and transforms.

    Note
    ----
    This wrapper runs single-GPU (``time_parallel=1``). The reference scoring
    runs shard the 8 frames over 4 GPUs; the different collective layout
    changes the bf16 summation order, so fields differ from the reference at
    numerical-noise level (well below the 0.1 % score gate).

    Parameters
    ----------
    loop : healda TrainingLoop
        A set-up loop with the checkpoint loaded
    dataset : torch.utils.data.Dataset
        The loop's inference dataset (train=False: no observation dropout)
    """

    def __init__(self, loop: Any, dataset: Any) -> None:
        super().__init__()
        self._loop = loop
        self._dataset = dataset
        dt = dataset.times[1] - dataset.times[0]
        self._final_times = pd.DatetimeIndex(
            dataset.times[: len(dataset)] + (dataset.time_length - 1) * dt
        )
        self._channels = list(loop.batch_info.channels)
        self._scale = torch.tensor(loop.batch_info.scales)[:, None, None].to(
            loop.device
        )
        self._mean = torch.tensor(loop.batch_info.center)[:, None, None].to(loop.device)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default model package from Hugging Face."""
        return Package(
            "hf://nvidia/healda-v2",
            cache_options={"same_names": True},
        )

    @classmethod
    def load_model(
        cls,
        package: Package,
        loop_name: str = DEFAULT_LOOP,
        years: list[int] | None = None,
        compile_dit: bool = False,
    ) -> "HealDAv2Native":
        """Build the healda loop, load the raw training checkpoint, set up
        the inference dataset.

        Parameters
        ----------
        package : Package
            Package holding ``healda_v2.checkpoint`` (a healda training state)
        loop_name : str, optional
            Name in healda's ``LOOPS``; supplies model geometry and the
            observation configuration, by default the published 18L arm
        years : list[int] | None, optional
            Years the observation dataset must cover, by default the current
            NNJA archive span configured in the environment
        compile_dit : bool, optional
            Compile the backbone (faster steady state, slow first call),
            by default False
        """
        # Single-process torch.distributed, as healda's setup expects a group.
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29511")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        healda_dist.init(timeout_infinite=True)

        loop = LOOPS[loop_name]
        loop.run_dir = tempfile.mkdtemp(prefix="healda_v2_native_")
        # Single-GPU inference carries the whole observation window on one
        # rank; the fused FiLM Triton kernel faults at that volume (the
        # reference runs shard it 4-way). The pure-torch tokenizer path uses
        # the same weights, so outputs are unchanged up to bf16 numerics.
        if loop.sensor_embedder_config is not None:
            loop.sensor_embedder_config = dataclasses.replace(
                loop.sensor_embedder_config, use_fused_mlp=False
            )
        loop.batch_size = 1
        loop.batch_gpu = 1
        loop.fsdp = False
        loop.time_parallel = 1
        loop.dataloader_num_workers = 0
        loop.compile_dit = compile_dit
        loop.setup_datasets = False
        loop.setup()

        checkpoint_path = package.resolve("healda_v2.checkpoint")
        logger.info(f"Loading healda training state from {checkpoint_path}")
        distributed_checkpoint.load(
            checkpoint_path, loop.net, optimizer=None, require_all=True
        )
        loop.net.eval()

        dataset = loop.get_dataset(train=False, years=years)
        return cls(loop, dataset)

    @property
    def analysis_times(self) -> pd.DatetimeIndex:
        """Valid times the dataset can produce an analysis for."""
        return self._final_times

    def __call__(self, time: np.datetime64 | str | pd.Timestamp) -> xr.DataArray:
        """Produce the analysis window ending at ``time``.

        Parameters
        ----------
        time : np.datetime64 | str | pd.Timestamp
            Analysis valid time (the final window frame)

        Returns
        -------
        xr.DataArray
            Physical-space analysis window with dimensions
            [time, lead_time, variable, lat, lon]; the analysis frame is
            ``lead_time == 0``.
        """
        loop = self._loop
        when = pd.Timestamp(time)
        matches = np.flatnonzero(self._final_times == when)
        if matches.size == 0:
            raise ValueError(
                f"{when} is not an available analysis time; see analysis_times"
            )
        index = int(matches[0])

        # healda datasets fetch in batches (__getitems__), like the DataLoader does
        batch = collate_v2(self._dataset.__getitems__([index]))
        batch = loop._device_transform(batch, transform=self._dataset.transform)

        condition = batch["condition"]
        noise_labels = torch.zeros([1], device=loop.device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = loop.net(
                condition,
                noise_labels=noise_labels,
                class_labels=batch.get("labels"),
                second_of_day=batch["second_of_day"],
                day_of_year=batch["day_of_year"],
                unified_obs=batch.get("unified_obs"),
                timestamp=batch["timestamp"],
                is_causal=loop.dit_temporal_attention_causal,
            )
        pred = prediction.out
        if pred.ndim == 5:
            pred = pred.flatten(3)

        with torch.no_grad():
            state = pred.float() * self._scale + self._mean
            state = state_transforms.to_physical_space(
                state, self._channels, loop.variable_config.name, channel_axis=1
            )
            if loop.latlon_decode:
                shape = state.shape
                masked = [c for c in state_masks.CHANNEL_DOMAIN if c in self._channels]
                if masked:
                    grid = state.reshape(*shape[:-1], LATLON_NLAT, LATLON_NLON)
                    state = state_transforms.restore_fill(
                        grid,
                        self._channels,
                        state_masks.LATLON_025,
                        None,
                        channel_axis=1,
                    ).reshape(shape)

        time_step = self._dataset.times[1] - self._dataset.times[0]
        lead = (
            (np.arange(loop.time_length) - (loop.time_length - 1))
            * pd.Timedelta(time_step)
        ).astype("timedelta64[ns]")
        values = state.reshape(
            1, len(self._channels), loop.time_length, LATLON_NLAT, LATLON_NLON
        )
        coords: OrderedDict[str, Any] = OrderedDict(
            time=np.array([np.datetime64(when, "ns")]),
            lead_time=lead,
            variable=np.array(self._channels),
            lat=np.linspace(90.0, -90.0, LATLON_NLAT, dtype=np.float32),
            lon=np.linspace(0.0, 360.0, LATLON_NLON, endpoint=False, dtype=np.float32),
        )
        return xr.DataArray(
            values.permute(0, 2, 1, 3, 4).cpu().numpy(),
            dims=list(coords),
            coords=coords,
        )

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
import re
from collections import OrderedDict

import hydra
import numpy as np
import torch
import xarray as xr
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from earth2studio.data import DataSource
from earth2studio.data.utils import datasource_cache_root
from src.utils import run_with_rank_ordered_execution


def resolve_oro_path(oro_path: str) -> str:
    """Return a local file path for the orography dataset.

    If *oro_path* is a HuggingFace URL the file is downloaded into the
    Earth2Studio cache under ``tc_hunt/`` and the cached path is returned.
    Local paths are returned unchanged.
    """

    hf_url_pattern = re.compile(
        r"^https://huggingface\.co/(?P<repo>[^/]+/[^/]+)/blob/(?P<revision>[^/]+)/(?P<path>.+)$"
    )
    match = hf_url_pattern.match(oro_path)
    if match is None:
        return oro_path

    repo_id = match.group("repo")
    revision = match.group("revision")
    filename = match.group("path")

    cache_dir = os.path.join(datasource_cache_root(), "tc_hunt")
    return hf_hub_download(repo_id, filename, revision=revision, local_dir=cache_dir)


def load_heights(oro_path: str) -> tuple[torch.Tensor, OrderedDict]:
    """Load orography data and convert geopotential to height.

    Parameters
    ----------
    oro_path : str
        Path to the orography dataset (NetCDF / Zarr) or a HuggingFace URL
        (e.g. ``https://huggingface.co/nvidia/fourcastnet3/blob/main/orography.nc``).
        When a URL is provided the file is downloaded into the Earth2Studio
        cache under ``tc_hunt/``.

    Returns
    -------
    tuple[torch.Tensor, OrderedDict]
        Height field as a tensor and the corresponding coordinate mapping
        with keys *variable*, *lat*, *lon*.
    """
    oro_path = run_with_rank_ordered_execution(resolve_oro_path, oro_path)
    oro = xr.load_dataset(oro_path)

    coords = OrderedDict(
        {
            "variable": np.array(["height"]),
            "lat": oro.latitude.values,
            "lon": oro.longitude.values,
        }
    )
    geop = (
        torch.Tensor(oro["Z"].to_numpy()) / 9.80665
    )  # divide by gravity to get height

    return geop, coords


class DataSourceManager:
    """Select the right :class:`DataSource` for a given forecast time.

    Supports either a single data source (simple Hydra target) or a mapping
    of named sources to year ranges, allowing different datasets for
    different periods, which is eg useful if data is split up in train/valid/test.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.  Must contain a ``data_source`` key that is
        either a single Hydra-instantiable target or a mapping of named
        sources each with ``source`` and ``years`` sub-keys.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # build dict of data sources
        self.data_sources = {}
        self.data_source_mapping = {}
        self.single_source = False
        for name, source in self.cfg.data_source.items():
            # catch classic case of single source
            if name == "_target_":
                self.single_source = True
                self.data_source = hydra.utils.instantiate(self.cfg.data_source)
                return

            # instantiate data source
            self.data_sources[name] = hydra.utils.instantiate(source.source)

            # extract valid years for source and build mapping
            years = source.years
            if isinstance(years, str):
                if "-" not in years:
                    years = int(years)

            if isinstance(years, int):
                years = [years]
            else:
                years = list(
                    range(int(years.split("-")[0]), int(years.split("-")[1]) + 1)
                )

            for year in years:
                self.data_source_mapping[year] = name

        return

    def select_data_source(self, time_stamps: np.datetime64 | np.ndarray) -> DataSource:
        """Return the data source appropriate for time_stamps.

        Parameters
        ----------
        time_stamps : np.datetime64 | np.ndarray
            One or more timestamps whose year determines the source.

        Returns
        -------
        DataSource
            The matching data source instance.

        Raises
        ------
        ValueError
            If the timestamps span multiple years or no source is
            configured for the year.
        """
        if self.single_source:
            return self.data_source

        year = np.unique(time_stamps.astype("datetime64[Y]").astype(int) + 1970)

        if len(year) > 1:
            raise ValueError("track spans multiple years, which is not yet supported")
        else:
            year = year[0]

        if year not in self.data_source_mapping:
            raise ValueError(f"no data source provided for year {year}")

        return self.data_sources[self.data_source_mapping[year]]

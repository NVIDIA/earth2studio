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

import asyncio
import os
import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime

import hydra
import numpy as np
import xarray as xr
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from earth2studio.data import DataSource
from earth2studio.data.utils import datasource_cache_root
from earth2studio.run.work import rank0_first
from earth2studio.utils.time import to_time_array
from earth2studio.utils.type import TimeArray, VariableArray


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


def load_heights(oro_path: str) -> xr.DataArray:
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
    xr.DataArray
        Surface height in metres, named ``height``, with dimensions
        ``(lat, lon)``.
    """
    # rank 0 downloads the file, the others then find it in the cache
    oro_path = rank0_first(lambda: resolve_oro_path(oro_path))
    oro = xr.load_dataset(oro_path)

    # divide geopotential by gravity to get height
    geop = oro["Z"].squeeze(drop=True).values / 9.80665

    return xr.DataArray(
        geop.astype(np.float32),
        dims=("lat", "lon"),
        coords={"lat": oro.latitude.values, "lon": oro.longitude.values},
        name="height",
    )


def data_source_from_config(cfg: DictConfig) -> DataSource:
    """Build the data source of the ``data_source`` section of a config.

    The section is either a single Hydra target, or named sources, each with
    a ``source`` target and the ``years`` it provides, a single year or a
    range such as ``"2000-2010"``. The latter is useful if the data is split
    by period, e.g. into training, validation and test years.

    Parameters
    ----------
    cfg : DictConfig
        The ``data_source`` section of the Hydra config

    Returns
    -------
    DataSource
        The source, a :class:`YearRoutedSource` for named sources
    """
    if "_target_" in cfg:
        return hydra.utils.instantiate(cfg)

    sources = {}
    for entry in cfg.values():
        source = hydra.utils.instantiate(entry.source)
        first, _, last = str(entry.years).partition("-")
        for year in range(int(first), int(last or first) + 1):
            sources[year] = source
    return YearRoutedSource(sources)


class _SourceWrapper:
    """Base of data sources passing requests on to other sources."""

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch fields, see :class:`~earth2studio.data.DataSource`."""
        raise NotImplementedError

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch fields asynchronously, running the request in a thread.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Times to fetch
        variable : str | list[str] | VariableArray
            Variables to fetch

        Returns
        -------
        xr.DataArray
            The fields
        """
        return await asyncio.to_thread(self, time, variable)


class YearRoutedSource(_SourceWrapper):
    """Data source passing each request on to the source of its year.

    Parameters
    ----------
    sources : Mapping[int, DataSource]
        Source of each year
    """

    def __init__(self, sources: Mapping[int, DataSource]) -> None:
        self.sources = dict(sources)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch the fields from the source of the requested year.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Times to fetch, all in the same year
        variable : str | list[str] | VariableArray
            Variables to fetch

        Returns
        -------
        xr.DataArray
            The fields, as returned by the source of the year

        Raises
        ------
        ValueError
            If the times span several years, or no source provides their year
        """
        years = np.unique(to_time_array(time).astype("datetime64[Y]")).astype(int)
        if len(years) != 1:
            raise ValueError(
                f"times span the years {(years + 1970).tolist()}, request each "
                "year on its own"
            )
        year = int(years[0]) + 1970
        if year not in self.sources:
            raise ValueError(f"no data source configured for the year {year}")
        return self.sources[year](time, variable)


class CachedSource(_SourceWrapper):
    """Data source keeping the fields of its last requests.

    Initial conditions are requested again for every batch of members started
    from them and for every retry of an unstable batch. Keeping the last few
    on the host saves fetching them again.

    Parameters
    ----------
    source : DataSource
        Source to fetch from on a cache miss
    size : int, optional
        Number of requests kept, by default 4. A model with two input times,
        like AIFS-ENS, makes two requests per initial condition.
    """

    def __init__(self, source: DataSource, size: int = 4) -> None:
        self.source = source
        self.size = size
        self._cache: OrderedDict[tuple, xr.DataArray] = OrderedDict()

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return the fields from the cache, or fetch and keep them.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Times to fetch
        variable : str | list[str] | VariableArray
            Variables to fetch

        Returns
        -------
        xr.DataArray
            The fields, shared with the cache: treat them as read-only
        """
        key = (
            to_time_array(time).astype("datetime64[s]").tobytes(),
            tuple(np.atleast_1d(variable).astype(str)),
        )
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = self.source(time, variable)
            while len(self._cache) > self.size:
                self._cache.popitem(last=False)
        return self._cache[key].copy(deep=False)


class SupplementedSource(_SourceWrapper):
    """Data source taking a few variables from a second source.

    For analyses lacking a variable a model needs. In this recipe it is used
    for one case only: WeatherNext 2 reads the sea surface temperature, which
    the IFS produces but ECMWF's open data does not include, so the WN2
    configs take ``sst`` from ERA5 and every other variable from the IFS. The
    supplement is only contacted for its variables: a model not reading them,
    such as FCN3 or AIFS-ENS, gets everything from ``source``. Both sources
    have to provide the same grid.

    Parameters
    ----------
    source : DataSource
        Source of all other variables
    supplement : DataSource
        Source of ``variables``
    variables : Sequence[str]
        Variables fetched from ``supplement``
    """

    def __init__(
        self, source: DataSource, supplement: DataSource, variables: Sequence[str]
    ) -> None:
        self.source = source
        self.supplement = supplement
        self.variables = {str(variable) for variable in variables}

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Fetch the fields, each variable from its source.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Times to fetch
        variable : str | list[str] | VariableArray
            Variables to fetch

        Returns
        -------
        xr.DataArray
            The fields, in the requested order of variables

        Raises
        ------
        ValueError
            If the sources return different grids
        """
        requested = [str(name) for name in np.atleast_1d(variable)]
        supplemented = [name for name in requested if name in self.variables]
        others = [name for name in requested if name not in self.variables]
        if not supplemented:
            return self.source(time, variable)
        if not others:
            return self.supplement(time, variable)

        fields = self.source(time, others)
        extra = self.supplement(time, supplemented)
        for dim in ("lat", "lon"):
            if fields[dim].shape != extra[dim].shape or not np.allclose(
                fields[dim].values, extra[dim].values
            ):
                raise ValueError(f"the sources differ in their {dim} coordinate")
        # coordinates of the main source, so rounding differences do not misalign
        extra = extra.assign_coords(
            time=fields.time, lat=fields.lat, lon=fields.lon
        ).transpose(*fields.dims)
        combined = xr.concat([fields, extra.astype(fields.dtype)], dim="variable")
        return combined.sel(variable=requested)

from collections import OrderedDict

import hydra
import numpy as np
import torch
import xarray as xr
from omegaconf import DictConfig

from earth2studio.data import DataSource


def load_heights(oro_path: str) -> tuple[torch.Tensor, OrderedDict]:
    """Load orography data and convert geopotential to height.

    Parameters
    ----------
    oro_path : str
        Path to the orography dataset (NetCDF / Zarr).

    Returns
    -------
    tuple[torch.Tensor, OrderedDict]
        Height field as a tensor and the corresponding coordinate mapping
        with keys *variable*, *lat*, *lon*.
    """
    oro = xr.load_dataset(oro_path)

    coords = OrderedDict(
        {"variable": np.array(["height"]), "lat": oro.latitude.values, "lon": oro.longitude.values}
    )
    geop = torch.Tensor(oro["Z"].to_numpy()) / 9.80665  # divide by gravity to get height

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
                if not "-" in years:
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
            raise ValueError(f"track spans multiple years, which is not yet supported")
        else:
            year = year[0]

        if year not in self.data_source_mapping:
            raise ValueError(f"no data source provided for year {year}")

        return self.data_sources[self.data_source_mapping[year]]

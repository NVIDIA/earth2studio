import glob
import json
import os
from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Tuple

import h5py
import numpy as np
import xarray as xr

from earth2studio.utils.type import TimeArray, VariableArray

DIMS = ["time", "variable", "lat", "lon"]


class LocalArchiveHDF5:
    def __init__(
        self,
        dirs: str | Iterable,
        metadata_file: str,
        latlon_box: tuple[tuple[float, float], tuple[float, int]] | None = None,
    ):
        if isinstance(dirs, str):
            dirs = [dirs]

        self.files_by_year = {}
        for dir in dirs:
            for path in glob.glob(os.path.join(dir, "????.h5")):
                year = int(os.path.basename(path).split(".")[0])
                self.files_by_year[year] = path

        with open(metadata_file) as f:
            metadata = json.load(f)
        self.lat = np.array(metadata["coords"]["lat"])
        self.lon = np.array(metadata["coords"]["lon"])
        if latlon_box is not None:
            (self.lat, self.lat_slice) = _find_index_range(self.lat, *latlon_box[0])
            (self.lon, self.lon_slice) = _find_index_range(self.lon, *latlon_box[1])
        else:
            self.lat_slice = self.lon_slice = slice(None)
        self.dt = timedelta(hours=metadata["dhours"])
        self.variables = metadata["coords"]["channel"]
        self.field_shape = (len(self.lat), len(self.lon))

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        if isinstance(time, datetime):
            time = [time]
        time = [
            datetime.fromisoformat(str(t)[:19]) if isinstance(t, np.datetime64) else t
            for t in time
        ]
        if isinstance(variable, str):
            variable = [variable]

        var_indices = [self.variables.index(v) for v in variable]

        x = np.zeros((len(time), len(variable)) + self.field_shape, dtype=np.float32)

        for k, t in enumerate(time):
            time_idx = int((t - datetime(t.year, 1, 1)) / self.dt)
            with h5py.File(self.files_by_year[t.year], "r") as ds:
                for i, var_idx in enumerate(var_indices):
                    x[k, i, :, :] = ds["fields"][
                        time_idx, var_idx, self.lat_slice, self.lon_slice
                    ]

        coords = {
            "time": np.array(time, copy=False),
            "variable": np.array(variable, copy=False),
            "lat": self.lat,
            "lon": self.lon,
        }

        return xr.DataArray(data=x, coords=coords, dims=DIMS)


def _find_index_range(seq, x0, x1):
    in_range = (x0 <= seq) & (seq <= x1)
    subseq = np.array(seq[in_range])
    ind = np.arange(len(seq))[in_range]
    i0 = ind[0]
    i1 = ind[-1] + 1

    return (subseq, slice(i0, i1))

from datetime import datetime, timedelta
from string import Template

import numpy as np
from fsspec.implementations.local import LocalFileSystem

from earth2studio.data import GFS, fetch_data
from earth2studio.io.v2.netcdf4 import NetCDF4Backend

io = NetCDF4Backend(
    root="outputs",
    fs=LocalFileSystem(),
    ft=Template("earth2studio_${step}.nc"),
)

data = GFS()
time = np.array([datetime(2024, 1, 1)])
lead_time = np.array([timedelta(0)])
variable = np.array(["t2m"])
device = "cuda"

x0, coords0 = fetch_data(
    source=data,
    time=time,
    variable=variable,
    lead_time=lead_time,
    device=device,
)

io.write(x0, coords0, {"step": 0})


time = np.array([datetime(2024, 2, 1)])
x0, coords0 = fetch_data(
    source=data,
    time=time,
    variable=variable,
    lead_time=lead_time,
    device=device,
)
io.write(x0, coords0, {"step": 1})


time = np.array([datetime(2024, 3, 1)])
x0, coords0 = fetch_data(
    source=data,
    time=time,
    variable=variable,
    lead_time=lead_time,
    device=device,
)
io.write(x0, coords0, {"step": 2})
io.close()


io.consolidate([{"step": i} for i in range(3)], concat_dims=["time"])

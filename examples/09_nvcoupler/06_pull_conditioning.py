# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# %%
"""Pull-pattern coupling with live DataArray conditioning."""

import numpy as np
import xarray as xr

import earth2studio.nvcoupler as nvc
from earth2studio.data.utils import fetch_data
from earth2studio.nvcoupler.testing import grid_coords

GRID = (8, 16)
T0 = np.datetime64("2024-01-01")


class MockPullModel:
    conditioning_variables = np.array(["u10m", "t2m"])

    def __init__(self):
        self.conditioning_data_source = None
        self.pull_log: list[xr.DataArray] = []

    def input_coords(self):
        return xr.DataArray(
            np.empty((1, 1, 1, *GRID)),
            dims=("time", "lead_time", "variable", "lat", "lon"),
            coords={
                "time": [T0],
                "lead_time": [np.timedelta64(0, "h")],
                "variable": ["refc"],
                **grid_coords(*GRID),
            },
        )

    def output_coords(self, input_coords):
        return input_coords.assign_coords(
            lead_time=input_coords.lead_time + np.timedelta64(1, "h")
        )

    def __call__(self, array):
        forcing = fetch_data(
            self.conditioning_data_source,
            time=np.atleast_1d(array.time),
            variable=self.conditioning_variables,
        )
        self.pull_log.append(forcing)
        increment = (
            1.0
            + forcing.sel(variable="u10m").mean()
            + 0.1 * forcing.sel(variable="t2m").mean()
        )
        return (array + increment).assign_coords(self.output_coords(array).coords)

    def to(self, device):
        return self


def global_step(array):
    wind = array.sel(variable="u10m", drop=True) + 1.0
    temperature = array.sel(variable="t2m", drop=True)
    return xr.concat(
        [wind, temperature],
        xr.IndexVariable("variable", ["u10m", "t2m"]),
    )


glob = nvc.CallableComponent(
    "global",
    global_step,
    timestep="1h",
    exports=["eastward_wind_10m", "air_temperature_2m"],
)

dictionary = nvc.FieldDictionary(nvc.DEFAULT_DICTIONARY)
dictionary.register(
    nvc.FieldEntry("radar_reflectivity", "dBZ", aliases=frozenset({"refc"}))
)
model = MockPullModel()
stormcast = nvc.PrognosticComponent(
    "stormcast",
    model,
    imports=["eastward_wind_10m", "air_temperature_2m"],
    exports=["radar_reflectivity"],
    import_adapter=nvc.PullAdapter(),
    variable_aliases={"refc": "radar_reflectivity"},
    dictionary=dictionary,
)

driver = nvc.Driver(
    {"global": glob, "stormcast": stormcast},
    sequence="""
    @1h
      global
      global -> stormcast
      stormcast
    @
    """,
    clock=nvc.Clock(T0, "2024-01-01T04:00", "1h"),
    connectors=[nvc.Connector(glob, stormcast)],
)

global_ic = xr.DataArray(
    np.stack([np.full(GRID, 2.0), np.full(GRID, 280.0)]),
    dims=("variable", "lat", "lon"),
    coords={"variable": ["u10m", "t2m"], **grid_coords(*GRID)},
)
regional_ic = xr.DataArray(
    np.zeros((1, 1, 1, *GRID)),
    dims=("time", "lead_time", "variable", "lat", "lon"),
    coords={
        "time": [T0],
        "lead_time": [np.timedelta64(0, "h")],
        "variable": ["refc"],
        **grid_coords(*GRID),
    },
)
driver.initialize({"global": global_ic, "stormcast": regional_ic})
output = driver.run()

pulled_wind = [float(array.sel(variable="u10m").mean()) for array in model.pull_log]
if pulled_wind != [3.0, 4.0, 5.0, 6.0]:
    raise ValueError("a pull saw stale conditioning")
print(output["stormcast"])

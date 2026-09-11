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

from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr

from earth2studio.data.utils import prep_data_inputs, prep_forecast_inputs
from earth2studio.grids import E2S_GRID_ID, GridDefinition, resolve_grid
from earth2studio.utils.type import FieldArray, LeadTimeArray, TimeArray, VariableArray

Domain: TypeAlias = Mapping[str, np.ndarray] | xr.DataArray | GridDefinition | str


def _domain(domain: Domain) -> tuple[tuple[str, ...], xr.Coordinates, dict[str, Any]]:
    if isinstance(domain, str):
        definition = resolve_grid(domain)
        return definition.dims, definition.coords(), {E2S_GRID_ID: domain}
    if isinstance(domain, GridDefinition):
        return domain.dims, domain.coords(), domain.attrs
    if isinstance(domain, xr.DataArray):
        dimensions = tuple(
            name
            for name in domain.dims
            if name not in {"batch", "time", "lead_time", "variable"}
        )
        coordinates = xr.Coordinates(
            {
                name: coordinate
                for name, coordinate in domain.coords.items()
                if set(coordinate.dims).issubset(dimensions)
            }
        )
        return dimensions, coordinates, dict(domain.attrs)
    coordinates = xr.Coordinates(domain)
    return tuple(domain), coordinates, {}


class Random:
    """A randomly generated normally distributed data. Primarily useful for testing.

    Parameters
    ----------
    domain_coords : Domain
        Spatial coordinates, coordinate signature, or registered grid.
    """

    def __init__(
        self,
        domain_coords: Domain,
    ):
        self.domain_dims, self.domain_coords, self.attrs = _domain(domain_coords)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve random gaussian data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Random data array
        """

        time, variable = prep_data_inputs(time, variable)

        shape = [
            len(time),
            len(variable),
            *(self.domain_coords.sizes[name] for name in self.domain_dims),
        ]
        coords = {"time": time, "variable": variable}

        coords.update(self.domain_coords)
        da = xr.DataArray(
            data=np.random.randn(*shape).astype(np.float32),
            dims=("time", "variable", *self.domain_dims),
            coords=coords,
            attrs=self.attrs,
        )

        return da


class Random_FX:
    """A randomly generated normally distributed data. Primarily useful for testing.

    Parameters
    ----------
    domain_coords : Domain
        Spatial coordinates, coordinate signature, or registered grid.
    """

    def __init__(
        self,
        domain_coords: Domain,
    ):
        self.domain_dims, self.domain_coords, self.attrs = _domain(domain_coords)

    def __call__(  # type: ignore[override]
        self,
        time: datetime | list[datetime] | TimeArray,
        lead_time: timedelta | list[timedelta] | LeadTimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Retrieve random gaussian data.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.

        Returns
        -------
        xr.DataArray
            Random data array
        """

        time, lead_time, variable = prep_forecast_inputs(time, lead_time, variable)

        shape = [
            len(time),
            len(lead_time),
            len(variable),
            *(self.domain_coords.sizes[name] for name in self.domain_dims),
        ]
        coords = {"time": time, "lead_time": lead_time, "variable": variable}
        coords.update(self.domain_coords)
        da = xr.DataArray(
            data=np.random.randn(*shape).astype(np.float32),
            dims=("time", "lead_time", "variable", *self.domain_dims),
            coords=coords,
            attrs=self.attrs,
        )
        return da


class RandomDataFrame:
    """A randomly generated DataFrame source. Primarily useful for testing.

    Generates random observations at random locations for specified times and variables.
    Each observation is a point in space-time with a random value.

    Parameters
    ----------
    n_obs : int, optional
        Number of random observations to generate per time step, by default 10
    tolerance : timedelta | np.timedelta64, optional
        Time tolerance; observations will be randomly sampled within +/- tolerance
        of each requested time, by default np.timedelta64(0)
    schema : pa.Schema | None, optional
        PyArrow schema to use for data generation. If None, uses default SCHEMA.
        Data will be generated dynamically based on schema field types, by default None
    field_generators : dict[str, Callable[[], Any]] | None, optional
        Dictionary mapping field names to generator functions. These will be merged
        with the default generators. Default generators include: time, lat, lon,
        observation, variable. User-provided generators will override defaults,
        by default None
    """

    SOURCE_ID = "earth2studio.data.RandomDataFrame"
    SCHEMA = pa.schema(
        [
            pa.field("time", pa.timestamp("ns")),
            pa.field("lat", pa.float32()),
            pa.field("lon", pa.float32()),
            pa.field("observation", pa.float32()),
            pa.field("variable", pa.string()),
        ]
    )

    def __init__(
        self,
        n_obs: int = 10,
        tolerance: timedelta | np.timedelta64 = np.timedelta64(0),
        schema: pa.Schema | None = None,
        field_generators: dict[str, Callable[[], Any]] | None = None,
    ):
        self.n_obs = n_obs
        # Normalize tolerance to python timedelta
        if isinstance(tolerance, np.timedelta64):
            self.tolerance = pd.to_timedelta(tolerance).to_pytimedelta()
        else:
            self.tolerance = tolerance
        # Use provided schema or default class schema
        self.schema = schema if schema is not None else self.SCHEMA
        # Default field generators
        default_generators: dict[str, Callable[[], Any]] = {
            "time": lambda: None,  # Will be set based on obs_time
            "lat": lambda: np.random.uniform(-90.0, 90.0),
            "lon": lambda: np.random.uniform(0.0, 360.0),
            "elev": lambda: np.random.uniform(0.0, 1000.0),
            "observation": lambda: np.random.randn(),
            "variable": lambda: None,  # Will be set based on variable v
        }
        user_generators = field_generators if field_generators is not None else {}
        self._field_generators = {**default_generators, **user_generators}

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | FieldArray | None = None,
    ) -> pd.DataFrame:
        """Retrieve random observation DataFrame.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Timestamps to return data for.
        variable : str | list[str] | VariableArray
            Strings or list of strings that refer to variables to return.
        fields : str | list[str] | pa.Schema | FieldArray | None, optional
            Fields to include in output, by default None (all fields).

        Returns
        -------
        pd.DataFrame
            Random observation DataFrame
        """
        time, variable = prep_data_inputs(time, variable)
        schema = self._resolve_fields(fields)

        # Generate random observations
        data = []
        for t in time:
            # Convert time to datetime if needed
            t_dt = pd.to_datetime(t)

            for v in variable:
                for _ in range(self.n_obs):
                    # Sample random time within tolerance range
                    if self.tolerance.total_seconds() > 0:
                        # Generate random time within tolerance window
                        random_offset_seconds = np.random.uniform(
                            -self.tolerance.total_seconds(),
                            self.tolerance.total_seconds(),
                        )
                        obs_time = t_dt + timedelta(seconds=random_offset_seconds)
                    else:
                        # No tolerance, use exact time
                        obs_time = t_dt

                    # Generate data dynamically based on schema
                    row = {}
                    for field in schema:
                        field_name = field.name

                        # Check if field has a generator function
                        if field_name in self._field_generators:
                            # Special handling for time and variable
                            if field_name == "time":
                                row[field_name] = obs_time
                            elif field_name == "variable":
                                row[field_name] = v
                            else:
                                generator = self._field_generators[field_name]
                                row[field_name] = generator()
                        else:
                            # Field not in dictionary
                            if field.nullable:
                                row[field_name] = None
                            else:
                                raise KeyError(
                                    f"Field '{field_name}' not found in field generators "
                                    f"and is not nullable. Available generators: "
                                    f"{list(self._field_generators.keys())}"
                                )

                    data.append(row)

        df = pd.DataFrame(data)
        df.attrs["source"] = self.SOURCE_ID

        # Filter to requested fields
        if fields is not None:
            field_names = [name for name in schema.names if name in df.columns]
            df = df[field_names]

        return df

    def _resolve_fields(
        self, fields: str | list[str] | pa.Schema | FieldArray | None
    ) -> pa.Schema:
        """Convert fields parameter into a validated PyArrow schema.

        Parameters
        ----------
        fields : str | list[str] | pa.Schema | FieldArray | None
            Field specification. Can be:
            - None: Returns the full instance schema
            - str: Single field name to select from schema
            - list[str]: List of field names to select from schema
            - pa.Schema: Validated against instance schema for compatibility

        Returns
        -------
        pa.Schema
            A PyArrow schema containing only the requested fields
        """
        if fields is None:
            return self.schema

        if isinstance(fields, str):
            fields = [fields]

        if isinstance(fields, pa.Schema):
            # Validate provided schema against instance schema
            for field in fields:
                if field.name not in self.schema.names:
                    raise KeyError(
                        f"Field '{field.name}' not found in schema. "
                        f"Available fields: {self.schema.names}"
                    )
                expected_type = self.schema.field(field.name).type
                if field.type != expected_type:
                    raise TypeError(
                        f"Field '{field.name}' has type {field.type}, "
                        f"expected {expected_type} from schema"
                    )
            return fields

        # fields is list[str] - select fields from instance schema
        selected_fields = []
        for name in fields:
            if name not in self.schema.names:
                raise KeyError(
                    f"Field '{name}' not found in schema. "
                    f"Available fields: {self.schema.names}"
                )
            selected_fields.append(self.schema.field(name))

        return pa.schema(selected_fields)

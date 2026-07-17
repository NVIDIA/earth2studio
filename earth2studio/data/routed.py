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

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa

from earth2studio.data.base import DataFrameSource
from earth2studio.utils.type import TimeArray, VariableArray


class RoutedObsSource:
    """Composite observation data source that routes variables to backends.

    Presents multiple :class:`~earth2studio.data.base.DataFrameSource` backends
    as a single source: each requested variable is dispatched to the backend
    that owns it, results are fetched per backend and concatenated into one
    DataFrame. Useful when a single logical observation stream is served by
    different archives (e.g. microwave sounders from NNJA and infrared
    sounders from the UFS Replay).

    All backends must share the columns needed by the consumer. The composite
    ``SCHEMA`` is the intersection of the backend schemas (fields matching in
    name and type), and by default only those common columns are returned so
    the concatenated frame is well-formed.

    Parameters
    ----------
    routes : dict[str | Sequence[str], DataFrameSource]
        Mapping from a variable name (or group of variable names) to the
        backend data source that serves it. Each variable may appear in only
        one route.

    Examples
    --------
    >>> sat_source = RoutedObsSource(
    ...     {
    ...         ("atms", "mhs", "amsua", "amsub"): NNJAObsSat(),
    ...         ("iasi", "crisfsr", "airs"): UFSObsSat(),
    ...     }
    ... )
    >>> df = sat_source(time, ["atms", "iasi"])  # doctest: +SKIP
    """

    def __init__(
        self,
        routes: dict[str | Sequence[str], DataFrameSource],
    ) -> None:
        if not routes:
            raise ValueError("At least one route must be provided")

        self._routes: dict[str, DataFrameSource] = {}
        self._sources: list[DataFrameSource] = []
        for key, source in routes.items():
            names = [key] if isinstance(key, str) else list(key)
            if not names:
                raise ValueError("Route variable groups cannot be empty")
            for name in names:
                if name in self._routes:
                    raise ValueError(
                        f"Variable {name!r} is mapped to more than one source"
                    )
                self._routes[name] = source
            if all(source is not existing for existing in self._sources):
                self._sources.append(source)

        self.SCHEMA = self._common_schema([s.SCHEMA for s in self._sources])
        if len(self.SCHEMA) == 0:
            raise ValueError("Backend sources share no common schema fields")

    @property
    def variables(self) -> list[str]:
        """Variables served by this source, in route order."""
        return list(self._routes.keys())

    @staticmethod
    def _common_schema(schemas: list[pa.Schema]) -> pa.Schema:
        """Intersection of schemas: fields matching in name and type,
        ordered as in the first schema."""
        common = [
            field
            for field in schemas[0]
            if all(
                schema.get_field_index(field.name) >= 0
                and schema.field(field.name).type == field.type
                for schema in schemas[1:]
            )
        ]
        return pa.schema(common)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
        fields: str | list[str] | pa.Schema | None = None,
    ) -> pd.DataFrame:
        """Fetch observations, routing each variable to its backend.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Datetime, list of datetimes or array of np.datetime64 to return
            data for.
        variable : str | list[str] | VariableArray
            String, list of strings or array of strings that refer to
            variables to return.
        fields : str | list[str] | pa.Schema | None, optional
            Fields / columns to return; must be available on every backend
            that serves a requested variable. If None, the common schema
            columns are returned, by default None

        Returns
        -------
        pd.DataFrame
            Concatenated observations from all involved backends
        """
        if isinstance(variable, str):
            variables = [variable]
        else:
            variables = [str(v) for v in np.asarray(variable).ravel()]

        unknown = [v for v in variables if v not in self._routes]
        if unknown:
            raise ValueError(
                f"Variable(s) {unknown} have no route; available: " f"{self.variables}"
            )

        if fields is None:
            fields = self.SCHEMA.names

        parts = []
        for source in self._sources:
            source_variables = [v for v in variables if self._routes[v] is source]
            if not source_variables:
                continue
            parts.append(source(time, source_variables, fields=fields))

        df = pd.concat(parts, ignore_index=True)
        df.attrs["source"] = "earth2studio.data.RoutedObsSource"
        return df

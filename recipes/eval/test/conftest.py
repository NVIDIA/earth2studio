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

"""Shared fixtures for the eval recipe tests.

Models and sources follow the Earth2Studio 1.0 DataArray protocols: a
4 x 8 latitude-longitude grid, ``Persistence`` as the prognostic model and
``SeededRandom`` as the data source, so every test runs on CPU without downloads.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict

from earth2studio.data import Random
from earth2studio.grids import LatLonGrid
from earth2studio.models.px import Persistence
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
VARIABLES = ["t2m", "z500"]
RECIPE_DIR = Path(__file__).resolve().parents[1]
CFG_DIR = RECIPE_DIR / "cfg"


class FakeDiagnostic(torch.nn.Module):
    """Diagnostic copying matching input channels into its output variables.

    Output variables absent from the input are zero.  Declares a dynamic
    ``batch`` prefix like the library's diagnostics, so leading ``time`` and
    ``lead_time`` axes pass through.
    """

    def __init__(
        self,
        input_variables: list[str],
        output_variables: list[str],
        grid: LatLonGrid | None = None,
    ) -> None:
        super().__init__()
        self._input_variables = list(input_variables)
        self._output_variables = list(output_variables)
        self._grid = grid or LatLonGrid(SMALL_LAT, SMALL_LON)
        self.calls = 0

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "variable", "lat", "lon"),
            {"variable": np.array(self._input_variables)},
            dynamic=("batch",),
            grid=self._grid,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        return coord_array_like(
            input_coords, {"variable": np.array(self._output_variables)}
        )

    def to(self, device: torch.device) -> FakeDiagnostic:  # type: ignore[override]
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        self.calls += 1
        parts = []
        for name in self._output_variables:
            if name in x["variable"].values:
                parts.append(x.sel(variable=[name]).assign_coords(variable=[name]))
            else:
                zeros = xr.zeros_like(x.isel(variable=[0])).assign_coords(
                    variable=[name]
                )
                parts.append(zeros)
        out = xr.concat(parts, dim="variable").transpose(*x.dims)
        out.attrs = dict(x.attrs)
        return out


@pytest.fixture()
def small_grid() -> LatLonGrid:
    return LatLonGrid(SMALL_LAT, SMALL_LON)


@pytest.fixture()
def small_domain() -> OrderedDict:
    return OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})


@pytest.fixture()
def prognostic(small_grid) -> Persistence:
    return Persistence(variable=VARIABLES, domain_coords=small_grid)


class SeededRandom(Random):
    """``Random`` whose fields depend only on the requested times.

    The library ``Random`` source draws from the global NumPy state, so two
    fetches of the same time differ; the runner parity tests need the same
    initial condition on every fetch.
    """

    def __call__(self, time, variable) -> xr.DataArray:
        da = super().__call__(time, variable)
        seed = int(
            np.asarray(da["time"].values).astype("datetime64[s]").astype(np.int64).sum()
        ) % (2**32)
        da.values[...] = np.random.default_rng(seed).standard_normal(da.shape)
        return da


@pytest.fixture()
def data_source(small_domain) -> Random:
    return SeededRandom(domain_coords=small_domain)


@pytest.fixture()
def fake_diagnostic(small_grid) -> FakeDiagnostic:
    """Diagnostic taking t2m and z500 and producing 'diag_a' (zeros) and 't2m'."""
    return FakeDiagnostic(["t2m", "z500"], ["diag_a", "t2m"], grid=small_grid)


def compose_cfg(tmp_path: Path, overrides: list[str] | None = None) -> DictConfig:
    """Compose the recipe's real default config with test overrides."""
    base = [
        "project=test_eval",
        "run_id=unit",
        f"output.path={tmp_path / 'outputs'}",
        "output.thread_writers=0",
        "output.io_backend=zarr",
        "require_predownload=false",
        "model.architecture=earth2studio.models.px.Persistence",
        "data_source._target_=test.conftest.SeededRandom",
    ]
    with initialize_config_dir(config_dir=str(CFG_DIR), version_base=None):
        cfg = compose(config_name="default", overrides=base + list(overrides or []))
    return cfg


@pytest.fixture()
def base_cfg(tmp_path) -> DictConfig:
    """Config running Persistence on the 4 by 8 grid for two 6-hour steps.

    ``Persistence`` needs its variables and grid at construction, which the
    ``architecture`` form cannot express, so the model node is a ``_target_``
    block.
    """
    cfg = compose_cfg(tmp_path, ["nsteps=2", "ensemble_size=1", "random_seed=42"])
    with open_dict(cfg):
        cfg.model = {
            "_target_": "earth2studio.models.px.Persistence",
            "variable": list(VARIABLES),
            "domain_coords": "latlon-small",
        }
        cfg.data_source = {
            "_target_": "test.conftest.SeededRandom",
            "domain_coords": {"lat": SMALL_LAT.tolist(), "lon": SMALL_LON.tolist()},
        }
        cfg.output.variables = list(VARIABLES)
    return cfg


@pytest.fixture(autouse=True)
def register_small_grid() -> None:
    """Register the test grid under a name Hydra configs can reference."""
    from earth2studio.grids import list_grids, register_grid

    if "latlon-small" not in list_grids():
        register_grid("latlon-small", LatLonGrid(SMALL_LAT, SMALL_LON))

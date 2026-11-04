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

"""Output regridding: the ``output.regrid`` block and its lazy regridder."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf
from src.output import OutputRegridder, build_regridders, regrid_spec

from earth2studio.grids import LatLonGrid, resolve_regridder
from earth2studio.utils.coords import coord_array
from test.conftest import SMALL_LAT, SMALL_LON

CROP = {
    "lat": {"start": 90.0, "stop": -90.0, "num": 2},
    "lon": {"start": 0.0, "stop": 180.0, "num": 2},
}
OFF_GRID = {
    "lat": {"start": 60.0, "stop": -60.0, "num": 2},
    "lon": {"start": 10.0, "stop": 190.0, "num": 2},
}


def _cfg(regrid) -> OmegaConf:
    return OmegaConf.create({"output": {"regrid": regrid}})


def _field() -> xr.DataArray:
    values = np.arange(len(SMALL_LAT) * len(SMALL_LON), dtype="float32")
    return xr.DataArray(
        values.reshape(1, len(SMALL_LAT), len(SMALL_LON)),
        dims=("variable", "lat", "lon"),
        coords={"variable": np.array(["t2m"]), "lat": SMALL_LAT, "lon": SMALL_LON},
    )


def test_regrid_spec_is_off_by_default() -> None:
    assert regrid_spec(_cfg(None)) is None


def test_regrid_spec_reads_the_options_block_and_flat_keys() -> None:
    spec = regrid_spec(
        _cfg(
            {
                "grid": CROP,
                "method": "nearest",
                "options": {"max_dist_km": "auto"},
                "fill_value": 0.0,
            }
        )
    )
    assert spec is not None
    assert spec.method == "nearest"
    assert spec.options == {"max_dist_km": "auto", "fill_value": 0.0}
    assert isinstance(spec.target, LatLonGrid)
    np.testing.assert_array_equal(spec.target_grid.coords()["lat"], [90.0, -90.0])


def test_regrid_spec_defaults_to_nearest_and_needs_a_grid() -> None:
    spec = regrid_spec(_cfg({"grid": "latlon-small"}))
    assert spec is not None and spec.method == "nearest" and spec.options == {}
    with pytest.raises(ValueError, match="grid"):
        regrid_spec(_cfg({"method": "nearest"}))


def test_output_regridder_crops_exact_subsets() -> None:
    regrid = OutputRegridder(regrid_spec(_cfg({"grid": CROP, "method": "bilinear"})))
    out = regrid(_field())
    assert out.shape == (1, 2, 2)
    np.testing.assert_array_equal(out["lat"].values, [90.0, -90.0])
    np.testing.assert_array_equal(
        out.values, _field().sel(lat=[90.0, -90.0], lon=[0.0, 180.0]).values
    )
    # Already on the target: untouched.
    assert regrid(out) is out


def test_output_regridder_uses_the_configured_method(small_grid) -> None:
    spec = regrid_spec(_cfg({"grid": OFF_GRID, "method": "bilinear"}))
    regrid = OutputRegridder(spec)
    field = _field()
    out = regrid(field)
    bilinear = resolve_regridder(small_grid, spec.target, "bilinear")(field)
    nearest = resolve_regridder(small_grid, spec.target, "nearest")(field)
    np.testing.assert_allclose(out.values, bilinear.values, rtol=1e-6)
    assert not np.allclose(out.values, nearest.values)
    # One regridder serves every field on the same source grid.
    assert regrid(field) is not out
    assert regrid._regridder is not None


def test_build_regridders_accepts_the_options_block(small_grid) -> None:
    cfg = _cfg({"grid": CROP, "method": "nearest", "options": {"fill_value": 0.0}})
    schema = coord_array(
        ("lead_time", "variable", "lat", "lon"),
        {
            "lead_time": np.array([np.timedelta64(0, "h")]),
            "variable": np.array(["t2m"]),
        },
        grid=small_grid,
    )
    regridders = build_regridders(cfg, {"forecast": schema})
    assert list(regridders) == ["forecast"]
    assert build_regridders(_cfg(None), {"forecast": schema}) == {}

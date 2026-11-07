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

import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.run as run
from earth2studio.data import Constant, Random, Random_FX, fetch_data
from earth2studio.grids import ProjectedGrid
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Persistence
from earth2studio.utils.checkpoint import Checkpoint
from earth2studio.utils.coords import coord_array


# This class is used to verify the workflow moved the model onto the right device
class TestPersistence(Persistence):
    def __init__(self, *args, target_device="cpu"):
        super().__init__(*args)
        self.target_device = torch.device(target_device)

    def _forward(
        self,
        x: xr.DataArray,
    ) -> xr.DataArray:
        assert x.e2s.to_torch()[0].device == self.target_device
        return super()._forward(x)


@pytest.mark.parametrize(
    "coords",
    [
        OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))]),
        OrderedDict([("c1", np.arange(10))]),
        OrderedDict([("c1", np.arange(5)), ("c2", np.arange(5)), ("c3", np.arange(5))]),
    ],
)
@pytest.mark.parametrize(
    "variable", [["t2m"], ["u10m", "v10m"], ["u10m", "u100", "nvidia"]]
)
@pytest.mark.parametrize("nsteps", [5, 10])
@pytest.mark.parametrize("time", [["2024-01-01"]])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_run_deterministic(coords, variable, nsteps, time, device):

    data = Random(domain_coords=coords)
    model = TestPersistence(variable, coords, target_device=device)

    io = ZarrBackend()

    io = run.deterministic(time, nsteps, model, data, io, device=device)

    for var in variable:
        assert io[var].shape[0] == len(time)
        assert io[var].shape[1] == nsteps + 1
        for i, (key, value) in enumerate(coords.items()):
            assert io[var].shape[i + 2] == value.shape[0]


@pytest.mark.parametrize(
    "output_coords",
    [
        OrderedDict({"variable": np.array(["u10m"])}),
        OrderedDict(
            {"variable": np.array(["v10m", "t2m"]), "lat": np.array([0, 1, 2, 3])}
        ),
        OrderedDict(
            {
                "variable": np.array(["nvidia"]),
                "lon": np.array([0, 1, 2, 3]),
                "lat": np.array([4, 5, 6]),
            }
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_deterministic_output_coords(output_coords, device):

    output_coords = output_coords.copy()
    coords = OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))])
    variable = ["u10m", "v10m", "u100", "t2m", "nvidia"]
    nsteps = 2
    time = ["1993-04-05T12:00:00"]

    data = Random(domain_coords=coords)
    model = TestPersistence(variable, coords, target_device=device)
    io = ZarrBackend()

    io = run.deterministic(time, nsteps, model, data, io, output_coords, device=device)

    for name in variable:
        if name not in output_coords["variable"]:
            assert name not in list(io.root.array_keys())
        else:
            assert name in list(io.root.array_keys())

    del output_coords["variable"]
    for key, value in output_coords.items():
        assert np.array_equal(io[key], value)


@pytest.mark.parametrize("source_type", [Random, Random_FX])
def test_native_projected_grid_and_forecast_source(source_type):
    grid = ProjectedGrid(np.arange(3) * 3000.0, np.arange(4) * 3000.0, "EPSG:3857")
    model = Persistence(["t2m"], grid, history=2)
    source = source_type(OrderedDict(grid.coords(only_index=True).items()))

    class NativeSource:
        def __call__(self, time, variable):
            return source(time, variable).assign_coords(grid.coords())

    class NativeForecastSource:
        def __call__(self, time, lead_time, variable):
            return source(time, lead_time, variable).assign_coords(grid.coords())

    seen = []

    def check_field(field):
        assert field.dims == ("time", "lead_time", "variable", "y", "x")
        xr.testing.assert_equal(field.coords["lat"], grid.coords()["lat"])
        assert (
            field.attrs["earth2studio_crs"]
            == model.input_coords().attrs["earth2studio_crs"]
        )
        seen.append(field)
        return field

    model.front_hook = check_field
    io = run.deterministic(
        ["2024-01-01"],
        2,
        model,
        NativeSource() if source_type is Random else NativeForecastSource(),
        ZarrBackend(),
        device="cpu",
        verbose=False,
    )
    assert len(seen) == 1
    assert io["t2m"].shape == (1, 3, 3, 4)
    assert not np.isnan(io["t2m"][:]).any()
    np.testing.assert_array_equal(io["t2m"][0, 0], io["t2m"][0, 2])


def test_native_mapping_preserves_storage_and_rejects_wrong_geometry():
    grid = ProjectedGrid(np.arange(3), np.arange(4), "EPSG:3857")
    signature = coord_array(("variable", "y", "x"), {"variable": ["t2m"]}, grid=grid)
    field = xr.DataArray(
        np.ones(signature.shape), dims=signature.dims, coords=signature.coords
    )
    mapped = run._map_field(field, signature)
    assert np.shares_memory(mapped.data, field.data)
    assert "earth2studio_crs" not in field.attrs
    with pytest.raises(ValueError, match="geometry"):
        run._map_field(field.assign_coords(lat=field.lat + 1), signature)

    # Execute the actual example's fetch/crop statements without loading weights.
    root = Path(__file__).resolve().parents[2]
    parent = ProjectedGrid(np.arange(6) * 3000.0, np.arange(7) * 3000.0, "EPSG:3857")
    crop = ProjectedGrid(parent.y[2:4], parent.x[1:4], parent.crs)
    model = Persistence(["t2m"], crop)

    def source(time, variable):
        return xr.DataArray(
            np.arange(len(time) * len(variable) * 42).reshape(
                len(time), len(variable), 6, 7
            ),
            dims=("time", "variable", "y", "x"),
            coords={"time": time, "variable": variable, **parent.coords()},
            attrs={**parent.attrs, "earth2studio_crs": parent.crs.to_string()},
        )

    tree = ast.parse(
        (root / "examples/05_data_assimilation/01_stormcast_sda.py").read_text()
    )
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"x", "input_signature", "truth_x", "truth_signature"}
    ]
    namespace = dict(
        np=np,
        fetch_data=fetch_data,
        _map_field=run._map_field,
        model=SimpleNamespace(
            grid=crop, variables=np.array(["t2m"]), input_coords=model.input_coords
        ),
        hrrr=source,
        init_time=np.array(["2024-01-01"], dtype="datetime64[h]"),
        plot_var="t2m",
        nsteps=2,
    )
    exec(  # noqa: S102 - execute selected local example statements without weight loading
        compile(ast.Module(body=assignments, type_ignores=[]), "crop-example", "exec"),
        namespace,
    )
    for name in ("x", "truth_x"):
        actual = namespace[name].e2s.as_numpy()
        assert actual.shape[-2:] == (2, 3)
        assert actual.attrs["shape"] == [2, 3]
        assert actual.attrs["earth2studio_crs"] == parent.crs.to_string()
        assert "earth2studio_grid_id" not in actual.attrs
        np.testing.assert_array_equal(actual.lat, crop.coords()["lat"])
        np.testing.assert_array_equal(actual.lon, crop.coords()["lon"])
        np.testing.assert_array_equal(
            actual.values[0, 0, 0], np.arange(42).reshape(6, 7)[2:4, 1:4]
        )
    invalid = namespace["x"].copy(deep=True)
    invalid.attrs["earth2studio_crs"] = "EPSG:4326"
    with pytest.raises(ValueError, match="CRS"):
        run._map_field(invalid, model.input_coords())


def test_native_checkpoint_resume(tmp_path):
    domain = OrderedDict(lat=np.arange(3), lon=np.arange(4))
    io = ZarrBackend()
    times = np.array([np.datetime64("2024-01-01")])
    model = Persistence(["t2m"], domain)
    # A store's extent is fixed at creation, so the shorter first run declares the
    # resumed run's full lead-time extent and writes a subset of its labels.
    output_coords = OrderedDict(
        lead_time=run._lead_times(model.output_coords(model.input_coords()), 3)
    )
    checkpoint = Checkpoint("native-run", path=tmp_path, level=2, flush_interval=1)
    with checkpoint as session:
        run.deterministic(
            list(times),
            1,
            Persistence(["t2m"], domain),
            Constant(domain, 7),
            io,
            output_coords,
            device="cpu",
            verbose=False,
            checkpoint=session,
        )
    with checkpoint as session:
        run.deterministic(
            list(times),
            3,
            Persistence(["t2m"], domain),
            Constant(domain, 99),
            io,
            output_coords,
            device="cpu",
            verbose=False,
            checkpoint=session,
        )
    np.testing.assert_array_equal(io["t2m"][:], 7)

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

"""Model shapes the built-in prognostic runner must carry unchanged.

Integration check 4 of ``dev/spec/EXECUTION_CONTRACT_SPEC.md``: a forced model
(StormCast-shaped) and fused DLESyM run through :class:`PrognosticRunner`
unchanged. The models here are shaped like those wrappers -- the same dynamic
prefixes, grids, forcing slots, lead-time cadences and seeding -- with trivial
arithmetic in place of networks, so the runner's stream schemas, fetches and
seeding are checked without downloading checkpoints.
"""

from collections.abc import Iterator, Sequence

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from earth2studio.grids import HEALPixGrid, ProjectedGrid
from earth2studio.io import KVBackend
from earth2studio.io.utils import output_template
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.run import PrognosticRunner, WorkItem
from earth2studio.run.work import member_seed
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

T0 = np.datetime64("2026-01-01T00")


class _GridSource:
    """Source of constant fields already on a model's own grid.

    Built from the model signature so auxiliary coordinates and grid metadata
    match exactly, as a real source does after regridding.
    """

    def __init__(
        self,
        signature: CoordinateSystem,
        variables: Sequence[str],
        fill: float = 1.0,
    ) -> None:
        self.signature = coord_array_like(
            signature, {"variable": np.array(list(variables))}
        ).isel(lead_time=0, drop=True)
        self.fill = fill
        self.times: list[np.datetime64] = []
        self.variables: list[str] = []

    def __call__(self, time: np.ndarray, variable: np.ndarray) -> xr.DataArray:
        time, variable = np.atleast_1d(time), np.atleast_1d(variable)
        self.times.extend(time)
        self.variables.extend(str(v) for v in variable)
        template = output_template(self.signature, leading={"time": time})
        values = np.full(template.shape, self.fill, dtype=np.float32)
        return template.copy(data=values).sel(variable=variable)


class _StormCastShaped(PrognosticMixin):
    """StormCast-shaped forced model: hourly steps on a cropped projected grid.

    Like :class:`~earth2studio.models.px.StormCast`: a ``(batch, time)`` dynamic
    prefix, one conditioning slot on the model grid at the input valid time,
    one-hour steps, and a sampling stream seeded through ``set_rng``.
    """

    stochastic = True
    variables = ["u10m", "t2m"]
    conditioning_variables = ["z500"]

    def __init__(self) -> None:
        self.grid = ProjectedGrid(
            y=np.arange(3.0) * 3000.0,
            x=np.arange(4.0) * 3000.0,
            coordinate_reference_system=CRS.from_proj4(
                "+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 "
                "+R=6371229 +units=m"
            ),
        )
        self.seeds: list[tuple[int, bool]] = []
        self.conditioning_variables_seen: list[list[str]] = []

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "time", "lead_time", "variable", "y", "x"),
            {
                "lead_time": np.array([0], dtype="timedelta64[h]"),
                "variable": np.array(self.variables),
            },
            dynamic=("batch", "time"),
            grid=self.grid,
        )

    def forcing_coords(self) -> CoordinateSystem:
        return coord_array_like(
            self.input_coords(), {"variable": np.array(self.conditioning_variables)}
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        lead = np.asarray(input_coords["lead_time"].values)
        return coord_array_like(
            input_coords, {"lead_time": lead + np.timedelta64(1, "h")}
        )

    def set_rng(self, seed: int, reset: bool = True) -> None:
        self.seeds.append((seed, reset))

    def _advance(self, x: xr.DataArray, conditioning: xr.DataArray) -> xr.DataArray:
        self.conditioning_variables_seen.append(
            [str(v) for v in conditioning["variable"].values]
        )
        y = x + 1.0 + float(conditioning.values.mean())
        y = y.assign_coords(lead_time=x["lead_time"] + np.timedelta64(1, "h"))
        return y.assign_attrs(x.attrs)

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> tuple[xr.DataArray, int]:
        return self._advance(x, conditioning), 0

    def step(  # type: ignore[override]
        self, y: xr.DataArray, conditioning: xr.DataArray, state: int = 0
    ) -> tuple[xr.DataArray, int]:
        return self._advance(y, conditioning), state + 1

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x, conditioning)


class _FusedDLESyMShaped(PrognosticMixin):
    """Fused DLESyM-shaped model: HEALPix faces and several leads per step.

    Like :class:`~earth2studio.models.px.DLESyM`: a single input slot carrying a
    multi-frame window, a face-oriented HEALPix grid, diagnostic output
    variables that are not fed back, and one iterator step publishing every
    output lead time at once.
    """

    stochastic = True  # DLESyM with conditional layer norm.
    atmos_variables = ["z500"]
    atmos_diagnostic_variables = ["tp06"]
    ocean_variables = ["sst"]
    input_times = np.array([-12, 0], dtype="timedelta64[h]")
    output_times = np.array([6, 12], dtype="timedelta64[h]")

    def __init__(self) -> None:
        self.grid = HEALPixGrid(
            1, ordering="xy", layout="face", xy_origin="north", xy_clockwise=True
        )
        self.seeds: list[tuple[int, bool]] = []

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "time", "lead_time", "variable", "face", "height", "width"),
            {
                "lead_time": self.input_times,
                "variable": np.array(self.atmos_variables + self.ocean_variables),
            },
            dynamic=("batch", "time"),
            grid=self.grid,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        lead = np.asarray(input_coords["lead_time"].values)
        return coord_array_like(
            input_coords,
            {
                "lead_time": lead[-1] + self.output_times,
                "variable": np.array(
                    self.atmos_variables
                    + self.atmos_diagnostic_variables
                    + self.ocean_variables
                ),
            },
        )

    def set_rng(self, seed: int, reset: bool = True) -> None:
        self.seeds.append((seed, reset))

    def _advance(self, x: xr.DataArray) -> xr.DataArray:
        """Publish every output lead time of one model step at once."""
        cell = x.isel(lead_time=-1, drop=True).isel(variable=0, drop=True)
        y = xr.full_like(cell, float(x.isel(lead_time=-1).values.mean()) + 1.0)
        variables = np.array(
            self.atmos_variables
            + self.atmos_diagnostic_variables
            + self.ocean_variables
        )
        y = y.expand_dims(variable=variables).expand_dims(
            lead_time=x["lead_time"].values[-1] + self.output_times
        )
        return y.transpose("time", "lead_time", "variable", *self.grid.dims)

    def _window(self, y: xr.DataArray) -> xr.DataArray:
        """Rebuild the next step's input window from the published outputs."""
        return y.isel(lead_time=[-2, -1]).sel(
            variable=self.atmos_variables + self.ocean_variables
        )

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray
    ) -> tuple[xr.DataArray, None]:
        return self._advance(x), None

    def step(  # type: ignore[override]
        self, y: xr.DataArray, state: None = None
    ) -> tuple[xr.DataArray, None]:
        return self._advance(self._window(y)), None

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        return self._default_create_iterator(x)


def _write_stream(
    runner: PrognosticRunner, item: WorkItem, stream: str = "forecast"
) -> KVBackend:
    """Plan a store from the runner's schema and write every step into it."""
    signature = runner.output_coords(item.horizon)[stream]
    io = KVBackend()
    io.add_array(
        output_template(
            signature,
            leading={"ensemble": np.array([0]), "time": np.array([item.time])},
        )
    )
    for step in runner.run_item(item):
        io.write(step[stream].expand_dims(ensemble=[0]))
    return io


def test_forced_model_rolls_out_with_its_conditioning() -> None:
    model = _StormCastShaped()
    runner = PrognosticRunner(
        model,
        _GridSource(model.input_coords(), model.variables, fill=0.0),
        forcing=(_GridSource(model.forcing_coords(), model.conditioning_variables),),
        seed=3,
    )
    item = WorkItem(T0, np.timedelta64(3, "h"), member_ids=(2,))

    signature = runner.output_coords(item.horizon)["forecast"]
    assert signature.dims == ("batch", "time", "lead_time", "variable", "y", "x")
    assert signature.data.nbytes == 0
    np.testing.assert_array_equal(
        signature["lead_time"], np.arange(4, dtype="timedelta64[h]")
    )
    # Auxiliary coordinates and CRS travel with the schema (X2).
    assert signature["lat"].dims == ("y", "x")
    assert signature.attrs["earth2studio_crs"] == model.grid.crs.to_string()

    io = _write_stream(runner, item)
    assert io["u10m"].shape == (1, 1, 4, 3, 4)
    # Steps advance by one each hour, forced by conditioning of ones.
    np.testing.assert_allclose(io["u10m"][0, 0, :, 0, 0], [0, 2, 4, 6])
    assert model.conditioning_variables_seen == [["z500"]] * 3
    assert model.seeds == [(member_seed(3, T0, 2), True)]


def test_forced_model_requests_conditioning_for_every_step() -> None:
    model = _StormCastShaped()
    runner = PrognosticRunner(
        model,
        _GridSource(model.input_coords(), model.variables, fill=0.0),
        forcing=(_GridSource(model.forcing_coords(), model.conditioning_variables),),
    )
    item = WorkItem(T0, np.timedelta64(3, "h"))
    requests = runner.data_requests(item)
    assert [sorted(set(map(str, r.variable))) for r in requests] == [
        ["t2m", "u10m"],
        ["z500"],
        ["z500"],
        ["z500"],
    ]
    assert [r.lead_time.tolist() for r in requests[1:]] == [
        [np.timedelta64(hours, "h")] for hours in (0, 1, 2)
    ]
    conditioning = runner.forcing[0]
    list(runner.run_item(item))
    # One conditioning fetch per step, at the valid time of the step's input.
    assert isinstance(conditioning, _GridSource)
    offsets = (np.array(conditioning.times) - T0).astype("timedelta64[h]").astype(int)
    assert offsets.tolist() == [0, 1, 2]


def test_fused_dlesym_publishes_every_lead_time_of_each_step() -> None:
    model = _FusedDLESyMShaped()
    source = _GridSource(
        model.input_coords(), model.atmos_variables + model.ocean_variables, fill=0.0
    )
    runner = PrognosticRunner(model, source, seed=1)
    item = WorkItem(T0, np.timedelta64(24, "h"), member_ids=(4,))

    # One model step covers two lead times, so the schema spans nsteps * 2 + 1.
    assert runner.nsteps(item.horizon) == 2
    signature = runner.output_coords(item.horizon)["forecast"]
    assert signature.dims == (
        "batch",
        "time",
        "lead_time",
        "variable",
        "face",
        "height",
        "width",
    )
    np.testing.assert_array_equal(
        signature["lead_time"], np.array([0, 6, 12, 18, 24], dtype="timedelta64[h]")
    )
    np.testing.assert_array_equal(signature["variable"], ["z500", "tp06", "sst"])
    # HEALPix grids declare index coordinates and grid metadata, no lat/lon (X2).
    np.testing.assert_array_equal(signature["face"], np.arange(12))
    assert "lat" not in signature.coords
    assert signature.attrs["nside"] == 2
    assert signature.attrs["dims"] == ["face", "height", "width"]

    steps = list(runner.run_item(item))
    assert [int(s["forecast"].sizes["lead_time"]) for s in steps] == [1, 2, 2]
    leads = np.concatenate([s["forecast"]["lead_time"].values for s in steps])
    np.testing.assert_array_equal(leads, signature["lead_time"].values)
    assert model.seeds == [(member_seed(1, T0, 4), True)]

    io = _write_stream(runner, item)
    assert io["z500"].shape == (1, 1, 5, 12, 2, 2)
    np.testing.assert_allclose(io["z500"][0, 0, :, 0, 0, 0], [0, 1, 1, 2, 2])


def test_fused_dlesym_initial_condition_omits_diagnostic_variables() -> None:
    """The published initial condition carries the model's *input* variables.

    DLESyM's output stream adds diagnostic variables that are not inputs, so the
    runner's first step is a label subset of the stream -- which IO accepts.
    """
    model = _FusedDLESyMShaped()
    source = _GridSource(
        model.input_coords(), model.atmos_variables + model.ocean_variables, fill=0.0
    )
    runner = PrognosticRunner(model, source)
    item = WorkItem(T0, np.timedelta64(12, "h"))
    first = next(iter(runner.run_item(item)))["forecast"]
    assert [str(v) for v in first["variable"].values] == ["z500", "sst"]
    assert "tp06" not in first["variable"].values
    io = _write_stream(runner, item)
    assert np.isnan(float(io["tp06"][0, 0, 0, 0, 0, 0]))


def test_member_batching_is_rejected_for_both_shapes() -> None:
    model = _StormCastShaped()
    runner = PrognosticRunner(
        model,
        _GridSource(model.input_coords(), model.variables),
        forcing=(_GridSource(model.forcing_coords(), model.conditioning_variables),),
    )
    assert runner.supports_member_batching is False
    with pytest.raises(ValueError):
        list(runner.run_item(WorkItem(T0, np.timedelta64(1, "h"), member_ids=(0, 1))))

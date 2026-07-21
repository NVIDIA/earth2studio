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

import itertools
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.px.datareplay import DataReplay

H, W = 8, 16
LAT = np.linspace(90, -90, H, endpoint=False)
LON = np.linspace(0, 360, W, endpoint=False)
VARS = ["t2m", "u10m", "z500"]
Y0 = np.datetime64("2020-01-01T00:00:00")


class HourSource:
    """Synthetic in-memory DataSource: each frame is a constant field equal to hours-since-Y0 (nlat rows)."""

    def __init__(self, nlat=H, fill=None):
        self.nlat = nlat
        self.fill = (
            fill  # if set, return this value (e.g. np.nan) to test the finite guard
        )

    def __call__(self, time, variable):
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
        variable = [variable] if isinstance(variable, str) else list(variable)
        arr = np.zeros((len(times), len(variable), self.nlat, W), dtype="float32")
        for i, t in enumerate(times):
            arr[i] = (
                self.fill
                if self.fill is not None
                else float((t - Y0) / np.timedelta64(1, "h"))
            )
        return xr.DataArray(
            arr,
            dims=["time", "variable", "lat", "lon"],
            coords=dict(
                time=times,
                variable=np.array(variable),
                lat=np.linspace(90, -90, self.nlat, endpoint=False),
                lon=LON,
            ),
        )


def _ic(src, t0):
    x = torch.from_numpy(src(np.array([t0]), VARS).values.astype("float32"))[
        None, :, None
    ]  # (1,1,1,V,H,W)
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARS),
        lat=LAT,
        lon=LON,
    )
    return x, coords


def test_datareplay_call():
    # __call__ advances one step: fetch the source at time + lead_time + step.
    src = HourSource()
    replay = DataReplay(src, VARS, OrderedDict(lat=LAT, lon=LON), step=6)
    t0 = Y0 + np.timedelta64(48, "h")
    x, coords = _ic(src, t0)
    xf, cf = replay(x, coords)
    assert xf.shape == (1, 1, 1, len(VARS), H, W)
    assert int(cf["lead_time"][-1] / np.timedelta64(1, "h")) == 6  # advanced one step
    # source at t0 + 6h == valid hour 54 (whole constant field)
    assert torch.allclose(xf, torch.full_like(xf, 54.0), atol=1e-4)


@pytest.mark.parametrize("step_h", [3, 6])
def test_datareplay_iter(step_h):
    src = HourSource()
    replay = DataReplay(src, VARS, OrderedDict(lat=LAT, lon=LON), step=step_h)
    t0 = Y0 + np.timedelta64(48, "h")
    x, coords = _ic(src, t0)
    leads = []
    # islice to exactly 3 frames (IC + 2 steps) so no extra fourth frame is fetched.
    for k, (xf, cf) in enumerate(
        itertools.islice(replay.create_iterator(x, coords), 3)
    ):
        leads.append(int(cf["lead_time"][-1] / np.timedelta64(1, "m")))
        assert xf.shape == (1, 1, 1, len(VARS), H, W)
        assert torch.isfinite(xf).all()
        # whole field is constant == valid hour (checks every cell, not just the mean)
        expected = 48.0 + k * step_h
        assert torch.allclose(xf, torch.full_like(xf, expected), atol=1e-4)
    assert leads == [0, step_h * 60, step_h * 120]


def test_datareplay_grid_mismatch_raises():
    replay = DataReplay(
        HourSource(nlat=H + 1), VARS, OrderedDict(lat=LAT, lon=LON), step=6
    )
    x, coords = _ic(HourSource(), Y0 + np.timedelta64(48, "h"))
    iterator = replay.create_iterator(x, coords)
    next(iterator)  # initial condition (from a valid source) must not raise
    with pytest.raises(ValueError, match="does not match"):
        next(iterator)  # first fetch hits the mismatched source grid


def test_datareplay_nonfinite_raises():
    replay = DataReplay(
        HourSource(fill=np.nan), VARS, OrderedDict(lat=LAT, lon=LON), step=6
    )
    x, coords = _ic(HourSource(), Y0 + np.timedelta64(48, "h"))
    iterator = replay.create_iterator(x, coords)
    next(iterator)  # initial condition (finite) must not raise
    with pytest.raises(ValueError, match="non-finite"):
        next(iterator)  # first fetch returns the NaN-filled field


@pytest.mark.parametrize(
    "step, expected",
    [
        (6, np.timedelta64(6, "h")),  # int hours
        (6.0, np.timedelta64(6, "h")),  # float hours
        (np.float32(6.0), np.timedelta64(6, "h")),  # numpy float hours
        (np.timedelta64(6, "h"), np.timedelta64(6, "h")),  # timedelta64 verbatim
        (
            np.timedelta64(90, "m"),
            np.timedelta64(90, "m"),
        ),  # sub-hour td64 kept (numeric-only check)
    ],
)
def test_datareplay_valid_step(step, expected):
    dom = OrderedDict(lat=LAT, lon=LON)
    assert DataReplay(HourSource(), VARS, dom, step=step).step == expected


@pytest.mark.parametrize(
    "step, exc, match",
    [
        (6.5, ValueError, "whole hours"),  # fractional numeric hours
        (0, ValueError, "positive"),  # positivity boundary
        (-1, ValueError, "positive"),  # negative
        ("6h", TypeError, "step must be"),  # wrong type
        # NaT is the timedelta NaN: it passes the timedelta64 type check, and NaT <= 0h is False (all
        # NaT comparisons are False), so it needs an explicit isnat() guard, not the positivity check.
        (np.timedelta64("NaT"), ValueError, "NaT"),
        (True, TypeError, "step must be"),  # bool subclasses int
        (float("nan"), ValueError, "finite"),  # NaN
        (float("inf"), ValueError, "finite"),  # inf
    ],
)
def test_datareplay_exceptions(step, exc, match):
    # invalid step values / types raise
    with pytest.raises(exc, match=match):
        DataReplay(HourSource(), VARS, OrderedDict(lat=LAT, lon=LON), step=step)


def test_datareplay_front_hook_applied():
    # front_hook transforms the provided input; the step-0 IC (derived from it) reflects the change,
    # while later frames are fetched fresh from the source and are not front-hooked.
    replay = DataReplay(HourSource(), VARS, OrderedDict(lat=LAT, lon=LON), step=6)
    x, coords = _ic(HourSource(), Y0 + np.timedelta64(48, "h"))
    replay.front_hook = lambda a, c: (a + 50.0, c)
    it = replay.create_iterator(x, coords)
    ic, _ = next(it)  # IC = input (hour 48) + 50 from front_hook
    assert torch.allclose(ic, torch.full_like(ic, 48.0 + 50.0), atol=1e-4)
    nxt, _ = next(it)  # fetched fresh from the source (hour 54), not front-hooked
    assert torch.allclose(nxt, torch.full_like(nxt, 54.0), atol=1e-4)


def test_datareplay_rear_hook_applied():
    # rear_hook transforms every emitted frame (the IC and each fetched frame); exact values (IC at
    # hour 48, fetched at 54/60) confirm the offset lands on every frame, not just some.
    replay = DataReplay(HourSource(), VARS, OrderedDict(lat=LAT, lon=LON), step=6)
    x, coords = _ic(HourSource(), Y0 + np.timedelta64(48, "h"))
    replay.rear_hook = lambda a, c: (a + 100.0, c)
    frames = [xf for xf, _ in itertools.islice(replay.create_iterator(x, coords), 3)]
    for xf, hour in zip(frames, [48.0, 54.0, 60.0]):
        assert torch.allclose(xf, torch.full_like(xf, hour + 100.0), atol=1e-4)


def test_datareplay_iter_coord_mismatch_raises():
    # The iterator path validates input coords (variable order) up front, like __call__ and
    # Persistence -- not just at the runtime _fetch grid check.
    replay = DataReplay(HourSource(), VARS, OrderedDict(lat=LAT, lon=LON), step=6)
    x, coords = _ic(HourSource(), Y0 + np.timedelta64(48, "h"))
    coords["variable"] = np.array(VARS[::-1])  # reversed order -> handshake mismatch
    with pytest.raises(ValueError, match="not the same"):
        next(replay.create_iterator(x, coords))

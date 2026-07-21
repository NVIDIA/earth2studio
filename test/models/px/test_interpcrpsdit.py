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
import types
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.px.interpcrpsdit as interp_mod
from earth2studio.models.auto import Package
from earth2studio.models.px.datareplay import DataReplay
from earth2studio.models.px.interpcrpsdit import (
    OPTIONAL_VARIABLES,
    PHYS_EXTRA,
    VARIABLES,
    InterpCRPSDiT,
)

# Small grid to keep the suite fast on CPU without weights or network access.
H, W = 16, 32
LAT = np.linspace(90, -90, H, endpoint=False)
LON = np.linspace(0, 360, W, endpoint=False)
Y0 = np.datetime64("2020-01-01T00:00:00")


class PhooDiT(torch.nn.Module):
    """Dummy DiT matching the physicsnemo-DiT interface InterpCRPSDiT calls. Returns a zero residual,
    so with center=0/scale=1 the interpolated field is exactly the linear base.
    """

    patch_size = (2, 2)

    def __init__(self) -> None:
        super().__init__()
        self.detokenizer = (
            types.SimpleNamespace()
        )  # _dit_forward sets .h_patches/.w_patches on it

    def forward(self, x, t, condition=None, attn_kwargs=None):
        return x.new_zeros(x.shape[0], len(VARIABLES), x.shape[-2], x.shape[-1])


class PhooSource:
    """Synthetic in-memory DataSource: each frame is a constant field equal to hours-since-Y0, so linear
    interpolation of two frames has a known closed form."""

    def __call__(self, time, variable):
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
        variable = [variable] if isinstance(variable, str) else list(variable)
        arr = np.zeros((len(times), len(variable), H, W), dtype="float32")
        for i, t in enumerate(times):
            arr[i] = float((t - Y0) / np.timedelta64(1, "h"))
        return xr.DataArray(
            arr,
            dims=["time", "variable", "lat", "lon"],
            coords=dict(time=times, variable=np.array(variable), lat=LAT, lon=LON),
        )


class GridSource:
    """Source on an exact subset of the base lat/lon grid (for sub-domain px_models); ``lat``/``lon``
    must be members of ``all_lat``/``all_lon`` (np.where needs exact matches). spatially varying: each
    cell = hours-since-Y0 + row-index + 0.01*col-index, keyed to absolute lat/lon, so a mis-offset
    trim of the correct size yields different values than the exact bounding-box sub-block.
    """

    def __init__(self, lat, lon, all_lat, all_lon):
        self.lat, self.lon = np.asarray(lat), np.asarray(lon)
        self.all_lat, self.all_lon = np.asarray(all_lat), np.asarray(all_lon)

    def __call__(self, time, variable):
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
        variable = [variable] if isinstance(variable, str) else list(variable)
        # global row/col index of each of this grid's cells (absolute coordinate -> unique value)
        ri = np.array(
            [int(np.where(self.all_lat == v)[0][0]) for v in self.lat], dtype="float32"
        )
        ci = np.array(
            [int(np.where(self.all_lon == v)[0][0]) for v in self.lon], dtype="float32"
        )
        field = (
            ri[:, None] + 0.01 * ci[None, :]
        )  # (nlat, nlon), distinct per absolute cell
        arr = np.zeros(
            (len(times), len(variable), len(self.lat), len(self.lon)), dtype="float32"
        )
        for i, t in enumerate(times):
            arr[i] = float((t - Y0) / np.timedelta64(1, "h")) + field[None]
        return xr.DataArray(
            arr,
            dims=["time", "variable", "lat", "lon"],
            coords=dict(
                time=times, variable=np.array(variable), lat=self.lat, lon=self.lon
            ),
        )


class NoiseDiT(PhooDiT):
    """Residual = mean of the 6 trailing noise channels (so the drawn latent z reaches the
    output). Used by the ensemble/seed test to observe member spread + seed reproducibility.
    """

    def forward(self, x, t, condition=None, attn_kwargs=None):
        z = x[:, -6:].mean(
            1, keepdim=True
        )  # DiT input = [x0(73), xT(73), cond(87), z(6)]
        return z.expand(x.shape[0], len(VARIABLES), x.shape[-2], x.shape[-1])


class AutocastProbeDiT(PhooDiT):
    """Records whether autocast is active during the forward, so the amp test can assert amp_dtype
    toggles autocast on/off."""

    def __init__(self) -> None:
        super().__init__()
        self.saw_autocast: bool | None = None

    def forward(self, x, t, condition=None, attn_kwargs=None):
        self.saw_autocast = torch.is_autocast_enabled(x.device.type)
        return x.new_zeros(x.shape[0], len(VARIABLES), x.shape[-2], x.shape[-1])


class GapCaptureDiT(PhooDiT):
    """Captures the gap-conditioning channel (index 13 of the 87-ch cond block) as built during
    create_iterator, so the gap-value test exercises the gap->gap_val derivation.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gap_seen: float | None = None

    def forward(self, x, t, condition=None, attn_kwargs=None):
        self.gap_seen = float(
            x[:, 73 + 73 + 13].mean()
        )  # cond starts at 146; gap is cond[13]
        return x.new_zeros(x.shape[0], len(VARIABLES), x.shape[-2], x.shape[-1])


def _build(
    gap_h=6,
    num_interp_steps=6,
    lon_pad=0,
    drop_variables=None,
    seed=None,
    amp_dtype=None,
    dit=None,
    device="cpu",
    t0=None,
):
    t0 = t0 if t0 is not None else Y0 + np.timedelta64(120, "h")
    src = PhooSource()
    lon2d, lat2d = np.meshgrid(LON, LAT)
    model = InterpCRPSDiT(
        dit=dit if dit is not None else PhooDiT(),
        center=torch.zeros(1, len(VARIABLES), 1, 1),
        scale=torch.ones(1, len(VARIABLES), 1, 1),
        static_inv=torch.zeros(12, H, W),
        lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
        lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
        px_model=DataReplay(src, VARIABLES, OrderedDict(lat=LAT, lon=LON), step=gap_h),
        num_interp_steps=num_interp_steps,
        lon_pad=lon_pad,
        seed=seed,
        drop_variables=drop_variables,
        amp_dtype=amp_dtype,
    ).to(device)
    x0 = torch.from_numpy(src(np.array([t0]), VARIABLES).values.astype("float32")).to(
        device
    )[None, :, None]
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=LAT,
        lon=LON,
    )
    return model, x0, coords, float((t0 - Y0) / np.timedelta64(1, "h"))


@pytest.mark.parametrize(
    "gap_h, num_interp_steps",
    [
        (6, 6),
        (6, 36),
        (3, 3),
        (10, 10),
    ],  # gaps 3/6/10h x resolutions 1h/10min (3,3) & (10,10) = range edges
)
def test_interpcrpsdit_gap_and_resolution(gap_h, num_interp_steps):
    model, x0, coords, h0 = _build(gap_h=gap_h, num_interp_steps=num_interp_steps)
    step_min = gap_h * 60 // num_interp_steps
    leads, frames = [], []
    # islice to IC + one full bracket + first frame of the next (no extra fetch past the seam).
    for k, (xf, cf) in enumerate(
        itertools.islice(model.create_iterator(x0, coords), num_interp_steps + 2)
    ):
        leads.append(int(cf["lead_time"][-1] / np.timedelta64(1, "m")))
        frames.append(xf)
        assert xf.shape == (1, 1, 1, len(VARIABLES), H, W)
        assert torch.isfinite(xf).all()
    # hourly/sub-hourly ladder, no drift; endpoint at gap; and the coarse endpoint is verbatim source
    assert leads == [step_min * k for k in range(len(leads))]
    # coarse endpoint == source frame at t0 + gap (whole constant field == hour)
    endpoint = frames[num_interp_steps]
    assert torch.allclose(endpoint, torch.full_like(endpoint, h0 + gap_h), atol=1e-4)


def test_interpcrpsdit_pin_linear_interior():
    # Phoo residual f=0 + center0/scale1 => interior frame == linear interp of the two coarse frames.
    model, x0, coords, h0 = _build(gap_h=6, num_interp_steps=6)
    it = model.create_iterator(x0, coords)
    next(it)  # IC (tau handled as verbatim)
    x1, c1 = next(it)  # first interior sub-step, tau = 1/6
    tau = 1 / 6
    assert c1["lead_time"][-1] == np.timedelta64(1, "h")  # 6h gap / 6 steps = 1h
    # whole interior field == linear interp of the two constant coarse frames
    assert torch.allclose(x1, torch.full_like(x1, h0 + tau * 6.0), atol=1e-4)


def test_interpcrpsdit_gap_channel_value():
    # gap-conditioning channel = (gap_h - 6)/3, derived from the coarse step (not passed in):
    # capture the cond the DiT actually receives during create_iterator.
    for gap_h, expect in [(3, -1.0), (6, 0.0), (9, 1.0)]:
        dit = GapCaptureDiT()
        model, x0, coords, _ = _build(gap_h=gap_h, num_interp_steps=3, dit=dit)
        it = model.create_iterator(x0, coords)
        next(it)  # IC (no DiT call)
        next(it)  # first interior -- calls the DiT, populating gap_seen
        assert dit.gap_seen == pytest.approx(expect, abs=1e-6)


def test_interpcrpsdit_indivisible_cadence_raises():
    # num_interp_steps must divide the coarse gap (min) -- hard error regardless of range.
    model, x0, coords, _ = _build(gap_h=6, num_interp_steps=7)
    with pytest.raises(ValueError, match="must divide"):
        next(model.create_iterator(x0, coords))  # gap validated at the first (IC) step


@pytest.mark.parametrize("gap_h", [12, 2])  # above and below the trained 3-10h range
def test_interpcrpsdit_out_of_range_warns_but_runs(gap_h):
    # Out-of-range gaps are off-distribution: warn, but still run (divisible cadence, so no raise).
    model, x0, coords, _ = _build(gap_h=gap_h, num_interp_steps=6)
    it = model.create_iterator(x0, coords)
    with pytest.warns(UserWarning, match="outside the trained"):
        x_ic, _ = next(it)  # IC -- gap validated here
        x_interior, _ = next(it)  # first interior -- interpolation actually runs
    assert torch.isfinite(x_ic).all()
    assert torch.isfinite(x_interior).all()


def test_interpcrpsdit_drop_variables():
    # Dropped optional channels are declared unavailable: the base need not supply them and they are NOT
    # emitted -- the model outputs VARIABLES minus the dropped set, with the kept channels exact.
    model, x0, coords, _ = _build(drop_variables=["sp", "tcwv"])
    assert list(model.drop_idx) == [VARIABLES.index("sp"), VARIABLES.index("tcwv")]
    present = [v for v in VARIABLES if v not in ("sp", "tcwv")]
    it = model.create_iterator(x0, coords)
    ic_x, ic_c = next(
        it
    )  # IC endpoint: present variables only (variable axis = dim -3)
    assert list(ic_c["variable"]) == present
    assert ic_x.shape[-3] == len(present)  # sp, tcwv not emitted
    x1, c1 = next(it)  # first interior
    assert list(c1["variable"]) == present
    assert x1.shape[-3] == len(present)
    assert torch.isfinite(x1).all()
    assert torch.all(
        x1 != 0
    )  # every emitted channel carries the linear interp (nothing zeroed in output)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"num_interp_steps": 0}, "num_interp_steps"),  # invalid value
        ({"amp_dtype": torch.float64}, "amp_dtype must be"),  # invalid value
        (
            {"drop_variables": ["z500"]},
            "trained-optional",
        ),  # invalid variable (non-optional)
    ],
)
def test_interpcrpsdit_exceptions(kwargs, match):
    # invalid constructor values / variables raise ValueError
    with pytest.raises(ValueError, match=match):
        _build(**kwargs)


def test_interpcrpsdit_ensemble_and_seed():
    # NoiseDiT routes the drawn latent z into the output, so this test covers seeding + spread.
    def one(seed: int) -> torch.Tensor:
        model, x0, coords, _ = _build(num_interp_steps=6, seed=seed, dit=NoiseDiT())
        x = x0.repeat(3, 1, 1, 1, 1, 1)  # 3 ensemble members
        c = OrderedDict(coords)
        c["batch"] = np.arange(3)
        it = model.create_iterator(x, c)
        next(it)
        xi, _ = next(it)  # first interior sub-step
        return xi

    seed0_a, seed0_b, seed1 = one(seed=0), one(seed=0), one(seed=1)
    assert torch.allclose(seed0_a, seed0_b)  # same seed -> reproducible
    assert not torch.allclose(seed0_a[0], seed0_a[1])  # distinct members (z per member)
    assert not torch.allclose(seed0_a, seed1)  # different seed -> different draw


def _build_sub(halo=0, min_cells=4, gap_h=6, num_interp_steps=6):
    # global 16x32 model -> regional sub-domain; attach a px_model on the sub's run grid.
    model, _, _, _ = _build(gap_h=gap_h, num_interp_steps=num_interp_steps)
    sub = model.set_domain(
        lat_min=10, lat_max=70, lon_min=50, lon_max=200, halo=halo, min_cells=min_cells
    )
    rc = sub.run_coords()  # public API: the grid to build a regional px_model against
    runlat, runlon = rc["lat"], rc["lon"]
    src = GridSource(runlat, runlon, LAT, LON)
    sub.px_model = DataReplay(
        src, VARIABLES, OrderedDict(lat=runlat, lon=runlon), step=gap_h
    )
    t0 = Y0 + np.timedelta64(120, "h")
    x0 = torch.from_numpy(src(np.array([t0]), VARIABLES).values.astype("float32"))[
        None, :, None
    ]
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=runlat,
        lon=runlon,
    )
    return sub, x0, coords


@pytest.mark.parametrize("halo, exp_halo", [(0, (0, 0, 1, 0)), (2, (2, 2, 3, 2))])
def test_interpcrpsdit_set_domain(halo, exp_halo):
    sub, x0, coords = _build_sub(halo=halo)
    frames = list(itertools.islice(sub.create_iterator(x0, coords), 3))
    # the emitted output grid (public: from the frame coords) == the exact bounding box interior
    outlat, outlon = frames[0][1]["lat"], frames[0][1]["lon"]
    np.testing.assert_array_equal(outlat, LAT[2:8])  # lat 10..70 on the 16-row grid
    np.testing.assert_array_equal(outlon, LON[5:18])  # lon 50..200 on the 32-col grid
    nlat_o, nlon_o = len(outlat), len(outlon)
    assert sub._halo == exp_halo  # asserts the per-side halo trim (internal _halo)
    # run grid (what the DiT/px see) is >= output; with halo, larger overall (per-side pinned by _halo)
    run = sub.run_coords()
    if halo:
        assert len(run["lat"]) > nlat_o and len(run["lon"]) > nlon_o
    # frame 0 is the IC, emitted verbatim -> must equal the source's exact bounding box sub-block (spatially
    # varying source, so a correct-size-but-wrong-offset trim would fail this).
    h0 = float((coords["time"][0] - Y0) / np.timedelta64(1, "h"))
    ri = np.arange(2, 8, dtype="float32")[:, None]
    ci = np.arange(5, 18, dtype="float32")[None, :]
    expected_ic = h0 + ri + 0.01 * ci  # (6, 13)
    for k, (xf, cf) in enumerate(frames):
        assert xf.shape[-2:] == (
            nlat_o,
            nlon_o,
        )  # every emitted frame is the trimmed bounding box
        np.testing.assert_array_equal(cf["lat"], outlat)
        np.testing.assert_array_equal(cf["lon"], outlon)
        assert torch.isfinite(xf).all()
        if k == 0:
            np.testing.assert_allclose(
                xf[0, 0, 0, 0].cpu().numpy(), expected_ic, atol=1e-4
            )


def test_interpcrpsdit_crops_superset_base_grid():
    # A base on a grid that contains the run grid (e.g. SFNO's 721 lat vs the model's 720) is
    # reconciled onto the run grid via map_coords -- here a base with one extra southern row.
    model, _, _, h0 = _build()
    step = LAT[0] - LAT[1]
    lat_big = np.concatenate(
        [LAT, [LAT[-1] - step]]
    )  # H+1 rows; LAT is the contiguous top slice
    src = GridSource(lat_big, LON, lat_big, LON)
    model.px_model = DataReplay(
        src, VARIABLES, OrderedDict(lat=lat_big, lon=LON), step=6
    )
    x0 = torch.from_numpy(
        src(np.array([Y0 + np.timedelta64(120, "h")]), VARIABLES).values.astype(
            "float32"
        )
    )[None, :, None]
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([Y0 + np.timedelta64(120, "h")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=lat_big,
        lon=LON,
    )
    xf, cf = next(
        model.create_iterator(x0, coords)
    )  # IC -- cropped to the run grid, retained values unmodified
    assert xf.shape[-2:] == (H, W)  # cropped from H+1 -> H
    # emitted on the run grid, not the base grid
    np.testing.assert_array_equal(cf["lat"], LAT)
    assert torch.isfinite(xf).all()
    # IC equals the source's top-H rows (the extra southern row is dropped, not blended in)
    expected = (
        h0
        + np.arange(H, dtype="float32")[:, None]
        + 0.01 * np.arange(W, dtype="float32")[None, :]
    )
    np.testing.assert_allclose(xf[0, 0, 0, 0].cpu().numpy(), expected, atol=1e-4)
    assert (
        xf[0, 0, 0, 0].max().item() < h0 + H
    )  # the extra row (values >= h0+H) was cropped, not averaged


def test_interpcrpsdit_input_coords_reports_px_grid():
    # input_coords reflects the wrapped px_model's grid (base is fed to px_model on its grid, then
    # map_coords'd onto the run grid), matching InterpModAFNO; run_coords is the model's own run grid.
    model, _, _, _ = _build()
    step = LAT[0] - LAT[1]
    lat_big = np.concatenate([LAT, [LAT[-1] - step]])
    model.px_model = DataReplay(
        GridSource(lat_big, LON, lat_big, LON),
        VARIABLES,
        OrderedDict(lat=lat_big, lon=LON),
        step=6,
    )
    # input_coords reports the full px grid (H+1 rows), value-for-value
    np.testing.assert_array_equal(model.input_coords()["lat"], lat_big)
    np.testing.assert_array_equal(model.run_coords()["lat"], LAT)  # run grid (H)


def test_interpcrpsdit_run_coords():
    model, _, _, _ = _build()
    # pre-set_domain: run grid == the full model grid
    np.testing.assert_array_equal(model.run_coords()["lat"], LAT)
    np.testing.assert_array_equal(model.run_coords()["lon"], LON)
    sub = model.set_domain(
        lat_min=10, lat_max=70, lon_min=50, lon_max=200, halo=2, min_cells=4
    )
    # post-set_domain: run grid == bounding box + halo, patch-snapped (rows 0..10, cols 2..20 for this bounding box)
    np.testing.assert_array_equal(sub.run_coords()["lat"], LAT[0:10])
    np.testing.assert_array_equal(sub.run_coords()["lon"], LON[2:20])
    # run grid is strictly larger than the trimmed output (bounding box interior LAT[2:8]/LON[5:18])
    assert len(sub.run_coords()["lat"]) > len(LAT[2:8])
    assert len(sub.run_coords()["lon"]) > len(LON[5:18])
    # run_coords reads the fixed grid directly, so detaching the px_model leaves it unchanged
    before = sub.run_coords()
    sub.px_model = None
    after = sub.run_coords()
    np.testing.assert_array_equal(after["lat"], before["lat"])
    np.testing.assert_array_equal(after["lon"], before["lon"])


def test_interpcrpsdit_set_domain_global_base():
    # A global base (not a regional replay) drives a sub-domain: map_coords crops global -> bounding box+halo,
    # exercising both map_coords legs plus interior interpolation on the cropped run grid.
    model, _, _, _ = _build()
    sub = model.set_domain(
        lat_min=10, lat_max=70, lon_min=50, lon_max=200, halo=2, min_cells=4
    )
    src = GridSource(LAT, LON, LAT, LON)  # full global grid, not the sub run grid
    sub.px_model = DataReplay(src, VARIABLES, OrderedDict(lat=LAT, lon=LON), step=6)
    t0 = Y0 + np.timedelta64(120, "h")
    x0 = torch.from_numpy(src(np.array([t0]), VARIABLES).values.astype("float32"))[
        None, :, None
    ]
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=LAT,
        lon=LON,
    )
    frames = list(itertools.islice(sub.create_iterator(x0, coords), 3))
    # emitted output grid == the bounding box interior, even though the base was global
    outlat, outlon = frames[0][1]["lat"], frames[0][1]["lon"]
    np.testing.assert_array_equal(outlat, LAT[2:8])
    np.testing.assert_array_equal(outlon, LON[5:18])
    nlat_o, nlon_o = len(outlat), len(outlon)
    # Zero residual (PhooDiT) + 1h substeps => frame k == h0 + k + (bounding box sub-block). Checking every
    # frame's exact values exercises both map_coords legs (correct crop offset, via the spatially
    # varying block) and the interior linear interp (via the +k).
    h0 = float((t0 - Y0) / np.timedelta64(1, "h"))
    block = (
        np.arange(2, 8, dtype="float32")[:, None]
        + 0.01 * np.arange(5, 18, dtype="float32")[None, :]
    )
    for k, (xf, cf) in enumerate(frames):
        assert xf.shape[-2:] == (
            nlat_o,
            nlon_o,
        )  # cropped global -> bounding box interior
        np.testing.assert_array_equal(cf["lat"], outlat)
        np.testing.assert_array_equal(cf["lon"], outlon)
        assert torch.isfinite(xf).all()
        np.testing.assert_allclose(
            xf[0, 0, 0, 0].cpu().numpy(), h0 + k + block, atol=1e-4
        )


def test_interpcrpsdit_set_domain_bbox_outside_raises():
    model, _, _, _ = _build()
    with pytest.raises(ValueError, match="outside the current grid"):
        model.set_domain(
            lat_min=-100, lat_max=70, lon_min=50, lon_max=200, min_cells=4
        )  # lat below grid


def test_interpcrpsdit_set_domain_too_small_raises():
    model, _, _, _ = _build()
    with pytest.raises(
        ValueError, match="per-side minimum"
    ):  # bounding box < min_cells per side
        model.set_domain(
            lat_min=10, lat_max=70, lon_min=50, lon_max=200, min_cells=1000
        )


@pytest.mark.parametrize(
    "amp_dtype, expect_autocast", [(None, False), (torch.bfloat16, True)]
)
def test_interpcrpsdit_amp_dtype(amp_dtype, expect_autocast):
    # amp_dtype toggles autocast around the DiT forward.
    dit = AutocastProbeDiT()
    model, x0, coords, _ = _build(amp_dtype=amp_dtype, dit=dit)
    frames = [
        xf for xf, _ in itertools.islice(model.create_iterator(x0, coords), 2)
    ]  # IC + 1 interior
    assert dit.saw_autocast is expect_autocast
    assert torch.isfinite(frames[-1]).all()


def test_interpcrpsdit_requires_px_model():
    lon2d, lat2d = np.meshgrid(LON, LAT)
    model = InterpCRPSDiT(
        dit=PhooDiT(),
        center=torch.zeros(1, len(VARIABLES), 1, 1),
        scale=torch.ones(1, len(VARIABLES), 1, 1),
        static_inv=torch.zeros(12, H, W),
        lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
        lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
        px_model=None,
    )
    # input_coords derives from px_model, so it raises when none is attached
    with pytest.raises(ValueError, match="px_model, must be set"):
        model.input_coords()
    # the run path guards separately (a different raise site)
    x = torch.zeros(1, 1, 1, len(VARIABLES), H, W)
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([Y0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=LAT,
        lon=LON,
    )
    with pytest.raises(ValueError, match="px_model, must be set"):
        next(model.create_iterator(x, coords))


def test_interpcrpsdit_set_domain_negative_halo_raises():
    model, _, _, _ = _build()
    with pytest.raises(ValueError, match="halo must be non-negative"):
        model.set_domain(lat_min=10, lat_max=70, lon_min=50, lon_max=200, halo=-1)


def test_interpcrpsdit_set_domain_bad_min_cells_raises():
    model, _, _, _ = _build()
    with pytest.raises(ValueError, match="min_cells must be >= 1"):
        model.set_domain(lat_min=10, lat_max=70, lon_min=50, lon_max=200, min_cells=0)


def test_interpcrpsdit_lon_pad_periodic():
    # lon_pad>0 exercises the circular longitude pad + periodic noise draw; with PhooDiT f=0 the
    # pinned output is still the exact linear interp (the pad is added, then cropped back off).
    model, x0, coords, h0 = _build(lon_pad=4)
    it = model.create_iterator(x0, coords)
    next(it)  # IC
    x1, _ = next(it)  # first interior sub-step (tau=1/6)
    assert x1.shape[-2:] == (
        H,
        W,
    )  # circular pad added then cropped back to the run grid
    assert torch.isfinite(x1).all()
    assert torch.allclose(x1, torch.full_like(x1, h0 + (1.0 / 6.0) * 6.0), atol=1e-4)


def test_interpcrpsdit_multi_time_independent_noise():
    # Two init times in one call get independent per-member latents. Using a time-invariant source so
    # the linear base is identical across times, any difference between the two outputs is purely noise.
    class ConstSource:
        def __call__(self, time, variable):
            times = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
            variable = [variable] if isinstance(variable, str) else list(variable)
            arr = np.full((len(times), len(variable), H, W), 5.0, dtype="float32")
            return xr.DataArray(
                arr,
                dims=["time", "variable", "lat", "lon"],
                coords=dict(time=times, variable=np.array(variable), lat=LAT, lon=LON),
            )

    model, _, _, _ = _build(num_interp_steps=6, seed=0, dit=NoiseDiT())
    model.px_model = DataReplay(
        ConstSource(), VARIABLES, OrderedDict(lat=LAT, lon=LON), step=6
    )
    t0 = Y0 + np.timedelta64(120, "h")
    x = torch.full(
        (1, 2, 1, len(VARIABLES), H, W), 5.0
    )  # two init times, identical field
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0, t0 + np.timedelta64(12, "h")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=LAT,
        lon=LON,
    )
    it = model.create_iterator(x, coords)
    next(it)  # IC
    xi, _ = next(it)  # first interior sub-step; shape (1, 2, 1, 73, H, W)
    assert xi.shape[1] == 2  # both init times emitted
    # identical base across times => the two differ only via their independent per-time latents
    assert not torch.allclose(xi[:, 0], xi[:, 1])


def test_interpcrpsdit_load_model(tmp_path, monkeypatch):
    # Exercise load_model with a mocked package: dummy .mdlus (from_checkpoint patched),
    # real set_phys.nc invariant stack + normalization stats.
    class _FakeMod:
        @staticmethod
        def from_checkpoint(path):
            return PhooDiT()

    monkeypatch.setattr(interp_mod, "PhysicsNemoModule", _FakeMod)
    (tmp_path / "CRPSModel.mdlus").write_bytes(
        b""
    )  # presence only; from_checkpoint is mocked
    (tmp_path / "config.json").write_text(
        '{"name": "test"}'
    )  # resolved for HF download tracking
    np.save(
        tmp_path / "global_means.npy", np.zeros((1, len(VARIABLES), 1, 1), "float32")
    )
    np.save(tmp_path / "global_stds.npy", np.ones((1, len(VARIABLES), 1, 1), "float32"))
    hh, ww = 4, 8
    lat = np.linspace(90, -90, hh, endpoint=False)  # descending, as load_model requires
    lon = np.linspace(0, 360, ww, endpoint=False)
    ramp = np.arange(hh * ww, dtype="float32").reshape(
        hh, ww
    )  # nonzero-variance invariant fields
    invars = ["lsm", "z"] + PHYS_EXTRA
    xr.Dataset(
        {
            v: (["latitude", "longitude"], (ramp + i).astype("float32"))
            for i, v in enumerate(invars)
        },
        coords=dict(latitude=lat, longitude=lon),
    ).to_netcdf(tmp_path / "set_phys.nc")

    model = InterpCRPSDiT.load_model(Package(str(tmp_path)))
    assert model.center.shape == (1, len(VARIABLES), 1, 1)
    assert model.static_inv.shape == (12, hh, ww)  # 4 sin/cos + lsm + z + 6 phys
    assert torch.isfinite(model.static_inv).all()
    assert (
        model._min_domain_cells == 64
    )  # PhooDiT exposes no attn_kernel -> pre-load default kept


@pytest.mark.parametrize("time_h", [120, 48])  # different init times
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
        ),
    ],
)
def test_interpcrpsdit_call(device, time_h):
    # __call__ (single-step forward) returns the initial condition (step 0) -- matching
    # create_iterator's first yield (the InterpModAFNO px contract), not a future sub-step.
    t0 = Y0 + np.timedelta64(time_h, "h")
    model, x0, coords, h0 = _build(device=device, t0=t0)
    x_call, c_call = model(x0, coords)
    x_iter, c_iter = next(model.create_iterator(x0, coords))
    assert torch.allclose(x_call, x_iter)  # __call__ == create_iterator step 0
    assert x_call.shape[-2:] == (H, W)
    assert torch.allclose(
        x_call, torch.full_like(x_call, h0), atol=1e-4
    )  # IC = source at t0
    np.testing.assert_array_equal(np.asarray(c_call["lat"]), np.asarray(c_iter["lat"]))


@pytest.mark.parametrize("ensemble", [1, 2])
def test_interpcrpsdit_iter(ensemble):
    # create_iterator yields the IC first, then advances lead_time; preserves the ensemble/batch dim.
    model, x0, coords, h0 = _build()
    x = x0.repeat(ensemble, 1, 1, 1, 1, 1)
    c = OrderedDict(coords)
    c["batch"] = np.arange(ensemble)
    it = model.create_iterator(x, c)
    ic_x, ic_c = next(it)  # step 0 = initial condition
    assert ic_x.shape[0] == ensemble  # ensemble/batch dim preserved
    assert int(ic_c["lead_time"][-1] / np.timedelta64(1, "m")) == 0  # IC at lead 0
    assert torch.allclose(ic_x, torch.full_like(ic_x, h0), atol=1e-4)
    s1_x, s1_c = next(it)  # first interior sub-step
    assert s1_x.shape[0] == ensemble
    assert s1_c["lead_time"][-1] > ic_c["lead_time"][-1]  # lead_time advanced
    assert torch.isfinite(s1_x).all()


@pytest.mark.package
def test_interpcrpsdit_package():
    # Real-weights smoke test (needs --package + a hosted default package + the interp-crps-dit extra;
    # skipped otherwise). Loads the default package, builds a regional model, runs one real DiT forward.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    package = InterpCRPSDiT.load_default_package()
    model = InterpCRPSDiT.load_model(package).to(device).eval()
    assert model.center.shape == (1, len(VARIABLES), 1, 1)
    assert model.static_inv.shape[0] == 12
    assert bool(torch.isfinite(model.static_inv).all())

    sub = model.set_domain(
        lat_min=10.0, lat_max=32.0, lon_min=100.0, lon_max=124.0, halo=0
    )
    rc = sub.run_coords()
    rlat, rlon = rc["lat"], rc["lon"]
    hs, ws = len(rlat), len(rlon)
    mean = model.center.detach().cpu().numpy().reshape(len(VARIABLES), 1, 1)

    class _MeanSrc:  # in-distribution constant field = per-variable mean
        def __call__(self, time, variable):
            times = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
            var = [variable] if isinstance(variable, str) else list(variable)
            arr = np.broadcast_to(mean, (len(times), len(VARIABLES), hs, ws)).astype(
                "float32"
            )
            return xr.DataArray(
                arr,
                dims=["time", "variable", "lat", "lon"],
                coords=dict(time=times, variable=np.array(var), lat=rlat, lon=rlon),
            )

    sub.px_model = DataReplay(
        _MeanSrc(), VARIABLES, OrderedDict(lat=rlat, lon=rlon), step=6
    )
    x0 = torch.from_numpy(
        np.broadcast_to(mean, (1, 1, 1, len(VARIABLES), hs, ws)).astype("float32")
    ).to(device)
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([Y0 + np.timedelta64(120, "h")]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=rlat,
        lon=rlon,
    )
    it = sub.create_iterator(x0, coords)
    next(it)  # IC (verbatim)
    xi, _ = next(it)  # first interior -- real natten2d_rope DiT forward
    assert bool(torch.isfinite(xi).all())


class ZCaptureDiT(PhooDiT):
    """Records the trailing 6 latent channels the DiT sees each forward, to check the per-bracket
    latent is drawn once and reused across all interior sub-steps."""

    def __init__(self) -> None:
        super().__init__()
        self.z_seen: list[torch.Tensor] = []

    def forward(self, x, t, condition=None, attn_kwargs=None):
        self.z_seen.append(x[:, -6:].clone())  # DiT input tail = z(6)
        return x.new_zeros(x.shape[0], len(VARIABLES), x.shape[-2], x.shape[-1])


class ShiftedSource:
    """Source on a grid offset from the model grid (no exact members), to trigger the nearest-regrid
    warning when it drives the interpolator."""

    def __init__(self, lat, lon):
        self.lat, self.lon = np.asarray(lat), np.asarray(lon)

    def __call__(self, time, variable):
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
        variable = [variable] if isinstance(variable, str) else list(variable)
        arr = np.zeros(
            (len(times), len(variable), len(self.lat), len(self.lon)), "float32"
        )
        for i, t in enumerate(times):
            arr[i] = float((t - Y0) / np.timedelta64(1, "h"))
        return xr.DataArray(
            arr,
            dims=["time", "variable", "lat", "lon"],
            coords=dict(
                time=times, variable=np.array(variable), lat=self.lat, lon=self.lon
            ),
        )


def test_interpcrpsdit_bracket_latent_coherent():
    # The per-member latent z is drawn once per bracket and reused across every interior sub-step
    # (temporal coherence within a bracket), but re-drawn fresh for each new bracket (independent noise
    # across brackets -- otherwise every bracket would share one realization).
    dit = ZCaptureDiT()
    model, x0, coords, _ = _build(gap_h=6, num_interp_steps=3, seed=0, dit=dit)
    # IC + two full brackets: each bracket = 2 interior DiT calls, so z_seen = [b1, b1, b2, b2].
    list(itertools.islice(model.create_iterator(x0, coords), 7))
    assert len(dit.z_seen) == 4
    assert (
        float(dit.z_seen[0].abs().sum()) > 0.0
    )  # latent actually flows (seeded, non-zero)
    assert torch.equal(
        dit.z_seen[0], dit.z_seen[1]
    )  # bracket 1: same latent across sub-steps
    assert torch.equal(
        dit.z_seen[2], dit.z_seen[3]
    )  # bracket 2: same latent across sub-steps
    assert not torch.equal(
        dit.z_seen[0], dit.z_seen[2]
    )  # different latent between brackets


def test_interpcrpsdit_front_hook_applied():
    # front_hook transforms each incoming base frame; with a zero-residual DiT the +5 offset propagates
    # through the linear base to every emitted frame (here checked on the IC = base at t0).
    model, x0, coords, h0 = _build(gap_h=6, num_interp_steps=3)
    calls = []

    def front(a, c):
        calls.append(True)
        return a + 5.0, c

    model.front_hook = front
    frames = list(itertools.islice(model.create_iterator(x0, coords), 4))
    assert calls  # invoked per base frame
    ic = frames[0][0]
    assert torch.allclose(ic, torch.full_like(ic, h0 + 5.0), atol=1e-4)


def test_interpcrpsdit_rear_hook_output_not_reused_as_anchor():
    # rear_hook (+100) transforms every emitted frame, but the interpolation anchor is the UN-hooked
    # base frame -- so exact interior values stay the correct linear ramp + the constant offset, with
    # no accumulation across brackets. If the hooked output were fed back as the next anchor, bracket 2
    # would drift; asserting exact values across two brackets proves the rollout state is not corrupted.
    model, x0, coords, h0 = _build(gap_h=6, num_interp_steps=3)
    seen = []

    def rear(a, c):
        seen.append(int(c["lead_time"][-1] / np.timedelta64(1, "m")))
        return a + 100.0, c

    model.rear_hook = rear
    frames = [xf for xf, _ in itertools.islice(model.create_iterator(x0, coords), 7)]
    # IC(0), bracket1 interior(1,2)+endpoint(3), bracket2 interior(4,5)+endpoint(6); PhooDiT f=0 ->
    # each interior is the linear interp of the un-hooked coarse endpoints (source hour = value).
    expected = [h0, h0 + 2, h0 + 4, h0 + 6, h0 + 8, h0 + 10, h0 + 12]
    assert len(seen) == len(frames)  # every emitted frame passed through rear_hook
    for xf, e in zip(frames, expected):
        assert torch.allclose(xf, torch.full_like(xf, e + 100.0), atol=1e-4)


def test_interpcrpsdit_regrid_warns_off_grid_base():
    # A base whose grid is not a subset of the model grid triggers the nearest-regrid warning
    # (map_coords falls back to interpolation instead of an exact crop).
    lat2 = LAT + 0.3  # offset -> not members of the model's LAT grid
    src = ShiftedSource(lat2, LON)
    lon2d, lat2d = np.meshgrid(LON, LAT)
    model = InterpCRPSDiT(
        dit=PhooDiT(),
        center=torch.zeros(1, len(VARIABLES), 1, 1),
        scale=torch.ones(1, len(VARIABLES), 1, 1),
        static_inv=torch.zeros(12, H, W),
        lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
        lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
        px_model=DataReplay(src, VARIABLES, OrderedDict(lat=lat2, lon=LON), step=6),
        num_interp_steps=6,
    )
    t0 = Y0 + np.timedelta64(120, "h")
    x0 = torch.from_numpy(src(np.array([t0]), VARIABLES).values.astype("float32"))[
        None, :, None
    ]
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=lat2,
        lon=LON,
    )
    with pytest.warns(UserWarning, match="nearest-regridded"):
        next(model.create_iterator(x0, coords))


def test_interpcrpsdit_load_model_min_domain_cells(tmp_path, monkeypatch):
    # load_model derives _min_domain_cells from the DiT's attn_kernel x patch_size (not the pre-load
    # default of 64). Mocked package -- no real weights.
    class AttnDiT(PhooDiT):
        attn_kernel = 23  # exposed like the real natten2d_rope DiT

    class _FakeMod:
        @staticmethod
        def from_checkpoint(path):
            return AttnDiT()

    monkeypatch.setattr(interp_mod, "PhysicsNemoModule", _FakeMod)
    (tmp_path / "CRPSModel.mdlus").write_bytes(b"")
    (tmp_path / "config.json").write_text(
        '{"name": "test"}'
    )  # resolved for HF download tracking
    np.save(
        tmp_path / "global_means.npy", np.zeros((1, len(VARIABLES), 1, 1), "float32")
    )
    np.save(tmp_path / "global_stds.npy", np.ones((1, len(VARIABLES), 1, 1), "float32"))
    hh, ww = 4, 8
    lat = np.linspace(90, -90, hh, endpoint=False)
    lon = np.linspace(0, 360, ww, endpoint=False)
    ramp = np.arange(hh * ww, dtype="float32").reshape(hh, ww)
    invars = ["lsm", "z"] + PHYS_EXTRA
    xr.Dataset(
        {
            v: (["latitude", "longitude"], (ramp + i).astype("float32"))
            for i, v in enumerate(invars)
        },
        coords=dict(latitude=lat, longitude=lon),
    ).to_netcdf(tmp_path / "set_phys.nc")
    model = InterpCRPSDiT.load_model(Package(str(tmp_path)))
    assert model._min_domain_cells == 23 * max(
        AttnDiT.patch_size
    )  # max kernel * max patch = 46


def test_interpcrpsdit_lon_pad_exceeds_width():
    # _cpad must stay genuinely periodic when lon_pad exceeds the grid width (indices wrap via modulo);
    # a slice-based pad would clamp and _sample_tau's crop would collapse to zero width. With PhooDiT
    # f=0 the round-trip must still yield the exact linear interp, not merely a finite field.
    model, x0, coords, h0 = _build(
        lon_pad=2 * W + 8
    )  # pad wider than the grid (and > 2*W)
    it = model.create_iterator(x0, coords)
    next(it)  # IC
    xi, _ = next(it)  # first interior sub-step (tau = 1/6)
    assert xi.shape[-2:] == (H, W)  # cropped back to the grid, not collapsed
    # exact linear interp of the two constant coarse frames (6 h gap / 6 steps -> tau = 1/6)
    assert torch.allclose(xi, torch.full_like(xi, h0 + (1.0 / 6.0) * 6.0), atol=1e-4)


class DropInputProbeDiT(PhooDiT):
    """Captures, for dropped channels, the DiT-visible normalized input (first 73 = x0n) and the
    presence-mask entry (last 73 of the 87-ch cond block), so a test can assert both are zero. DiT input
    layout: [x0n(73), xTn(73), cond(87), z(6)]; cond = static(12) + cos_zenith + gap + mask(73).
    """

    def __init__(self, drop_idx) -> None:
        super().__init__()
        self.drop_idx = drop_idx
        self.dropped_input: torch.Tensor | None = None
        self.dropped_mask: torch.Tensor | None = None

    def forward(self, x, t, condition=None, attn_kwargs=None):
        self.dropped_input = x[:, self.drop_idx].clone()  # x0n at the dropped positions
        mask0 = (
            73 + 73 + 14
        )  # presence mask starts at cond[14] (after static + cos_zenith + gap)
        self.dropped_mask = x[:, [mask0 + i for i in self.drop_idx]].clone()
        return x.new_zeros(x.shape[0], len(VARIABLES), x.shape[-2], x.shape[-1])


def test_interpcrpsdit_missing_optional_base():
    # A base that supplies only VARIABLES minus the optional four (like Pangu's 69) drives the
    # interpolator when those four are dropped: output is the 69 present variables (endpoint-exact),
    # and the absent channels are expanded to normalized zero internally (never emitted).
    present = [v for v in VARIABLES if v not in OPTIONAL_VARIABLES]  # 69
    drop_idx = [VARIABLES.index(v) for v in OPTIONAL_VARIABLES]
    dit = DropInputProbeDiT(drop_idx)
    lon2d, lat2d = np.meshgrid(LON, LAT)
    src = (
        PhooSource()
    )  # returns exactly the requested variables as hour-constant fields
    model = InterpCRPSDiT(
        dit=dit,
        center=torch.full(
            (1, len(VARIABLES), 1, 1), 5.0
        ),  # nonzero -> pad must use center, not 0
        scale=torch.ones(1, len(VARIABLES), 1, 1),
        static_inv=torch.zeros(12, H, W),
        lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
        lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
        px_model=DataReplay(src, present, OrderedDict(lat=LAT, lon=LON), step=6),
        num_interp_steps=6,
        lon_pad=0,  # small synthetic grid
        drop_variables=OPTIONAL_VARIABLES,
    )
    t0 = Y0 + np.timedelta64(120, "h")
    h0 = float((t0 - Y0) / np.timedelta64(1, "h"))
    x0 = torch.from_numpy(src(np.array([t0]), present).values.astype("float32"))[
        None, :, None
    ]
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(present),
        lat=LAT,
        lon=LON,
    )
    it = model.create_iterator(x0, coords)
    ic_x, ic_c = next(
        it
    )  # IC endpoint: only the 69 present variables, exact base values
    assert list(ic_c["variable"]) == present
    assert ic_x.shape[-3] == len(present)
    assert torch.allclose(ic_x, torch.full_like(ic_x, h0), atol=1e-4)  # endpoint-exact
    x1, c1 = next(it)  # first interior -- DiT forward runs
    assert list(c1["variable"]) == present
    assert torch.isfinite(x1).all()
    # padded absent channels reach the DiT as normalized zero (center-filled, not physical zero)
    assert dit.dropped_input is not None
    assert torch.allclose(
        dit.dropped_input, torch.zeros_like(dit.dropped_input), atol=1e-6
    )
    # and their presence-mask entries are zero (declared absent to the DiT)
    assert dit.dropped_mask is not None
    assert torch.all(dit.dropped_mask == 0)


def test_interpcrpsdit_fractional_minute_gap_raises():
    # A base whose coarse step is not a whole number of minutes must raise -- rounding it would
    # silently shift every emitted lead time.
    src = PhooSource()
    lon2d, lat2d = np.meshgrid(LON, LAT)
    model = InterpCRPSDiT(
        dit=PhooDiT(),
        center=torch.zeros(1, len(VARIABLES), 1, 1),
        scale=torch.ones(1, len(VARIABLES), 1, 1),
        static_inv=torch.zeros(12, H, W),
        lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
        lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
        px_model=DataReplay(
            src, VARIABLES, OrderedDict(lat=LAT, lon=LON), step=np.timedelta64(90, "s")
        ),  # 1.5 min
        num_interp_steps=6,
        lon_pad=0,
    )
    t0 = Y0 + np.timedelta64(120, "h")
    x0 = torch.from_numpy(src(np.array([t0]), VARIABLES).values.astype("float32"))[
        None, :, None
    ]
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(VARIABLES),
        lat=LAT,
        lon=LON,
    )
    with pytest.raises(ValueError, match="whole number of minutes"):
        next(model.create_iterator(x0, coords))


def _write_pkg(tmp_path, monkeypatch, means=None, stds=None, nlat=4, nlon=8, dit=None):
    # Write a minimal mocked InterpCRPSDiT package (no real weights) and patch from_checkpoint.
    class _FakeMod:
        @staticmethod
        def from_checkpoint(path):
            return dit if dit is not None else PhooDiT()

    monkeypatch.setattr(interp_mod, "PhysicsNemoModule", _FakeMod)
    (tmp_path / "CRPSModel.mdlus").write_bytes(b"")
    (tmp_path / "config.json").write_text('{"name": "test"}')
    means = np.zeros((len(VARIABLES),), "float32") if means is None else means
    stds = np.ones((len(VARIABLES),), "float32") if stds is None else stds
    np.save(tmp_path / "global_means.npy", means)
    np.save(tmp_path / "global_stds.npy", stds)
    # 721 spans 90..-90 inclusive (a real 0.25 deg grid ending at the -90 pole); smaller grids match the
    # module LAT (endpoint excluded).
    lat = np.linspace(90, -90, nlat, endpoint=(nlat == 721))
    lon = np.linspace(0, 360, nlon, endpoint=False)
    ramp = np.arange(nlat * nlon, dtype="float32").reshape(nlat, nlon)
    invars = ["lsm", "z"] + PHYS_EXTRA
    xr.Dataset(
        {
            v: (["latitude", "longitude"], (ramp + i).astype("float32"))
            for i, v in enumerate(invars)
        },
        coords=dict(latitude=lat, longitude=lon),
    ).to_netcdf(tmp_path / "set_phys.nc")


@pytest.mark.parametrize(
    "kind, match",
    [
        ("zero_std", "strictly positive"),  # a zero scale -> would divide to inf
        ("short_means", "must have"),  # wrong channel count
        ("nonfinite_std", "non-finite"),  # NaN/inf normalization
    ],
)
def test_interpcrpsdit_load_model_bad_normalization(tmp_path, monkeypatch, kind, match):
    means = np.zeros((len(VARIABLES),), "float32")
    stds = np.ones((len(VARIABLES),), "float32")
    if kind == "zero_std":
        stds[3] = 0.0
    elif kind == "short_means":
        means = means[:-1]  # 72 channels
    elif kind == "nonfinite_std":
        stds[5] = np.nan
    _write_pkg(tmp_path, monkeypatch, means=means, stds=stds)
    with pytest.raises(ValueError, match=match):
        InterpCRPSDiT.load_model(Package(str(tmp_path)))


def test_interpcrpsdit_load_model_crops_721_to_720(tmp_path, monkeypatch):
    # A 721-lat set_phys.nc is cropped to the 720 training grid (the -90 pole row dropped).
    _write_pkg(tmp_path, monkeypatch, nlat=721, nlon=8)
    model = InterpCRPSDiT.load_model(Package(str(tmp_path)))
    assert model.static_inv.shape == (12, 720, 8)  # pole row dropped
    assert model.lat2d.shape == (720, 8)
    # last retained row is -89.75 (the -90 pole was dropped from the 0.25 deg grid)
    assert model.lat2d[-1, 0].item() == pytest.approx(-89.75)


def test_interpcrpsdit_pad_fills_dropped_with_center():
    # _pad_present_to_full fills dropped channels with their physical center (not zero); a per-channel
    # distinct center makes a wrong fill detectable. Present channels are preserved verbatim.
    center = (
        torch.arange(len(VARIABLES), dtype=torch.float32).reshape(
            1, len(VARIABLES), 1, 1
        )
        + 10.0
    )
    lon2d, lat2d = np.meshgrid(LON, LAT)
    model = InterpCRPSDiT(
        dit=PhooDiT(),
        center=center,
        scale=torch.ones(1, len(VARIABLES), 1, 1),
        static_inv=torch.zeros(12, H, W),
        lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
        lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
        px_model=None,
        num_interp_steps=6,
        lon_pad=0,
        drop_variables=OPTIONAL_VARIABLES,
    )
    x_present = torch.randn(1, 1, 1, len(model.present_idx), H, W)
    full = model._pad_present_to_full(x_present)
    assert full.shape[-3] == len(VARIABLES)
    assert torch.equal(
        full[..., model.present_idx, :, :], x_present
    )  # present preserved
    for i in model.drop_idx:  # dropped == that channel's physical center
        assert torch.allclose(
            full[..., i, :, :], torch.full((1, 1, 1, H, W), center[0, i, 0, 0].item())
        )


def test_interpcrpsdit_missing_optional_base_reorders_variables():
    # The base may emit present variables in its own native order (Pangu's here); map_coords must
    # reconcile them into VARIABLES order before padding, so every output channel carries its own values.
    from earth2studio.models.px.pangu import (
        VARIABLES as PANGU,
    )  # 69 vars, Pangu's native order

    present = [v for v in VARIABLES if v not in OPTIONAL_VARIABLES]
    assert set(PANGU) == set(present)  # same set of variables
    assert (
        list(PANGU) != present
    )  # but a different order -> genuinely exercises the reorder

    class VarValueSource:  # variable v -> constant field == its index in VARIABLES (distinct per var)
        def __call__(self, time, variable):
            times = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
            var = [variable] if isinstance(variable, str) else list(variable)
            arr = np.zeros((len(times), len(var), H, W), "float32")
            for j, v in enumerate(var):
                arr[:, j] = float(VARIABLES.index(v))
            return xr.DataArray(
                arr,
                dims=["time", "variable", "lat", "lon"],
                coords=dict(time=times, variable=np.array(var), lat=LAT, lon=LON),
            )

    src = VarValueSource()
    lon2d, lat2d = np.meshgrid(LON, LAT)
    model = InterpCRPSDiT(
        dit=PhooDiT(),
        center=torch.zeros(1, len(VARIABLES), 1, 1),
        scale=torch.ones(1, len(VARIABLES), 1, 1),
        static_inv=torch.zeros(12, H, W),
        lat2d=torch.as_tensor(lat2d, dtype=torch.float32),
        lon2d=torch.as_tensor(lon2d, dtype=torch.float32),
        px_model=DataReplay(
            src, list(PANGU), OrderedDict(lat=LAT, lon=LON), step=6
        ),  # base in Pangu order
        num_interp_steps=6,
        lon_pad=0,
        drop_variables=OPTIONAL_VARIABLES,
    )
    t0 = Y0 + np.timedelta64(120, "h")
    x0 = torch.from_numpy(src(np.array([t0]), list(PANGU)).values.astype("float32"))[
        None, :, None
    ]
    coords = OrderedDict(
        batch=np.array([0]),
        time=np.array([t0]),
        lead_time=np.array([np.timedelta64(0, "h")]),
        variable=np.array(PANGU),
        lat=LAT,
        lon=LON,
    )
    ic_x, ic_c = next(model.create_iterator(x0, coords))
    assert (
        list(ic_c["variable"]) == present
    )  # emitted in VARIABLES order (minus dropped)
    for j, v in enumerate(
        present
    ):  # each channel carries ITS variable's value -> correct reorder
        assert torch.allclose(
            ic_x[:, :, :, j],
            torch.full_like(ic_x[:, :, :, j], float(VARIABLES.index(v))),
            atol=1e-4,
        )

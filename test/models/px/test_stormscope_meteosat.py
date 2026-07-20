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

from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px.stormscope_meteosat import VARIABLES, StormScopeMeteosatEU
from earth2studio.utils import handshake_dim


# Spoof diffusion model with the same call signature as an EDMPreconditioner-wrapped
# model: forward(x, sigma, class_labels=None, condition=None) -> denoised x0.
class PhooMeteosatDiffusionModel(torch.nn.Module):
    def __init__(self, nvar=4):
        super().__init__()
        self.sigma_min = 0.0
        self.sigma_max = 88.0
        self.nvar = nvar
        self.call_count = 0

    def forward(self, x, noise, class_labels=None, condition=None):
        self.call_count += 1
        return x[:, : self.nvar]

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def create_spoof_model(
    nvar=4,
    n_inv=2,
    h_pkg=8,
    w_pkg=10,
    inference_mtg_box=None,
    mtg_ylim=None,
    mtg_xlim=None,
    variables=None,
    means=None,
    stds=None,
    scale_factor=None,
    add_offset=None,
    earth_mask=None,
    device="cpu",
    input_times=None,
    output_times=None,
    ir_38_warm_scale_factor=0.024222141,
    ir_38_warm_threshold=4095.0,
    num_diffusion_steps=2,
    sigma_threshold=1.0,
    batch_size=1,
    use_amp=False,
):
    """Create a spoof StormScopeMeteosatEU model for testing"""
    if inference_mtg_box is None:
        inference_mtg_box = ((100, 100 + h_pkg), (200, 200 + w_pkg))
    if mtg_ylim is None:
        mtg_ylim = inference_mtg_box[0]
    if mtg_xlim is None:
        mtg_xlim = inference_mtg_box[1]

    if variables is None:
        variables = np.array(VARIABLES[:nvar])
    nvar = len(variables)

    if input_times is None:
        input_times = np.array([-1, 0]) * np.timedelta64(10, "m")
    if output_times is None:
        output_times = np.array([np.timedelta64(10, "m")])

    lat = torch.linspace(30.0, 55.0, h_pkg).unsqueeze(1).repeat(1, w_pkg)
    lon = torch.linspace(-10.0, 30.0, w_pkg).unsqueeze(0).repeat(h_pkg, 1)
    if earth_mask is None:
        earth_mask = torch.ones(h_pkg, w_pkg, dtype=torch.bool)
    mtg_y = np.arange(*inference_mtg_box[0])
    mtg_x = np.arange(*inference_mtg_box[1])

    if means is None:
        means = torch.zeros(nvar, 1, 1)
    if stds is None:
        stds = torch.ones(nvar, 1, 1)
    if scale_factor is None:
        scale_factor = torch.ones(nvar, 1, 1)
    if add_offset is None:
        add_offset = torch.zeros(nvar, 1, 1)
    invariants = torch.zeros(1, n_inv, h_pkg, w_pkg)

    model_low = PhooMeteosatDiffusionModel(nvar=nvar)
    model_high = PhooMeteosatDiffusionModel(nvar=nvar)

    model = StormScopeMeteosatEU(
        model_low=model_low,
        model_high=model_high,
        means=means,
        stds=stds,
        scale_factor=scale_factor,
        add_offset=add_offset,
        invariants=invariants,
        lat=lat,
        lon=lon,
        earth_mask=earth_mask,
        mtg_y=mtg_y,
        mtg_x=mtg_x,
        mtg_ylim=mtg_ylim,
        mtg_xlim=mtg_xlim,
        inference_mtg_box=inference_mtg_box,
        variables=variables,
        sampler_args={"sigma_min": 0.02, "sigma_max": 10.0},
        input_times=input_times,
        output_times=output_times,
        ir_38_warm_scale_factor=ir_38_warm_scale_factor,
        ir_38_warm_threshold=ir_38_warm_threshold,
        num_diffusion_steps=num_diffusion_steps,
        sigma_threshold=sigma_threshold,
        batch_size=batch_size,
        use_amp=use_amp,
    ).to(device)

    return model


def test_stormscope_meteosat_coords():
    model = create_spoof_model()
    in_coords = model.input_coords()

    assert list(in_coords.keys()) == [
        "batch",
        "time",
        "lead_time",
        "variable",
        "y",
        "x",
    ]
    assert np.array_equal(in_coords["lead_time"], model.input_times)
    assert (in_coords["variable"] == model.variables).all()
    assert len(in_coords["y"]) == len(model.mtg_y)
    assert len(in_coords["x"]) == len(model.mtg_x)


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-06-01T00:00")]),
        np.array(
            [
                np.datetime64("2024-06-01T00:00"),
                np.datetime64("2024-06-01T06:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "batch, model_batch_size",
    [
        (1, 1),
        (2, 1),
        (3, 2),
    ],
)
def test_stormscope_meteosat_call(time, device, batch, model_batch_size):
    """Test basic StormScopeMeteosatEU call functionality"""
    nvar = 4
    model = create_spoof_model(nvar=nvar, batch_size=model_batch_size, device=device)

    dc = OrderedDict([("y", model.mtg_y), ("x", model.mtg_x)])
    r = Random(dc)

    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    x = x.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch)})
    coords.move_to_end("batch", last=False)

    out, out_coords = model(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    h, w = len(model.mtg_y), len(model.mtg_x)
    assert out.shape == torch.Size([batch, len(time), 1, nvar, h, w])
    assert torch.isfinite(out).all()
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    handshake_dim(out_coords, "x", 5)
    handshake_dim(out_coords, "y", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_meteosat_expert_selection(device):
    """_conditioned_x0_predictor routes to model_high at/above sigma_threshold
    and model_low below it, for both single- and multi-sample sigma batches."""
    nvar = 3
    sigma_threshold = 5.0
    model = create_spoof_model(
        nvar=nvar, sigma_threshold=sigma_threshold, device=device
    )

    h, w = model.lat.shape
    x_noisy = torch.randn(2, nvar, h, w, device=device)
    condition = torch.zeros(2, 1, h, w, device=device)
    x0_predictor = model._conditioned_x0_predictor(condition)

    t_low = torch.full((2,), sigma_threshold - 1.0, device=device)
    x0_predictor(x_noisy, t_low)
    assert model.model_low.call_count == 1
    assert model.model_high.call_count == 0

    t_high = torch.full((2,), sigma_threshold + 1.0, device=device)
    x0_predictor(x_noisy, t_high)
    assert model.model_low.call_count == 1
    assert model.model_high.call_count == 1

    # Exactly at the threshold routes to the high-sigma expert (>=)
    t_eq = torch.full((1,), sigma_threshold, device=device)
    x0_predictor(x_noisy[:1], t_eq)
    assert model.model_high.call_count == 2


@pytest.mark.parametrize("use_amp", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_meteosat_amp(use_amp, device):
    """AMP autocast should leave inference behavior intact."""
    nvar = 4
    model = create_spoof_model(nvar=nvar, use_amp=use_amp, device=device)

    time = np.array([np.datetime64("2024-06-01T00:00")])
    dc = OrderedDict([("y", model.mtg_y), ("x", model.mtg_x)])
    r = Random(dc)

    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)
    x = x.unsqueeze(0)
    coords.update({"batch": np.arange(1)})
    coords.move_to_end("batch", last=False)

    out, _ = model(x, coords)

    assert model.use_amp == use_amp
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-06-01T00:00")]),
        np.array(
            [
                np.datetime64("2024-06-01T00:00"),
                np.datetime64("2024-06-01T06:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_meteosat_iter(time, batch, device):
    """Test StormScopeMeteosatEU autoregressive iterator, including the
    per-analysis-time solar-angle window rolling for multiple ``time`` entries."""
    nvar = 4
    model = create_spoof_model(nvar=nvar, device=device)

    dc = OrderedDict([("y", model.mtg_y), ("x", model.mtg_x)])
    r = Random(dc)

    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    x = x.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch)})
    coords.move_to_end("batch", last=False)

    p_iter = model.create_iterator(x, coords)

    # First value from the generator is the unchanged initial condition
    x0, coords0 = next(p_iter)
    assert torch.equal(x0, x)
    assert np.array_equal(coords0["lead_time"], lead_time)

    h, w = len(model.mtg_y), len(model.mtg_x)
    time_step = model.output_times[0]
    for i, (out, out_coords) in enumerate(p_iter):
        assert out.shape == torch.Size([batch, len(time), 1, nvar, h, w])
        assert torch.isfinite(out).all()
        assert (out_coords["batch"] == np.arange(batch)).all()
        assert out_coords["lead_time"][0] == time_step * (i + 1)

        if i > 1:
            break


def test_stormscope_meteosat_default_bbox():
    """mtg_ylim/mtg_xlim/inference_mtg_box default to Model_FCI_BBox when omitted"""
    box = StormScopeMeteosatEU.Model_FCI_BBox
    h_pkg = box[0][1] - box[0][0]
    w_pkg = box[1][1] - box[1][0]

    nvar = 2
    variables = np.array(VARIABLES[:nvar])
    lat = torch.zeros(h_pkg, w_pkg)
    lon = torch.zeros(h_pkg, w_pkg)
    earth_mask = torch.ones(h_pkg, w_pkg, dtype=torch.bool)
    mtg_y = np.arange(*box[0])
    mtg_x = np.arange(*box[1])
    means = torch.zeros(nvar, 1, 1)
    stds = torch.ones(nvar, 1, 1)
    scale_factor = torch.ones(nvar, 1, 1)
    add_offset = torch.zeros(nvar, 1, 1)
    invariants = torch.zeros(1, 1, h_pkg, w_pkg)

    # mtg_ylim, mtg_xlim, and inference_mtg_box are all intentionally omitted
    # here to exercise the Model_FCI_BBox class-level defaults.
    model = StormScopeMeteosatEU(
        model_low=PhooMeteosatDiffusionModel(nvar=nvar),
        model_high=PhooMeteosatDiffusionModel(nvar=nvar),
        means=means,
        stds=stds,
        scale_factor=scale_factor,
        add_offset=add_offset,
        invariants=invariants,
        lat=lat,
        lon=lon,
        earth_mask=earth_mask,
        mtg_y=mtg_y,
        mtg_x=mtg_x,
        variables=variables,
    )

    assert model.mtg_ylim == box[0]
    assert model.mtg_xlim == box[1]
    assert model.lat.shape == (h_pkg, w_pkg)
    assert np.array_equal(model.mtg_y, mtg_y)
    assert np.array_equal(model.mtg_x, mtg_x)


def test_stormscope_meteosat_sub_region():
    """mtg_ylim/mtg_xlim select a sub-region of the inference_mtg_box package extent"""
    h_pkg, w_pkg = 20, 30
    y0, x0 = 500, 1000
    inference_mtg_box = ((y0, y0 + h_pkg), (x0, x0 + w_pkg))

    # Full package extent, no cropping
    model_full = create_spoof_model(
        h_pkg=h_pkg, w_pkg=w_pkg, inference_mtg_box=inference_mtg_box
    )
    assert model_full.lat.shape == (h_pkg, w_pkg)
    assert np.array_equal(model_full.mtg_y, np.arange(*inference_mtg_box[0]))
    assert np.array_equal(model_full.mtg_x, np.arange(*inference_mtg_box[1]))

    # Cropped sub-region, distinct from inference_mtg_box
    mtg_ylim = (y0 + 5, y0 + 15)
    mtg_xlim = (x0 + 10, x0 + 25)
    model_sub = create_spoof_model(
        h_pkg=h_pkg,
        w_pkg=w_pkg,
        inference_mtg_box=inference_mtg_box,
        mtg_ylim=mtg_ylim,
        mtg_xlim=mtg_xlim,
    )

    assert model_sub.mtg_ylim == mtg_ylim
    assert model_sub.mtg_xlim == mtg_xlim
    assert np.array_equal(model_sub.mtg_y, np.arange(*mtg_ylim))
    assert np.array_equal(model_sub.mtg_x, np.arange(*mtg_xlim))

    expected_shape = (mtg_ylim[1] - mtg_ylim[0], mtg_xlim[1] - mtg_xlim[0])
    assert model_sub.lat.shape == expected_shape
    assert model_sub.lon.shape == expected_shape
    assert model_sub.earth_mask.shape == expected_shape

    # Sub-region grid/lat/lon should be an exact slice of the full package arrays
    yi0, yi1 = mtg_ylim[0] - y0, mtg_ylim[1] - y0
    xi0, xi1 = mtg_xlim[0] - x0, mtg_xlim[1] - x0
    assert torch.equal(model_sub.lat, model_full.lat[yi0:yi1, xi0:xi1])
    assert torch.equal(model_sub.lon, model_full.lon[yi0:yi1, xi0:xi1])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_meteosat_raw_conversions(device):
    """raw_to_physical / physical_to_raw and raw_to_normalized / normalized_to_raw round trip."""
    nvar = len(VARIABLES)
    means = torch.linspace(0.0, 30.0, nvar).reshape(nvar, 1, 1)
    stds = torch.linspace(1.0, 4.0, nvar).reshape(nvar, 1, 1)
    scale_factor = torch.linspace(0.005, 0.03, nvar).reshape(nvar, 1, 1)
    add_offset = torch.linspace(-1.0, 1.0, nvar).reshape(nvar, 1, 1)

    model = create_spoof_model(
        nvar=nvar,
        means=means,
        stds=stds,
        scale_factor=scale_factor,
        add_offset=add_offset,
        device=device,
    )

    h, w = model.lat.shape
    x_raw = torch.rand(2, 1, 1, nvar, h, w, device=device) * 4000
    x_raw[0, 0, 0, VARIABLES.index("fci38ir")] *= 2  # to trigger HDR scaling

    x_phys = model.raw_to_physical(x_raw.clone())
    x_raw_back = model.physical_to_raw(x_phys.clone())
    assert torch.allclose(x_raw_back, x_raw, atol=1e-4)

    x_norm = model.raw_to_normalized(x_raw.clone())
    x_raw_back2 = model.normalized_to_raw(x_norm.clone())
    assert torch.allclose(x_raw_back2, x_raw, atol=1e-4)

    x_n = model.normalize(x_phys.clone())
    x_dn = model.denormalize(x_n.clone())
    assert torch.allclose(x_dn, x_phys, atol=1e-3)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_meteosat_off_earth_mask(device):
    """Off-earth pixels are zero-filled after normalize and NaN after denormalize"""
    nvar = 3
    h_pkg, w_pkg = 6, 8
    earth_mask = torch.ones(h_pkg, w_pkg, dtype=torch.bool)
    earth_mask[:2, :2] = False

    model = create_spoof_model(
        nvar=nvar, h_pkg=h_pkg, w_pkg=w_pkg, earth_mask=earth_mask, device=device
    )

    x = torch.rand(1, 1, 1, nvar, h_pkg, w_pkg, device=device) + 1.0
    x_norm = model.normalize(x.clone())
    assert torch.all(x_norm[..., :2, :2] == 0)
    assert torch.all(x_norm[..., 2:, 2:] != 0)

    x_denorm = model.denormalize(x_norm.clone())
    assert torch.all(torch.isnan(x_denorm[..., :2, :2]))
    assert torch.all(torch.isfinite(x_denorm[..., 2:, 2:]))


def test_stormscope_meteosat_ir38_warm_scaling():
    """The ir_38 channel applies extra scaling above ir_38_warm_threshold"""
    variables = np.array(["fci04vis", "fci38ir", "fci105ir"])
    nvar = len(variables)
    scale_factor = torch.full((nvar, 1, 1), 0.01)
    add_offset = torch.zeros(nvar, 1, 1)
    threshold = 0.5
    warm_scale = 2.0

    model = create_spoof_model(
        variables=variables,
        scale_factor=scale_factor,
        add_offset=add_offset,
        ir_38_warm_scale_factor=warm_scale,
        ir_38_warm_threshold=threshold,
    )

    assert model.ir_38_channel == 1

    h, w = model.lat.shape
    x = torch.zeros(1, 1, 1, nvar, h, w)
    below_val, above_val = 0.3, 0.8
    x[..., 1, 0, 0] = below_val
    x[..., 1, 0, 1] = above_val

    x_phys = model.raw_to_physical(x.clone())

    below = x_phys[0, 0, 0, 1, 0, 0]
    above = x_phys[0, 0, 0, 1, 0, 1]

    expected_below = below_val * 0.01
    expected_above = (above_val - threshold) * warm_scale + threshold * 0.01
    assert torch.allclose(below, torch.tensor(expected_below))
    assert torch.allclose(above, torch.tensor(expected_above), atol=1e-6)

    # round trip back to raw values
    x_raw_back = model.physical_to_raw(x_phys.clone())
    assert torch.allclose(x_raw_back[..., 1, :, :], x[..., 1, :, :], atol=1e-4)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_meteosat_azimuth_zenith(device):
    """_azimuth_zenith returns finite solar angles, masked to zero off-earth"""
    h_pkg, w_pkg = 6, 8
    earth_mask = torch.ones(h_pkg, w_pkg, dtype=torch.bool)
    earth_mask[0, 0] = False

    model = create_spoof_model(
        h_pkg=h_pkg, w_pkg=w_pkg, earth_mask=earth_mask, device=device
    )

    times = [
        np.datetime64("2024-06-01T00:00"),
        np.datetime64("2024-06-01T00:10"),
    ]
    zen_azi = model._azimuth_zenith(times)

    assert zen_azi.shape == torch.Size([3, len(times), h_pkg, w_pkg])
    assert torch.isfinite(zen_azi).all()
    assert torch.all(zen_azi[:, :, ~earth_mask] == 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_meteosat_compile(device):
    """compile_model wraps the diffusion sub-models with torch.compile"""
    model = create_spoof_model(device=device)
    orig_low_forward = model.model_low.forward
    orig_high_forward = model.model_high.forward

    model.compile_model()

    assert model.model_low.forward is not orig_low_forward
    assert model.model_high.forward is not orig_high_forward
    assert callable(model.model_low.forward)
    assert callable(model.model_high.forward)


def test_stormscope_meteosat_exceptions():
    """output_coords should reject mismatched variable order or grid size"""
    nvar = 3
    model = create_spoof_model(nvar=nvar)

    in_coords = model.input_coords()
    time = np.array([np.datetime64("2024-06-01T00:00")])

    bad_variable_order = OrderedDict(
        {
            "batch": np.arange(1),
            "time": time,
            "lead_time": in_coords["lead_time"],
            "variable": in_coords["variable"][::-1].copy(),
            "y": model.mtg_y,
            "x": model.mtg_x,
        }
    )
    with pytest.raises(ValueError):
        model.output_coords(bad_variable_order)

    bad_grid_size = OrderedDict(
        {
            "batch": np.arange(1),
            "time": time,
            "lead_time": in_coords["lead_time"],
            "variable": in_coords["variable"],
            "y": model.mtg_y[:-1],
            "x": model.mtg_x,
        }
    )
    with pytest.raises(ValueError):
        model.output_coords(bad_grid_size)

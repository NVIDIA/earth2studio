# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the CorrDiffEra5Hrrr diagnostic wrapper.

Construction-based with a stand-in network on a small synthetic grid: they
exercise the coordinate contracts, the conditioning assembly (interpolation,
cosine zenith, invariants, scalar conditions) and the three sampling paths
(x-prediction / velocity rectified flow, EDM) end to end on the CPU.
"""

from collections import OrderedDict
from datetime import datetime

import numpy as np
import pytest
import torch

pytest.importorskip("physicsnemo")
pytest.importorskip("natten")

from earth2studio.models.dx.corrdiff_era5_hrrr import (  # noqa: E402
    CorrDiffEra5Hrrr,
)

ERA5_VARIABLES = ["u10m", "v10m", "t2m", "z500"]
OUTPUT_VARIABLES = ["u10m", "v10m", "t2m", "refc", "q1hl"]
N_INV = 3
H, W = 8, 12


class PhooNet(torch.nn.Module):
    """Stand-in for ConcatConditionWrapper(DiT): returns a scaled slice of x.

    Records the number of conditioning channels it received so the tests can
    check the assembled background.
    """

    def __init__(self, n_out: int, gain: float = 0.0):
        super().__init__()
        self.n_out = n_out
        self.gain = gain
        self.seen_cond_channels: int | None = None
        self.seen_cond_vec: int | None = None
        self.seen_t: list[torch.Tensor] = []

    def forward(self, x, t, condition=None):
        self.seen_cond_channels = condition["cond_concat"].shape[1]
        self.seen_cond_vec = condition["cond_vec"].shape[1]
        self.seen_t.append(t.detach().clone())
        return self.gain * x[:, : self.n_out]


def _build(kind="rectified_flow", prediction_type="x0", **overrides):
    lat_in = np.arange(30.0, 26.0, -0.5, dtype=np.float32)  # 8, descending
    lon_in = np.arange(260.0, 265.0, 0.5, dtype=np.float32)  # 10
    # curvilinear-ish output grid strictly inside the input footprint
    lat2d, lon2d = np.meshgrid(
        np.linspace(27.0, 29.0, H), np.linspace(260.5, 264.0, W), indexing="ij"
    )
    lat2d = lat2d + 0.05 * np.sin(lon2d)
    net = PhooNet(len(OUTPUT_VARIABLES), gain=overrides.pop("gain", 0.0))
    kwargs = dict(
        network=net,
        network_kind=kind,
        era5_variables=ERA5_VARIABLES,
        output_variables=OUTPUT_VARIABLES,
        lat_input_grid=torch.tensor(lat_in),
        lon_input_grid=torch.tensor(lon_in),
        lat_output_grid=torch.tensor(lat2d, dtype=torch.float32),
        lon_output_grid=torch.tensor(lon2d, dtype=torch.float32),
        hrrr_y=torch.arange(H, dtype=torch.float64) * 3000.0,
        hrrr_x=torch.arange(W, dtype=torch.float64) * 3000.0,
        era5_center=torch.zeros(len(ERA5_VARIABLES)),
        era5_scale=torch.ones(len(ERA5_VARIABLES)),
        out_center=torch.tensor([0.0, 0.0, 280.0, -5.0, 0.0]),
        out_scale=torch.tensor([3.0, 3.0, 10.0, 8.0, 1.0]),
        invariants=torch.randn(N_INV, H, W),
        presence_flags=["tcwv", "sp"],
        day_of_year=True,
        prediction_type=prediction_type,
        number_of_samples=2,
        number_of_steps=3,
        shift=4.0,
        seed=0,
        amp=False,
    )
    kwargs.update(overrides)
    return CorrDiffEra5Hrrr(**kwargs), net


def _input(model, n_time=1):
    ic = model.input_coords()
    x = torch.randn(1, n_time, len(ERA5_VARIABLES), ic["lat"].size, ic["lon"].size)
    coords = OrderedDict(
        {
            "batch": np.array([0]),
            "time": np.array(
                [
                    np.datetime64("2025-10-24T00:00") + np.timedelta64(6 * i, "h")
                    for i in range(n_time)
                ]
            ),
            "variable": np.array(ERA5_VARIABLES),
            "lat": ic["lat"],
            "lon": ic["lon"],
        }
    )
    return x, coords


def test_coords_contract():
    model, _ = _build()
    ic = model.input_coords()
    assert list(ic) == ["batch", "time", "variable", "lat", "lon"]
    _, coords = _input(model)
    oc = model.output_coords(coords)
    assert list(oc) == ["batch", "sample", "time", "variable", "hrrr_y", "hrrr_x"]
    assert oc["sample"].size == 2 and oc["hrrr_y"].size == H and oc["hrrr_x"].size == W
    assert list(oc["variable"]) == OUTPUT_VARIABLES


def test_output_coords_rejects_wrong_grid():
    model, _ = _build()
    _, coords = _input(model)
    coords["lat"] = coords["lat"] + 1.0
    with pytest.raises(ValueError):
        model.output_coords(coords)


def test_preprocess_assembles_conditioning():
    model, _ = _build()
    era5 = torch.randn(
        len(ERA5_VARIABLES), model.lat_input_numpy.size, model.lon_input_numpy.size
    )
    cond = model.preprocess_input(era5, datetime(2025, 7, 1, 18))
    # ERA5 channels + cos zenith + invariants
    assert cond["cond_concat"].shape == (1, len(ERA5_VARIABLES) + 1 + N_INV, H, W)
    # sin/cos day-of-year + two presence flags (always 1)
    assert cond["cond_vec"].shape == (1, 4)
    assert torch.all(cond["cond_vec"][0, 2:] == 1.0)
    cz = cond["cond_concat"][0, len(ERA5_VARIABLES)]
    assert torch.all(cz.abs() <= 1.0)
    assert torch.isfinite(cond["cond_concat"]).all()


@pytest.mark.parametrize(
    "kind,prediction_type",
    [("rectified_flow", "x0"), ("rectified_flow", "flow"), ("edm", "x0")],
)
def test_call_shapes_and_seeding(kind, prediction_type):
    if kind == "edm":
        pytest.importorskip("physicsnemo.diffusion.preconditioners")
    model, net = _build(kind=kind, prediction_type=prediction_type)
    if kind == "edm":
        from physicsnemo.diffusion.preconditioners import EDMPreconditioner

        model.network = EDMPreconditioner(net, sigma_data=0.5)
    x, coords = _input(model, n_time=2)
    out, oc = model(x, coords)
    assert out.shape == (1, 2, 2, len(OUTPUT_VARIABLES), H, W)
    assert torch.isfinite(out).all()
    assert list(oc) == ["batch", "sample", "time", "variable", "hrrr_y", "hrrr_x"]
    # deterministic given the seed; members differ from each other
    out2, _ = model(x, coords)
    assert torch.allclose(out, out2)
    assert not torch.allclose(out[:, 0], out[:, 1])
    if kind == "rectified_flow":
        # the network sees the time scaled for its embedder (t in [0, 1] x 999)
        assert max(float(t.max()) for t in net.seen_t) <= 999.0 * 0.999 + 1e-3
        assert max(float(t.max()) for t in net.seen_t) > 1.0


def test_rf_time_grid_shift():
    model, _ = _build(shift=1.0)
    from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler

    sch = RectifiedFlowNoiseScheduler(t_max=0.99)
    t_unshifted = model._rf_time_steps(torch.device("cpu"), sch)
    model.shift = 16.0
    t_shifted = model._rf_time_steps(torch.device("cpu"), sch)
    assert t_shifted.shape == t_unshifted.shape
    assert float(t_shifted[-1]) == 0.0
    # the shift pushes interior steps toward the noise end
    assert torch.all(t_shifted[1:-1] > t_unshifted[1:-1])


def test_constructor_validation():
    with pytest.raises(ValueError):
        _build(kind="ddpm")
    with pytest.raises(ValueError):
        _build(prediction_type="epsilon")
    with pytest.raises(ValueError):
        _build(number_of_samples=0)
    with pytest.raises(ValueError):
        _build(t_max=1.0)

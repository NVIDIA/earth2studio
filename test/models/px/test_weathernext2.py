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
from unittest import mock

import numpy as np
import pytest
import torch

try:
    from weathernext.weathernext2 import fgn
except ImportError:
    pytest.importorskip("weathernext")

from earth2studio.data import Random, fetch_data
from earth2studio.models.px.weathernext2_cyclones_mini import (
    OUTPUT_VARIABLES,
    WeatherNext2CyclonesMini,
)
from earth2studio.utils import handshake_dim


def mocked_chunked_prediction(
    predictor_fn,
    rng,
    inputs,
    targets_template,
    forcings,
    num_steps_per_chunk=None,
    verbose=None,
):
    return targets_template


def mocked_chunked_prediction_generator(
    self,
    predictor_fn,
    rng,
    inputs,
    targets_template,
    batch,
    forcings,
):
    yield targets_template.isel(time=[0])
    while True:
        yield targets_template.isel(time=[0])


@pytest.fixture
def mock_weathernext2_model():
    ckpt = fgn.CheckPoint(params={}, description="mock", license="license")
    with mock.patch.object(
        WeatherNext2CyclonesMini, "_load_run_forward_from_checkpoint", return_value=None
    ):
        return WeatherNext2CyclonesMini(
            ckpt,
            land_sea_mask=np.ones((9, 12), dtype=np.float32),
            geopotential_at_surface=np.ones((9, 12), dtype=np.float32),
            jit_compile=False,
        )


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2025-01-01T00:00")]),
        np.array(
            [np.datetime64("2025-01-01T00:00"), np.datetime64("2025-01-02T00:00")]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@mock.patch("weathernext.utils.rollout.chunked_prediction", mocked_chunked_prediction)
def test_weathernext2_call(time, device, mock_weathernext2_model):
    p = mock_weathernext2_model.to(device)
    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    x, coords = fetch_data(
        r,
        time,
        p.input_coords()["variable"],
        p.input_coords()["lead_time"],
        device=device,
    )
    out, out_coords = p(x, coords)

    assert out.shape == torch.Size([len(time), 1, len(OUTPUT_VARIABLES), 9, 12])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@mock.patch.object(
    WeatherNext2CyclonesMini,
    "_chunked_prediction_generator",
    mocked_chunked_prediction_generator,
)
def test_weathernext2_iter(device, mock_weathernext2_model):
    time = np.array([np.datetime64("2025-01-01T00:00")])
    p = mock_weathernext2_model.to(device)
    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    x, coords = fetch_data(
        r,
        time,
        p.input_coords()["variable"],
        p.input_coords()["lead_time"],
        device=device,
    )
    p_iter = p.create_iterator(x, coords)

    input, input_coords = next(p_iter)
    assert input_coords["lead_time"] == np.timedelta64(0, "h")
    assert input.shape == torch.Size([1, 1, len(p.input_coords()["variable"]), 9, 12])

    for i, (out, out_coords) in enumerate(p_iter):
        assert out.shape == torch.Size([1, 1, len(OUTPUT_VARIABLES), 9, 12])
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"] == np.timedelta64(6 * (i + 1), "h")
        if i > 5:
            break


@mock.patch("weathernext.utils.rollout.chunked_prediction")
def test_weathernext2_rng_advances(chunked_prediction, mock_weathernext2_model):
    rngs = []

    def _mock_prediction(predictor_fn, rng, inputs, targets_template, forcings):
        rngs.append(np.asarray(rng))
        return targets_template

    chunked_prediction.side_effect = _mock_prediction
    time = np.array([np.datetime64("2025-01-01T00:00")])
    p = mock_weathernext2_model
    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)

    x, coords = fetch_data(
        r,
        time,
        p.input_coords()["variable"],
        p.input_coords()["lead_time"],
        device="cpu",
    )
    p(x, coords)
    p(x, coords)

    assert len(rngs) == 2
    assert not np.array_equal(rngs[0], rngs[1])


def test_weathernext2_cyclone_tracks_inactive(mock_weathernext2_model):
    with mock.patch(
        "earth2studio.models.px.weathernext2_cyclones_mini.logger.warning"
    ) as warning:
        tracks = mock_weathernext2_model.cyclone_tracks

    assert tracks.empty
    warning.assert_called_once_with(
        "Cyclone tracking is currently not active on this model."
    )


@mock.patch("weathernext.utils.rollout.chunked_prediction", mocked_chunked_prediction)
def test_weathernext2_call_updates_cyclone_tracks(mock_weathernext2_model):
    mock_weathernext2_model.track_cyclones = True
    time = np.array([np.datetime64("2025-01-01T00:00")])
    dc = mock_weathernext2_model.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    r = Random(dc)
    x, coords = fetch_data(
        r,
        time,
        mock_weathernext2_model.input_coords()["variable"],
        mock_weathernext2_model.input_coords()["lead_time"],
        device="cpu",
    )

    with mock.patch.object(mock_weathernext2_model, "_reset_cyclone_tracks") as reset:
        with mock.patch.object(
            mock_weathernext2_model, "_update_cyclone_tracks"
        ) as update:
            out, out_coords = mock_weathernext2_model(x, coords)

    assert out.shape == torch.Size([1, 1, len(OUTPUT_VARIABLES), 9, 12])
    assert (
        out_coords["variable"]
        == mock_weathernext2_model.output_coords(coords)["variable"]
    ).all()
    reset.assert_called_once_with()
    update.assert_called_once()


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(9)}),
        OrderedDict({"lat": np.random.randn(9), "phoo": np.random.randn(12)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weathernext2_exceptions(dc, device, mock_weathernext2_model):
    time = np.array([np.datetime64("2025-01-01T00:00")])
    p = mock_weathernext2_model.to(device)
    r = Random(dc)

    x, coords = fetch_data(
        r,
        time,
        p.input_coords()["variable"],
        p.input_coords()["lead_time"],
        device=device,
    )
    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.mark.package
def test_weathernext2_package():
    package = WeatherNext2CyclonesMini.load_default_package()
    model = WeatherNext2CyclonesMini.load_model(package, jit_compile=False)
    assert model.input_coords()["lat"].shape == (181,)
    assert model.input_coords()["lon"].shape == (360,)

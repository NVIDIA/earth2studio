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
import pandas as pd
import pytest
import torch
import xarray as xr

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

TEST_TIME = np.array([np.datetime64("2025-01-01T00:00")])
DEVICES = ["cpu", "cuda:0"]


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


def fetch_random_input(model, time=TEST_TIME, device="cpu"):
    input_coords = model.input_coords()
    random_coords = input_coords.copy()
    for dim in ("batch", "time", "lead_time", "variable"):
        del random_coords[dim]
    return fetch_data(
        Random(random_coords),
        time,
        input_coords["variable"],
        input_coords["lead_time"],
        device=device,
    )


def assert_output(model, out, out_coords, coords, time):
    assert out.shape == torch.Size([len(time), 1, len(OUTPUT_VARIABLES), 9, 12])
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    for dim, index in (
        ("lon", 4),
        ("lat", 3),
        ("variable", 2),
        ("lead_time", 1),
        ("time", 0),
    ):
        handshake_dim(out_coords, dim, index)


@pytest.mark.parametrize(
    "time",
    [
        TEST_TIME,
        np.array(
            [np.datetime64("2025-01-01T00:00"), np.datetime64("2025-01-02T00:00")]
        ),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
@mock.patch("weathernext.utils.rollout.chunked_prediction", mocked_chunked_prediction)
def test_weathernext2_call(time, device, mock_weathernext2_model):
    model = mock_weathernext2_model.to(device)
    x, coords = fetch_random_input(model, time, device)
    out, out_coords = model(x, coords)
    assert_output(model, out, out_coords, coords, time)


@pytest.mark.parametrize("device", DEVICES)
@mock.patch.object(
    WeatherNext2CyclonesMini,
    "_chunked_prediction_generator",
    mocked_chunked_prediction_generator,
)
def test_weathernext2_iter(device, mock_weathernext2_model):
    model = mock_weathernext2_model.to(device)
    x, coords = fetch_random_input(model, device=device)
    model_iter = model.create_iterator(x, coords)

    out, out_coords = next(model_iter)
    assert out_coords["lead_time"] == np.timedelta64(0, "h")
    assert out.shape == torch.Size([1, 1, len(model.input_coords()["variable"]), 9, 12])

    for i in range(7):
        out, out_coords = next(model_iter)
        assert_output(model, out, out_coords, coords, TEST_TIME)
        assert out_coords["lead_time"] == np.timedelta64(6 * (i + 1), "h")


@mock.patch("weathernext.utils.rollout.chunked_prediction")
def test_weathernext2_rng_advances(chunked_prediction, mock_weathernext2_model):
    rngs = []

    def mock_prediction(predictor_fn, rng, inputs, targets_template, forcings):
        rngs.append(np.asarray(rng))
        return targets_template

    chunked_prediction.side_effect = mock_prediction
    x, coords = fetch_random_input(mock_weathernext2_model)
    mock_weathernext2_model(x, coords)
    mock_weathernext2_model(x, coords)

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


def test_weathernext2_cyclone_tracks_have_e2s_observation_names(
    mock_weathernext2_model,
):
    mock_weathernext2_model.track_cyclones = True
    mock_weathernext2_model._cyclone_tracker = mock.Mock(
        return_value=pd.DataFrame(
            {
                "track_id": ["storm-0"],
                "lead_time": [pd.Timedelta(hours=6)],
                "valid_time": [pd.Timestamp("2025-01-01T06:00")],
                "lat": [10.0],
                "lon": [20.0],
                "minimum_sea_level_pressure_hpa": [990.0],
                "maximum_sustained_wind_speed_knots": [20.0],
            }
        )
    )

    mock_weathernext2_model._update_cyclone_tracks(
        xr.Dataset(
            {
                "cyclone_probability": xr.DataArray(
                    np.ones((1, 1, 1)), dims=("time", "lat", "lon")
                )
            }
        ),
        OrderedDict(
            {
                "time": TEST_TIME,
                "lead_time": np.array([np.timedelta64(6, "h")]),
            }
        ),
        accumulate_predictions=False,
    )

    tracks = mock_weathernext2_model.cyclone_tracks
    assert set(["lat", "lon", "tcmsl", "tcw10m"]).issubset(tracks.columns)
    np.testing.assert_allclose(tracks[["lat", "lon", "tcmsl"]], [[10.0, 20.0, 99000.0]])
    np.testing.assert_allclose(tracks["tcw10m"], [10.28888])


@mock.patch("weathernext.utils.rollout.chunked_prediction", mocked_chunked_prediction)
def test_weathernext2_call_updates_cyclone_tracks(mock_weathernext2_model):
    mock_weathernext2_model.track_cyclones = True
    x, coords = fetch_random_input(mock_weathernext2_model)

    with mock.patch.object(mock_weathernext2_model, "_reset_cyclone_tracks") as reset:
        with mock.patch.object(
            mock_weathernext2_model, "_update_cyclone_tracks"
        ) as update:
            out, out_coords = mock_weathernext2_model(x, coords)

    assert_output(mock_weathernext2_model, out, out_coords, coords, TEST_TIME)
    reset.assert_called_once_with()
    update.assert_called_once()


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(9)}),
        OrderedDict({"lat": np.random.randn(9), "phoo": np.random.randn(12)}),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_weathernext2_exceptions(dc, device, mock_weathernext2_model):
    model = mock_weathernext2_model.to(device)
    x, coords = fetch_data(
        Random(dc),
        TEST_TIME,
        model.input_coords()["variable"],
        model.input_coords()["lead_time"],
        device=device,
    )
    with pytest.raises((KeyError, ValueError)):
        model(x, coords)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_weathernext2_package(device):
    torch.cuda.empty_cache()
    model = WeatherNext2CyclonesMini.load_model(
        WeatherNext2CyclonesMini.load_default_package(), jit_compile=False
    ).to(device)

    assert model.input_coords()["lat"].shape == (181,)
    assert model.input_coords()["lon"].shape == (360,)
    assert model.output_coords(model.input_coords())["variable"].shape == (84,)

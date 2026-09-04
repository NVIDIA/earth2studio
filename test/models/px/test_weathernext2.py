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

try:
    from weathernext.weathernext2 import fgn
except ImportError:
    pytest.importorskip("weathernext")

from earth2studio.data import Random, fetch_data
from earth2studio.models.px.weathernext2_cyclones_mini import (
    OUTPUT_VARIABLES,
    WeatherNext2CyclonesMini,
    _add_e2s_cyclone_columns,
)

TEST_TIME = np.array([np.datetime64("2025-01-01T00:00")])


def mocked_chunked_prediction(*args, targets_template, **kwargs):
    return targets_template


def mocked_chunked_prediction_generator(self, *args, targets_template, **kwargs):
    while True:
        yield targets_template.isel(time=[0])


@pytest.fixture
def mock_weathernext2_model():
    grid = np.ones((9, 12), dtype=np.float32)
    ckpt = fgn.CheckPoint(params={}, description="mock", license="license")
    with mock.patch.object(
        WeatherNext2CyclonesMini, "_load_run_forward_from_checkpoint", return_value=None
    ):
        return WeatherNext2CyclonesMini(ckpt, grid, grid, jit_compile=False)


def fetch_random_input(model, time=TEST_TIME, device="cpu"):
    coords = model.input_coords()
    spatial = OrderedDict((dim, coords[dim]) for dim in ("lat", "lon"))
    return fetch_data(
        Random(spatial), time, coords["variable"], coords["lead_time"], device=device
    )


def assert_output(out, coords, time=TEST_TIME):
    assert out.shape == (len(time), 1, len(OUTPUT_VARIABLES), 9, 12)
    assert list(coords) == ["time", "lead_time", "variable", "lat", "lon"]
    assert np.array_equal(coords["variable"], OUTPUT_VARIABLES)
    assert np.array_equal(coords["time"], time)


@pytest.mark.parametrize(
    "time,device",
    [
        (TEST_TIME, "cpu"),
        (
            np.array(
                [
                    np.datetime64("2025-01-01T00:00"),
                    np.datetime64("2025-01-02T00:00"),
                ]
            ),
            "cuda:0",
        ),
    ],
)
@mock.patch("weathernext.utils.rollout.chunked_prediction", mocked_chunked_prediction)
def test_weathernext2_call(time, device, mock_weathernext2_model):
    model = mock_weathernext2_model.to(device)
    x, coords = fetch_random_input(model, time, device)
    assert_output(*model(x, coords), time)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@mock.patch.object(
    WeatherNext2CyclonesMini,
    "_chunked_prediction_generator",
    mocked_chunked_prediction_generator,
)
def test_weathernext2_iter(device, mock_weathernext2_model):
    model = mock_weathernext2_model.to(device)
    x, coords = fetch_random_input(model, device=device)
    iterator = model.create_iterator(x, coords)

    out, out_coords = next(iterator)
    assert_output(out, out_coords)
    assert out_coords["lead_time"] == np.timedelta64(0, "h")
    tp06 = OUTPUT_VARIABLES.index("tp06")
    assert torch.count_nonzero(out[:, :, tp06]) == 0
    assert torch.equal(
        torch.cat((out[:, :, :tp06], out[:, :, tp06 + 1 :]), dim=2), x[:, 1:]
    )

    out, out_coords = next(iterator)
    assert_output(out, out_coords)
    assert out_coords["lead_time"] == np.timedelta64(6, "h")


@mock.patch("weathernext.utils.rollout.chunked_prediction")
def test_weathernext2_rng_advances(prediction, mock_weathernext2_model):
    rngs = []

    def record_rng(*args, rng, targets_template, **kwargs):
        rngs.append(np.asarray(rng))
        return targets_template

    prediction.side_effect = record_rng
    x, coords = fetch_random_input(mock_weathernext2_model)
    mock_weathernext2_model(x, coords)
    mock_weathernext2_model(x, coords)
    assert len(rngs) == 2 and not np.array_equal(*rngs)


def test_weathernext2_cyclone_tracks_inactive(mock_weathernext2_model):
    with mock.patch(
        "earth2studio.models.px.weathernext2_cyclones_mini.logger.warning"
    ) as warning:
        assert mock_weathernext2_model.cyclone_tracks.empty
    warning.assert_called_once()


def test_weathernext2_cyclone_track_aliases():
    tracks = _add_e2s_cyclone_columns(
        pd.DataFrame(
            {
                "minimum_sea_level_pressure_hpa": [990.0],
                "maximum_sustained_wind_speed_knots": [20.0],
            }
        )
    )
    np.testing.assert_allclose(tracks[["tcmsl", "tcw10m"]], [[99000.0, 10.28888]])


@mock.patch("weathernext.utils.rollout.chunked_prediction", mocked_chunked_prediction)
def test_weathernext2_call_updates_cyclone_tracks(mock_weathernext2_model):
    model = mock_weathernext2_model
    model.track_cyclones = True
    x, coords = fetch_random_input(model)
    with (
        mock.patch.object(model, "_reset_cyclone_tracks") as reset,
        mock.patch.object(model, "_update_cyclone_tracks") as update,
    ):
        model(x, coords)
    reset.assert_called_once_with()
    update.assert_called_once()


@pytest.mark.parametrize(
    "coords,device",
    [
        (OrderedDict(lat=np.random.randn(9)), "cpu"),
        (OrderedDict(lat=np.random.randn(9), phoo=np.random.randn(12)), "cuda:0"),
    ],
)
def test_weathernext2_exceptions(coords, device, mock_weathernext2_model):
    model = mock_weathernext2_model.to(device)
    x, coords = fetch_data(
        Random(coords),
        TEST_TIME,
        model.input_coords()["variable"],
        model.input_coords()["lead_time"],
        device=device,
    )
    with pytest.raises((KeyError, ValueError)):
        model(x, coords)


@pytest.mark.package
def test_weathernext2_package():
    torch.cuda.empty_cache()
    model = WeatherNext2CyclonesMini.load_model(
        WeatherNext2CyclonesMini.load_default_package(), jit_compile=False
    ).to("cuda:0")
    assert (
        len(model.input_coords()["lat"]),
        len(model.input_coords()["lon"]),
        len(model.output_coords(model.input_coords())["variable"]),
    ) == (181, 360, 84)

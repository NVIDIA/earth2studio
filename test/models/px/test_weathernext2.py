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
    from weathernext.utils import fiddle_config_io
    from weathernext.weathernext2 import fgn
except ImportError:
    pytest.importorskip("weathernext")

from earth2studio.data import Random, fetch_data
from earth2studio.models.px.weathernext2_cyclones import (
    OUTPUT_VARIABLES,
    WeatherNext2Cyclones,
    WeatherNext2CyclonesMini,
    _add_e2s_cyclone_columns,
    _add_tisr_batched,
)

TEST_TIME = np.array([np.datetime64("2025-01-01T00:00")])


def mocked_chunked_prediction(*args, targets_template, **kwargs):
    return targets_template


def mocked_chunked_prediction_generator(self, *args, targets_template, batch, **kwargs):
    value = float(batch["2m_temperature"].isel(time=-1).mean())
    while True:
        yield targets_template.isel(time=[0]).fillna(value)


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


@mock.patch.object(
    WeatherNext2CyclonesMini,
    "_chunked_prediction_generator",
    mocked_chunked_prediction_generator,
)
def test_weathernext2_concurrent_iterators(mock_weathernext2_model):
    x, coords = fetch_random_input(mock_weathernext2_model)
    first = mock_weathernext2_model.create_iterator(x, coords)
    second = mock_weathernext2_model.create_iterator(x + 1, coords)
    next(first), next(second)
    assert not torch.equal(next(first)[0], next(second)[0])


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


def test_weathernext2_set_rng(mock_weathernext2_model):
    mock_weathernext2_model.set_rng(123)
    key = np.asarray(mock_weathernext2_model.prng_key)
    mock_weathernext2_model.set_rng(456, reset=False)
    np.testing.assert_array_equal(key, mock_weathernext2_model.prng_key)
    mock_weathernext2_model.set_rng(456)
    assert not np.array_equal(key, mock_weathernext2_model.prng_key)
    mock_weathernext2_model.set_rng(123)
    np.testing.assert_array_equal(key, mock_weathernext2_model.prng_key)


def test_weathernext2_target_order(mock_weathernext2_model):
    targets = fiddle_config_io.get_fiddle_config_by_name(
        "weathernext2/configs/WeatherNextCyclones_Mini"
    ).task.target_variables
    expected = tuple(
        variable for variable in targets if not variable.startswith("cyclone")
    )
    assert mock_weathernext2_model.task_config.target_variables == expected


def test_weathernext2_cyclone_tracks_inactive(mock_weathernext2_model):
    with mock.patch(
        "earth2studio.models.px.weathernext2_cyclones.logger.warning"
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


def test_weathernext2_operational_checkpoint():
    assert WeatherNext2Cyclones._params_path(1).endswith("_<2025_model1.npz")
    assert WeatherNext2Cyclones._params_path(4).endswith("_<2025_model4.npz")
    with pytest.raises(ValueError, match="1 through 4"):
        WeatherNext2Cyclones._params_path(0)


@pytest.mark.package
def test_weathernext2_operational_package():
    model = WeatherNext2Cyclones.load_model(
        WeatherNext2Cyclones.load_default_package(), jit_compile=False
    )
    assert tuple(len(model.input_coords()[dim]) for dim in ("lat", "lon")) == (
        721,
        1440,
    )


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


@pytest.mark.parametrize("n_batch", [1, 4])
def test_weathernext2_tisr_batched(n_batch):
    """TISR is broadcast across members and matches the single-member value."""
    import xarray as xr
    from weathernext.utils import data_utils

    lat = np.linspace(-90.0, 90.0, 9)
    lon = np.linspace(0.0, 330.0, 12)
    start = np.datetime64("2025-01-01T00:00")
    stamps = np.array([start, start + np.timedelta64(6, "h")])

    def make(n):
        return xr.Dataset(
            {
                "x": (
                    ("batch", "time", "lat", "lon"),
                    np.zeros((n, 2, lat.size, lon.size), dtype=np.float32),
                )
            },
            coords={
                "batch": np.arange(n),
                "time": np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")]),
                "lat": lat,
                "lon": lon,
                "datetime": (("batch", "time"), np.tile(stamps, (n, 1))),
            },
        )

    tisr = getattr(data_utils, "TISR", "toa_incident_solar_radiation")
    reference = make(1)
    data_utils.add_tisr_var(reference)
    expected = reference[tisr].isel(batch=0).values

    data = make(n_batch)
    _add_tisr_batched(data)

    assert data.sizes["batch"] == n_batch
    for member in range(n_batch):
        np.testing.assert_allclose(data[tisr].isel(batch=member).values, expected)


@pytest.mark.parametrize("n_batch", [1, 4])
def test_weathernext2_from_dataarray_batched(n_batch, mock_weathernext2_model):
    """Converted datasets keep the caller's batch width on data and datetime."""
    import xarray as xr

    model = mock_weathernext2_model
    coords = model.input_coords()
    coords["batch"] = np.arange(n_batch)
    coords["time"] = TEST_TIME
    shape = tuple(len(v) for v in coords.values())
    data = xr.DataArray(torch.randn(*shape, dtype=torch.float32).numpy(), coords=coords)

    out, _ = model.from_dataarray_to_dataset(data, 6)

    assert out.sizes["batch"] == n_batch
    assert out["datetime"].sizes["batch"] == n_batch
    batched = [n for n in out.data_vars if "batch" in out[n].dims]
    assert batched, "no data variable carried a batch dimension"
    for name in batched:
        assert out[name].sizes["batch"] == n_batch, name
    # Static fields are shared across members and stay unbatched.
    assert "batch" not in out["land_sea_mask"].dims

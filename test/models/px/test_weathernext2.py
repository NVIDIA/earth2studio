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
from dataclasses import make_dataclass
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

try:
    from weathernext.utils import fiddle_config_io
    from weathernext.weathernext2 import fgn
except ImportError:
    fiddle_config_io = fgn = None

from test_graphcast import (
    _check_device_selector,
    _input,
    _offline_backend,
    _prediction,
    _require_device,
)

import earth2studio.models.px.weathernext2_cyclones as module
from earth2studio.data import Random, fetch_data
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px.weathernext2_cyclones import (
    OUTPUT_VARIABLES,
    WeatherNext2Cyclones,
    WeatherNext2CyclonesMini,
    _add_e2s_cyclone_columns,
)

TEST_TIME = np.array([np.datetime64("2025-01-01T00:00")])


def mocked_chunked_prediction(*args, targets_template, **kwargs):
    return targets_template


def mocked_chunked_prediction_generator(self, *args, targets_template, batch, **kwargs):
    value = float(batch["2m_temperature"].isel(time=-1).mean())
    while True:
        yield targets_template.isel(time=[0]).fillna(value)


@pytest.fixture(params=[WeatherNext2CyclonesMini, WeatherNext2Cyclones])
def mock_weathernext2_model(request, monkeypatch):
    if module.jax is not None:
        ckpt = fgn.CheckPoint(params={}, description="test", license="test")
        grid = np.ones((9, 12), dtype=np.float32)
        p = request.param(ckpt, grid, grid, jit_compile=False)
    else:
        _offline_backend(monkeypatch, module)
        monkeypatch.setattr(module, "pd", pd)
        p = request.param.__new__(request.param)
        torch.nn.Module.__init__(p)
        p.register_buffer("device_buffer", torch.empty(0))
        p.land_sea_mask = np.ones((9, 12), dtype=np.float32)
        p.geopotential_at_surface = p.land_sea_mask.copy()
        p.task_config = make_dataclass(
            "Task", [("target_variables", tuple), ("forcing_variables", tuple)]
        )((), ("year_progress_sin", "toa_incident_solar_radiation"))
        p.track_cyclones = False
        p._cyclone_tracks = pd.DataFrame()
        p._cyclone_prediction_history = []
        p._cyclone_tracker = None
        p.set_rng(0)

    def prediction(**kwargs):
        noise = (
            float(module.jax.random.uniform(kwargs["rng"])) / 100
            if module.hk is not None
            else float(kwargs["rng"][0]) / 100
        )
        return _prediction(**kwargs) + noise

    p.run_forward = prediction
    _check_device_selector(monkeypatch, p)
    return p


def fetch_random_input(model, time=TEST_TIME, device="cpu"):
    return _input(model, time, device)


def assert_output(out, time=TEST_TIME):
    assert out.shape == (len(time), 1, len(OUTPUT_VARIABLES), 9, 12)
    assert list(out.dims) == ["time", "lead_time", "variable", "lat", "lon"]
    assert np.array_equal(
        out.coords["variable"],
        ["tp:sum:6h" if v == "tp06" else v for v in OUTPUT_VARIABLES],
    )
    assert np.array_equal(out.time, time)


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
def test_weathernext2_call(time, device, mock_weathernext2_model):
    _require_device(module, device)
    model = mock_weathernext2_model.to(device)
    x = fetch_random_input(model, time)
    out = model(x)
    assert_output(out, time)
    assert out.name == x.name and out.encoding == x.encoding and out.marker == 7


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_weathernext2_iter(device, mock_weathernext2_model):
    _require_device(module, device)
    model = mock_weathernext2_model.to(device)
    x = fetch_random_input(model).expand_dims(member=2)
    saved = x.copy(deep=True)
    calls = []

    def hook(field):
        assert field.dims == x.dims and "member" not in field.coords
        calls.append(field.sizes["lead_time"])
        field.data += 1
        return field

    model.front_hook = model.rear_hook = hook
    iterator = model.create_iterator(x)
    initial = next(iterator)
    assert calls == []
    first = next(iterator)
    before = first.copy(deep=True)
    second = next(iterator)
    assert calls == [2, 1, 2, 1]
    assert second.lead_time.values == np.timedelta64(12, "h")
    delta = (
        second.sel(variable="t2m").e2s.to_torch()[0]
        - first.sel(variable="t2m").e2s.to_torch()[0]
    )
    assert torch.all((delta >= 3) & (delta < 3.021))
    xr.testing.assert_identical(x, saved)
    xr.testing.assert_identical(first, before)
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))
    model.clear_hooks()
    iterator = model.create_iterator(x)
    next(iterator)
    next(iterator)
    assert next(iterator).lead_time.values == np.timedelta64(12, "h")


def test_weathernext2_concurrent_iterators(mock_weathernext2_model):
    x = fetch_random_input(mock_weathernext2_model)
    first = mock_weathernext2_model.create_iterator(x)
    other = x.copy(deep=True)
    other.data += 1
    second = mock_weathernext2_model.create_iterator(other)
    next(first), next(second)
    assert not next(first).equals(next(second))


def test_weathernext2_rng_advances(monkeypatch, mock_weathernext2_model):
    rngs = []

    def record_rng(*args, rng, targets_template, **kwargs):
        rngs.append(np.asarray(rng))
        return targets_template

    monkeypatch.setattr(module.rollout, "chunked_prediction", record_rng)
    x = fetch_random_input(mock_weathernext2_model)
    mock_weathernext2_model(x)
    mock_weathernext2_model(x)
    assert len(rngs) == 2 and not np.array_equal(*rngs)


def test_weathernext2_conformance(mock_weathernext2_model):
    model = mock_weathernext2_model
    check_prognostic_contract(model)


def test_weathernext2_set_rng(mock_weathernext2_model):
    mock_weathernext2_model.set_rng(123)
    key = np.asarray(mock_weathernext2_model.prng_key)
    mock_weathernext2_model.set_rng(456, reset=False)
    np.testing.assert_array_equal(key, mock_weathernext2_model.prng_key)
    mock_weathernext2_model.set_rng(456)
    assert not np.array_equal(key, mock_weathernext2_model.prng_key)
    mock_weathernext2_model.set_rng(123)
    np.testing.assert_array_equal(key, mock_weathernext2_model.prng_key)


def test_weathernext2_target_order():
    pytest.importorskip("weathernext")
    targets = fiddle_config_io.get_fiddle_config_by_name(
        "weathernext2/configs/WeatherNextCyclones_Mini"
    ).task.target_variables
    expected = tuple(
        variable for variable in targets if not variable.startswith("cyclone")
    )
    model = WeatherNext2CyclonesMini.__new__(WeatherNext2CyclonesMini)
    torch.nn.Module.__init__(model)
    model.track_cyclones = False
    assert model._load_task_config().target_variables == expected


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


def test_weathernext2_call_updates_cyclone_tracks(mock_weathernext2_model):
    model = mock_weathernext2_model
    model.track_cyclones = True
    x = fetch_random_input(model)
    with (
        mock.patch.object(model, "_reset_cyclone_tracks") as reset,
        mock.patch.object(model, "_update_cyclone_tracks") as update,
    ):
        model(x)
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
    x = fetch_data(
        Random(coords),
        TEST_TIME,
        model.input_coords()["variable"],
        model.input_coords()["lead_time"],
        device=device,
    )
    with pytest.raises((KeyError, ValueError)):
        model(x)


def test_weathernext2_operational_checkpoint():
    assert WeatherNext2Cyclones._params_path(1).endswith("_<2025_model1.npz")
    assert WeatherNext2Cyclones._params_path(4).endswith("_<2025_model4.npz")
    with pytest.raises(ValueError, match="1 through 4"):
        WeatherNext2Cyclones._params_path(0)
    for cls in (WeatherNext2Cyclones, WeatherNext2CyclonesMini):
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        shape = (721, 1440) if cls is WeatherNext2Cyclones else (181, 360)
        model.land_sea_mask = np.ones(shape)
        signature = model.input_coords()
        assert signature.shape == (0, 0, 2, 83, *shape)
        if cls is WeatherNext2Cyclones:
            assert signature.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
        assert "tp:sum:6h" in model.output_coords(signature).coords["variable"]


@pytest.mark.package
def test_weathernext2_operational_package():
    pytest.importorskip("weathernext")
    model = WeatherNext2Cyclones.load_model(
        WeatherNext2Cyclones.load_default_package(), jit_compile=False
    )
    assert tuple(len(model.input_coords()[dim]) for dim in ("lat", "lon")) == (
        721,
        1440,
    )


@pytest.mark.package
def test_weathernext2_package():
    pytest.importorskip("weathernext")
    torch.cuda.empty_cache()
    model = WeatherNext2CyclonesMini.load_model(
        WeatherNext2CyclonesMini.load_default_package(), jit_compile=False
    ).to("cuda:0")
    assert (
        len(model.input_coords()["lat"]),
        len(model.input_coords()["lon"]),
        len(model.output_coords(model.input_coords())["variable"]),
    ) == (181, 360, 84)

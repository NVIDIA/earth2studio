# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
from unittest import mock

import numpy as np
import pytest
import torch
import xarray as xr
from graphcast import graphcast

from earth2studio.data import Random, fetch_data
from earth2studio.models.px.graphcast_operational import GraphCastOperational
from earth2studio.models.px.graphcast_small import GraphCastSmall
from earth2studio.utils import handshake_dim


def mocked_chunked_prediction_generator(
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


@pytest.fixture
def mock_GraphCastSmall_model():
    # Spoof model
    model_config = graphcast.ModelConfig(
        resolution=1.0,
        mesh_size=5,
        latent_size=512,
        gnn_msg_steps=16,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
    )
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[13],
        input_duration=graphcast.TASK.input_duration,
    )

    class CKPT:
        def __init__(self, model_config, task_config):
            self.model_config = model_config
            self.task_config = task_config
            self.params = {}
            self.description = "some"
            self.license = "license"

    static_data = {}
    for v in (
        graphcast.ALL_ATMOSPHERIC_VARS
        + graphcast.TARGET_SURFACE_VARS
        + graphcast.FORCING_VARS
    ):
        if v in graphcast.TARGET_ATMOSPHERIC_VARS:
            static_data[v] = ("level", np.ones(len(graphcast.PRESSURE_LEVELS[37])))
        else:
            static_data[v] = 1

    diffs_stddev_by_level = xr.Dataset(
        static_data, coords={"level": list(graphcast.PRESSURE_LEVELS[37])}
    )
    mean_by_level = xr.Dataset(
        static_data, coords={"level": list(graphcast.PRESSURE_LEVELS[37])}
    )
    stddev_by_level = xr.Dataset(
        static_data, coords={"level": list(graphcast.PRESSURE_LEVELS[37])}
    )

    # Create a proper checkpoint with initialized parameters
    ckpt = CKPT(model_config, task_config)

    # Initialize the model with the checkpoint
    p = GraphCastSmall(
        ckpt,
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level,
        np.ones((181, 360)),
        np.ones((181, 360)),
    )

    # Set chunked_prediction_generator to the mocked function
    p._chunked_prediction_generator = mocked_chunked_prediction_generator

    return p


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [np.datetime64("2001-06-04T00:00")]
        ),  # Only len 1 time array is supported by GraphCastSmall model
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@mock.patch("graphcast.rollout.chunked_prediction", mocked_chunked_prediction)
def test_graphcast_small_call(time, device, mock_GraphCastSmall_model):

    p = mock_GraphCastSmall_model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 83, 181, 360])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@mock.patch(
    "graphcast.rollout.chunked_prediction_generator",
    mocked_chunked_prediction_generator,
)
def test_graphcast_small_iter(ensemble, device, mock_GraphCastSmall_model):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = mock_GraphCastSmall_model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    input, input_coords = next(p_iter)  # Skip first which should return the input
    assert input_coords["lead_time"] == np.timedelta64(0, "h")
    assert len(input.shape) == 6
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 83, 181, 360])
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"] == np.timedelta64(6 * (i + 1), "h")

        if i > 5:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_graphcast_small_exceptions(dc, device, mock_GraphCastSmall_model):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = mock_GraphCastSmall_model.to(device)
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.fixture(scope="function")
def model() -> GraphCastSmall:
    package = GraphCastSmall.load_default_package()
    p = GraphCastSmall.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.timeout(360)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_graphcast_small_package(model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package graphcast
    p = model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Check iter
    p_iter = p.create_iterator(x, coords)
    for i in range(3):
        out, out_coords = next(p_iter)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 83, 181, 360])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.fixture
def mock_GraphCastOperational_model():
    model_config = graphcast.ModelConfig(
        resolution=0.25,
        mesh_size=5,
        latent_size=512,
        gnn_msg_steps=16,
        hidden_layers=1,
        radius_query_fraction_edge_length=0.6,
    )
    task_config = graphcast.TaskConfig(
        input_variables=graphcast.TASK.input_variables,
        target_variables=graphcast.TASK.target_variables,
        forcing_variables=graphcast.TASK.forcing_variables,
        pressure_levels=graphcast.PRESSURE_LEVELS[13],
        input_duration=graphcast.TASK.input_duration,
    )

    class CKPT:
        def __init__(self, model_config, task_config):
            self.model_config = model_config
            self.task_config = task_config
            self.params = {}
            self.description = "some"
            self.license = "license"

    static_data = {}
    for v in (
        graphcast.ALL_ATMOSPHERIC_VARS
        + graphcast.TARGET_SURFACE_VARS
        + graphcast.FORCING_VARS
    ):
        if v in graphcast.TARGET_ATMOSPHERIC_VARS:
            static_data[v] = ("level", np.ones(len(graphcast.PRESSURE_LEVELS[13])))
        else:
            static_data[v] = 1
    diffs_stddev_by_level = xr.Dataset(
        static_data, coords={"level": list(graphcast.PRESSURE_LEVELS[13])}
    )
    mean_by_level = xr.Dataset(
        static_data, coords={"level": list(graphcast.PRESSURE_LEVELS[13])}
    )
    stddev_by_level = xr.Dataset(
        static_data, coords={"level": list(graphcast.PRESSURE_LEVELS[13])}
    )
    ckpt = CKPT(model_config, task_config)
    p = GraphCastOperational(
        ckpt,
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level,
        np.ones((721, 1440)),
        np.ones((721, 1440)),
    )
    p._chunked_prediction_generator = mocked_chunked_prediction_generator
    return p


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@mock.patch("graphcast.rollout.chunked_prediction", mocked_chunked_prediction)
def test_graphcast_operational_call(device, mock_GraphCastOperational_model):

    p = mock_GraphCastOperational_model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    time = np.array([np.datetime64("2010-01-01T00:00")])
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)
    out, out_coords = p(x, coords)
    assert out.shape == (1, 1, 83, 721, 1440)
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@mock.patch(
    "graphcast.rollout.chunked_prediction_generator",
    mocked_chunked_prediction_generator,
)
def test_graphcast_operational_iter(ensemble, device, mock_GraphCastOperational_model):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = mock_GraphCastOperational_model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    input, input_coords = next(p_iter)  # Skip first which should return the input
    assert input_coords["lead_time"] == np.timedelta64(0, "h")
    assert input.shape == torch.Size(
        [ensemble, len(time), 1, 83, 721, 1440]
    )  # 83, tp06 included in output
    assert len(input.shape) == 6
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 83, 721, 1440])
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"] == np.timedelta64(6 * (i + 1), "h")

        if i > 5:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_graphcast_operational_exceptions(dc, device, mock_GraphCastOperational_model):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = mock_GraphCastOperational_model.to(device)
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.fixture(scope="function")
def operational_model() -> GraphCastOperational:
    package = GraphCastOperational.load_default_package()
    p = GraphCastOperational.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_graphcast_operational_package(operational_model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package graphcast
    p = operational_model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Check iter
    p_iter = p.create_iterator(x, coords)
    for i in range(3):
        out, out_coords = next(p_iter)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 83, 721, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)

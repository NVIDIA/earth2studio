# From https://github.com/NVIDIA/earth2studio/pull/208

from collections import OrderedDict
from collections.abc import Iterable
from unittest import mock

import numpy as np
import pytest
import torch
import xarray as xr
from graphcast import graphcast

from earth2studio.data import Random, fetch_data
from earth2studio.models.px.graphcast import GraphCast
from earth2studio.utils import handshake_dim

CUDA_DEVICE = 0


def mocked_chunked_prediction_generator(
    predictor_fn,
    rng,
    inputs,
    targets_template,
    forcings,
    num_steps_per_chunk=None,
    verbose=None,
):
    # iterator returns 1 template lead time at a time
    while True:
        if "ensemble" in targets_template.dims:
            targets_template = targets_template.squeeze("batch").rename(
                {"ensemble": "batch"}
            )
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


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [np.datetime64("2001-06-04T00:00")]
        ),  # Only len 1 time array is supported by Auroral model
    ],
)
@pytest.mark.parametrize("device", ["cpu", f"cuda:{CUDA_DEVICE}"])
@mock.patch("graphcast.rollout.chunked_prediction", mocked_chunked_prediction)
def test_graphcast_call(time, device):

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
            self.params = None
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

    p = GraphCast(
        CKPT(model_config, task_config),
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level,
    ).to(device)

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
    print("coords: ", coords)

    nsteps = 1
    out, out_coords = p.set_nsteps(1)(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1 + nsteps, 85, 181, 360])
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
@pytest.mark.parametrize("device", ["cpu", f"cuda:{CUDA_DEVICE}"])
@mock.patch(
    "graphcast.rollout.chunked_prediction_generator",
    mocked_chunked_prediction_generator,
)
def test_graphcast_iter(ensemble, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
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
            self.params = None
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

    p = GraphCast(
        CKPT(model_config, task_config),
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level,
    ).to(device)

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

    nsteps = 5
    p = p.set_nsteps(nsteps)
    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    input, input_coords = next(p_iter)  # Skip first which should return the input
    assert len(input.shape) == 6
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 85, 181, 360])
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
@pytest.mark.parametrize("device", ["cpu", f"cuda:{CUDA_DEVICE}"])
def test_graphcast_exceptions(dc, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
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
            self.params = None
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

    p = GraphCast(
        CKPT(model_config, task_config),
        diffs_stddev_by_level,
        mean_by_level,
        stddev_by_level,
    ).to(device)
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p.set_nsteps(1)(x, coords)


@pytest.fixture(scope="module")
def model(model_cache_context) -> GraphCast:
    # Test only on cuda device
    with model_cache_context():
        package = GraphCast.load_default_package()
        p = GraphCast.load_model(package)
        return p


@pytest.mark.ci_cache
# @pytest.mark.timeout()
@pytest.mark.parametrize("device", ["cpu", f"cuda:{CUDA_DEVICE}"])
def test_graphcast_package(model, device):
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

    nsteps = 1
    out, out_coords = p.set_nsteps(nsteps)(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), nsteps + 1, 85, 181, 360])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)

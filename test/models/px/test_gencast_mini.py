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
from unittest import mock

import numpy as np
import pytest
import torch
import xarray as xr

try:
    from graphcast import denoiser, graphcast
    from graphcast import gencast as gencast_module
except ImportError:
    pytest.importorskip("graphcast")

from earth2studio.data import Random, fetch_data
from earth2studio.models.px.gencast_mini import (
    ATMOS_VARIABLES,
    GENERATED_FORCING_VARS,
    INPUT_VARIABLES,
    OUTPUT_VARIABLES,
    PRESSURE_LEVELS,
    GenCastMini,
)
from earth2studio.utils import handshake_dim

# GenCast-specific variable lists (matching graphcast module names)
GENCAST_TARGET_SURFACE_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_surface_temperature",
    "total_precipitation_12hr",
)

GENCAST_TARGET_SURFACE_NO_PRECIP_VARS = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "sea_surface_temperature",
)

GENCAST_TARGET_ATMOS_VARS = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)

GENCAST_STATIC_VARS = ("geopotential_at_surface", "land_sea_mask")


def mocked_chunked_prediction_generator(
    predictor_fn,
    rng,
    inputs,
    targets_template,
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


def _build_fake_ckpt(n_lat: int, n_lon: int):
    """Build a fake GenCast checkpoint for testing."""
    # Use the GenCast TASK config from the graphcast module
    task_config = graphcast.TaskConfig(
        input_variables=(
            GENCAST_TARGET_SURFACE_NO_PRECIP_VARS
            + graphcast.TARGET_ATMOSPHERIC_VARS
            + tuple(GENERATED_FORCING_VARS)
            + GENCAST_STATIC_VARS
        ),
        target_variables=(
            GENCAST_TARGET_SURFACE_VARS + graphcast.TARGET_ATMOSPHERIC_VARS
        ),
        forcing_variables=tuple(GENERATED_FORCING_VARS),
        pressure_levels=graphcast.PRESSURE_LEVELS_WEATHERBENCH_13,
        input_duration="24h",
    )

    # Build a minimal DenoiserArchitectureConfig
    # SparseTransformerConfig and DenoiserArchitectureConfig live in the
    # ``denoiser`` module, not ``gencast``.
    sparse_transformer_config = denoiser.SparseTransformerConfig(
        attention_k_hop=4,
        d_model=512,
        num_layers=2,
        num_heads=4,
        attention_type="triblockdiag_mha",
        mask_type="full",
    )
    denoiser_architecture_config = denoiser.DenoiserArchitectureConfig(
        sparse_transformer_config=sparse_transformer_config,
        mesh_size=4,
        latent_size=512,
    )

    sampler_config = gencast_module.SamplerConfig()
    noise_config = gencast_module.NoiseConfig()
    noise_encoder_config = denoiser.NoiseEncoderConfig()

    class CKPT:
        def __init__(self):
            self.params = {}
            self.task_config = task_config
            self.denoiser_architecture_config = denoiser_architecture_config
            self.sampler_config = sampler_config
            self.noise_config = noise_config
            self.noise_encoder_config = noise_encoder_config
            self.description = "test"
            self.license = "test"

    return CKPT()


def _build_fake_stats(n_levels: int = 13):
    """Build fake normalization stats datasets."""
    pressure_levels = list(graphcast.PRESSURE_LEVELS_WEATHERBENCH_13)

    static_data = {}
    for v in (
        GENCAST_TARGET_ATMOS_VARS
        + GENCAST_TARGET_SURFACE_VARS
        + tuple(GENERATED_FORCING_VARS)
    ):
        if v in GENCAST_TARGET_ATMOS_VARS:
            static_data[v] = ("level", np.ones(n_levels, dtype=np.float32))
        else:
            static_data[v] = np.float32(1.0)

    coords = {"level": pressure_levels}
    diffs_stddev = xr.Dataset(static_data, coords=coords)
    mean = xr.Dataset(static_data, coords=coords)
    stddev = xr.Dataset(static_data, coords=coords)
    min_vals = xr.Dataset(static_data, coords=coords)

    return diffs_stddev, mean, stddev, min_vals


# ===== 1.0-degree (181x360) tests =====


@pytest.fixture
def mock_GenCastMini_model():
    n_lat, n_lon = 181, 360
    ckpt = _build_fake_ckpt(n_lat, n_lon)
    diffs_stddev, mean, stddev, min_vals = _build_fake_stats()

    p = GenCastMini(
        ckpt,
        diffs_stddev,
        mean,
        stddev,
        min_vals,
        np.ones((n_lat, n_lon), dtype=np.float32),
        np.ones((n_lat, n_lon), dtype=np.float32),
        np.ones((n_lat, n_lon), dtype=bool),
    )

    # Mock the iterator to avoid needing real JAX inference
    p._chunked_prediction_generator = mocked_chunked_prediction_generator

    return p


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2010-01-01T00:00")]),
        np.array(
            [np.datetime64("2010-01-01T00:00"), np.datetime64("2010-01-02T00:00")]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@mock.patch("graphcast.rollout.chunked_prediction", mocked_chunked_prediction)
def test_gencast_mini_call(time, device, mock_GenCastMini_model):

    p = mock_GenCastMini_model.to(device)

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
    assert out.shape == torch.Size([len(time), 1, 84, 181, 360])
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
def test_gencast_mini_iter(ensemble, device, mock_GenCastMini_model):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = mock_GenCastMini_model.to(device)

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
        [ensemble, len(time), 1, 84, 181, 360]
    )  # 84 output vars, tp12 included
    assert len(input.shape) == 6
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 84, 181, 360])
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"] == np.timedelta64(12 * (i + 1), "h")

        if i > 5:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(180)}),
        OrderedDict({"lat": np.random.randn(180), "phoo": np.random.randn(360)}),
    ],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_gencast_mini_exceptions(dc, device, mock_GenCastMini_model):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    p = mock_GenCastMini_model.to(device)
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


def test_gencast_mini_variables():
    """Test that variable lists are consistent."""
    assert len(INPUT_VARIABLES) == 83  # 5 surface + 6*13 atmos
    assert len(OUTPUT_VARIABLES) == 84  # 6 surface (with tp12) + 6*13 atmos
    assert len(PRESSURE_LEVELS) == 13
    assert len(ATMOS_VARIABLES) == 6
    # tp12 is in output but not input
    assert "tp12" not in INPUT_VARIABLES
    assert "tp12" in OUTPUT_VARIABLES


@pytest.fixture(scope="function")
def model() -> GenCastMini:
    package = GenCastMini.load_default_package()
    p = GenCastMini.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_gencast_mini_package(model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package gencast mini
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

    assert out.shape == torch.Size([len(time), 1, 84, 181, 360])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)

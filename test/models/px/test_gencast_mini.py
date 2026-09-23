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

from dataclasses import make_dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr
from test_graphcast import (
    _check_device_selector,
    _input,
    _offline_backend,
    _prediction,
    _require_device,
    _stats,
)

import earth2studio.models.px.gencast_mini as module
from earth2studio.models.conformance import ContractException, check_prognostic_contract
from earth2studio.models.px.gencast_mini import (
    ATMOS_VARIABLES,
    INPUT_VARIABLES,
    OUTPUT_VARIABLES,
    PRESSURE_LEVELS,
    GenCastMini,
)


@pytest.fixture
def mock_GenCastMini_model(monkeypatch):
    if module.jax is not None:
        from weathernext.weathernext1_gen import denoiser

        surface = (
            "2m_temperature",
            "mean_sea_level_pressure",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "sea_surface_temperature",
        )
        atmos = module.gencast.TARGET_ATMOSPHERIC_VARS
        task = module.graphcast.TaskConfig(
            input_variables=surface
            + atmos
            + module.GENERATED_FORCING_VARS
            + ("geopotential_at_surface", "land_sea_mask"),
            target_variables=surface + ("total_precipitation_12hr",) + atmos,
            forcing_variables=module.GENERATED_FORCING_VARS,
            pressure_levels=tuple(PRESSURE_LEVELS),
            input_duration="24h",
        )
        ckpt = module.gencast.CheckPoint(
            params={},
            task_config=task,
            denoiser_architecture_config=denoiser.DenoiserArchitectureConfig(
                sparse_transformer_config=denoiser.SparseTransformerConfig(
                    attention_k_hop=4,
                    d_model=512,
                    num_layers=2,
                    num_heads=4,
                    attention_type="triblockdiag_mha",
                    mask_type="full",
                ),
                mesh_size=4,
                latent_size=512,
            ),
            sampler_config=module.gencast.SamplerConfig(),
            noise_config=module.gencast.NoiseConfig(),
            noise_encoder_config=denoiser.NoiseEncoderConfig(),
            description="test",
            license="test",
        )
        stats = _stats(task)
        p = GenCastMini(
            ckpt,
            stats,
            stats,
            stats,
            stats,
            np.ones((5, 8)),
            np.ones((5, 8)),
            np.ones((5, 8), dtype=bool),
            jit_compile=False,
        )
    else:
        _offline_backend(monkeypatch, module)
        p = GenCastMini.__new__(GenCastMini)
        torch.nn.Module.__init__(p)
        p.register_buffer("device_buffer", torch.empty(0))
        p.land_sea_mask = np.ones((5, 8))
        p.geopotential_at_surface = np.ones((5, 8))
        p.sst_nan_mask = np.ones((5, 8), dtype=bool)
        p.seed = 0
        p.ckpt = SimpleNamespace(
            task_config=make_dataclass("Task", [("forcing_variables", tuple)])(())
        )

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


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_gencast_mini_call(device, mock_GenCastMini_model):
    _require_device(module, device)
    p = mock_GenCastMini_model.to(device)
    x = _input(p, np.array(["2010-01-01", "2010-01-02"], dtype="datetime64[ns]"))
    before = x.copy(deep=True)
    out = p(x)
    assert out.shape == (2, 1, 84, 5, 8)
    assert out.dims == x.dims and out.name == x.name and out.encoding == x.encoding
    assert out.lead_time.values == np.timedelta64(12, "h")
    assert "tp:sum:12h" in out.coords["variable"]
    delta = (
        out.sel(variable="t2m").e2s.to_torch()[0].cpu()
        - x.sel(variable="t2m").e2s.to_torch()[0][:, -1:]
    )
    assert torch.all((delta >= 1) & (delta < 1.011))
    xr.testing.assert_identical(x, before)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_gencast_mini_iter(device, mock_GenCastMini_model):
    _require_device(module, device)
    p = mock_GenCastMini_model.to(device)
    x = _input(p).expand_dims(member=2)
    before = x.copy(deep=True)
    calls = []

    def hook(field):
        assert field.dims == x.dims and "member" not in field.coords
        calls.append(field.sizes["lead_time"])
        field.data += 1
        return field

    p.front_hook = p.rear_hook = hook
    iterator = p.create_iterator(x)
    initial = next(iterator)
    assert calls == []
    first = next(iterator)
    saved = first.copy(deep=True)
    second = next(iterator)
    assert calls == [2, 1, 2, 1]
    assert second.lead_time.values == np.timedelta64(24, "h")
    delta = (
        second.sel(variable="t2m").e2s.to_torch()[0]
        - first.sel(variable="t2m").e2s.to_torch()[0]
    )
    assert torch.all((delta >= 3) & (delta < 3.011))
    xr.testing.assert_identical(x, before)
    xr.testing.assert_identical(first, saved)
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))


def test_gencast_mini_exceptions(mock_GenCastMini_model):
    p = mock_GenCastMini_model
    x = _input(p)
    for bad in (
        x.transpose(..., "lon", "lat"),
        x.assign_coords(lead_time=[0, 12]),
        x.drop_vars("lead_time"),
    ):
        with pytest.raises(ValueError):
            p(bad)


def test_gencast_mini_conformance(mock_GenCastMini_model):
    mock_GenCastMini_model.seed = None
    with pytest.raises(ContractException) as exc_info:
        check_prognostic_contract(mock_GenCastMini_model)
    assert exc_info.value.violations == [
        "P13: repeated runs with the same input and seed disagree"
    ]


def test_gencast_mini_variables(mock_GenCastMini_model):
    assert len(INPUT_VARIABLES) == 83
    assert len(OUTPUT_VARIABLES) == 84
    assert len(PRESSURE_LEVELS) == 13
    assert len(ATMOS_VARIABLES) == 6
    assert "tp12" not in INPUT_VARIABLES and "tp12" in OUTPUT_VARIABLES
    p = mock_GenCastMini_model
    signature = p.input_coords()
    assert signature.shape == (0, 0, 2, 83, 5, 8)
    assert "earth2studio_grid_id" not in signature.attrs
    assert "earth2studio_crs" in signature.attrs
    assert "tp:sum:12h" in p.output_coords(signature).coords["variable"]


@pytest.fixture
def model():
    pytest.importorskip("weathernext")
    return GenCastMini.load_model(GenCastMini.load_default_package())


@pytest.mark.package
def test_gencast_mini_package(model):
    p = model.to("cuda:0")
    x = _input(p, device="cuda:0")
    iterator = p.create_iterator(x)
    xr.testing.assert_identical(next(iterator), x.isel(lead_time=slice(-1, None)))
    next(iterator)
    out = next(iterator)
    assert out.shape == (1, 1, 84, 181, 360)
    assert out.lead_time.values == np.timedelta64(24, "h")

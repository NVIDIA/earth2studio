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

from contextlib import nullcontext
from dataclasses import make_dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.px.graphcast_operational as operational
import earth2studio.models.px.graphcast_small as small
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px.graphcast_operational import GraphCastOperational
from earth2studio.models.px.graphcast_small import GraphCastSmall
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


def _offline_backend(monkeypatch, module):
    if module.jax is not None:
        return module.jax.random
    # Substitute unavailable numerical dependencies, retaining wrapper dataset
    # conversion, native recurrent generator, forcing cadence and tensor ordering.
    monkeypatch.setattr(module, "chex", SimpleNamespace(PRNGKey=np.ndarray))
    random = SimpleNamespace(
        PRNGKey=lambda seed: np.array([seed, 0], dtype=np.int64),
        split=lambda key: (key + [0, 1], key + [1, 1]),
        fold_in=lambda key, i: key + [0, i],
    )
    monkeypatch.setattr(
        module,
        "jax",
        SimpleNamespace(
            random=random,
            default_device=lambda device: nullcontext(),
            devices=lambda kind: [kind],
        ),
    )

    def extract(data, target_lead_times, **kwargs):
        if hasattr(module, "FORCING_VARIABLES") or hasattr(
            module, "WN2_TARGET_VARIABLES"
        ):
            tisr(data)
        inputs = data.isel(time=slice(0, 2)).drop_vars("datetime")
        targets = data.isel(time=slice(2, None)).drop_vars("datetime")
        targets = targets.drop_vars(
            [
                "land_sea_mask",
                "geopotential_at_surface",
                "toa_incident_solar_radiation",
            ],
            errors="ignore",
        )
        forcing = xr.Dataset(coords={"time": targets.time})
        return inputs, targets, forcing

    def derived(data):
        data["year_progress_sin"] = xr.zeros_like(data["datetime"], dtype=float)

    def tisr(data):
        if "toa_incident_solar_radiation" in data:
            return
        # The pinned backend squeezes batch before computing solar radiation.
        sample = data.squeeze("batch")
        values = xr.ones_like(sample["2m_temperature"])
        data["toa_incident_solar_radiation"] = values.expand_dims("batch", axis=0)

    data_utils = SimpleNamespace(
        extract_inputs_targets_forcings=extract,
        add_derived_vars=derived,
        add_tisr_var=tisr,
        get_year_progress=lambda seconds: np.zeros_like(seconds, dtype=float),
        get_day_progress=lambda seconds, lon: np.zeros((len(seconds), len(lon))),
        featurize_progress=lambda *args: {},
    )
    monkeypatch.setattr(module, "data_utils", data_utils)

    def next_inputs(inputs, next_frame):
        next_frame = next_frame.assign_coords(time=inputs.time.values[-1:])
        return xr.concat(
            (inputs.isel(time=slice(-1, None)), next_frame),
            dim="time",
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )

    def prediction(predictor, rng, inputs, targets_template, forcings):
        _, step_rng = random.split(rng)
        return predictor(
            rng=step_rng,
            inputs=inputs,
            targets_template=targets_template,
            forcings=forcings,
        )

    monkeypatch.setattr(
        module,
        "rollout",
        SimpleNamespace(
            chunked_prediction=prediction,
            _get_next_inputs=next_inputs,
        ),
    )
    if hasattr(module, "FORCING_VARIABLES"):
        monkeypatch.setattr(
            module,
            "FORCING_VARIABLES",
            ("year_progress_sin", "toa_incident_solar_radiation"),
        )
    return random


def _prediction(rng, inputs, targets_template, forcings):
    result = targets_template.copy(deep=True)
    for name in result.data_vars:
        if name in inputs:
            result[name] = (
                inputs[name].isel(time=slice(-1, None)).assign_coords(time=result.time)
                + 1
            )
        else:
            result[name] = xr.zeros_like(result[name])
    return result


def _input(model, time=None, device="cpu"):
    if time is None:
        time = np.array(["2001-06-04"], dtype="datetime64[ns]")
    signature = coord_array_like(model.input_coords(), {"time": time})
    signature = coord_array_like(signature, {"batch": [0]})
    x = from_torch(torch.rand(signature.shape, device=device), signature).isel(
        batch=0, drop=True
    )
    x.name = "weather"
    x.attrs["source"] = "test"
    x.encoding["note"] = "kept"
    return x.assign_coords(marker=7)


def _stats(task):
    from weathernext.utils import variables

    return xr.Dataset(
        {
            name: (
                ("level", np.ones(len(task.pressure_levels), dtype=np.float32))
                if name in variables.ALL_ATMOSPHERIC_VARS
                else np.float32(1)
            )
            for name in set(
                task.input_variables + task.target_variables + task.forcing_variables
            )
        },
        coords={"level": list(task.pressure_levels)},
    )


def _require_device(module, device):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        if module.hk is not None:
            try:
                module.jax.devices("gpu")
            except RuntimeError:
                pytest.skip("JAX GPU backend unavailable")


def _check_device_selector(monkeypatch, model):
    selector = model.get_jax_device_from_tensor

    def select(tensor):
        assert tensor is model.device_buffer
        assert tensor.numel() == 0
        return selector(tensor)

    monkeypatch.setattr(model, "get_jax_device_from_tensor", select)


@pytest.fixture(params=[small, operational], ids=["small", "operational"])
def graphcast(request, monkeypatch):
    module = request.param
    cls = GraphCastSmall if module is small else GraphCastOperational
    if module.jax is not None:
        backend = module.graphcast
        task = backend.TASK_13 if module is small else backend.TASK_13_PRECIP_OUT
        config = backend.ModelConfig(
            resolution=1.0 if module is small else 0.25,
            mesh_size=5,
            latent_size=512,
            gnn_msg_steps=16,
            hidden_layers=1,
            radius_query_fraction_edge_length=0.6,
        )
        ckpt = backend.CheckPoint(
            params={},
            model_config=config,
            task_config=task,
            description="test",
            license="test",
        )
        stats = _stats(task)
        p = cls(ckpt, stats, stats, stats, np.ones((5, 8)), np.ones((5, 8)))
    else:
        random = _offline_backend(monkeypatch, module)
        p = cls.__new__(cls)
        torch.nn.Module.__init__(p)
        p.register_buffer("device_buffer", torch.empty(0))
        p.land_sea_mask = np.ones((5, 8))
        p.geopotential_at_surface = np.ones((5, 8))
        p.prng_key = random.PRNGKey(0)
        p.ckpt = SimpleNamespace(
            task_config=make_dataclass("Task", [("forcing_variables", tuple)])(
                module.FORCING_VARIABLES
            )
        )
    p.run_forward = _prediction
    native = p.input_coords()
    signature = coord_array(
        native.dims,
        {
            "lead_time": native.lead_time.values,
            "variable": native.coords["variable"].values,
            "lat": np.linspace(90, -90, 5),
            "lon": np.arange(8) * 45.0,
        },
        dynamic=("batch", "time"),
    )
    monkeypatch.setattr(p, "input_coords", lambda: signature)
    _check_device_selector(monkeypatch, p)
    return p


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_graphcast_small_call(graphcast, device):
    _require_device(
        small if isinstance(graphcast, GraphCastSmall) else operational, device
    )
    p = graphcast.to(device)
    x = _input(p, np.array(["1993-04-05", "2001-06-04"], dtype="datetime64[ns]"))
    before = x.copy(deep=True)
    out = p(x)
    assert out.shape == (2, 1, 83, 5, 8)
    assert out.dims == x.dims
    assert out.name == x.name and out.encoding == x.encoding
    assert out.attrs["source"] == "test" and out.marker == 7
    assert out.lead_time.values == np.timedelta64(6, "h")
    torch.testing.assert_close(
        out.sel(variable="t2m").e2s.to_torch()[0].cpu(),
        x.sel(variable="t2m").e2s.to_torch()[0][:, -1:] + 1,
    )
    xr.testing.assert_identical(x, before)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_graphcast_small_iter(graphcast, device):
    _require_device(
        small if isinstance(graphcast, GraphCastSmall) else operational, device
    )
    p = graphcast.to(device)
    x = _input(p).expand_dims(member=2)
    before = x.copy(deep=True)
    calls = []

    def hook(field):
        assert field.dims == x.dims and "member" not in field.coords
        calls.append(field.sizes["lead_time"])
        field.data += 1
        return field

    p.front_hook = p.rear_hook = hook
    it = p.create_iterator(x)
    initial = next(it)
    assert calls == []
    first = next(it)
    saved = first.copy(deep=True)
    second = next(it)
    assert calls == [2, 1, 2, 1]
    assert second.lead_time.values == np.timedelta64(12, "h")
    torch.testing.assert_close(
        second.sel(variable="t2m").e2s.to_torch()[0],
        first.sel(variable="t2m").e2s.to_torch()[0] + 3,
    )
    xr.testing.assert_identical(x, before)
    xr.testing.assert_identical(first, saved)
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))
    p.clear_hooks()
    iterator = p.create_iterator(x)
    next(iterator)
    next(iterator)
    assert next(iterator).lead_time.values == np.timedelta64(12, "h")


def test_graphcast_small_exceptions(graphcast):
    x = _input(graphcast)
    for bad in (
        x.transpose(..., "lon", "lat"),
        x.assign_coords(lead_time=[0, 6]),
        x.drop_vars("lead_time"),
    ):
        with pytest.raises(ValueError):
            graphcast(bad)


def test_graphcast_small_conformance(graphcast):
    check_prognostic_contract(graphcast)
    module = small if isinstance(graphcast, GraphCastSmall) else operational
    forcing = xr.Dataset(
        {
            "2m_temperature": (
                ("batch", "time", "lat", "lon"),
                np.zeros((2, 1, 2, 2), dtype=np.float32),
            )
        },
        coords={
            "batch": [9, 3],
            "time": np.array([6], dtype="timedelta64[h]"),
            "lat": [-30.0, 30.0],
            "lon": [0.0, 180.0],
            "datetime": (
                ("batch", "time"),
                np.array(
                    [["2001-06-04T00"], ["2001-06-04T12"]], dtype="datetime64[ns]"
                ),
            ),
        },
    )
    expected = []
    for i in range(2):
        sample = forcing.isel(batch=slice(i, i + 1)).copy(deep=True)
        module.data_utils.add_tisr_var(sample)
        expected.append(sample.toa_incident_solar_radiation)
    operational._add_tisr_batched(forcing, module.data_utils)
    xr.testing.assert_identical(
        forcing.toa_incident_solar_radiation, xr.concat(expected, dim="batch")
    )
    np.testing.assert_array_equal(forcing.batch, [9, 3])
    prepared = []
    original = graphcast.from_dataarray_to_dataset

    def prepare(*args, **kwargs):
        prepared.append(1)
        return original(*args, **kwargs)

    graphcast.from_dataarray_to_dataset = prepare
    graphcast.rear_hook = lambda field: field.assign_coords(hook_marker=1)
    iterator = graphcast.create_iterator(_input(graphcast))
    next(iterator)
    next(iterator)
    assert next(iterator).hook_marker == 1
    assert len(prepared) == 1
    p = type(graphcast).__new__(type(graphcast))
    torch.nn.Module.__init__(p)
    signature = p.input_coords()
    assert signature.sizes["lat"] == (181 if isinstance(p, GraphCastSmall) else 721)
    if isinstance(p, GraphCastOperational):
        assert signature.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    assert "tp06" in p.output_coords(signature).coords["variable"]
    shifted = coord_array_like(
        signature, {"lead_time": np.array([6, 12], dtype="timedelta64[h]")}
    )
    assert p.output_coords(shifted).lead_time.values == np.timedelta64(18, "h")


@pytest.fixture(params=[GraphCastSmall, GraphCastOperational])
def model(request):
    pytest.importorskip("weathernext")
    cls = request.param
    return cls.load_model(cls.load_default_package())


@pytest.mark.package
@pytest.mark.timeout(360)
def test_graphcast_small_package(model):
    p = model.to("cuda:0")
    x = _input(p, device="cuda:0")
    iterator = p.create_iterator(x)
    initial = next(iterator)
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))
    next(iterator)
    out = next(iterator)
    assert out.shape == (
        1,
        1,
        83,
        p.input_coords().sizes["lat"],
        p.input_coords().sizes["lon"],
    )
    assert out.lead_time.values == np.timedelta64(12, "h")

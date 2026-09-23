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

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.models.conformance import (
    ContractException,
    check_prognostic_contract,
)
from earth2studio.models.dx import (
    CorrDiffTaiwan,
    DerivedSurfacePressure,
    DerivedWS,
    PrecipitationAFNOv2,
    SolarRadiationAFNO1H,
)
from earth2studio.models.px import FCN3, DiagnosticWrapper, Persistence
from earth2studio.utils import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


@pytest.fixture(autouse=True)
def optional_backend(request):
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")
    if request.node.name.startswith(
        (
            "test_dxwrapper_call",
            "test_dxwrapper_iter",
            "test_dxwrapper_run",
            "test_fcn3_conformance",
        )
    ):
        pytest.importorskip("makani")
        pytest.importorskip("physicsnemo.utils.zenith_angle")


def make_input(model, times, device="cpu"):
    signature = model.input_coords().rename(batch="time")
    signature.attrs["earth2studio_dynamic_dims"] = ("time",)
    signature = coord_array_like(signature, {"time": np.asarray(times)})
    return from_torch(torch.randn(signature.shape, device=device), signature)


class PhooFCN3Preprocessor(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.register_buffer(
            "state",
            torch.randn(
                10,
            ),
        )

    def set_internal_state(self, state: torch.Tensor):
        self.state = state.to(self.state.device)

    def get_internal_state(self, tensor=True):
        return self.state

    def update_internal_state(self, replace_state=True):
        self.state = torch.randn((10,), device=self.state.device)


class PhooFCN3Model(torch.nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor


class PhooFCN3ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._generator = None

    def forward(self, x, t, normalized_data: bool = False, replace_state: bool = False):
        # Deterministic (identity) unless set_rng() has seeded a local generator,
        # mirroring the real core model's noise-conditioned forward so that FCN3's
        # declared stochastic=True is exercisable by the conformance rollout rules.
        if self._generator is None:
            return x
        noise = torch.randn(x.shape, generator=self._generator).to(x.device)
        return x + noise

    def set_rng(self, reset: bool = True, seed: int = 333):
        if reset or self._generator is None:
            self._generator = torch.Generator().manual_seed(seed)


class PhooAFNOPrecipV2(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, :, :]


class PhooAFNOSolarRadiation(torch.nn.Module):
    """Mock model for testing."""

    def forward(self, x):
        # x: (batch, variables, lat, lon)
        # The model expects input shape (batch, variables, lat, lon)
        # where variables includes the input variables plus sza, sincos_latlon, orography, and landsea_mask
        # We'll return a tensor of the same shape but with only one variable
        return torch.zeros_like(x[:, :1, :, :])


class PhooCorrDiff(torch.nn.Module):
    img_out_channels = 4
    img_resolution = 448
    sigma_min = 0
    sigma_max = float("inf")

    def __init__(self):
        super().__init__()
        self.register_buffer("device_buffer", torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    @device.setter
    def device(self, value) -> None:
        dev = torch.device(value)
        self.device_buffer = torch.empty(0, device=dev)

    def forward(self, x, img_lr, class_labels=None, force_fp32=False, **model_kwargs):
        return x[:, :4]

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


@pytest.mark.parametrize("device", ["cuda:0"])  # Removing CPU here too slow atm "cpu",
@pytest.mark.parametrize("model_type", ["precip", "solar"])
@pytest.mark.parametrize(
    "times",
    [
        [np.datetime64("2025-08-21T00:00:00")],
        [np.datetime64("2025-08-21T00:00:00"), np.datetime64("2025-08-22T00:00:00")],
    ],
)
def test_dxwrapper_call(device, model_type, times):
    # Spoof models
    fcn3_model = PhooFCN3ModelWrapper(PhooFCN3Model(PhooFCN3Preprocessor()))
    px_model = FCN3(fcn3_model)

    if model_type == "precip":
        precipafnov2_model = PhooAFNOPrecipV2()
        center = torch.zeros(20, 1, 1)
        scale = torch.ones(20, 1, 1)
        landsea_mask = torch.zeros(1, 1, 720, 1440)
        orography = torch.zeros(1, 1, 720, 1440)

        dx_model = PrecipitationAFNOv2(
            precipafnov2_model, landsea_mask, orography, center, scale
        ).to(device)
    elif model_type == "solar":
        era5_mean = torch.zeros(24, 1, 1)
        era5_std = torch.ones(24, 1, 1)
        ssrd_mean = torch.zeros(1, 1, 1)
        ssrd_std = torch.ones(1, 1, 1)
        orography = torch.zeros(1, 1, 721, 1440)
        landsea_mask = torch.zeros(1, 1, 721, 1440)
        sincos_latlon = torch.zeros(1, 4, 721, 1440)

        dx_model = SolarRadiationAFNO1H(
            core_model=PhooAFNOSolarRadiation(),
            freq="1h",
            era5_mean=era5_mean,
            era5_std=era5_std,
            ssrd_mean=ssrd_mean,
            ssrd_std=ssrd_std,
            orography=orography,
            landsea_mask=landsea_mask,
            sincos_latlon=sincos_latlon,
        ).to(device)

    px_out_coords = px_model.output_coords(px_model.input_coords())
    sp_model = DerivedSurfacePressure(
        p_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        surface_geopotential=torch.zeros(721, 1440),
        surface_geopotential_coords=OrderedDict(
            {"lat": px_out_coords["lat"], "lon": px_out_coords["lon"]}
        ),
    )
    ws_model = DerivedWS(levels=["100m"])

    wrapped_model = DiagnosticWrapper(
        px_model=px_model, dx_model=[sp_model, ws_model]
    ).to(device=device)
    wrapped_model = DiagnosticWrapper(px_model=wrapped_model, dx_model=[dx_model]).to(
        device=device
    )

    field = make_input(wrapped_model, times, device)
    output = wrapped_model(field)
    handshake_dataarray(output, wrapped_model.output_coords(field))
    assert output.dims == ("time", "lead_time", "variable", "lat", "lon")


@pytest.mark.parametrize("device", ["cuda:0"])  # Removing CPU here too slow atm "cpu",
@pytest.mark.parametrize(
    "times,number_of_samples",
    [
        ([np.datetime64("2025-08-21T00:00:00")], 2),
        (
            [
                np.datetime64("2025-08-21T00:00:00"),
                np.datetime64("2025-08-22T00:00:00"),
            ],
            1,
        ),
    ],
)
def test_dxwrapper_iter(device, times, number_of_samples):
    # Spoof models
    model = PhooCorrDiff()
    in_center = torch.zeros(12, 1, 1)
    in_scale = torch.ones(12, 1, 1)
    out_center = torch.zeros(4, 1, 1)
    out_scale = torch.ones(4, 1, 1)
    lat = torch.as_tensor(np.linspace(19.5, 27, 450, endpoint=True))
    lon = torch.as_tensor(np.linspace(117, 125, 450, endpoint=False))
    out_lon, out_lat = torch.meshgrid(lon, lat)
    corrdiff_model = CorrDiffTaiwan(
        model,
        model,
        in_center,
        in_scale,
        out_center,
        out_scale,
        out_lat,
        out_lon,
        number_of_samples=number_of_samples,
    ).to(device)

    # Create persistence prognostic model
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 360, 1440, endpoint=False)
    domain_coords = OrderedDict({"lat": lat, "lon": lon})
    px_model = Persistence(
        variable=corrdiff_model.input_coords()["variable"].values.tolist(),
        domain_coords=domain_coords,
        dt=np.timedelta64(6, "h"),
    ).to(device)

    wrapped_model = DiagnosticWrapper(
        px_model=px_model,
        dx_model=corrdiff_model,
    ).to(device=device)

    x = make_input(wrapped_model, times, device)
    # Get generator
    p_iter = wrapped_model.create_iterator(x)
    xr.testing.assert_identical(next(p_iter), x)
    for i, out in enumerate(p_iter):
        expected = wrapped_model.output_coords(x)
        assert out.shape == expected.shape

        if i == 2:
            break


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize(
    "times,number_of_samples",
    [
        ([np.datetime64("2025-08-21T00:00:00")], 1),
    ],
)
def test_dxwrapper_run(device, times, number_of_samples):

    model = PhooCorrDiff()
    in_center = torch.zeros(12, 1, 1)
    in_scale = torch.ones(12, 1, 1)
    out_center = torch.zeros(4, 1, 1)
    out_scale = torch.ones(4, 1, 1)
    lat = torch.as_tensor(np.linspace(19.5, 27, 450, endpoint=True))
    lon = torch.as_tensor(np.linspace(117, 125, 450, endpoint=False))
    out_lon, out_lat = torch.meshgrid(lon, lat)
    corrdiff_model = CorrDiffTaiwan(
        model,
        model,
        in_center,
        in_scale,
        out_center,
        out_scale,
        out_lat,
        out_lon,
        number_of_samples=number_of_samples,
    ).to(device)

    # Create persistence prognostic model
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 360, 1440, endpoint=False)
    domain_coords = OrderedDict({"lat": lat, "lon": lon})
    px_model = Persistence(
        variable=corrdiff_model.input_coords()["variable"].values.tolist(),
        domain_coords=domain_coords,
        dt=np.timedelta64(6, "h"),
    ).to(device)

    wrapped_model = DiagnosticWrapper(
        px_model=px_model,
        dx_model=corrdiff_model,
    ).to(device=device)

    field = make_input(wrapped_model, times, device)
    iterator = wrapped_model.create_iterator(field)
    xr.testing.assert_identical(next(iterator), field)
    for step in range(1, 3):
        output = next(iterator)
        assert output.lead_time.values[0] == np.timedelta64(step * 6, "h")
    iterator.close()


def test_fcn3_conformance():
    """Check the mock FCN3 model against the Earth2Studio model contract.

    FCN3 declares stochastic=True and delegates set_rng to its core model. The
    Phoo core model seeds a local torch.Generator and adds noise from it once
    seeded, so P13 (reproducibility) and P14 (RNG isolation) are exercisable.

    Not conformant; same violations as
    test/models/px/test_fcn3.py::test_fcn3_conformance, which documents each.
    """
    fcn3_model = PhooFCN3ModelWrapper(PhooFCN3Model(PhooFCN3Preprocessor()))
    px_model = FCN3(fcn3_model)
    with pytest.raises(ContractException) as exc_info:
        check_prognostic_contract(px_model)
    assert {v.split(":")[0] for v in exc_info.value.violations} == {"P14"}


def test_persistence_conformance():
    """Check the Persistence prognostic model against the model contract.

    Persistence does not declare `stochastic`, so P14 is reported as an
    informational skip rather than a violation.
    """
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 360, 1440, endpoint=False)
    domain_coords = OrderedDict({"lat": lat, "lon": lon})
    px_model = Persistence(
        variable=["t2m", "u10m"],
        domain_coords=domain_coords,
        dt=np.timedelta64(6, "h"),
    )
    check_prognostic_contract(px_model)


def test_diagnosticwrapper_conformance(tmp_path):
    from earth2studio.grids import ProjectedGrid
    from earth2studio.models.px.dxwrapper import PrepareInputCoordsDefault

    projected = ProjectedGrid(
        np.array([1000.0, 2000.0]), np.array([3000.0, 4000.0, 5000.0]), "EPSG:3857"
    )
    diagnostic = DerivedWS(["10m"], grid=projected)
    source = Persistence(
        ["u10m", "v10m"], LatLonGrid(np.array([0.0, 1.0]), np.array([0.0, 1.0, 2.0]))
    )
    prepared = PrepareInputCoordsDefault()(
        source.output_coords(source.input_coords()), diagnostic.input_coords()
    )
    diagnostic.output_coords(prepared)
    assert prepared.attrs["earth2studio_crs"] == "EPSG:3857"
    np.testing.assert_array_equal(prepared.y, projected.y)
    np.testing.assert_array_equal(prepared.x, projected.x)
    wrapper = DiagnosticWrapper(source, diagnostic)
    field = make_input(wrapper, [np.datetime64("2024-01-01")])
    field.data[..., 0, :, :] = 3
    field.data[..., 1, :, :] = 4
    result = wrapper(field)
    handshake_dataarray(result, wrapper.output_coords(field))
    np.testing.assert_allclose(result.values, 5)
    assert result.attrs["earth2studio_crs"] == "EPSG:3857"
    grid = LatLonGrid(np.linspace(30, 40, 4), np.arange(6))
    px_model = Persistence(["u10m", "v10m", "u100m", "v100m"], grid, history=2)
    inner = DiagnosticWrapper(px_model, DerivedWS(["10m"], grid=grid))
    wrapped_model = DiagnosticWrapper(inner, DerivedWS(["100m"], grid=grid))
    check_prognostic_contract(wrapped_model)
    field = make_input(wrapped_model, [np.datetime64("2024-01-01")]).expand_dims(
        member=["a", "b"]
    )
    field.name = "weather"
    field.attrs["nested"] = {"owner": ["caller"]}
    field.encoding = {"nested": {"owner": ["caller"]}}
    field = field.assign_coords(aux=("member", [1, 2]), units=("variable", ["m/s"] * 4))
    before = field.copy(deep=True)
    calls = []

    def front(x):
        calls.append(("front", x.dims))
        x.attrs["nested"]["owner"].append("hook")
        return x.drop_vars("aux", errors="ignore")

    def rear(x):
        calls.append(("rear", x.dims))
        return x

    wrapped_model.front_hook = front
    wrapped_model.rear_hook = rear
    output = wrapped_model(field)
    assert calls == []
    handshake_dataarray(output, wrapped_model.output_coords(field))
    iterator = wrapped_model.create_iterator(field)
    initial = next(iterator)
    assert calls == []
    first = next(iterator)
    assert [c[0] for c in calls] == ["front", "rear"]
    assert calls[0][1] == field.dims
    assert "aux" not in first.coords
    assert "units" not in first.coords
    assert first.name == field.name and first.encoding == field.encoding
    frozen = first.copy(deep=True)
    next(iterator)
    xr.testing.assert_identical(first, frozen)
    xr.testing.assert_identical(field, before)
    xr.testing.assert_identical(initial, before.isel(lead_time=slice(-1, None)))
    iterator.close()
    wrapped_model.clear_hooks()

    class SamplingWind(DerivedWS):
        def __call__(self, x):
            result = super().__call__(x)
            result.data += torch.randn(result.shape).numpy()
            return result

    sampled = DiagnosticWrapper(px_model, SamplingWind(["10m"], grid=grid))
    with pytest.raises(ContractException) as exc_info:
        check_prognostic_contract(sampled)
    assert exc_info.value.violations == [
        "P13: repeated runs with the same input and seed disagree"
    ]

    from earth2studio.utils.checkpoint import Checkpoint

    checkpoint = Checkpoint("wrapper", path=tmp_path, mode="append", level=2)
    field = field.assign_coords(member=np.arange(field.sizes["member"])).copy(deep=True)
    with checkpoint as ckpt:
        base = Persistence(["u10m", "v10m", "u100m", "v100m"], grid, history=2)
        iterator = base.create_iterator(field)
        next(iterator)
        saved = next(iterator)
        ckpt.write(lead_time=saved.lead_time.values[-1])
        ckpt.flush()
        iterator.close()
    with checkpoint.select(-1):
        base = Persistence(["u10m", "v10m", "u100m", "v100m"], grid, history=2)
        wrapper = DiagnosticWrapper(
            DiagnosticWrapper(base, DerivedWS(["10m"], grid=grid)),
            DerivedWS(["100m"], grid=grid),
        )
        calls.clear()
        wrapper.front_hook, wrapper.rear_hook = front, rear
        iterator = wrapper.create_iterator(field)
        resumed = next(iterator)
        assert resumed.lead_time.values[0] == np.timedelta64(12, "h")
        assert {"ws10m", "ws100m"}.issubset(resumed.coords["variable"].values)
        assert [c[0] for c in calls] == ["front", "rear"]
        assert "aux" not in resumed.coords
        assert next(iterator).lead_time.values[0] == np.timedelta64(18, "h")
        iterator.close()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output1(device):
    """Test strategy 1: Direct concatenation when all coords match"""
    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    # Create matching coordinate systems
    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m", "u10m"])),
            ("lat", np.linspace(-90, 90, 10)),
            ("lon", np.linspace(0, 360, 20)),
        ]
    )

    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["tp:sum:6h"])),
                ("lat", np.linspace(-90, 90, 10)),
                ("lon", np.linspace(0, 360, 20)),
            ]
        )
    ]

    # Create tensors
    px_x = torch.randn(1, 2, 10, 20, device=device)
    dx_x = [torch.randn(1, 1, 10, 20, device=device)]

    px_field = from_torch(px_x, coord_array(tuple(px_coords), px_coords))
    dx_fields = [
        from_torch(t, coord_array(tuple(c), c)) for t, c in zip(dx_x, dx_coords)
    ]
    output = prepare_output(px_field, dx_fields)
    x_out, _ = output.e2s.to_torch()
    coords_out = output.coords

    # Verify shape and concatenation
    assert x_out.shape == (1, 3, 10, 20)
    assert list(coords_out["variable"]) == ["t2m", "u10m", "tp:sum:6h"]
    assert set(output.attrs["earth2studio_statistics"]) == {"tp:sum:6h"}
    assert x_out.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output2(device):
    """Test strategy 2: Subregion extraction when dx is a lat/lon subregion of px"""
    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    # Create px coords with larger spatial domain
    lat_px = np.linspace(-90, 90, 20)
    lon_px = np.linspace(0, 360, 40)
    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m", "u10m"])),
            ("lat", lat_px),
            ("lon", lon_px),
        ]
    )

    # Create dx coords with subregion (middle section)
    lat_dx = lat_px[5:15]  # Contiguous subregion
    lon_dx = lon_px[10:30]  # Contiguous subregion
    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", lat_dx),
                ("lon", lon_dx),
            ]
        )
    ]

    # Create tensors
    px_x = torch.randn(1, 2, 20, 40, device=device)
    dx_x = [torch.randn(1, 1, 10, 20, device=device)]

    px_field = from_torch(px_x, coord_array(tuple(px_coords), px_coords))
    dx_fields = [
        from_torch(t, coord_array(tuple(c), c)) for t, c in zip(dx_x, dx_coords)
    ]
    output = prepare_output(px_field, dx_fields)
    x_out, _ = output.e2s.to_torch()
    coords_out = output.coords

    # Verify shape and concatenation
    assert x_out.shape == (1, 3, 10, 20)
    assert list(coords_out["variable"]) == ["t2m", "u10m", "precip"]

    # Verify that the sliced region was extracted correctly
    # The first 2 variables should match the subregion of px_x
    expected_slice = px_x[:, :, 5:15, 10:30]
    assert torch.allclose(x_out[:, :2, :, :], expected_slice)
    assert x_out.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output3(device):

    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    # Create incompatible coordinate systems (different time dimension)
    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m", "u10m"])),
            ("lat", np.linspace(-90, 90, 10)),
            ("lon", np.linspace(0, 360, 20)),
        ]
    )

    dx_coords = [
        OrderedDict(
            [
                (
                    "time",
                    np.array(
                        [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
                    ),
                ),
                ("variable", np.array(["precip"])),
                ("lat", np.linspace(-90, 90, 10)),
                ("lon", np.linspace(0, 360, 20)),
            ]
        )
    ]

    # Create tensors
    px_x = torch.randn(1, 2, 10, 20, device=device)
    dx_x = [torch.randn(2, 1, 10, 20, device=device)]

    # Test forward pass
    px_field = from_torch(px_x, coord_array(tuple(px_coords), px_coords))
    dx_fields = [
        from_torch(t, coord_array(tuple(c), c)) for t, c in zip(dx_x, dx_coords)
    ]
    output = prepare_output(px_field, dx_fields)
    x_out, _ = output.e2s.to_torch()
    coords_out = output.coords

    # Verify only dx outputs are used
    assert x_out.shape == (2, 1, 10, 20)
    assert list(coords_out["variable"]) == ["precip"]
    assert x_out.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output_subregion(device):
    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    # Create px coords
    lat_px = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    lon_px = np.linspace(0, 360, 40)
    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m"])),
            ("lat", lat_px),
            ("lon", lon_px),
        ]
    )

    # Create dx coords with non-contiguous lat indices (skip some values)
    lat_dx = np.array([0, 20, 40, 60, 80])  # Non-contiguous in the array
    lon_dx = lon_px[10:30]  # Contiguous
    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", lat_dx),
                ("lon", lon_dx),
            ]
        )
    ]
    px_x = torch.randn(1, 1, 10, 40, device=device)
    dx_x = [torch.randn(1, 1, 5, 20, device=device)]

    px_field = from_torch(px_x, coord_array(tuple(px_coords), px_coords))
    dx_fields = [
        from_torch(t, coord_array(tuple(c), c)) for t, c in zip(dx_x, dx_coords)
    ]
    output = prepare_output(px_field, dx_fields)
    x_out, _ = output.e2s.to_torch()
    coords_out = output.coords

    assert x_out.shape == (1, 1, 5, 20)
    assert list(coords_out["variable"]) == ["precip"]
    assert torch.equal(x_out, dx_x[0])
    assert x_out.device.type == device.split(":")[0]

    # Create a dx with a lat lon domain thats out of bounds of prognostic
    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", np.linspace(-10, 80, 8)),
                ("lon", np.linspace(200, 360, 10)),
            ]
        )
    ]
    px_x = torch.randn(1, 1, 10, 40, device=device)
    dx_x = [torch.randn(1, 1, 8, 10, device=device)]

    px_field = from_torch(px_x, coord_array(tuple(px_coords), px_coords))
    dx_fields = [
        from_torch(t, coord_array(tuple(c), c)) for t, c in zip(dx_x, dx_coords)
    ]
    output = prepare_output(px_field, dx_fields)
    x_out, _ = output.e2s.to_torch()
    coords_out = output.coords

    assert x_out.shape == (1, 1, 8, 10)
    assert list(coords_out["variable"]) == ["precip"]
    assert torch.equal(x_out, dx_x[0])
    assert x_out.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output_tensor_multiple_dx_models(device):
    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m", "u10m"])),
            ("lat", np.linspace(-90, 90, 10)),
            ("lon", np.linspace(0, 360, 20)),
        ]
    )

    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", np.linspace(-90, 90, 10)),
                ("lon", np.linspace(0, 360, 20)),
            ]
        ),
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["solar"])),
                ("lat", np.linspace(-90, 90, 10)),
                ("lon", np.linspace(0, 360, 20)),
            ]
        ),
    ]
    px_x = torch.randn(1, 2, 10, 20, device=device)
    dx_x = [
        torch.randn(1, 1, 10, 20, device=device),
        torch.randn(1, 1, 10, 20, device=device),
    ]

    px_field = from_torch(px_x, coord_array(tuple(px_coords), px_coords))
    dx_fields = [
        from_torch(t, coord_array(tuple(c), c)) for t, c in zip(dx_x, dx_coords)
    ]
    output = prepare_output(px_field, dx_fields)
    x_out, _ = output.e2s.to_torch()
    coords_out = output.coords

    # Verify shape and concatenation with multiple dx models
    assert x_out.shape == (1, 4, 10, 20)
    assert list(coords_out["variable"]) == ["t2m", "u10m", "precip", "solar"]
    assert x_out.device.type == device.split(":")[0]

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


import numpy as np
import pytest
import torch
import xarray as xr
from test_dlesym import (
    CoupledAtmosModel,
    CoupledOceanModel,
    HEALPixFill,
    _assert_coupled_reproduces_fused,
    _coupled,
    _streams,
    build_components,
    hours,
    mock_checkpoints,
    write_package,
)
from test_dlesym import optional_backend as grid_backend  # noqa: F401

import earth2studio.models.px.dlesym as dlesym_src
import earth2studio.models.px.dlesym_v0_isccp_era5 as v0_src
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import (
    DLESyMv0_ISCCP_ERA5,
    DLESyMv0_ISCCP_ERA5Atmos,
    DLESyMv0_ISCCP_ERA5Ocean,
)
from earth2studio.run import WorkItem
from earth2studio.utils.coords import coord_array_like, handshake_metadata
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import OptionalDependencyFailure


@pytest.fixture(autouse=True)
def optional_backend(monkeypatch, request):
    request.getfixturevalue("grid_backend")
    if v0_src.__file__ in OptionalDependencyFailure.failures:
        for module in (dlesym_src, v0_src):
            monkeypatch.delitem(
                OptionalDependencyFailure.failures, module.__file__, raising=False
            )
        monkeypatch.setattr(
            dlesym_src,
            "insolation",
            lambda times, lat, lon: np.zeros(
                (len(times), *lat.shape), dtype=np.float32
            ),
        )


# Upstream-DLESyM variable layout (model space). The wrapper swaps ``rlut`` ->
# ``ttr`` in input_coords when use_ttr=True.
_ATMOS_VARIABLES = [
    "z500",
    "tau300-700",
    "z1000",
    "t2m",
    "tcwv",
    "t850",
    "z250",
    "rlut",
    "ws10m",
]
_OCEAN_VARIABLES = ["sst"]
_ATMOS_COUPLING_VARIABLES = ["sst"]
_OCEAN_COUPLING_VARIABLES = ["z1000", "ws10m", "rlut"]

# The atmos model is a HEALPixRecUNet with presteps=1 and input_time_dim=2, so
# it consumes (presteps + 1) * input_time_dim = 4 history timesteps (the first
# two initialize the recurrent hidden state). The ocean model is a plain
# HEALPixUNet (presteps=0) and consumes only its input_time_dim window.
_ATMOS_INPUT_TIMES = np.array([-18, -12, -6, 0], dtype="timedelta64[h]")
_OCEAN_INPUT_TIMES = np.array([-48, 0], dtype="timedelta64[h]")

_ATMOS_OUTPUT_TIMES = np.array(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],
    dtype="timedelta64[h]",
)
_OCEAN_OUTPUT_TIMES = np.array([48, 96], dtype="timedelta64[h]")


class PhooAtmosModel(torch.nn.Module):
    """Mock atmosphere model. Output ones; shape mirrors HEALPixRecUNet."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = len(_ATMOS_OUTPUT_TIMES)
        self.input_time_dim = len(_ATMOS_INPUT_TIMES)

    def forward(self, in_list):
        x = in_list[0]
        b, t = x.shape[:2]
        return torch.ones(b, t, self.output_time_dim, *x.shape[3:], device=x.device)


class PhooOceanModel(torch.nn.Module):
    """Mock ocean model."""

    def __init__(self):
        super().__init__()
        self.output_time_dim = len(_OCEAN_OUTPUT_TIMES)
        self.input_time_dim = len(_OCEAN_INPUT_TIMES)

    def forward(self, in_list):
        x = in_list[0]
        b, t = x.shape[:2]
        return torch.ones(b, t, self.output_time_dim, *x.shape[3:], device=x.device)


def _build_climatology(nside: int, n_doy: int = 366) -> dict:
    """Return synthetic per-doy climatology arrays of the right shape."""
    rng = np.random.default_rng(0)
    return {
        "ttr_clim_mean": rng.standard_normal((n_doy, 12, nside, nside)).astype(
            "float32"
        ),
        "ttr_clim_std": np.ones((n_doy, 12, nside, nside), dtype="float32"),
        "olr_clim_mean": rng.standard_normal((n_doy, 12, nside, nside)).astype(
            "float32"
        ),
        "olr_clim_std": np.ones((n_doy, 12, nside, nside), dtype="float32"),
    }


def _build_model(
    device,
    nside: int = 16,
    use_ttr: bool = True,
    atmos_model=None,
    ocean_model=None,
    scaled: bool = False,
) -> DLESyMv0_ISCCP_ERA5:
    """Build DLESyMv0_ISCCP_ERA5 with mock components. Uses nside=16 for fast tests."""
    n_vars = len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES)
    # A local generator, not the global np.random state: drawing from the global
    # generator made whether the conformance checker's probes observed a value
    # change depend on whatever earlier test in the suite last touched np.random,
    # i.e. test order.
    rng = np.random.default_rng(0)
    hpx_lat = rng.standard_normal((12, nside, nside))
    hpx_lon = rng.standard_normal((12, nside, nside))
    center = np.zeros((1, 1, 1, n_vars, 1, 1, 1))
    scale = np.ones((1, 1, 1, n_vars, 1, 1, 1))
    if scaled:
        center = rng.uniform(-1.0, 1.0, center.shape)
        scale = rng.uniform(0.5, 2.0, scale.shape)
    atmos_constants = rng.standard_normal((12, 2, nside, nside))
    ocean_constants = rng.standard_normal((12, 2, nside, nside))

    clim = _build_climatology(nside) if use_ttr else {}

    model = DLESyMv0_ISCCP_ERA5(
        atmos_model=atmos_model if atmos_model is not None else PhooAtmosModel(),
        ocean_model=ocean_model if ocean_model is not None else PhooOceanModel(),
        hpx_lat=hpx_lat,
        hpx_lon=hpx_lon,
        nside=nside,
        center=center,
        scale=scale,
        atmos_constants=atmos_constants,
        ocean_constants=ocean_constants,
        atmos_input_times=_ATMOS_INPUT_TIMES,
        ocean_input_times=_OCEAN_INPUT_TIMES,
        atmos_output_times=_ATMOS_OUTPUT_TIMES,
        ocean_output_times=_OCEAN_OUTPUT_TIMES,
        atmos_variables=_ATMOS_VARIABLES,
        ocean_variables=_OCEAN_VARIABLES,
        atmos_coupling_variables=_ATMOS_COUPLING_VARIABLES,
        ocean_coupling_variables=_OCEAN_COUPLING_VARIABLES,
        use_ttr=use_ttr,
        **clim,
    ).to(device)
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("use_ttr", [True, False])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_dlesym_v0_isccp_era5_forward(device, use_ttr, batch_size):
    """Forward pass with mock atmos/ocean models, both with and without TTR."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 16
    model = _build_model(device, nside=nside, use_ttr=use_ttr)

    # When use_ttr=True the wrapper advertises ``ttr`` in input_coords.
    in_coords = model.input_coords()
    expected_input_var = "ttr" if use_ttr else "rlut"
    assert expected_input_var in list(in_coords["variable"])

    time = np.array([np.datetime64("2020-01-01T00:00")])
    x = torch.randn(
        batch_size,
        len(time),
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        12,
        nside,
        nside,
        device=device,
    )

    in_coords = coord_array_like(
        in_coords, {"batch": np.arange(batch_size), "time": time}
    )

    field = from_torch(x, in_coords)
    with pytest.raises(ValueError, match="metadata"):
        handshake_metadata(
            field.assign_attrs(origin="south", clockwise=False),
            in_coords,
            ("origin", "clockwise"),
        )
    out = model(field)
    out_coords = out.coords
    expected_coords = model.output_coords(in_coords)

    # Output should always be in model variable space (rlut), regardless of
    # whether the input was provided as ttr.
    assert "rlut" in list(out_coords["variable"])
    assert "ttr" not in list(out_coords["variable"])
    assert out.shape == (
        batch_size,
        len(time),
        len(_ATMOS_OUTPUT_TIMES),
        len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES),
        12,
        nside,
        nside,
    )
    for key in out_coords:
        np.testing.assert_array_equal(out_coords[key], expected_coords[key])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_dlesym_v0_isccp_era5_ttr_transform_changes_values(device):
    """The TTR->OLR transform must actually run when use_ttr=True."""
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 16
    model = _build_model(device, nside=nside, use_ttr=True)

    in_coords = model.input_coords()
    in_coords = coord_array_like(
        in_coords,
        {"batch": np.array([0]), "time": np.array([np.datetime64("2020-07-15T00:00")])},
    )

    x_input = torch.zeros(
        1,
        1,
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        12,
        nside,
        nside,
        device=device,
    )
    x_transformed = model._apply_ttr_to_olr(
        x_input, {k: v.values for k, v in in_coords.coords.items()}
    )

    # Find the ttr channel and confirm the transform pulled values away from zero.
    ttr_idx = list(in_coords["variable"]).index("ttr")
    transformed_channel = x_transformed[:, :, :, ttr_idx]
    other_channels = torch.cat(
        [
            x_transformed[:, :, :, :ttr_idx],
            x_transformed[:, :, :, ttr_idx + 1 :],
        ],
        dim=3,
    )
    assert not torch.allclose(
        transformed_channel, torch.zeros_like(transformed_channel)
    )
    assert torch.allclose(other_channels, torch.zeros_like(other_channels))
    # Climate years must retain the same day-of-year lookup without ns overflow.
    climate_coords = {k: v.values for k, v in in_coords.coords.items()}
    climate_coords["time"] = np.array(["3020-07-15T00:00:00"], dtype="datetime64[s]")
    torch.testing.assert_close(
        model._apply_ttr_to_olr(x_input, climate_coords), x_transformed
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("use_ttr", [True, False])
def test_dlesym_v0_isccp_era5_iterator(device, batch_size, use_ttr, monkeypatch):
    if device == "cuda:0" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    nside = 16
    model = _build_model(device, nside=nside, use_ttr=use_ttr)
    conversions = []
    transform = model._apply_ttr_to_olr

    def record_conversion(tensor, coords):
        conversions.append(True)
        return transform(tensor, coords)

    monkeypatch.setattr(model, "_apply_ttr_to_olr", record_conversion)
    in_coords = model.input_coords()

    time = np.array([np.datetime64("2020-01-01T00:00")])
    x = torch.randn(
        batch_size,
        len(time),
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        12,
        nside,
        nside,
        device=device,
    )
    in_coords = coord_array_like(
        in_coords, {"batch": np.arange(batch_size), "time": time}
    )

    field = from_torch(x, in_coords)
    iterator = model.create_iterator(field)

    coupler_step = _ATMOS_OUTPUT_TIMES[-1]

    for i in range(2):
        out = next(iterator)
        coords = out.coords
        assert out.shape == (
            batch_size,
            len(time),
            len(_ATMOS_OUTPUT_TIMES),
            len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES),
            12,
            nside,
            nside,
        )
        assert np.all(coords["lead_time"] == _ATMOS_OUTPUT_TIMES + coupler_step * i)
        assert "rlut" in coords["variable"].values
        assert "ttr" not in coords["variable"].values
        assert len(conversions) == int(use_ttr)
        assert model.use_ttr is use_ttr


def test_dlesym_v0_isccp_era5_missing_clim_raises():
    """Direct construction without climatology should raise."""
    nside = 16
    n_vars = len(_ATMOS_VARIABLES) + len(_OCEAN_VARIABLES)
    hpx_lat = np.random.randn(12, nside, nside)
    hpx_lon = np.random.randn(12, nside, nside)
    center = np.zeros((1, 1, 1, n_vars, 1, 1, 1))
    scale = np.ones((1, 1, 1, n_vars, 1, 1, 1))
    atmos_constants = np.random.randn(12, 2, nside, nside)
    ocean_constants = np.random.randn(12, 2, nside, nside)

    with pytest.raises(ValueError, match="climatology"):
        DLESyMv0_ISCCP_ERA5(
            atmos_model=PhooAtmosModel(),
            ocean_model=PhooOceanModel(),
            hpx_lat=hpx_lat,
            hpx_lon=hpx_lon,
            nside=nside,
            center=center,
            scale=scale,
            atmos_constants=atmos_constants,
            ocean_constants=ocean_constants,
            atmos_input_times=_ATMOS_INPUT_TIMES,
            ocean_input_times=_OCEAN_INPUT_TIMES,
            atmos_output_times=_ATMOS_OUTPUT_TIMES,
            ocean_output_times=_OCEAN_OUTPUT_TIMES,
            atmos_variables=_ATMOS_VARIABLES,
            ocean_variables=_OCEAN_VARIABLES,
            atmos_coupling_variables=_ATMOS_COUPLING_VARIABLES,
            ocean_coupling_variables=_OCEAN_COUPLING_VARIABLES,
            use_ttr=True,
            # climatology omitted
        )


def test_dlesym_v0_isccp_era5_missing_rlut_raises():
    """Atmos variables must contain rlut (model space name)."""
    nside = 16
    bad_atmos = list(_ATMOS_VARIABLES)
    bad_atmos[bad_atmos.index("rlut")] = "ttr"  # wrong; should be model space rlut
    n_vars = len(bad_atmos) + len(_OCEAN_VARIABLES)

    with pytest.raises(ValueError, match="rlut"):
        DLESyMv0_ISCCP_ERA5(
            atmos_model=PhooAtmosModel(),
            ocean_model=PhooOceanModel(),
            hpx_lat=np.random.randn(12, nside, nside),
            hpx_lon=np.random.randn(12, nside, nside),
            nside=nside,
            center=np.zeros((1, 1, 1, n_vars, 1, 1, 1)),
            scale=np.ones((1, 1, 1, n_vars, 1, 1, 1)),
            atmos_constants=np.random.randn(12, 2, nside, nside),
            ocean_constants=np.random.randn(12, 2, nside, nside),
            atmos_input_times=_ATMOS_INPUT_TIMES,
            ocean_input_times=_OCEAN_INPUT_TIMES,
            atmos_output_times=_ATMOS_OUTPUT_TIMES,
            ocean_output_times=_OCEAN_OUTPUT_TIMES,
            atmos_variables=bad_atmos,
            ocean_variables=_OCEAN_VARIABLES,
            atmos_coupling_variables=_ATMOS_COUPLING_VARIABLES,
            ocean_coupling_variables=_OCEAN_COUPLING_VARIABLES,
            use_ttr=False,
        )


def test_dlesym_v0_isccp_era5_conformance():
    model = _build_model("cpu", nside=8, use_ttr=True)
    check_prognostic_contract(model)


def _coupled_model(use_ttr: bool = True) -> DLESyMv0_ISCCP_ERA5:
    return _build_model(
        "cpu",
        nside=8,
        use_ttr=use_ttr,
        atmos_model=CoupledAtmosModel(),
        ocean_model=CoupledOceanModel(),
        scaled=True,
    )


def _components(model: DLESyMv0_ISCCP_ERA5):
    """Standalone components equivalent to a fused mock ``model``."""
    climatology = {}
    if model.use_ttr:
        climatology = {
            name: getattr(model, name).cpu().numpy()
            for name in (
                "ttr_clim_mean",
                "ttr_clim_std",
                "olr_clim_mean",
                "olr_clim_std",
            )
        }
    return build_components(
        model,
        atmos_cls=DLESyMv0_ISCCP_ERA5Atmos,
        ocean_cls=DLESyMv0_ISCCP_ERA5Ocean,
        atmos_extra={"use_ttr": model.use_ttr, **climatology},
    )


@pytest.mark.parametrize("use_ttr", [True, False])
def test_dlesym_v0_isccp_era5_component_declarations(use_ttr):
    atmos, ocean = _components(_coupled_model(use_ttr))
    radiation = "ttr" if use_ttr else "rlut"
    # The atmosphere accepts ERA5 ttr in its history and reports OLR.
    assert radiation in atmos.input_coords()["variable"].values
    outputs = atmos.output_coords(atmos.input_coords())["variable"].values
    assert "rlut" in outputs and "ttr" not in outputs
    assert list(atmos.forcing_coords()["variable"].values) == ["sst"]
    # The ocean is also forced with the window mean of the atmosphere's OLR.
    assert list(ocean.forcing_coords()["variable"].values) == [
        "z1000:mean:48h",
        "ws10m:mean:48h",
        "rlut:mean:48h",
    ]
    assert hours(ocean.output_coords(ocean.input_coords())) == [48, 96]


@pytest.mark.parametrize("use_ttr", [True, False])
def test_dlesym_v0_isccp_era5_coupled_reproduces_fused(use_ttr):
    model = _coupled_model(use_ttr)
    variables = list(model.input_coords()["variable"].values)
    variables.remove("sst")
    _assert_coupled_reproduces_fused(
        model, _components(model), HEALPixFill(model), variables
    )


def test_dlesym_v0_isccp_era5_component_conformance():
    atmos, ocean = _components(_build_model("cpu", nside=8, use_ttr=True))
    check_prognostic_contract(atmos)
    check_prognostic_contract(ocean)


def test_dlesym_v0_isccp_era5_atmos_validation():
    model = _build_model("cpu", nside=8, use_ttr=False)
    with pytest.raises(ValueError, match="climatology"):
        build_components(
            model,
            atmos_cls=DLESyMv0_ISCCP_ERA5Atmos,
            ocean_cls=DLESyMv0_ISCCP_ERA5Ocean,
            atmos_extra={"use_ttr": True},
        )
    model.atmos_variables = [v for v in model.atmos_variables if v != "rlut"]
    with pytest.raises(ValueError, match="rlut"):
        DLESyMv0_ISCCP_ERA5Atmos(
            CoupledAtmosModel(),
            hpx_lat=model.hpx_lat,
            hpx_lon=model.hpx_lon,
            nside=8,
            center=np.zeros(8),
            scale=np.ones(8),
            coupling_center=np.zeros(1),
            coupling_scale=np.ones(1),
            constants=np.zeros((12, 2, 8, 8)),
            input_times=_ATMOS_INPUT_TIMES,
            output_times=_ATMOS_OUTPUT_TIMES,
            variables=model.atmos_variables,
            coupling_variables=["sst"],
            use_ttr=False,
        )


_PACKAGE_CONFIG = """
io:
  atmos_variables: [z500, tau300-700, z1000, t2m, tcwv, t850, z250, ws10m, rlut]
  ocean_variables: [sst]
  atmos_coupling_variables: [sst]
  ocean_coupling_variables: [z1000, ws10m, rlut]
  atmos_input_times: [-18, -12, -6, 0]
  ocean_input_times: [-48, 0]
  atmos_output_times: [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96]
  ocean_output_times: [48, 96]
  atmos_constants: [land_sea_mask, topography]
  ocean_constants: [land_sea_mask]
models:
  atmos_model_checkpoints: [atmos_model_0.mdlus]
  ocean_model_checkpoints: [ocean_model_0.mdlus]
data:
  nside: 8
  scaling:
    z500: {mean: 0.5, std: 1.5}
    tau300-700: {mean: -0.5, std: 0.5}
    z1000: {mean: 0.25, std: 2.0}
    t2m: {mean: 1.0, std: 0.75}
    tcwv: {mean: -1.0, std: 1.25}
    t850: {mean: 0.1, std: 0.9}
    z250: {mean: -0.2, std: 1.1}
    ws10m: {mean: 0.3, std: 0.6}
    rlut: {mean: -0.7, std: 1.7}
    sst: {mean: 0.8, std: 1.3}
"""


def _write_climatology(root, nside: int = 8, n_doy: int = 3) -> None:
    """Write the per-doy climatology files of the TTR -> OLR transform."""
    rng = np.random.default_rng(2)
    dims = ("dayofyear", "face", "height", "width")
    shape = (n_doy, 12, nside, nside)
    for file, (mean, std) in (
        (v0_src._TTR_CLIM_FILE, v0_src._TTR_CLIM_VARS),
        (v0_src._OLR_CLIM_FILE, v0_src._OLR_CLIM_VARS),
    ):
        xr.Dataset(
            {
                mean: (dims, rng.standard_normal(shape).astype("float32")),
                std: (dims, np.ones(shape, dtype="float32")),
            },
            coords={"dayofyear": np.arange(1, n_doy + 1)},
        ).to_netcdf(root / file)


@pytest.mark.parametrize("use_ttr", [True, False])
def test_dlesym_v0_isccp_era5_components_load_from_package(
    tmp_path, monkeypatch, use_ttr
):
    """Components load their own core from the fused model's unversioned package."""
    package = write_package(tmp_path, _PACKAGE_CONFIG)
    _write_climatology(tmp_path)
    loaded = mock_checkpoints(monkeypatch)

    model = DLESyMv0_ISCCP_ERA5.load_model(package, use_ttr=use_ttr)
    assert [p.removeprefix(str(tmp_path) + "/") for p in loaded] == [
        "atmos_model_0.mdlus",
        "ocean_model_0.mdlus",
    ]
    loaded.clear()
    package.resolved.clear()
    ocean = DLESyMv0_ISCCP_ERA5Ocean.load_model(package)
    assert [p.removeprefix(str(tmp_path) + "/") for p in loaded] == [
        "ocean_model_0.mdlus"
    ]
    # The ocean needs neither the atmosphere's constants nor the climatology.
    assert not {"topography.npy", v0_src._TTR_CLIM_FILE} & set(package.resolved)
    np.testing.assert_allclose(ocean.coupling_center.numpy(), [0.25, 0.3, -0.7])
    np.testing.assert_allclose(ocean.coupling_scale.numpy(), [2.0, 0.6, 1.7])

    atmos = DLESyMv0_ISCCP_ERA5Atmos.load_model(package, use_ttr=use_ttr)
    assert atmos.use_ttr is use_ttr
    assert isinstance(atmos, DLESyMv0_ISCCP_ERA5Atmos)
    assert isinstance(ocean, DLESyMv0_ISCCP_ERA5Ocean)
    if use_ttr:
        torch.testing.assert_close(atmos.ttr_clim_mean, model.ttr_clim_mean)

    variables = list(model.input_coords()["variable"].values)
    variables.remove("sst")
    _assert_coupled_reproduces_fused(
        model, (atmos, ocean), HEALPixFill(model), variables
    )


def test_dlesym_v0_isccp_era5_package_checks_history(tmp_path, monkeypatch):
    """A recurrent core needs (presteps + 1) * input_time_dim history steps."""
    package = write_package(tmp_path, _PACKAGE_CONFIG)

    class Recurrent(CoupledAtmosModel):
        presteps = 1

    mock_checkpoints(monkeypatch, atmos_core=Recurrent)
    with pytest.raises(ValueError, match="requires 8 history timesteps"):
        DLESyMv0_ISCCP_ERA5Atmos.load_model(package, use_ttr=False)
    with pytest.raises(ValueError, match="requires 8 history timesteps"):
        DLESyMv0_ISCCP_ERA5.load_model(package, use_ttr=False)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_dlesym_v0_isccp_era5_package(device):
    """Load the real HuggingFace package and run one coupled forward step on HEALPix."""
    torch.cuda.empty_cache()
    package = DLESyMv0_ISCCP_ERA5.load_default_package()
    model = DLESyMv0_ISCCP_ERA5.load_model(package).to(device)

    nside = model.nside
    batch_size = 1
    time = np.array([np.datetime64("2020-01-01T00:00")])
    in_coords = model.input_coords()
    in_coords = coord_array_like(
        in_coords, {"batch": np.arange(batch_size), "time": time}
    )

    x = torch.randn(
        batch_size,
        len(time),
        len(in_coords["lead_time"]),
        len(in_coords["variable"]),
        12,
        nside,
        nside,
        device=device,
    )

    out = model(from_torch(x, in_coords))
    out_coords = out.coords
    expected_coords = model.output_coords(in_coords)

    n_vars = len(in_coords["variable"])
    assert out.shape == (
        batch_size,
        len(time),
        len(_ATMOS_OUTPUT_TIMES),
        n_vars,
        12,
        nside,
        nside,
    )
    assert "rlut" in list(out_coords["variable"])
    assert "ttr" not in list(out_coords["variable"])
    for key in out_coords:
        np.testing.assert_array_equal(out_coords[key], expected_coords[key])


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_dlesym_v0_isccp_era5_component_package(device):
    """Load the packaged components and run one coupled cycle on HEALPix."""
    torch.cuda.empty_cache()
    package = DLESyMv0_ISCCP_ERA5Atmos.load_default_package()
    atmos = DLESyMv0_ISCCP_ERA5Atmos.load_model(package).to(device)
    ocean = DLESyMv0_ISCCP_ERA5Ocean.load_model(package).to(device)

    class Inputs:
        """The variables both components fetch, on their grid."""

        nside = atmos.nside

        @staticmethod
        def input_coords():
            variables = list(atmos.input_coords()["variable"].values)
            return coord_array_like(
                atmos.input_coords(),
                {"variable": np.array(variables + ocean.variables)},
            )

    streams = _streams(
        _coupled((atmos, ocean), HEALPixFill(Inputs)),
        WorkItem(np.datetime64("2020-06-01T00"), np.timedelta64(96, "h")),
    )
    assert hours(streams["atmos"][1]) == list(range(6, 97, 6))
    assert hours(streams["ocean"][1]) == [48, 96]
    variables = list(streams["atmos"][1]["variable"].values)
    assert "rlut" in variables and "ttr" not in variables
    assert not np.isnan(streams["ocean"][1].values).any()

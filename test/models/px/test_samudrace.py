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

import dataclasses
import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.px.samudrace import SamudrACE
from earth2studio.utils import handshake_dim

pytest.importorskip("fme")

# Tiny coupled test configuration: 6 hour atmosphere steps, 18 hour ocean
# steps (3 inner atmosphere steps per coupled cycle), on a 6 x 8 grid with
# south-to-north model latitudes (so the wrapper's internal flip to the
# public north-to-south convention is exercised).
N_LAT = 6
N_LON = 8
MODEL_LAT = torch.linspace(-75.0, 75.0, N_LAT)
MODEL_LON = torch.arange(0.0, 360.0, 360.0 / N_LON)
N_INNER_STEPS = 3
ATMOS_TIMESTEP = datetime.timedelta(hours=6)
OCEAN_TIMESTEP = datetime.timedelta(hours=18)

ATMOS_IN_NAMES = [
    "a_prog",
    "a_sfc_temp",
    "ocean_fraction",
    "land_fraction",
    "DSWRFtoa",
    "o_prog",
]
ATMOS_OUT_NAMES = ["a_prog", "a_sfc_temp", "a_diag"]
OCEAN_IN_NAMES = ["o_prog", "o_sfc_temp", "lake_fraction", "land_fraction", "a_diag"]
OCEAN_OUT_NAMES = ["o_prog", "o_sfc_temp", "o_diag"]

ATMOS_PROG_NAMES = sorted(["a_prog", "a_sfc_temp"])
OCEAN_PROG_NAMES = sorted(["o_prog", "o_sfc_temp"])
IN_VARS = ATMOS_PROG_NAMES + OCEAN_PROG_NAMES
OUT_VARS = ATMOS_OUT_NAMES + OCEAN_OUT_NAMES

# Exogenous forcing variables required from the forcing data source
ATMOS_FORCING_NAMES = sorted(["ocean_fraction", "land_fraction", "DSWRFtoa"])
OCEAN_FORCING_NAMES = ["lake_fraction"]  # land_fraction is shared with the atmosphere

# The same forcing variables in Earth2Studio names, as a data source serves them
FORCING_NAMES_E2S = {
    "ocean_fraction": "ocean_abs",
    "land_fraction": "land_abs",
    "DSWRFtoa": "mtdwswrf",
    "lake_fraction": "lake_abs",
}


class DeterministicForcing:
    """Deterministic forcing data source used by both the wrapper and the
    direct fme parity trajectory: values depend only on (time, variable)."""

    lat = MODEL_LAT.numpy()[::-1].copy()  # public north-to-south convention
    lon = MODEL_LON.numpy().copy()

    def __call__(self, time, variable) -> xr.DataArray:
        """Build deterministic fields for the requested times and variables."""
        time = np.atleast_1d(np.asarray(time, dtype="datetime64[s]"))
        variable = np.atleast_1d(np.asarray(variable))
        constants = {"ocean_abs": 0.7, "land_abs": 0.2, "lake_abs": 1.0}
        pattern = np.outer(
            np.sin(np.deg2rad(self.lat)), np.cos(np.deg2rad(self.lon))
        ).astype(np.float32)
        data = np.empty((len(time), len(variable), N_LAT, N_LON), dtype=np.float32)
        epoch = np.datetime64("2001-01-01T00:00:00", "s")
        for i, t in enumerate(time):
            hours = float((t - epoch) / np.timedelta64(1, "h"))
            for j, name in enumerate(variable):
                if name in constants:
                    data[i, j] = constants[name]
                else:
                    phase = sum(ord(c) for c in str(name)) % 7
                    data[i, j] = 0.5 + 0.25 * np.sin(
                        2 * np.pi * hours / 120.0 + phase
                    ) * (1.0 + 0.5 * pattern)
        return xr.DataArray(
            data=data,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": self.lat,
                "lon": self.lon,
            },
        )


def build_coupled_stepper():
    """Construct a genuine tiny fme CoupledStepper on CPU.

    Uses prebuilt deterministic 1x1 convolution modules for both components
    so the coupled trajectory is reproducible and cheap.
    """
    from fme.ace.stepper import StepperConfig
    from fme.core.coordinates import (
        DepthCoordinate,
        HybridSigmaPressureCoordinate,
        LatLonCoordinates,
    )
    from fme.core.dataset_info import DatasetInfo
    from fme.core.mask_provider import MaskProvider
    from fme.core.normalizer import (
        NetworkAndLossNormalizationConfig,
        NormalizationConfig,
    )
    from fme.core.ocean import OceanConfig
    from fme.core.registry.corrector import CorrectorSelector
    from fme.core.registry.module import ModuleSelector
    from fme.core.step.single_module import SingleModuleStepConfig
    from fme.core.step.step import StepSelector
    from fme.coupled.dataset_info import CoupledDatasetInfo
    from fme.coupled.stepper import ComponentConfig, CoupledStepperConfig

    torch.manual_seed(0)
    atmos_module = torch.nn.Conv2d(len(ATMOS_IN_NAMES), len(ATMOS_OUT_NAMES), 1)
    ocean_module = torch.nn.Conv2d(len(OCEAN_IN_NAMES), len(OCEAN_OUT_NAMES), 1)

    def norm(names):
        """Identity normalization config over the given variable names."""
        return NetworkAndLossNormalizationConfig(
            network=NormalizationConfig(
                means={name: 0.0 for name in names},
                stds={name: 1.0 for name in names},
            ),
        )

    config = CoupledStepperConfig(
        atmosphere=ComponentConfig(
            timedelta="6h",
            stepper=StepperConfig(
                step=StepSelector(
                    type="single_module",
                    config=dataclasses.asdict(
                        SingleModuleStepConfig(
                            builder=ModuleSelector(
                                type="prebuilt", config={"module": atmos_module}
                            ),
                            in_names=ATMOS_IN_NAMES,
                            out_names=ATMOS_OUT_NAMES,
                            normalization=norm(set(ATMOS_IN_NAMES + ATMOS_OUT_NAMES)),
                            ocean=OceanConfig(
                                surface_temperature_name="a_sfc_temp",
                                ocean_fraction_name="ocean_fraction",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        ocean=ComponentConfig(
            timedelta="18h",
            stepper=StepperConfig(
                step=StepSelector(
                    type="single_module",
                    config=dataclasses.asdict(
                        SingleModuleStepConfig(
                            builder=ModuleSelector(
                                type="prebuilt", config={"module": ocean_module}
                            ),
                            in_names=OCEAN_IN_NAMES,
                            out_names=OCEAN_OUT_NAMES,
                            next_step_forcing_names=["a_diag"],
                            normalization=norm(set(OCEAN_IN_NAMES + OCEAN_OUT_NAMES)),
                            corrector=CorrectorSelector("ocean_corrector", {}),
                        ),
                    ),
                ),
            ),
        ),
        sst_name="o_sfc_temp",
    )
    hcoord = LatLonCoordinates(lat=MODEL_LAT, lon=MODEL_LON)
    dataset_info = CoupledDatasetInfo(
        ocean=DatasetInfo(
            horizontal_coordinates=hcoord,
            vertical_coordinate=DepthCoordinate(
                torch.arange(3), torch.ones(N_LAT, N_LON, 2)
            ),
            mask_provider=MaskProvider(),
            timestep=OCEAN_TIMESTEP,
        ),
        atmosphere=DatasetInfo(
            horizontal_coordinates=hcoord,
            vertical_coordinate=HybridSigmaPressureCoordinate(
                torch.arange(3), torch.arange(3)
            ),
            mask_provider=MaskProvider(),
            timestep=ATMOS_TIMESTEP,
        ),
    )
    return config.get_stepper(dataset_info)


@pytest.fixture(autouse=True)
def fme_distributed():
    """Enter fme's distributed context around each test."""
    from fme.core.distributed.distributed import Distributed

    try:
        Distributed.get_instance()
        entered = True
    except RuntimeError:
        entered = False
    if entered:
        yield
    else:
        with Distributed.context():
            yield


@pytest.fixture
def model():
    """SamudrACE wrapper around the synthetic coupled stepper."""
    return SamudrACE(build_coupled_stepper(), DeterministicForcing())


def build_input(model, time, batch=1):
    """Build a deterministic initial-condition tensor and coordinates."""
    torch.manual_seed(1)
    in_coords = model.input_coords()
    x = torch.randn(batch, len(time), 1, len(in_coords["variable"]), N_LAT, N_LON)
    coords = in_coords.copy()
    coords["batch"] = np.arange(batch)
    coords["time"] = time
    return x, coords


device_params = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing"),
    ),
]


@pytest.mark.parametrize("device", device_params)
def test_samudrace_call(model, device):
    time = np.array([np.datetime64("2001-01-01T00:00")])
    p = model.to(device)
    x, coords = build_input(p, time)
    x = x.to(device)

    out, out_coords = p(x, coords)

    assert out.device == torch.device(device)
    assert out.shape == (1, len(time), 1, len(OUT_VARS), N_LAT, N_LON)
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    assert out_coords["lead_time"][0] == np.timedelta64(6, "h")
    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)
    # Public latitude convention is north-to-south
    assert out_coords["lat"][0] > out_coords["lat"][-1]
    # Atmosphere fields are finite predictions
    for name in ATMOS_OUT_NAMES:
        j = list(out_coords["variable"]).index(name)
        assert torch.isfinite(out[:, :, :, j]).all()


def test_samudrace_input_coords(model):
    in_coords = model.input_coords()
    assert list(in_coords["variable"]) == IN_VARS
    assert in_coords["lead_time"][0] == np.timedelta64(0, "h")
    assert in_coords["lat"][0] > in_coords["lat"][-1]
    assert in_coords["lat"].shape == (N_LAT,)
    assert in_coords["lon"].shape == (N_LON,)
    out_coords = model.output_coords(model.input_coords())
    # The stepper owns the ordering within each component; atmosphere output
    # variables come first, then ocean output variables
    out_list = list(out_coords["variable"])
    assert sorted(out_list) == sorted(OUT_VARS)
    assert sorted(out_list[: len(ATMOS_OUT_NAMES)]) == sorted(ATMOS_OUT_NAMES)
    assert sorted(out_list[len(ATMOS_OUT_NAMES) :]) == sorted(OCEAN_OUT_NAMES)
    assert out_coords["lead_time"][0] == np.timedelta64(6, "h")


@pytest.mark.parametrize("batch", [1, 2])
def test_samudrace_iter(model, batch):
    time = np.array([np.datetime64("2001-01-01T00:00")])
    x, coords = build_input(model, time, batch=batch)

    var_list = list(model.output_coords(coords.copy())["variable"])
    ocean_prog_idx = {name: var_list.index(name) for name in OCEAN_PROG_NAMES}
    ocean_diag_idx = var_list.index("o_diag")
    in_var_list = list(coords["variable"])

    p_iter = model.create_iterator(x, coords)

    # First yield is the initial condition
    out, out_coords = next(p_iter)
    assert out.shape == (batch, 1, 1, len(OUT_VARS), N_LAT, N_LON)
    assert out_coords["lead_time"][0] == np.timedelta64(0, "h")
    # Prognostic channels equal the input state; diagnostics are NaN
    for name in IN_VARS:
        j_out = var_list.index(name)
        j_in = in_var_list.index(name)
        assert torch.equal(out[:, :, 0, j_out], x[:, :, 0, j_in])
    assert torch.isnan(out[:, :, :, var_list.index("a_diag")]).all()
    assert torch.isnan(out[:, :, :, ocean_diag_idx]).all()

    outputs = [out]
    for i, (out, out_coords) in enumerate(p_iter):
        assert out.shape == (batch, 1, 1, len(OUT_VARS), N_LAT, N_LON)
        assert (out_coords["variable"] == np.array(var_list, dtype=object)).all()
        assert (out_coords["batch"] == np.arange(batch)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
        outputs.append(out)
        if i + 1 >= 2 * N_INNER_STEPS:
            break

    # Ocean prognostic fields are held constant between cycle boundaries and
    # update exactly at each boundary
    for name, j in ocean_prog_idx.items():
        for step in range(1, N_INNER_STEPS):
            assert torch.equal(outputs[step][:, :, :, j], outputs[0][:, :, :, j])
        assert not torch.equal(
            outputs[N_INNER_STEPS][:, :, :, j], outputs[0][:, :, :, j]
        )
        for step in range(N_INNER_STEPS + 1, 2 * N_INNER_STEPS):
            assert torch.equal(
                outputs[step][:, :, :, j], outputs[N_INNER_STEPS][:, :, :, j]
            )
    # Ocean diagnostics are NaN until the first boundary, then finite
    for step in range(N_INNER_STEPS):
        assert torch.isnan(outputs[step][:, :, :, ocean_diag_idx]).all()
    for step in range(N_INNER_STEPS, 2 * N_INNER_STEPS + 1):
        assert torch.isfinite(outputs[step][:, :, :, ocean_diag_idx]).all()
    # Atmosphere fields update every step
    a_prog_idx = var_list.index("a_prog")
    for step in range(1, 2 * N_INNER_STEPS + 1):
        assert not torch.equal(
            outputs[step][:, :, :, a_prog_idx], outputs[step - 1][:, :, :, a_prog_idx]
        )


def test_samudrace_parity(model):
    """The concatenated iterator trajectory equals a direct predict_paired
    trajectory over the same stepper, initial condition, and forcing."""
    import cftime
    from fme.ace.data_loading.batch_data import BatchData, PrognosticState
    from fme.coupled.data_loading.batch_data import (
        CoupledBatchData,
        CoupledPrognosticState,
    )

    n_cycles = 2
    time = np.array([np.datetime64("2001-01-01T00:00")])
    x, coords = build_input(model, time)

    # Iterator trajectory through the Earth2Studio seam
    p_iter = model.create_iterator(x, coords)
    next(p_iter)  # initial condition
    outputs = [
        out for out, _ in (next(p_iter) for _ in range(n_cycles * N_INNER_STEPS))
    ]
    var_list = list(model.output_coords(coords.copy())["variable"])

    # Direct fme trajectory: one predict_paired call over n_cycles coupled
    # steps with an independently assembled forcing window
    forcing_source = DeterministicForcing()
    base = cftime.DatetimeProlepticGregorian(2001, 1, 1, 0)

    def flip(tensor):
        """Flip the public north-to-south latitude to model orientation."""
        return torch.flip(tensor, dims=[-2])

    def forcing_window(names, step, n_steps):
        """Assemble an fme forcing BatchData window in model orientation."""
        times = [np.datetime64("2001-01-01T00:00") + k * step for k in range(n_steps)]
        da = forcing_source(
            np.array(times, dtype="datetime64[s]"),
            np.array([FORCING_NAMES_E2S[name] for name in names]),
        )
        tensor = flip(torch.as_tensor(da.values)).transpose(0, 1)
        return BatchData(
            data={name: tensor[j].unsqueeze(0) for j, name in enumerate(names)},
            time=xr.DataArray(
                np.array(
                    [
                        [
                            base
                            + datetime.timedelta(
                                hours=int(k * step / np.timedelta64(1, "h"))
                            )
                            for k in range(n_steps)
                        ]
                    ],
                    dtype=object,
                ),
                dims=["sample", "time"],
            ),
            horizontal_dims=["lat", "lon"],
        )

    in_var_list = list(coords["variable"])
    xm = flip(x[0, :, 0])  # [time=1(sample), variable, lat, lon]
    ic_time = xr.DataArray(np.array([[base]], dtype=object), dims=["sample", "time"])

    def component_state(names):
        """Build a component PrognosticState from the initial condition."""
        data = {name: xm[:, in_var_list.index(name)].unsqueeze(1) for name in names}
        return PrognosticState(
            BatchData(data=data, time=ic_time, horizontal_dims=["lat", "lon"])
        )

    ic = CoupledPrognosticState(
        ocean_data=component_state(OCEAN_PROG_NAMES),
        atmosphere_data=component_state(ATMOS_PROG_NAMES),
    )
    forcing = CoupledBatchData(
        ocean_data=forcing_window(
            OCEAN_FORCING_NAMES, np.timedelta64(18, "h"), n_cycles + 1
        ),
        atmosphere_data=forcing_window(
            ATMOS_FORCING_NAMES, np.timedelta64(6, "h"), n_cycles * N_INNER_STEPS + 1
        ),
    )
    paired, _ = model.stepper.predict_paired(ic, forcing)

    # Atmosphere fields match at every 6 hour step
    for name in ATMOS_OUT_NAMES:
        j = var_list.index(name)
        for step in range(n_cycles * N_INNER_STEPS):
            expected = flip(paired.atmosphere_data.prediction[name][:, step])
            torch.testing.assert_close(
                outputs[step][0, :, 0, j], expected, rtol=1e-5, atol=1e-5
            )
    # Ocean fields match at cycle boundaries
    for name in OCEAN_OUT_NAMES:
        j = var_list.index(name)
        for cycle in range(n_cycles):
            expected = flip(paired.ocean_data.prediction[name][:, cycle])
            boundary = outputs[(cycle + 1) * N_INNER_STEPS - 1]
            torch.testing.assert_close(
                boundary[0, :, 0, j], expected, rtol=1e-5, atol=1e-5
            )


@pytest.mark.parametrize("device", device_params)
def test_samudrace_device(model, device):
    p = model.to(device)
    assert p.device_buffer.device == torch.device(device)
    for param in p.parameters():
        assert param.device == torch.device(device)


def test_samudrace_exceptions(model):
    time = np.array([np.datetime64("2001-01-01T00:00")])
    x, coords = build_input(model, time)

    # Wrong number of dimensions
    with pytest.raises(ValueError):
        model(x[0], {k: v for k, v in coords.items() if k != "batch"})

    # Wrong variable coordinates
    bad_coords = coords.copy()
    bad_coords["variable"] = np.array(list(reversed(IN_VARS)), dtype=object)
    with pytest.raises((KeyError, ValueError)):
        model(x, bad_coords)

    # Wrong latitude orientation
    bad_coords = coords.copy()
    bad_coords["lat"] = coords["lat"][::-1]
    with pytest.raises((KeyError, ValueError)):
        model(x, bad_coords)


def test_samudrace_forcing_window_from_file(model, tmp_path):
    """The wrapper drives a file-backed forcing source over one cycle.

    A synthetic no-leap forcing NetCDF (built here, in the layout of the
    published SamudrACE forcing files) is served through
    ``SamudrACEForcingData`` and consumed by the wrapper, exercising the
    per-cycle exogenous window assembly (atmosphere at 6 hour resolution,
    year-ignoring time matching, static-field broadcast).
    """
    from unittest.mock import patch

    import cftime

    from earth2studio.data.samudrace import SamudrACEForcingData

    lat = MODEL_LAT.numpy()
    lon = MODEL_LON.numpy()
    # 6-hourly no-leap year of forcing, as in the published files
    times = [
        cftime.DatetimeNoLeap(311, 1, 1, 0) + datetime.timedelta(hours=6 * k)
        for k in range(4 * 365)
    ]
    time_varying = "DSWRFtoa"
    static = ["ocean_fraction", "land_fraction", "lake_fraction"]
    ds = xr.Dataset(
        {
            time_varying: (
                ["time", "lat", "lon"],
                np.stack(
                    [
                        np.full((N_LAT, N_LON), 0.5 + 0.01 * (k % 20), dtype=np.float32)
                        for k in range(len(times))
                    ]
                ),
            ),
            **{
                name: (
                    ["lat", "lon"],
                    np.full((N_LAT, N_LON), 0.25 * (j + 1), dtype=np.float32),
                )
                for j, name in enumerate(static)
            },
        },
        coords={"time": times, "lat": lat, "lon": lon},
    )
    path = tmp_path / "forcing_synthetic.nc"
    ds.to_netcdf(path)

    def fake_fetch(self, filename):
        """Return the local path of the synthetic forcing file."""
        return str(path)

    with patch.object(SamudrACEForcingData, "_fetch_file", fake_fetch):
        source = SamudrACEForcingData(scenario="0311", verbose=False)
        p = SamudrACE(model.stepper, source)
        time = np.array([np.datetime64("0311-01-01T00:00:00")])
        x, coords = build_input(p, time)

        p_iter = p.create_iterator(x, coords)
        next(p_iter)  # initial condition
        outputs = [out for out, _ in (next(p_iter) for _ in range(N_INNER_STEPS))]

    var_list = list(p.output_coords(coords.copy())["variable"])
    for out in outputs:
        assert out.shape == (1, 1, 1, len(OUT_VARS), N_LAT, N_LON)
        for name in ATMOS_OUT_NAMES:
            assert torch.isfinite(out[:, :, :, var_list.index(name)]).all()


def test_samudrace_forcing_out_of_calendar(model, tmp_path):
    """A trajectory crossing February 29 fails loudly on forcing lookup."""
    from unittest.mock import patch

    import cftime

    from earth2studio.data.samudrace import SamudrACEForcingData

    times = [
        cftime.DatetimeNoLeap(311, 2, 28, 0) + datetime.timedelta(hours=6 * k)
        for k in range(8)
    ]
    ds = xr.Dataset(
        {
            "DSWRFtoa": (
                ["time", "lat", "lon"],
                np.zeros((len(times), N_LAT, N_LON), dtype=np.float32),
            ),
            "ocean_fraction": (
                ["lat", "lon"],
                np.zeros((N_LAT, N_LON), dtype=np.float32),
            ),
            "land_fraction": (
                ["lat", "lon"],
                np.zeros((N_LAT, N_LON), dtype=np.float32),
            ),
            "lake_fraction": (
                ["lat", "lon"],
                np.ones((N_LAT, N_LON), dtype=np.float32),
            ),
        },
        coords={
            "time": times,
            "lat": MODEL_LAT.numpy(),
            "lon": MODEL_LON.numpy(),
        },
    )
    path = tmp_path / "forcing_feb.nc"
    ds.to_netcdf(path)

    def fake_fetch(self, filename):
        """Return the local path of the synthetic forcing file."""
        return str(path)

    with patch.object(SamudrACEForcingData, "_fetch_file", fake_fetch):
        p = SamudrACE(
            model.stepper, SamudrACEForcingData(scenario="0311", verbose=False)
        )
        # 2000 is a leap year in the proleptic Gregorian time coordinate
        time = np.array([np.datetime64("2000-02-28T18:00:00")])
        x, coords = build_input(p, time)
        with pytest.raises(ValueError, match="no counterpart"):
            p(x, coords)


def test_samudrace_load_default_package():
    from earth2studio.data.samudrace import HF_REPO_ID, HF_REVISION

    package = SamudrACE.load_default_package()
    assert package.root == f"hf://{HF_REPO_ID}@{HF_REVISION}"


@pytest.mark.package
def test_samudrace_package():
    """Load the real SamudrACE checkpoint and run one forward pass on CPU."""
    from huggingface_hub import snapshot_download

    from earth2studio.data.samudrace import HF_REPO_ID, HF_REVISION, SamudrACEData
    from earth2studio.models.auto import Package

    # The checkpoint is fetched through the shared HuggingFace hub cache
    snapshot_path = snapshot_download(
        HF_REPO_ID,
        revision=HF_REVISION,
        allow_patterns=["samudrACE_CM4_piControl_ckpt.tar"],
    )
    package = Package(snapshot_path)
    model = SamudrACE.load_model(package, scenario="0151")

    in_coords = model.input_coords()
    out_vars = list(model.output_coords(model.input_coords())["variable"])
    # Variable lists derive from the checkpoint through the lexicon
    assert "t2m" in in_coords["variable"]
    assert "thetao2p5m" in in_coords["variable"]
    assert "mslhf" in out_vars
    assert "siconc" in out_vars
    assert in_coords["lat"].shape == (180,)
    assert in_coords["lat"][0] > in_coords["lat"][-1]

    # Published initial condition for the 0151 scenario
    time = np.array([np.datetime64("0151-01-06T00:00:00")])
    da = SamudrACEData(verbose=False)(time, in_coords["variable"])
    x = torch.as_tensor(da.values, dtype=torch.float32)[None, :, None]
    coords = in_coords.copy()
    coords["batch"] = np.arange(1)
    coords["time"] = time

    out, out_coords = model(x, coords)

    assert out.shape == (1, 1, 1, len(out_vars), 180, 360)
    assert out_coords["lead_time"][0] == np.timedelta64(6, "h")
    for name in ["t2m", "sp", "mslhf"]:
        j = out_vars.index(name)
        assert torch.isfinite(out[:, :, :, j]).all()

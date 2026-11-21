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

"""Structural tests for the DLESyM split adapter (no real weights).

MockDLESyM replicates the attribute surface split_dlesym relies on with
tiny deterministic sub-models on an nside=8 HEALPix grid; sub-model inputs
and outputs are captured so tests can assert exactly what crossed the
coupling seam.
"""

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import CallableComponent
from earth2studio.nvcoupler.connector import Connector
from earth2studio.nvcoupler.dlesym_split import (
    DLESyMAtmosComponent,
    DLESyMOceanComponent,
    build_dlesym_driver,
    split_dlesym,
)
from earth2studio.nvcoupler.errors import CouplingError

NSIDE = 8
START = np.datetime64("2024-01-01", "ns")
STEP = np.timedelta64(96, "h")


# ---------------------------------------------------------------------------
# Mock DLESyM
# ---------------------------------------------------------------------------
class MockAtmosModel:
    """HEALPixRecUNet-shaped callable: inputs [state (B,F,Tin,C,H,W),
    insolation (B,F,Tsol,1,H,W), constants (F,Cc,H,W),
    coupling (Tc,B,Cv,F,H,W)] -> (B,F,Tout,C,H,W)."""

    input_time_dim = 4
    output_time_dim = 16

    def __init__(self, log: list):
        self.calls: list[dict] = []
        self._log = log

    def __call__(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        state, sol, const, coupling = inputs
        B, F, tin, C, H, W = state.shape
        assert (F, H, W) == (12, NSIDE, NSIDE)
        assert tin == self.input_time_dim
        assert sol.shape == (B, F, tin + self.output_time_dim, 1, H, W)
        assert const.shape[0] == F
        assert coupling.shape[0] == 1 + self.output_time_dim // self.input_time_dim
        leads = torch.arange(1, self.output_time_dim + 1, dtype=state.dtype)
        leads = leads.view(1, 1, -1, 1, 1, 1)
        cp = coupling.mean(dim=0).mean(dim=1)  # (B, F, H, W)
        out = (
            state[:, :, -1].unsqueeze(2)
            + 0.01 * leads
            + 0.1 * cp[:, :, None, None, :, :]
        )
        self.calls.append({"inputs": inputs, "output": out})
        self._log.append("atmos")
        return out


class MockOceanModel:
    """HEALPixUNet-shaped callable; coupling arrives as
    (lead=1, B, n_window * n_coupling_vars, F, H, W)."""

    input_time_dim = 2
    output_time_dim = 2

    def __init__(self, log: list):
        self.calls: list[dict] = []
        self._log = log

    def __call__(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        state, sol, const, coupling = inputs
        B, F, tin, C, H, W = state.shape
        assert (F, H, W) == (12, NSIDE, NSIDE)
        assert coupling.shape[:2] == (1, B) and coupling.shape[3] == F
        leads = torch.arange(1, self.output_time_dim + 1, dtype=state.dtype)
        leads = leads.view(1, 1, -1, 1, 1, 1)
        cp = coupling.mean(dim=(0, 2))  # (B, F, H, W)
        out = (
            state[:, :, -1].unsqueeze(2)
            + 0.05 * leads
            + 0.2 * cp[:, :, None, None, :, :]
        )
        self.calls.append({"inputs": inputs, "output": out})
        self._log.append("ocean")
        return out


class MockDLESyM:
    """Attribute-compatible stand-in for earth2studio.models.px.DLESyM."""

    def __init__(self, nside: int = NSIDE):
        self.nside = nside
        self.atmos_variables = ["z1000", "ws10m", "t2m"]
        self.ocean_variables = ["sst"]
        self.atmos_coupling_variables = ["sst"]
        self.ocean_coupling_variables = ["z1000", "ws10m"]

        self.atmos_input_times = np.array([-18, -12, -6, 0], dtype="timedelta64[h]")
        self.ocean_input_times = np.array([-48, 0], dtype="timedelta64[h]")
        self.atmos_output_times = np.arange(6, 97, 6).astype("timedelta64[h]")
        self.ocean_output_times = np.array([48, 96], dtype="timedelta64[h]")
        self.full_input_times = np.arange(
            self.ocean_input_times[0],
            self.atmos_input_times[-1] + 1,
            self.atmos_input_times[1] - self.atmos_input_times[0],
        )
        self.atmos_sol_times = np.concatenate(
            [self.atmos_input_times, self.atmos_output_times]
        )
        self.ocean_sol_times = np.concatenate(
            [self.ocean_input_times, self.ocean_output_times]
        )

        self.call_log: list[str] = []
        self.atmos_model = MockAtmosModel(self.call_log)
        self.ocean_model = MockOceanModel(self.call_log)

        n_atmos_steps = 1 + max(
            self.atmos_model.output_time_dim // self.atmos_model.input_time_dim, 1
        )
        full = list(self.full_input_times)
        out = list(self.atmos_output_times)
        self.atmos_input_lt_idx = [full.index(t) for t in self.atmos_input_times]
        self.ocean_input_lt_idx = [full.index(t) for t in self.ocean_input_times]
        self.atmos_coupled_input_lt_idx = [
            full.index(self.atmos_input_times[-1])
        ] * n_atmos_steps
        self.ocean_output_lt_idx = [out.index(t) for t in self.ocean_output_times]

        variables = self.atmos_variables + self.ocean_variables
        self.atmos_var_idx = [variables.index(v) for v in self.atmos_variables]
        self.ocean_var_idx = [variables.index(v) for v in self.ocean_variables]
        self.atmos_coupling_var_idx = [
            variables.index(v) for v in self.atmos_coupling_variables
        ]
        self.ocean_coupling_var_idx = [
            variables.index(v) for v in self.ocean_coupling_variables
        ]

        nvar = len(variables)
        self.center = torch.zeros(1, 1, 1, nvar, 1, 1, 1)
        self.scale = torch.ones(1, 1, 1, nvar, 1, 1, 1)
        self.atmos_constants = torch.zeros(12, 2, nside, nside)
        self.ocean_constants = torch.zeros(12, 1, nside, nside)

    def _make_insolation_tensor(
        self, anchor_times: np.ndarray, timedeltas: np.ndarray
    ) -> torch.Tensor:
        return torch.zeros(
            len(anchor_times), 12, len(timedeltas), 1, self.nside, self.nside
        )

    # Copied verbatim from earth2studio.models.px.DLESyM — dlesym_split now
    # calls these parent methods instead of re-implementing their math, so
    # the mock must expose them with identical semantics.
    def _make_atmos_coupling(self, x: torch.Tensor, coords) -> torch.Tensor:
        atmos_coupling = x[:, self.atmos_coupled_input_lt_idx][
            ..., self.atmos_coupling_var_idx, :, :, :
        ].permute(1, 0, 2, 3, 4, 5)
        return atmos_coupling

    def _make_ocean_coupling(self, x: torch.Tensor, coords) -> torch.Tensor:
        ocean_coupling = x[:, :, :, self.ocean_coupling_var_idx, :, :]
        slices = ocean_coupling.chunk(len(self.ocean_output_times), dim=2)
        ocean_coupling = torch.concat(
            [s.mean(dim=2, keepdim=True) for s in slices], dim=3
        )
        ocean_coupling = ocean_coupling.permute(2, 0, 3, 1, 4, 5)
        return ocean_coupling


def make_ic(mock: MockDLESyM, seed: int = 0) -> tuple[torch.Tensor, OrderedDict]:
    g = torch.Generator().manual_seed(seed)
    nvar = len(mock.atmos_variables + mock.ocean_variables)
    nlead = len(mock.full_input_times)
    x = torch.rand(1, 1, nlead, nvar, 12, mock.nside, mock.nside, generator=g)
    coords = OrderedDict(
        {
            "batch": np.arange(1),
            "time": np.array([START]),
            "lead_time": mock.full_input_times.copy(),
            "variable": np.array(mock.atmos_variables + mock.ocean_variables),
            "face": np.arange(12),
            "height": np.arange(mock.nside),
            "width": np.arange(mock.nside),
        }
    )
    return x, coords


def expected_ocean_coupling(atmos_out: torch.Tensor, mock: MockDLESyM) -> torch.Tensor:
    """Replicate DLESyM._make_ocean_coupling on a (B,F,L,C,H,W) atmos output."""
    oc = atmos_out[:, :, :, mock.ocean_coupling_var_idx, :, :]
    slices = oc.chunk(len(mock.ocean_output_times), dim=2)
    oc = torch.concat([s.mean(dim=2, keepdim=True) for s in slices], dim=3)
    return oc.permute(2, 0, 3, 1, 4, 5)


# ---------------------------------------------------------------------------
# Connector face-identity probe (a concurrent change makes identical HEALPix
# grids pass through as identity; xfail integration tests until it lands)
# ---------------------------------------------------------------------------
def _face_identity_supported() -> bool:
    coords = OrderedDict(
        {
            "variable": np.array(["sst"]),
            "face": np.arange(12),
            "height": np.arange(2),
            "width": np.arange(2),
        }
    )
    src = CallableComponent("src", lambda x, c: (x, c), "6h", exports=["sst"])
    dst = CallableComponent("dst", lambda x, c: (x, c), "6h", imports=["sst"])
    x = torch.zeros(1, 12, 2, 2)
    src.initialize(x, coords)
    dst.initialize(x, coords)
    try:
        Connector(src, dst).execute(np.datetime64("2024-01-01"))
    except CouplingError:
        return False
    return True


face_xfail = pytest.mark.xfail(
    condition=not _face_identity_supported(),
    reason="pending connector face fix",
    strict=False,
)


# ---------------------------------------------------------------------------
# (1) structural: split produces the right components
# ---------------------------------------------------------------------------
def test_split_structure():
    mock = MockDLESyM()
    atmos, ocean = split_dlesym(mock)

    assert isinstance(atmos, DLESyMAtmosComponent)
    assert isinstance(ocean, DLESyMOceanComponent)
    assert atmos.timestep == np.timedelta64(96, "h").astype("timedelta64[ns]")
    assert ocean.timestep == atmos.timestep

    assert atmos.import_names == ["sea_surface_temperature"]
    derived = [
        "geopotential_at_1000hpa_48h_mean",
        "wind_speed_10m_48h_mean",
    ]
    for name in ["geopotential_at_1000hpa", "wind_speed_10m", "air_temperature_2m"]:
        assert name in atmos.export_names
    for name in derived:
        assert name in atmos.export_names

    assert ocean.import_names == derived
    assert ocean.export_names == ["sea_surface_temperature"]


def test_split_registers_unknown_variables():
    mock = MockDLESyM()
    mock.atmos_variables = ["z1000", "ws10m", "mystery_var"]
    atmos, _ = split_dlesym(mock)
    assert "mystery_var" in atmos.dictionary
    assert "mystery_var" in atmos.export_names


def test_split_rejects_uneven_windows():
    mock = MockDLESyM()
    mock.ocean_output_times = np.array([32, 64, 96], dtype="timedelta64[h]")
    with pytest.raises(CouplingError, match="chunk evenly"):
        split_dlesym(mock)


# ---------------------------------------------------------------------------
# (2a) one step, standalone (no connectors): coupling tensor math
# ---------------------------------------------------------------------------
def _standalone_pair(mock, n_steps=1):
    atmos, ocean = split_dlesym(mock)
    clock = Clock(START, START + n_steps * STEP, dt=STEP)
    atmos.realize(clock)
    ocean.realize(clock)
    x, coords = make_ic(mock)
    atmos.initialize(x, coords)
    ocean.initialize(x, coords)
    return atmos, ocean, x


def _manual_step(atmos, ocean, time):
    # ocean -> atmos (lagged SST), atmos, atmos -> ocean, ocean
    atmos.import_state.add(ocean.export_state["sea_surface_temperature"])
    atmos.run(time)
    for name in ocean.import_names:
        ocean.import_state.add(atmos.export_state[name])
    ocean.run(time)


def test_standalone_step_delivers_chunk_mean_coupling():
    mock = MockDLESyM()
    atmos, ocean, x = _standalone_pair(mock)
    _manual_step(atmos, ocean, START + STEP)

    assert mock.call_log == ["atmos", "ocean"]

    # atmos coupling = IC SST at lead 0, persisted over all internal sub-steps
    atmos_coupling = mock.atmos_model.calls[-1]["inputs"][3]
    sst0 = x[:, :, -1, 3].reshape(-1, 12, NSIDE, NSIDE)  # normalized == physical
    assert atmos_coupling.shape[0] == len(mock.atmos_coupled_input_lt_idx)
    for k in range(atmos_coupling.shape[0]):
        torch.testing.assert_close(atmos_coupling[k, :, 0], sst0)

    # ocean coupling tensor == _make_ocean_coupling chunk means of atmos output
    atmos_out = mock.atmos_model.calls[-1]["output"]
    expected = expected_ocean_coupling(atmos_out, mock)
    received = mock.ocean_model.calls[-1]["inputs"][3]
    torch.testing.assert_close(received, expected)

    # SST export is the ocean output at the 96 h lead
    ocean_out = mock.ocean_model.calls[-1]["output"]
    sst_export = ocean.export_state["sea_surface_temperature"]
    torch.testing.assert_close(
        sst_export.data.reshape(-1, 12, NSIDE, NSIDE), ocean_out[:, :, -1, 0]
    )
    assert sst_export.valid_time == START + STEP


def test_standalone_two_steps_lagged_sst_and_window():
    mock = MockDLESyM()
    atmos, ocean, _ = _standalone_pair(mock, n_steps=2)
    _manual_step(atmos, ocean, START + STEP)
    _manual_step(atmos, ocean, START + 2 * STEP)

    # lagged feedback: atmos step-2 coupling == ocean step-1 SST at 96 h
    sst_96h = mock.ocean_model.calls[0]["output"][:, :, -1, 0]
    coupling2 = mock.atmos_model.calls[1]["inputs"][3]
    for k in range(coupling2.shape[0]):
        torch.testing.assert_close(coupling2[k, :, 0], sst_96h)

    # sliding window: atmos step-2 state == step-1 outputs at 78/84/90/96 h
    out1 = mock.atmos_model.calls[0]["output"]  # (B, F, 16, C, H, W)
    state2 = mock.atmos_model.calls[1]["inputs"][0]
    torch.testing.assert_close(state2, out1[:, :, 12:16])

    # ocean window: step-2 state == step-1 outputs at 48/96 h
    oout1 = mock.ocean_model.calls[0]["output"]
    ostate2 = mock.ocean_model.calls[1]["inputs"][0]
    torch.testing.assert_close(ostate2, oout1)


def test_ocean_run_without_imports_raises():
    mock = MockDLESyM()
    _, ocean, _ = _standalone_pair(mock)
    with pytest.raises(CouplingError, match="atmos -> ocean connector"):
        ocean.run(START + STEP)


def test_gradients_flow_across_split():
    mock = MockDLESyM()
    atmos, ocean, _ = _standalone_pair(mock)
    x, coords = make_ic(mock)
    x = x.clone().requires_grad_(True)
    atmos.initialize(x, coords)
    ocean.initialize(x, coords)
    _manual_step(atmos, ocean, START + STEP)
    loss = ocean.export_state["sea_surface_temperature"].data.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# (2b)/(3) driver integration through connectors (identity on HEALPix grid)
# ---------------------------------------------------------------------------
@face_xfail
def test_driver_one_step_atmos_then_ocean():
    mock = MockDLESyM()
    driver = build_dlesym_driver(mock, START, START + STEP)
    x, coords = make_ic(mock)
    driver.initialize({"atmos": (x, coords), "ocean": (x, coords)})
    for _time, _states in driver.steps():
        pass

    assert mock.call_log == ["atmos", "ocean"]
    atmos_out = mock.atmos_model.calls[-1]["output"]
    expected = expected_ocean_coupling(atmos_out, mock)
    received = mock.ocean_model.calls[-1]["inputs"][3]
    torch.testing.assert_close(received, expected)


@face_xfail
def test_driver_two_steps_sst_feedback():
    mock = MockDLESyM()
    driver = build_dlesym_driver(mock, START, START + 2 * STEP)
    x, coords = make_ic(mock)
    driver.initialize({"atmos": (x, coords), "ocean": (x, coords)})
    for _time, _states in driver.steps():
        pass

    assert mock.call_log == ["atmos", "ocean", "atmos", "ocean"]
    sst_96h = mock.ocean_model.calls[0]["output"][:, :, -1, 0]
    coupling2 = mock.atmos_model.calls[1]["inputs"][3]
    for k in range(coupling2.shape[0]):
        torch.testing.assert_close(coupling2[k, :, 0], sst_96h)

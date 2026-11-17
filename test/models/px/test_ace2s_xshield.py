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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import xarray as xr

import earth2studio.models.px.ace2 as ace2_mod
import earth2studio.models.px.ace2s_xshield as ace_src
from earth2studio.data import ACE2SXSHiELDData, Random, fetch_data
from earth2studio.data.ace2 import ACE_GRID_LAT, ACE_GRID_LON
from earth2studio.lexicon.ace import ACELexicon
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px.ace2s_xshield import ACE2SXSHiELD
from earth2studio.utils.imports import OptionalDependencyFailure


@pytest.fixture(autouse=True)
def optional_backend(monkeypatch):
    for module in (ace_src, ace2_mod):
        if module.__file__ in OptionalDependencyFailure.failures:
            monkeypatch.delitem(OptionalDependencyFailure.failures, module.__file__)

    if not hasattr(ace2_mod.BatchData, "new_on_device"):

        class Batch:
            @classmethod
            def new_on_device(cls, **kwargs):
                return SimpleNamespace(**kwargs)

        monkeypatch.setattr(ace2_mod, "BatchData", Batch)
        monkeypatch.setattr(
            ace2_mod, "PrognosticState", lambda data: SimpleNamespace(_data=data)
        )


class _Output:
    prediction: dict = {}


class _Stepper(torch.nn.Module):
    prognostic_names = ["UGRD10m", "VGRD10m", "PRESsfc"]
    _input_only_names = ["DSWRFtoa", "global_mean_co2"]
    out_names = ["UGRD10m", "VGRD10m", "PRESsfc", "PRATEsfc", "PRMSL"]

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("device_buffer", torch.empty(0))

    def predict_paired(self, ic, forcing_batch):
        batch, _, lat, lon = ic._data.data[self.prognostic_names[0]].shape
        output = _Output()
        output.prediction = {
            key: torch.randn(batch, 1, lat, lon, device=self.device_buffer.device)
            for key in self.out_names
        }
        return output, None


def _forcing(model, x, continuation=False):
    signature = model.forcing_coords()
    leads = x.lead_time.values[-1:] + signature.lead_time.values
    if continuation:
        leads = leads[-1:]
    forcing = fetch_data(
        Random({"lat": ACE_GRID_LAT, "lon": ACE_GRID_LON}),
        x.time.values,
        signature.coords["variable"].values,
        leads,
        device=x.e2s.to_torch()[0].device,
    )
    for dim in reversed(x.dims[: x.get_axis_num("time")]):
        forcing = forcing.expand_dims({dim: x.coords[dim].values}).copy(deep=True)
    return forcing


def _initial(model, device):
    time = np.array([np.datetime64("2014-01-01T00:00")])
    signature = model.input_coords()
    x = fetch_data(
        Random({"lat": ACE_GRID_LAT, "lon": ACE_GRID_LON}),
        time,
        signature.coords["variable"].values,
        signature.lead_time.values,
        device=device,
    )
    kept = {
        key: value
        for key, value in signature.attrs.items()
        if key
        not in (
            "earth2studio_kind",
            "earth2studio_schema_version",
            "earth2studio_dynamic_dims",
        )
    }
    x.attrs.update(kept)
    return x


def test_msl_lexicon():
    assert ACELexicon["msl"][0] == "PRMSL"
    assert ACELexicon.get_e2s_from_fme("PRMSL") == "msl"


def test_default_sources_are_xshield():
    model = ACE2SXSHiELD(_Stepper())
    initial, forcing = model.default_sources()
    assert isinstance(initial, ACE2SXSHiELDData)
    assert isinstance(forcing, ACE2SXSHiELDData)
    assert initial._mode == "initial_conditions"
    assert forcing._mode == "forcing"
    assert initial._HF_REPO_ID == "allenai/HiRO-ACEv2"


# fme 2026.8.1 rejects tensors that are not already on get_device(), which is
# CUDA whenever a GPU is visible.
_DEVICE = ["cuda:0" if torch.cuda.is_available() else "cpu"]


@pytest.mark.parametrize("device", _DEVICE)
def test_rollout_replay_ignores_later_set_rng(device):
    model = ACE2SXSHiELD(_Stepper()).to(device)
    model.set_rng(0)
    x = _initial(model, device)
    forcing = _forcing(model, x)
    y, state = model.initialize(x, forcing)
    assert "tp" in set(y.coords["variable"].values)
    assert "msl" in set(y.coords["variable"].values)
    assert y.lead_time.values[0] == np.timedelta64(6, "h")
    nxt = _forcing(model, y, continuation=True)
    before = x.copy(deep=True)
    advanced, _ = model.step(y, nxt, state)
    model.set_rng(99)
    replayed, _ = model.step(y, nxt, state)
    xr.testing.assert_allclose(advanced, replayed)
    xr.testing.assert_identical(x, before)
    model.set_rng(1)
    other, _ = model.initialize(x, forcing)
    y_tensor, _ = y.e2s.to_torch()
    other_tensor, _ = other.e2s.to_torch()
    assert not torch.allclose(y_tensor, other_tensor)


@pytest.mark.parametrize("device", _DEVICE)
def test_ace2s_xshield_conformance(device):
    model = ACE2SXSHiELD(_Stepper()).to(device)
    assert check_prognostic_contract(model, device=device) == [
        "P21: checkpoint serialization requires component-specific tests"
    ]

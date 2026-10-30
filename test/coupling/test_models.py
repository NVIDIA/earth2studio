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
from coupling_toys import (
    ToyAtmosphere,
    ToyImpact,
    ToyOcean,
    ToyPrognostic,
    toy_grid,
)

from earth2studio.coupling import CouplingError
from earth2studio.coupling._models import inspect_model
from earth2studio.models.px import Persistence


def test_prognostic_declarations_are_read_from_the_contract():
    spec = inspect_model("ocean", ToyOcean())
    assert spec.kind == "prognostic"
    assert spec.cadence == np.timedelta64(48, "h")
    assert spec.imports == {"z1000:mean:48h": 0}
    assert spec.exports == {"sst": 0}
    assert spec.state == {"sst": 0}
    assert spec.import_kind == "forcing"
    assert spec.stream(0) == "ocean" and spec.stream(1) == "ocean.1"


def test_state_variables_are_outputs_also_declared_as_inputs():
    spec = inspect_model("atmos", ToyPrognostic(["z1000", "sst"]))
    assert spec.state == {"z1000": 0, "sst": 0}
    assert spec.imports == {}
    assert spec.forcing == ()


def test_real_persistence_model_is_inspectable():
    spec = inspect_model("persist", Persistence(["t2m", "z1000"], toy_grid(8, 16)))
    assert spec.cadence == np.timedelta64(6, "h")
    assert set(spec.exports) == {"t2m", "z1000"}
    assert set(spec.state) == {"t2m", "z1000"}


def test_diagnostic_declarations_and_cadence_override():
    spec = inspect_model("impact", ToyImpact(), cadence="12h")
    assert spec.kind == "diagnostic"
    assert spec.import_kind == "input"
    assert spec.imports == {"z1000": 0}
    assert spec.exports == {"ws10m": 0}
    assert spec.cadence == np.timedelta64(12, "h")
    with pytest.raises(CouplingError, match="no cadence"):
        inspect_model("impact", ToyImpact()).required_cadence()
    with pytest.raises(CouplingError, match="positive"):
        inspect_model("impact", ToyImpact(), cadence="0h")


def test_prognostic_cadence_cannot_be_overridden():
    with pytest.raises(CouplingError, match="cannot be overridden"):
        inspect_model("atmos", ToyAtmosphere(), cadence="6h")


class _NotAModel:
    pass


class _NoAdvance(ToyPrognostic):
    def output_coords(self, input_coords):  # type: ignore[override]
        return input_coords


class _Broken(ToyPrognostic):
    def output_coords(self, input_coords):  # type: ignore[override]
        raise ValueError("bad")


@pytest.mark.parametrize(
    "model,error,match",
    [
        (_NotAModel(), TypeError, "declares no input_coords"),
        (_NoAdvance(["z1000"]), CouplingError, "does not advance"),
        (_Broken(["z1000"]), CouplingError, "rejected its own"),
    ],
)
def test_inspect_model_rejects(model, error, match):
    with pytest.raises(error, match=match):
        inspect_model("m", model)

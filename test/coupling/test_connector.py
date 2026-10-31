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

import pytest
from coupling_toys import toy_grid

from earth2studio.coupling import Connector
from earth2studio.grids import NearestRegridder


def test_connector_is_a_frozen_declaration():
    connector = Connector("atmos", "ocean", variables=["z1000:mean:48h"])
    assert connector.name == "atmos->ocean"
    assert connector.variables == ("z1000:mean:48h",)
    assert hash(connector) == hash(Connector("atmos", "ocean", ("z1000:mean:48h",)))
    with pytest.raises(AttributeError):
        connector.src = "other"  # type: ignore[misc]
    assert "variables=['z1000:mean:48h']" in repr(connector)
    assert "variables=auto" in repr(Connector("a", "b"))


def test_connector_accepts_single_label_and_regridder():
    regridder = NearestRegridder(toy_grid(16, 32), toy_grid(32, 64))
    connector = Connector("ocean", "atmos", variables="sst", regridder=regridder)
    assert connector.variables == ("sst",)
    assert "NearestRegridder" in repr(connector)
    assert "transform" in repr(Connector("a", "b", transform=lambda x: x))


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"src": "a", "dst": "a"}, ValueError),
        ({"src": "a", "dst": "b", "variables": ()}, ValueError),
        ({"src": "a", "dst": "b", "regridder": object()}, TypeError),
        ({"src": "a", "dst": "b", "transform": 3}, TypeError),
    ],
)
def test_connector_rejects(kwargs, error):
    with pytest.raises(error):
        Connector(**kwargs)

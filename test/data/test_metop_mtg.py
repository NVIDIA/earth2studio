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

import pathlib
import shutil
from datetime import datetime

import numpy as np
import pytest

from earth2studio.data import MetOpMTG
from earth2studio.lexicon import MetOpMTGLexicon


@pytest.mark.timeout(15)
def test_metop_mtg_grid():
    lat, lon = MetOpMTG.grid(resolution="2km")
    assert lat.shape == (5568, 5568)
    assert lon.shape == (5568, 5568)
    assert not np.all(np.isnan(lat))
    assert not np.all(np.isnan(lon))


@pytest.mark.timeout(60)
def test_metop_mtg_grid_1km():
    lat, lon = MetOpMTG.grid(resolution="1km")
    assert lat.shape == (11136, 11136)
    assert lon.shape == (11136, 11136)


@pytest.mark.timeout(15)
def test_metop_mtg_grid_invalid_resolution():
    with pytest.raises(ValueError):
        MetOpMTG.grid(resolution="500m")


@pytest.mark.timeout(15)
def test_metop_mtg_available():
    assert MetOpMTG.available(datetime(2024, 1, 16, 12, 0))
    assert MetOpMTG.available(datetime(2025, 3, 1, 0, 0))
    assert not MetOpMTG.available(datetime(2023, 12, 31, 0, 0))


@pytest.mark.timeout(15)
def test_metop_mtg_lexicon():
    assert "mtg_vis_04" in MetOpMTGLexicon.VOCAB
    assert "mtg_ir_133" in MetOpMTGLexicon.VOCAB
    assert len(MetOpMTGLexicon.VOCAB) == 12

    channel_name, modifier = MetOpMTGLexicon["mtg_vis_04"]
    assert channel_name == "vis_04"
    assert callable(modifier)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
def test_metop_mtg_mixed_resolution_error():
    ds = MetOpMTG(resolution="2km")
    with pytest.raises(ValueError, match="resolution"):
        ds(
            datetime(2025, 1, 1, 0, 0),
            ["mtg_vis_04", "mtg_ir_87"],
        )


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_metop_mtg_fetch():
    ds = MetOpMTG(resolution="2km", cache=False)
    time = datetime(2025, 3, 1, 12, 0)
    variable = "mtg_ir_87"
    data = ds(time, variable)
    shape = data.shape
    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 5568
    assert shape[3] == 5568
    assert not np.all(np.isnan(data.values))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
def test_metop_mtg_fetch_multistep():
    ds = MetOpMTG(resolution="2km", cache=False)
    time = [
        datetime(2025, 3, 1, 12, 0),
        datetime(2025, 3, 1, 12, 10),
    ]
    variable = ["mtg_ir_87", "mtg_ir_133"]
    data = ds(time, variable)
    shape = data.shape
    assert shape[0] == 2
    assert shape[1] == 2
    assert shape[2] == 5568
    assert shape[3] == 5568


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("cache", [True, False])
def test_metop_mtg_cache(cache):
    ds = MetOpMTG(resolution="2km", cache=cache)
    time = datetime(2025, 3, 1, 12, 0)
    variable = "mtg_ir_87"
    data = ds(time, variable)
    assert data.shape[0] == 1

    assert pathlib.Path(ds.cache).is_dir() == cache

    # Reload from cache
    data2 = ds(time, variable)
    assert data2.shape[0] == 1

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass

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

from earth2studio.lexicon.gefs import GEFSLexicon, GEFSLexiconSel


def test_gefs_lexicon():
    key, mod = GEFSLexicon.get_item("u10m")
    assert key == "pgrb2a::UGRD::10 m above ground"
    assert mod(np.array([1.0])) == np.array([1.0])

    key, mod = GEFSLexicon.get_item("z1000")
    assert key == "pgrb2a::HGT::1000 mb"
    assert mod(np.array([1.0])) == np.array([9.81])

    # Test invalid key
    with pytest.raises(KeyError):
        GEFSLexicon.get_item("invalid_key")


def test_gefs_lexicon_sel():
    key, mod = GEFSLexiconSel.get_item("u10m")
    assert key == "pgrb2s::UGRD::10 m above ground"
    assert mod(np.array([1.0])) == np.array([1.0])

    # Test invalid key
    with pytest.raises(KeyError):
        GEFSLexiconSel.get_item("invalid_key")

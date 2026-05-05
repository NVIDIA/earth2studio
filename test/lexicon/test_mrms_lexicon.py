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

from earth2studio.lexicon.mrms import MRMSLexicon


def test_mrms_lexicon_basic_mapping_and_modifier_identity():
    key, mod = MRMSLexicon.get_item("refc")
    assert key == "MergedReflectivityQCComposite_00.50"

    x = np.array([0.0, 1.5, -3.2], dtype=np.float32)
    y = mod(x.copy())
    assert np.array_equal(y, x)


def test_mrms_lexicon_refc_base_mapping_and_modifier_identity():
    key, mod = MRMSLexicon.get_item("refc_base")
    assert key == "MergedBaseReflectivityQC_00.50"

    x = np.array([0.0, 1.5, -3.2], dtype=np.float32)
    y = mod(x.copy())
    assert np.array_equal(y, x)


def test_mrms_lexicon_invalid_key():
    with pytest.raises(KeyError):
        MRMSLexicon.get_item("invalid_key")

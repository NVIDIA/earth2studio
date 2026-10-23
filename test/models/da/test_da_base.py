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

from earth2studio.models.da import (
    AssimilationModel,
    CorrDiffCosmoEra5SDA,
    HealDA,
    InterpEquirectangular,
    StatefulAssimilationModel,
    StormCastSDA,
)


@pytest.mark.parametrize(
    "model,stateful",
    [
        (HealDA, False),
        (InterpEquirectangular, False),
        (CorrDiffCosmoEra5SDA, False),
        (StormCastSDA, True),
    ],
)
def test_assimilation_capabilities(model, stateful):
    assert issubclass(model, AssimilationModel)
    assert issubclass(model, StatefulAssimilationModel) is stateful
    assert not hasattr(model, "create_generator")


def test_streaming_alone_does_not_imply_stateful_assimilation():
    class StreamingInterpolation(InterpEquirectangular):
        def create_iterator(self, batches):
            for batch in batches:
                yield self(batch)

    assert issubclass(StreamingInterpolation, AssimilationModel)
    assert not issubclass(StreamingInterpolation, StatefulAssimilationModel)

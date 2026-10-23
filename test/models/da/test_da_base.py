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
import torch

from earth2studio.models.da import (
    AssimilationModel,
    CorrDiffCosmoEra5SDA,
    HealDA,
    InterpEquirectangular,
    StatefulAssimilationModel,
    StormCastConusSDA,
    StormCastSDA,
)


@pytest.mark.parametrize(
    "model,stateful,stochastic",
    [
        (HealDA, False, False),
        (InterpEquirectangular, False, False),
        (CorrDiffCosmoEra5SDA, False, True),
        (StormCastSDA, True, True),
        (StormCastConusSDA, True, True),
    ],
)
def test_assimilation_capabilities(model, stateful, stochastic):
    # Check structural declarations without loading weights or optional backends.
    instance = model.__new__(model)
    assert isinstance(instance, AssimilationModel)
    assert isinstance(instance, StatefulAssimilationModel) is stateful
    assert instance.stochastic is stochastic
    assert not hasattr(model, "create_generator")
    if not stochastic:
        rng = torch.get_rng_state()
        instance.set_rng(42)
        instance.set_rng(99, reset=False)
        assert torch.equal(rng, torch.get_rng_state())


def test_streaming_alone_does_not_imply_stateful_assimilation():
    class StreamingInterpolation(InterpEquirectangular):
        def create_iterator(self, batches):
            for batch in batches:
                yield self(batch)

    model = StreamingInterpolation()
    assert isinstance(model, AssimilationModel)
    assert not isinstance(model, StatefulAssimilationModel)

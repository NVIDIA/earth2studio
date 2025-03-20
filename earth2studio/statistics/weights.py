# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from typing import TypeVar

import numpy as np
import torch

T = TypeVar("T", np.ndarray, torch.Tensor)


def lat_weight(lat: T) -> T:
    """
    Compute the cosine-based latitude weighting for a given tensor
    representing latitude coordinates. The latitude coordinates
    are assumed to be in degrees.

    Uses the formula weights = cos(lat * pi / 180.0)

    Parameters
    ----------
    lat: Union[np.ndarray, torch.Tensor]
        Array of latitude values, in degrees. Should be in the range [-90, 90].

    Returns
    -------
    weights: Union[np.ndarray, torch.Tensor]
        Array of latitude weights, formally defined above.

    """
    lib = np if isinstance(lat, np.ndarray) else torch
    weights = lib.cos(lat * lib.pi / 180.0)
    return weights / weights.mean()

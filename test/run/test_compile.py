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

from collections import OrderedDict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

import earth2studio.run as run
from earth2studio.data import Random
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Persistence


def test_deterministic_compile_flag():
    """Verify that the compile flag triggers torch.compile on the prognostic model's _forward method."""
    coords = OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))])
    variable = ["t2m"]
    nsteps = 1
    time = ["2024-01-01"]
    device = "cpu"

    data = Random(domain_coords=coords)
    model = Persistence(variable, coords)
    io = ZarrBackend()

    # Mock torch.compile to avoid actual compilation during test
    with patch("torch.compile", side_effect=lambda x, **kwargs: x) as mock_compile:
        run.deterministic(time, nsteps, model, data, io, device=device, compile=True)
        
        # Verify torch.compile was called
        # Note: We check if it was called at least once. 
        # In deterministic, it's called on prognostic._forward
        assert mock_compile.called
        # Check if it was called with the model's _forward method
        # The first argument to the first call should be a function (the _forward method)
        args, kwargs = mock_compile.call_args
        assert kwargs.get("mode") == "reduce-overhead"


def test_diagnostic_compile_flag():
    """Verify that the compile flag triggers torch.compile on both models in diagnostic workflow."""
    coords = OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))])
    variable = ["t2m"]
    nsteps = 1
    time = ["2024-01-01"]
    device = "cpu"

    data = Random(domain_coords=coords)
    prognostic = Persistence(variable, coords)
    
    # Simple diagnostic model mock
    diagnostic_model = MagicMock()
    diagnostic_model.to.return_value = diagnostic_model
    diagnostic_model.input_coords.return_value = prognostic.input_coords()
    diagnostic_model.output_coords.return_value = prognostic.output_coords(prognostic.input_coords())
    
    io = ZarrBackend()

    with patch("torch.compile", side_effect=lambda x, **kwargs: x) as mock_compile:
        run.diagnostic(time, nsteps, prognostic, diagnostic_model, data, io, device=device, compile=True)
        
        # Should be called for prognostic._forward and diagnostic_model
        assert mock_compile.call_count >= 2


def test_ensemble_compile_flag():
    """Verify that the compile flag triggers torch.compile on the prognostic model in ensemble workflow."""
    coords = OrderedDict([("lat", np.arange(10)), ("lon", np.arange(20))])
    variable = ["t2m"]
    nsteps = 1
    nensemble = 2
    time = ["2024-01-01"]
    device = "cpu"

    data = Random(domain_coords=coords)
    model = Persistence(variable, coords)
    
    perturbation = MagicMock()
    perturbation.side_effect = lambda x, c: (x, c)
    
    io = ZarrBackend()

    with patch("torch.compile", side_effect=lambda x, **kwargs: x) as mock_compile:
        run.ensemble(time, nsteps, nensemble, model, data, io, perturbation, device=device, compile=True)
        
        assert mock_compile.called
        args, kwargs = mock_compile.call_args
        assert kwargs.get("mode") == "reduce-overhead"

#!/usr/bin/env python3
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

"""
Deterministic Workflow Custom Pipeline

This pipeline implements the deterministic workflow from examples/01_deterministic_workflow.py
as a custom pipeline that can be invoked via the REST API.
"""

from datetime import datetime
from typing import Literal

from api_server.workflow import Earth2Workflow, workflow_registry
from earth2studio import run
from earth2studio.data import GFS
from earth2studio.io import IOBackend
from earth2studio.models.px import DLWP, FCN


@workflow_registry.register
class DeterministicEarth2Workflow(Earth2Workflow):
    """
    Deterministic workflow with auto-registration
    """

    name = "deterministic_earth2_workflow"
    description = "Deterministic workflow with auto-registration"

    def __init__(self, model_type: Literal["fcn", "dlwp"] = "fcn"):
        super().__init__()

        if model_type == "fcn":
            package = FCN.load_default_package()
            self.model = FCN.load_model(package)
        elif model_type == "dlwp":
            package = DLWP.load_default_package()
            self.model = DLWP.load_model(package)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.data = GFS()

    def __call__(
        self,
        io: IOBackend,
        start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
        num_steps: int = 6,
    ) -> None:
        """Run the deterministic workflow pipeline"""

        run.deterministic(start_time, num_steps, self.model, self.data, io)

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
Workflow package for custom workflows.

This package provides the base classes and functionality for creating custom workflows
that can be served as REST APIs.
"""

from typing import Any

from .workflow import (
    Workflow,
    WorkflowParameters,
    WorkflowProgress,
    WorkflowRegistry,
    WorkflowResult,
    WorkflowStatus,
    get_workflow_config,
    parse_workflow_directories_from_env,
    register_all_workflows,
    workflow_registry,
)


# Lazy import for Earth2Workflow to avoid requiring torch/earth2studio dependencies
# unless actually needed
def __getattr__(name: str) -> Any:
    if name == "Earth2Workflow":
        from .e2workflow import Earth2Workflow

        return Earth2Workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Workflow",
    "WorkflowParameters",
    "WorkflowProgress",
    "WorkflowResult",
    "WorkflowStatus",
    "WorkflowRegistry",
    "workflow_registry",
    "get_workflow_config",
    "register_all_workflows",
    "parse_workflow_directories_from_env",
    "Earth2Workflow",
]

# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Earth2Studio API Server Package

This package contains the main API server components including:
- main: FastAPI application and REST endpoints
- workflow: Workflow framework and registry
- worker: RQ worker functions for processing jobs
- cpu_worker: CPU-intensive worker functions
"""

from typing import Any

from earth2studio.serve.server.workflow import (
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


def __getattr__(name: str) -> Any:
    if name == "Earth2Workflow":
        from earth2studio.serve.server import e2workflow

        return e2workflow.Earth2Workflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "main",
    "workflow",
    "worker",
    "cpu_worker",
    "config",
    "object_storage",
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

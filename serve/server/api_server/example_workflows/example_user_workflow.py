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
Example User Workflow

This is a template/example showing how users should structure their workflow files
for automatic discovery and registration.

To use this workflow:
1. Copy this file to your workflow directory (e.g., ~/my_workflows/)
2. Set WORKFLOW_DIR environment variable: export WORKFLOW_DIR=~/my_workflows
3. Start the server - this workflow will be auto-registered
4. Access via POST /v1/infer/example_user_workflow

This file serves as both documentation and a working example.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from api_server.workflow import (
    Workflow,
    WorkflowParameters,
    WorkflowProgress,
    workflow_registry,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExampleUserWorkflowParameters(WorkflowParameters):
    """
    Parameters for the example user workflow.

    Define your workflow parameters here using Pydantic Field for validation.
    """

    task_name: str = Field(
        default="example_task",
        description="Name of the task to execute",
    )

    num_iterations: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of iterations to perform",
    )

    delay_seconds: float = Field(
        default=0.5,
        ge=0,
        le=10,
        description="Delay between iterations in seconds",
    )

    generate_output: bool = Field(
        default=True, description="Whether to generate output files"
    )


@workflow_registry.register
class ExampleUserWorkflow(Workflow):
    """
    Example user workflow demonstrating the workflow pattern.

    This workflow:
    1. Accepts parameters via the REST API
    2. Performs a simple iterative task
    3. Updates progress using WorkflowProgress during execution
    4. Generates output files
    5. Returns results

    Users should extend this pattern for their own workflows.
    """

    name = "example_user_workflow"
    description = "Example user workflow demonstrating auto-registration pattern"
    Parameters = ExampleUserWorkflowParameters

    # No __init__ needed - name and description are set by the registry during registration
    # You only need to define __init__ if you have custom initialization logic

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | ExampleUserWorkflowParameters
    ) -> ExampleUserWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return ExampleUserWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | ExampleUserWorkflowParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """
        Execute the workflow with given parameters.

        This is the main entry point that gets called when a user submits
        a workflow execution request via POST /v1/infer/example_user_workflow

        Args:
            parameters: Workflow parameters (dict or Parameter object)
            execution_id: Unique ID for this execution

        Returns:
            Dictionary with execution results
        """

        # ========================================
        # 1. Parse and validate parameters
        # ========================================
        parameters = self.validate_parameters(parameters)

        # ========================================
        # 2. Initialize execution tracking
        # ========================================
        metadata = {"parameters": parameters.model_dump()}

        try:
            # Store metadata separately
            self.update_execution_data(execution_id, {"metadata": metadata})

            # ========================================
            # 3. Execute workflow logic
            # ========================================

            # Update status to show progress
            progress = WorkflowProgress(
                progress="Starting workflow...",
                current_step=0,
                total_steps=parameters.num_iterations,
            )
            self.update_execution_data(execution_id, progress)

            # Simulate iterative processing
            results = []
            for i in range(parameters.num_iterations):
                # Update progress
                progress = WorkflowProgress(
                    progress=f"Processing iteration {i + 1}/{parameters.num_iterations}",
                    current_step=i + 1,
                    total_steps=parameters.num_iterations,
                )
                self.update_execution_data(execution_id, progress)

                # Do some work (simulated with sleep)
                time.sleep(parameters.delay_seconds)

                # Store iteration result
                result = {
                    "iteration": i + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "value": i * i,  # Just an example calculation
                }
                results.append(result)

            # ========================================
            # 4. Generate output files (if requested)
            # ========================================

            if parameters.generate_output:
                # Create results file
                results_path = self.get_output_path(execution_id) / "results.json"
                with open(results_path, "w") as f:
                    json.dump(
                        {
                            "task_name": parameters.task_name,
                            "iterations": results,
                            "summary": {
                                "total_iterations": len(results),
                                "execution_id": execution_id,
                            },
                        },
                        f,
                        indent=2,
                    )

                # Create a summary text file
                summary_path = self.get_output_path(execution_id) / "summary.txt"
                with open(summary_path, "w") as f:
                    f.write("Workflow Execution Summary\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Execution ID: {execution_id}\n")
                    f.write(f"Task Name: {parameters.task_name}\n")
                    f.write(f"Iterations: {parameters.num_iterations}\n")
                    f.write(f"Completed: {datetime.now(timezone.utc).isoformat()}\n")
                    f.write("=" * 50 + "\n")

            # ========================================
            # 5. Calculate execution time and complete
            # ========================================
            # Mark workflow as completed
            progress = WorkflowProgress(
                progress="Complete!",
                current_step=parameters.num_iterations,
                total_steps=parameters.num_iterations,
            )
            self.update_execution_data(execution_id, progress)

            # Update final metadata with results summary
            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": f"Completed {parameters.num_iterations} iterations",
                    }
                },
            )

            # ========================================
            # 6. Return results
            # ========================================
            return {
                "status": "success",
                "execution_id": execution_id,
                "message": f"Successfully completed {parameters.num_iterations} iterations",
                "results": {
                    "total_iterations": len(results),
                    "last_value": results[-1]["value"] if results else None,
                },
            }

        except Exception as e:
            # ========================================
            # Error handling
            # ========================================
            progress = WorkflowProgress(progress="Failed!", error_message=str(e))
            self.update_execution_data(execution_id, progress)
            logger.error(f"Error in ExampleUserWorkflow: {e}")
            raise

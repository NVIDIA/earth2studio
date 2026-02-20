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
Workflow module

This module provides the base classes and functionality for custom workflows.
"""

import importlib.util
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Union

import redis  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Import configuration
from api_server.config import (
    get_config,
    get_config_manager,
    get_workflow_config,
)

# Get configuration
config = get_config()
config_manager = get_config_manager()

# Configure logging
config_manager.setup_logging()
logger = logging.getLogger(__name__)


class WorkflowStatus:
    """Workflow execution status constants"""

    QUEUED = "queued"
    RUNNING = "running"
    PENDING_RESULTS = "pending_results"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class WorkflowArgsBase(BaseModel):
    """Base parameters for workflow execution"""

    # Add strict validation - reject unknown fields
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def validate(
        cls, data: Union[dict[str, Any], "WorkflowArgsBase"]
    ) -> "WorkflowArgsBase":
        """
        Validate and convert parameters to the correct type.

        This method performs both field validation and model validation using Pydantic.
        It ensures that all required fields are present, types are correct, and any
        custom validators defined in the parameter class are executed.

        Args:
            data: Input data as dict or WorkflowArgsBase instance

        Returns:
            Validated WorkflowArgsBase instance

        Raises:
            ValueError: If validation fails
        """
        if isinstance(data, cls):
            # Already the correct type, re-validate to ensure all constraints are met
            return cls(**data.model_dump())
        elif isinstance(data, dict):
            # Convert dict to parameter class with validation
            return cls(**data)
        else:
            raise ValueError(
                f"Parameters must be dict or {cls.__name__}, got {type(data)}"
            )


class WorkflowParameters(WorkflowArgsBase):
    """
    Base class for workflow parameters with common validation.

    This class provides centralized validation for common workflow parameters
    like forecast_times (ISO 8601 datetime format validation).
    """

    @field_validator("forecast_times", "start_time", mode="before", check_fields=False)
    @classmethod
    def validate_datetime_list(cls, v: Any, info: Any) -> Any:
        """
        Validate that forecast_times and start_time are in valid ISO 8601 datetime format.

        This validator handles both string lists (forecast_times) and can validate
        datetime objects or ISO 8601 strings (start_time).

        Requires full datetime format with 'T' separator (not just date) for strings.

        Args:
            v: The value (can be list of strings/datetime objects, single value, or None)
            info: Validation context info

        Returns:
            The validated value

        Raises:
            ValueError: If any value is not in valid ISO 8601 datetime format
        """
        from datetime import datetime

        # If None or not present, skip validation
        if v is None:
            return v

        field_name = info.field_name

        # Ensure v is a list
        times = v if isinstance(v, list) else [v]

        # Validate each time value
        for i, time_val in enumerate(times):
            # If already a datetime object, it's valid
            if isinstance(time_val, datetime):
                continue

            # Must be a string if not datetime
            if not isinstance(time_val, str):
                raise ValueError(
                    f"{field_name}[{i}] must be a string or datetime, got {type(time_val).__name__}"
                )

            # Require the 'T' or space separator to ensure it's a datetime, not just a date
            if "T" not in time_val and " " not in time_val:
                raise ValueError(
                    f"{field_name}[{i}] = '{time_val}' is not a valid ISO 8601 datetime format. "
                    f"Expected format with time component like '2024-01-01T00:00:00' or '2024-01-01T00:00:00Z'."
                )

            try:
                # Try to parse as ISO 8601 datetime
                # This handles various ISO 8601 formats including:
                # - 2024-01-01T00:00:00
                # - 2024-01-01T00:00:00Z
                # - 2024-01-01T00:00:00+00:00
                # - 2024-01-01T00:00:00.123456
                datetime.fromisoformat(time_val.replace("Z", "+00:00"))
            except (ValueError, AttributeError) as e:
                raise ValueError(
                    f"{field_name}[{i}] = '{time_val}' is not a valid ISO 8601 datetime format. "
                    f"Expected format like '2024-01-01T00:00:00' or '2024-01-01T00:00:00Z'. "
                    f"Error: {str(e)}"
                ) from e

        return v


class WorkflowConfig(WorkflowArgsBase):
    """
    Specialize in case we want to differentiate parameters and config handling  later.
    """

    pass


class WorkflowProgress(BaseModel):
    """
    Base class for workflow progress tracking.

    This class provides standard progress tracking fields that can be extended
    by custom workflows to include additional tracking information.

    Attributes:
        progress: Human-readable progress message
        current_step: Current step number in the workflow
        total_steps: Total number of steps in the workflow
        error_message: Error message to report specific errors when workflow fails

    Example:
        Basic usage:
            progress = WorkflowProgress(
                progress="Processing data...",
                current_step=3,
                total_steps=10
            )

        Error reporting:
            progress = WorkflowProgress(
                progress="Failed!",
                error_message="Connection timeout: Could not reach data source"
            )

        Extended for custom workflow:
            class MyCustomProgress(WorkflowProgress):
                data_processed_gb: float = 0.0
                error_count: int = 0

            progress = MyCustomProgress(
                progress="Processing batch 5",
                current_step=5,
                total_steps=20,
                data_processed_gb=150.5,
                error_count=2
            )
    """

    progress: str | None = None
    current_step: int | None = None
    total_steps: int | None = None
    error_message: str | None = None


class WorkflowResult(BaseModel):
    """Base result structure for workflow execution"""

    workflow_name: str
    execution_id: str
    status: str
    position: int | None = (
        None  # Queue position (0-indexed), only set when status is QUEUED
    )
    progress: WorkflowProgress | None = None
    start_time: str | None = None
    end_time: str | None = None
    execution_time_seconds: float | None = None
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict)

    def __init__(
        self, workflow_name: str, execution_id: str, status: str, **kwargs: Any
    ) -> None:
        data = {
            "workflow_name": workflow_name,
            "execution_id": execution_id,
            "status": status,
            **kwargs,
        }
        super().__init__(**data)


class Workflow(ABC):
    """
    Base class for custom workflows.

    Users should subclass this class and implement the run() method
    to create custom workflows that will be exposed as FastAPI endpoints.
    """

    name: str
    description: str = ""
    Config: type[WorkflowConfig] = WorkflowConfig
    Parameters: type[WorkflowParameters] = WorkflowParameters

    def __init__(self) -> None:
        """
        Initialize workflow.

        Note: name and description are set by the registry during workflow instantiation.
        """
        self.redis_client: redis.Redis | None = None
        self.output_dir = Path(config.paths.default_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_redis_client(self, redis_client: redis.Redis) -> None:
        """Set Redis client for state management"""
        self.redis_client = redis_client

    @classmethod
    @abstractmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        """
        Validate and convert input parameters to the workflow-specific parameter type.

        This method should use the WorkflowParameters.validate() method to perform
        Pydantic field validation and model validation. Subclasses should implement
        this method to specify their parameter type.

        Args:
            parameters: Input parameters as dict or WorkflowParameters instance

        Returns:
            Validated WorkflowParameters instance of the workflow-specific type

        Raises:
            ValueError: If validation fails

        Example:
            @classmethod
            def validate_parameters(cls, parameters):
                return MyWorkflowParameters.validate(parameters)
        """
        raise NotImplementedError(
            "Subclasses must implement validate_parameters() method"
        )

    @abstractmethod
    def run(
        self, parameters: dict[str, Any] | WorkflowParameters, execution_id: str
    ) -> dict[str, Any]:
        """
        Execute the workflow with given parameters.

        This method should implement the actual workflow logic.
        It will be called when a POST request is made to /v1/infer/{workflow_name}.

        Args:
            parameters: Input parameters for the workflow (dict or WorkflowParameters)
            execution_id: Unique execution identifier

        Returns:
            Dictionary containing workflow results
        """
        raise NotImplementedError("Subclasses must implement run() method")

    def update_execution_data(
        self, execution_id: str, updates: dict[str, Any] | WorkflowProgress
    ) -> None:
        """
        Update execution data in Redis with dict updates or WorkflowProgress object.

        This method can be called by workflow implementations to update specific
        fields in the execution data without replacing the entire record.

        Args:
            execution_id: Unique execution identifier
            updates: Dictionary of fields to update, or WorkflowProgress instance.
                    If a WorkflowProgress object is provided, it will update the
                    'progress' field of WorkflowResult. Dict updates can include
                    any WorkflowResult fields including 'progress', 'metadata', etc.

        Raises:
            RuntimeError: If Redis client is not set

        Example:
            # Using dict for general updates
            self.update_execution_data(execution_id, {
                "metadata": {"custom_field": "value"}
            })

            # Using WorkflowProgress for progress updates
            progress = WorkflowProgress(
                progress="Processing data...",
                current_step=3,
                total_steps=10
            )
            self.update_execution_data(execution_id, progress)

            # Using custom WorkflowProgress subclass
            class MyProgress(WorkflowProgress):
                data_processed_gb: float = 0.0

            progress = MyProgress(
                progress="Batch processing",
                current_step=5,
                total_steps=20,
                data_processed_gb=150.5
            )
            self.update_execution_data(execution_id, progress)
        """
        if not self.redis_client:
            raise RuntimeError("Redis client not set. Call set_redis_client() first.")

        # Get current data to merge updates
        current_data = self._get_execution_data(
            self.redis_client, self.name, execution_id
        )

        # Convert WorkflowProgress object to dict and merge with existing progress
        if isinstance(updates, WorkflowProgress):
            # Get existing progress data (if any)
            existing_progress = {}
            if current_data.progress:
                existing_progress = current_data.progress.model_dump(exclude_none=True)

            # Merge new progress with existing progress
            new_progress = updates.model_dump(exclude_none=True)
            merged_progress = {**existing_progress, **new_progress}
            updates = {"progress": merged_progress}

        # Prevent status from being overwritten by users
        if "status" in updates:
            updates["status"] = current_data.status

        self._update_execution_data(self.redis_client, self.name, execution_id, updates)

    def get_output_path(self, execution_id: str) -> Path:
        """
        Get output path for storing results.

        This method can be called by workflow implementations to get the standard
        output directory path for a specific execution. The directory is automatically
        created if it doesn't exist.

        Args:
            execution_id: Unique execution identifier

        Returns:
            Path object pointing to the output directory
        """
        workflow_dir = self.output_dir / self.name / execution_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        return workflow_dir

    # The following methods are only called by the system, and should not be called by workflow implementations.
    @classmethod
    def _get_execution_data(
        cls, redis_client: redis.Redis, workflow_name: str, execution_id: str
    ) -> WorkflowResult:
        """
        Get the execution data of a workflow execution.

        This method can only be called by the system, and should not be called by workflow implementations.

        Args:
            redis_client: Redis client instance
            workflow_name: Name of the workflow
            execution_id: Unique execution identifier

        Returns:
            WorkflowResult with execution data
        """
        try:
            # Get execution data from Redis
            data = redis_client.get(
                f"workflow_execution:{workflow_name}:{execution_id}"
            )
            if not data:
                raise ValueError(
                    f"Execution {execution_id} not found for workflow {workflow_name}"
                )
            return WorkflowResult.model_validate_json(data)

        except ValueError:
            raise
        except Exception as e:
            logger.exception(f"Failed to get status for {workflow_name}:{execution_id}")
            return WorkflowResult(
                workflow_name=workflow_name,
                execution_id=execution_id,
                status=WorkflowStatus.FAILED,
                error_message=str(e),
            )

    @classmethod
    def _save_execution_data(
        cls,
        redis_client: redis.Redis,
        workflow_name: str,
        execution_id: str,
        data: WorkflowResult,
    ) -> None:
        """
        Save execution data to Redis.

        This method can only be called by the system, and should not be called by workflow implementations.

        Args:
            redis_client: Redis client instance
            workflow_name: Name of the workflow
            execution_id: Unique execution identifier
            data: WorkflowResult to save
        """
        try:
            redis_client.setex(
                f"workflow_execution:{workflow_name}:{execution_id}",
                config.redis.retention_ttl,
                json.dumps(data.model_dump(mode="json"), default=json_serial),
            )
        except Exception:
            logger.exception(
                f"Failed to store execution data for {workflow_name}:{execution_id}"
            )
            raise

    @classmethod
    def _update_execution_data(
        cls,
        redis_client: redis.Redis,
        workflow_name: str,
        execution_id: str,
        updates: dict[str, Any],
    ) -> None:
        """
        Update execution data in Redis with dict updates.

        This method can only be called by the system, and should not be called by workflow implementations.

        Args:
            redis_client: Redis client instance
            workflow_name: Name of the workflow
            execution_id: Unique execution identifier
            updates: Dictionary of fields to update
        """
        try:
            # Get current data
            current_data_json = redis_client.get(
                f"workflow_execution:{workflow_name}:{execution_id}"
            )
            if current_data_json:
                current_data = json.loads(current_data_json)
                current_data.update(updates)
                redis_client.setex(
                    f"workflow_execution:{workflow_name}:{execution_id}",
                    config.redis.retention_ttl,
                    json.dumps(current_data, default=json_serial),
                )
        except Exception:
            logger.exception(
                f"Failed to update execution data for {workflow_name}:{execution_id}"
            )
            raise


def json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return obj.total_seconds()
    raise TypeError(f"Type {type(obj)} is not serializable")


class WorkflowRegistry:
    """Registry for managing custom workflows"""

    def __init__(self, **config: Any) -> None:
        self._workflows: dict[str, type[Workflow]] = {}
        self._workflow_instances: dict[str, Workflow] = {}

    def register(self, workflow_class: type[Workflow]) -> type[Workflow]:
        """
        Register a workflow class.

        Args:
            workflow_class: Workflow class (not instance) to register

        Returns:
            workflow_class (enabling the use of ``register`` as a decorator)

        Raises:
            TypeError: If workflow_class is not a class or not a Workflow subclass
            ValueError: If workflow name is already registered
        """
        # Skip registration in a non-API environment
        api_server_running = os.environ.get("EARTH2STUDIO_API_ACTIVE", "0") == "1"
        if __name__ == "__main__" or not api_server_running:
            return workflow_class

        # Validate that a class (not instance) was passed
        if not isinstance(workflow_class, type):
            raise TypeError(
                f"Expected a Workflow class, but got an instance of {type(workflow_class).__name__}. "
                f"Use MyWorkflow (the class) not MyWorkflow() (an instance)."
            )

        if not issubclass(workflow_class, Workflow):
            raise TypeError(f"{workflow_class.__name__} must be a subclass of Workflow")

        workflow_name = workflow_class.name

        # Prevent reserved workflow names that would conflict with API endpoints
        if workflow_name == "workflows":
            raise ValueError(
                f"Workflow name '{workflow_name}' is reserved and cannot be used. "
                f"It would conflict with the /v1/infer/workflows endpoint."
            )

        if workflow_name in self._workflows:
            raise ValueError(f"Workflow '{workflow_name}' is already registered")

        self._workflows[workflow_name] = workflow_class
        logger.info(f"Registered workflow: {workflow_name}")

        return workflow_class

    def unregister(self, name: str) -> None:
        """Unregister a workflow"""
        if name not in self._workflows:
            raise ValueError(f"Workflow '{name}' is not registered")

        del self._workflows[name]
        # Clear cached instance if it exists
        if name in self._workflow_instances:
            del self._workflow_instances[name]
        logger.info(f"Unregistered workflow: {name}")

    def get_workflow_class(self, name: str) -> type[Workflow] | None:
        """
        Get a workflow class by name.

        Args:
            name: Name of the workflow

        Returns:
            Workflow class or None if not found
        """
        return self._workflows.get(name)

    def get(
        self, name: str, redis_client: redis.Redis | None = None
    ) -> Workflow | None:
        """
        Get a cached instance of a workflow by name.

        Creates a new instance on first call and caches it for subsequent calls.
        The instance is automatically initialized with the registered name and description.

        Args:
            name: Name of the workflow
            redis_client: Optional Redis client to set on the instance

        Returns:
            Cached workflow instance or None if not found
        """
        workflow_class = self.get_workflow_class(name)
        if workflow_class is None:
            return None

        # Check cache first
        if name in self._workflow_instances:
            instance = self._workflow_instances[name]
        else:
            # Get workflow config
            config = get_workflow_config(name)
            try:
                workflow_class.Config.validate(config)
            except Exception as e:
                raise ValueError(
                    f"Invalid parameters for {workflow_class.__name__}: {e}"
                ) from e
            # Create a new instance
            instance = workflow_class(**config)
            self._workflow_instances[name] = instance

        # Set Redis client if provided (update even for cached instances)
        if redis_client is not None:
            instance.set_redis_client(redis_client)

        return instance

    def list_workflows(self) -> dict[str, str]:
        """List all registered workflows"""
        return {
            name: workflow_class.description
            for name, workflow_class in self._workflows.items()
        }

    def discover_and_register_from_directories(
        self, workflow_dirs: list, include_builtin: bool = True
    ) -> tuple:
        """
        Discover and register workflows from specified directories.

        Args:
            workflow_dirs: List of directory paths to search for workflow modules
            include_builtin: Whether to automatically include built-in example_workflows

        Returns:
            tuple: (successful_imports, failed_imports) counts
        """
        # Always include built-in workflows if requested
        if include_builtin:
            builtin_workflows_dir = Path(__file__).parent.parent / "example_workflows"
            already_included = builtin_workflows_dir in {Path(d) for d in workflow_dirs}
            if builtin_workflows_dir.exists() and not already_included:
                # Add to the beginning so built-in workflows are discovered first
                workflow_dirs.insert(0, str(builtin_workflows_dir))
                logger.info(
                    f"Including built-in example workflows from: {builtin_workflows_dir}"
                )

        successful_imports = 0
        failed_imports = 0
        logger.info(f"Workflow directories: {workflow_dirs}")

        for directory in workflow_dirs:
            dir_path = Path(directory)

            if not dir_path.exists():
                logger.warning(f"Workflow directory does not exist: {directory}")
                continue

            if not dir_path.is_dir():
                logger.warning(f"Path is not a directory: {directory}")
                continue

            logger.info(f"Discovering workflows in: {directory}")

            # Find all Python files in the directory (non-recursively)
            python_files = list(dir_path.glob("*.py"))

            # Filter out __init__.py and __pycache__ files
            python_files = [
                f
                for f in python_files
                if f.name != "__init__.py" and "__pycache__" not in f.parts
            ]

            logger.info(f"Found {len(python_files)} Python file(s) in {directory}")

            for py_file in python_files:
                try:
                    # Create a unique module name based on the file path
                    rel_path = py_file.relative_to(dir_path)
                    module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

                    logger.info(
                        f"Importing workflow module: {py_file.name} as {module_name}"
                    )

                    # Add the directory to sys.path for pickle compatibility
                    # (torch.compile requires the module to be importable by name)
                    dir_path_str = str(dir_path)
                    if dir_path_str not in sys.path:
                        sys.path.insert(0, dir_path_str)

                    # Load the module dynamically
                    spec = importlib.util.spec_from_file_location(module_name, py_file)
                    if spec is None or spec.loader is None:
                        logger.error(f"Could not load spec for {py_file}")
                        failed_imports += 1
                        continue

                    module = importlib.util.module_from_spec(spec)

                    # Add to sys.modules to make it importable
                    sys.modules[module_name] = module

                    # Execute the module (this should trigger registration)
                    spec.loader.exec_module(module)

                    logger.info(
                        f"Successfully imported workflow module: {py_file.name}"
                    )
                    successful_imports += 1

                except Exception:
                    logger.exception(f"Failed to import workflow from {py_file}")
                    failed_imports += 1

        return successful_imports, failed_imports

    def auto_register_workflows(self, redis_client: redis.Redis) -> None:
        """
        Auto-discover and register all workflows, then set Redis client.

        This is the main entry point for workflow registration during server startup.

        Args:
            redis_client: Redis client instance to set on registered workflows
        """
        try:
            logger.info("=" * 60)
            logger.info("Auto-discovering workflows...")
            logger.info("=" * 60)

            # Parse user-provided workflow directories from environment variables
            workflow_dirs = parse_workflow_directories_from_env()

            # Discover and register workflows (includes built-in by default)
            # Always try to discover built-in workflows
            logger.info(
                f"Searching for workflows in {len(workflow_dirs) + 1} directory/directories"
            )
            successful, failed = self.discover_and_register_from_directories(
                workflow_dirs, include_builtin=True
            )

            logger.info("=" * 60)
            logger.info(
                f"Workflow discovery complete: {successful} successful, {failed} failed"
            )
            logger.info("=" * 60)

            # Log summary of registered workflows
            registered_workflows = self.list_workflows()
            logger.info("=" * 60)
            logger.info(f"Total registered workflows: {len(registered_workflows)}")
            for name, description in registered_workflows.items():
                logger.info(f"  {name}: {description}")
            logger.info("=" * 60)

        except Exception:
            logger.exception("Failed to register workflows")
            raise


# Helper function for parsing environment variables
def parse_workflow_directories_from_env() -> list[str]:
    """
    Parse workflow directories from environment variables.

    Supports WORKFLOW_DIR environment variable which can contain:
    - Single directory path
    - Comma-separated list of directories
    - Colon-separated list of directories

    Returns:
        list: List of directory paths to search for workflows
    """
    workflow_dirs = []

    # Check WORKFLOW_DIR environment variable
    env_value = os.environ.get("WORKFLOW_DIR")
    if env_value:
        # Support both comma and colon as separators
        separators = [",", ":"]
        dirs = [env_value]

        for sep in separators:
            if sep in env_value:
                dirs = [d.strip() for d in env_value.split(sep) if d.strip()]
                break

        workflow_dirs.extend(dirs)
        logger.info(f"Found {len(dirs)} directory/directories in WORKFLOW_DIR: {dirs}")

    # Remove duplicates while preserving order
    workflow_dirs = list(dict.fromkeys(workflow_dirs))

    return workflow_dirs


# Global workflow registry instance
workflow_registry = WorkflowRegistry()


# Convenience function for backward compatibility and ease of use
def register_all_workflows(redis_client: redis.Redis) -> None:
    """
    Register all workflows (built-in and user-provided).

    This is a convenience function that calls the registry's auto_register_workflows method.
    It auto-discovers workflows from:
    1. Built-in example_workflows directory (always included)
    2. User directories specified via WORKFLOW_DIR environment variable

    Args:
        redis_client: Redis client instance to set on registered workflows

    Example:
        >>> import redis
        >>> from api_server.workflow import register_all_workflows
        >>> redis_client = redis.Redis(host='localhost', port=6379)
        >>> register_all_workflows(redis_client)
    """
    workflow_registry.auto_register_workflows(redis_client)

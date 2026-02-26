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

import inspect
import logging
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar, get_type_hints

import torch
import zarr
from pydantic import Field, create_model

from api_server.config import get_config
from earth2studio.io import IOBackend, NetCDF4Backend, ZarrBackend
from earth2studio.utils.type import CoordSystem

from .workflow import Workflow, WorkflowConfig, WorkflowParameters, WorkflowProgress


def _convert_param(param: inspect.Parameter, typeinfo: type) -> tuple[type, Field]:
    """Convert parameter to Pydantic Field declaration with validation constraints"""
    field_kwargs = {}
    if param.default is not param.empty:
        field_kwargs["default"] = param.default

    # Add validation constraints for specific parameter names and types
    # This ensures reasonable bounds for numeric workflow parameters
    if param.name in ("num_steps", "nsteps"):
        # Steps parameters should be positive and have reasonable upper bounds
        field_kwargs["ge"] = 1  # At least 1 step
        field_kwargs["le"] = 1000  # Maximum 1000 steps to prevent resource exhaustion
        field_kwargs["description"] = "Number of steps (must be between 1 and 1000)"

    return (typeinfo, Field(**field_kwargs))


def func_to_model(
    function: Callable,
    model_name: str = "Parameters",
    exclude_params: set = set(),
    base: type[WorkflowParameters] = WorkflowParameters,
) -> type[WorkflowParameters]:
    """Create a Pydantic model from function call signature"""
    params = dict(inspect.signature(function).parameters)
    typeinfo = get_type_hints(function)
    exclude_params = exclude_params | {"self", "args", "kwargs"}
    converted_params = {
        name: _convert_param(param, typeinfo[name])
        for (name, param) in params.items()
        if name not in exclude_params
    }
    return create_model(model_name, __base__=base, **converted_params)


class AutoParameters(ABCMeta):
    """Automatically:
    - Assign the ``name`` attribute of the class to ``__name__`` if not set.
    - Create Pydantic model corresponding to the ``__init__`` signature of the class
      and assign it to the Config attribute.
    - Create Pydantic model corresponding to the ``__call__`` signature of the class
      and assign it to the Parameters attribute.
    """

    RESERVED_PARAMS: ClassVar[set[str]] = {"io"}

    def __new__(cls: type, clsname: str, bases: tuple, attrs: dict) -> type:
        mod_attrs = attrs.copy()
        if "name" not in attrs:
            mod_attrs["name"] = clsname
        mod_attrs["Config"] = func_to_model(
            attrs["__init__"], model_name=f"{clsname}Config", base=WorkflowConfig
        )
        mod_attrs["Parameters"] = func_to_model(
            attrs["__call__"],
            model_name=f"{clsname}Parameters",
            exclude_params=AutoParameters.RESERVED_PARAMS,
            base=WorkflowParameters,
        )
        return super().__new__(cls, clsname, bases, mod_attrs)  # type: ignore[misc]


class Earth2Workflow(Workflow, metaclass=AutoParameters):
    """Base class for implementing workflows in Earth2Studio that can be served as APIs."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, io: IOBackend) -> None:
        """Subclasses must implement the workflow logic in the __call__ method.
        The method must accept an IOBackend in the ``io`` argument to store
        the results.
        """
        pass

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return cls.Parameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters for {cls.__name__}: {e}") from e

    def run(
        self, parameters: dict[str, Any] | WorkflowParameters, execution_id: str
    ) -> dict[str, Any]:
        """Run custom workflow"""

        # Validate and convert parameters
        parameters = self.validate_parameters(parameters)

        # Initialize metadata for tracking
        metadata = {"parameters": parameters.model_dump()}

        try:
            # Store metadata separately
            self.update_execution_data(execution_id, {"metadata": metadata})

            # Import Earth2Studio components
            progress = WorkflowProgress(progress="Initializing workflow")
            self.update_execution_data(execution_id, progress)

            # Configure output
            output_dir = self.get_output_path(execution_id)
            output_format = get_config().paths.output_format
            if output_format == "zarr":
                output_path = str(output_dir / "results.zarr")
                results_io: IOBackend = ZarrBackend(output_path)
            elif output_format == "netcdf4":
                output_path = str(output_dir / "results.nc")
                results_io = NetCDF4Backend(output_path)  # type: ignore[assignment]
            else:
                raise ValueError(
                    f"Unsupported output format: {output_format}. Supported formats are 'zarr' and 'netcdf4'"
                )
            results_io = BackendProgress(results_io, self, execution_id)

            # Run the forecast!
            progress = WorkflowProgress(progress="Starting workflow")
            self.update_execution_data(execution_id, progress)
            self(io=results_io, **dict(parameters))

            # Consolidate zarr metadata for faster remote access
            if output_format == "zarr":
                zarr.consolidate_metadata(output_path)

            # Update final metadata and progress
            progress = WorkflowProgress(progress="Finished workflow successfully")
            self.update_execution_data(execution_id, progress)

            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": "Generated forecast",
                    }
                },
            )

            return {"status": "success"}

        except Exception as e:
            # Update error progress and metadata
            progress = WorkflowProgress(
                progress="Workflow failed", error_message=str(e)
            )
            self.update_execution_data(execution_id, progress)
            raise


logger = logging.getLogger(__name__)


class BackendProgress:
    """Wrap an IOBackend to automatically update writing progress to API server."""

    def __init__(
        self,
        io: IOBackend,
        workflow: Workflow,
        execution_id: str,
        progress_dim: str = "lead_time",
    ) -> None:
        self.io = io
        self.workflow = workflow
        self.execution_id = execution_id
        self.progress_dim = progress_dim
        self.progress_coords: list[Any] | None = None
        self.progress_array: str | None = None

    def add_array(
        self,
        coords: CoordSystem,
        array_name: str | list[str],
        **kwargs: dict[str, Any],
    ) -> None:
        """Add array and initialize progress report."""
        self.io.add_array(coords, array_name, **kwargs)
        if (
            self.progress_dim in coords and self.progress_array is None
        ):  # perform updates only with one array to avoid redundant updates
            self.progress_array = (
                array_name if isinstance(array_name, str) else array_name[0]
            )
            self.progress_coords = list(coords[self.progress_dim].copy())
            # Initialize progress tracking using WorkflowProgress
            if self.progress_coords is not None:
                progress = WorkflowProgress(
                    current_step=0, total_steps=len(self.progress_coords)
                )
                self.workflow.update_execution_data(self.execution_id, progress)

    def write(
        self,
        x: torch.Tensor | list[torch.Tensor],
        coords: CoordSystem,
        array_name: str | list[str],
    ) -> None:
        """Write results and log progress."""
        self.io.write(x, coords, array_name)
        if not isinstance(array_name, str):
            array_name = array_name[0]
        if array_name == self.progress_array and self.progress_coords is not None:
            current_coord = coords[self.progress_dim][-1]
            step_index = self.progress_coords.index(current_coord)
            # Update progress using WorkflowProgress
            progress = WorkflowProgress(current_step=step_index + 1)
            self.workflow.update_execution_data(self.execution_id, progress)

    def __getattr__(self, name: str) -> Any:
        """Allow passthrough of unwrapped attributes."""
        return getattr(self.io, name)

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
Deterministic FCN Workflow Custom Pipeline

This pipeline implements the deterministic workflow from examples/01_deterministic_workflow.py
as a custom pipeline that can be invoked via the REST API.

It loads the FCN model once and stores it in the instance.
"""

import json
import logging
from typing import Any, Literal

import zarr
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


class DeterministicFCNWorkflowParameters(WorkflowParameters):
    """Parameters for the deterministic workflow"""

    # Forecast configuration
    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    nsteps: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Number of forecast steps (each step is 6 hours for FCN)",
    )

    # Data source configuration
    data_source: Literal["gfs"] = Field(
        default="gfs", description="Data source for initialization (currently gfs)"
    )

    # IO configuration
    output_format: Literal["zarr"] = Field(
        default="zarr", description="Output format (currently zarr)"
    )

    # Post-processing options
    create_plots: bool = Field(
        default=True, description="Whether to create visualization plots"
    )
    plot_variable: Literal["t2m", "msl", "u10m", "v10m", "tcwv", "z500"] = Field(
        default="t2m",
        description="Variable to plot (t2m=temperature, msl=pressure, u10m/v10m=wind, tcwv=water vapor, z500=geopotential)",
    )
    plot_step: int = Field(
        default=4,
        ge=0,
        description="Forecast step to plot (step 4 = 24 hours for FCN)",
    )


@workflow_registry.register
class DeterministicFCNWorkflow(Workflow):
    """
    Deterministic workflow that runs Earth2Studio deterministic forecasts.

    This workflow:
    1. Loads a prognostic model (FCN)
    2. Sets up a data source (GFS, ERA5)
    3. Runs deterministic forecast
    4. Saves results in specified format
    5. Optionally creates visualization plots
    """

    name = "deterministic_fcn_workflow"
    description = "Earth2Studio deterministic forecast workflow with FCN model"
    Parameters = DeterministicFCNWorkflowParameters

    def __init__(self) -> None:
        super().__init__()
        from earth2studio.models.px import FCN

        # load the model once and store it in the instance
        self.package = FCN.load_default_package()
        self.model = FCN.load_model(self.package)

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | DeterministicFCNWorkflowParameters
    ) -> DeterministicFCNWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return DeterministicFCNWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | DeterministicFCNWorkflowParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the deterministic workflow pipeline"""

        # Validate and convert parameters to the correct type
        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            # Store metadata separately
            self.update_execution_data(execution_id, {"metadata": metadata})

            # Import Earth2Studio components
            progress = WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=4,
            )
            self.update_execution_data(execution_id, progress)

            from earth2studio import run
            from earth2studio.data import GFS
            from earth2studio.io import ZarrBackend

            # Set up data source
            progress = WorkflowProgress(
                progress=f"Setting up {parameters.data_source} data source...",
                current_step=2,
                total_steps=4,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.data_source.lower() == "gfs":
                data = GFS()
            else:
                raise ValueError(f"Unsupported data source: {parameters.data_source}")

            # Set up IO backend
            output_dir = self.get_output_path(execution_id)
            if parameters.output_format.lower() == "zarr":
                io = ZarrBackend(file_name=str(output_dir / "results.zarr"))
            else:
                raise ValueError(
                    f"Unsupported output format: {parameters.output_format}"
                )

            # Run deterministic workflow
            progress = WorkflowProgress(
                progress=f"Running deterministic forecast ({parameters.nsteps} steps)...",
                current_step=3,
                total_steps=4,
            )
            self.update_execution_data(execution_id, progress)

            # Execute the workflow
            io_result = run.deterministic(  # type: ignore[assignment]
                parameters.forecast_times, parameters.nsteps, self.model, data, io
            )
            io = io_result  # type: ignore[assignment]

            # Consolidate zarr metadata for faster remote access
            if parameters.output_format.lower() == "zarr":
                zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            # Save metadata about the forecast
            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "nsteps": parameters.nsteps,
                "model_type": "FCN",
                "data_source": parameters.data_source,
                "output_format": parameters.output_format,
            }

            # Save forecast metadata
            output_dir = self.get_output_path(execution_id)
            metadata_path = output_dir / "forecast_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(forecast_info, f, indent=2)

            # Create visualization plots if requested
            if parameters.create_plots:
                progress = WorkflowProgress(
                    progress="Creating visualization plots...",
                    current_step=4,
                    total_steps=4,
                )
                self.update_execution_data(execution_id, progress)
                self.create_forecast_plot(io, parameters, execution_id)

            # Update completion status
            progress = WorkflowProgress(
                progress="Complete!", current_step=4, total_steps=4
            )
            self.update_execution_data(execution_id, progress)

            # Update final metadata with results summary
            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": f"Generated {parameters.nsteps}-step forecast for {len(parameters.forecast_times)} time(s)",
                        "forecast_info": forecast_info,
                    }
                },
            )

            return {
                "status": "success",
                "output_path": str(output_dir),
                "forecast_info": forecast_info,
            }

        except Exception as e:
            # Mark workflow as failed
            progress = WorkflowProgress(progress="Failed!", error_message=str(e))
            self.update_execution_data(execution_id, progress)
            raise e

    def create_forecast_plot(
        self, io: Any, parameters: DeterministicFCNWorkflowParameters, execution_id: str
    ) -> None:
        """Create a forecast visualization plot"""
        try:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt

            # Create plot
            forecast_time = parameters.forecast_times[0]
            variable = parameters.plot_variable
            step = min(parameters.plot_step, parameters.nsteps - 1)

            plt.close("all")

            # Create a Robinson projection
            projection = ccrs.Robinson()

            # Create a figure and axes with the specified projection
            _, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(12, 8))

            # Get data from the IO object
            lon = io["lon"][:]
            lat = io["lat"][:]

            # Handle the case where the variable might not exist
            if variable in io:
                data = io[variable][0, step]  # First forecast time, specified step
            else:
                # Fallback to first available variable
                available_vars = [
                    key for key in io if key not in ["lon", "lat", "time"]
                ]
                if available_vars:
                    variable = available_vars[0]
                    data = io[variable][0, step]
                else:
                    raise ValueError("No data variables found in forecast output")

            # Plot the field using pcolormesh
            im = ax.pcolormesh(
                lon,
                lat,
                data,
                transform=ccrs.PlateCarree(),
                cmap="Spectral_r",
            )

            # Add colorbar
            cbar = plt.colorbar(
                im, ax=ax, orientation="horizontal", pad=0.1, shrink=0.8
            )
            cbar.set_label(f"{variable}")

            # Calculate lead time in hours (assuming 6-hour steps for FCN)
            lead_time_hours = step * 6

            # Set title
            ax.set_title(
                f"{forecast_time} + {lead_time_hours}hrs - {variable}", fontsize=14
            )

            # Add coastlines and gridlines
            ax.coastlines()
            ax.gridlines(alpha=0.5)

            # Save plot
            output_dir = self.get_output_path(execution_id)
            plot_path = output_dir / f"forecast_plot_{variable}_step{step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            # Log the error but don't fail the entire pipeline
            logger.exception("Could not create forecast plot")
            raise

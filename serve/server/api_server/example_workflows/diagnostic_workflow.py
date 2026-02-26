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
Diagnostic Custom Workflow

This workflow implements the recipe examples/02_diagnostic_workflow.py
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


class DiagnosticWorkflowParameters(WorkflowParameters):
    """Parameters for the diagnostic workflow"""

    # Forecast configuration
    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO format)",
    )
    nsteps: int = Field(
        default=6,
        ge=1,
        le=100,
        description="Number of forecast steps",
    )

    # Prognostic Model configuration
    prognostic_model_type: Literal["dlwp", "fcn"] = Field(
        default="fcn", description="Prognostic model type (dlwp, fcn)"
    )

    # Diagnostic Model configuration
    diagnostic_model_type: Literal["precipitation_afno"] = Field(
        default="precipitation_afno",
        description="Diagnostic model type (currently precipitation_afno)",
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
    plot_variable: Literal["tp", "t2m", "msl", "u10m", "v10m", "tcwv", "z500"] = Field(
        default="tp",
        description="Variable to plot (tp=precipitation, t2m=temperature, msl=pressure, u10m/v10m=wind, tcwv=water vapor, z500=geopotential)",
    )
    plot_step: int = Field(
        default=4,
        ge=0,
        description="Forecast step to plot (step 4 = 24 hours for DLWP)",
    )


@workflow_registry.register
class DiagnosticWorkflow(Workflow):
    """
    Diagnostic workflow that runs Earth2Studio diagnostic forecasts.

    This workflow:
    1. Loads a prognostic model (DLWP, FCN, etc.)
    2. Loads a diagnostic model (e.g. precipitation_afno)
    3. Sets up a data source (GFS, ERA5)
    4. Runs diagnostic forecast
    5. Saves results in specified format
    6. Optionally creates visualization plots
    """

    name = "diagnostic_workflow"
    description = "Earth2Studio diagnostic forecast workflow with visualization"
    Parameters = DiagnosticWorkflowParameters

    # No __init__ needed - name and description are set by the registry during registration

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | DiagnosticWorkflowParameters
    ) -> DiagnosticWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return DiagnosticWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | DiagnosticWorkflowParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the diagnostic workflow pipeline"""

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
                total_steps=7,
            )
            self.update_execution_data(execution_id, progress)

            from earth2studio import run
            from earth2studio.data import GFS
            from earth2studio.io import ZarrBackend
            from earth2studio.models.dx import PrecipitationAFNO
            from earth2studio.models.px import DLWP, FCN

            # Load prognostic model
            progress = WorkflowProgress(
                progress=f"Loading {parameters.prognostic_model_type} prognostic model...",
                current_step=2,
                total_steps=7,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.prognostic_model_type.lower() == "dlwp":
                package = DLWP.load_default_package()
                prognostic_model = DLWP.load_model(package)
            elif parameters.prognostic_model_type.lower() == "fcn":
                package = FCN.load_default_package()
                prognostic_model = FCN.load_model(package)
            else:
                raise ValueError(
                    f"Unsupported prognostic model type: {parameters.prognostic_model_type}"
                )

            # Load diagnostic model
            progress = WorkflowProgress(
                progress=f"Loading {parameters.diagnostic_model_type} diagnostic model...",
                current_step=3,
                total_steps=7,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.diagnostic_model_type.lower() == "precipitation_afno":
                package = PrecipitationAFNO.load_default_package()
                diagnostic_model = PrecipitationAFNO.load_model(package)
            else:
                raise ValueError(
                    f"Unsupported diagnostic model type: {parameters.diagnostic_model_type}"
                )

            # Set up data source
            progress = WorkflowProgress(
                progress=f"Setting up {parameters.data_source} data source...",
                current_step=4,
                total_steps=7,
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

            # Run diagnostic workflow
            progress = WorkflowProgress(
                progress=f"Running diagnostic forecast ({parameters.nsteps} steps)...",
                current_step=5,
                total_steps=7,
            )
            self.update_execution_data(execution_id, progress)

            # Execute the workflow
            io = run.diagnostic(  # type: ignore[assignment]
                parameters.forecast_times,
                parameters.nsteps,
                prognostic_model,
                diagnostic_model,
                data,
                io,
            )

            # Consolidate zarr metadata for faster remote access
            if parameters.output_format.lower() == "zarr":
                zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            # Save metadata about the forecast
            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "nsteps": parameters.nsteps,
                "prognostic_model_type": parameters.prognostic_model_type,
                "diagnostic_model_type": parameters.diagnostic_model_type,
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
                    current_step=6,
                    total_steps=7,
                )
                self.update_execution_data(execution_id, progress)
                self.create_forecast_plot(io, parameters, execution_id)

            # Update completion status
            progress = WorkflowProgress(
                progress="Complete!", current_step=7, total_steps=7
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
            raise

    def create_forecast_plot(
        self, io: Any, parameters: DiagnosticWorkflowParameters, execution_id: str
    ) -> None:
        """Create a forecast visualization plot"""
        try:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt
            import numpy as np

            # Create plot
            forecast_time = parameters.forecast_times[0]
            variable = parameters.plot_variable
            step = min(parameters.plot_step, parameters.nsteps - 1)

            plt.close("all")

            # Create a Robinson projection
            projection = ccrs.Orthographic(-100, 40)

            # Create a figure and axes with the specified projection
            _, ax = plt.subplots(subplot_kw={"projection": projection}, figsize=(10, 6))

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

            # Plot the field using contourf
            levels = np.arange(0.0, 0.01, 0.001)

            im = ax.contourf(
                lon,
                lat,
                data,
                levels,
                transform=ccrs.PlateCarree(),
                vmax=0.01,
                vmin=0.00,
                cmap="terrain",
            )

            # Add colorbar
            plt.colorbar(
                im,
                ax=ax,
                ticks=levels,
                shrink=0.75,
                pad=0.04,
                label="Total precipitation (m)",
            )

            # Calculate lead time in hours (assuming 6-hour steps for DLWP)
            lead_time_hours = step * 6

            # Set title
            ax.set_title(
                f"{forecast_time} + {lead_time_hours}hrs - {variable}", fontsize=14
            )

            # Add coastlines and gridlines
            ax.set_extent([220, 340, 20, 70])  # [lat min, lat max, lon min, lon max]
            ax.coastlines()
            ax.gridlines()

            # Save plot
            output_dir = self.get_output_path(execution_id)
            plot_path = output_dir / f"forecast_plot_{variable}_step{step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            logger.exception("Could not create forecast plot")
            raise

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
Ensemble Workflow Custom Pipeline

This pipeline implements the ensemble workflow from examples/03_ensemble_workflow.py
as a custom pipeline that can be invoked via the REST API.
"""

import json
import logging
from collections import OrderedDict
from typing import Any, Literal

import numpy as np
import zarr
from pydantic import Field

from earth2studio.serve.server.workflow import (
    Workflow,
    WorkflowParameters,
    WorkflowProgress,
    workflow_registry,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnsembleWorkflowParameters(WorkflowParameters):
    """Parameters for the ensemble workflow"""

    # Forecast configuration
    forecast_times: list[str] = Field(
        default=["2024-01-01T00:00:00"],
        description="List of forecast initialization times (ISO 8601 datetime format, e.g. '2024-01-01T00:00:00')",
    )
    nsteps: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of forecast steps (each step is 6 hours for FCN)",
    )
    nensemble: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Number of ensemble members",
    )
    batch_size: int = Field(
        default=2,
        ge=1,
        le=32,
        description="Number of ensemble members per batch",
    )

    # Model configuration
    model_type: Literal["fcn"] = Field(
        default="fcn",
        description="Prognostic model type (ensemble workflow currently supports fcn)",
    )

    # Perturbation configuration (SphericalGaussian)
    noise_amplitude: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Noise amplitude for SphericalGaussian perturbation",
    )

    # Data source configuration
    data_source: Literal["gfs"] = Field(
        default="gfs", description="Data source for initialization (currently gfs)"
    )

    # IO configuration
    output_format: Literal["zarr"] = Field(
        default="zarr", description="Output format (currently zarr)"
    )
    output_variables: list[str] | None = Field(
        default=None,
        description="Variables to save (e.g. ['t2m', 'tcwv']). If None, save all.",
    )

    # Post-processing options
    create_plots: bool = Field(
        default=True, description="Whether to create visualization plots"
    )
    plot_variable: Literal["t2m", "msl", "u10m", "v10m", "tcwv", "z500"] = Field(
        default="tcwv",
        description="Variable to plot (t2m, msl, u10m, v10m, tcwv, z500)",
    )
    plot_step: int = Field(
        default=4,
        ge=0,
        description="Forecast step to plot (step 4 = 24 hours for FCN)",
    )


@workflow_registry.register
class EnsembleWorkflow(Workflow):
    """
    Ensemble workflow that runs Earth2Studio ensemble forecasts.

    This workflow:
    1. Loads a prognostic model (FCN)
    2. Sets up SphericalGaussian perturbation
    3. Sets up a data source (GFS)
    4. Runs ensemble forecast with multiple members
    5. Saves results in zarr format
    6. Optionally creates visualization plots (members + std)
    """

    name = "ensemble_workflow"
    description = (
        "Earth2Studio ensemble forecast workflow with perturbation and visualization"
    )
    Parameters = EnsembleWorkflowParameters

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | EnsembleWorkflowParameters
    ) -> EnsembleWorkflowParameters:
        """Validate and convert input parameters"""
        try:
            return EnsembleWorkflowParameters.validate(parameters)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}") from e

    def run(
        self,
        parameters: dict[str, Any] | EnsembleWorkflowParameters,
        execution_id: str,
    ) -> dict[str, Any]:
        """Run the ensemble workflow pipeline"""

        parameters = self.validate_parameters(parameters)
        metadata = {"parameters": parameters.model_dump()}

        try:
            self.update_execution_data(execution_id, {"metadata": metadata})

            progress = WorkflowProgress(
                progress="Importing Earth2Studio components...",
                current_step=1,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            from earth2studio import run
            from earth2studio.data import GFS
            from earth2studio.io import ZarrBackend
            from earth2studio.models.px import FCN
            from earth2studio.perturbation import SphericalGaussian

            # Load prognostic model
            progress = WorkflowProgress(
                progress=f"Loading {parameters.model_type} model...",
                current_step=2,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            if parameters.model_type.lower() == "fcn":
                package = FCN.load_default_package()
                model = FCN.load_model(package)
            else:
                raise ValueError(f"Unsupported model type: {parameters.model_type}")

            # Perturbation method
            progress = WorkflowProgress(
                progress="Setting up perturbation method...",
                current_step=3,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)
            sg = SphericalGaussian(noise_amplitude=parameters.noise_amplitude)

            # Data source
            if parameters.data_source.lower() == "gfs":
                data = GFS()
            else:
                raise ValueError(f"Unsupported data source: {parameters.data_source}")

            # IO backend
            output_dir = self.get_output_path(execution_id)
            chunks = {"ensemble": 1, "time": 1, "lead_time": 1}
            io = ZarrBackend(
                file_name=str(output_dir / "results.zarr"),
                chunks=chunks,
                backend_kwargs={"overwrite": True},
            )

            # Optional output coordinate filter (variable subset)
            output_coords: OrderedDict = OrderedDict()
            if parameters.output_variables:
                output_coords = OrderedDict(
                    {"variable": np.array(parameters.output_variables)}
                )

            # Run ensemble workflow
            progress = WorkflowProgress(
                progress=f"Running ensemble forecast ({parameters.nensemble} members, {parameters.nsteps} steps)...",
                current_step=4,
                total_steps=6,
            )
            self.update_execution_data(execution_id, progress)

            io_result = run.ensemble(  # type: ignore[assignment]
                parameters.forecast_times,
                parameters.nsteps,
                parameters.nensemble,
                model,
                data,
                io,
                sg,
                batch_size=parameters.batch_size,
                output_coords=output_coords,
            )
            io = io_result  # type: ignore[assignment]

            # Consolidate zarr metadata
            zarr.consolidate_metadata(str(output_dir / "results.zarr"))

            forecast_info = {
                "forecast_times": parameters.forecast_times,
                "nsteps": parameters.nsteps,
                "nensemble": parameters.nensemble,
                "batch_size": parameters.batch_size,
                "model_type": parameters.model_type,
                "noise_amplitude": parameters.noise_amplitude,
                "data_source": parameters.data_source,
                "output_format": parameters.output_format,
            }

            metadata_path = output_dir / "forecast_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(forecast_info, f, indent=2)

            if parameters.create_plots:
                progress = WorkflowProgress(
                    progress="Creating visualization plots...",
                    current_step=5,
                    total_steps=6,
                )
                self.update_execution_data(execution_id, progress)
                self.create_ensemble_plot(io, parameters, execution_id)

            progress = WorkflowProgress(
                progress="Complete!", current_step=6, total_steps=6
            )
            self.update_execution_data(execution_id, progress)

            self.update_execution_data(
                execution_id,
                {
                    "metadata": {
                        **metadata,
                        "results_summary": (
                            f"Generated {parameters.nensemble}-member, "
                            f"{parameters.nsteps}-step ensemble for "
                            f"{len(parameters.forecast_times)} time(s)"
                        ),
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
            progress = WorkflowProgress(progress="Failed!", error_message=str(e))
            self.update_execution_data(execution_id, progress)
            raise

    def create_ensemble_plot(
        self,
        io: Any,
        parameters: EnsembleWorkflowParameters,
        execution_id: str,
    ) -> None:
        """Create ensemble visualization (two members + std)."""
        try:
            import cartopy.crs as ccrs
            import matplotlib.pyplot as plt

            variable = parameters.plot_variable
            step = min(parameters.plot_step, parameters.nsteps - 1)
            forecast_time = parameters.forecast_times[0]

            if variable not in io:
                available = [
                    k for k in io if k not in ["lon", "lat", "time", "lead_time"]
                ]
                variable = available[0] if available else "t2m"

            plt.close("all")
            projection = ccrs.Robinson()
            fig, (ax1, ax2, ax3) = plt.subplots(
                nrows=1, ncols=3, subplot_kw={"projection": projection}, figsize=(16, 3)
            )

            lon = io["lon"][:]
            lat = io["lat"][:]
            lead_hrs = 6 * step

            def plot_field(
                axi: Any,
                data: np.ndarray,
                title: str,
                cmap: str = "Blues",
            ) -> None:
                im = axi.pcolormesh(
                    lon,
                    lat,
                    data,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                )
                plt.colorbar(im, ax=axi, shrink=0.6, pad=0.04)
                axi.set_title(title)
                axi.coastlines()
                axi.gridlines()

            # Member 0, Member 1, Std across members
            plot_field(
                ax1,
                io[variable][0, 0, step],
                f"{forecast_time} - Lead: {lead_hrs}hrs - Member 0",
            )
            if parameters.nensemble >= 2:
                plot_field(
                    ax2,
                    io[variable][1, 0, step],
                    f"{forecast_time} - Lead: {lead_hrs}hrs - Member 1",
                )
            else:
                ax2.set_visible(False)
            std_data = np.std(io[variable][:, 0, step], axis=0)
            plot_field(
                ax3,
                std_data,
                f"{forecast_time} - Lead: {lead_hrs}hrs - Std",
            )

            output_dir = self.get_output_path(execution_id)
            plot_path = output_dir / f"ensemble_plot_{variable}_step{step}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

        except Exception:
            logger.exception("Could not create ensemble plot")
            raise

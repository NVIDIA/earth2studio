# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import os
from collections.abc import Iterator
from datetime import datetime
from math import ceil

import numpy as np
import torch
import xarray as xr
from loguru import logger
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array

from .hens_utilities import TCTracking, cat_coords, get_batchid_from_ensid
from .hens_utilities_reproduce import calculate_torch_seed

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class EnsembleBase:
    """Ensemble inference pipeline with options to add a diagnostic
    model or tropical cyclone tracking in the loop.

    Parameters
    ----------
    time : list[str] | list[datetime] | list[np.datetime64]
        IC times.
    nsteps : int
        number of forecast steps.
    nensemble : int
        ensemble size.
    prognostic : PrognosticModel
        forecast model.
    data : DataSource
        Data source from which to obtain ICs.
    io_dict : dict[str, IOBackend]
        Data object for storing generated data.
    perturbation : Perturbation
        Method for perturbing initial conditions.
    output_coords_dict : dict[str, CoordSystem]
        Dictionary of coordinate systems of data that shall be stored.
    dx_model_dict : dict[str, DiagnosticModel], optional
        Dictionary of diagnostic models.
    cyclone_tracking : DiagnosticModel, optional
        Cyclone tracking diagnostic model.
    batch_size : int, optional
        batch size.
    device : torch.device, optional
        device on which to run inference.
    ensemble_idx_base : int, optional
        Initial value for counting ensemble members.
    batch_ids_produce : list[int]
        List of batch IDs that will be processed.
    base_seed_string : str
        Random seed that will be used as a basis.
    pkg : str
        Model package name, used for naming the output files for cyclone tracking.
    """

    def __init__(
        self,
        time: list[str] | list[datetime] | list[np.datetime64],
        nsteps: int,
        nensemble: int,
        prognostic: PrognosticModel,
        data: DataSource,
        io_dict: dict[str, IOBackend],
        perturbation: Perturbation,
        output_coords_dict: dict[str, CoordSystem],
        dx_model_dict: dict[str, DiagnosticModel] = {},
        cyclone_tracking: TCTracking | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        ensemble_idx_base: int = 0,
        batch_ids_produce: list[int] = [],
        base_seed_string: str = "0",
        pkg: str = "",
    ) -> None:

        logger.info("Setting up HENS.")

        self.io_dict = io_dict
        self.nensemble = nensemble
        self.ensemble_idx = ensemble_idx_base
        self.batch_ids_produce = batch_ids_produce
        self.nsteps = nsteps
        self.output_coords_dict = output_coords_dict
        self.perturbation = perturbation
        self.base_seed_string = base_seed_string
        self.pkg = pkg.split("/")[-1].split("seed")[-1]

        if cyclone_tracking:
            self.cyclone_tracking = cyclone_tracking.tracker
            self.cyclone_tracking_out_path = cyclone_tracking.out_path
        else:
            self.cyclone_tracking = None

        if len(time) > 1:
            raise ValueError("Only a single IC can be passed here")
        self.ic = time[0]

        # Load model onto the device
        self.move_models_to_device(prognostic, dx_model_dict, device)

        # Fetch data from data source and load onto device
        self.fetch_ics(data=data, time=time)

        # Compute batch sizes
        self.set_batch_size(batch_size)

        # Set up IO backend with information from output_coords (if applicable).
        self.setup_data_output()

        return

    def move_models_to_device(
        self,
        prognostic: PrognosticModel,
        dx_model_dict: dict[str, DiagnosticModel] = {},
        device: torch.device | None = None,
    ) -> None:
        """Moves model dictionary to device

        Parameters
        ----------
        prognostic : PrognosticModel
            Forecast model.
        dx_model_dict : dict[str, DiagnosticModel], optional
            Diagnostic model dictionary, by default {}
        cyclone_tracking : DiagnosticModel | None, optional
            Cyclone tracking diagnostic, by default None
        device : torch.device, optional
            PyTorch device. If None, will select cuda if available, by default None
        """
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.info(f"Inference device: {self.device}")
        self.prognostic = prognostic.to(self.device)
        self.prognositc_ic = prognostic.input_coords()

        self.dx_model_dict = dx_model_dict
        dx_ic_dict = {}
        for k, dx_model in dx_model_dict.items():
            dx_model.to(self.device)
            dx_ic_dict[k] = dx_model.input_coords()
        self.dx_ic_dict = dx_ic_dict

        if self.cyclone_tracking:
            self.cyclone_tracking.to(self.device)
            self.cyclone_tracking_ic = self.cyclone_tracking.input_coords()

    def fetch_ics(
        self,
        data: DataSource,
        time: list[str] | list[datetime] | list[np.datetime64],
    ) -> None:
        """Fetch initial conditions

        Parameters
        ----------
        data : DataSource
            Data source from which to obtain ICs
        time : list[str] | list[datetime] | list[np.datetime64]
            IC times
        """
        self.time = to_time_array(time)
        self.x0, self.coords0 = fetch_data(
            source=data,
            time=time,
            variable=self.prognositc_ic["variable"],
            lead_time=self.prognositc_ic["lead_time"],
            device="cpu",
        )
        logger.success(f"Fetched data from {data.__class__.__name__}")

        return

    def setup_data_output(self) -> None:
        """Assemble output coords and initialise IO backend with coords."""

        # assemble output coords from fetched IC coords and ensemble IDs
        ensemble_members = np.arange(self.nensemble) + self.ensemble_idx
        batch_ids = [
            get_batchid_from_ensid(self.nensemble, self.batch_size, ensid)
            for ensid in ensemble_members
        ]
        # determine which ensemble members we keep (based on batch_ids_produce)
        ensemble_members_to_produce = ensemble_members[
            np.isin(batch_ids, self.batch_ids_produce)
        ]
        total_coords = {"ensemble": ensemble_members_to_produce} | self.coords0.copy()

        # add lead time dimension
        total_coords["lead_time"] = np.asarray(
            [
                self.prognostic.output_coords(self.prognostic.input_coords())[
                    "lead_time"
                ]
                * ii
                for ii in range(self.nsteps + 1)
            ]
        ).flatten()

        for i, (k, oc) in enumerate(self.output_coords_dict.items()):

            # augment and overwrite total coords with dimensions of output coords
            for key, value in total_coords.items():
                total_coords[key] = oc.get(key, value)

            if i == 0:
                # initialise place for variables in io backend
                variables_to_save = total_coords.pop("variable")

            if (
                self.io_dict[k] is not None
            ):  # cyclone tracker still missing field output, to be fixed
                self.io_dict[k].add_array(total_coords, variables_to_save)

        return

    def set_batch_size(self, batch_size: int | None = None) -> None:
        """Calculate batch size and number of mini batches to inference.

        Parameters
        ----------
        batch_size : int
            targeted batch size
        """
        if batch_size is None:
            batch_size = self.nensemble
        self.batch_size = min(self.nensemble, batch_size)
        self.number_of_batches = ceil(self.nensemble / self.batch_size)

    def prep_loop(self, batch_id: int) -> tuple[Iterator[tuple], int, str, int]:
        """Preparing mini batch for inference by setting ensemble IDs, perturbing
        ICs and creating the inference iterator of the prognostic model.

        Parameters
        ----------
        batch_id : int
            mini batch index

        Returns
        -------
        tuple[Iterator[tuple], int, str, int]
            Tuple containing iterator of prognostic model, mini batch size, seed string
            and PyTorch seed.
        """

        # Get fresh batch data
        xx = self.x0.to(self.device)

        # calculate mini batch size and define coords for ensemble
        num_batches_per_ic = int(np.ceil(self.nensemble / self.batch_size))
        mini_batch_sizes = [
            min((self.nensemble - ii * self.batch_size), self.batch_size)
            for ii in range(num_batches_per_ic)
        ]
        batch_id_ic = batch_id % num_batches_per_ic
        mini_batch_size = mini_batch_sizes[batch_id_ic]

        coords = {
            "ensemble": np.array(
                [
                    sum(mini_batch_sizes[0 : batch_id % num_batches_per_ic]) + t
                    for t in range(0, mini_batch_size)
                ]
            )
            + self.ensemble_idx
        } | self.coords0.copy()

        # Unsqueeze xx for batching ensemble
        xx = xx.unsqueeze(0).repeat(mini_batch_size, *([1] * xx.ndim))

        # Map lat and lon if needed
        xx, coords = map_coords(xx, coords, self.prognositc_ic)

        # set torch random seed for reproducibility
        # every batch gets different random seed by concatenating base string with batch id and using hash algortihm
        full_seed_string = self.base_seed_string + "_" + str(batch_id)
        torch_seed = calculate_torch_seed(full_seed_string)
        torch.manual_seed(torch_seed)

        # Perturb ensemble
        xx, coords = self.perturbation(xx, coords)

        # Create prognostic iterator
        model = self.prognostic.create_iterator(xx, coords)

        return model, mini_batch_size, full_seed_string, torch_seed

    @torch.inference_mode()
    def __call__(self) -> dict[str, IOBackend]:
        """Run ensemble inference pipeline with diagnostic model on top
        saving specified variables.

        Returns
        -------
        dict[str, IOBackend]
            Dictionary of io objects containing data of ensemble inference
        """
        logger.info(
            f"Starting {self.nensemble} member ensemble inference with"
            + f" {len(self.batch_ids_produce)} batches."
        )

        for batch_id in tqdm(
            self.batch_ids_produce,
            total=len(self.batch_ids_produce),
            desc="Total Ensemble Batches",
        ):

            model, nsamples, full_seed_string, torch_seed = self.prep_loop(batch_id)
            # If cyclone tracking reset path buffer
            if self.cyclone_tracking:
                self.cyclone_tracking.reset_path_buffer()

            with tqdm(
                total=self.nsteps + 1,
                desc=f"Inferencing batch {batch_id} ({nsamples} samples)",
                leave=False,
            ) as pbar:
                for step, (xx, coords) in enumerate(model):

                    for dx_name, dx_model in self.dx_model_dict.items():
                        # select input vars, remove lead time dim and apply diagnostic model
                        yy, codia = map_coords(xx, coords, self.dx_ic_dict[dx_name])
                        yy, codib = dx_model(yy, codia)

                        # concatenate diagnostic variable to forecast vars
                        xx, coords = cat_coords(xx, coords, yy, codib, "variable")

                    if self.cyclone_tracking:
                        # Delete lead_time, no need for it in the tc tracks since
                        # steps are present in the tracks
                        xx_tc = xx[:, :, 0]
                        coords_tc = coords.copy()
                        del coords_tc["lead_time"]
                        # get and collect track elements for each time step
                        tracks_tensor, track_coords = self.cyclone_tracking(
                            *map_coords(xx_tc, coords_tc, self.cyclone_tracking_ic)
                        )

                    # pass output variables to io backend
                    for k in self.io_dict.keys():
                        output_coords = self.output_coords_dict[k]
                        xx_sub, coords_sub = map_coords(xx, coords, output_coords)
                        self.io_dict[k].write(*split_coords(xx_sub, coords_sub))

                    pbar.update(1)
                    if step == self.nsteps:
                        break

            # If cyclone tracks add to list of data arrays
            if self.cyclone_tracking:
                # Create DataArray for the tracks
                tracks_da = xr.DataArray(
                    data=tracks_tensor.cpu().numpy(),
                    coords=track_coords,
                    dims=list(track_coords.keys()),
                )
                os.makedirs(self.cyclone_tracking_out_path, exist_ok=True)
                file_name = np.datetime_as_string(self.ic, unit="s")
                file_name = f"tracks_pkg_{self.pkg}_{file_name}_batch_{batch_id}.nc"
                tracks_da.to_netcdf(
                    os.path.join(self.cyclone_tracking_out_path, file_name)
                )

        logger.success("Inference complete")
        return self.io_dict

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

from collections import OrderedDict
from collections.abc import Iterator
from datetime import datetime
from math import ceil

import numpy as np
import pandas as pd
import torch
from loguru import logger
from reproduce_utilities import calculate_torch_seed
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.dx.cyclone_tracking import get_tracks_from_positions
from earth2studio.models.px import PrognosticModel
from earth2studio.perturbation import Perturbation
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords
from earth2studio.utils.time import to_time_array

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


class EnsembleBase:
    """
    Ensemble inference pipeline with options to add a diagnostic
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
        dx_model_dict: dict[str, DiagnosticModel] | None = None,
        cyclone_tracking: DiagnosticModel | None = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        ensemble_idx_base: int = 0,
        batch_ids_produce: list[int] | None = None,
        base_seed_string: str = "0",
    ) -> None:

        logger.info("Setting up HENS.")

        self.io_dict = io_dict
        self.nensemble = nensemble
        self.ensemble_idx = ensemble_idx_base
        self.batch_ids_produce = batch_ids_produce
        self.nsteps = nsteps
        self.output_coords_dict = output_coords_dict
        self.perturbation = perturbation
        self.cyclone_tracking = cyclone_tracking
        self.base_seed_string = base_seed_string

        # Load model onto the device
        self.move_models_to_device(prognostic, dx_model_dict, cyclone_tracking, device)

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
        dx_model_dict: DiagnosticModel | None = None,
        cyclone_tracking: DiagnosticModel | None = None,
        device: torch.device | None = None,
    ) -> None:
        """
        Move models to device and obtain their input coordinates.

        Parameters
        ----------
        prognostic : PrognosticModel
            forecast model.
        diagnostic : DiagnosticModel | None
            diagnostic model [optional]
        device : torch.device
            device on which to run inference
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

        self.cyclone_tracking = cyclone_tracking
        if cyclone_tracking is not None:
            self.cyclone_tracking.to(self.device)
            self.cyclone_tracking_ic = self.cyclone_tracking.input_coords()

        return

    def fetch_ics(
        self,
        data: DataSource,
        time: list[str] | list[datetime] | list[np.datetime64],
    ) -> None:
        """
        Fetch initial conditions.

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
        """
        Assemble output coords and initialise IO backend with coords.
        """

        # assemble output coords from fetched IC coords and ensemble IDs
        ensemble_members = np.arange(self.nensemble) + self.ensemble_idx
        batch_ids = np.arange(self.nensemble) // self.batch_size
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
        """
        calculate batch size and number of mini batches to inference.

        Parameters
        ----------
        batch_size : int
            targeted batch size
        """
        if batch_size is None:
            batch_size = self.nensemble
        self.batch_size = min(self.nensemble, batch_size)
        self.number_of_batches = ceil(self.nensemble / self.batch_size)

    def prep_loop(self, batch_id: int) -> tuple[Iterator, int]:
        """
        preparing mini batch for inference by setting ensemble IDs, perturbing
        ICs and creating the inference iterator of the prognostic model.

        Parameters
        ----------
        batch_id : int
            mini batch index

        Returns
        -------
        tuple[Iterator, int]
            Tuple containing iterator of prognostic model and mini batch size.
        """

        # Get fresh batch data
        xx = self.x0.to(self.device)

        # calculate mini batch size and define coords for ensemble
        mini_batch_size = min(
            self.batch_size, self.nensemble - batch_id * self.batch_size
        )
        coords = {
            "ensemble": np.arange(
                batch_id * self.batch_size, batch_id * self.batch_size + mini_batch_size
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
    def __call__(self) -> IOBackend:
        """
        Run ensemble inference pipeline with diagnostic model on top
        saving specified variables.

        Returns
        -------
        IOBackend
            io object containing data of ensemble inference.
        """
        logger.info(
            f"Starting {self.nensemble} Member Ensemble inference with"
            + f" {len(self.batch_ids_produce)} number of batches."
        )

        tracks_dict = {kk: [] for kk in self.output_coords_dict.keys()}
        seed_dict = {}
        for batch_id in tqdm(
            self.batch_ids_produce,
            total=len(self.batch_ids_produce),
            desc="Total Ensemble Batches",
        ):

            model, nsamples, full_seed_string, torch_seed = self.prep_loop(batch_id)
            seed_dict[batch_id] = (full_seed_string, torch_seed)
            with tqdm(
                total=self.nsteps + 1,
                desc=f"Inferencing batch {batch_id} ({nsamples} samples)",
                leave=False,
            ) as pbar:
                if self.cyclone_tracking:
                    track_element_dict = {
                        "track_element_list": [],
                        "track_coords_list": [],
                        "time_list": [],
                    }
                for step, (xx, coords) in enumerate(model):

                    for dx_name, dx_model in self.dx_model_dict.items():
                        # select input vars, remove lead time dim and apply diagnostic model
                        yy, codia = map_coords(xx, coords, self.dx_ic_dict[dx_name])
                        yy, codib = dx_model(yy, codia)

                        # concatenate diagnostic variable to forecast vars
                        xx, coords = cat_coords(xx, coords, yy, codib, "variable")

                    if self.cyclone_tracking:
                        # get and collect track elements for each time step
                        track_element_dict = self.detect_tc_centres(
                            xx, coords, track_element_dict
                        )

                    # pass output variables to io backend
                    for k in self.io_dict.keys():
                        output_coords = self.output_coords_dict[k]
                        xx_sub, coords_sub = map_coords(xx, coords, output_coords)
                        self.io_dict[k].write(*split_coords(xx_sub, coords_sub))

                    pbar.update(1)
                    if step == self.nsteps:
                        break

                if self.cyclone_tracking:
                    # combine track elements to get full tracks (includes threading)
                    df_tracks_dict = self.connect_centres_to_tracks(track_element_dict)
                    for k, df_tracks in df_tracks_dict.items():
                        df_tracks = self.add_meta_data_to_trackds_df(
                            df_tracks, full_seed_string, torch_seed
                        )
                        tracks_dict[k].append(df_tracks)
        df_tracks_dict = self.concat_tracks_for_each_region(tracks_dict)
        logger.success("Inference complete")

        return df_tracks_dict, self.io_dict, seed_dict

    @staticmethod
    def concat_tracks_for_each_region(
        tracks_dict: dict[str, list[pd.DataFrame]]
    ) -> dict[str, pd.DataFrame]:
        """
        Concatenates track data for each region into single DataFrames.

        Parameters
        ----------
        tracks_dict : dict[str, list[pd.DataFrame]]
            Dictionary where keys are region identifiers and values are lists of DataFrames,
            each representing a track for a specific ensemble member and region.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary where keys are region identifiers and values are single concatenated
            DataFrames containing all track data for that region.
        """
        df_tracks_dict = {}
        if tracks_dict:
            for k in tracks_dict.keys():
                if tracks_dict[k]:
                    df_tracks_dict[k] = pd.concat(tracks_dict[k])
                else:
                    df_tracks_dict[k] = pd.DataFrame()
        return df_tracks_dict

    @staticmethod
    def get_lon_range_from_coordinates(coords: OrderedDict) -> int:
        """
        calculate which longitudes range is being used

        Parameters
        ----------
        coords: OrderedDict
            Input tensor

        Returns
        -------
        lon_range: int
            Either 180 or 360
        """
        lon_range = (
            360 if ((coords["lon"].max() > 180) and (coords["lon"].min() >= 0)) else 180
        )
        return lon_range

    def adjust_geographic_extent(self, tt_mem: torch.Tensor) -> torch.Tensor:
        """
        adjust geographic extent of data to

        Parameters
        ----------
        tt_mem: torch.Tensor
            Input tensor

        Returns
        -------
        tt_mem_filtered: torch.Tensor
            Data cropped to region of interest
        """

        tt_mem_filtered_dict = {}
        for k, output_coords in self.output_coords_dict.items():
            # determine longitude ranges of input and output
            lon_range_in = self.get_lon_range_from_coordinates(self.cyclone_tracking_ic)
            lon_range_out = self.get_lon_range_from_coordinates(output_coords)

            # adjust longitude range
            if lon_range_in != lon_range_out:
                if lon_range_in == 360 and lon_range_out == 180:
                    tt_mem[:, 1, :] = ((tt_mem[:, 1, :] + 180) % 360) - 180
                elif lon_range_in == 180 and lon_range_out == 360:
                    tt_mem[:, 1, :] = tt_mem[:, 1, :] % 360

            # filter area
            c1 = tt_mem[:, 1, :] >= output_coords["lon"].min()
            c2 = tt_mem[:, 1, :] <= output_coords["lon"].max()
            c3 = tt_mem[:, 0, :] >= output_coords["lat"].min()
            c4 = tt_mem[:, 0, :] <= output_coords["lat"].max()
            tt_mem_filtered = torch.where(
                (c1 & c2 & c3 & c4).repeat(4, 1, 1).swapaxes(1, 0),
                tt_mem,
                np.nan,
            )
            tt_mem_filtered_dict[k] = tt_mem_filtered
        return tt_mem_filtered_dict

    def detect_tc_centres(
        self,
        xx: torch.Tensor,
        coords: CoordSystem,
        track_element_dict: dict,
    ) -> dict:
        """
        Detects TC centers and appends them to the corresponding lists for data, coordinates, and verification time.

        Parameters
        ----------
        xx : torch.Tensor
            Input tensor containing the data for which TC centers need to be detected.
        coords : CoordSystem
            Ordered dictionary representing the coordinate system that describes `xx`.
        track_element_dict : dict
            Dictionary containing lists for TC center candidates, their coordinates, and verification times.
            It should have the following keys:
            - 'track_element_list': list of torch.Tensor objects containing TC center candidates for each timestep.
            - 'track_coords_list': list of CoordSystem objects corresponding to each element in 'track_element_list'.
            - 'time_list': list of np.datetime64 objects representing the verification times for each element in 'track_element_list'.

        Returns
        -------
        dict
            A dictionary with the updated lists for TC center candidates, their coordinates, and verification times.
        """
        xx2, coords2 = map_coords(xx, coords, self.cyclone_tracking_ic)
        rt = coords2["time"][0]  # run time/initialisation time
        lt = coords2["lead_time"][0]  # lead time
        vt = rt + lt  # verification time

        track_element_dict["time_list"].append(vt)

        coords2["time"] = vt

        track_value, track_coords = self.cyclone_tracking(xx2, coords2)
        del track_coords["lead_time"]
        track_value = track_value.squeeze(2)
        track_element_dict["track_element_list"].append(track_value)
        track_element_dict["track_coords_list"].append(track_coords)
        return track_element_dict

    def combine_track_elements(
        self,
        track_element_dict: dict,
    ) -> tuple[torch.Tensor, OrderedDict, np.ndarray[np.int64]]:
        """
        Combines individual track elements (representing a time step) into a large tensor representing all timesteps.

        Parameters
        ----------
        track_element_dict : dict
            Dictionary containing lists for TC center candidates, their coordinates, and verification times.
            It should have the following keys:
            - 'track_element_list': List of torch.Tensor objects containing TC center candidates for each timestep.
            - 'track_coords_list': List of CoordSystem objects corresponding to each element in 'track_element_list'.
            - 'time_list': List of np.datetime64 objects representing the verification times for each element in 'track_element_list'.

        Returns
        -------
        tuple[torch.Tensor, OrderedDict, np.ndarray[np.int64]]
            - out_tensor: torch.Tensor containing TC center candidates for all timesteps.
            - track_coords_final: OrderedDict representing the final coordinate system.
            - member_ids: np.ndarray[np.int64] containing ensemble member IDs.
        """
        track_element_list = track_element_dict["track_element_list"]
        track_coords_list = track_element_dict["track_coords_list"]
        time_list = track_element_dict["time_list"]
        out_tensor = torch.nested.nested_tensor(
            track_element_list, dtype=torch.float32, device=track_element_list[0].device
        )
        out_tensor = torch.swapaxes(
            torch.nested.to_padded_tensor(out_tensor, torch.nan), 0, 2
        )
        # remove lead time dim
        out_tensor = out_tensor.squeeze(0)

        track_coords_final = track_coords_list[-1]
        member_ids = track_coords_final["ensemble"]
        del track_coords_final["ensemble"]
        track_coords_final["time"] = time_list
        track_coords_final["point"] = np.sort(
            np.unique(np.concatenate([x["point"] for x in track_coords_list]))
        )
        return out_tensor, track_coords_final, member_ids

    def connect_centres_to_tracks(
        self, track_element_dict: dict
    ) -> dict[str, pd.DataFrame]:
        """
        Combines TC center candidates into tracks using the provided track elements.

        Parameters
        ----------
        track_element_dict : dict
            Dictionary containing lists for TC center candidates, their coordinates, and verification times.
            It should have the following keys:
            - 'track_element_list': List of torch.Tensor objects containing TC center candidates for each timestep.
            - 'track_coords_list': List of CoordSystem objects corresponding to each element in 'track_element_list'.
            - 'time_list': List of np.datetime64 objects representing the verification times for each element in 'track_element_list'.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary where keys are region identifiers and values are DataFrames containing track information
            for each region.
        """
        tt, track_coords_final, member_ids = self.combine_track_elements(
            track_element_dict
        )
        tracks_dict = {}
        for k in self.output_coords_dict.keys():
            tracks_dict[k] = []
        # iterate over individual ensemble members
        for i_member in range(tt.shape[0]):
            tt_mem = tt[i_member]
            tt_mem_filtered_dict = self.adjust_geographic_extent(tt_mem)

            for k, tt_mem_filtered in tt_mem_filtered_dict.items():
                # threading/combine center locations to tracks
                tracks_df = get_tracks_from_positions(
                    tt_mem_filtered, track_coords_final
                )
                tracks_df.insert(0, "ens_member", member_ids[i_member])
                tracks_dict[k].append(tracks_df)

        df_tracks_dict = {}
        for k in tracks_dict.keys():
            df_tracks_dict[k] = pd.concat(tracks_dict[k]).reset_index(drop=True)
        return df_tracks_dict

    def add_meta_data_to_trackds_df(self, tracks_df, full_seed_string, torch_seed):

        tracks_df = tracks_df.assign(full_seed_string=full_seed_string)
        tracks_df = tracks_df.assign(torch_seed=str(torch_seed))
        tracks_df = tracks_df.assign(
            batch_id=(tracks_df["ens_member"] - self.ensemble_idx) // self.batch_size
        )
        tracks_df = tracks_df.assign(batch_size=self.batch_size)
        tracks_df = tracks_df.assign(nensemble=self.nensemble)
        return tracks_df


def cat_coords(
    xx: torch.Tensor,
    cox: CoordSystem,
    yy: torch.Tensor,
    coy: CoordSystem,
    dim: str = "variable",
) -> tuple[torch.Tensor, CoordSystem]:
    """
    concatenate data along coordinate dimension.

    Parameters
    ----------
    xx : torch.Tensor
        First input tensor which to concatenate
    cox : CoordSystem
        Ordered dict representing coordinate system that describes xx
    yy : torch.Tensor
        Second input tensor which to concatenate
    coy : CoordSystem
        Ordered dict representing coordinate system that describes yy
    dim : str
        name of dimension along which to concatenate

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict from
        concatenated data.
    """

    if dim not in cox:
        raise ValueError(f"dim {dim} is not in coords: {list(cox)}.")
    if dim not in coy:
        raise ValueError(f"dim {dim} is not in coords: {list(coy)}.")

    # fix difference in latitude
    _cox = cox.copy()
    _cox["lat"] = coy["lat"]
    xx, cox = map_coords(xx, cox, _cox)

    coords = cox.copy()
    dim_index = list(coords).index(dim)

    zz = torch.cat((xx, yy), dim=dim_index)
    coords[dim] = np.append(cox[dim], coy[dim])

    return zz, coords


def squeeze_coord(
    xx: torch.Tensor, inco: CoordSystem, dim: str
) -> tuple[torch.Tensor, CoordSystem]:
    """
    remove a coordinate dimension of length 1.

    Parameters
    ----------
    xx : torch.Tensor
        Input tensor
    inco : CoordSystem
        Ordered dict representing coordinate system that describes xx
    dim : str
        name of dimension along which to concatenate

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict from
        concatenated data.
    """
    idx = list(inco).index(dim)
    ouco = inco.copy()
    ouco.pop(dim)
    if xx.shape[idx] != 1:
        raise ValueError(
            "cannot remove dimension with len>1,"
            + f" dim {dim} has length {xx.shape[idx]}"
        )

    return xx.squeeze(idx), ouco

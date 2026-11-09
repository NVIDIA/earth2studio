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

from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data import DataSource, fetch_data
from earth2studio.grids import LatLonGrid, infer_grid
from earth2studio.io import IOBackend
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import step_interval
from earth2studio.perturbation import HemisphericCentredBredVector, Perturbation
from earth2studio.run import Pipeline, WorkItem
from earth2studio.run._fields import _map_field, _with_lead_time
from earth2studio.run.runner import _signature_for_stream
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.time import to_time_array
from earth2studio.utils.type import CoordinateSystem

from .hens_distributed import local_device
from .hens_utilities_reproduce import calculate_torch_seed


def _regional_field(x: xr.DataArray, selection: Mapping[str, Any]) -> xr.DataArray:
    """Select a HENS lat/lon region and refresh its grid metadata."""
    if "lon" in selection:
        longitude = np.asarray(selection["lon"])
        if (longitude < 0).any() and (x.lon.values >= 0).all():
            x = x.assign_coords(lon=(x.lon + 180) % 360 - 180).sortby("lon")
        elif (longitude > 180).any() and (x.lon.values < 0).any():
            x = x.assign_coords(lon=x.lon % 360).sortby("lon")
    x = _map_field(x, selection)
    signature = coord_array(
        x.dims,
        dict(x.coords),
        dynamic=x.attrs.get("earth2studio_dynamic_dims", ()),
        sizes=x.sizes,
        grid=LatLonGrid(x.lat.values, x.lon.values),
        dtype=x.dtype,
        attrs=x.attrs,
    )
    x.attrs = signature.attrs
    return x


class HENSRunner:
    """Run one complete HENS perturbation batch as a Pipeline work item.

    Batches, rather than individual members, are the reproducibility unit: bred
    vectors center and pair members together. Diagnostics are chained in insertion
    order. Regional streams are written by Pipeline; tracking and downscaling
    retain their per-batch files.
    """

    supports_member_batching = True

    def __init__(
        self,
        prognostic: PrognosticModel,
        data: DataSource,
        perturbation: Perturbation,
        output_coords_dict: Mapping[str, Mapping[str, Any]],
        nensemble: int,
        batch_size: int,
        base_seed_string: str,
        dx_model_dict: Mapping[str, DiagnosticModel] | None = None,
        cd_model_dict: Mapping[str, Any] | None = None,
        cyclone_tracking: Any = None,
        pkg: str = "",
    ) -> None:
        self.prognostic = prognostic
        self.data = data
        self.perturbation = perturbation
        self.regions = {
            name: dict(coords) for name, coords in output_coords_dict.items()
        }
        self.nensemble = nensemble
        self.batch_size = batch_size
        self.base_seed_string = base_seed_string
        self.diagnostics = dict(dx_model_dict or {})
        self.downscaling = dict(cd_model_dict or {})
        self.tracking = cyclone_tracking
        self.pkg = pkg.split("/")[-1].split("seed")[-1]
        self.device = torch.device("cpu")
        inputs = prognostic.input_coords()
        if isinstance(inputs, tuple):
            raise ValueError("HENS requires a single prognostic input field")
        self.inputs: CoordinateSystem = inputs
        outputs = prognostic.output_coords(self.inputs)
        if isinstance(outputs, tuple):
            raise ValueError("HENS requires a single prognostic output field")
        self.step = step_interval(
            self.inputs.lead_time.values, outputs.lead_time.values
        )
        if len(outputs.lead_time) != 1 or self.inputs.lead_time.values[
            -1
        ] != np.timedelta64(0, "h"):
            raise ValueError(
                "HENS requires a single forecast frame per step and an analysis-time input"
            )
        self._initial: dict[np.datetime64, xr.DataArray] = {}

    def to(self, device: torch.device) -> "HENSRunner":
        """Move all models to the worker's device."""
        self.device = torch.device(device)
        self.prognostic.to(self.device)
        for model in self.diagnostics.values():
            model.to(self.device)
        for wrapper in self.downscaling.values():
            wrapper.corrdiff.to(self.device)
        if self.tracking is not None:
            self.tracking.tracker.to(self.device)
        return self

    def _nsteps(self, final_lead_time: np.timedelta64) -> int:
        if final_lead_time < np.timedelta64(
            0, "s"
        ) or final_lead_time % self.step != np.timedelta64(0, "s"):
            raise ValueError(
                "Final lead time must be a nonnegative multiple of the model step"
            )
        return int(final_lead_time // self.step)

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Plan independent regional stores, including the initial condition."""
        signature = self.prognostic.output_coords(self.inputs)
        if isinstance(signature, tuple):
            raise ValueError("HENS requires a single prognostic output field")
        variables = list(signature["variable"].values)
        for diagnostic in self.diagnostics.values():
            inputs = _map_field(signature, diagnostic.input_coords())
            output = diagnostic.output_coords(inputs)
            if isinstance(output, tuple):
                raise ValueError("HENS requires single-field diagnostics")
            variables.extend(output["variable"].values)
            signature = coord_array_like(signature, {"variable": variables})
        leads = np.arange(self._nsteps(final_lead_time) + 1) * self.step
        signature = _signature_for_stream(signature, leads)
        return {
            name: _regional_field(signature, coords)
            for name, coords in self.regions.items()
        }

    def data_requests(self, item: WorkItem) -> None:
        """Bred-vector perturbations own additional data fetches internally."""
        return None

    def _fetch(self, item: WorkItem) -> xr.DataArray:
        if item.time not in self._initial:
            self._initial[item.time] = fetch_data(
                source=self.data,
                time=np.array([item.time]),
                variable=self.inputs["variable"].values,
                lead_time=self.inputs.lead_time.values,
                target_grid=infer_grid(self.inputs),
                device="cpu",
            )
        # Copy the expanded batch so perturbations cannot change cached ICs.
        x = (
            self._initial[item.time]
            .astype(self.inputs.dtype)
            .expand_dims(ensemble=list(item.member_ids))
            .copy(deep=True)
        )
        return x.e2s.to_backend(
            "cupy" if self.device.type == "cuda" else "numpy", device=self.device
        )

    def _seed(self, batch_id: int) -> None:
        seed = calculate_torch_seed(f"{self.base_seed_string}_{batch_id}")
        torch.manual_seed(seed)
        for model in [
            self.prognostic,
            *self.diagnostics.values(),
            *(w.corrdiff for w in self.downscaling.values()),
        ]:
            if getattr(model, "stochastic", False):
                model.set_rng(seed, reset=True)

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish the perturbed IC once, then exactly the requested advances."""
        num_batches = -(-self.nensemble // self.batch_size)
        member = item.member_ids[0]
        batch_id = (member // self.nensemble) * num_batches + (
            member % self.nensemble
        ) // self.batch_size
        with torch.inference_mode():
            bred = getattr(self.perturbation, "perturbation", self.perturbation)
            if isinstance(bred, HemisphericCentredBredVector):
                # Odd batches carry a paired member from the preceding batch.
                # Reconstruct that state for isolated/reordered work items, without
                # forecasting or writing the predecessor. Even batches start clean.
                bred._residual = []
                local_batch = (member % self.nensemble) // self.batch_size
                if self.batch_size % 2 and local_batch % 2:
                    previous = WorkItem(
                        item.time,
                        item.final_lead_time,
                        tuple(range(member - self.batch_size, member)),
                    )
                    self._seed(batch_id - 1)
                    self.perturbation(self._fetch(previous))
            self._seed(batch_id)
            x = self.perturbation(self._fetch(item))
            tracker = self.tracking.tracker if self.tracking is not None else None
            if tracker is not None:
                tracker.reset_path_buffer()
            downscaled: dict[str, list[xr.DataArray]] = {
                name: [] for name in self.downscaling
            }
            tracks = None
            iterator = self.prognostic.create_iterator(x)
            try:
                for step in range(self._nsteps(item.final_lead_time) + 1):
                    field = initial_condition(x) if step == 0 else next(iterator)
                    if isinstance(field, tuple):
                        raise ValueError(
                            "HENS requires a single prognostic output field"
                        )
                    for diagnostic in self.diagnostics.values():
                        result = diagnostic(
                            _map_field(field, diagnostic.input_coords())
                        )
                        result = _with_lead_time(result, field.lead_time.values)
                        result = _map_field(
                            result, {dim: field[dim].values for dim in ("lat", "lon")}
                        )
                        field = xr.concat([field, result], dim="variable", join="exact")
                    for name, wrapper in self.downscaling.items():
                        result = wrapper.corrdiff(
                            _map_field(field, wrapper.corrdiff.input_coords())
                        )
                        result = _with_lead_time(result, field.lead_time.values)
                        downscaled[name].append(
                            result.e2s.to_backend("numpy", device="cpu")
                        )
                    if tracker is not None:
                        tracks = tracker(
                            _map_field(
                                field.isel(lead_time=0, drop=True),
                                tracker.input_coords(),
                            )
                        )
                    yield {
                        name: _regional_field(field, coords)
                        for name, coords in self.regions.items()
                    }
            finally:
                iterator.close()

            stamp = np.datetime_as_string(item.time, unit="s")
            if tracks is not None and self.tracking is not None:
                path = Path(self.tracking.out_path)
                path.mkdir(parents=True, exist_ok=True)
                tracks.e2s.to_backend("numpy", device="cpu").to_netcdf(
                    path / f"tracks_pkg_{self.pkg}_{stamp}_batch_{batch_id}.nc"
                )
            for name, fields in downscaled.items():
                path = Path(self.downscaling[name].out_path)
                path.mkdir(parents=True, exist_ok=True)
                xr.concat(fields, dim="lead_time", join="exact").to_dataset(
                    dim="variable"
                ).to_netcdf(
                    path / f"corrdiff_{name}_pkg_{self.pkg}_{stamp}_batch_{batch_id}.nc"
                )


class EnsembleBase:
    """Configure a local Pipeline for one HENS checkpoint/initial-condition pair.

    The public recipe entry point retains its existing arguments. Work items keep
    the original global batch IDs and member groups, including a short last batch.
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
        output_coords_dict: Mapping[str, Mapping[str, Any]],
        dx_model_dict: Mapping[str, DiagnosticModel] | None = None,
        cd_model_dict: Mapping[str, Any] | None = None,
        cyclone_tracking: Any = None,
        batch_size: int | None = None,
        device: torch.device | None = None,
        ensemble_idx_base: int = 0,
        batch_ids_produce: list[int] | None = None,
        base_seed_string: str = "0",
        pkg: str = "",
    ) -> None:
        if len(time) != 1 or nsteps < 0 or nensemble < 1:
            raise ValueError(
                "HENS needs one IC, nonnegative nsteps and a positive ensemble size"
            )
        batch_size = nensemble if batch_size is None else batch_size
        if batch_size < 1 or ensemble_idx_base < 0 or ensemble_idx_base % nensemble:
            raise ValueError("Invalid batch size or checkpoint ensemble offset")
        batch_size = min(batch_size, nensemble)
        num_batches = -(-nensemble // batch_size)
        first_batch = ensemble_idx_base // nensemble * num_batches
        batches = (
            list(range(first_batch, first_batch + num_batches))
            if batch_ids_produce is None
            else sorted(set(batch_ids_produce))
        )
        if any(
            batch < first_batch or batch >= first_batch + num_batches
            for batch in batches
        ):
            raise ValueError("Batch IDs do not belong to this checkpoint")
        self.io_dict = io_dict
        self.runner = HENSRunner(
            prognostic,
            data,
            perturbation,
            {name: output_coords_dict[name] for name in io_dict},
            nensemble,
            batch_size,
            base_seed_string,
            dx_model_dict,
            cd_model_dict,
            cyclone_tracking,
            pkg,
        )
        self.items = []
        for batch in batches:
            start = (batch - first_batch) * batch_size
            members = tuple(
                ensemble_idx_base + i
                for i in range(start, min(start + batch_size, nensemble))
            )
            self.items.append(
                WorkItem(to_time_array(time)[0], nsteps * self.runner.step, members)
            )
        self.pipeline = Pipeline(
            self.runner, output=io_dict, device=device or local_device()
        )

    def __call__(self) -> dict[str, IOBackend]:
        """Run all selected batches and return the readable, closed backends."""
        if torch.distributed.is_initialized():
            raise RuntimeError(
                "HENS assigns checkpoint/IC pairs per process; launch without initializing a Torch process group"
            )
        self.pipeline.run(self.items)
        return self.io_dict

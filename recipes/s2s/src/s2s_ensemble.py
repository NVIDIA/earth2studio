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

from collections.abc import Callable, Iterator, Mapping, Sequence
from math import ceil

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.data import DataSource, fetch_data
from earth2studio.grids import LatLonGrid, infer_grid
from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import step_interval
from earth2studio.perturbation import Perturbation
from earth2studio.run import WorkItem

# TODO: switch to public equivalents once core exposes them
from earth2studio.run._fields import _map_field, _with_lead_time
from earth2studio.utils.coords import coord_array, coord_array_like
from earth2studio.utils.type import CoordinateSystem

from .s2s_perturbation import HENSPerturbation
from .s2s_utilities import (
    OutputRegion,
    calculate_torch_seed,
    create_base_seed_string,
    get_batchid_from_ensid,
)


class S2SRunner:
    """Run one S2S perturbation batch, for one checkpoint, as a ``Pipeline`` item.

    Global member IDs are laid out checkpoint-major: member ``m`` belongs to
    package ``m // nperturbed``, so a single ``Pipeline`` writes every
    checkpoint into one store per region. The runner swaps checkpoint weights
    (and rebuilds the perturbation around them) whenever an item belongs to a
    different package than the last one; work items are ordered package-major,
    so each rank swaps at most a few times.

    Each batch is seeded with the recipe's sha256 seed of
    ``{random_seed}_{package}_{ic}_{batch_id}``, so a batch reproduces
    regardless of rank, world size or which other batches run. Diagnostics are
    concatenated to the forecast in insertion order, and every step yields the
    configured output variables cropped to each output region.

    Parameters
    ----------
    model : PrognosticModel
        Prognostic model loaded from ``packages[0]``.
    load_model : Callable[[str], PrognosticModel]
        Loads the prognostic model for a package name.
    packages : Sequence[str]
        Package names, one per checkpoint. Also part of each batch's seed.
    data : DataSource
        Source of initial conditions.
    perturbation_factory : Callable[[PrognosticModel, DataSource], Perturbation]
        Builds the initial-condition perturbation for the active checkpoint.
    regions : Mapping[str, OutputRegion]
        Output regions, one stream (and store) each.
    output_variables : Sequence[str]
        Variables saved, from the prognostic and diagnostic outputs.
    nperturbed : int
        Perturbed members per checkpoint.
    batch_size : int
        Members per work item.
    base_random_seed : str | int
        Base of every batch seed.
    diagnostics : Mapping[str, DiagnosticModel], optional
        Diagnostics applied to every step, by default none.
    """

    supports_member_batching = True

    def __init__(
        self,
        model: PrognosticModel,
        load_model: Callable[[str], PrognosticModel],
        packages: Sequence[str],
        data: DataSource,
        perturbation_factory: Callable[[PrognosticModel, DataSource], Perturbation],
        regions: Mapping[str, OutputRegion],
        output_variables: Sequence[str],
        nperturbed: int,
        batch_size: int,
        base_random_seed: str | int,
        diagnostics: Mapping[str, DiagnosticModel] | None = None,
    ) -> None:
        self.load_model = load_model
        self.packages = list(packages)
        self.data = data
        self.perturbation_factory = perturbation_factory
        self.regions = dict(regions)
        self.output_variables = list(output_variables)
        self.nperturbed = nperturbed
        self.batch_size = min(batch_size, nperturbed)
        self.base_random_seed = str(base_random_seed)
        self.diagnostics = dict(diagnostics or {})
        self.device = torch.device("cpu")

        inputs = model.input_coords()
        outputs = model.output_coords(inputs)
        if isinstance(inputs, tuple) or isinstance(outputs, tuple):
            raise ValueError("S2S requires a single prognostic input and output field")
        if model.forcing_coords() is not None:
            raise ValueError("S2S does not support models with forcing inputs")
        in_leads = inputs["lead_time"].values
        if in_leads[-1] != np.timedelta64(0, "s"):
            raise ValueError("S2S requires a model input ending at the analysis time")
        self.inputs: CoordinateSystem = inputs
        self.output_leads = outputs["lead_time"].values
        self.step = step_interval(in_leads, self.output_leads)
        self._region_attrs = {k: r.grid_attrs() for k, r in self.regions.items()}

        # Variables available after the diagnostics run on forecasts, and whether
        # the diagnostics can also run on the initial condition to publish it.
        forecast_vars = list(outputs["variable"].values)
        ic_vars = list(inputs["variable"].values)
        ic_ok = True
        # Diagnostics chain: each sees the forecast plus earlier diagnostics' outputs
        signature = outputs
        for diagnostic in self.diagnostics.values():
            dx_in = set(diagnostic.input_coords()["variable"].values)
            if not dx_in.issubset(forecast_vars):
                raise ValueError(
                    f"Diagnostic {type(diagnostic).__name__} needs variables "
                    f"{sorted(dx_in.difference(forecast_vars))} the forecast lacks"
                )
            dx_signature = _map_field(signature, diagnostic.input_coords())
            dx_out = list(diagnostic.output_coords(dx_signature)["variable"].values)
            ic_ok = ic_ok and dx_in.issubset(ic_vars)
            forecast_vars += dx_out
            ic_vars += dx_out
            signature = coord_array_like(signature, {"variable": forecast_vars})
        missing = set(self.output_variables).difference(forecast_vars)
        if missing:
            raise ValueError(
                f"Output variables {sorted(missing)} are not produced; "
                f"available: {forecast_vars}"
            )
        self.publish_initial_condition = ic_ok and set(self.output_variables).issubset(
            ic_vars
        )
        if not self.publish_initial_condition:
            logger.warning(
                "Output variables cannot all be derived from the initial condition; "
                "outputs start at the first forecast step"
            )

        self.model = model
        self._loaded = 0  # package of self.model
        self._package: int | None = None  # package of self.perturbation
        self.perturbation: Perturbation | None = None
        self._initial: tuple[np.datetime64, xr.DataArray] | None = None

    # -- Runner protocol ----------------------------------------------------

    def to(self, device: torch.device) -> "S2SRunner":
        """Move all models to the worker's device."""
        self.device = torch.device(device)
        self.model.to(self.device)
        for diagnostic in self.diagnostics.values():
            diagnostic.to(self.device)
        return self

    def nsteps(self, final_lead_time: np.timedelta64) -> int:
        """Number of model steps reaching ``final_lead_time``."""
        if final_lead_time < np.timedelta64(0, "s") or final_lead_time % self.step:
            raise ValueError(
                f"Final lead time {final_lead_time} is not a nonnegative multiple "
                f"of the model step {self.step}"
            )
        return int(final_lead_time // self.step)

    def lead_times(self, final_lead_time: np.timedelta64) -> np.ndarray:
        """Lead times every stream covers through ``final_lead_time``."""
        leads = []
        if self.publish_initial_condition:
            leads.append(np.array([np.timedelta64(0, "h")]))
        leads += [
            self.output_leads + k * self.step
            for k in range(self.nsteps(final_lead_time))
        ]
        return np.concatenate(leads).astype("timedelta64[ns]")

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """One stream per output region; ``ensemble`` and ``time`` are dynamic."""
        leads = self.lead_times(final_lead_time)
        return {
            name: coord_array(
                ("time", "lead_time", "variable", "lat", "lon"),
                {"lead_time": leads, "variable": np.array(self.output_variables)},
                dynamic=("time",),
                grid=LatLonGrid(region.lat, region.lon),
            )
            for name, region in self.regions.items()
        }

    def data_requests(self, item: WorkItem) -> None:
        """Perturbations may fetch additional data internally."""
        return None

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish the perturbed initial condition, if possible, then each step."""
        member = item.member_ids[0]
        package = member // self.nperturbed
        if any(m // self.nperturbed != package for m in item.member_ids):
            raise ValueError(f"Members {item.member_ids} span several checkpoints")
        self._activate(package)

        with torch.inference_mode():
            x = self._fetch(item)
            # Seed after fetching, so only the perturbation draws from the stream
            torch.manual_seed(self.batch_seed(item))
            if isinstance(self.perturbation, HENSPerturbation):
                if len(item.member_ids) % 2:
                    raise ValueError(
                        "Bred-vector perturbations perturb members in +/- pairs; "
                        "use an even batch_size and nperturbed"
                    )
                self.perturbation.perturbation._residual = []
            if self.perturbation is not None:
                x = self.perturbation(x)

            if self.publish_initial_condition:
                yield self._publish(initial_condition(x))
            nsteps = self.nsteps(item.final_lead_time)
            if nsteps == 0:
                return
            iterator = self.model.create_iterator(x)
            try:
                for _ in range(nsteps):
                    yield self._publish(next(iterator))
            finally:
                iterator.close()

    # -- helpers ------------------------------------------------------------

    def batch_seed(self, item: WorkItem) -> int:
        """Torch seed of an item: sha256 of package, IC, base seed and batch ID."""
        member = item.member_ids[0]
        batch_id = get_batchid_from_ensid(self.nperturbed, self.batch_size, member)
        seed_string = create_base_seed_string(
            self.packages[member // self.nperturbed], item.time, self.base_random_seed
        )
        return calculate_torch_seed(f"{seed_string}_{batch_id}")

    def _activate(self, package: int) -> None:
        """Load the weights of ``package`` and rebuild the perturbation for them."""
        if package == self._package:
            return
        if package != self._loaded:
            logger.info(f"Loading checkpoint {self.packages[package]}")
            self.model.to("cpu")
            self.model = self.load_model(self.packages[package])
            self.model.to(self.device)
            self._loaded = package
        self.perturbation = self.perturbation_factory(self.model, self.data)
        self._package = package

    def _fetch(self, item: WorkItem) -> xr.DataArray:
        """Initial condition for ``item``, expanded over its members."""
        # Cache only the latest IC: items are ordered by time within a package
        if self._initial is None or self._initial[0] != item.time:
            x = fetch_data(
                source=self.data,
                time=np.array([item.time]),
                variable=self.inputs["variable"].values,
                lead_time=self.inputs["lead_time"].values,
                target_grid=infer_grid(self.inputs),
                device="cpu",
            )
            self._initial = (item.time, x)
        # Copy so perturbations cannot change the cached IC
        x = self._initial[1].expand_dims(ensemble=list(item.member_ids)).copy(deep=True)
        backend = "cupy" if self.device.type == "cuda" else "numpy"
        return x.e2s.to_backend(backend, device=self.device)

    def _publish(self, field: xr.DataArray) -> Mapping[str, xr.DataArray]:
        """Add diagnostics, keep the output variables and crop to each region."""
        for diagnostic in self.diagnostics.values():
            result = diagnostic(_map_field(field, diagnostic.input_coords()))
            result = _with_lead_time(result, field.lead_time.values)
            result = _map_field(
                result, {dim: field[dim].values for dim in ("lat", "lon")}
            )
            field = xr.concat([field, result], dim="variable", join="exact")
        field = field.sel(variable=self.output_variables)
        return {
            name: region.crop(field, self._region_attrs[name])
            for name, region in self.regions.items()
        }


def build_work_items(
    times: Sequence[np.datetime64],
    final_lead_time: np.timedelta64,
    npackages: int,
    nperturbed: int,
    batch_size: int,
    batch_ids: Sequence[int] | None = None,
) -> list[WorkItem]:
    """Work items for every checkpoint, initial condition and member batch.

    Items are ordered package-major, so the contiguous slices ``Pipeline`` hands
    to each rank need few checkpoint swaps. Global batch ``b`` of package ``p``
    is ``p * ceil(nperturbed / batch_size) + b``; ``batch_ids`` keeps only those
    batches (the ``batch_ids_reproduce`` config option).

    Parameters
    ----------
    times : Sequence[np.datetime64]
        Initial condition times.
    final_lead_time : np.timedelta64
        Final lead time of every item.
    npackages : int
        Number of checkpoints.
    nperturbed : int
        Perturbed members per checkpoint.
    batch_size : int
        Members per item.
    batch_ids : Sequence[int], optional
        Global batch IDs to run, by default all.

    Returns
    -------
    list[WorkItem]
        The work items.
    """
    batch_size = min(batch_size, nperturbed)
    nbatches = ceil(nperturbed / batch_size)
    keep = None if batch_ids is None else set(batch_ids)
    items = []
    for package in range(npackages):
        for time in times:
            for batch in range(nbatches):
                if keep is not None and package * nbatches + batch not in keep:
                    continue
                start = package * nperturbed + batch * batch_size
                stop = min(start + batch_size, (package + 1) * nperturbed)
                items.append(
                    WorkItem(
                        np.datetime64(time, "ns"),
                        final_lead_time,
                        tuple(range(start, stop)),
                    )
                )
    return items

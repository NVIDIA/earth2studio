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

"""Ensemble rollouts on top of the library's :class:`PrognosticRunner`.

:class:`~earth2studio.run.PrognosticRunner` rolls one member per item and has
no perturbation stage.  :class:`EnsembleRunner` adds both: an item's
``member_ids`` ride on a leading ``ensemble`` axis, each member gets its own
perturbation
under its own :func:`~earth2studio.run.work.member_seed`, and the members roll
out together through the model's batch dimension, as ``run.ensemble`` does.
Deterministic items (no members) fall through to the library runner.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.models.dx import DiagnosticModel
from earth2studio.models.px import PrognosticModel
from earth2studio.models.px.utils import initial_condition
from earth2studio.models.utils import fork_rng
from earth2studio.perturbation import Perturbation
from earth2studio.run import PrognosticRunner, WorkItem
from earth2studio.run.runner import Source, _fetch, _seed_model
from earth2studio.run.work import member_seed


def reseed_perturbation(perturbation: Perturbation, seed: int) -> None:
    """Point a perturbation's own random stream at ``seed``.

    :class:`~earth2studio.perturbation.Gaussian` and its relatives draw from a
    private ``torch.Generator`` seeded from their ``seed`` attribute on first
    use; resetting both restarts that stream.  Perturbations that draw from the
    global Torch RNG fall inside the :func:`fork_rng` scope the caller opens.

    Parameters
    ----------
    perturbation : object
        The :class:`earth2studio.perturbation.Perturbation` to reseed.
    seed : int
        New seed.
    """
    if hasattr(perturbation, "seed") and hasattr(perturbation, "generator"):
        perturbation.seed = seed
        perturbation.generator = None


class EnsembleRunner(PrognosticRunner):
    """Roll a batch of ensemble members of one initial condition together.

    Parameters
    ----------
    prognostic : PrognosticModel
        Model to roll out; its stream is ``forecast``.
    source : DataSource | ForecastSource, optional
        Source of initial conditions, by default the model's recommendation
    forcing : tuple[DataSource | ForecastSource, ...], optional
        One source per forcing slot, by default the model's recommendations
    diagnostics : Mapping[str, DiagnosticModel], optional
        Diagnostics applied every step, each yielding its own stream
    seed : int, optional
        Base seed for :func:`~earth2studio.run.work.member_seed`, by default 0
    perturbation : Perturbation, optional
        Initial-condition perturbation, drawn per member under that member's
        seed, by default None.  Models whose stochasticity drives the spread
        need none.
    """

    supports_member_batching = True

    def __init__(
        self,
        prognostic: PrognosticModel,
        source: Source | None = None,
        forcing: tuple[Source, ...] | None = None,
        diagnostics: Mapping[str, DiagnosticModel] | None = None,
        seed: int = 0,
        perturbation: Perturbation | None = None,
    ) -> None:
        super().__init__(prognostic, source, forcing, diagnostics, seed)
        self.perturbation = perturbation

    def to(self, device: torch.device) -> EnsembleRunner:
        """Move every model to ``device`` and fetch onto it."""
        super().to(device)
        return self

    def _perturb(self, x: xr.DataArray, item: WorkItem) -> xr.DataArray:
        """Perturb each member of ``x`` under its own seed.

        Drawing members one at a time keeps a member's perturbation identical
        whether it runs alone or batched with others.
        """
        if self.perturbation is None:
            return x
        members = [int(m) for m in x.coords["ensemble"].values]
        perturbed = []
        for member in members:
            seed = member_seed(self.seed, item.time, member)
            reseed_perturbation(self.perturbation, seed)
            with fork_rng(seed, device=self.device):
                perturbed.append(self.perturbation(x.sel(ensemble=[member])))
        out = xr.concat(perturbed, dim="ensemble")
        out.attrs = dict(x.attrs)
        return out.transpose(*x.dims)

    def _with_members(self, x: xr.DataArray, members: np.ndarray) -> xr.DataArray:
        """Give a fetched field a leading, writable ``ensemble`` axis."""
        return x.expand_dims(ensemble=members, axis=0).copy(deep=True)

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Publish the perturbed initial state, then every forecast step.

        A deterministic item runs through the library runner unchanged.
        Stochastic models receive one seed per item, from the first member's
        seed, so a member's trajectory in a batch is a valid but different
        draw from its single-member run; perturbations are per member.
        """
        if not item.member_ids:
            yield from super().run_item(item)
            return
        members = np.asarray(item.member_ids, dtype=np.int64)
        first = int(members[0])
        _seed_model(self.seed, item.time, first, self.prognostic)
        for diagnostic in self.diagnostics.values():
            _seed_model(self.seed, item.time, first, diagnostic)
        if len(members) > 1 and getattr(self.prognostic, "stochastic", False):
            logger.warning(
                f"{type(self.prognostic).__name__} is stochastic and rolls "
                f"{len(members)} members in one batch seeded once; members will "
                "not reproduce a one-member-per-item run (the ensemble is still a "
                "valid draw)."
            )
        nsteps = self.nsteps(item.final_lead_time)
        x = _fetch(self._initial_request(item), self.input_signature, self.device)
        x = self._perturb(self._with_members(x, members), item)
        yield self._publish(initial_condition(x))
        if nsteps == 0:
            return
        forcing = tuple(
            self._with_members(f, members)
            for f in self._fetch_forcing(self._forcing_requests(item))
        )
        iterator = self.prognostic.create_iterator(x, *forcing)
        try:
            y = next(iterator)
            yield self._publish(y)
            for _ in range(nsteps - 1):
                leads = y.coords["lead_time"].values
                step_forcing = tuple(
                    self._with_members(f, members)
                    for f in self._fetch_forcing(self._forcing_requests(item, leads))
                )
                y = iterator.send(step_forcing or None)
                yield self._publish(y)
        finally:
            iterator.close()

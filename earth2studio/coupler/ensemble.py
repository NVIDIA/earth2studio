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

"""Ensemble extension of the coupled runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import xarray as xr

from earth2studio.coupler.runner import CoupledRunner
from earth2studio.perturbation import Perturbation
from earth2studio.run.work import WorkItem, member_seed


class EnsembleCoupledRunner(CoupledRunner):
    """:class:`CoupledRunner` that perturbs chosen models' initial conditions.

    Supported: one member per work item (``member_batch=1``), with each model's
    initial condition perturbed from a seed derived from the base seed, the
    model's name, the time and the member. Not supported: several members in one
    item, which raises, and perturbing anything but initial conditions.

    Parameters
    ----------
    models, sources, couplings, window, seed
        As for :class:`CoupledRunner`
    perturbations : Mapping[str, Callable[[int], Perturbation]]
        Model name to a factory building that model's perturbation from the
        member's seed, e.g. ``lambda seed: Gaussian(0.05, seed=seed)``. Models not
        listed start unperturbed

    Raises
    ------
    ValueError
        If ``perturbations`` names a model that is not in ``models``
    """

    def __init__(
        self,
        *args: Any,
        perturbations: Mapping[str, Callable[[int], Perturbation]],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.perturbations = dict(perturbations)
        for name in self.perturbations:
            if name not in self.models:
                raise ValueError(f"Perturbation for unknown model {name!r}")

    def _start(
        self, item: WorkItem, member_id: int, initial: dict[str, xr.DataArray]
    ) -> dict[str, xr.DataArray]:
        """Perturb the listed models' initial conditions for this member."""
        start = dict(initial)
        for name, factory in self.perturbations.items():
            seed = member_seed(self._salt(name), item.time, member_id)
            start[name] = factory(seed)(start[name])
        return start

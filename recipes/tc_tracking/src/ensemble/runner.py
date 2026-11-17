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

"""Ensemble rollouts that retry unstable batches with new seeds.

:class:`StableRunner` extends Earth2Studio's
:class:`~earth2studio.run.PrognosticRunner` with a stability check: a batch of
members whose forecast drifts away from its initial state is rolled out again
with the next seed. Other parts of the pipeline follow the attempts of each
batch as a :class:`BatchObserver`.
"""

from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import xarray as xr
from loguru import logger
from tqdm import tqdm

from earth2studio.models.px import PrognosticModel
from earth2studio.run import PrognosticRunner, WorkItem
from earth2studio.run.work import member_seed, rank_info

# a batch's forecast steps, keyed by stream, as yielded by a runner
Step = Mapping[str, xr.DataArray]


class BatchObserver(Protocol):
    """Follows the attempts of each batch of :class:`StableRunner`."""

    def attempt_started(self, item: WorkItem, attempt: int, seed: int) -> None:
        """A rollout of ``item`` starts, replacing the steps of an earlier attempt.

        Parameters
        ----------
        item : WorkItem
            The batch
        attempt : int
            Number of the attempt, 0 for the first
        seed : int
            Seed of the model's random state for this attempt
        """
        ...

    def batch_finished(self, item: WorkItem, seed: int | None, attempts: int) -> None:
        """All attempts of ``item`` are done.

        Called after the last step of a stable batch, and before the NaN steps
        of a failed one.

        Parameters
        ----------
        item : WorkItem
            The batch
        seed : int | None
            Seed of the stable attempt, ``None`` if every attempt was unstable
        attempts : int
            Number of attempts made, including those of earlier runs
        """
        ...


class InstabilityDetection:
    """Detect forecasts drifting away from their initial state.

    Compares the global mean of each variable and member to its value at the
    first step after :meth:`reset`, usually the initial condition.

    Parameters
    ----------
    variables : Sequence[str]
        Variables to monitor
    thresholds : Sequence[float]
        Largest allowed change of each variable's global mean

    Raises
    ------
    ValueError
        If the number of thresholds does not match the number of variables
    """

    def __init__(self, variables: Sequence[str], thresholds: Sequence[float]) -> None:
        if len(variables) != len(thresholds):
            raise ValueError("please provide exactly one threshold per variable")
        self.variables = list(variables)
        self.thresholds = np.asarray(thresholds)
        self.reset()

    def reset(self) -> None:
        """Forget the reference, the next call sets a new one."""
        self.reference: np.ndarray | None = None

    def __call__(self, x: xr.DataArray) -> bool:
        """Check whether all members of a step are within the thresholds.

        Parameters
        ----------
        x : xr.DataArray
            One forecast step, with dimension ``variable`` and optionally
            ``ensemble``; all other dimensions are averaged over

        Returns
        -------
        bool
            True if every member is stable

        Raises
        ------
        ValueError
            If the reference, the first step after :meth:`reset`, contains NaN,
            which would fail every check
        """
        x = x.sel(variable=self.variables)
        kept = [dim for dim in ("ensemble", "variable") if dim in x.dims]
        # skipna=False: NaN makes the mean NaN, which fails the check below
        means = x.mean(dim=[d for d in x.dims if d not in kept], skipna=False)
        means = means.transpose(..., "variable").e2s.as_numpy().values

        if self.reference is None:
            if np.isnan(means).any():
                raise ValueError(
                    f"stability check variables {self.variables} contain NaN in "
                    "the initial condition, choose other variables"
                )
            self.reference = means

        return bool((np.abs(means - self.reference) < self.thresholds).all())


class StableRunner(PrognosticRunner):
    """Prognostic runner that rolls out unstable batches again with new seeds.

    Each step of an attempt is checked before it is yielded. Once a step is
    unstable, the attempt is stopped and the batch starts again from its
    initial condition, so the steps of the next attempt replace those already
    written. Attempt ``n`` seeds the model like :class:`PrognosticRunner` with
    base seed ``seed + n``, i.e. with
    ``member_seed(seed + n, item.time, item.member_ids[0])``. If all attempts
    are unstable, every step of every stream is yielded as NaN, so that the
    outputs of a failed batch hold no data of an unstable attempt.

    Parameters
    ----------
    prognostic : PrognosticModel
        Model to roll out
    stability_check : InstabilityDetection | None
        Check of each step of stream ``"forecast"``, None to accept every step
    attempts : Callable[[WorkItem], range], optional
        Attempts to make for a batch, by default only the first, ``range(1)``.
        An attempt number is the offset to the base seed, see above.
    observers : Sequence[BatchObserver], optional
        Told about the attempts of each batch, by default none
    seed : int, optional
        Base seed, by default 0
    **kwargs : Any
        Further arguments of :class:`PrognosticRunner`, e.g. ``source`` and
        ``diagnostics``
    """

    def __init__(
        self,
        prognostic: PrognosticModel,
        stability_check: InstabilityDetection | None,
        attempts: Callable[[WorkItem], range] | None = None,
        observers: Sequence[BatchObserver] = (),
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(prognostic, seed=seed, **kwargs)
        # PrognosticRunner seeds from self.seed, which each attempt offsets
        self.base_seed = seed
        self.stability_check = stability_check
        self.attempts = attempts or (lambda item: range(1))
        self.observers = list(observers)
        # last step yielded, the template of the NaN steps of a failed batch
        self._last_step: Step = {}

    def batch_seed(self, item: WorkItem, attempt: int) -> int:
        """Seed of the model's random state for one attempt of a batch.

        Parameters
        ----------
        item : WorkItem
            The batch
        attempt : int
            Number of the attempt

        Returns
        -------
        int
            The seed :class:`PrognosticRunner` derives for the attempt
        """
        first_member = item.member_ids[0] if item.member_ids else 0
        return member_seed(self.base_seed + attempt, item.time, first_member)

    def run_item(self, item: WorkItem) -> Iterator[Step]:
        """Yield the steps of the first stable attempt, or NaN steps if none is.

        Parameters
        ----------
        item : WorkItem
            The batch

        Yields
        ------
        Step
            Outputs of one step, keyed by stream
        """
        rank, _ = rank_info()
        ic = np.datetime_as_string(item.time, unit="s")
        name = f"rank {rank}: {ic} members {list(item.member_ids)}"
        attempts = self.attempts(item)
        for attempt in attempts:
            seed = self.batch_seed(item, attempt)
            logger.info(f"{name}: attempt {attempt}, seed {seed}")
            for observer in self.observers:
                observer.attempt_started(item, attempt, seed)
            stable = yield from self._attempt(item, attempt)
            if stable:
                self._last_step = {}
                self._notify_finished(item, seed, attempt + 1)
                return
            logger.warning(f"{name}: unstable with seed {seed}")

        logger.error(f"{name}: unstable in all {attempts.stop} attempts, giving up")
        self._notify_finished(item, None, attempts.stop)
        # no step without an attempt left: an earlier run wrote the NaN steps
        template, self._last_step = self._last_step, {}
        if template:
            schemas = self.output_coords(item.final_lead_time)
            yield from _nan_steps(template, schemas)

    def _attempt(self, item: WorkItem, attempt: int) -> Generator[Step, None, bool]:
        """Yield the steps of one attempt until one is unstable, True if none is."""
        if self.stability_check is not None:
            self.stability_check.reset()
        # PrognosticRunner seeds the model from self.seed when the rollout starts
        self.seed = self.base_seed + attempt
        steps = super().run_item(item)
        total = self.nsteps(item.final_lead_time) + self.publishes_initial_condition
        try:
            for step in tqdm(steps, total=total):
                if self.stability_check is not None and not self.stability_check(
                    step["forecast"]
                ):
                    return False
                self._last_step = step
                yield step
        finally:
            # closes the model's iterator, also when the attempt is stopped
            steps.close()
            self.seed = self.base_seed
        return True

    def _notify_finished(self, item: WorkItem, seed: int | None, attempts: int) -> None:
        for observer in self.observers:
            observer.batch_finished(item, seed, attempts)


def _nan_steps(template: Step, schemas: Mapping[str, xr.DataArray]) -> Iterator[Step]:
    """NaN steps for every lead time of each stream, shaped like ``template``.

    The fields are broadcast views of a single NaN, so they take no memory
    until a backend writes the selected variables.
    """
    leads = {stream: schemas[stream].coords["lead_time"].values for stream in template}
    for lead in np.unique(np.concatenate(list(leads.values()))):
        step = {}
        for stream, x in template.items():
            if lead in leads[stream]:
                nan = np.broadcast_to(np.array(np.nan, dtype=x.dtype), x.shape)
                step[stream] = x.copy(data=nan).assign_coords(lead_time=[lead])
        yield step

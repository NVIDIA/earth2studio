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

"""Ensemble rollouts that stop unstable batches early.

:class:`StableRunner` extends Earth2Studio's
:class:`~earth2studio.run.PrognosticRunner` with a stability check: the rollout
of a batch of members whose forecast drifts away from its initial state is
stopped at the first unstable step. Each run of a batch is one attempt with its
own seed; :class:`~src.ensemble.schedule.RetryScheduler` runs the next attempt
on whichever rank is free. Other parts of the pipeline follow the attempts as a
:class:`BatchObserver`.
"""

from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from typing import Any, Protocol, cast

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
        """An attempt of ``item`` is done.

        Called after the last step of a stable attempt, once an unstable one is
        stopped, and before the NaN steps of an unstable last attempt.

        Parameters
        ----------
        item : WorkItem
            The batch
        seed : int | None
            Seed of the attempt if it was stable, ``None`` if it was unstable
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
    """Prognostic runner that stops the rollout of a batch once it is unstable.

    Each run of a batch is one attempt. Every step is checked before it is
    yielded, and the attempt is stopped at the first unstable step; a later
    attempt of the batch replaces the steps already written. Attempt ``n``
    seeds the model like :class:`PrognosticRunner` with base seed ``seed + n``,
    i.e. with ``member_seed(seed + n, item.time, item.member_ids[0])``.
    Attempts from ``max_retries`` on are the last of a batch: if one is
    unstable, every step of every stream is yielded as NaN, so that the outputs
    of a failed batch hold no data of an unstable attempt.

    Parameters
    ----------
    prognostic : PrognosticModel
        Model to roll out
    stability_check : InstabilityDetection | None
        Check of each step of stream ``"forecast"``, None to accept every step
    attempt : Callable[[WorkItem], int], optional
        Number of the attempt to run for a batch, the offset to the base seed,
        by default always the first, 0
    max_retries : int, optional
        Number of the last attempt, by default 0, i.e. no retries
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
        attempt: Callable[[WorkItem], int] | None = None,
        max_retries: int = 0,
        observers: Sequence[BatchObserver] = (),
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(prognostic, seed=seed, **kwargs)
        # PrognosticRunner seeds from self.seed, which each attempt offsets
        self.base_seed = seed
        self.stability_check = stability_check
        self.attempt = attempt or (lambda item: 0)
        self.max_retries = max_retries
        self.observers = list(observers)
        # NaN copy of the last step yielded, the template of the NaN steps of a
        # failed batch; holds no field data, so no GPU memory between batches
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
        """Yield the steps of one attempt until one is unstable.

        If the last attempt of the batch is unstable, all steps are then
        yielded again as NaN.

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
        attempt = self.attempt(item)
        seed = self.batch_seed(item, attempt)
        logger.info(f"{name}: attempt {attempt}, seed {seed}")
        for observer in self.observers:
            observer.attempt_started(item, attempt, seed)

        stable = yield from self._attempt(item, attempt)
        template, self._last_step = self._last_step, {}
        if stable:
            self._notify_finished(item, seed, attempt + 1)
            return
        if attempt < self.max_retries:
            logger.warning(f"{name}: unstable with seed {seed}, to be retried")
            self._notify_finished(item, None, attempt + 1)
            return
        logger.error(f"{name}: unstable in all {attempt + 1} attempts, giving up")
        self._notify_finished(item, None, attempt + 1)
        yield from _nan_steps(template, self.output_coords(item.final_lead_time))

    def _attempt(self, item: WorkItem, attempt: int) -> Generator[Step, None, bool]:
        """Yield the steps of one attempt until one is unstable, True if none is."""
        if self.stability_check is not None:
            self.stability_check.reset()
        # PrognosticRunner seeds the model from self.seed when the rollout starts
        self.seed = self.base_seed + attempt
        # a generator, annotated as Iterator by PrognosticRunner
        steps = cast(Generator[Step, None, None], super().run_item(item))
        total = self.nsteps(item.final_lead_time) + self.publishes_initial_condition
        try:
            for step in tqdm(steps, total=total):
                if self.stability_check is not None and not self.stability_check(
                    step["forecast"]
                ):
                    return False
                self._last_step = {name: _nan_like(x) for name, x in step.items()}
                yield step
        finally:
            # closes the model's iterator, also when the attempt is stopped
            steps.close()
            self.seed = self.base_seed
        return True

    def _notify_finished(self, item: WorkItem, seed: int | None, attempts: int) -> None:
        for observer in self.observers:
            observer.batch_finished(item, seed, attempts)


def _nan_like(x: xr.DataArray) -> xr.DataArray:
    """``x`` with its data replaced by NaN, keeping dims, coordinates and attrs.

    The data is a broadcast view of a single NaN on the host, so it takes no
    memory until a backend writes it, and drops the reference to the fields.
    """
    nan = np.broadcast_to(np.array(np.nan, dtype=x.dtype), x.shape)
    return x.copy(deep=False, data=nan)


def _nan_steps(template: Step, schemas: Mapping[str, xr.DataArray]) -> Iterator[Step]:
    """NaN steps for every lead time of each stream, shaped like ``template``.

    ``template`` holds NaN fields, see :func:`_nan_like`.
    """
    leads = {stream: schemas[stream].coords["lead_time"].values for stream in template}
    for lead in np.unique(np.concatenate(list(leads.values()))):
        step = {}
        for stream, x in template.items():
            if lead in leads[stream]:
                step[stream] = x.assign_coords(lead_time=[lead])
        yield step

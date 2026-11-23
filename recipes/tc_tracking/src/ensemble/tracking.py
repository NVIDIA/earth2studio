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

"""Cyclone tracking as part of an Earth2Studio pipeline.

The tracker needs fields that are not necessarily written to the output.
:class:`TrackerInputs` therefore runs as a diagnostic of the runner, selecting
the tracker's input variables into a stream of their own, and
:class:`TrackingScorer` tracks the cyclones in that stream, one batch of members
at a time.
"""

from typing import Any

import numpy as np
import xarray as xr

from earth2studio.run import WorkItem
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.type import CoordinateSystem
from src.tempest_extremes import TempestExtremes

# name of the stream holding the tracker's input fields
TRACKING_STREAM = "tracking"


class TrackerInputs:
    """Diagnostic passing on the fields a tracker needs.

    Its input coordinates are those of the tracker, so the runner selects the
    tracker's variables and checks its grid; the fields themselves pass
    unchanged.

    Parameters
    ----------
    tracker : TempestExtremes
        Tracker whose input coordinates are used
    """

    stochastic = False

    def __init__(self, tracker: TempestExtremes) -> None:
        self.tracker = tracker

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system, the tracker's.

        Returns
        -------
        CoordinateSystem
            Signature of the tracker's input fields
        """
        return self.tracker.input_coords()

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system, the same as the input's.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Signature of the input fields

        Returns
        -------
        CoordinateSystem
            The same signature
        """
        return coord_array_like(input_coords)

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Return the fields unchanged.

        Parameters
        ----------
        x : xr.DataArray
            The tracker's input fields

        Returns
        -------
        xr.DataArray
            The same fields
        """
        return x

    def default_sources(self) -> None:
        """No recommended data source, the inputs come from the forecast."""
        return None

    def to(self, device: Any) -> "TrackerInputs":
        """Nothing to move, the tracker runs on the CPU.

        Parameters
        ----------
        device : Any
            Ignored

        Returns
        -------
        TrackerInputs
            This diagnostic
        """
        return self


class TrackingScorer:
    """Track cyclones in every batch, as scorer of an Earth2Studio pipeline.

    Records the fields of stream :data:`TRACKING_STREAM` at every step and,
    once the batch is done, writes one track file per member,
    ``tracks_<ic>_mem_<member>_seed_<seed>_bs_<batch_size>.csv``. The batch
    counts as done only once its track files are written, so a resumed run
    never skips a batch without tracks.

    As a :class:`~src.ensemble.runner.BatchObserver` of the runner, it
    discards the steps of an unstable attempt and does not track it.

    Parameters
    ----------
    tracker : TempestExtremes
        Tracker to run
    batch_size : int
        Batch size of the run, part of the track file names
    """

    def __init__(self, tracker: TempestExtremes, batch_size: int) -> None:
        self.tracker = tracker
        self.batch_size = batch_size
        # seed of the batch's attempt once it is stable, None while it runs
        self._seed: int | None = None
        # set once the attempt was unstable
        self._failed = False

    def begin_item(self, item: WorkItem) -> None:
        """Start a batch.

        Parameters
        ----------
        item : WorkItem
            The batch
        """
        self._seed = None
        self._failed = False

    def attempt_started(self, item: WorkItem, attempt: int, seed: int) -> None:
        """Discard the steps of an earlier attempt.

        Parameters
        ----------
        item : WorkItem
            The batch
        attempt : int
            Number of the attempt
        seed : int
            Seed of the attempt
        """
        self.tracker.reset()

    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None:
        """Record the tracker's input fields of one step.

        Parameters
        ----------
        item : WorkItem
            The batch
        stream : str
            Stream of the fields; only :data:`TRACKING_STREAM` is recorded
        x : xr.DataArray
            Fields of one step
        """
        # a failed batch yields NaN steps, which are not tracked
        if stream != TRACKING_STREAM or self._failed:
            return
        if "ensemble" not in x.dims:
            # a batch of one member is run without ensemble dimension
            x = x.expand_dims(ensemble=list(item.member_ids))
        self.tracker.record_state(x)

    def batch_finished(self, item: WorkItem, seed: int | None, attempts: int) -> None:
        """Note the seed of a stable attempt, or discard an unstable one.

        Parameters
        ----------
        item : WorkItem
            The batch
        seed : int | None
            Seed of the attempt, None if it was unstable
        attempts : int
            Number of attempts made
        """
        self._seed = seed
        self._failed = seed is None
        if self._failed:
            self.tracker.reset()

    def finish_item(self, item: WorkItem) -> None:
        """Track the cyclones of a stable batch and wait for its track files.

        Parameters
        ----------
        item : WorkItem
            The batch

        Raises
        ------
        ChildProcessError
            If tracking failed for a member
        """
        if self._failed:
            return
        ic = np.datetime_as_string(item.time, unit="s")
        self.tracker(
            out_file_names=[
                f"tracks_{ic}_mem_{member:04d}_seed_{self._seed}"
                f"_bs_{self.batch_size}.csv"
                for member in item.member_ids
            ]
        )
        self.tracker.wait()

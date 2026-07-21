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

from collections import OrderedDict
from collections.abc import Generator, Iterator

import numpy as np
import torch

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem


class DataReplay(torch.nn.Module, PrognosticMixin):
    """Adapt any :class:`~earth2studio.data.DataSource` into a prognostic model, so a stored or
    observed dataset can be stepped through the same iterator interface as a forecast model.

    Each step after the initial condition fetches ``source`` at the next valid time (advancing
    ``lead_time`` by ``step``) instead of forecasting, yielding the source's own frames as the
    trajectory. Any pipeline built around the prognostic iterator can then be driven by
    reanalysis/analysis frames rather than a forecast rollout -- for example, supplying source frames
    to a downstream model (such as a temporal interpolator), building a reference trajectory to score
    forecasts against, or sub-sampling a finer source in time. It complements
    :class:`~earth2studio.models.px.persistence.Persistence` (which echoes the initial state forward)
    by instead pulling a fresh frame each step. For the forecast-as-trajectory case, prefer
    :class:`~earth2studio.data.ForecastSource`.

    ``step`` sets the spacing between emitted frames (equivalently, the ``lead_time`` increment, and the
    sub-sampling interval when ``source`` is available at a finer resolution).

    Notes
    -----
    * **Deterministic / single realization.** ``source`` is fetched once and broadcast across the batch
      (ensemble) dim -- all members get the same coarse frame; member diversity must come from the
      downstream model (a stochastic source is sampled once, not per member).
    * **On-grid source required.** The fetched data is not regridded; ``_fetch`` raises if the source
      grid does not match ``domain_coords`` (crop/interpolate the source beforehand if it differs), and
      also raises on non-finite (fill/masked) values since the downstream model has no NaN handling.
    * **Stateless.** No history window, no checkpoint state -- a resume re-queries ``source``, so a
      resumed run reproduces the original only insofar as the source itself is reproducible. The step-0 initial condition is assumed to be
      ``source`` at ``time0`` (true when the same object is passed to the run's ``data`` and here).

    Parameters
    ----------
    source : DataSource
        The data source to replay (queried by absolute valid time).
    variable : str | list[str]
        Variables to emit (Earth2Studio ids), in the order the downstream model expects.
    domain_coords : CoordSystem
        Spatial coordinates (``lat``/``lon``) the source is expected to provide.
    step : np.timedelta64 | int | float, optional
        Spacing between emitted frames (also the ``lead_time`` increment). ``int``/``float`` is
        interpreted as whole hours (a non-integral float raises). Default ``np.timedelta64(6, "h")``.
    """

    def __init__(
        self,
        source: DataSource,
        variable: str | list[str],
        domain_coords: CoordSystem,
        step: np.timedelta64 | int | float = np.timedelta64(6, "h"),
    ) -> None:
        super().__init__()
        if isinstance(variable, str):
            variable = [variable]
        if isinstance(
            step, bool
        ):  # bool subclasses int; reject before the numeric branch
            raise TypeError(
                f"step must be int/float hours or np.timedelta64, got {type(step)}"
            )
        if isinstance(
            step, np.timedelta64
        ):  # checked first: np.timedelta64 subclasses np.integer
            if np.isnat(step):
                raise ValueError("step must be a finite positive duration, got NaT")
        elif isinstance(step, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(step)):
                raise ValueError(
                    f"int/float step must be finite whole hours, got {step}"
                )
            if float(step) != int(step):
                raise ValueError(f"int/float step must be whole hours, got {step}")
            step = np.timedelta64(int(step), "h")
        else:
            raise TypeError(
                f"step must be int/float hours or np.timedelta64, got {type(step)}"
            )
        if step <= np.timedelta64(0, "h"):
            raise ValueError(f"step must be positive, got {step}")
        self.source = source
        self.step = step
        self._variable = np.array(variable)
        self._domain = OrderedDict(domain_coords)
        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(variable),
            }
        )
        for key, value in self._domain.items():
            self._input_coords[key] = value

    def __str__(self) -> str:
        return "DataReplay"

    def input_coords(self) -> CoordSystem:
        """Input coordinate system.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary (``batch, time, lead_time, variable, lat, lon``).
        """
        return self._input_coords.copy()

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system: one coarse ``step`` past the input lead time.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary.
        """
        output_coords = self._input_coords.copy()
        target = self.input_coords()
        handshake_dim(input_coords, "variable", -3)
        handshake_dim(input_coords, "lat", -2)
        handshake_dim(input_coords, "lon", -1)
        handshake_coords(input_coords, target, "variable")
        handshake_coords(input_coords, target, "lat")
        handshake_coords(input_coords, target, "lon")
        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords.get("time", np.empty(0))
        output_coords["lead_time"] = (
            np.array([self.step]) + input_coords["lead_time"][-1]
        )
        return output_coords

    @torch.inference_mode()
    def _fetch(
        self,
        times: np.ndarray,
        offset: np.timedelta64,
        batch: int,
        device: torch.device,
    ) -> torch.Tensor:
        # Fetch the source at ``times + offset``; shape (batch, time, 1, var, lat, lon).
        # Raises if the source grid does not match ``domain_coords`` (no silent regrid/mislabel).
        # Returns a materialized (contiguous) tensor so the broadcast batch dim does not alias storage.
        data, out = fetch_data(
            self.source,
            time=times,
            variable=self._variable,
            lead_time=np.array([offset]),
            device=device,
        )
        for ax in ("lat", "lon"):
            got, want = np.asarray(out[ax]), np.asarray(self._domain[ax])
            if got.shape != want.shape or not np.allclose(got, want):
                raise ValueError(
                    f"source {ax} grid (n={got.shape}) does not match domain_coords (n={want.shape}); "
                    "DataReplay does not regrid -- provide a source already on the domain grid."
                )
        if not torch.isfinite(data).all():
            raise ValueError(
                "DataReplay: source returned non-finite values (fill/ocean-masked field?); "
                "the downstream model has no NaN handling -- provide gap-filled data."
            )
        return data.unsqueeze(0).expand(batch, *data.shape).contiguous()

    @staticmethod
    def _require_time(coords: CoordSystem) -> None:
        if "time" not in coords or len(np.atleast_1d(coords["time"])) == 0:
            raise ValueError(
                "DataReplay requires coords['time'] (absolute valid time(s)); none provided."
            )

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Advance one coarse step: fetch the source at ``time + lead_time + step``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (used only for batch size / device / dtype).
        coords : CoordSystem
            Input coordinate system; must contain ``time``.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            The fetched frame and its coordinate system, one ``step`` ahead.

        Notes
        -----
        ``front_hook`` / ``rear_hook`` are **not** applied here -- they fire only in
        :meth:`create_iterator` (matching :class:`~earth2studio.models.px.persistence.Persistence`).
        Use :meth:`create_iterator` when hook transformations are required.
        """
        self._require_time(coords)
        out_coords = self.output_coords(coords)
        data = self._fetch(
            coords["time"], self.step + coords["lead_time"][-1], x.shape[0], x.device
        )
        return data.to(x.dtype), out_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        self._require_time(coords)
        # Validate input dim order + grid up front, so the iterator rejects a variable/lat/lon mismatch
        # instead of silently mislabelling fetched frames.
        self.output_coords(coords)
        # front_hook transforms the provided input before the replay begins. DataReplay has no forward
        # pass, so it fires once on the input that defines the step-0 initial condition and the query times.
        x, coords = self.front_hook(x, coords)
        # Step 0: the provided initial condition is the source at time0 (the last input lead slice).
        # rear_hook transforms every emitted frame (the IC and each fetched frame); the default identity
        # hook leaves them unmodified, preserving the verbatim-replay contract.
        coords0 = coords.copy()
        coords0["lead_time"] = coords["lead_time"][-1:]
        yield self.rear_hook(x[:, :, -1:], coords0)
        k = 1
        while True:
            offset = coords["lead_time"][-1] + k * self.step
            data = self._fetch(coords["time"], offset, x.shape[0], x.device).to(x.dtype)
            out = coords.copy()
            out["lead_time"] = np.array([offset])
            yield self.rear_hook(data, out)
            k += 1

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Iterator yielding the initial condition, then successive source frames at each coarse step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system; must contain ``time``.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Each coarse-step frame and its coordinate system.
        """
        yield from self._default_generator(x, coords)

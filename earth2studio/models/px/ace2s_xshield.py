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

"""ACE2S-X-SHiELD prognostic wrapper.

The stepper, packing and forcing window match :class:`ACE2ERA5`. This subclass
adds a reproducible noise stream for the stochastic ACE2S checkpoint and
points the packaged weights at ``allenai/HiRO-ACEv2``. A later ACE2S-ERA5
release can subclass this and override :meth:`load_default_package`,
:meth:`load_model` and :meth:`default_sources`.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Generator
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data import ACE2SXSHiELDData
from earth2studio.data.base import DataSource
from earth2studio.models.auto import Package
from earth2studio.models.px.ace2 import ACE2ERA5
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils.coords import handshake_nonempty
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    from fme.ace.stepper.single_module import load_stepper
except ImportError:
    OptionalDependencyFailure("ace2")
    load_stepper = Any


def _indexed_seed(rollout_seed: int, index: int) -> int:
    """Derive the noise seed for step ``index`` of one rollout.

    Parameters
    ----------
    rollout_seed : int
        Seed drawn once when the rollout is initialized.
    index : int
        Zero-based step within that rollout.

    Returns
    -------
    int
        Seed for that step. The same pair always reproduces it.
    """
    generator = torch.Generator()
    generator.manual_seed(int(rollout_seed))
    draws = torch.randint(0, 2**31 - 1, (index + 1,), generator=generator)
    return int(draws[-1].item())


@contextlib.contextmanager
def _fork_global_rng(seed: int) -> Generator[None, None, None]:
    """Seed the global generators fme reads, then restore them.

    fme's ``randn`` helpers call ``torch.randn`` with no generator argument.
    The fork covers the Torch state on CPU and CUDA. NumPy and the Python
    ``random`` module are saved and restored around the same block.
    """
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if devices:
                torch.cuda.manual_seed_all(int(seed))
            np.random.seed(int(seed) % (2**32))
            random.seed(int(seed))
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


@check_optional_dependencies()
class ACE2SXSHiELD(ACE2ERA5):
    """ACE2S-X-SHiELD stochastic climate emulator.

    ACE2S-X-SHiELD is a stochastic Ai2 Climate Emulator trained to reproduce a
    1-degree coarsening of GFDL's X-SHiELD simulation. It steps the global
    atmosphere every 6 hours. Precipitation rate, 10 m winds and sea-level
    pressure from this rollout are the coarse inputs of :class:`HiROv2`.

    The wrapper keeps the noise position of one rollout in the state returned
    by :meth:`initialize` and :meth:`step`. It does not write an Earth2Studio
    checkpoint catalog. Call :meth:`set_rng` before a new rollout. Continuing
    an existing ``(y, state)`` pair ignores later :meth:`set_rng` calls.

    Parameters
    ----------
    stepper : Stepper
        Loaded ``fme.ace.stepper.single_module.Stepper``.
    dt : numpy.timedelta64, optional
        Lead-time increment, by default 6 hours.

    References
    ----------
    - HiRO-ACEv2: https://arxiv.org/abs/2610.17693
    - ACE code: https://github.com/ai2cm/ace
    - Hugging Face: https://huggingface.co/allenai/HiRO-ACEv2

    Warning
    -------
    Use input data on the same CUDA device the package was loaded on.
    ``torch.cuda.current_device()`` at load time is the device fme keeps.

    Badges
    ------
    region:global class:climate product:wind product:precip product:temp product:atmos
    year:2026 provider:ai2 backend:pytorch
    """

    stochastic: bool = True
    _CHECKPOINT_FILE = "ACE2S-X-SHiELD.ckpt"

    def __init__(
        self,
        stepper: Any,
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ) -> None:
        super().__init__(stepper=stepper, dt=dt)
        self._rng: torch.Generator | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the stream used to start new rollouts.

        Parameters
        ----------
        seed : int
            Seed for rollouts created after this call.
        reset : bool, optional
            When True, replace any existing stream. When False, seed only if
            no stream exists yet, by default True.
        """
        if reset or self._rng is None:
            self._rng = torch.Generator()
            self._rng.manual_seed(int(seed))

    def _next_rollout_seed(self) -> int:
        if self._rng is None:
            return int(torch.randint(0, 2**31 - 1, (1,)).item())
        return int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())

    def _predict(
        self, x: xr.DataArray, forcing: xr.DataArray, seed: int | None = None
    ) -> xr.DataArray:
        step_seed = self._next_rollout_seed() if seed is None else int(seed)
        with _fork_global_rng(step_seed):
            return super()._predict(x, forcing)

    def _retain_missing(self, x: xr.DataArray) -> xr.DataArray:
        missing = [
            name
            for name in self._prog_vars_e2s
            if name not in self._all_out_variables_e2s
        ]
        if not missing:
            retained = x.isel(variable=slice(0, 0))
        else:
            retained = x.sel(variable=missing)
        return retained.copy(deep=True)

    # The state widens ACE2ERA5's (missing, forcing) pair with the rollout seed and
    # step index, which is why the override signature differs.
    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray, int, int]]:
        """Predict the first 6-hour state and keep the rollout noise seed.

        Parameters
        ----------
        x : xr.DataArray
            Initial prognostic fields matching ``input_coords()``.
        forcing : xr.DataArray
            Forcing window matching ``forcing_coords()`` at the input time and
            the first forecast time.

        Returns
        -------
        tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray, int, int]]
            First forecast, then state ``(missing prognostics, last forcing
            frame, rollout seed, step index)``. The step index is 0 for this
            forecast. Iterator hooks are not applied.
        """
        handshake_nonempty(x)
        rollout_seed = self._next_rollout_seed()
        y = self._predict(x, forcing, _indexed_seed(rollout_seed, 0))
        return y, (
            self._retain_missing(x),
            forcing.isel(lead_time=slice(-1, None)).copy(deep=True),
            int(rollout_seed),
            0,
        )

    def step(  # type: ignore[override]
        self,
        y: xr.DataArray,
        forcing: xr.DataArray,
        state: tuple[xr.DataArray, xr.DataArray, int, int],
    ) -> tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray, int, int]]:
        """Advance one 6-hour step from the previous forecast and its state.

        Parameters
        ----------
        y : xr.DataArray
            Previous forecast, including any caller-applied edits.
        forcing : xr.DataArray
            Forcing frame at the next forecast time. The frame before it is
            the second entry of ``state``.
        state : tuple[xr.DataArray, xr.DataArray, int, int]
            Missing prognostic fields, previous forcing frame, rollout seed
            and the step index of ``y``.

        Returns
        -------
        tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray, int, int]]
            Next forecast and updated state. ``y`` and ``state`` are not
            modified, and iterator hooks are not applied.
        """
        missing, previous_forcing, rollout_seed, step_index = state
        self._validate_forcing(y, forcing, continuation=True)
        window = xr.concat(
            [previous_forcing, forcing],
            dim="lead_time",
            coords="minimal",
            compat="override",
            join="exact",
        )
        present_names = [
            name for name in self._prog_vars_e2s if name in self._all_out_variables_e2s
        ]
        present = y.sel(variable=present_names)
        if missing.sizes.get("variable", 0) == 0:
            x = present
        else:
            carried = missing.assign_coords(lead_time=y.lead_time.values)
            x = xr.concat(
                [present, carried],
                dim="variable",
                coords="minimal",
                compat="override",
                join="exact",
            )
        x = x.sel(variable=self._prog_vars_e2s)
        x.attrs, x.encoding = deepcopy(y.attrs), deepcopy(y.encoding)
        y_next = self._predict(
            x, window, _indexed_seed(int(rollout_seed), step_index + 1)
        )
        return y_next, (
            self._retain_missing(x),
            window.isel(lead_time=slice(-1, None)).copy(deep=True),
            int(rollout_seed),
            int(step_index) + 1,
        )

    def default_sources(self) -> tuple[DataSource, DataSource]:
        """Recommend X-SHiELD initial conditions and forcing.

        Returns
        -------
        tuple[DataSource, DataSource]
            ``ACE2SXSHiELDData`` in initial-condition mode, then forcing mode.
            Both read ``allenai/HiRO-ACEv2``.
        """
        return (
            ACE2SXSHiELDData(mode="initial_conditions"),
            ACE2SXSHiELDData(mode="forcing"),
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the HiRO-ACEv2 package that contains the ACE2S checkpoint.

        Returns
        -------
        Package
            Hugging Face package ``allenai/HiRO-ACEv2``.
        """
        return Package(
            "hf://allenai/HiRO-ACEv2",
            cache_options={
                "cache_storage": Package.default_cache("hiro_acev2"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ) -> PrognosticModel:
        """Load ACE2S-X-SHiELD from a package.

        Parameters
        ----------
        package : Package
            Package that contains ``ACE2S-X-SHiELD.ckpt``.
        dt : numpy.timedelta64, optional
            Lead-time increment, by default 6 hours.

        Returns
        -------
        PrognosticModel
            Loaded ACE2S-X-SHiELD wrapper.
        """
        checkpoint_path = package.resolve(cls._CHECKPOINT_FILE)
        stepper = load_stepper(checkpoint_path)
        return cls(stepper=stepper, dt=dt)

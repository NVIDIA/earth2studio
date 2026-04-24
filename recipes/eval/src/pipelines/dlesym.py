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

"""Coupled Earth-system DLESyM / DLESyMLatLon forecast pipeline."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
from omegaconf import DictConfig
from physicsnemo.distributed import DistributedManager
from tqdm import tqdm

from earth2studio.data import DataSource, fetch_data
from earth2studio.models.px.dlesym import DLESyM
from earth2studio.utils.coords import CoordSystem, map_coords

from ..models import load_prognostic
from ..work import WorkItem
from .base import PredownloadStore
from .forecast import ForecastPipeline


class DLESyMPipeline(ForecastPipeline):
    """Coupled Earth-system forecast pipeline for DLESyM / DLESyMLatLon.

    DLESyM differs from standard prognostic models in three ways that make it
    incompatible with :class:`ForecastPipeline`:

    1. **Multi-lead-time per iterator step.**  Each call to the model
       advances the state by 96 hours and yields 16 output lead times
       (6h, 12h, …, 96h) in a single chunk — rather than one lead time per
       iterator step.  :attr:`nsteps` therefore counts *model steps*, not
       output lead times; the zarr schema covers ``nsteps × 16 + 1`` lead
       times at 6-hour stride.
    2. **Coupled atmosphere / ocean with different time scales.**  Ocean
       variables are only valid at the 48h and 96h ticks within each 96-hour
       step; atmosphere variables are valid at all 16 ticks.  Invalid ocean
       values are masked to NaN before writing so the zarr store reflects
       physical validity.
    3. **Initial condition yield.**  The model iterator yields the IC window
       (lead_times ``[-48h .. 0h]``) as its first output; those negative
       lead times are outside the output-zarr schema and are explicitly
       skipped.

    The model's own :meth:`retrieve_valid_ocean_outputs` determines which
    lead times are ocean-valid — we don't hardcode the [48h, 96h] list, so
    variants of DLESyM with different ocean cadences would continue to
    work.

    Diagnostics on top of DLESyM are not supported — :meth:`setup` raises
    if ``cfg.diagnostics`` is configured.  The build-in deterministic
    model has no ``set_rng`` method, so per-member seeding is a no-op
    (but the ensemble dim is still honored for repeated runs with
    different perturbations).
    """

    _ocean_variables: list[str]

    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        if cfg.get("diagnostics"):
            raise ValueError(
                "DLESyMPipeline does not support diagnostic models; "
                "remove the 'diagnostics' config key."
            )
        super().setup(cfg, device)
        if not hasattr(self.prognostic, "ocean_variables"):
            raise AttributeError(
                "DLESyMPipeline expects the loaded prognostic to expose "
                "`ocean_variables` (DLESyM / DLESyMLatLon from earth2studio). "
                f"Got: {type(self.prognostic).__name__}"
            )
        self._ocean_variables = list(self.prognostic.ocean_variables)

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare IC + optional verification stores for a DLESyM run.

        Verification times are computed by flattening the model's
        per-step output lead times across ``nsteps`` iterator steps —
        this produces the full 6-hourly grid of valid times rather than
        the single-tick-per-step output that
        :meth:`ForecastPipeline.predownload_stores` would give.
        """
        from ..predownload_utils import (
            declare_single_source_stores,
            single_source_stores_disabled,
        )
        from ..work import build_work_items

        if single_source_stores_disabled(cfg):
            return []

        # CPU-inspect the model to infer IC requirements + output lead times.
        model = load_prognostic(cfg)
        ic_coords = model.input_coords()
        out_coords = model.output_coords(ic_coords)
        spatial_ref = out_coords  # lat/lon (LatLon) or face/height/width (raw)

        all_items = build_work_items(cfg)
        unique_ic_times: list[np.datetime64] = sorted({i.time for i in all_items})

        ic_variables = list(ic_coords["variable"])
        ic_lead_times = ic_coords["lead_time"]
        ic_fetch_times: list[np.datetime64] = sorted(
            {t + lt for t in unique_ic_times for lt in ic_lead_times}
        )

        # All unique forecast valid times across the full nsteps rollout.
        # Flattens per-step output lead times so every 6h tick is covered.
        verif_times = _unique_forecast_valid_times(
            unique_ic_times, out_coords["lead_time"], cfg.nsteps
        )

        return declare_single_source_stores(
            cfg,
            ic_variables=ic_variables,
            ic_times=ic_fetch_times,
            verif_variables=list(cfg.output.variables),
            verif_times=verif_times,
            spatial_ref=spatial_ref,
        )

    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        x, coords = fetch_data(
            source=data_source,
            time=[item.time],
            variable=self._prognostic_ic["variable"],
            lead_time=self._prognostic_ic["lead_time"],
            device=device,
        )
        x, coords = map_coords(x, coords, self._prognostic_ic)

        if self.perturbation is not None:
            torch.manual_seed(item.seed)
            x, coords = self.perturbation(x, coords)

        if hasattr(self.prognostic, "set_rng"):
            self.prognostic.set_rng(seed=item.seed, reset=True)

        model_iter = self.prognostic.create_iterator(x, coords)

        # Skip the IC yield — its lead_times are in the input window
        # ([-48h..0h] for DLESyM), outside the output zarr schema.
        next(model_iter)

        if not DistributedManager.is_initialized():
            DistributedManager.initialize()
        rank = DistributedManager().rank

        for step, (x_step, coords_step) in enumerate(
            tqdm(
                model_iter,
                total=self.nsteps,
                desc=f"IC {item.time}",
                position=1,
                leave=False,
                disable=rank != 0,
            )
        ):
            x_step = self._mask_invalid_ocean(x_step, coords_step)
            yield x_step, coords_step

            if step + 1 >= self.nsteps:
                break

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _mask_invalid_ocean(
        self,
        x_step: torch.Tensor,
        coords_step: CoordSystem,
    ) -> torch.Tensor:
        """Replace ocean-variable values at non-valid lead times with NaN.

        DLESyM's ocean component predicts only at a subset of the atmos
        output lead times (by default 48h and 96h).  The atmos output
        tensor still has entries for every lead time — those entries are
        meaningless for ocean variables and should not be written to the
        output zarr as if they were forecasts.

        The set of valid ocean lead times is queried from the model
        (via ``retrieve_valid_ocean_outputs``) rather than hardcoded, so
        DLESyM variants with different ocean cadences continue to work.
        """
        if not self._ocean_variables:
            return x_step

        if not isinstance(self.prognostic, DLESyM):
            raise ValueError(
                "DLESyMPipeline expects the loaded prognostic to be a DLESyM model; "
                f"Got: {type(self.prognostic).__name__}"
            )
        _, valid_coords = self.prognostic.retrieve_valid_ocean_outputs(
            x_step, coords_step
        )
        valid_lt = set(valid_coords["lead_time"].tolist())
        all_lt = list(coords_step["lead_time"].tolist())
        all_vars = list(coords_step["variable"])
        ocean_set = set(self._ocean_variables)

        lt_invalid = torch.tensor(
            [lt not in valid_lt for lt in all_lt],
            device=x_step.device,
        )
        var_ocean = torch.tensor(
            [v in ocean_set for v in all_vars],
            device=x_step.device,
        )
        if not lt_invalid.any() or not var_ocean.any():
            return x_step

        lt_axis = list(coords_step.keys()).index("lead_time")
        var_axis = list(coords_step.keys()).index("variable")

        lt_shape = [1] * x_step.ndim
        lt_shape[lt_axis] = -1
        var_shape = [1] * x_step.ndim
        var_shape[var_axis] = -1

        mask = lt_invalid.view(lt_shape) & var_ocean.view(var_shape)
        nan = torch.full_like(x_step, float("nan"))
        return torch.where(mask, nan, x_step)


def _unique_forecast_valid_times(
    ic_times: list[np.datetime64],
    step_lead_times: np.ndarray,
    nsteps: int,
) -> list[np.datetime64]:
    """Flatten per-step output lead times into the full set of valid times.

    For a model whose single forward pass outputs multiple lead times
    (e.g. DLESyM: 16 per step), :func:`compute_verification_times` from
    ``predownload_utils`` *undercounts* — it assumes one tick per step.
    This helper takes the model's output lead times and stride (derived
    from the last entry) and walks ``nsteps`` iterator steps to produce
    the full time set, including the IC itself.

    Parameters
    ----------
    ic_times : list[np.datetime64]
        Unique initial-condition times.
    step_lead_times : np.ndarray
        Output lead times from a single model step, e.g. ``[6h, 12h, …, 96h]``.
    nsteps : int
        Number of iterator steps.

    Returns
    -------
    list[np.datetime64]
        Sorted unique set of ``{ic + lead}`` across all ICs and all
        flattened lead times.
    """
    step_stride = step_lead_times[-1]
    zero = np.array([np.timedelta64(0, "ns")])
    all_offsets = np.concatenate(
        [
            zero,
            np.asarray([step_lead_times + step_stride * i for i in range(nsteps)])
            .flatten()
            .astype("timedelta64[ns]"),
        ]
    )
    return sorted({t + off for t in ic_times for off in all_offsets})

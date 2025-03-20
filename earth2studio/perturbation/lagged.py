# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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


import torch

from earth2studio.data import DataSource, fetch_data
from earth2studio.utils.coords import handshake_dim
from earth2studio.utils.type import CoordSystem, LeadTimeArray


class LaggedEnsemble:
    """Lagged Ensemble perturbation method. This method creates an ensemble
    by collecting initial conditions from different times, but verifying
    at the same time.

    Parameters
    ----------
    source: DataSource
        The source for which to draw initial conditions from.
    lags: LeadTimeArray
        The list lags, in the form of LeadTimeArray, i.e., np.timedelta64, to include.
        Positive lags refer to future times whereas negative lags refer to
        previous times.  Using positive lags should not be used for forecasting
        purposes but can be used for hindcasting. A lag of zero is effectively
        an unperturbed ensemble member.

        For example,
            lags = np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")])
            creates an ensemble with two lags, one of which is the current time and
            one is 6 hours in the past.

    Note
    ----
    For additional information:

    - https://doi.org/10.1111/j.1600-0870.1983.tb00189.x
    """

    def __init__(
        self,
        source: DataSource,
        lags: LeadTimeArray,
    ):
        self.source = source
        self.lags = lags

    @torch.inference_mode()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Apply perturbation method

        Parameters
        ----------
        x : torch.Tensor
            Input tensor intended to apply perturbation on
        coords : CoordSystem
            Ordered dict representing coordinate system that describes the tensor

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]:
            Output tensor and respective coordinate system dictionary
        """

        # Assume order of dimensions here
        handshake_dim(coords, "ensemble", 0)
        handshake_dim(coords, "time", 1)
        handshake_dim(coords, "lead_time", 2)

        if len(coords["ensemble"]) != len(self.lags):
            raise ValueError(
                f"Warning! The number of lags, {len(self.lags)}, does not match "
                f"the number of ensemble members requested, {len(coords['ensemble'])}."
            )
        y = torch.clone(x)
        for i, lag in enumerate(self.lags):
            y[i] = fetch_data(
                source=self.source,
                time=coords["time"] + lag,
                variable=coords["variable"],
                lead_time=coords["lead_time"],
                device=y.device,
            )[0]

        return y, coords

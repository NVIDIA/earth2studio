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


import os
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from AI_WQ_package import retrieve_evaluation_data


def get_verif_data(
    forecast_date: str, variable: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """Retrieve official AIWQ verification data for a given forecast date and variable.

    Parameters
    ----------
    forecast_date : str
        The forecast date in the format YYYYMMDD.
    variable : str
        The variable to retrieve verification data for.

    Returns
    -------
    obs1 : xr.DataArray
        The verification data for the first week.
    obs2 : xr.DataArray
        The verification data for the second week.
    """

    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)

    # Download observations
    obs1 = retrieve_evaluation_data.retrieve_weekly_obs(
        fc_valid_date1, variable, password=os.getenv("AIWQ_SUBMIT_PWD")
    )
    obs2 = retrieve_evaluation_data.retrieve_weekly_obs(
        fc_valid_date2, variable, password=os.getenv("AIWQ_SUBMIT_PWD")
    )

    return obs1, obs2


def get_quintile_clim(
    forecast_date: str, variable: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """Retrieve the official AIWQ quintile climatology for a given forecast date and variable.

    Parameters
    ----------
    forecast_date : str
        The forecast date in the format YYYYMMDD.
    variable : str
        The variable to retrieve quintile climatology for.

    Returns
    -------
    clim1 : xr.DataArray
        The quintile climatology for the first week.
    clim2 : xr.DataArray
        The quintile climatology for the second week.
    """

    fc_valid_date1, fc_valid_date2 = valid_dates(forecast_date)

    clim1 = retrieve_evaluation_data.retrieve_20yr_quintile_clim(
        fc_valid_date1, variable, password=os.getenv("AIWQ_SUBMIT_PWD")
    )
    clim2 = retrieve_evaluation_data.retrieve_20yr_quintile_clim(
        fc_valid_date2, variable, password=os.getenv("AIWQ_SUBMIT_PWD")
    )

    return clim1, clim2


def valid_dates(forecast_date: str) -> tuple[str, str]:
    """Specify the valid dates for a given forecast date.

    Parameters
    ----------
    forecast_date : str
        The forecast date in the format YYYYMMDD.

    Returns
    -------
    tuple[str, str]
        A tuple containing the valid dates for the forecast.
    """

    date_obj = datetime.strptime(
        forecast_date, "%Y%m%d"
    )  # get initial date as a date obj

    # add number of days to date object depending on lead time
    fc_valid_date_obj1 = date_obj + timedelta(
        days=4 + (7 * 2)
    )  # get to the next Monday then add number of weeks
    fc_valid_date_obj2 = date_obj + timedelta(days=4 + (7 * 3))

    fc_valid_date1 = fc_valid_date_obj1.strftime(
        "%Y%m%d"
    )  # convert date obj back to a string
    fc_valid_date2 = fc_valid_date_obj2.strftime(
        "%Y%m%d"
    )  # convert date obj back to a string

    return fc_valid_date1, fc_valid_date2


def convert_to_quintile_probs(data: xr.DataArray, edges: xr.DataArray) -> xr.DataArray:
    """Convert data to quintile probabilities based on the given edges.

    Parameters
    ----------
    data : xr.DataArray
        The data to convert to quintiles.
    edges : xr.DataArray
        The edges of the quintiles.

    Returns
    -------
    quintile_probs : xr.DataArray
        The quintile probabilities.
    """

    full_edges = xr.concat(
        [
            xr.full_like(edges.isel(quantile=0), -np.inf),
            edges,
            xr.full_like(edges.isel(quantile=-1), np.inf),
        ],
        dim="quantile",
    )

    # Now digitize the ensemble values into these bins
    # np.digitize assigns bin indices in [1, ..., len(bins)-1]
    # So subtract 1 to get bin index in [0, ..., n_bins-1]
    bin_indices = (
        xr.apply_ufunc(
            np.digitize,
            data,
            full_edges,
            input_core_dims=[["ensemble"], ["quantile"]],
            output_core_dims=[["ensemble"]],
            vectorize=True,
            dask="parallelized",
            dask_gufunc_kwargs={"allow_rechunk": True},
            output_dtypes=[int],
        )
        - 1
    )

    # Count occurrences of each bin index (0-4) along 'ensemble'
    quintile_probs = []
    for i in range(5):
        prob = (bin_indices == i).sum(dim="ensemble") / data.sizes["ensemble"]
        quintile_probs.append(prob)

    # Stack along new 'quintile' dimension
    result = xr.concat(quintile_probs, dim="quantile")
    result = result.assign_coords(
        quantile=np.linspace(0.2, 1.0, endpoint=True, num=5)
    ).rename({"quantile": "quintile"})

    return result

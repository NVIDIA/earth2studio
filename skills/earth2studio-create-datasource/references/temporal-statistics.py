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

"""OPTIONAL reference: native temporal statistics in a data source.

This file is a TEMPLATE, not importable code. Load it at Step 7b ONLY when
the user wants the source to compute temporal statistics itself.

Temporal statistics are variable labels such as ``"t2m:mean:24h"`` (window
``[-24h, 0h)``) or ``"tp:sum:+0h:+6h"`` (explicit offsets). Methods: ``mean``,
``sum``, ``min``, ``max``. Windows are left-closed/right-open and must be a
multiple of the source cadence.

NOT every source needs this. A source whose lexicon does not declare native
statistics keeps working: ``fetch_data`` fetches the extra times and
reduces them itself with an explicit ``delta_t``. Native support is worth it when
the source can share fetches (one file per time) or reuse its own cache. Skip it
for sources with irregular cadence (DataFrame/station/satellite sources) or
when the user did not ask.

Contract
--------
1. Document support in the source class docstring using a ``Note`` headed
   ``Temporal statistics:`` (see below). Store capability metadata on its lexicon::

       TIME_STATISTICS: tuple[str, ...] = ("mean", "sum", "min", "max")

   Expose the lexicon as ``source.lexicon``. Declare a subset if only some methods
   make sense; omit the metadata for a lexicon without native support. Pass cadence
   internally to the planner. Neither ``time_step`` nor ``time_statistics`` is a
   source class attribute.
2. ``split_statistic_variables`` separates plain labels from statistic labels.
   It raises ``ValueError`` for malformed labels and methods outside
   ``supported``.
3. A planner builds one ``StatisticJob`` per (variable[, init time]). Each job
   owns a ``StatisticAggregator`` that knows which extra times (analysis) or
   lead times (forecast) are needed. Plain variables are jobs with a ``None``
   modifier, so everything goes through one code path.
4. ``sample_coords(jobs)`` is the union of extra coordinates: validate them
   with the source's own ``_validate_time`` / ``_validate_leadtime`` BEFORE any
   download so bad windows fail fast.
5. ``run_statistic_jobs`` fetches each distinct field once with bounded
   concurrency and folds it into every window that contains it. Memory is one
   field per statistic plus the output, independent of the window length.
6. Set ``xr_array.attrs[E2S_STATISTICS] = statistic_attrs(stat_groups)`` so
   downstream code knows the result is already reduced.
7. With ``verbose=True`` call ``log_statistics`` to tell the user what is
   averaged / accumulated / min'd / max'd and how many samples it needs.

Helpers live in ``earth2studio/data/utils.py``; parsing and offsets in
``earth2studio/utils/time_statistics.py``. Do not write new reducers.

Real examples:
- Analysis, Zarr: earth2studio/data/wb2.py, earth2studio/data/arco.py
- Analysis + forecast, GRIB: earth2studio/data/gfs.py (GFS, GFS_FX)
- Mock fixtures: test/data/conftest.py (mock_statistic_zarr/_grib)
"""

# ---------------------------------------------------------------------------
# Imports (add to the existing earth2studio.data.utils import)
# ---------------------------------------------------------------------------
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr

from earth2studio.data.utils import (
    StatisticJob,
    analysis_jobs,  # FILL: forecast_jobs for a ForecastSource
    as_datetime,  # FILL: as_timedelta for a ForecastSource
    log_statistics,
    run_statistic_jobs,
    sample_coords,
    split_statistic_variables,
    statistic_attrs,
)
from earth2studio.utils.coords import E2S_STATISTICS


class SourceName:
    """FILL: source description and parameters.

    Note
    ----
    Temporal statistics: ``mean``, ``sum``, ``min``, and ``max`` accept labels
    ``variable:method:window`` or ``variable:method:start:end``. Windows are
    left-closed/right-open, sampled every six hours, and relative to the requested
    analysis time. For example, ``t2m:mean:24h`` averages the preceding four
    samples, excluding the requested time. FILL: accumulation caveats.
    """

    # FILL: import the source lexicon and declare TIME_STATISTICS there
    lexicon = SourceNameLexicon  # noqa: F821

    # ----------------------------------------------------------------------
    # DataSource (analysis): extra samples are other valid TIMES
    # ----------------------------------------------------------------------
    async def fetch(self, time, variable):  # type: ignore[no-untyped-def]
        # Existing prep + validation of the requested times
        time_list, variable_list = prep_data_inputs(time, variable)  # noqa: F821
        self._validate_time(time_list)

        # 1. Split labels. Plain entries are (slot, name); groups share a modifier
        plain, stat_groups = split_statistic_variables(
            variable_list,
            supported=self.lexicon.TIME_STATISTICS,  # reject unsupported methods
        )

        # 2. Output keeps the labels exactly as requested ("t2m:mean:24h")
        xr_array = xr.DataArray(
            data=np.full((len(time_list), len(variable_list), 181, 360), np.nan),
            dims=["time", "variable", "lat", "lon"],
            coords={"time": time_list, "variable": variable_list},
        )

        # 3. Plan: jobs write into xr_array.values[:, slot] in place
        # FILL: cadence of the stored data; windows must be multiples of it
        jobs = analysis_jobs(
            stat_groups, time_list, np.timedelta64(6, "h"), xr_array.values, plain
        )

        # 4. Validate every extra time before reading anything
        self._validate_time([as_datetime(t) for t in sample_coords(jobs)])
        log_statistics("SourceName", stat_groups, jobs, self._verbose)

        # 5. Fetch each distinct instantaneous field once, reduce on arrival
        async def fetch_sample(job: StatisticJob, coord: Any) -> np.ndarray:
            # job.base is the lexicon name WITHOUT the statistic suffix
            return await self.fetch_array(as_datetime(coord), job.base)

        await run_statistic_jobs(
            jobs,
            fetch_sample,
            desc="Fetching SourceName data",
            verbose=self._verbose,
            async_workers=self._async_workers,
        )

        # 6. Declare the reduction
        if stat_groups:
            xr_array.attrs[E2S_STATISTICS] = statistic_attrs(stat_groups)
        return xr_array

    # ----------------------------------------------------------------------
    # ForecastSource: extra samples are other LEAD TIMES of the same init time
    # ----------------------------------------------------------------------
    # plain, stat_groups = split_statistic_variables(
    #     variable_list, self.lexicon.TIME_STATISTICS
    # )
    # jobs = forecast_jobs(
    #     stat_groups, time_list, lead_list, np.timedelta64(1, "h"),
    #     xr_array.values,          # dims (time, lead_time, variable, ...)
    #     plain,
    # )
    # self._validate_leadtime([as_timedelta(lt) for lt in sample_coords(jobs)])
    # fetch_sample receives job.init_time (a datetime) and a lead-time coord:
    #     return await self.fetch_array(job.init_time, as_timedelta(coord), job.base)
    #
    # A lead time of 0 with a trailing window (e.g. "mean:3h") reaches before
    # the start of the forecast and fails validation: that is intended.


# ---------------------------------------------------------------------------
# Source-specific caveats to handle (and tell the user about in verbose mode)
# ---------------------------------------------------------------------------
# - Fields that are already accumulations (precip buckets that reset every 6h)
#   do not sum/mean meaningfully; log a warning like GFS does for ``tp``.
# - Composite sources must not reduce: forward labels unchanged to components.
# - Missing samples: reducers skip NaN (sum/mean keep per-cell valid counts),
#   so a missing file degrades the window instead of crashing it.


# ===========================================================================
# MOCK TEST SAMPLE (copy into test/data/test_<source>.py)
# ===========================================================================
# Exactly ONE statistics test per source: test_<source>_time_statistics_mock.
# It must run without network and cover mean, sum, min, max, the attrs, the
# set of fetched samples (each field once), and an invalid window.
#
# Fake the I/O layer, not the statistics code. Make the fake field value a
# function of the sample time so expected results are closed-form. Existing
# fixtures: ``mock_statistic_zarr`` and ``mock_statistic_grib`` in
# test/data/conftest.py (field value == hour index / valid hour). Reuse them
# if the backend matches, otherwise add a similar fixture there.
#
#     @pytest.mark.timeout(30)
#     def test_source_time_statistics_mock(mock_statistic_zarr):
#         ds = SourceName(cache=True, verbose=True)
#         reads = mock_statistic_zarr(ds)       # returns the list of reads
#         time = datetime.datetime(2020, 1, 2, 12)
#         index = ds._get_time_index(time)      # value of the field at `time`
#         variables = [
#             "t2m",                 # plain field passes through unchanged
#             "t2m:mean:24h",        # window [-24h, 0h): 4 six-hourly samples
#             "t2m:sum:24h",
#             "t2m:max:12h",
#             "t2m:min:12h",
#         ]
#         data = ds(time, variables)
#         assert data.shape == (1, 5, 2, 2)
#         values = [float(data.sel(variable=v).values[0, 0, 0]) for v in variables]
#         assert values == [index, index - 2.5, 4 * index - 10, index - 1, index - 2]
#         # Result declares what was reduced
#         assert set(data.attrs["earth2studio_statistics"]) == set(variables[1:])
#         # Every distinct sample is read exactly once, including the plain one
#         assert sorted(reads) == list(range(index - 4, index + 1))
#         # Window must be a multiple of the cadence
#         with pytest.raises(ValueError, match="divisible"):
#             ds(time, ["t2m:mean:9h"])
#         # Forward-looking window uses the explicit-offset form
#         reads.clear()
#         data = ds(time, ["t2m:mean:+0h:+12h"])
#         assert sorted(reads) == [index, index + 1]
#
# Forecast sources additionally assert lead-time behavior: pass two lead times
# whose windows overlap, check shifted results, that each overlapping lead is
# fetched once (``len(set(calls)) == len(calls)``), and that a lead time
# reaching before 0 raises ``ValueError``.

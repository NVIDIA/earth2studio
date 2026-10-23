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
2. ``compile_statistic_requests`` parses labels, expands windows and returns
   ``requests, attrs, finalize``: deduplicated requests paired with processing
   callbacks, metadata, and a finalization function. It rejects
   malformed/duplicate labels and methods outside ``supported``.
3. Each request carries ``time``, ``lead_time``, ``variable`` and ``modifier``.
   Validate those coordinates with the source's validators before downloading.
4. Fetch the base variable and apply its ordinary lexicon modifier, then call
   ``request.modifier(sample)`` once. This callback copies or aggregates the field
   into all consuming outputs, without retaining it. ``run_data_requests`` is a
   bounded async loop that does this; it contains no statistic planning logic.
5. Call ``finalize()`` once after fetching to check completeness and finish means.
   Field memory is output views, per-cell mean counts, and concurrent samples;
   coordinate plans still grow with the number of requested samples. Backend
   adapters may fetch additional components for derived variables.
6. Set ``xr_array.attrs.update(attrs)`` so
   downstream code knows the result is already reduced.
7. Pass ``verbose=True`` to the compiler to tell the user what is
   averaged / accumulated / min'd / max'd and how many samples it needs.

Helpers live in ``earth2studio/data/utils.py``; parsing and offsets in
``earth2studio/utils/time_statistics.py``. Do not write new reducers.
Contracts: ``dev/spec/TIME_STATISTICS_SPEC.md`` and ``dev/spec/FETCH_DATA_SPEC.md``.
Call path: compile_statistic_requests -> source validation -> run_data_requests
-> fetch_sample -> request.modifier -> finalize() -> output attrs.
Custom block reductions registered in time_statistics are not automatically
supported by native streaming sources; use fetch_data with explicit delta_t.

Real examples:
- Analysis, Zarr: earth2studio/data/wb2.py, earth2studio/data/arco.py
- Analysis + forecast, GRIB: earth2studio/data/gfs.py (GFS, GFS_FX)
- Mock fixtures: test/data/conftest.py (mock_statistic_zarr/_grib)
"""

# ---------------------------------------------------------------------------
# Imports (add to the existing earth2studio.data.utils import)
# ---------------------------------------------------------------------------
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

from earth2studio.data.utils import (
    DataRequest,
    compile_statistic_requests,
    run_data_requests,
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

        # Output keeps the labels exactly as requested ("t2m:mean:24h")
        xr_array = xr.DataArray(
            data=np.full((len(time_list), len(variable_list), 181, 360), np.nan),
            dims=["time", "variable", "lat", "lon"],
            coords={"time": time_list, "variable": variable_list},
        )

        # Compile requests with callbacks writing into output views
        # FILL: cadence of the stored data; windows must be multiples of it
        requests, attrs, finalize = compile_statistic_requests(
            variable_list,
            time_list,
            np.timedelta64(6, "h"),
            xr_array.values,
            supported=self.lexicon.TIME_STATISTICS,
            source="SourceName",
            verbose=self._verbose,
        )

        # Validate every extra time before reading anything
        self._validate_time(list({r.time for r in requests}))

        # Fetch each base field using the source's normal lexicon conversion
        async def fetch_sample(request: DataRequest) -> np.ndarray:
            return await self.fetch_array(request.time, request.variable)

        # Equivalent to: request.modifier(await fetch_sample(request)) per request
        await run_data_requests(
            requests,
            fetch_sample,
            desc="Fetching SourceName data",
            verbose=self._verbose,
            async_workers=self._async_workers,
        )

        finalize()
        xr_array.attrs.update(attrs)
        return xr_array

    # ----------------------------------------------------------------------
    # ForecastSource: extra samples are other LEAD TIMES of the same init time
    # ----------------------------------------------------------------------
    # requests, attrs, finalize = compile_statistic_requests(
    #     variable_list, time_list, np.timedelta64(1, "h"),
    #     xr_array.values,          # dims (time, lead_time, variable, ...)
    #     lead_time=lead_list,
    #     supported=self.lexicon.TIME_STATISTICS,
    #     source="SourceName", verbose=self._verbose,
    # )
    # self._validate_leadtime(list({r.lead_time for r in requests}))
    # Forecast fetch_sample uses the already resolved time and lead:
    #     return await self.fetch_array(request.time, request.lead_time, request.variable)
    #
    # A lead time of 0 with a trailing window (e.g. "mean:3h") reaches before
    # the start of the forecast and fails validation: that is intended.


# ---------------------------------------------------------------------------
# Source-specific caveats to handle (and tell the user about in verbose mode)
# ---------------------------------------------------------------------------
# - Reductions sum/average stored values, not rates integrated over time. Already
#   accumulated fields may overlap or reset; document their periods and warn like
#   GFS does for ``tp``. Totals are only rainfall totals for disjoint intervals.
# - Composite sources must not reduce: forward labels unchanged to components.
# - NaNs are skipped; only mean keeps per-cell valid counts. All-NaN windows
#   yield NaN for mean/min/max and zero for sum. Download errors still propagate;
#   a source may explicitly return NaN fields for missing records. finalize()
#   rejects incomplete sample counts.


# ===========================================================================
# MOCK TEST SAMPLE (copy into test/data/test_<source>.py)
# ===========================================================================
# Exactly ONE statistics test per source: extend test_<source>_call_mock when
# present, otherwise add test_<source>_time_statistics_mock.
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
#         time = datetime(2020, 1, 2, 12)
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
# reaching before 0 raises ``ValueError``. Use multiple initializations to check
# that equal valid times from distinct forecasts are not merged. Also check NaN
# handling and that invalid windows cause no backend reads.

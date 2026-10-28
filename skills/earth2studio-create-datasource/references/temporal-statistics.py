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
``sum``, ``min``, ``max``. Windows are left-closed/right-open. Regular coordinate
helpers require cadence-aligned windows; reducers impose no sampling cadence.

NOT every source needs this. A source whose lexicon does not declare native
statistics keeps working: ``fetch_data`` fetches the extra times and
reduces them itself with an explicit ``delta_t``. Native support is worth it when
the source can share fetches (one file per time) or reuse its own cache. Irregular
sources select available samples using parsed window offsets. Means are sample
weighted, not duration weighted. Skip native support when the user did not ask.

Contract
--------
1. Document support in the source class docstring using a ``Note`` headed
   ``Temporal statistics:`` (see below). Store capability metadata on its lexicon::

       TIME_STATISTICS: tuple[str, ...] = ("mean", "sum", "min", "max")

   Expose the lexicon as ``source.lexicon``. Declare a subset if only some methods
    make sense; omit the metadata for a lexicon without native support.
    Neither ``time_step`` nor ``time_statistics`` is a
   source class attribute.
2. ``plan_statistics`` parses labels and returns ``plain, stats, attrs``. ``plain``
   is ``[(slot, name)]`` (unqualified labels and raw ``::`` identifiers) for the
   source's ordinary path; ``stats`` is ``[(slot, base, modifier, samples)]`` with
   ``samples`` shaped ``(target, sample)``. It rejects malformed/duplicate labels
   and methods outside ``supported``.
3. ``validate`` receives every unique sampled coordinate before downloading. Irregular sources build
   their own sample arrays from ``time_statistic_metadata`` offsets.
4. ``await reduce_statistics(out, stats, fetch)``: ``fetch(coord, base)`` is a
   source coroutine returning one field with its ordinary lexicon conversion.
   Each target's samples are fetched by a bounded pool into one transient array, reduced
   through the registered statistic (``reduce_samples``; any registered statistic
   works if listed in ``TIME_STATISTICS``), written to ``out`` and released.
   Overlapping windows re-read shared samples (disk caches make this cheap).
5. Propagate fetch failures (the shared helper cancels outstanding fetches).
   Memory is the output plus a few transient window stacks.
6. Set ``xr_array.attrs.update(attrs)`` so downstream code knows the result is
   already reduced.
7. Pass ``verbose=True`` to ``plan_statistics`` to tell the user what is
   averaged / accumulated / min'd / max'd and the window.

Helpers live in ``earth2studio/data/utils.py``; parsing and offsets in
``earth2studio/utils/time_statistics.py``. Do not write new reducers; register
them with ``register_time_statistic``.
Contracts: ``dev/spec/TIME_STATISTICS_SPEC.md`` and ``dev/spec/FETCH_DATA_SPEC.md``.
Call path: plan_statistics -> validation -> reduce_statistics -> fetch(coord, base)
-> reduce_samples -> out[target, slot] -> output attrs.

Real examples:
- Analysis, Zarr: earth2studio/data/wb2.py, earth2studio/data/arco.py
- Analysis + forecast, GRIB: earth2studio/data/gfs.py (GFS, GFS_FX)
- Mock fixtures: test/data/conftest.py (mock_statistic_zarr/_grib)
"""

# ---------------------------------------------------------------------------
# Imports (add to the existing earth2studio.data.utils import)
# ---------------------------------------------------------------------------
import numpy as np
import xarray as xr

from earth2studio.data.utils import as_datetime, plan_statistics, reduce_statistics


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

        # This example source stores six-hourly fields. Irregular sources instead
        # build their sample arrays inside the parsed offsets.
        times = np.asarray(time_list, dtype="datetime64[ns]")
        plain, stats, attrs = plan_statistics(
            variable_list,
            times,
            np.timedelta64(6, "h"),
            supported=self.lexicon.TIME_STATISTICS,
            validate=self._validate_time,  # every sample, before any download
            source="SourceName",
            verbose=self._verbose,
        )

        # Plain variables: the source's existing request path, unchanged
        for slot, name in plain:
            for i, t in enumerate(time_list):
                xr_array.values[i, slot] = await self.fetch_array(t, name)

        # Statistics: ``fetch_array`` returns one lexicon-converted field
        await reduce_statistics(
            xr_array.values,
            stats,
            lambda t, base: self.fetch_array(as_datetime(t), base),
            async_workers=self._async_workers,
            desc="Fetching SourceName statistics",
            verbose=self._verbose,
        )
        xr_array.attrs.update(attrs)
        return xr_array

    # ----------------------------------------------------------------------
    # ForecastSource: extra samples are other LEAD TIMES of the same init time
    # ----------------------------------------------------------------------
    # plain, stats, attrs = plan_statistics(
    #     variable_list, lead_times, np.timedelta64(1, "h"),  # timedelta64 targets
    #     supported=self.lexicon.TIME_STATISTICS,
    #     validate=self._validate_leadtime,  # unique sampled leads, as timedeltas
    #     source="SourceName",
    # )
    # Then for each initialization time t:
    #   await reduce_statistics(xr_array.values[i], stats,
    #       lambda lt, base, t=t: self.fetch_array(t, as_timedelta(lt), base))
    # Equal valid times from different initializations are never mixed.
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
# - NaNs are skipped. All-NaN windows yield NaN for mean/min/max and zero for
#   sum. Download errors still propagate; a source may explicitly return NaN
#   fields for missing records. The planned samples are all fetched or the
#   request fails; no partial window is written.


# ===========================================================================
# MOCK TEST SAMPLE (copy into test/data/test_<source>.py)
# ===========================================================================
# Exactly ONE statistics test per source: extend test_<source>_call_mock when
# present, otherwise add test_<source>_time_statistics_mock.
# It must run without network and cover mean, sum, min, max, the attrs, the
# set of fetched samples, and an invalid window.
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
#         # Every planned sample is read, including the plain one
#         assert set(reads) == set(range(index - 4, index + 1))
#         # Window must be a multiple of the cadence
#         with pytest.raises(ValueError, match="divisible"):
#             ds(time, ["t2m:mean:9h"])
#         # Forward-looking window uses the explicit-offset form
#         reads.clear()
#         data = ds(time, ["t2m:mean:+0h:+12h"])
#         assert set(reads) == {index, index + 1}
#
# Forecast sources additionally assert lead-time behavior: pass two lead times
# whose windows overlap, check shifted results, and that a lead time
# reaching before 0 raises ``ValueError``. Use multiple initializations to check
# that equal valid times from distinct forecasts are not merged. Also check NaN
# handling and that invalid windows cause no backend reads.

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

"""Tests for TimeWindow data wrapper."""

from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from earth2studio.data.time_window import TimeWindow


class MockDataSource:
    """Mock datasource for testing TimeWindow."""

    def __init__(self, available_times=None, available_vars=None, fail_times=None):
        """Initialize mock datasource.

        Parameters
        ----------
        available_times : list[datetime], optional
            Times that are available. If None, all times are available.
        available_vars : list[str], optional
            Variables that are available. If None, all variables are available.
        fail_times : list[datetime], optional
            Times that should raise ValueError when requested.
        """
        self.available_times = available_times
        self.available_vars = available_vars
        self.fail_times = fail_times or []
        self.call_history = []

    def __call__(self, time, variable):
        """Mock data fetch."""
        # Record the call
        self.call_history.append({"time": time, "variable": variable})

        # Handle single time/variable
        if isinstance(time, datetime):
            time = [time]
        if isinstance(variable, str):
            variable = [variable]

        # Check for failures
        for t in time:
            if t in self.fail_times:
                raise ValueError(f"Time {t} not available")

        # Check availability
        if self.available_times is not None:
            for t in time:
                if t not in self.available_times:
                    raise ValueError(f"Time {t} not in available times")

        if self.available_vars is not None:
            for v in variable:
                if v not in self.available_vars:
                    raise KeyError(f"Variable {v} not available")

        # Create mock data
        data = np.random.randn(len(time), len(variable), 10, 20)
        return xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": time,
                "variable": variable,
                "lat": np.linspace(-90, 90, 10),
                "lon": np.linspace(0, 360, 20),
            },
        )


class TestTimeWindowInitialization:
    """Test TimeWindow initialization and validation."""

    def test_init_valid(self):
        """Test successful initialization with valid parameters."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)],
            suffixes=["_tm1", "_t", "_tp1"],
            group_by="variable",
        )
        assert tw.datasource is ds
        assert len(tw.offsets) == 3
        assert len(tw.suffixes) == 3
        assert tw.group_by == "variable"

    def test_init_empty_offsets(self):
        """Test that empty offsets raises ValueError."""
        ds = MockDataSource()
        with pytest.raises(ValueError, match="offsets must be a non-empty list"):
            TimeWindow(datasource=ds, offsets=[], suffixes=["_t"])

    def test_init_empty_suffixes(self):
        """Test that empty suffixes raises ValueError."""
        ds = MockDataSource()
        with pytest.raises(ValueError, match="suffixes must be a non-empty list"):
            TimeWindow(datasource=ds, offsets=[timedelta(0)], suffixes=[])

    def test_init_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        ds = MockDataSource()
        with pytest.raises(ValueError, match="must have the same length"):
            TimeWindow(
                datasource=ds,
                offsets=[timedelta(0), timedelta(hours=6)],
                suffixes=["_t"],
            )

    def test_init_invalid_group_by(self):
        """Test that invalid group_by raises ValueError."""
        ds = MockDataSource()
        with pytest.raises(ValueError, match="must be 'variable' or 'offset'"):
            TimeWindow(
                datasource=ds,
                offsets=[timedelta(0)],
                suffixes=["_t"],
                group_by="invalid",
            )


class TestTimeWindowVariableOrdering:
    """Test variable ordering in TimeWindow output."""

    def test_ordering_group_by_variable(self):
        """Test variable ordering when group_by='variable'."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)],
            suffixes=["_tm1", "_t", "_tp1"],
            group_by="variable",
        )

        result = tw(datetime(2024, 1, 1), ["t2m", "u10m"])

        # Expected order: all offsets for t2m, then all offsets for u10m
        expected_vars = [
            "t2m_tm1",
            "t2m_t",
            "t2m_tp1",
            "u10m_tm1",
            "u10m_t",
            "u10m_tp1",
        ]
        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == expected_vars

    def test_ordering_group_by_offset(self):
        """Test variable ordering when group_by='offset'."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)],
            suffixes=["_tm1", "_t", "_tp1"],
            group_by="offset",
        )

        result = tw(datetime(2024, 1, 1), ["t2m", "u10m"])

        # Expected order: all variables for tm1, then all for t, then all for tp1
        expected_vars = [
            "t2m_tm1",
            "u10m_tm1",
            "t2m_t",
            "u10m_t",
            "t2m_tp1",
            "u10m_tp1",
        ]
        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == expected_vars

    def test_ordering_preserves_request_order(self):
        """Test that variable order in request is preserved in output."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(0), timedelta(hours=6)],
            suffixes=["_t", "_tp1"],
            group_by="variable",
        )

        # Request in specific order
        result = tw(datetime(2024, 1, 1), ["z500", "t2m", "u10m"])

        expected_vars = ["z500_t", "z500_tp1", "t2m_t", "t2m_tp1", "u10m_t", "u10m_tp1"]
        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == expected_vars

    def test_ordering_with_presuffixed_variables(self):
        """Test ordering when variables already have suffixes."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)],
            suffixes=["_tm1", "_t", "_tp1"],
            group_by="variable",
        )

        # Request specific suffixed variables
        result = tw(datetime(2024, 1, 1), ["t2m_t", "u10m_tm1", "u10m_tp1"])

        # Should preserve the requested order
        expected_vars = ["t2m_t", "u10m_tm1", "u10m_tp1"]
        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == expected_vars

    def test_ordering_mixed_suffixed_unsuffixed(self):
        """Test ordering with mix of suffixed and unsuffixed variables."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0)],
            suffixes=["_tm1", "_t"],
            group_by="variable",
        )

        # Mix of suffixed and unsuffixed
        result = tw(datetime(2024, 1, 1), ["t2m_t", "u10m"])

        # Suffixed variables come first (in request order), then expanded unsuffixed
        expected_vars = ["t2m_t", "u10m_tm1", "u10m_t"]
        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == expected_vars


class TestTimeWindowSuffixHandling:
    """Test suffix handling and variable name parsing."""

    def test_suffix_stripping(self):
        """Test that suffixes are correctly stripped from variable names."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0)],
            suffixes=["_tm1", "_t"],
        )

        tw(datetime(2024, 1, 1), ["t2m_tm1", "t2m_t"])

        # Should have called datasource with base variable "t2m" twice
        assert len(ds.call_history) == 2
        # First call for tm1 offset
        assert ds.call_history[0]["variable"] == ["t2m"]
        # Second call for t offset
        assert ds.call_history[1]["variable"] == ["t2m"]

    @pytest.mark.skip(reason="Empty suffix handling needs implementation")
    def test_empty_suffix(self):
        """Test handling of empty suffix string."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0)],
            suffixes=["_tm1", ""],  # Empty suffix for current time
        )

        result = tw(datetime(2024, 1, 1), ["t2m"])

        # Should create variables with and without suffix
        expected_vars = ["t2m_tm1", "t2m"]
        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == expected_vars

    def test_suffix_not_matching(self):
        """Test variables that don't match any suffix are expanded to all offsets."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0)],
            suffixes=["_tm1", "_t"],
        )

        # Request variable without matching suffix
        result = tw(datetime(2024, 1, 1), ["temperature"])

        # Should be expanded to all offsets
        expected_vars = ["temperature_tm1", "temperature_t"]
        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == expected_vars


class TestTimeWindowOffsetCalculation:
    """Test time offset calculations."""

    def test_offset_times_calculated_correctly(self):
        """Test that offset times are correctly calculated."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)],
            suffixes=["_tm1", "_t", "_tp1"],
        )

        base_time = datetime(2024, 1, 1, 12, 0)
        tw(base_time, ["t2m"])

        # Check that datasource was called with correct offset times
        assert len(ds.call_history) == 3
        assert ds.call_history[0]["time"] == [datetime(2024, 1, 1, 6, 0)]  # -6h
        assert ds.call_history[1]["time"] == [datetime(2024, 1, 1, 12, 0)]  # 0h
        assert ds.call_history[2]["time"] == [datetime(2024, 1, 1, 18, 0)]  # +6h

    def test_offset_with_multiple_times(self):
        """Test offset calculation with multiple requested times."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(days=-1), timedelta(days=0)],
            suffixes=["_prev", "_curr"],
        )

        times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        tw(times, ["t2m"])

        # Check offset times for both base times
        assert ds.call_history[0]["time"] == [
            datetime(2023, 12, 31),
            datetime(2024, 1, 1),
        ]  # -1 day
        assert ds.call_history[1]["time"] == [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ]  # 0 days

    def test_time_coordinate_alignment(self):
        """Test that output time coordinates match input (not offset times)."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=6)],
            suffixes=["_tm1", "_tp1"],
        )

        base_time = datetime(2024, 1, 1, 12, 0)
        result = tw(base_time, ["t2m"])

        # Output time should match input time, not offset times
        assert len(result.time) == 1
        assert result.time.values[0] == np.datetime64(base_time)


class TestTimeWindowErrorHandling:
    """Test error handling in TimeWindow."""

    def test_missing_time_raises_error(self):
        """Test that missing time data raises ValueError with context."""
        fail_time = datetime(2024, 1, 1, 6, 0)
        ds = MockDataSource(fail_times=[fail_time])
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0)],
            suffixes=["_tm1", "_t"],
        )

        with pytest.raises(ValueError, match="Failed to fetch data for offset"):
            tw(datetime(2024, 1, 1, 12, 0), ["t2m"])

    def test_missing_variable_raises_error(self):
        """Test that missing variable raises KeyError with context."""
        ds = MockDataSource(available_vars=["t2m"])
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=0)],
            suffixes=["_t"],
        )

        with pytest.raises(KeyError, match="Failed to fetch data for offset"):
            tw(datetime(2024, 1, 1), ["nonexistent"])

    def test_partial_window_missing(self):
        """Test behavior when only part of time window is missing."""
        # Only t-6h is missing, t and t+6h are available
        available_times = [
            datetime(2024, 1, 1, 12, 0),  # t
            datetime(2024, 1, 1, 18, 0),  # t+6h
        ]
        ds = MockDataSource(available_times=available_times)
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)],
            suffixes=["_tm1", "_t", "_tp1"],
        )

        # Should raise error for the missing offset
        with pytest.raises(ValueError, match="Time .* not in available times"):
            tw(datetime(2024, 1, 1, 12, 0), ["t2m"])

    def test_no_variables_fetched_raises_error(self):
        """Test that fetching no variables raises ValueError."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=0)],
            suffixes=["_t"],
        )

        # This should raise because no variables match
        with pytest.raises(ValueError, match="No variables fetched"):
            tw(datetime(2024, 1, 1), [])


class TestTimeWindowDataIntegrity:
    """Test data integrity and shape preservation."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0), timedelta(hours=6)],
            suffixes=["_tm1", "_t", "_tp1"],
        )

        result = tw(datetime(2024, 1, 1), ["t2m", "u10m"])

        # Should have 6 variables (2 vars * 3 offsets)
        assert result.shape[0] == 1  # time
        assert result.shape[1] == 6  # variables
        assert result.shape[2] == 10  # lat
        assert result.shape[3] == 20  # lon

    def test_multiple_times_shape(self):
        """Test output shape with multiple times."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=0), timedelta(hours=6)],
            suffixes=["_t", "_tp1"],
        )

        times = [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)]
        result = tw(times, ["t2m"])

        assert result.shape[0] == 3  # times
        assert result.shape[1] == 2  # variables (1 var * 2 offsets)

    def test_dimensions_preserved(self):
        """Test that spatial dimensions are preserved."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=0)],
            suffixes=["_t"],
        )

        result = tw(datetime(2024, 1, 1), ["t2m"])

        # Check dimensions are correct
        assert "time" in result.dims
        assert "variable" in result.dims
        assert "lat" in result.dims
        assert "lon" in result.dims

    def test_coordinates_preserved(self):
        """Test that coordinates are preserved correctly."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=0)],
            suffixes=["_t"],
        )

        result = tw(datetime(2024, 1, 1), ["t2m"])

        # Check coordinates exist and have correct length
        assert len(result.coords["lat"]) == 10
        assert len(result.coords["lon"]) == 20
        assert len(result.coords["time"]) == 1
        assert len(result.coords["variable"]) == 1


class TestTimeWindowEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_offset(self):
        """Test with single offset (no actual windowing)."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=0)],
            suffixes=["_t"],
        )

        result = tw(datetime(2024, 1, 1), ["t2m"])

        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == ["t2m_t"]

    def test_many_offsets(self):
        """Test with many offsets."""
        ds = MockDataSource()
        offsets = [timedelta(hours=i) for i in range(-12, 13, 3)]  # 9 offsets
        suffixes = [f"_t{i:+d}" for i in range(-12, 13, 3)]
        tw = TimeWindow(datasource=ds, offsets=offsets, suffixes=suffixes)

        result = tw(datetime(2024, 1, 1), ["t2m"])

        assert result.shape[1] == 9  # 1 var * 9 offsets

    def test_single_variable(self):
        """Test with single variable."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=-6), timedelta(hours=0)],
            suffixes=["_tm1", "_t"],
        )

        result = tw(datetime(2024, 1, 1), "t2m")  # Single string

        assert result.shape[1] == 2  # 1 var * 2 offsets

    def test_duplicate_variables_in_request(self):
        """Test behavior with duplicate variables in request."""
        ds = MockDataSource()
        tw = TimeWindow(
            datasource=ds,
            offsets=[timedelta(hours=0)],
            suffixes=["_t"],
        )

        # Request same variable twice
        result = tw(datetime(2024, 1, 1), ["t2m", "t2m"])

        # TimeWindow does NOT deduplicate - requesting twice gives two instances
        assert result.shape[1] == 2
        actual_vars = [str(v) for v in result.coords["variable"].values]
        assert actual_vars == ["t2m_t", "t2m_t"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

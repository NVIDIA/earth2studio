"""
Unit tests for data/earth2studio/nclimgrid.py
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from earth2studio.data.nclimgrid import NClimGrid
from earth2studio.lexicon.nclimgrid import NClimGridLexicon

# ---------------------------------------------------------------------
# GLOBAL DATASET FIXTURE  (VERY IMPORTANT)
# ---------------------------------------------------------------------


@pytest.fixture(scope="session")
def ds():
    """
    Open dataset ONCE for entire test session.

    Prevents:
    - repeated Zarr open
    - repeated S3 metadata scan
    - test hangs
    - extreme runtime
    """
    return NClimGrid(cache=True, verbose=False)


# ---------------------------------------------------------------------
# OFFLINE TESTS
# ---------------------------------------------------------------------


class TestNClimGridOffline:

    def test_schema_fields(self):
        assert NClimGrid.SCHEMA.names == [
            "time",
            "lat",
            "lon",
            "observation",
            "variable",
        ]

    def test_schema_types(self):
        assert NClimGrid.SCHEMA.field("time").type == pa.timestamp("ns")
        assert NClimGrid.SCHEMA.field("lat").type == pa.float32()
        assert NClimGrid.SCHEMA.field("lon").type == pa.float32()
        assert NClimGrid.SCHEMA.field("observation").type == pa.float32()
        assert NClimGrid.SCHEMA.field("variable").type == pa.string()

    def test_resolve_fields_invalid(self):
        with pytest.raises(KeyError):
            NClimGrid.resolve_fields(["not_real"])

    def test_source_id(self):
        assert NClimGrid.SOURCE_ID == "earth2studio.data.nclimgrid"

    def test_lexicon_variables(self):
        for v in ["t2m_max", "t2m_min", "tp", "spi"]:
            desc, mod = NClimGridLexicon[v]
            assert isinstance(desc, str)
            assert callable(mod)

    def test_unit_conversion_kelvin(self):
        _, mod = NClimGridLexicon["t2m_max"]
        np.testing.assert_allclose(mod(np.array([25.0])), [298.15])

    def test_unit_conversion_precip(self):
        _, mod = NClimGridLexicon["tp"]
        np.testing.assert_allclose(mod(np.array([100.0])), [0.1])

    def test_spi_identity(self):
        _, mod = NClimGridLexicon["spi"]
        arr = np.array([1.2])
        np.testing.assert_allclose(mod(arr), arr)


# ---------------------------------------------------------------------
# ONLINE TESTS
# ---------------------------------------------------------------------


@pytest.mark.network
class TestNClimGridOnline:

    DATE = datetime(2010, 7, 1)

    # ---------------- functional ----------------

    def test_single_variable(self, ds):
        df = ds(self.DATE, "t2m_max")
        assert len(df) > 1000
        assert set(df["variable"]) == {"t2m_max"}

    def test_multi_variable(self, ds):
        df = ds(self.DATE, ["t2m_max", "tp"])
        assert set(df["variable"]) == {"t2m_max", "tp"}

    def test_multiple_dates(self, ds):
        dates = [self.DATE, self.DATE + timedelta(days=1)]
        df = ds(dates, "t2m_max")
        assert df["time"].nunique() == 2

    def test_slice_semantics(self, ds):
        df = ds(slice(datetime(2010, 7, 1), datetime(2010, 7, 3)), "t2m_max")
        assert df["time"].nunique() == 3

    # ---------------- grid integrity ----------------

    def test_lat_lon_bounds(self, ds):
        df = ds(self.DATE, "t2m_max")
        assert df["lat"].between(20, 55).all()
        assert df["lon"].between(-130, -60).all()

    def test_unique_grid_density(self, ds):
        df = ds(self.DATE, "t2m_max")
        assert df["lat"].nunique() > 100
        assert df["lon"].nunique() > 100

    def test_no_nan_coordinates(self, ds):
        df = ds(self.DATE, "t2m_max")
        assert df["lat"].notna().all()
        assert df["lon"].notna().all()

    # ---------------- scientific sanity ----------------

    def test_temperature_mean_range(self, ds):
        df = ds(self.DATE, "t2m_max")
        assert 250 < df["observation"].mean() < 320

    def test_temperature_extreme_range(self, ds):
        df = ds(self.DATE, "t2m_max")
        assert df["observation"].min() > 200
        assert df["observation"].max() < 350

    def test_precip_nonnegative(self, ds):
        df = ds(self.DATE, "tp")
        valid = df["observation"].dropna()
        assert len(valid) > 0
        assert (valid >= 0).all()

    # ---------------- scaling ----------------

    def test_multi_variable_multi_time_scaling(self, ds):
        dates = [self.DATE + timedelta(days=i) for i in range(5)]
        df = ds(dates, ["t2m_max", "tp"])
        assert df["time"].nunique() == 5
        assert set(df["variable"]) == {"t2m_max", "tp"}

    def test_large_time_window(self, ds):
        df = ds(slice(datetime(2010, 7, 1), datetime(2010, 7, 10)), "t2m_max")
        assert df["time"].nunique() == 10

    def test_duplicate_time_input(self, ds):
        df = ds([self.DATE, self.DATE], "t2m_max")
        assert df["time"].nunique() == 1

    def test_time_order_invariance(self, ds):
        d1 = [datetime(2010, 7, 1), datetime(2010, 7, 2)]
        d2 = list(reversed(d1))
        assert set(ds(d1, "t2m_max")["time"]) == set(ds(d2, "t2m_max")["time"])

    # ---------------- caching ----------------

    def test_cache_speedup(self, ds):
        t0 = time.time()
        ds(self.DATE, "t2m_max")
        first = time.time() - t0

        t0 = time.time()
        ds(self.DATE, "t2m_max")
        second = time.time() - t0

        assert second <= first

    # ---------------- dataframe integrity ----------------

    def test_output_types(self, ds):
        df = ds(self.DATE, "t2m_max")
        assert pd.api.types.is_datetime64_any_dtype(df["time"])
        assert pd.api.types.is_float_dtype(df["lat"])
        assert pd.api.types.is_float_dtype(df["lon"])
        assert pd.api.types.is_float_dtype(df["observation"])
        assert pd.api.types.is_string_dtype(df["variable"])

    def test_source_attr(self, ds):
        df = ds(self.DATE, "t2m_max")
        assert df.attrs["source"] == NClimGrid.SOURCE_ID

    def test_invalid_variable(self, ds):
        with pytest.raises(KeyError):
            ds(self.DATE, "not_real_variable")

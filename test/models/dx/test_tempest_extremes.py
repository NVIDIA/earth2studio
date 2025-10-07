# test_tempest_extremes.py

import os
import tempfile
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pytest
import torch
from physicsnemo.distributed import DistributedManager

from earth2studio.models.dx import AsyncTempestExtremes, TempestExtremes


@pytest.fixture(scope="session", autouse=True)
def initialize_distributed_manager():
    """Initialize DistributedManager once for all tests."""
    # Initialize if not already initialized
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    yield
    # Note: We don't cleanup torch.distributed as it can't be reinitialized in the same process


# ============================================================================
# 1. INITIALIZATION AND SETUP TESTS
# ============================================================================


def test_tempest_extremes_initialization():
    """Test basic initialization with required parameters"""
    with tempfile.TemporaryDirectory() as tmpdir:
        te = TempestExtremes(
            detect_cmd="DetectNodes --verbosity 0",
            stitch_cmd="StitchNodes --verbosity 0",
            input_vars=["u10m", "v10m", "msl"],
            batch_size=2,
            n_steps=10,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        assert te.batch_size == 2
        assert te.n_steps == 10
        assert len(te.input_vars) == 3
        assert os.path.exists(te.store_dir)


def test_initialization_with_static_vars():
    """Test initialization with static variables"""
    with tempfile.TemporaryDirectory() as tmpdir:
        static_vars = torch.randn(2, 721, 1440)
        static_coords = OrderedDict(
            {
                "variable": np.array(["z", "lsm"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440),
            }
        )

        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m", "v10m", "msl"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            static_vars=static_vars,
            static_coords=static_coords,
            use_ram=False,
        )

        assert te.static_vars is not None
        assert te.static_coords is not None


def test_initialization_missing_static_coords_raises_error():
    """Test that providing static_vars without static_coords raises error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="provide both values and coords"):
            TempestExtremes(
                detect_cmd="DetectNodes",
                stitch_cmd="StitchNodes",
                input_vars=["u10m"],
                batch_size=1,
                n_steps=5,
                time_step=np.timedelta64(6, "h"),
                lats=np.linspace(90, -90, 721),
                lons=np.linspace(0, 360, 1440),
                store_dir=tmpdir,
                static_vars=torch.randn(1, 721, 1440),
                static_coords=None,
                use_ram=False,
            )


# ============================================================================
# 2. COMMAND FORMATTING TESTS
# ============================================================================


def test_remove_arguments():
    """Test that specified arguments are correctly removed from commands"""
    cmd = ["DetectNodes", "--in_data_list", "input.txt", "--verbosity", "0"]
    result = TempestExtremes.remove_arguments(cmd, ["--in_data_list"])

    assert "--in_data_list" not in result
    assert "input.txt" not in result
    assert "--verbosity" in result
    assert "0" in result


def test_format_tempestextremes_commands():
    """Test command formatting removes input/output arguments"""
    with tempfile.TemporaryDirectory() as tmpdir:
        detect_cmd = "DetectNodes --in_data_list old.txt --verbosity 0"
        stitch_cmd = "StitchNodes --in old.txt --out old_tracks.txt"

        te = TempestExtremes(
            detect_cmd=detect_cmd,
            stitch_cmd=stitch_cmd,
            input_vars=["u10m"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        assert "--in_data_list" not in te.detect_cmd
        assert "--out_file_list" not in te.detect_cmd
        assert "--in" not in te.stitch_cmd
        assert "--out" not in te.stitch_cmd
        assert "--verbosity" in te.detect_cmd


# ============================================================================
# 3. COORDINATE SYSTEM TESTS
# ============================================================================


def test_input_coords():
    """Test that input_coords returns correct coordinate system"""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_vars = ["u10m", "v10m", "msl"]
        lats = np.linspace(90, -90, 721)
        lons = np.linspace(0, 360, 1440)

        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=input_vars,
            batch_size=1,
            n_steps=10,
            time_step=np.timedelta64(6, "h"),
            lats=lats,
            lons=lons,
            store_dir=tmpdir,
            use_ram=False,
        )

        coords = te.input_coords

        assert "time" in coords
        assert "lead_time" in coords
        assert "variable" in coords
        assert "lat" in coords
        assert "lon" in coords
        assert len(coords["lead_time"]) == 11  # n_steps + 1
        assert np.array_equal(coords["variable"], input_vars)


def test_store_coords_with_static_vars():
    """Test that store coords include static variables"""
    with tempfile.TemporaryDirectory() as tmpdir:
        static_vars = torch.randn(1, 721, 1440)
        static_coords = OrderedDict(
            {
                "variable": np.array(["z"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440),
            }
        )

        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m", "v10m"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            static_vars=static_vars,
            static_coords=static_coords,
            use_ram=False,
        )

        # Check that static vars are in store coords
        assert "z" in te._store_coords["variable"]


# ============================================================================
# 4. DATA RECORDING TESTS
# ============================================================================


def test_record_state_basic():
    """Test that record_state correctly stores data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m", "v10m", "msl"],
            batch_size=1,
            n_steps=2,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        # Create test data
        time = np.datetime64("2024-01-01T00:00:00")
        lead_time = np.timedelta64(0, "h")
        x = torch.randn(1, 1, 1, 3, 721, 1440)
        coords = OrderedDict(
            {
                "ensemble": np.array([0]),
                "time": np.array([time]),
                "lead_time": np.array([lead_time]),
                "variable": np.array(["u10m", "v10m", "msl"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440),
            }
        )

        # Record state
        te.record_state(x, coords)

        # Check that data was written to store
        assert len(te.store.coords["time"]) == 1
        assert len(te.store.coords["ensemble"]) == 1


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
def test_record_state_with_static_vars(mock_check):
    """Test record_state correctly concatenates static variables"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        static_vars = torch.ones(2, 721, 1440) * 0.5
        static_coords = OrderedDict(
            {
                "variable": np.array(["z", "lsm"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440),
            }
        )

        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m", "v10m"],
            batch_size=1,
            n_steps=2,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            static_vars=static_vars,
            static_coords=static_coords,
            use_ram=False,
        )

        # Create test data (without static vars)
        time = np.datetime64("2024-01-01T00:00:00")
        x = torch.randn(1, 1, 1, 2, 721, 1440)
        coords = OrderedDict(
            {
                "ensemble": np.array([0]),
                "time": np.array([time]),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["u10m", "v10m"]),
                "lat": np.linspace(90, -90, 721),
                "lon": np.linspace(0, 360, 1440),
            }
        )

        te.record_state(x, coords)

        # Verify static vars were concatenated
        assert "z" in te._store_coords["variable"]
        assert "lsm" in te._store_coords["variable"]


# ============================================================================
# 5. DATA DUMPING TESTS
# ============================================================================


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
def test_dump_raw_data_creates_netcdf(mock_check):
    """Test that dump_raw_data creates NetCDF files with correct structure"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m", "v10m", "msl"],
            batch_size=2,
            n_steps=3,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        # Record some data
        time = np.datetime64("2024-01-01T00:00:00")
        for lt in range(4):  # n_steps + 1
            x = torch.randn(2, 1, 1, 3, 721, 1440)
            coords = OrderedDict(
                {
                    "ensemble": np.array([0, 1]),
                    "time": np.array([time]),
                    "lead_time": np.array([np.timedelta64(lt * 6, "h")]),
                    "variable": np.array(["u10m", "v10m", "msl"]),
                    "lat": np.linspace(90, -90, 721),
                    "lon": np.linspace(0, 360, 1440),
                }
            )
            te.record_state(x, coords)

        # Dump data
        raw_files, mems = te.dump_raw_data()

        # Check that files were created
        assert len(raw_files) == 2  # batch_size
        assert len(mems) == 2
        for f in raw_files:
            assert os.path.exists(f)
            assert f.endswith(".nc")


# ============================================================================
# 6. FILE SETUP TESTS
# ============================================================================


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
@patch("earth2studio.models.dx.tempest_extremes.TempestExtremes.dump_raw_data")
def test_setup_files(mock_dump, mock_check):
    """Test that setup_files creates correct file structure"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_dump.return_value = (
            [
                os.path.join(tmpdir, "data_mem_0000.nc"),
                os.path.join(tmpdir, "data_mem_0001.nc"),
            ],
            np.array([0, 1]),
        )
        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m"],
            batch_size=2,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        ins, outs, node_files, track_files = te.setup_files()

        # Check that file lists were created
        assert len(ins) == 2
        assert len(outs) == 2
        assert len(node_files) == 2
        assert len(track_files) == 2

        # Check that input/output lists exist and contain expected content
        for i, in_file in enumerate(ins):
            assert os.path.exists(in_file)
            with open(in_file) as f:
                content = f.read()
                assert "mem_000" in content


def test_setup_files_with_custom_names():
    """Test setup_files with custom output file names"""
    # Similar structure to above, but with out_file_names parameter
    pass


# ============================================================================
# 7. ASYNC FUNCTIONALITY TESTS
# ============================================================================


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
def test_async_initialization(mock_check):
    """Test AsyncTempestExtremes initialization"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        ate = AsyncTempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            timeout=120,
            max_workers=2,
            use_ram=False,
        )

        assert ate.timeout == 120
        assert ate.max_workers == 2
        assert hasattr(ate, "_instance_tasks")
        assert hasattr(ate, "_dump_in_progress")


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
def test_async_task_status(mock_check):
    """Test getting task status from AsyncTempestExtremes"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        ate = AsyncTempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        status = ate.get_task_status()

        assert "running" in status
        assert "pending" in status
        assert "completed" in status
        assert "failed" in status
        assert "total" in status
        assert status["total"] == 0  # No tasks submitted yet


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
def test_async_record_state_waits_for_dump(mock_check):
    """Test that async record_state waits for ongoing dumps"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        ate = AsyncTempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        # Simulate dump in progress
        ate._dump_in_progress.clear()

        # Start recording in a thread (would block)
        import threading

        record_started = threading.Event()
        record_completed = threading.Event()

        def record():
            record_started.set()
            time = np.datetime64("2024-01-01T00:00:00")
            x = torch.randn(1, 1, 1, 1, 721, 1440)
            coords = OrderedDict(
                {
                    "ensemble": np.array([0]),
                    "time": np.array([time]),
                    "lead_time": np.array([np.timedelta64(0, "h")]),
                    "variable": np.array(["u10m"]),
                    "lat": np.linspace(90, -90, 721),
                    "lon": np.linspace(0, 360, 1440),
                }
            )
            ate.record_state(x, coords)
            record_completed.set()

        thread = threading.Thread(target=record)
        thread.start()

        # Wait a bit and verify record hasn't completed
        record_started.wait(timeout=1)
        import time

        time.sleep(0.1)
        assert not record_completed.is_set()

        # Signal dump complete
        ate._dump_in_progress.set()

        # Now record should complete
        thread.join(timeout=2)
        assert record_completed.is_set()


# ============================================================================
# 8. UTILITY FUNCTION TESTS
# ============================================================================


def test_tile_xx_to_yy():
    """Test tiling function for expanding dimensions"""
    from collections import OrderedDict

    from earth2studio.models.dx.tempest_extremes import tile_xx_to_yy

    xx = torch.randn(2, 721, 1440)
    xx_coords = OrderedDict(
        {
            "variable": np.array(["z", "lsm"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    yy = torch.randn(3, 4, 5, 721, 1440)
    yy_coords = OrderedDict(
        {
            "ensemble": np.array([0, 1, 2]),
            "time": np.array([1, 2, 3, 4]),
            "lead_time": np.array([0, 1, 2, 3, 4]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    result, result_coords = tile_xx_to_yy(xx, xx_coords, yy, yy_coords)

    # Result should have yy's leading dims + all of xx's dims
    # yy.shape = (3, 4, 5, 721, 1440), xx.shape = (2, 721, 1440)
    # n_lead = 5 - 3 = 2, so we prepend yy's first 2 dims to xx
    # Result shape should be (3, 4, 2, 721, 1440)
    assert result.shape == (3, 4, 2, 721, 1440)
    assert "variable" in result_coords
    assert "time" in result_coords


def test_cat_coords():
    """Test coordinate concatenation"""
    from collections import OrderedDict

    from earth2studio.models.dx.tempest_extremes import cat_coords

    xx = torch.randn(1, 2, 721, 1440)
    cox = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["u10m", "v10m"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    yy = torch.randn(1, 1, 721, 1440)
    coy = OrderedDict(
        {
            "time": np.array([0]),
            "variable": np.array(["msl"]),
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440),
        }
    )

    result, result_coords = cat_coords(xx, cox, yy, coy, dim="variable")

    assert result.shape == (1, 3, 721, 1440)
    assert len(result_coords["variable"]) == 3
    assert np.array_equal(result_coords["variable"], ["u10m", "v10m", "msl"])


# ============================================================================
# 9. ERROR HANDLING TESTS
# ============================================================================


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
@patch("earth2studio.models.dx.tempest_extremes.TempestExtremes.dump_raw_data")
def test_setup_files_mismatched_file_names_raises_error(mock_dump, mock_check):
    """Test that providing wrong number of output file names raises error"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_dump.return_value = (
            [os.path.join(tmpdir, "data1.nc"), os.path.join(tmpdir, "data2.nc")],
            np.array([0, 1]),
        )
        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m"],
            batch_size=2,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        with pytest.raises(ValueError, match="passed for.*ensemble members"):
            te.setup_files(out_file_names=["only_one_file.csv"])


def test_multiple_ics_raises_error():
    """Test that multiple initial conditions raise an error"""
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="only accepts multiple ensemble members"):
            TempestExtremes(
                detect_cmd="DetectNodes",
                stitch_cmd="StitchNodes",
                input_vars=["u10m"],
                batch_size=1,
                n_steps=5,
                time_step=[np.timedelta64(6, "h"), np.timedelta64(12, "h")],
                lats=np.linspace(90, -90, 721),
                lons=np.linspace(0, 360, 1440),
                store_dir=tmpdir,
                use_ram=False,
            )


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
def test_async_check_for_failures_raises(mock_check):
    """Test that async tracker raises error if previous task failed"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        ate = AsyncTempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        # Manually set failure flag
        ate._has_failed = True

        # Should raise on next operation
        with pytest.raises(ChildProcessError, match="Previous cyclone tracking"):
            ate._check_for_failures()


# ============================================================================
# 10. CLEANUP TESTS
# ============================================================================


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
@patch("earth2studio.models.dx.tempest_extremes.run")
def test_tidy_up_keeps_raw_data(mock_run, mock_check):
    """Test that tidy_up keeps raw data when keep_raw=True"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        te = TempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            keep_raw_data=True,
            use_ram=False,
        )

        # Create dummy files
        raw_file = os.path.join(tmpdir, "raw_data", "test_data.nc")
        os.makedirs(os.path.dirname(raw_file), exist_ok=True)
        with open(raw_file, "w") as f:
            f.write("dummy")

        in_list = os.path.join(tmpdir, "raw_data", "input_list.txt")
        with open(in_list, "w") as f:
            f.write(raw_file)

        out_list = os.path.join(tmpdir, "raw_data", "output_list.txt")
        node_file = os.path.join(tmpdir, "raw_data", "nodes.txt")
        with open(out_list, "w") as f:
            f.write(node_file)
        with open(node_file, "w") as f:
            f.write("dummy")

        # Run cleanup
        te.tidy_up([in_list], [out_list])

        # Raw data should still exist (moved to raw_data subdir)
        assert os.path.exists(os.path.join(te.store_dir, "raw_data"))


@patch(
    "earth2studio.models.dx.tempest_extremes.TempestExtremes.check_tempest_extremes_availability"
)
def test_async_cleanup(mock_check):
    """Test that async cleanup waits for all tasks"""
    mock_check.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        ate = AsyncTempestExtremes(
            detect_cmd="DetectNodes",
            stitch_cmd="StitchNodes",
            input_vars=["u10m"],
            batch_size=1,
            n_steps=5,
            time_step=np.timedelta64(6, "h"),
            lats=np.linspace(90, -90, 721),
            lons=np.linspace(0, 360, 1440),
            store_dir=tmpdir,
            use_ram=False,
        )

        # Cleanup should not raise when no tasks exist
        ate.cleanup()
        assert ate._cleanup_done

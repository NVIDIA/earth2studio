import atexit
import copy
import os
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from subprocess import run
from types import TracebackType
from typing import Any

import numpy as np
import torch
from physicsnemo.distributed import DistributedManager
from rich.console import Console
from rich.table import Table
from rich.traceback import Traceback

from earth2studio.io import KVBackend
from earth2studio.utils.coords import CoordSystem, map_coords, split_coords


def tile_xx_to_yy(
    xx: torch.Tensor, xx_coords: CoordSystem, yy: torch.Tensor, yy_coords: CoordSystem
) -> tuple[torch.Tensor, CoordSystem]:
    """Tile tensor xx to match the leading dimensions of tensor yy.

    Parameters
    ----------
    xx : torch.Tensor
        Source tensor to be tiled
    xx_coords : CoordSystem
        Coordinate system for xx tensor
    yy : torch.Tensor
        Target tensor whose shape determines tiling
    yy_coords : CoordSystem
        Coordinate system for yy tensor

    Returns
    -------
    Tuple[torch.Tensor, CoordSystem]
        Tuple containing the tiled tensor and updated coordinate system

    Raises
    ------
    ValueError
        If xx has more dimensions than yy
    """
    n_lead = len(yy.shape) - len(xx.shape)

    if n_lead < 0:
        raise ValueError("xx must have fewer dimensions than yy.")

    out_shape = yy.shape[:n_lead] + tuple([-1 for _ in range(len(xx.shape))])
    out_coords = copy.deepcopy(yy_coords)
    for key, val in xx_coords.items():
        out_coords[key] = val

    return xx.expand(out_shape), out_coords


def cat_coords(
    xx: torch.Tensor,
    cox: CoordSystem,
    yy: torch.Tensor,
    coy: CoordSystem,
    dim: str = "variable",
) -> tuple[torch.Tensor, CoordSystem]:
    """
    concatenate data along coordinate dimension.

    Parameters
    ----------
    xx : torch.Tensor
        First input tensor which to concatenate
    cox : CoordSystem
        Ordered dict representing coordinate system that describes xx
    yy : torch.Tensor
        Second input tensor which to concatenate
    coy : CoordSystem
        Ordered dict representing coordinate system that describes yy
    dim : str
        name of dimension along which to concatenate

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Tuple containing output tensor and coordinate OrderedDict from
        concatenated data.
    """

    if dim not in cox:
        raise ValueError(f"dim {dim} is not in coords: {list(cox)}.")
    if dim not in coy:
        raise ValueError(f"dim {dim} is not in coords: {list(coy)}.")

    # fix difference in latitude
    _cox = cox.copy()
    _cox["lat"] = coy["lat"]
    xx, cox = map_coords(xx, cox, _cox)

    coords = cox.copy()
    dim_index = list(coords).index(dim)

    zz = torch.cat((xx, yy), dim=dim_index)
    coords[dim] = np.append(cox[dim], coy[dim])

    return zz, coords


class TempestExtremes:
    """TempestExtremes cyclone tracking diagnostic class.

    This class provides functionality for detecting and tracking tropical cyclones
    using the TempestExtremes (TE) software package directly on model outputs.
    To be compatible with TE, model outputs are buffered as a netcdf file with the
    coordinates (time, lat, lon). Optionally, this netcdf file can be stored in RAM to
    avoid writing large amount of data to disk.
    Since TE natively only provides a command-line interface, a subporcess is spawned
    calling TE. To provide users with as much flexibility as possible, the command to
    execude TE has to be specified by the user and passed to the class during
    initialisation. Any arguments regarding input and output files will be ignored
    and automatically populated with the files holding the model output.
    Additionally, this apporach allows the user to point the class to TE executibles
    at any location of the system.
    TE is purely CPU-based, but for most appliations still faster than producing the
    atmospheric data on the GPU. AsyncTempestExtremes provides an implementation in
    which a CPU thread applies TE to the gerneated data, while the GPU continues
    producing the next prediction.

    The result will be a TE-genereated track file, stored in a user-defined location

    NOTE:
        - currently works on batch size 1 only (currently worked on)
        - length of time direction in input coords must be 1 (add test)

    Tip: To iterate on the TE command without having to always run a model, set
    keep_raw_data to True and work with TE directly on the netcdf file that the
    class generates.

    Parameters
    ----------
    detect_cmd : str
        TempestExtremes DetectNodes command with arguments
        Note that --in_data_list and --out_file_list will be ignored if provdied
        example: "/path/to/DetectNodes --mergedist 6 --closedcontourcmd _DIFF(z300,z500),-58.8,6.5,0;msl,200.,5.5,0 --searchbymin msl --outputcmd msl,min,0;_VECMAG(u10m,v10m),max,5;height,min,0"
    stitch_cmd : str
        TempestExtremes StitchNodes command with arguments
        Note that --in and --out will be ignored if provdied
        example: "/path/to/StitchNodes --in_fmt lon,lat,msl,wind_speed,height --range 8.0 --mintime 54h --maxgap 4 --out_file_format csv --threshold wind_speed,>=,10.,10;lat,<=,50.,10;lat,>=,-50.,10;height,<=,150.,10"
    input_vars :
        List of variables which are required for the tracking algorithm
    n_steps :
        Number of time steps
    time_step :
        Time step interval
    lats :
        Latitude coordinates
    lons :
        Longitude coordinates
    static_vars : torch.Tensor, optional
        A tensor holding static data like orography, by default None
    static_coords : dict, optional
        Coordinates of the static data, by default None
    store_dir : str, optional
        Path for storing output files, by default None.
        if use_ram==False, intermediate files will also be written to that path
    keep_raw_data : bool, optional
        Whether to keep raw data files, by default False
    print_te_output : bool, optional
        Whether to print TempestExtremes command line output, by default False
    use_ram : bool, optional
        Whether to use RAM (/dev/shm) for temporary storage, by default True
    **kwargs
        Additional keyword arguments for robustness
    """

    def __init__(
        self,
        detect_cmd: str,
        stitch_cmd: str,
        input_vars: list[str],
        batch_size: int,
        n_steps: int,
        time_step: np.ndarray | list[np.timedelta64] | np.timedelta64,
        lats: np.ndarray | torch.Tensor,
        lons: np.ndarray | torch.Tensor,
        store_dir: str,
        static_vars: torch.Tensor | None = None,
        static_coords: dict | None = None,
        keep_raw_data: bool = False,
        print_te_output: bool = False,
        use_ram: bool = True,
        **kwargs: Any,
    ):  # leave kwars to make call robust to switching between this and async version

        self.rank = DistributedManager().rank
        self.print_te_output = print_te_output
        self.keep_raw = keep_raw_data
        self.static_vars = static_vars
        self.static_coords = static_coords
        self.time_step = time_step
        self.input_vars = input_vars
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.lats = lats
        self.lons = lons

        if isinstance(self.time_step, np.ndarray) or isinstance(self.time_step, list):
            if not len(self.time_step) == 1:
                raise ValueError(
                    "TempestExtremes connector currently only accepts multiple ensemble members from a single ICs."
                )
            self.time_step = self.time_step[0]
        if (static_vars is None) != (static_coords is None):
            raise ValueError(
                "provide both values and coords for static fields or don't."
                + " currently, only one of them is provided"
            )

        self.format_tempestextremes_commands(detect_cmd, stitch_cmd)

        self.check_tempest_extremes_availability()

        self.initialise_store_coords()

        self.initialise_te_kvstore()

        self.setup_output_directories(use_ram, store_dir)

        return

    def format_tempestextremes_commands(self, detect_cmd: str, stitch_cmd: str) -> None:
        """Format TempestExtremes commands by removing input/output file arguments.
        These arguments are populated dynamically by the method.

        Parameters
        ----------
        detect_cmd : str
            TempestExtremes detect command
        stitch_cmd : str
            TempestExtremes stitch command
        """
        self.detect_cmd = detect_cmd.split(" ")
        self.stitch_cmd = stitch_cmd.split(" ")

        self.detect_cmd = self.remove_arguments(
            self.detect_cmd, ["--in_data_list", "--out_file_list"]
        )
        self.stitch_cmd = self.remove_arguments(self.stitch_cmd, ["--in", "--out"])

        return

    def check_tempest_extremes_availability(self) -> None:
        """Check if TempestExtremes executables are available on the system.

        Verifies that both detect and stitch commands can be executed by attempting
        to run them. If either command fails, raises an OptionalDependencyFailure.

        Raises
        ------
        OptionalDependencyFailure
            If TempestExtremes executables are not available or cannot be executed
        """
        for cmd in (self.detect_cmd[0], self.stitch_cmd[0]):
            try:
                run(cmd, capture_output=True)  # noqa: S603
            except Exception as e:
                # TODO discuss implementation of below before merging
                self.dependency_failure(e.__traceback__, e)

        return

    def dependency_failure(
        self, error_traceback: TracebackType | None, error_value: BaseException
    ) -> None:
        """Display a formatted error message for TempestExtremes dependency failures.

        This method creates and prints information about TempestExtremes dependency
        errors and raises a RuntimeError to halt execution.

        Parameters
        ----------
        error_traceback : TracebackType | None
            The traceback object from the exception (typically from exception.__traceback__)
        error_value : BaseException
            The exception object that was raised

        Raises
        ------
        RuntimeError
            Always raised after displaying the error message to halt execution
        """
        doc_url = "https://nvidia.github.io/earth2studio/userguide/about/install.html#optional-dependencies"

        console = Console()
        table = Table(show_header=False, show_lines=True)
        table.add_row(
            "[blue]Earth2Studio Extra Dependency Error\n"
            + "This error typically indicates an extra dependency group is needed.\n"
            + "Don't panic, this is usually an easy fix.[/blue]"
        )
        table.add_row(
            "[yellow]The TempestExtremes class is marked needing optional dependencies'.\n\n"
            + "unlike other dependencies, TempestExtremes has to be installed to the system by the user`\n\n"
            + "For more information, visit the install documentation: \n"
            + f"{doc_url}[/yellow]"
        )
        if error_value:
            table.add_row(
                Traceback(
                    Traceback.extract(
                        type(error_value),
                        error_value,
                        error_traceback,
                        show_locals=False,
                    )
                )
            )
        console.print(table)

        raise RuntimeError(
            "TempestExtremes dependency not available. "
            f"Please install TempestExtremes. See {doc_url}"
        )

    @staticmethod
    def remove_arguments(cmd: list[str], args: list[str]) -> list[str]:
        """Remove specified arguments from command list.

        Parameters
        ----------
        cmd : List[str]
            Command as list of strings
        args : List[str]
            Arguments to remove

        Returns
        -------
        List[str]
            Command with specified arguments removed
        """
        for arg in args:
            if arg in cmd:
                remove = [cmd.index(arg)]
                if len(cmd) > remove[0] + 1:
                    if not cmd[remove[0] + 1].startswith("--"):
                        remove.append(remove[0] + 1)
                cmd = [_arg for ii, _arg in enumerate(cmd) if ii not in remove]

        return cmd

    def setup_output_directories(self, use_ram: bool, store_dir: str) -> None:
        """Set up output directories for final data storage and for storing
        auxiliary files.

        Parameters
        ----------
        use_ram : bool
            Whether to use RAM for temporary storage
        store_dir : Optional[str]
            Base storage directory path
        """
        self.store_dir = os.path.abspath(os.path.join(store_dir, "cyclone_tracks_te"))

        if self.rank == 0:
            os.makedirs(self.store_dir, exist_ok=True)  # output for track data
            if self.keep_raw:
                os.makedirs(
                    os.path.join(self.store_dir, "raw_data"), exist_ok=True
                )  # output for raw data

        if use_ram:
            self.ram_dir = os.path.join("/dev/shm", "cyclone_tracking")  # noqa: S108
            if DistributedManager().local_rank == 0:
                os.makedirs(
                    self.ram_dir, exist_ok=True
                )  # RAM location for processing fields
        else:
            self.ram_dir = os.path.join(self.store_dir, "raw_data")
            if self.rank == 0:
                os.makedirs(
                    self.ram_dir, exist_ok=True
                )  # disk location for processing fields

        return

    def initialise_store_coords(self) -> None:
        """Initialize coordinate system for data storage.

        Creates the internal coordinate system for storing to the internal
        KV io backend, with optionally adding static variables at each time step.
        """
        # extract time step in case it's passed as list or array
        _input_vars = self.input_vars
        if self.static_coords:
            _input_vars = np.array(
                list(self.input_vars) + list(self.static_coords["variable"])
            )

        self._store_coords = OrderedDict(
            {
                "lead_time": np.asarray(
                    [self.time_step * i for i in range(self.n_steps + 1)]
                ).flatten(),
                "variable": np.array(_input_vars),
                "lat": np.array(self.lats),
                "lon": np.array(self.lons),
            }
        )

        return

    @property
    def input_coords(self) -> OrderedDict:
        """Returns the coords of the data to be passed to record_state.

        Returns
        -------
        OrderedDict
            Coordinate system for input data
        """
        return OrderedDict(
            {
                "time": np.empty(0),
                "lead_time": np.asarray(
                    [self.time_step * i for i in range(self.n_steps + 1)]
                ).flatten(),
                "variable": np.array(self.input_vars),
                "lat": np.array(self.lats),
                "lon": np.array(self.lons),
            }
        )

    def initialise_te_kvstore(self) -> None:
        """Initialize the KV store that stores model output in TE compatible format."""
        # create store
        self.store = KVBackend()

        # add arrays to the store
        oco = copy.deepcopy(self._store_coords)
        out_vars = oco.pop("variable")

        oco["time"] = np.array([None])
        oco["ensemble"] = np.array([None] * self.batch_size)
        oco.move_to_end("time", last=False)
        oco.move_to_end("ensemble", last=False)

        self.store.add_array(coords=oco, array_name=out_vars)

        return

    def dump_raw_data(self) -> tuple[list[str], np.ndarray]:
        """Dump raw data from store to NetCDF file.

        Morphs the store to TempestExtremes-compatible [time, lat, lon] structure
        and writes to a (optianlly in-RAM) NetCDF file, which can be ingested by TE.

        Returns
        -------
        str
            Path to the created NetCDF file
        """
        if len(self.store.coords["time"]) > 1:
            raise ValueError(
                "Currently, TempestExtremes interface only works for single IC per evaluation cycle."
            )

        ic = self.store.coords["time"][0]
        mems = self.store.coords["ensemble"]

        # morph store to tempes extremes-compatible [time, lat, lon] structure
        lead_times = self.store.coords.pop("lead_time")
        self.store.coords["time"] = [ic + lt for lt in lead_times]

        for var in self.store.root.keys():
            self.store.root[var] = self.store.root[var].squeeze(1)
            self.store.dims[var] = ["ensemble", "time", "lat", "lon"]

        # for each member, write te compatible nc file
        raw_files = []

        # in case last batch has less ensemble members than batch size, expand coords of store so to_xarray() does not fail
        if len(mems) < self.batch_size:
            self.store.coords["ensemble"] = list(self.store.coords["ensemble"]) + [
                None
            ] * (self.batch_size - len(mems))

        for mem in mems:
            raw_dat = f"{np.datetime_as_string(ic, unit='s')}_mem_{mem:04d}.nc"
            raw_dat = os.path.join(self.ram_dir, raw_dat)
            raw_files.append(raw_dat)
            kw_args = {"path": raw_dat, "format": "NETCDF4", "mode": "w"}

            _store = self.store.to_xarray().sel(ensemble=mem).drop_vars("ensemble")
            _store.to_netcdf(**kw_args)

        # new clean store
        self.initialise_te_kvstore()

        return raw_files, mems

    def setup_files(
        self, out_file_names: list[str] | None = None
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Set up input and output files for TE processing.

        Creates necessary file lists and paths for TE detect and stitch operations.

        Returns
        -------
        tuple[list[str], list[str], list[str], list[str]]
            Tuple containing (input_file_list, output_file_list, node_file, track_file)
        """
        # write store to in-memory file
        raw_files, mems = self.dump_raw_data()

        # make sure if file names are passed that there's one per member
        if out_file_names is not None:
            if len(out_file_names) != len(mems):
                raise ValueError(
                    f"{len(out_file_names)} passed for {len(mems)} ensemble members"
                )

        ins, outs, node_files, track_files = [], [], [], []
        for ii, (mem, raw_dat) in enumerate(zip(mems, raw_files)):
            # in_files to file
            _ins = os.path.join(
                self.ram_dir, f"input_files_{self.rank:04d}_mem_{mem:05d}.txt"
            )
            with open(_ins, "w") as fl:
                fl.write(raw_dat + "\n")

            _outs = os.path.join(
                self.ram_dir, f"output_files_{self.rank:04d}_mem_{mem:05d}.txt"
            )
            base_name = os.path.basename(raw_dat)
            base_name = base_name.rsplit(".", 1)[0]
            node_file = os.path.join(
                self.ram_dir, base_name + f"_mem_{mem:05d}_tc_centres.txt"
            )
            with open(_outs, "w") as fl:
                fl.write(node_file + "\n")

            track_file = (
                out_file_names[ii] if out_file_names else "tracks_" + base_name + ".csv"
            )
            track_file = os.path.join(self.store_dir, track_file)

            ins.append(_ins)
            outs.append(_outs)
            node_files.append(node_file)
            track_files.append(track_file)

        return ins, outs, node_files, track_files

    def record_state(self, xx: torch.Tensor, coords: CoordSystem) -> None:
        """Record state data to the internal store.

        Updates store with current time and ensemble information,
        concatenates static data if available, and writes to store.

        Parameters
        ----------
        xx : torch.Tensor
            Input tensor data
        coords : CoordSystem
            Coordinate system for the input data
        """
        # update store mem and time
        for dim in ["time", "ensemble"]:
            if not np.array_equal(self.store.coords[dim], coords[dim]):
                self.store.coords[dim] = coords[dim]

        # concatenate static data
        if self.static_vars is not None:
            self.static_vars = self.static_vars.to(xx.device)
            if len(xx.shape) > len(self.static_vars.shape):
                _static_vars, _static_coords = tile_xx_to_yy(
                    self.static_vars, self.static_coords, xx, coords
                )
            else:
                _static_vars, _static_coords = self.static_vars, self.static_coords
            xx, coords = cat_coords(xx, coords, _static_vars, _static_coords)

        # select output data and write to store
        xx_sub, coords_sub = map_coords(xx, coords, self._store_coords)
        self.store.write(*split_coords(xx_sub, coords_sub, dim="variable"))

        return

    @staticmethod
    def run_te(command: list[str], print_output: bool) -> None:
        """Run TempestExtremes command and handle output.

        Parameters
        ----------
        command : list[str]
            Command to execute as list of strings
        print_output : bool
            Whether to print command output to terminal

        Raises
        ------
        ChildProcessError
            If TempestExtremes command fails (detected by "EXCEPTION" in output)
        """
        # detect nodes
        out = run(command, capture_output=True)  # noqa: S603

        # Print output to terminal if requested
        if print_output:
            print(out.stdout.decode("utf-8"))

        # unfortunately, TE does not fail with proper returncode
        if "EXCEPTION" in out.stdout.decode("utf-8"):
            print(out.stdout.decode("utf-8"))
            raise ChildProcessError(
                f"\nERROR: {command[0]} failed, see output above from TempestExtremes for details."
            )

        return

    def track_cyclones(self, out_file_names: list[str] | None = None) -> None:
        """Execute cyclone tracking using TempestExtremes.

        Runs the complete tracking pipeline including node detection
        and stitching operations.

        Raises
        ------
        ChildProcessError
            If TempestExtremes detect or stitch operations fail
        """
        then = time.time()

        # set up TE helper files
        insies, outsies, node_files, self.track_files = self.setup_files(out_file_names)

        for ins, outs, node_file, track_file in zip(
            insies, outsies, node_files, self.track_files
        ):
            # detect nodes
            self.run_te(
                command=self.detect_cmd
                + ["--in_data_list", ins, "--out_file_list", outs],
                print_output=self.print_te_output,
            )

            # stitch them together
            self.run_te(
                command=self.stitch_cmd + ["--in", node_file, "--out", track_file],
                print_output=self.print_te_output,
            )

        # remove helper files
        self.tidy_up(insies, outsies)

        print(f"took {(time.time() - then):.1f}s to track cyclones")

        return

    def __call__(self, out_file_names: list[str] | None = None) -> None:
        """Make the class callable to execute cyclone tracking.

        Returns
        -------
        None
            Calls track_cyclones method
        """
        return self.track_cyclones(out_file_names)

    def tidy_up(self, insies: list[str], outsies: list[str]) -> None:
        """Clean up temporary files after processing.

        Handles raw data files based on keep_raw_data setting and
        removes TempestExtremes helper files.

        Parameters
        ----------
        ins : str
            Path to input file list
        outs : str
            Path to output file list
        """
        for ins, outs in zip(insies, outsies):
            with open(ins) as fl:
                raw_dat = fl.read().strip()

            if self.keep_raw:  # move field data to disk
                out_path = os.path.join(self.store_dir, "raw_data")
                raw_dest = os.path.join(out_path, os.path.basename(raw_dat))
                if raw_dat != raw_dest:  # nothing to do if use_ram=False
                    if os.path.exists(raw_dest):
                        os.remove(raw_dest)
                    shutil.move(raw_dat, out_path)
            else:  # delete field data
                os.remove(raw_dat)

            # delete TE helper files
            files = [ins, outs]
            with open(outs) as fl:
                _files = [line.strip() for line in fl]
            files.extend(_files)

            for file in files:
                os.remove(file)

        return


# Global thread pool for background TempestExtremes processing
_tempest_executor: ThreadPoolExecutor | None = None
_tempest_background_tasks: list[Future] = []
_tempest_executor_lock: threading.Lock = threading.Lock()


def get_tempest_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor for TempestExtremes processing."""
    global _tempest_executor
    with _tempest_executor_lock:
        if _tempest_executor is None:
            # Reduced worker count for long-running tasks to avoid resource exhaustion
            max_workers = min(4, os.cpu_count() or 1)
            _tempest_executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="tempest_extremes"
            )
            # Ensure cleanup on exit
            atexit.register(cleanup_tempest_executor)
    return _tempest_executor


def cleanup_tempest_executor(timeout_per_task: int = 360) -> None:
    """Clean up the thread pool executor.

    Parameters
    ----------
    timeout_per_task : int, optional
        Timeout in seconds for each task, by default 360

    Raises
    ------
    ChildProcessError
        If one or more background TempestExtremes tasks failed
    """
    """Clean up the thread pool executor."""
    global _tempest_executor, _tempest_background_tasks
    with _tempest_executor_lock:
        if _tempest_executor is not None:
            # Wait for all background tasks to complete
            child_process_errors = []
            for i, future in enumerate(_tempest_background_tasks):
                try:
                    future.result(
                        timeout=timeout_per_task
                    )  # Configurable timeout per task
                except ChildProcessError as e:
                    print(
                        f"Background TempestExtremes task {i+1} failed with ChildProcessError: {e}"
                    )
                    child_process_errors.append(e)
                except Exception as e:
                    print(f"Background TempestExtremes task {i+1} failed: {e}")
            _tempest_executor.shutdown(wait=True)
            _tempest_executor = None
            _tempest_background_tasks.clear()

            # Re-raise ChildProcessError if any occurred
            if child_process_errors:
                raise ChildProcessError(
                    f"One or more background TempestExtremes tasks failed: {child_process_errors}"
                )


class AsyncTempestExtremes(TempestExtremes):
    """Asynchronous version of TempestExtremes that runs cyclone tracking in background threads.

    This class extends TempestExtremes to provide asynchronous cyclone tracking capabilities
    using background thread pools. It allows for non-blocking cyclone tracking operations
    while maintaining data integrity through synchronization mechanisms.

    Follows the parallelization paradigm from async_tracking_utils.py.

    Parameters
    ----------
    *args
        Positional arguments passed to parent TempestExtremes class
    **kwargs
        Keyword arguments passed to parent TempestExtremes class.
        Special parameters:
        - timeout : int, optional
            Timeout in seconds for operations, by default 60
        - max_workers : int, optional
            Maximum number of worker threads for parallel ensemble member processing.
            If None (default), uses min(n_members, cpu_count). If provided, caps
            the number of workers to this value, by default None
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Extract async-specific parameters with default values
        self.timeout = kwargs.pop("timeout", 60)
        self.max_workers = kwargs.pop("max_workers", None)

        # Initialize the parent class
        super().__init__(*args, **kwargs)

        # Track background tasks for this instance
        self._instance_tasks: list[Future] = []
        self._instance_lock = threading.Lock()
        self._dump_in_progress = threading.Event()
        self._dump_in_progress.set()  # Initially no dump in progress
        self._has_failed = False  # Track if any previous task failed
        self._cleanup_done = False  # Track if cleanup has been performed

    def record_state(self, xx: torch.Tensor, coords: CoordSystem) -> None:
        """Record state but ensure dump_raw_data has finished successfully before proceeding.

        This method ensures data integrity by waiting for any ongoing dump operations
        to complete before recording new state data.

        Parameters
        ----------
        xx : torch.Tensor
            Input tensor data
        coords : CoordSystem
            Coordinate system for the input data

        Raises
        ------
        ChildProcessError
            If any previous cyclone tracking task failed
        """
        # Check if any previous task failed and raise error immediately
        self._check_for_failures()

        # Wait for any ongoing dump_raw_data to complete
        self._dump_in_progress.wait()

        # Call parent's record_state method
        super().record_state(xx, coords)

    def dump_raw_data(self) -> tuple[list[str], np.ndarray]:
        """Dump raw data with synchronization to prevent concurrent access.

        Uses threading events to ensure only one dump operation occurs at a time
        to maintain data integrity.

        Returns
        -------
        tuple[list[str], np.ndarray]
            Tuple containing list of raw file paths and array of member indices
        """
        # Signal that dump is in progress
        self._dump_in_progress.clear()

        try:
            # Call parent's dump_raw_data method
            result = super().dump_raw_data()
            return result
        finally:
            # Signal that dump is complete
            self._dump_in_progress.set()

    def _check_for_failures(self) -> None:
        """Check if any previous tasks failed with ChildProcessError and raise immediately.

        This method checks the status of all background tasks and raises an exception
        if any have failed, ensuring early failure detection.

        Raises
        ------
        ChildProcessError
            If any previous cyclone tracking task failed
        Exception
            If any task failed with a different exception type
        """
        if self._has_failed:
            raise ChildProcessError(
                "Previous cyclone tracking task failed - stopping execution"
            )

        # Check completed tasks for failures
        with self._instance_lock:
            for future in self._instance_tasks:
                if future.done():
                    try:
                        future.result()  # This will raise if the task failed
                    except ChildProcessError as e:
                        self._has_failed = True
                        raise e  # Re-raise immediately
                    except Exception as e:
                        self._has_failed = True
                        raise e  # Re-raise other exceptions too

    def track_cyclones_async(self, out_file_names: list[str] | None = None) -> Future:
        """Submit cyclone tracking to background thread pool.

        This method submits cyclone tracking operations to a background thread pool
        for asynchronous execution, allowing the main thread to continue processing.

        Returns
        -------
        Future
            A Future object that can be used to check status or get results

        Raises
        ------
        ChildProcessError
            If any previous cyclone tracking task failed
        """
        # Check if any previous task failed and raise error immediately
        self._check_for_failures()

        global _tempest_background_tasks

        # Clean up completed tasks to avoid memory buildup
        with _tempest_executor_lock:
            _tempest_background_tasks = [
                f for f in _tempest_background_tasks if not f.done()
            ]

        with self._instance_lock:
            self._instance_tasks = [f for f in self._instance_tasks if not f.done()]

        # Check if we have too many running tasks
        with _tempest_executor_lock:
            running_tasks = len(_tempest_background_tasks)
            if running_tasks >= 3:  # Conservative limit for long-running tasks
                print(
                    f"Warning: {running_tasks} TempestExtremes tasks already running. Consider waiting for some to complete."
                )

        # Submit the task
        executor = get_tempest_executor()
        future = executor.submit(self.track_cyclones, out_file_names)

        # Add to both global and instance task lists
        with _tempest_executor_lock:
            _tempest_background_tasks.append(future)

        with self._instance_lock:
            self._instance_tasks.append(future)

        return future

    def get_task_status(self) -> dict[str, int]:
        """Get status of background tasks for this instance.

        Returns
        -------
        Dict[str, int]
            Dictionary containing task status counts with keys:
            - 'running': Number of currently running tasks
            - 'pending': Number of pending tasks
            - 'completed': Number of completed tasks
            - 'failed': Number of failed tasks
            - 'total': Total number of tasks
        """
        with self._instance_lock:
            running = sum(1 for f in self._instance_tasks if f.running())
            pending = sum(
                1 for f in self._instance_tasks if not f.done() and not f.running()
            )
            completed = sum(1 for f in self._instance_tasks if f.done())
            failed = sum(
                1
                for f in self._instance_tasks
                if f.done() and f.exception() is not None
            )

            return {
                "running": running,
                "pending": pending,
                "completed": completed,
                "failed": failed,
                "total": len(self._instance_tasks),
            }

    def get_task_results(
        self, raise_on_error: bool = True
    ) -> dict[str, Any] | tuple[list[Any], list[tuple[str, Exception]]]:
        """Get results from all completed tasks.

        Parameters
        ----------
        raise_on_error : bool, optional
            If True, raises ChildProcessError if any task failed with that error.
            If False, returns results and exceptions separately, by default True

        Returns
        -------
        Union[Dict[str, Any], tuple[list[Any], list[tuple[str, Exception]]]]
            If raise_on_error=True: dict with successful results
            If raise_on_error=False: tuple of (results, exceptions)

        Raises
        ------
        ChildProcessError
            If raise_on_error=True and any task failed with ChildProcessError
        Exception
            If raise_on_error=True and any task failed with other exceptions
        """
        with self._instance_lock:
            tasks_to_check = list(self._instance_tasks)

        results = []
        exceptions: list[tuple[str, Exception]] = []

        for i, future in enumerate(tasks_to_check):
            if future.done():
                try:
                    result = future.result()
                    results.append(result)
                except ChildProcessError as e:
                    if raise_on_error:
                        raise  # Re-raise ChildProcessError immediately
                    exceptions.append(("ChildProcessError", e))
                except Exception as e:
                    if raise_on_error:
                        raise  # Re-raise any other exceptions
                    exceptions.append(("Exception", e))

        if raise_on_error:
            return {"results": results, "task_count": len(tasks_to_check)}
        else:
            return results, exceptions

    def wait_for_completion(self, timeout_per_task: int = 360) -> None:
        """Wait for all background tasks associated with this instance to complete.

        This method blocks until all background tasks have finished execution.

        Parameters
        ----------
        timeout_per_task : int, optional
            Timeout in seconds for each task, by default 360

        Raises
        ------
        ChildProcessError
            If any background task failed with ChildProcessError
        Exception
            If any background task failed with other exceptions
        """
        with self._instance_lock:
            tasks_to_wait = list(self._instance_tasks)

        if tasks_to_wait:
            print(
                f"Waiting for {len(tasks_to_wait)} TempestExtremes tasks to complete..."
            )

            for i, future in enumerate(tasks_to_wait):
                try:
                    print(f"Waiting for task {i+1}/{len(tasks_to_wait)} to complete...")
                    future.result(timeout=timeout_per_task)
                except ChildProcessError as e:
                    print(f"Task {i+1} failed with ChildProcessError: {e}")
                    raise  # Re-raise ChildProcessError to propagate it
                except Exception as e:
                    print(f"Task {i+1} failed: {e}")
                    raise  # Re-raise any other exceptions as well
        else:
            print("No background tasks to wait for.")

    def _process_single_member(
        self, ins: str, outs: str, node_file: str, track_file: str
    ) -> None:
        """Process a single ensemble member (detect + stitch).

        This method is designed to run in a separate thread for parallel processing
        of ensemble members.

        Parameters
        ----------
        ins : str
            Path to input file list for this member
        outs : str
            Path to output file list for this member
        node_file : str
            Path to node file for this member
        track_file : str
            Path to track file for this member

        Raises
        ------
        ChildProcessError
            If TempestExtremes detect or stitch operations fail
        """
        then = time.time()

        # detect nodes
        self.run_te(
            command=self.detect_cmd + ["--in_data_list", ins, "--out_file_list", outs],
            print_output=self.print_te_output,
        )

        # stitch them together
        self.run_te(
            command=self.stitch_cmd + ["--in", node_file, "--out", track_file],
            print_output=self.print_te_output,
        )

        if self.print_te_output:
            print(f"took {(time.time() - then):.1f}s to track cyclones for one member")

    def track_cyclones(self, out_file_names: list[str] | None = None) -> None:
        """Execute cyclone tracking with parallel processing per ensemble member.

        This override of the parent method dumps data synchronously (within this thread),
        then spawns separate threads for each ensemble member's processing. This allows
        multiple ensemble members to be processed in parallel while avoiding race
        conditions during data dumping.

        Uses a dedicated local thread pool for member processing to avoid conflicts
        with the global executor during shutdown.

        Parameters
        ----------
        out_file_names : list of str, optional
            Custom output file names for each ensemble member

        Raises
        ------
        ChildProcessError
            If TempestExtremes detect or stitch operations fail for any member
        """
        then = time.time()

        # Dump data and setup files (synchronous within this thread to avoid race conditions)
        insies, outsies, node_files, self.track_files = self.setup_files(out_file_names)

        # Create a dedicated local thread pool for member processing
        # This avoids conflicts with the global executor during shutdown
        n_members = len(insies)
        max_workers = min(n_members, os.cpu_count() or 1)
        if self.max_workers is not None:
            max_workers = min(self.max_workers, max_workers)

        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="te_members"
        ) as executor:
            # Submit one task per member to process in parallel
            member_futures = []

            for ins, outs, node_file, track_file in zip(
                insies, outsies, node_files, self.track_files
            ):
                future = executor.submit(
                    self._process_single_member, ins, outs, node_file, track_file
                )
                member_futures.append(future)

            # Wait for all members to complete
            exceptions = []
            for i, future in enumerate(member_futures):
                try:
                    future.result()  # This will raise if the task failed
                except Exception as e:
                    print(f"Member {i+1} processing failed: {e}")
                    exceptions.append((i, e))

        # Raise if any failed
        if exceptions:
            error_msg = (
                f"Processing failed for {len(exceptions)} member(s): {exceptions}"
            )
            raise ChildProcessError(error_msg)

        # Clean up after all are done
        self.tidy_up(insies, outsies)

        print(
            f"took {(time.time() - then):.1f}s to track cyclones for all {len(member_futures)} members"
        )

    def cleanup(self, timeout_per_task: int = 360) -> None:
        """Explicitly clean up and wait for all background tasks to complete.

        This method should be called before the object is destroyed or the program exits
        to ensure all cyclone tracking tasks complete successfully.

        Parameters
        ----------
        timeout_per_task : int, optional
            Timeout in seconds for each task, by default 360

        Raises
        ------
        ChildProcessError
            If any background task failed
        Exception
            If any task failed with other exceptions
        """
        if self._cleanup_done:
            return

        try:
            # Wait for any ongoing dump to complete
            if hasattr(self, "_dump_in_progress"):
                self._dump_in_progress.wait(timeout=60)

            # Wait for all instance tasks to complete
            if hasattr(self, "_instance_tasks") and hasattr(self, "_instance_lock"):
                with self._instance_lock:
                    tasks_to_wait = list(self._instance_tasks)

                if tasks_to_wait:
                    print(
                        f"AsyncTempestExtremes: waiting for {len(tasks_to_wait)} background tasks to complete..."
                    )

                    for i, future in enumerate(tasks_to_wait):
                        try:
                            print(f"  Waiting for task {i+1}/{len(tasks_to_wait)}...")
                            future.result(timeout=timeout_per_task)
                            print(
                                f"  Task {i+1}/{len(tasks_to_wait)} completed successfully"
                            )
                        except ChildProcessError as e:
                            print(
                                f"  Task {i+1}/{len(tasks_to_wait)} failed with ChildProcessError: {e}"
                            )
                            raise  # Re-raise to propagate the error
                        except Exception as e:
                            print(f"  Task {i+1}/{len(tasks_to_wait)} failed: {e}")
                            raise  # Re-raise to propagate the error

                    print(
                        f"All {len(tasks_to_wait)} background tasks completed successfully"
                    )

            self._cleanup_done = True

        except Exception as _:
            self._cleanup_done = True  # Mark as done even on failure to avoid retry
            raise

    def __del__(self) -> None:
        """Destructor that ensures all processes have finished before the class gets destroyed.

        This method performs cleanup operations when the object is being destroyed,
        ensuring all background tasks complete and resources are properly released.
        """
        try:
            # Use getattr to handle case where initialization failed before _cleanup_done was set
            if not getattr(self, "_cleanup_done", True):
                self.cleanup(timeout_per_task=30)  # Shorter timeout in destructor
        except Exception as e:
            print(
                "\033[95mError in AsyncTempestExtremes destructor: had no time to cleanup.\033[0m"
            )
            print(
                "\033[95mplease call self.cleanup() from script before class goes out of scope.\033[0m"
            )
            print(f"\033[95mThis is resulting in the following error: {e}\033[0m")
            # Note: In destructor, we log but don't re-raise to avoid issues during interpreter shutdown

    def __call__(self, out_file_names: list[str] | None = None) -> Future[None]:  # type: ignore[override]
        """Override call method to use async version by default.

        This method makes the class callable and uses the asynchronous
        tracking method by default.

        Returns
        -------
        Future[None]
            A Future object for the submitted cyclone tracking task

        Raises
        ------
        ChildProcessError
            If any previous cyclone tracking task failed
        """
        # Check if any previous task failed and raise error immediately
        self._check_for_failures()
        return self.track_cyclones_async(out_file_names)

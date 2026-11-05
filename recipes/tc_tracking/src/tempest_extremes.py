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
import shutil
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from subprocess import run

import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig

# in-memory file system, the default location of TE's input files
SHARED_MEMORY = "/dev/shm"  # noqa: S108


class TempestExtremes:
    """Track tropical cyclones in model output with TempestExtremes (TE).

    Forecast steps of a batch are buffered with :meth:`record_state`. Calling
    the tracker writes one NetCDF file per ensemble member, with dimensions
    (time, lat, lon), and runs TE's DetectNodes and StitchNodes on it, which
    produces one CSV track file per member in ``<store_dir>/cyclone_tracks_te``.

    TE only provides a command-line interface, so the DetectNodes and
    StitchNodes commands are passed in as strings, giving full control over the
    tracking criteria. Input and output file arguments are filled in by the
    class and ignored if present.

    TE runs on the CPU. With ``asynchronous=True``, the members are tracked in
    background threads while the GPU produces the next forecast. Failures are
    raised at the next call of the tracker or at :meth:`close`, which waits for
    all members and has to be called when done.

    Tip: to iterate on the TE commands without running a model, set
    ``keep_raw_data=True`` and call TE directly on the stored NetCDF files.

    Parameters
    ----------
    detect_cmd : str
        DetectNodes command with arguments, e.g. "DetectNodes --mergedist 6
        --closedcontourcmd _DIFF(z300,z500),-58.8,6.5,0;msl,200.,5.5,0
        --searchbymin msl --outputcmd msl,min,0;_VECMAG(u10m,v10m),max,5"
    stitch_cmd : str
        StitchNodes command with arguments, e.g. "StitchNodes --in_fmt
        lon,lat,msl,wind_speed --range 8.0 --mintime 54h --maxgap 4
        --out_file_format csv"
    input_vars : list[str]
        Variables the TE commands use
    store_dir : str
        Directory under which the track files are stored
    heights : xr.DataArray, optional
        Surface height on the (lat, lon) grid of the tracked fields, added to
        the input files as variable ``height``, by default None
    keep_raw_data : bool, optional
        Keep the NetCDF input files of TE in ``<store_dir>/cyclone_tracks_te/
        raw_data``, by default False
    print_te_output : bool, optional
        Print the command line output of TE, by default False
    scratch_dir : str, optional
        Directory for the TE input files, by default /dev/shm (in memory) if
        available, else the store directory. Each tracker uses its own
        temporary subdirectory, which :meth:`close` removes.
    asynchronous : bool, optional
        Track in background threads, by default False
    max_workers : int, optional
        Number of members tracked in parallel in asynchronous mode, by default
        the number of CPU cores per GPU
    timeout : float, optional
        Seconds :meth:`close` waits for each member, by default 120
    """

    def __init__(
        self,
        detect_cmd: str,
        stitch_cmd: str,
        input_vars: list[str],
        store_dir: str,
        heights: xr.DataArray | None = None,
        keep_raw_data: bool = False,
        print_te_output: bool = False,
        scratch_dir: str | None = None,
        asynchronous: bool = False,
        max_workers: int | None = None,
        timeout: float = 120,
    ):
        self.input_vars = list(input_vars)
        self.heights = heights
        self.keep_raw = keep_raw_data
        self.print_te_output = print_te_output
        self.timeout = timeout

        # file arguments are set per member
        self.detect_cmd = self._without_arguments(
            detect_cmd.split(),
            ["--in_data", "--in_data_list", "--out", "--out_file_list"],
        )
        self.stitch_cmd = self._without_arguments(
            stitch_cmd.split(), ["--in", "--in_list", "--out"]
        )
        for cmd in (self.detect_cmd[0], self.stitch_cmd[0]):
            if shutil.which(cmd) is None:
                raise RuntimeError(
                    f"TempestExtremes executable {cmd} not found. Install "
                    "TempestExtremes (the recipe's container includes it) or "
                    "give the full path in the command."
                )

        self.store_dir = os.path.abspath(os.path.join(store_dir, "cyclone_tracks_te"))
        os.makedirs(self.store_dir, exist_ok=True)
        if self.keep_raw:
            os.makedirs(os.path.join(self.store_dir, "raw_data"), exist_ok=True)

        if scratch_dir is None:
            scratch_dir = (
                SHARED_MEMORY if os.path.isdir(SHARED_MEMORY) else self.store_dir
            )
        self.scratch_dir = tempfile.mkdtemp(prefix="tc_tracking_", dir=scratch_dir)

        # forecast steps of the current batch, filled by record_state
        self._buffer: list[xr.DataArray] = []

        self._executor: ThreadPoolExecutor | None = None
        self._tasks: list[Future] = []
        if asynchronous:
            if max_workers is None:
                max_workers = max(
                    1, (os.cpu_count() or 1) // max(1, torch.cuda.device_count())
                )
            self._executor = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix="tempest_extremes"
            )

    @classmethod
    def from_config(
        cls, cfg: DictConfig, store_dir: str, asynchronous: bool | None = None
    ) -> "TempestExtremes":
        """Create a tracker from the ``cyclone_tracking`` section of a config.

        Parameters
        ----------
        cfg : DictConfig
            The ``cyclone_tracking`` section of the Hydra config
        store_dir : str
            Directory under which the track files are stored
        asynchronous : bool, optional
            Overrides ``cfg.asynchronous``, by default None

        Returns
        -------
        TempestExtremes
            Configured tracker
        """
        from src.data.tc_hunt_data_utils import load_heights

        if asynchronous is None:
            asynchronous = cfg.get("asynchronous", False)
        return cls(
            detect_cmd=cfg.detect_cmd,
            stitch_cmd=cfg.stitch_cmd,
            input_vars=cfg.vars,
            store_dir=store_dir,
            heights=(
                load_heights(cfg.orography_path) if "orography_path" in cfg else None
            ),
            keep_raw_data=cfg.get("keep_raw_data", False),
            print_te_output=cfg.get("print_te_output", False),
            scratch_dir=cfg.get("scratch_dir", None),
            asynchronous=asynchronous,
            max_workers=cfg.get("max_workers_per_rank", None),
            timeout=cfg.get("task_timeout_seconds", 120),
        )

    @staticmethod
    def _without_arguments(cmd: list[str], args: list[str]) -> list[str]:
        """Remove arguments and their values from a command."""
        out: list[str] = []
        skip = False
        for ii, token in enumerate(cmd):
            if skip:
                skip = False
            elif token in args:
                # skip the value as well, unless the argument has none
                skip = ii + 1 < len(cmd) and not cmd[ii + 1].startswith("--")
            else:
                out.append(token)
        return out

    def reset(self) -> None:
        """Discard the buffered forecast steps, e.g. of an unstable batch."""
        self._buffer = []

    def record_state(self, x: xr.DataArray) -> None:
        """Buffer the tracking variables of one forecast step on the host.

        Parameters
        ----------
        x : xr.DataArray
            Forecast with dimensions (ensemble, time, lead_time, variable,
            lat, lon) on the grid of ``heights``
        """
        self._buffer.append(x.sel(variable=self.input_vars).e2s.as_numpy())

    def __call__(self, out_file_names: list[str] | None = None) -> list[str]:
        """Track cyclones in the buffered forecast, one track file per member.

        Parameters
        ----------
        out_file_names : list[str], optional
            Names of the track files, one per member, by default
            ``tracks_<ic>_mem_<member>.csv``

        Returns
        -------
        list[str]
            Paths of the track files, written later in asynchronous mode

        Raises
        ------
        ChildProcessError
            If TE failed for a previously submitted member
        """
        self._raise_failures()

        raw_files = self.dump_raw_data()
        if out_file_names is None:
            out_file_names = [
                "tracks_" + os.path.basename(raw).removesuffix(".nc") + ".csv"
                for raw in raw_files
            ]
        if len(out_file_names) != len(raw_files):
            raise ValueError(
                f"{len(out_file_names)} file names passed for {len(raw_files)} members"
            )

        track_files = [os.path.join(self.store_dir, name) for name in out_file_names]
        for raw, track in zip(raw_files, track_files):
            if self._executor is None:
                self._track_member(raw, track)
            else:
                self._tasks.append(
                    self._executor.submit(self._track_member, raw, track)
                )

        return track_files

    def dump_raw_data(self) -> list[str]:
        """Write the buffered forecast to one NetCDF file per ensemble member.

        TE expects fields with dimensions (time, lat, lon), so the lead times
        of the buffer are converted to absolute times.

        Returns
        -------
        list[str]
            Paths of the NetCDF files, one per member
        """
        fields = xr.concat(self._buffer, dim="lead_time")
        self._buffer = []

        if fields.sizes["time"] != 1:
            raise ValueError("all members of a batch must start from the same IC")
        ic = fields.coords["time"].values[0]

        # (ensemble, time, lat, lon) per variable, without model metadata
        fields = fields.isel(time=0, drop=True).reset_coords(drop=True)
        fields = fields.assign_coords(lead_time=ic + fields.coords["lead_time"].values)
        fields = fields.rename(lead_time="time")
        fields.attrs = {}
        data = fields.to_dataset(dim="variable")

        if self.heights is not None:
            data["height"] = (
                ("time", "lat", "lon"),
                np.broadcast_to(
                    self.heights.values, (data.sizes["time"], *self.heights.shape)
                ),
            )

        # written in the calling thread, as HDF5 is not thread-safe
        raw_files = []
        for mem in fields.coords["ensemble"].values:
            raw = os.path.join(
                self.scratch_dir,
                f"{np.datetime_as_string(ic, unit='s')}_mem_{mem:04d}.nc",
            )
            data.sel(ensemble=mem, drop=True).to_netcdf(raw, format="NETCDF4", mode="w")
            raw_files.append(raw)

        return raw_files

    def _track_member(self, raw_file: str, track_file: str) -> None:
        """Run DetectNodes and StitchNodes on the input file of one member."""
        then = time.time()
        node_file = raw_file.removesuffix(".nc") + "_nodes.txt"
        try:
            self._run(self.detect_cmd + ["--in_data", raw_file, "--out", node_file])
            self._run(self.stitch_cmd + ["--in", node_file, "--out", track_file])
        finally:
            if os.path.exists(node_file):
                os.remove(node_file)
            if self.keep_raw:
                raw_dir = os.path.join(self.store_dir, "raw_data")
                shutil.move(raw_file, os.path.join(raw_dir, os.path.basename(raw_file)))
            else:
                os.remove(raw_file)
        logger.info(
            f"took {time.time() - then:.1f}s to track {os.path.basename(track_file)}"
        )

    def _run(self, command: list[str]) -> None:
        """Run a TE command, raising ChildProcessError if it failed.

        TE exits with 0 even on errors, but reports them as EXCEPTION. A
        non-zero exit code means the operating system killed the process.
        """
        out = run(command, capture_output=True, text=True)  # noqa: S603
        failed = out.returncode != 0 or "EXCEPTION" in out.stdout
        if self.print_te_output or failed:
            print(out.stdout, out.stderr)
        if failed:
            raise ChildProcessError(
                f"{command[0]} failed (exit code {out.returncode}), see output above"
            )

    def _raise_failures(self) -> None:
        """Raise the error of the first failed background task, if any."""
        for task in self._tasks:
            if task.done() and task.exception() is not None:
                raise ChildProcessError("cyclone tracking failed") from task.exception()
        self._tasks = [task for task in self._tasks if not task.done()]

    def close(self) -> None:
        """Wait for all members to be tracked and remove the scratch directory.

        Raises
        ------
        ChildProcessError
            If TE failed for any member
        """
        try:
            for task in self._tasks:
                task.result(timeout=self.timeout)
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
            shutil.rmtree(self.scratch_dir, ignore_errors=True)

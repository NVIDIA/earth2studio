#!/usr/bin/env python3
"""Download MRMS fields to Zarr v3 with UTC-only timestamps.

- Default cadence: 10 minutes.
- For each target time, MRMS resolves nearest available file within tolerance.
- Zarr chunking: time=1, spatial dims full-domain.
- Includes a short test mode: 1 hour (6 frames) on 2024-05-15.
- Downloads both variables together: refc and refc_base.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import sleep

import numpy as np
import zarr
from tqdm import tqdm

from earth2studio.data import MRMS

DEFAULT_TIME_INTERVAL_MIN = 10
DEFAULT_MAX_OFFSET_MIN = 5
MRMS_VARIABLES = ["refc", "refc_base"]
MRMS_MIN_TIME_UTC = datetime(2020, 10, 14, 0, 0, 0, tzinfo=timezone.utc)
HTTP_RETRY_MAX_ATTEMPTS = 3
HTTP_RETRY_BASE_BACKOFF_S = 0.5


def _is_http_client_error(e: Exception) -> bool:
    # s3fs/aiohttp sometimes raises HTTPClientError with messages like:
    # "An HTTP Client raised an unhandled exception: File descriptor ... is used by transport ..."
    name = type(e).__name__
    msg = str(e)
    return name == "HTTPClientError" or "An HTTP Client raised an unhandled exception" in msg


def _ensure_utc(dt: datetime) -> datetime:
    """Return timezone-aware UTC datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def generate_times(start_utc: datetime, n_steps: int, interval_min: int) -> list[datetime]:
    """Generate UTC-aware target timestamps."""
    start_utc = _ensure_utc(start_utc)
    return [start_utc + timedelta(minutes=interval_min * i) for i in range(n_steps)]


def create_zarr_v3_store(
    store_path: str,
    n_time: int,
    variables: list[str],
    lat_size: int,
    lon_size: int,
) -> zarr.Group:
    """Create/open Zarr v3 store with requested chunking."""
    os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)

    shape = (n_time, lat_size, lon_size)
    chunks = (1, lat_size, lon_size)

    if os.path.exists(store_path):
        root = zarr.open_group(store_path, mode="a")
        return root

    root = zarr.open_group(store_path, mode="w")
    for var in variables:
        root.create(
            var,
            shape=shape,
            chunks=chunks,
            dtype="float32",
            dimension_names=["time", "lat", "lon"],
        )
    root.create(
        "time",
        shape=(n_time,),
        chunks=(n_time,),
        dtype="datetime64[s]",
        dimension_names=["time"],
    )
    # Per-variable resolved times (MRMS may resolve different products to different nearest files).
    for var in variables:
        root.create(
            f"actual_time_{var}",
            shape=(n_time,),
            chunks=(n_time,),
            dtype="datetime64[s]",
            dimension_names=["time"],
        )

    epoch = np.datetime64("1970-01-01T00:00:00", "s")
    root["time"][:] = epoch
    for var in variables:
        root[f"actual_time_{var}"][:] = epoch
    return root


def infer_grid(source: MRMS, sample_time: datetime) -> tuple[int, int]:
    """Infer lat/lon sizes from a sample fetch."""
    da = source([_ensure_utc(sample_time)], [MRMS_VARIABLES[0]])
    return int(da.coords["lat"].shape[0]), int(da.coords["lon"].shape[0])


class WorkManager:
    """Process/rank helper that supports plain runs and SLURM multi-node arrays.

    Global rank/world-size are resolved in this priority:
    1) Explicit launcher vars: RANK/WORLD_SIZE
    2) SLURM rank vars:
       - local rank from SLURM_PROCID
       - local world from SLURM_NPROCS or SLURM_NTASKS
       - optional array expansion via SLURM_ARRAY_TASK_{ID,MIN,MAX}
    3) Fallback single-process: rank=0, world_size=1
    """

    def __init__(self) -> None:
        if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
            self.rank = int(os.environ.get("RANK", "0"))
            self.world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
            return

        slurm_rank = int(os.environ.get("SLURM_PROCID", os.environ.get("PMI_RANK", "0")))
        slurm_world_size = int(
            os.environ.get("SLURM_NPROCS", os.environ.get("SLURM_NTASKS", os.environ.get("PMI_SIZE", "1")))
        )

        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))
        task_min = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "1"))
        task_max = int(os.environ.get("SLURM_ARRAY_TASK_MAX", "1"))

        array_world_size = task_max - task_min + 1
        array_rank = task_id - task_min

        self.rank = array_rank * slurm_world_size + slurm_rank
        self.world_size = max(1, array_world_size * slurm_world_size)

    def split(self, tasks: list[int], shuffle: bool = False, mode: str = "contiguous") -> list[int]:
        """Split task ids across global ranks.

        Why this matters for MRMS:
        - MRMS resolves each *requested time* to a nearest-available file within tolerance.
        - Adjacent requested times can map to the *same* underlying MRMS file.
        - If ranks are assigned interleaved indices (stride split), multiple ranks may try to
          download/read the same file concurrently and can corrupt a shared on-disk cache if the
          underlying downloader isn't fully atomic.

        Modes:
        - contiguous (default): each rank gets a contiguous block -> best cache locality and fewer
          concurrent collisions on the same MRMS file.
        - stride: legacy behavior (rank gets tasks[rank::world_size]).

        When shuffle=True, tasks are shuffled deterministically before splitting.
        """
        local = tasks.copy()
        if shuffle:
            random.seed(0)
            random.shuffle(local)

        if mode == "stride":
            return local[self.rank :: self.world_size]

        if mode != "contiguous":
            raise ValueError(f"Unknown split mode: {mode}")

        n = len(local)
        # Even-ish contiguous partitioning.
        start = (n * self.rank) // self.world_size
        end = (n * (self.rank + 1)) // self.world_size
        return local[start:end]


def wait_for_file(path: str, timeout_s: int = 1800, poll_s: float = 1.0) -> None:
    """Wait until a file exists (simple cross-process barrier)."""
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out waiting for init barrier file: {path}")
        sleep(poll_s)


def wait_for_all_files(
    paths: list[str], timeout_s: int = 7200, poll_s: float = 1.0
) -> None:
    """Wait until all files exist (simple multi-rank completion barrier)."""
    t0 = time.time()
    while True:
        if all(os.path.exists(p) for p in paths):
            return
        if time.time() - t0 > timeout_s:
            missing = [p for p in paths if not os.path.exists(p)]
            raise TimeoutError(
                f"Timed out waiting for completion markers. Missing: {missing[:5]}"
            )
        sleep(poll_s)


def _append_slice_log(
    log_path: str,
    *,
    rank: int,
    world: int,
    time_index: int,
    target_time: datetime,
    status: str,
    actual_time: str,
    elapsed_s: float,
) -> None:
    """Append one per-slice processing record with an inter-process file lock."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    ts_utc = datetime.now(timezone.utc).isoformat()
    line = (
        f"{ts_utc},rank={rank}/{world},index={time_index},"
        f"target={target_time.isoformat()},actual={actual_time},status={status},elapsed_s={elapsed_s:.6f}\n"
    )
    with open(log_path, "a", encoding="utf-8") as fout:
        fcntl.flock(fout.fileno(), fcntl.LOCK_EX)
        fout.write(line)
        fout.flush()
        fcntl.flock(fout.fileno(), fcntl.LOCK_UN)


def fetch_to_zarr(
    source: MRMS,
    time_utc: datetime,
    variables: list[str],
    time_index: int,
    root: zarr.Group,
    skip_if_filled: bool = True,
    rank: int = 0,
    world: int = 1,
    log_path: str | None = None,
) -> bool:
    """Fetch one time slice and write it to Zarr.

    Returns True if this index was written in this call, False if skipped.
    """
    epoch = np.datetime64("1970-01-01T00:00:00", "s")
    # Treat a slice as "filled" only if time and all per-variable actual_time_<var> are present.
    #
    # Why: writes happen to multiple Zarr arrays. If anything fails between writing
    # the data arrays and writing metadata (`time`/`actual_time_<var>`), you can end up with
    # correct-looking values but
    # per-variable actual_time_<var> still at epoch (1970). If we only check `time`,
    # reruns would skip and never backfill the per-variable resolved times.
    filled_time = root["time"][time_index] != epoch
    filled_per_var = True
    for var in variables:
        name = f"actual_time_{var}"
        if name in root and root[name][time_index] == epoch:
            filled_per_var = False
            break
    if skip_if_filled and filled_time and filled_per_var:
        if log_path is not None:
            _append_slice_log(
                log_path,
                rank=rank,
                world=world,
                time_index=time_index,
                target_time=_ensure_utc(time_utc),
                status="skip_filled",
                actual_time="",
                elapsed_s=0.0,
            )
        return False

    t = _ensure_utc(time_utc)
    try:
        http_attempt = 0
        while True:
            http_attempt += 1
            try:
                t_fetch0 = time.perf_counter()
                da = source([t], variables)  # dims: [time, variable, lat, lon]
                elapsed_s = time.perf_counter() - t_fetch0
                break
            except Exception as e:
                if _is_http_client_error(e) and http_attempt < HTTP_RETRY_MAX_ATTEMPTS:
                    backoff = HTTP_RETRY_BASE_BACKOFF_S * (2 ** (http_attempt - 1))
                    jitter = random.uniform(0.0, min(0.25, backoff * 0.25))
                    sleep_s = backoff + jitter
                    print(
                        f"HTTP retry rank={rank}/{world} index={time_index} time={t.isoformat()} "
                        f"attempt={http_attempt}/{HTTP_RETRY_MAX_ATTEMPTS} sleep_s={sleep_s:.2f} "
                        f"err={type(e).__name__}: {e}"
                    )
                    sleep(sleep_s)
                    continue
                raise

        data = da.values.astype(np.float32)  # [1, nvar, y, x]
        for var_idx, var in enumerate(variables):
            root[var][time_index, :, :] = data[0, var_idx, :, :]

        # Avoid noisy NumPy warning about parsing tz-aware datetimes.
# Use integer epoch seconds -> datetime64[s] instead.
        req_np = np.datetime64(int(t.timestamp()), "s")
        root["time"][time_index] = req_np

        # Store per-variable actual times if present in MRMS output coords.
        per_var_actual: dict[str, np.datetime64] = {}
        for var in variables:
            coord = f"actual_time_{var}"
            arr_name = f"actual_time_{var}"
            if arr_name in root:
                if coord in da.coords:
                    val = np.datetime64(da.coords[coord].values[0], "s")
                    root[arr_name][time_index] = val
                    per_var_actual[var] = val
                else:
                    # Do NOT fall back to requested time; that would silently corrupt semantics.
                    # Instead, warn and leave this per-variable actual time invalid (epoch),
                    # so reruns can backfill and consumers can detect missing metadata.
                    print(
                        f"WARNING rank={rank}/{world} index={time_index} time={t.isoformat()} "
                        f"missing coord {coord} in MRMS output; leaving {arr_name}=epoch"
                    )
                    root[arr_name][time_index] = epoch
                    per_var_actual[var] = epoch

        if log_path is not None:
            _append_slice_log(
                log_path,
                rank=rank,
                world=world,
                time_index=time_index,
                target_time=t,
                status="ok",
                actual_time=";".join(f"{v}={per_var_actual.get(v, epoch)}" for v in variables),
                elapsed_s=elapsed_s,
            )
        return True
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        print(
            f"Error rank={rank}/{world} index={time_index} time={t.isoformat()} "
            f"-> {type(e).__name__}: {msg}"
        )
        for var in variables:
            root[var][time_index, :, :] = np.nan
        # Keep the requested target time even on error (so the dataset time axis remains meaningful).
        # Use per-variable actual_time_<var>=epoch to indicate we did not successfully resolve a file.
        root["time"][time_index] = np.datetime64(int(t.timestamp()), "s")
        for var in variables:
            name = f"actual_time_{var}"
            if name in root:
                root[name][time_index] = epoch
        if log_path is not None:
            _append_slice_log(
                log_path,
                rank=rank,
                world=world,
                time_index=time_index,
                target_time=t,
                status=f"error:{type(e).__name__}",
                actual_time=str(epoch),
                elapsed_s=0.0,
            )
        return True


def run_main(args: argparse.Namespace) -> None:
    """Full-year download path (rank-aware multiprocess execution)."""
    wm = WorkManager()
    rank, world = wm.rank, wm.world_size
    year = args.year
    if args.start_utc is None:
        start = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    else:
        # Accept ISO8601 strings; if no tzinfo, treat as UTC.
        start = _ensure_utc(datetime.fromisoformat(args.start_utc))

    if args.n_steps is None:
        days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        n_steps = days * 24 * (60 // args.time_interval)
    else:
        n_steps = int(args.n_steps)
        if n_steps <= 0:
            raise ValueError("--n-steps must be > 0")

    times_utc = generate_times(start, n_steps, args.time_interval)

    zarr_name = args.zarr_name or f"{year}.zarr"
    out_path = os.path.join(args.output_dir, zarr_name)
    barrier_file = str(Path(args.output_dir) / f".mrms_{year}.init.done")
    log_path = str(Path(args.output_dir) / f"processed_time_slices_{year}.txt")
    done_dir = Path(args.output_dir) / f".mrms_{year}.done_ranks"

    # Rank 0 initializes the store once, others wait.
    if rank == 0:
        source0 = MRMS(
            max_offset_minutes=args.max_offset_minutes,
            cache=args.cache,
            verbose=True,
            max_workers=args.max_workers,
        )
        try:
            lat_size, lon_size = infer_grid(source0, times_utc[0])
        except Exception as e:  # noqa: BLE001
            print(f"Grid inference failed; using fallback 3500x7000: {e}")
            lat_size, lon_size = 3500, 7000
        create_zarr_v3_store(out_path, n_steps, MRMS_VARIABLES, lat_size, lon_size)
        Path(barrier_file).parent.mkdir(parents=True, exist_ok=True)
        Path(barrier_file).write_text("ok")
        done_dir.mkdir(parents=True, exist_ok=True)
        for p in done_dir.glob("rank_*.done"):
            p.unlink()
        with open(log_path, "a", encoding="utf-8") as fout:
            if fout.tell() == 0:
                fout.write("event_utc,rank,index,target,actual,status,elapsed_s\n")
    else:
        wait_for_file(barrier_file)

    root = zarr.open_group(out_path, mode="a")
    source = MRMS(
        max_offset_minutes=args.max_offset_minutes,
        cache=args.cache,
        verbose=True,
        max_workers=args.max_workers,
    )

    # Global-rank split (optionally shuffled for load balancing).
    local_indices = wm.split(list(range(n_steps)), shuffle=False, mode=args.split_mode)
    # Ensure every rank emits at least one line early (helps when per-task logs are mostly empty).
    if local_indices:
        first_i, last_i = local_indices[0], local_indices[-1]
        print(
            f"[rank {rank}] start world={world} split_mode={args.split_mode} "
            f"local_indices={len(local_indices)} first_index={first_i} first_target={times_utc[first_i].isoformat()} "
            f"last_index={last_i} last_target={times_utc[last_i].isoformat()}"
        )
    else:
        print(
            f"[rank {rank}] start world={world} split_mode={args.split_mode} local_indices=0"
        )
    t0 = time.time()
    wrote = 0
    for i in tqdm(local_indices, desc=f"Rank {rank}/{world}", disable=(rank != 0)):
        wrote += int(
            fetch_to_zarr(
                source,
                times_utc[i],
                MRMS_VARIABLES,
                i,
                root,
                skip_if_filled=True,
                rank=rank,
                world=world,
                log_path=log_path,
            )
        )
    t1 = time.time()
    print(
        f"[rank {rank}] processed={len(local_indices)} wrote={wrote} "
        f"elapsed={t1 - t0:.2f}s"
    )

    # Write completion marker for this rank.
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / f"rank_{rank}.done").write_text(
        f"rank={rank}\nworld={world}\nfinished_utc={datetime.now(timezone.utc).isoformat()}\n"
    )

    # Rank 0 consolidates metadata after all ranks are done.
    if rank == 0:
        done_paths = [str(done_dir / f"rank_{r}.done") for r in range(world)]
        wait_for_all_files(done_paths, timeout_s=max(7200, world * 120), poll_s=1.0)
        try:
            zarr.consolidate_metadata(out_path)
            print(f"[rank 0] Consolidated zarr metadata for {out_path}")
        except Exception as e:  # noqa: BLE001
            print(f"[rank 0] WARNING: failed to consolidate metadata: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MRMS fields to Zarr v3 (UTC-only)")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="./mrms_data")
    parser.add_argument(
        "--zarr-name",
        type=str,
        default=None,
        help="Optional output Zarr directory name (e.g. 2020_retry.zarr). Defaults to <year>.zarr.",
    )
    parser.add_argument("--time-interval", type=int, default=DEFAULT_TIME_INTERVAL_MIN)
    parser.add_argument("--max-offset-minutes", type=float, default=DEFAULT_MAX_OFFSET_MIN)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument(
        "--start-utc",
        type=str,
        default=None,
        help="Optional ISO8601 UTC start time for benchmarking (e.g. 2021-01-15T00:00:00+00:00). Defaults to Jan 1 of --year.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Optional number of time steps to run (for benchmarking). Defaults to full year.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="contiguous",
        choices=["contiguous", "stride"],
        help="How to split time indices across ranks. 'contiguous' reduces cache collisions; 'stride' is legacy.",
    )
    run_main(parser.parse_args())

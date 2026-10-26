import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import zarr

from earth2studio.data import GFS, Random
from earth2studio.io import ZarrBackend
from earth2studio.models.px import Persistence
from earth2studio.run import deterministic
from od_experiments.earth2_001.config import load_config
from od_experiments.earth2_001.provenance import (
    compute_sha256,
    create_provenance_manifest,
)
from od_experiments.earth2_001.validate import evaluate_forecast
from od_experiments.earth2_001.visualize import plot_field


def run_forecast(
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    device: str = "cpu",
    override_source: str | None = None,  # "gfs" or "random"
) -> dict[str, Any]:
    """
    Executes a genuine weather forecast workflow, reading output arrays directly
    from Earth2Studio's Zarr store, saving verified tensors and plots, and
    distinguishing between SYNTHETIC_SMOKE_TEST and GENUINE_ATMOSPHERIC_EXPERIMENT.
    """
    start_time = time.time()
    config = load_config(config_path)

    out_dir = Path(
        output_dir or config.get("output", {}).get("directory", "outputs/earth2_001")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    data_source_name = (
        override_source or config.get("data_source", {}).get("name", "gfs").lower()
    )
    variables = config.get("simulation", {}).get("variables", ["t2m"])
    n_steps = config.get("simulation", {}).get("n_steps", 1)
    lead_time_hours = config.get("simulation", {}).get("lead_time_hours", 6)
    init_time_str = config.get("simulation", {}).get(
        "initial_time", "2024-01-01T00:00:00"
    )
    init_time = datetime.fromisoformat(init_time_str)

    # Clean existing Zarr store in target directory to avoid stale array collisions
    zarr_path = out_dir / "forecast.zarr"
    if zarr_path.exists():
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
        else:
            zarr_path.unlink()

    from collections import OrderedDict

    # Define standard domain coordinates
    dc = OrderedDict(
        {
            "lat": np.linspace(90, -90, 721),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )

    if data_source_name == "gfs":
        classification = "GENUINE_ATMOSPHERIC_EXPERIMENT"
        data_source_type = "HISTORICAL_GFS"
        print(
            f"[i] Using REAL NOAA GFS Historical Data from {init_time.isoformat()} UTC..."
        )
        data_source = GFS()
        verification_time = init_time + timedelta(hours=lead_time_hours * n_steps)
    else:
        classification = "SYNTHETIC_SMOKE_TEST"
        data_source_type = "SYNTHETIC_RANDOM"
        print("[i] Using SYNTHETIC Random Field Data (Smoke Test Mode)...")
        data_source = Random(domain_coords=dc)
        verification_time = None

    # Initialize Persistence baseline model
    model = Persistence(variable=variables, domain_coords=dc, history=1)

    # Execute deterministic inference with Zarr backend
    io_backend = ZarrBackend(str(zarr_path))
    io_result = deterministic(
        time=[init_time],
        nsteps=n_steps,
        prognostic=model,
        data=data_source,
        io=io_backend,
        device=torch.device(device),
    )

    # Extract actual predicted array directly from the generated Zarr Store
    z_root = zarr.open(str(zarr_path), mode="r")

    # Earth2Studio ZarrBackend stores arrays by variable name or under root
    artifacts: list[dict[str, Any]] = []
    forecast_arrays: dict[str, np.ndarray] = {}

    for var in variables:
        if var in z_root:
            z_arr = z_root[var]
            # Shape is typically (time, lead_time, lat, lon)
            arr = np.array(z_arr)
            # Take the final predicted lead time step
            if arr.ndim >= 4:
                final_step = arr[0, -1]
            elif arr.ndim == 3:
                final_step = arr[-1]
            else:
                final_step = arr
            forecast_arrays[var] = final_step

            # Save extracted numpy array
            npy_file = out_dir / f"forecast_{var}.npy"
            np.save(npy_file, final_step)
            artifacts.append(
                {
                    "path": str(npy_file),
                    "sha256": compute_sha256(npy_file),
                    "bytes": npy_file.stat().st_size,
                }
            )

            # Generate scientific visualization plot
            plot_file = out_dir / f"forecast_{var}.png"
            title = f"{var.upper()} Forecast (+{lead_time_hours*n_steps}h from {init_time_str} UTC)"
            plot_field(
                final_step,
                title,
                plot_file,
                variable_name=var,
                units="K" if "t" in var else "m/s",
            )
            artifacts.append(
                {
                    "path": str(plot_file),
                    "sha256": compute_sha256(plot_file),
                    "bytes": plot_file.stat().st_size,
                }
            )

    # Fetch reference truth for genuine atmospheric verification if using GFS
    reference_arrays: dict[str, np.ndarray] | None = None
    verification_ref_desc = None

    if data_source_name == "gfs" and verification_time:
        try:
            print(
                f"[i] Fetching genuine reference analysis from NOAA GFS at {verification_time.isoformat()} UTC..."
            )
            gfs_ref_source = GFS()
            ref_da = gfs_ref_source(time=verification_time, variable=variables)
            reference_arrays = {}
            for idx, var in enumerate(variables):
                # DataArray shape (time, variable, lat, lon)
                ref_arr = np.array(ref_da.sel(variable=var).values)
                if ref_arr.ndim > 2:
                    ref_arr = ref_arr[0]
                reference_arrays[var] = ref_arr

                ref_npy = out_dir / f"reference_{var}.npy"
                np.save(ref_npy, ref_arr)
                artifacts.append(
                    {
                        "path": str(ref_npy),
                        "sha256": compute_sha256(ref_npy),
                        "bytes": ref_npy.stat().st_size,
                    }
                )
            verification_ref_desc = (
                f"NOAA GFS Analysis at {verification_time.isoformat()} UTC"
            )
        except Exception as e:
            print(f"[!] Warning: Could not retrieve reference analysis from GFS ({e}).")
            reference_arrays = None
            verification_ref_desc = f"Failed to fetch GFS reference: {str(e)}"

    # Evaluate scientific validation metrics
    lats = np.array(dc["lat"])
    val_results = evaluate_forecast(
        forecast_data=forecast_arrays,
        reference_data=reference_arrays,
        lats=lats,
        output_dir=out_dir,
    )

    val_file = out_dir / "validation_report.json"
    artifacts.append(
        {
            "path": str(val_file),
            "sha256": compute_sha256(val_file),
            "bytes": val_file.stat().st_size,
        }
    )

    exec_time = time.time() - start_time

    # Generate cryptographic provenance manifest
    manifest = create_provenance_manifest(
        experiment_id=config.get("experiment_id", "OD-EARTH2-001"),
        configuration=config,
        artifacts=artifacts,
        execution_outcome="EXECUTION_SUCCESS",
        validation_status=val_results.get("validation_status", "NOT_VALIDATED"),
        classification=classification,
        model_type="BASELINE_PERSISTENCE",
        data_source_type=data_source_type,
        verification_reference=verification_ref_desc,
        execution_time_seconds=round(exec_time, 3),
        device=device,
        output_dir=out_dir,
    )

    return {
        "status": "SUCCESS",
        "classification": classification,
        "output_directory": str(out_dir),
        "manifest": manifest,
        "validation": val_results,
    }

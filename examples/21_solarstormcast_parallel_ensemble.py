 

import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from loguru import logger

os.environ["EARTH2STUDIO_CACHE"] = "/workspace/earth2studio_cache"




# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Configuration for ensemble inference."""
    solar_package_path: str = "/output/solar_package_3km_fd_goes6km"
    stormscope_model_name: str = "6km_10min_natten_pure_obs_zenith_6steps" #"3km_10min_natten_pure_obs_cos_zenith_input_eoe" 6km_10min_natten_pure_obs_zenith_6steps
    output_dir: str = "/output/solarstormcast_ensembles"
    
    num_steps: int = 12  # 12 steps × 10 min = 2 hours
    step_minutes: int = 10
    ensemble_per_gpu: int = 3
    
    compile_models: bool = True
    stormscope_num_steps: int = 96
    solar_num_steps: int = 24


# =============================================================================
# Setup Functions
# =============================================================================

def setup_performance():
    """Configure PyTorch for optimal inference performance."""
    # Enable cudnn benchmarking for faster convolutions
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Enable TF32 for faster matrix multiplications on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def initialize_distributed():
    """Initialize distributed training and return DistributedManager."""
    from physicsnemo.distributed import DistributedManager
    
    DistributedManager.initialize()
    dm = DistributedManager()
    
    logger.info(f"Rank {dm.rank} of {dm.world_size} on device {dm.device}")
    
    # Seed based on rank for ensemble diversity
    torch.manual_seed(dm.rank)
    torch.cuda.manual_seed(dm.rank)
    
    return dm


def load_models(dm, config: Config):
    """Load StormScope and SolarStormCast models."""
    from earth2studio.data import GFS_FX
    from earth2studio.models.auto import Package
    from earth2studio.models.px import SolarStormCast
    from earth2studio.models.px.stormscope import StormScopeGOES
    
    t_start = time.perf_counter()
    
    # StormScope - rank 0 downloads first
    stormscope_package = StormScopeGOES.load_default_package()
    if dm.rank == 0:
        stormscope = StormScopeGOES.load_model(
            package=stormscope_package,
            conditioning_data_source=GFS_FX(),
            model_name=config.stormscope_model_name,
        )
    
    dist.barrier()
    if dm.rank != 0:
        stormscope = StormScopeGOES.load_model(
            package=stormscope_package,
            conditioning_data_source=GFS_FX(),
            model_name=config.stormscope_model_name,
        )
    
    stormscope = stormscope.to(dm.device).eval()
    stormscope.sampler_args = {"num_steps": config.stormscope_num_steps}
    
    # SolarStormCast
    solar_package = Package(config.solar_package_path)
    solar_model = SolarStormCast.load_model(solar_package, sampler_type="stochastic")
    solar_model.sampler_args = {
        "num_steps": config.solar_num_steps,
        "sigma_min": 0.001,
        "sigma_max": 800,
        "rho": 7,
        "S_churn": 10,
        "S_noise": 1,
    }
    solar_model = solar_model.to(dm.device).eval()
    
    # Optionally compile models
    if config.compile_models:
        stormscope, solar_model = compile_models(dm, stormscope, solar_model)
    
    t_load = time.perf_counter() - t_start
    if dm.rank == 0:
        logger.info(f"Models loaded in {t_load:.2f}s")
    
    return stormscope, solar_model


def compile_models(dm, stormscope, solar_model):
    """Compile models with torch.compile for faster inference."""
    if dm.rank == 0:
        logger.info("Compiling models with torch.compile...")
    
    compile_options = {
        "mode": "reduce-overhead",
        "fullgraph": False,
    }
    
    if hasattr(stormscope, "model") and stormscope.model is not None:
        stormscope.model = torch.compile(stormscope.model, **compile_options)
    
    if hasattr(solar_model, "regression_model") and solar_model.regression_model is not None:
        solar_model.regression_model = torch.compile(solar_model.regression_model, **compile_options)
    if hasattr(solar_model, "model") and solar_model.model is not None:
        solar_model.model = torch.compile(solar_model.model, **compile_options)
    
    if dm.rank == 0:
        logger.info("Models compiled")
    
    return stormscope, solar_model


def compute_domain_intersection(dm, stormscope, solar_model, margin_deg: float = 0.5):
    """
    Compute intersection between StormScope and SolarStormCast domains.
    Subsets SolarStormCast to only operate on the overlapping region.
    Uses StormScope's valid_mask to find the actual valid region.
    
    Parameters
    ----------
    margin_deg : float
        Margin in degrees to shrink the domain to ensure all pixels have valid
        StormScope data after interpolation. Default 0.5 degrees.
    """
    # Get lat/lon from both models
    stormscope_lat = stormscope.latitudes.cpu().numpy()
    stormscope_lon = stormscope.longitudes.cpu().numpy() - 360  # Convert 0-360 to -180 to 180
    stormscope_valid = stormscope.valid_mask.cpu().numpy()
    
    solar_lat = solar_model.latitudes.cpu().numpy()
    solar_lon = solar_model.longitudes.cpu().numpy()
    solar_valid = solar_model.valid_mask.cpu().numpy()
    
    if dm.rank == 0:
        logger.info(f"StormScope domain: {stormscope_lat.shape}, "
                    f"lat [{stormscope_lat.min():.2f}, {stormscope_lat.max():.2f}], "
                    f"lon [{stormscope_lon.min():.2f}, {stormscope_lon.max():.2f}]")
        logger.info(f"StormScope valid pixels: {stormscope_valid.sum()} / {stormscope_valid.size}")
        logger.info(f"SolarStormCast domain: {solar_lat.shape}, "
                    f"lat [{solar_lat.min():.2f}, {solar_lat.max():.2f}], "
                    f"lon [{solar_lon.min():.2f}, {solar_lon.max():.2f}]")
        logger.info(f"SolarStormCast valid pixels: {solar_valid.sum()} / {solar_valid.size}")
    
    # Find bounds of VALID pixels in StormScope (not the full domain)
    stormscope_lat_valid = stormscope_lat[stormscope_valid]
    stormscope_lon_valid = stormscope_lon[stormscope_valid]
    
    if dm.rank == 0:
        logger.info(f"StormScope valid region: "
                    f"lat [{stormscope_lat_valid.min():.2f}, {stormscope_lat_valid.max():.2f}], "
                    f"lon [{stormscope_lon_valid.min():.2f}, {stormscope_lon_valid.max():.2f}]")
    
    # Find common lat/lon bounds using StormScope's VALID region
    # Add margin to ensure interpolation doesn't reach outside valid region
    lat_min = max(stormscope_lat_valid.min() + margin_deg, solar_lat.min())
    lat_max = min(stormscope_lat_valid.max() - margin_deg, solar_lat.max())
    lon_min = max(stormscope_lon_valid.min() + margin_deg, solar_lon.min())
    lon_max = min(stormscope_lon_valid.max() - margin_deg, solar_lon.max())
    
    if dm.rank == 0:
        logger.info(f"Common region (with {margin_deg}° margin): lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}]")
    
    # Find slice in SolarStormCast grid that corresponds to common region
    mask = (solar_lat >= lat_min) & (solar_lat <= lat_max) & \
           (solar_lon >= lon_min) & (solar_lon <= lon_max)
    
    valid_rows = np.any(mask, axis=1)
    valid_cols = np.any(mask, axis=0)
    
    y_start = np.argmax(valid_rows)
    y_end = len(valid_rows) - np.argmax(valid_rows[::-1])
    x_start = np.argmax(valid_cols)
    x_end = len(valid_cols) - np.argmax(valid_cols[::-1])
    
    y_slice = slice(y_start, y_end)
    x_slice = slice(x_start, x_end)
    
    if dm.rank == 0:
        logger.info(f"Subsetting SolarStormCast: y={y_slice}, x={x_slice}")
        logger.info(f"New shape: ({y_end - y_start}, {x_end - x_start})")
    
    # Apply subset to solar model
    solar_model.subset_domain(y_slice, x_slice)
    
    # Verify new domain is within StormScope VALID bounds
    new_lat = solar_model.latitudes.cpu().numpy()
    new_lon = solar_model.longitudes.cpu().numpy()
    
    # Check that subsetted solar domain is inside StormScope's VALID region
    solar_inside_stormscope = (
        new_lat.min() >= stormscope_lat_valid.min() and
        new_lat.max() <= stormscope_lat_valid.max() and
        new_lon.min() >= stormscope_lon_valid.min() and
        new_lon.max() <= stormscope_lon_valid.max()
    )
    
    if dm.rank == 0:
        logger.info(f"SolarStormCast after subset: {new_lat.shape}, "
                    f"lat [{new_lat.min():.2f}, {new_lat.max():.2f}], "
                    f"lon [{new_lon.min():.2f}, {new_lon.max():.2f}]")
        
        if solar_inside_stormscope:
            logger.info("✓ Solar domain is inside StormScope valid region")
        else:
            logger.warning("⚠ Solar domain extends beyond StormScope valid region!")
            logger.warning(f"  StormScope valid bounds: lat [{stormscope_lat_valid.min():.2f}, {stormscope_lat_valid.max():.2f}], "
                          f"lon [{stormscope_lon_valid.min():.2f}, {stormscope_lon_valid.max():.2f}]")
            logger.warning(f"  Solar bounds:            lat [{new_lat.min():.2f}, {new_lat.max():.2f}], "
                          f"lon [{new_lon.min():.2f}, {new_lon.max():.2f}]")


def setup_interpolators(dm, stormscope, solar_model):
    """Build interpolators for data regridding."""
    from earth2studio.data import GOES, GFS_FX
    
    t_start = time.perf_counter()
    
    # Rank 0 gets grid info first (may involve downloads), then broadcasts
    # Note: NCCL backend requires CUDA tensors for broadcast, so we move to GPU
    if dm.rank == 0:
        goes_lat, goes_lon = GOES.grid(satellite="goes16", scan_mode="C")
        goes_lat_tensor = torch.from_numpy(goes_lat).float().to(dm.device)
        goes_lon_tensor = torch.from_numpy(goes_lon).float().to(dm.device)
        grid_shape = torch.tensor(goes_lat.shape, dtype=torch.long, device=dm.device)
    else:
        grid_shape = torch.zeros(2, dtype=torch.long, device=dm.device)
    
    dist.broadcast(grid_shape, src=0)
    
    if dm.rank != 0:
        goes_lat_tensor = torch.zeros(grid_shape.tolist(), dtype=torch.float32, device=dm.device)
        goes_lon_tensor = torch.zeros(grid_shape.tolist(), dtype=torch.float32, device=dm.device)
    
    dist.broadcast(goes_lat_tensor, src=0)
    dist.broadcast(goes_lon_tensor, src=0)
    
    # Move back to CPU for numpy conversion
    goes_lat = goes_lat_tensor.cpu().numpy()
    goes_lon = goes_lon_tensor.cpu().numpy()
    
    # Build interpolators
    stormscope.build_input_interpolator(goes_lat, goes_lon, max_dist_km=12.0)
    stormscope.build_conditioning_interpolator(GFS_FX.GFS_LAT, GFS_FX.GFS_LON, max_dist_km=26.0)
    
    solar_model.build_input_interpolator(goes_lat, goes_lon, max_dist_km=12.0, name="goes_raw")
    solar_model.build_input_interpolator(
        stormscope.latitudes.cpu().numpy(),
        stormscope.longitudes.cpu().numpy(),
        max_dist_km=12.0,
        name="obscast"
    )
    
    t_interp = time.perf_counter() - t_start
    if dm.rank == 0:
        logger.info(f"Interpolators built in {t_interp:.2f}s")
    
    dist.barrier()
    
    return goes_lat, goes_lon


def fetch_goes_data(dm, stormscope, init_time: datetime, config: Config, goes_lat: np.ndarray, goes_lon: np.ndarray):
    """Fetch GOES data on rank 0 and broadcast to all ranks."""
    from earth2studio.data import GOES, fetch_data
    
    t_start = time.perf_counter()
    
    goes_satellite = "goes19" if init_time >= datetime(2025, 4, 7, 20, 0, 0) else "goes16"
    obscast_vars = stormscope.input_coords()["variable"]
    in_coords = stormscope.input_coords()
    lead_times = in_coords["lead_time"]
    
    if dm.rank == 0:
        goes = GOES(satellite=goes_satellite, scan_mode="C")
        x_single, _ = fetch_data(
            goes,
            time=[np.datetime64(init_time)],
            variable=np.array(obscast_vars),
            lead_time=lead_times,
            device=dm.device,
        )
        # Ensure [T, L, C, H, W]
        if x_single.dim() == 4:
            x_single = x_single.unsqueeze(0)
        if x_single.dim() != 5:
            raise ValueError(f"Expected 4D or 5D GOES data, got shape {tuple(x_single.shape)}")

        target_lead_len = len(lead_times)
        if target_lead_len > 1 and x_single.shape[1] == 1:
            x_single = x_single.repeat(1, target_lead_len, 1, 1, 1)

        shape_5d = torch.tensor(x_single.shape, device=dm.device, dtype=torch.long)
        x = x_single.unsqueeze(0).repeat(config.ensemble_per_gpu, 1, 1, 1, 1, 1).to(dtype=torch.float32)
        logger.info(f"x.shape: {x.shape}")
    else:
        shape_5d = torch.zeros(5, device=dm.device, dtype=torch.long)

    dist.broadcast(shape_5d, src=0)

    if dm.rank != 0:
        x = torch.zeros(
            config.ensemble_per_gpu,
            int(shape_5d[0]),
            int(shape_5d[1]),
            int(shape_5d[2]),
            int(shape_5d[3]),
            int(shape_5d[4]),
            dtype=torch.float32,
            device=dm.device,
        )

    dist.broadcast(x, src=0)
    
    # Build coords with GOES lat/lon (must match the data shape 1500x2500)
    # This is critical: stormscope.input_coords() has model lat/lon (1024x1792)
    # but our data is in GOES coords (1500x2500) - use coords from fetch_data pattern
    from earth2studio.utils.type import CoordSystem
    x_coords = CoordSystem({
        "batch": np.arange(config.ensemble_per_gpu),
        "time": np.array([np.datetime64(init_time)]),
        "lead_time": in_coords["lead_time"],
        "variable": np.array(obscast_vars),
        "lat": goes_lat,
        "lon": goes_lon,
    })
    
    t_fetch = time.perf_counter() - t_start
    if dm.rank == 0:
        logger.info(f"  Data fetch + broadcast: {t_fetch:.2f}s")
    
    return x, x_coords


def run_forecast(
    dm,
    stormscope,
    solar_model,
    x: torch.Tensor,
    x_coords,
    init_time: datetime,
    config: Config,
    local_goes: torch.Tensor,
    local_ghi: torch.Tensor,
) -> List[datetime]:
    """Run ensemble forecast for one initialization time."""
    
    H, W = local_ghi.shape[-2:]
    
    # Interpolate initial GOES to StormScope grid, then to solar model grid
    x_on_stormscope, _ = stormscope.prep_input(x, x_coords)
    goes_interp = solar_model.interpolate(x_on_stormscope, interpolator="obscast")
    if goes_interp.dim() == 6:
        goes_2d = goes_interp[:, 0, -1]
    elif goes_interp.dim() == 5:
        goes_2d = goes_interp[:, -1]
    else:
        goes_2d = goes_interp
    
    if dm.rank == 0:
        logger.info(f"goes_2d.shape: {goes_2d.shape}")
    
    # StormScope handles interpolation internally via prep_input
    # Pass raw GOES data directly (following 20_stormscope_goes_example.py pattern)
    y, y_coords = x, x_coords
    
    if dm.rank == 0:
        logger.info(f"StormScope input shape: {y.shape}")
    
    # Reset storage
    local_goes.zero_()
    local_ghi.zero_()
    valid_times = [init_time]
    
    # Step 0: Initial prediction from raw GOES
    t_step0_start = time.perf_counter()
    
    solar_state = torch.zeros(
        config.ensemble_per_gpu, len(solar_model.variables), H, W, device=dm.device
    )
    conditioning = solar_model.build_conditioning(goes_2d, init_time)
    
    with torch.inference_mode():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            solar_output = solar_model(solar_state, conditioning, first_step=True)
    
    torch.cuda.synchronize()
    t_step0 = time.perf_counter() - t_step0_start
    
    local_goes[:, 0] = goes_2d
    local_ghi[:, 0] = solar_output[:, 0, 0]
    solar_state = solar_output
    
    if dm.rank == 0:
        logger.info(f"  Step 0 (StormScope-interpolated GOES): {t_step0:.2f}s")
    
    # Iterative steps
    step_times = []
    
    with torch.inference_mode():
        for step in range(config.num_steps):
            t_step_start = time.perf_counter()
            
            forecast_time = init_time + timedelta(minutes=(step + 1) * config.step_minutes)
            valid_times.append(forecast_time)
            
            # StormScope forecasts GOES (handles interpolation internally)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                y_pred, y_pred_coords = stormscope(y, y_coords)
            
            # Interpolate to solar model grid
            goes_interp = solar_model.interpolate(y_pred, interpolator="obscast")
            if goes_interp.dim() == 6:
                goes_2d = goes_interp[:, 0, 0]
            elif goes_interp.dim() == 5:
                goes_2d = goes_interp[:, 0]
            else:
                goes_2d = goes_interp
            
            # Build conditioning (SZA computed for forecast_time)
            conditioning = solar_model.build_conditioning(goes_2d, forecast_time)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                solar_output = solar_model(solar_state, conditioning)
            
            # Sync every 4 steps for timing
            if step == config.num_steps - 1 or (step + 1) % 4 == 0:
                torch.cuda.synchronize()
            
            t_step = time.perf_counter() - t_step_start
            step_times.append(t_step)
            
            local_goes[:, step + 1] = goes_2d
            local_ghi[:, step + 1] = solar_output[:, 0, 0]
            solar_state = solar_output
            
            y, y_coords = stormscope.next_input(y_pred, y_pred_coords, y, y_coords)
            
            if dm.rank == 0:
                logger.info(f"  Step {step + 1}/{config.num_steps}: {t_step:.2f}s | {forecast_time.strftime('%H:%M')}")
            
    
    if dm.rank == 0:
        avg_step_time = np.mean(step_times)
        logger.info(f"  Avg step time: {avg_step_time:.2f}s (total: {sum(step_times):.2f}s)")
    
    return valid_times


def gather_and_save(
    dm,
    local_goes: torch.Tensor,
    local_ghi: torch.Tensor,
    valid_times: List[datetime],
    init_time: datetime,
    config: Config,
    lat_out: np.ndarray,
    lon_out: np.ndarray,
):
    """Gather results from all ranks and save to zarr on rank 0."""
    
    torch.cuda.synchronize()
    
    t_gather_start = time.perf_counter()
    dist.barrier()
    
    total_ensemble = dm.world_size * config.ensemble_per_gpu
    
    goes_list = [torch.zeros_like(local_goes) for _ in range(dm.world_size)] if dm.rank == 0 else None
    ghi_list = [torch.zeros_like(local_ghi) for _ in range(dm.world_size)] if dm.rank == 0 else None
    
    dist.gather(local_goes, goes_list, dst=0)
    dist.gather(local_ghi, ghi_list, dst=0)
    
    t_gather = time.perf_counter() - t_gather_start
    if dm.rank == 0:
        logger.info(f"  Gather: {t_gather:.2f}s")
    
    if dm.rank == 0:
        t_save_start = time.perf_counter()
        
        all_goes = torch.cat(goes_list, dim=0).cpu().numpy()
        all_ghi = torch.cat(ghi_list, dim=0).cpu().numpy()
        
        valid_times_np = np.array([np.datetime64(t) for t in valid_times])
        n_goes_channels = all_goes.shape[2]
        
        ds = xr.Dataset(
            {
                "ghi": (["ensemble", "time", "y", "x"], all_ghi),
                "goes": (["ensemble", "time", "channel", "y", "x"], all_goes),
                "latitude": (["y", "x"], lat_out),
                "longitude": (["y", "x"], lon_out),
            },
            coords={
                "time": valid_times_np,
                "ensemble": np.arange(total_ensemble),
                "channel": np.arange(n_goes_channels),
            },
            attrs={"init_time": str(init_time), "model": "SolarStormCast"},
        )
        
        res_match = re.search(r"(\d+(?:\.\d+)?)km", config.stormscope_model_name)
        res_tag = f"{res_match.group(1)}km" if res_match else "resUnknown"
        output_file = os.path.join(
            config.output_dir,
            f"solarstormcast_ensemble_{init_time.strftime('%Y%m%d_%H%M')}_{res_tag}_{config.stormscope_num_steps}.zarr"
        )
        ds.to_zarr(output_file, mode="w")
        
        t_save = time.perf_counter() - t_save_start
        logger.info(f"  Save: {t_save:.2f}s")
        logger.info(f"  Saved: {output_file} (ensemble={total_ensemble})")
    
    dist.barrier()


# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point for ensemble inference."""
    
    # Configuration
    config = Config()
    init_times = [
        datetime(2024, 5, 5, 0, 0, 0),
        datetime(2024, 5, 5, 6, 0, 0),
        datetime(2024, 5, 5, 12, 0, 0),
        datetime(2024, 5, 5, 18, 0, 0),
        datetime(2024, 5, 6, 0, 0, 0),
        datetime(2024, 5, 6, 6, 0, 0),
        datetime(2024, 5, 6, 12, 0, 0),
        datetime(2024, 5, 6, 18, 0, 0),
        datetime(2024, 5, 7, 0, 0, 0),
        datetime(2024, 5, 7, 6, 0, 0),
        datetime(2024, 5, 7, 12, 0, 0),
        datetime(2024, 5, 7, 18, 0, 0),
        datetime(2024, 5, 8, 0, 0, 0),
        datetime(2024, 5, 8, 6, 0, 0),
        datetime(2024, 5, 8, 12, 0, 0),
        datetime(2024, 5, 8, 18, 0, 0),
        datetime(2024, 5, 9, 0, 0, 0),
        datetime(2024, 5, 9, 6, 0, 0),
        datetime(2024, 5, 9, 12, 0, 0),
        datetime(2024, 5, 9, 18, 0, 0),
        # datetime(2024, 5, 10, 0, 0, 0),
        # datetime(2024, 5, 10, 6, 0, 0),
        # datetime(2024, 5, 10, 12, 0, 0),
        # datetime(2024, 5, 10, 18, 0, 0),
        # datetime(2024, 5, 11, 0, 0, 0),
        # datetime(2024, 5, 11, 6, 0, 0),
        # datetime(2024, 5, 11, 12, 0, 0),
        # datetime(2024, 5, 11, 18, 0, 0),
        # datetime(2024, 5, 12, 0, 0, 0),
        # datetime(2024, 5, 12, 6, 0, 0),
        # datetime(2024, 5, 12, 12, 0, 0),
        # datetime(2024, 5, 12, 18, 0, 0),
        # datetime(2024, 5, 13, 0, 0, 0),
        # datetime(2024, 5, 13, 6, 0, 0),
        # datetime(2024, 5, 13, 12, 0, 0),
        # datetime(2024, 5, 13, 18, 0, 0),
        # datetime(2024, 5, 14, 0, 0, 0),
        # datetime(2024, 5, 14, 6, 0, 0),
        # datetime(2024, 5, 14, 12, 0, 0),
        # datetime(2024, 5, 14, 18, 0, 0),
        # datetime(2024, 5, 15, 0, 0, 0),
        # datetime(2024, 5, 15, 6, 0, 0),
        # datetime(2024, 5, 15, 12, 0, 0),
        # datetime(2024, 5, 15, 18, 0, 0),
    ]
    
    # Setup
    setup_performance()
    dm = initialize_distributed()
    
    if dm.rank == 0:
        logger.info(f"World size: {dm.world_size}")
        logger.info(f"Ensemble per GPU: {config.ensemble_per_gpu}")
        logger.info(f"Total ensemble size: {dm.world_size * config.ensemble_per_gpu}")
        os.makedirs(config.output_dir, exist_ok=True)
    
    dist.barrier()
    
    # Load models
    stormscope, solar_model = load_models(dm, config)
    
    # Use full SolarStormCast domain (no subsetting)
    lat_out = solar_model.latitudes.cpu().numpy()
    lon_out = solar_model.longitudes.cpu().numpy()
    H, W = lat_out.shape
    n_goes_channels = len(stormscope.variables)
    
    if dm.rank == 0:
        logger.info(f"Grid after subset: {H}x{W}")
    
    # Setup interpolators (uses subsetted solar model grid)
    goes_lat, goes_lon = setup_interpolators(dm, stormscope, solar_model)
    
    # Pre-allocate storage tensors
    local_goes = torch.zeros(
        config.ensemble_per_gpu, config.num_steps + 1, n_goes_channels, H, W, device=dm.device
    )
    local_ghi = torch.zeros(
        config.ensemble_per_gpu, config.num_steps + 1, H, W, device=dm.device
    )
    
    torch.cuda.empty_cache()
    
    # Run forecasts
    total_start = time.perf_counter()
    
    for init_idx, init_time in enumerate(init_times):
        t_init_start = time.perf_counter()
        
        if dm.rank == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Init time {init_idx + 1}/{len(init_times)}: {init_time}")
            logger.info(f"{'='*60}")
        
        # Fetch data
        x, x_coords = fetch_goes_data(dm, stormscope, init_time, config, goes_lat, goes_lon)
        
        # Run forecast
        valid_times = run_forecast(
            dm, stormscope, solar_model, x, x_coords, init_time, config,
            local_goes, local_ghi
        )
        
        # Gather and save
        gather_and_save(
            dm, local_goes, local_ghi, valid_times, init_time, config,
            lat_out, lon_out
        )
        
        if dm.rank == 0:
            t_init_total = time.perf_counter() - t_init_start
            logger.info(f"  Total for init time: {t_init_total:.2f}s")
    
    # Summary
    total_time = time.perf_counter() - total_start
    
    if dm.rank == 0:
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total init times: {len(init_times)}")
        logger.info(f"Total ensemble size: {dm.world_size * config.ensemble_per_gpu}")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        logger.info(f"Avg per init time: {total_time/len(init_times):.2f}s")
        logger.info("Done!")


if __name__ == "__main__":
    main()

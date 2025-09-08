# Script for generating dataset to train cbottle super resolution model

import datetime
import numpy as np
import torch
import zarr
import tqdm
import random
import matplotlib.pyplot as plt
import os
import yaml
import sys
from pathlib import Path

# Optional MPI support
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

from earth2studio.data import JPSS
from earth2grid import healpix
from cbottle import healpix_utils

def load_config(config_path="configs/config.yaml"):
    """Load and validate configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        print("Please create a config file or check the path.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)
    
    # Validate required sections
    required_sections = ['dates', 'dataset', 'viirs', 'output']
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required section '{section}' in config file")
            sys.exit(1)
    
    return config

def parse_config(config):
    """Parse configuration values into usable variables."""
    # Parse dates
    start_date = datetime.datetime.strptime(config['dates']['start_date'], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(config['dates']['end_date'], "%Y-%m-%d")
    gfs_temporal_resolution = datetime.timedelta(hours=config['dates']['temporal_resolution_hours'])
    
    # Calculate derived values
    nr_times = int((end_date - start_date) / gfs_temporal_resolution)
    nr_samples_per_time = config['dataset']['samples_per_time']
    patch_size = config['dataset']['patch_size']
    hpx_level = config['dataset']['hpx_level']
    
    # Grid calculations
    nr_cells = 12 * (2**hpx_level) ** 2

    # VIIRS variables
    viirs_variables = config['viirs']['variables']
    
    # Output path
    zarr_root = config['output']['zarr_path']
    
    # Random sampling config
    random_time_min = config['viirs']['random_time_offset']['min_hours']
    random_time_max = config['viirs']['random_time_offset']['max_hours']
    
    # Debug settings
    save_debug_plots = config.get('debug', {}).get('save_plots', False)
    max_debug_samples = config.get('debug', {}).get('max_debug_samples', 3)
    
    # Processing settings
    valid_pixel_threshold = config.get('processing', {}).get('valid_pixel_threshold', 0.5)
    max_attempts_multiplier = config['dataset'].get('max_attempts_multiplier', 10)
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'gfs_temporal_resolution': gfs_temporal_resolution,
        'nr_times': nr_times,
        'nr_samples_per_time': nr_samples_per_time,
        'patch_size': patch_size,
        'hpx_level': hpx_level,
        'nr_cells': nr_cells,
        'viirs_variables': viirs_variables,
        'zarr_root': zarr_root,
        'random_time_min': random_time_min,
        'random_time_max': random_time_max,
        'save_debug_plots': save_debug_plots,
        'max_debug_samples': max_debug_samples,
        'valid_pixel_threshold': valid_pixel_threshold,
        'max_attempts_multiplier': max_attempts_multiplier,
        'satellites': config['viirs']['satellites']
    }

def print_config_summary(config_dict):
    """Print a summary of the loaded configuration."""
    print("="*60)
    print("VIIRS Dataset Generation Configuration")
    print("="*60)
    print(f"Date range: {config_dict['start_date'].strftime('%Y-%m-%d')} to {config_dict['end_date'].strftime('%Y-%m-%d')}")
    print(f"Temporal resolution: {config_dict['gfs_temporal_resolution'].total_seconds()/3600:.0f} hours")
    print(f"Number of time steps: {config_dict['nr_times']}")
    print(f"Samples per time step: {config_dict['nr_samples_per_time']}")
    print(f"Total samples: {config_dict['nr_times'] * config_dict['nr_samples_per_time']}")
    print(f"Patch size: {config_dict['patch_size']}x{config_dict['patch_size']}")
    print(f"HEALPix level: {config_dict['hpx_level']}")
    print(f"Variables: {', '.join(config_dict['viirs_variables'])}")
    print(f"Satellites: {', '.join(config_dict['satellites'])}")
    print(f"Output: {config_dict['zarr_root']}")
    print(f"Random time range: {config_dict['random_time_min']:.1f}-{config_dict['random_time_max']:.1f} hours")
    print(f"Debug plots: {'Yes' if config_dict['save_debug_plots'] else 'No'}")
    print("="*60)

def setup_zarr_array(config_dict):
    """Set up zarr array with proper structure and metadata."""
    zarr_root = config_dict['zarr_root']
    nr_times = config_dict['nr_times']
    nr_samples_per_time = config_dict['nr_samples_per_time']
    patch_size = config_dict['patch_size']
    hpx_level = config_dict['hpx_level']
    nr_cells = config_dict['nr_cells']
    viirs_variables = config_dict['viirs_variables']
    start_date = config_dict['start_date']
    end_date = config_dict['end_date']
    gfs_temporal_resolution = config_dict['gfs_temporal_resolution']
    
    # Open or create zarr array
    if os.path.exists(zarr_root):
        print(f"Found existing zarr dataset at {zarr_root}, opening in append mode")
        zarr_array = zarr.open(zarr_root, mode="r+")
        print(f"Loaded existing dataset structure with arrays: {list(zarr_array.keys())}")
    else:
        print(f"Creating new zarr dataset at {zarr_root}")
        zarr_array = zarr.open(zarr_root, mode="w")

    # Create dimension coordinate arrays (only if they don't exist)
    if "time" not in zarr_array:
        zarr_array.create_array("time", shape=(nr_times,), dtype='datetime64[ns]', chunks=(nr_times,))
    if "nr_samples_per_time" not in zarr_array:
        zarr_array.create_array("nr_samples_per_time", shape=(nr_samples_per_time,), dtype=np.int32, chunks=(nr_samples_per_time,))
    if "viirs_channel" not in zarr_array:
        zarr_array.create_array("viirs_channel", shape=(len(viirs_variables),), dtype='U50', chunks=(len(viirs_variables),))
    if "y" not in zarr_array:
        zarr_array.create_array("y", shape=(patch_size,), dtype=np.float32, chunks=(patch_size,))
    if "x" not in zarr_array:
        zarr_array.create_array("x", shape=(patch_size,), dtype=np.float32, chunks=(patch_size,))

    # Create main data arrays (only if they don't exist)
    if "jpss_patch" not in zarr_array:
        zarr_array.create_array(
            "jpss_patch",
            shape=(nr_times, nr_samples_per_time, len(viirs_variables), patch_size, patch_size), 
            dtype=np.float32,
            chunks=(1, 1, len(viirs_variables), patch_size, patch_size)
        )
    if "jpss_patch_mask" not in zarr_array:
        zarr_array.create_array(
            "jpss_patch_mask",
            shape=(nr_times, nr_samples_per_time, 1, patch_size, patch_size),
            dtype=np.bool_,
            chunks=(1, 1, 1, patch_size, patch_size)
        )
    if "jpss_patch_coords" not in zarr_array:
        zarr_array.create_array(
            "jpss_patch_coords",
            shape=(nr_times, nr_samples_per_time, 1, patch_size, patch_size),
            dtype=np.int32,
            chunks=(1, 1, 1, patch_size, patch_size)
        )
    if "jpss_patch_id" not in zarr_array:
        zarr_array.create_array(
            "jpss_patch_id",
            shape=(nr_times, nr_samples_per_time, 1),
            dtype=np.int32,
            chunks=(1, 1, 1)
        )
    if "jpss_patch_lead_time" not in zarr_array:
        zarr_array.create_array(
            "jpss_patch_lead_time",
            shape=(nr_times, nr_samples_per_time, 1),
            dtype=np.float32,
            chunks=(1, 1, 1)
        )
    if "jpss_finished_fetching" not in zarr_array:
        zarr_array.create_array(
            "jpss_finished_fetching",
            shape=(nr_times, 1),
            dtype=np.bool_,
            chunks=(1, 1)
        )
        # Initialize all to False (not finished)
        zarr_array["jpss_finished_fetching"][:] = False
        zarr_array["jpss_finished_fetching"].attrs['_ARRAY_DIMENSIONS'] = ['time', 'finished_dim']

    # Populate coordinate arrays (only if they're new)
    if "time" in zarr_array and zarr_array["time"].shape[0] == nr_times:
        # Check if time array needs to be populated
        if np.all(zarr_array["time"][:] == np.datetime64('1970-01-01')):  # Default value check
            zarr_array["time"][:] = np.array([start_date + gfs_temporal_resolution * i for i in range(nr_times)])

    if "nr_samples_per_time" in zarr_array:
        zarr_array["nr_samples_per_time"][:] = np.array([0 for i in range(nr_samples_per_time)])

    if "viirs_channel" in zarr_array:
        zarr_array["viirs_channel"][:] = np.array(viirs_variables)

    if "y" in zarr_array:
        zarr_array["y"][:] = np.linspace(-90, 90, patch_size)  # Latitude-like coordinates

    if "x" in zarr_array:
        zarr_array["x"][:] = np.linspace(0, 360, patch_size)   # Longitude-like coordinates

    # Set attributes for xarray compatibility (safe to set multiple times)
    if "jpss_patch" in zarr_array:
        zarr_array["jpss_patch"].attrs['_ARRAY_DIMENSIONS'] = ['time', 'nr_samples_per_time', 'viirs_channel', 'y', 'x']
    if "jpss_patch_mask" in zarr_array:
        zarr_array["jpss_patch_mask"].attrs['_ARRAY_DIMENSIONS'] = ['time', 'nr_samples_per_time', 'mask_dim', 'y', 'x']
    if "jpss_patch_coords" in zarr_array:
        zarr_array["jpss_patch_coords"].attrs['_ARRAY_DIMENSIONS'] = ['time', 'nr_samples_per_time', 'coord_channel', 'y', 'x']
    if "time" in zarr_array:
        zarr_array["time"].attrs['_ARRAY_DIMENSIONS'] = ['time']
    if "nr_samples_per_time" in zarr_array:
        zarr_array["nr_samples_per_time"].attrs['_ARRAY_DIMENSIONS'] = ['nr_samples_per_time']
    if "viirs_channel" in zarr_array:
        zarr_array["viirs_channel"].attrs['_ARRAY_DIMENSIONS'] = ['viirs_channel']
    if "y" in zarr_array:
        zarr_array["y"].attrs['_ARRAY_DIMENSIONS'] = ['y']
    if "x" in zarr_array:
        zarr_array["x"].attrs['_ARRAY_DIMENSIONS'] = ['x']

    # Add global metadata
    zarr_array.attrs['description'] = 'CBottle training dataset with GFS input and VIIRS target data'
    zarr_array.attrs['patch_size'] = patch_size
    zarr_array.attrs['hpx_level'] = hpx_level
    zarr_array.attrs['nr_cells'] = nr_cells
    zarr_array.attrs['start_date'] = str(start_date)
    zarr_array.attrs['end_date'] = str(end_date)
    zarr_array.attrs['nr_times'] = nr_times
    zarr_array.attrs['nr_samples_per_time'] = nr_samples_per_time
    zarr_array.attrs['gfs_temporal_resolution_hours'] = gfs_temporal_resolution.total_seconds() / 3600
    zarr_array.attrs['created'] = str(datetime.datetime.now())
    zarr_array.attrs['conventions'] = 'CF-1.8'
    
    return zarr_array

def get_mpi_info():
    """Get MPI rank and size, returns (0, 1) if MPI not available."""
    if MPI_AVAILABLE and 'SLURM_PROCID' in os.environ:
        # We're in an MPI environment
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        return rank, size, comm
    else:
        # Single process
        return 0, 1, None

def distribute_time_steps(time_steps, rank, size):
    """Distribute time steps across MPI ranks."""
    if size == 1:
        return time_steps
    
    # Calculate how many time steps each rank should process
    nr_times = len(time_steps)
    base_count = nr_times // size
    remainder = nr_times % size
    
    # Ranks 0 to remainder-1 get one extra time step
    if rank < remainder:
        my_count = base_count + 1
        my_start = rank * my_count
    else:
        my_count = base_count
        my_start = rank * base_count + remainder
    
    my_end = my_start + my_count
    my_time_steps = time_steps[my_start:my_end]
    
    return my_time_steps

if __name__ == "__main__":
    # Get MPI information
    rank, size, comm = get_mpi_info()
    
    ## Set rank-specific cache directory to avoid conflicts
    #base_cache_dir = os.environ.get('EARTH2STUDIO_CACHE', '/lustre/fsw/portfolios/coreai/users/ohennigh/.cache/earth2studio')
    #rank_cache_dir = f"{base_cache_dir}_rank{rank}"
    #os.environ['EARTH2STUDIO_CACHE'] = rank_cache_dir
    
    # Create cache directory if it doesn't exist
    #os.makedirs(rank_cache_dir, exist_ok=True)
    
    #if rank == 0:
    #    print(f"Setting up rank-specific cache directories...")
    #print(f"Rank {rank}: Using cache directory: {rank_cache_dir}")
    
    # Load configuration (all ranks)
    config = load_config()
    config_dict = parse_config(config)
    
    # Print configuration summary (only rank 0)
    if rank == 0:
        if size > 1:
            print(f"🚀 Starting MPI VIIRS dataset generation with {size} processes")
        print_config_summary(config_dict)
    
    # Extract variables from config
    start_date = config_dict['start_date']
    end_date = config_dict['end_date']
    gfs_temporal_resolution = config_dict['gfs_temporal_resolution']
    nr_times = config_dict['nr_times']
    nr_samples_per_time = config_dict['nr_samples_per_time']
    patch_size = config_dict['patch_size']
    hpx_level = config_dict['hpx_level']
    nr_cells = config_dict['nr_cells']
    viirs_variables = config_dict['viirs_variables']
    zarr_root = config_dict['zarr_root']
    
    # Create data sources from config
    jpss = []
    for satellite in config_dict['satellites']:
        jpss.append(JPSS(satellite=satellite, band_type="L2", cache=True))

    # Create grid
    hpx_grid = healpix.Grid(
        level=hpx_level, pixel_order=healpix.HEALPIX_PAD_XY
    )

    # Create coordinate map
    nside = 2**hpx_level
    ids = torch.arange(12 * nside**2, dtype=torch.float32)
    coordinate_map = ids.view(1, 12, nside, nside)

    # Set up zarr array with all metadata and structure (only rank 0)
    if rank == 0:
        zarr_array = setup_zarr_array(config_dict)
    
    # Wait for rank 0 to finish setting up zarr structure
    if comm is not None:
        comm.Barrier()
    
    # All ranks open the zarr array
    if rank != 0:
        zarr_array = zarr.open(config_dict['zarr_root'], mode="r+")
     
    # Get Data
    # Check which time steps are already completed
    completed_times = []
    if "jpss_finished_fetching" in zarr_array:
        for time_idx in range(nr_times):
            if zarr_array["jpss_finished_fetching"][time_idx, 0]:
                completed_times.append(time_idx)
    
    # Distribute work across MPI ranks
    remaining_times = [t for t in range(nr_times) if t not in completed_times]
    my_time_steps = distribute_time_steps(remaining_times, rank, size)
    
    if rank == 0:
        print(f"Generating dataset with {nr_times} time steps, {nr_samples_per_time} samples per time step")
        if completed_times:
            print(f"Found {len(completed_times)} already completed time steps: {completed_times}")
            print(f"Will process remaining {len(remaining_times)} time steps")
        if size > 1:
            print(f"Distributing work across {size} MPI processes")
    
    print(f"Rank {rank}: Processing {len(my_time_steps)} time steps: {my_time_steps}")
    
    # Loop through assigned time steps
    for time_idx in my_time_steps:
        # Calculate the base time for this time step
        base_time = start_date + time_idx * gfs_temporal_resolution
        
        print(f"Rank {rank}: Processing time step {time_idx + 1}/{nr_times}: {base_time}")
        
        # Collect samples for this time step
        sample_idx = 0
        attempts = 0
        max_attempts = nr_samples_per_time * config_dict['max_attempts_multiplier']  # Prevent infinite loops
        
        with tqdm.tqdm(total=nr_samples_per_time, desc=f"Collecting samples for {base_time.strftime('%Y-%m-%d %H:%M')}") as pbar:
            while sample_idx < nr_samples_per_time and attempts < max_attempts:
                attempts += 1
                
                # Sample within configured time range
                random_offset_hours = random.uniform(config_dict['random_time_min'], config_dict['random_time_max'])
                random_time = base_time + datetime.timedelta(hours=random_offset_hours)
                
                # Randomly select a satellite
                selected_jpss = random.choice(jpss)

                # Get VIIRS data
                viirs_data = selected_jpss([random_time], viirs_variables)

                # Make pytorch tensor
                viirs_data_tensor = torch.from_numpy(viirs_data.values[0])
                
                # Clean up viirs_data immediately
                del viirs_data

                # Get lat lon coordinates (last dimension is lat, second to last is lon)
                viirs_lat = viirs_data_tensor[-2, :, :].flatten()
                viirs_lon = viirs_data_tensor[-1, :, :].flatten()
                viirs_data_tensor = viirs_data_tensor[0:-2, :, :].flatten(start_dim=1)

                # Get healpix coordinates
                pix = hpx_grid.ang2pix(viirs_lon, viirs_lat)

                # Check for invalid pixel indices (negative, NaN, or out of bounds)
                valid_mask = (pix >= 0) & (pix < nr_cells) & torch.isfinite(pix.float())
                invalid_count = torch.sum(~valid_mask).item()
                
                if invalid_count > 0:
                    print(f"  Warning: Found {invalid_count}/{len(pix)} invalid pixel indices, filtering them out")
                    # Filter out invalid pixels and corresponding data
                    pix = pix[valid_mask]
                    viirs_data_tensor = viirs_data_tensor[:, valid_mask]
                    
                    # Skip this sample if too many pixels are invalid
                    if len(pix) == 0 or invalid_count / len(valid_mask) > 0.5:  # More than 50% invalid
                        print(f"  Skipping sample due to too many invalid coordinates ({invalid_count}/{len(valid_mask)})")
                        # Clean up before continuing
                        del viirs_lat, viirs_lon, viirs_data_tensor, pix, valid_mask
                        torch.cuda.empty_cache()
                        continue
                
                # Clean up coordinate arrays after validation
                del viirs_lat, viirs_lon, valid_mask

                # Use bincount to aggregate data
                binned_viirs_data = [torch.bincount(pix, weights=viirs_data_tensor[i], minlength=nr_cells) for i in range(viirs_data_tensor.shape[0])]
                binned_viirs_data = torch.stack(binned_viirs_data)
                count_viirs_data = torch.bincount(pix, minlength=nr_cells)
                count_viirs_data = count_viirs_data.unsqueeze(0)
                
                # Clean up after bincount operations
                del pix, viirs_data_tensor
                
                # Compute average
                avg_viirs_data = binned_viirs_data / count_viirs_data

                assert avg_viirs_data.shape[-1] == nr_cells

                # Delete binned_viirs_data
                del binned_viirs_data
                torch.cuda.empty_cache()

                # Patchify data
                patches = healpix_utils.to_patches(
                    [avg_viirs_data.float(), count_viirs_data.float()],
                    patch_size=patch_size,
                    pre_padded_tensors=[coordinate_map.float()],
                )

                # Store patches
                for patch_id, (viirs_patch, count_viirs_patch, patch_coord_map, _) in enumerate(patches):
                    # Check if sample slot is still available
                    if sample_idx >= nr_samples_per_time:
                        # Clean up patch data before breaking
                        del viirs_patch, count_viirs_patch, patch_coord_map
                        break

                    # Check if too many pixels are invalid (based on config threshold)
                    invalid_fraction = torch.sum(count_viirs_patch == 0).float() / count_viirs_patch.numel()
                    if invalid_fraction > config_dict['valid_pixel_threshold']:
                        # Clean up patch data before continuing
                        del viirs_patch, count_viirs_patch, patch_coord_map
                        continue

                    # Store data directly in zarr arrays using indexing
                    zarr_array["jpss_patch"][time_idx, sample_idx, :, :, :] = viirs_patch[0].float().numpy()
                    zarr_array["jpss_patch_mask"][time_idx, sample_idx, :, :, :] = (count_viirs_patch != 0).bool().numpy()
                    zarr_array["jpss_patch_coords"][time_idx, sample_idx, :, :, :] = patch_coord_map.int().numpy()
                    zarr_array["jpss_patch_id"][time_idx, sample_idx, :] = patch_id
                    zarr_array["jpss_patch_lead_time"][time_idx, sample_idx, :] = random_offset_hours * 60
                    
                    sample_idx += 1
                    pbar.update(1)
                    
                    # Optional: Save debug plots for first few samples (if enabled in config)
                    if (config_dict['save_debug_plots'] and 
                        time_idx == 0 and 
                        sample_idx <= config_dict['max_debug_samples']):
                        for var_idx in range(viirs_patch.shape[1]):
                            plt.figure(figsize=(8, 6))
                            plt.imshow(viirs_patch[0, var_idx].numpy())
                            plt.title(f"Time {time_idx}, Sample {sample_idx}: {viirs_variables[var_idx]}")
                            plt.colorbar()
                            plt.savefig(f"debug_t{time_idx}_s{sample_idx}_{viirs_variables[var_idx]}.png")
                            plt.close()
                    
                    # Clean up patch data immediately after use
                    del viirs_patch, count_viirs_patch, patch_coord_map

                # Clean up remaining data from the patches loop
                del avg_viirs_data, count_viirs_data
                del patches  # Clean up the generator
                torch.cuda.empty_cache()
                        
        if sample_idx < nr_samples_per_time:
            print(f"Warning: Only collected {sample_idx}/{nr_samples_per_time} samples for time step {time_idx}")
            print(f"Time step {time_idx} marked as incomplete due to insufficient samples")
        else:
            # Mark this time step as completed
            zarr_array["jpss_finished_fetching"][time_idx, 0] = True
            print(f"Rank {rank}: Time step {time_idx} completed successfully - all {nr_samples_per_time} samples collected")
        
        # Clean up any remaining variables from this time step
        torch.cuda.empty_cache()
    
    # Wait for all ranks to complete
    if comm is not None:
        comm.Barrier()
    
    # Print final summary (only rank 0)
    if rank == 0:
        print("\nDataset generation complete!")
        
        # Print final summary
        completed_count = np.sum(zarr_array["jpss_finished_fetching"][:, 0])
        print(f"Final status: {completed_count}/{nr_times} time steps completed")
        if completed_count < nr_times:
            incomplete_times = [i for i in range(nr_times) if not zarr_array["jpss_finished_fetching"][i, 0]]
            print(f"Incomplete time steps: {incomplete_times}")
            print("You can restart the script to process the remaining time steps")
        else:
            print("🎉 All time steps completed successfully!")

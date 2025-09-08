# Script for adding GFS data to existing CBottle dataset
# NOTE: Run generate_viirs_dataset.py first to create the base zarr structure

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

from earth2studio.data import ARCO, GFS, JPSS, CDS, WB2ERA5
from earth2studio.lexicon import CBottleLexicon

from earth2grid import latlon, healpix
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
    required_sections = ['dates', 'dataset', 'gfs', 'output']
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required section '{section}' in config file")
            sys.exit(1)
    
    return config

def parse_gfs_config(config):
    """Parse configuration specifically for GFS data processing."""
    # GFS variables configuration
    gfs_variables = list(CBottleLexicon.VOCAB.keys())
    gfs_variables_count = config.get('gfs', {}).get('variables_count', 1)
    gfs_variables = gfs_variables[0:gfs_variables_count]
    
    # Lead times configuration
    lead_times = config.get('gfs', {}).get('lead_times_hours', [0, 6])
    
    # Output configuration
    zarr_root = config['output']['zarr_path']
    
    # Debug settings
    save_debug_plots = config.get('debug', {}).get('save_plots', False)
    max_debug_samples = config.get('debug', {}).get('max_debug_samples', 3)
    
    return {
        'gfs_variables': gfs_variables,
        'lead_times': lead_times,
        'zarr_root': zarr_root,
        'gfs_variables_count': gfs_variables_count,
        'save_debug_plots': save_debug_plots,
        'max_debug_samples': max_debug_samples
    }

def print_gfs_config_summary(config_dict, gfs_config):
    """Print a summary of the GFS processing configuration."""
    print("="*60)
    print("GFS Dataset Addition Configuration")
    print("="*60)
    print(f"Reading from: {gfs_config['zarr_root']}")
    print(f"GFS variables: {gfs_config['gfs_variables']}")
    print(f"Lead times: {gfs_config['lead_times']} hours")
    print(f"Number of GFS channels: {len(gfs_config['gfs_variables']) * 2} (HPX + LR)")
    print(f"Debug plots: {'Yes' if gfs_config['save_debug_plots'] else 'No'}")
    if gfs_config['save_debug_plots']:
        print(f"Max debug samples: {gfs_config['max_debug_samples']}")
    print("="*60)

def setup_gfs_zarr_arrays(zarr_array, gfs_config, nr_times, nr_samples_per_time, patch_size, gfs_patch_size):
    """Set up GFS-specific zarr arrays."""
    gfs_variables = gfs_config['gfs_variables']
    nr_gfs_channels = len(gfs_variables)  # hpx and lr versions
    
    # Add GFS-specific arrays if they don't already exist
    if "gfs_channel" not in zarr_array:
        print("Creating gfs_channel array...")
        zarr_array.create_array("gfs_channel", shape=(nr_gfs_channels,), dtype='U50', chunks=(nr_gfs_channels,))
        # Populate GFS channel array
        zarr_array["gfs_channel"][:] = np.array(
            [f"{var}" for var in gfs_variables]
        )
        # Set attributes for xarray compatibility
        zarr_array["gfs_channel"].attrs['_ARRAY_DIMENSIONS'] = ['gfs_channel']
    else:
        print("gfs_channel array already exists, skipping creation")

    if "gfs_patch" not in zarr_array:
        print("Creating gfs_patch array...")
        zarr_array.create_array(
            "gfs_patch",
            shape=(
                nr_times,
                nr_samples_per_time,
                len(gfs_config['lead_times']),
                nr_gfs_channels,
                gfs_patch_size,
                gfs_patch_size
            ), 
            dtype=np.float32,
            chunks=(1, 1, len(gfs_config['lead_times']), nr_gfs_channels, gfs_patch_size, gfs_patch_size)
        )
        # Set attributes for xarray compatibility
        zarr_array["gfs_patch"].attrs['_ARRAY_DIMENSIONS'] = ['time', 'nr_samples_per_time', 'lead_time', 'gfs_channel', 'y', 'x']
    else:
        print("gfs_patch array already exists, skipping creation")

    if "gfs_lr" not in zarr_array:
        print("Creating gfs_lr array...")
        zarr_array.create_array(
            "gfs_lr",
            shape=(
                nr_times,
                len(gfs_config['lead_times']),
                nr_gfs_channels,
                patch_size,
                patch_size),
            dtype=np.float32,
            chunks=(1, len(gfs_config['lead_times']), nr_gfs_channels, patch_size, patch_size)
        )
        
    # Add GFS completion tracking
    if "gfs_finished_fetching" not in zarr_array:
        print("Creating gfs_finished_fetching array...")
        zarr_array.create_array(
            "gfs_finished_fetching",
            shape=(nr_times, 1),
            dtype=np.bool_,
            chunks=(1, 1)
        )
        # Initialize all to False (not finished)
        zarr_array["gfs_finished_fetching"][:] = False
        zarr_array["gfs_finished_fetching"].attrs['_ARRAY_DIMENSIONS'] = ['time', 'finished_dim']
    else:
        print("gfs_finished_fetching array already exists, skipping creation")

    # Update global metadata for GFS
    if 'gfs_variables' not in zarr_array.attrs:
        zarr_array.attrs['gfs_variables'] = gfs_variables
        zarr_array.attrs['gfs_added'] = str(datetime.datetime.now())
    
    return nr_gfs_channels

def check_viirs_completion(zarr_array, nr_times):
    """Check which time steps have completed VIIRS data."""
    print("Checking VIIRS completion status...")
    viirs_completed_times = []
    if "jpss_finished_fetching" in zarr_array:
        for time_idx in range(nr_times):
            if zarr_array["jpss_finished_fetching"][time_idx, 0]:
                viirs_completed_times.append(time_idx)

    print(f"Found {len(viirs_completed_times)} time steps with completed VIIRS data")
    if len(viirs_completed_times) == 0:
        print("No completed VIIRS time steps found. Please run generate_viirs_dataset.py first.")
        sys.exit(1)
    
    return viirs_completed_times

def check_gfs_completion(zarr_array, nr_times):
    """Check which time steps have completed GFS data."""
    gfs_completed_times = []
    if "gfs_finished_fetching" in zarr_array:
        for time_idx in range(nr_times):
            if zarr_array["gfs_finished_fetching"][time_idx, 0]:
                gfs_completed_times.append(time_idx)
    return gfs_completed_times

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
    
    ## Create cache directory if it doesn't exist
    #os.makedirs(rank_cache_dir, exist_ok=True)
    
    #if rank == 0:
    #    print(f"Setting up rank-specific cache directories...")
    #print(f"Rank {rank}: Using cache directory: {rank_cache_dir}")
    
    # Load configuration (all ranks)
    config = load_config()
    gfs_config = parse_gfs_config(config)
    
    # Parse main dataset config (similar to VIIRS script)
    start_date = datetime.datetime.strptime(config['dates']['start_date'], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(config['dates']['end_date'], "%Y-%m-%d")
    gfs_temporal_resolution = datetime.timedelta(hours=config['dates']['temporal_resolution_hours'])
    nr_times = int((end_date - start_date) / gfs_temporal_resolution)
    nr_samples_per_time = config['dataset']['samples_per_time']
    patch_size = config['dataset']['patch_size']
    hpx_level = config['dataset']['hpx_level']
    gfs_hpx_level = config['dataset']['gfs_hpx_level']
    gfs_patch_size = patch_size // 2**(hpx_level - gfs_hpx_level)
    nr_cells = 12 * (2**hpx_level) ** 2
    
    # Check if zarr dataset exists
    zarr_root = gfs_config['zarr_root']
    if not os.path.exists(zarr_root):
        if rank == 0:
            print(f"Error: Zarr dataset not found at {zarr_root}")
            print("Please run generate_viirs_dataset.py first to create the base dataset")
        sys.exit(1)

    # Open existing zarr array to read parameters (rank 0 first)
    if rank == 0:
        print("Reading parameters from existing zarr dataset...")
        zarr_array = zarr.open(zarr_root, mode="r+")
        print(f"Loaded existing dataset with arrays: {list(zarr_array.keys())}")
    
    # Wait for rank 0 to finish reading
    if comm is not None:
        comm.Barrier()
    
    # All ranks open the zarr array
    if rank != 0:
        zarr_array = zarr.open(zarr_root, mode="r+")

    if rank == 0:
        if size > 1:
            print(f"🚀 Starting MPI GFS dataset addition with {size} processes")
        print(f"Configuration parameters:")
        print(f"  Start date: {start_date}")
        print(f"  End date: {end_date}")
        print(f"  Temporal resolution: {gfs_temporal_resolution}")
        print(f"  Samples per time: {nr_samples_per_time}")
        print(f"  Number of times: {nr_times}")
        print(f"  Patch size: {patch_size}")
        print(f"  HPX level: {hpx_level}")
        print(f"  Number of cells: {nr_cells}")

        # Print configuration summary
        print_gfs_config_summary(config, gfs_config)
        
        print(f"Using {len(gfs_config['gfs_variables'])} GFS variables: {gfs_config['gfs_variables']}")

    # Create data sources
    gfs = GFS()

    # Create mapping and regridders
    lat_lon_gfs_grid = latlon.equiangular_lat_lon_grid(
        721, 1440, includes_south_pole=False  # GFS grid resolution
    )
    hpx_grid = healpix.Grid(
        level=gfs_hpx_level, pixel_order=healpix.HEALPIX_PAD_XY
    )

    # Regridder from GFS lat/lon to healpix
    lat_lon_to_hpx = lat_lon_gfs_grid.get_bilinear_regridder_to(hpx_grid.lat, hpx_grid.lon)

    # Low resolution lat/lon grid for additional context
    lr_lat = torch.linspace(-90, 90, patch_size)[:, None].cpu().numpy()
    lr_lon = torch.linspace(0, 360, patch_size)[None, :].cpu().numpy()
    lat_lon_to_lr_lat_lon = lat_lon_gfs_grid.get_bilinear_regridder_to(
        lr_lat, lr_lon
    ).to("cpu")

    # Set up GFS-specific zarr arrays (only rank 0)
    if rank == 0:
        nr_gfs_channels = setup_gfs_zarr_arrays(zarr_array, gfs_config, nr_times, nr_samples_per_time, patch_size, gfs_patch_size)
    
    # Wait for rank 0 to finish setting up arrays
    if comm is not None:
        comm.Barrier()

    # Check which time steps have completed VIIRS data
    viirs_completed_times = check_viirs_completion(zarr_array, nr_times)
    
    # Check which time steps have completed GFS data
    gfs_completed_times = check_gfs_completion(zarr_array, nr_times)
    
    # Only process time steps that have VIIRS but not GFS
    pending_gfs_times = [t for t in viirs_completed_times if t not in gfs_completed_times]
    
    # Distribute work across MPI ranks
    my_time_steps = distribute_time_steps(pending_gfs_times, rank, size)
    
    if rank == 0:
        print(f"Found {len(viirs_completed_times)} time steps with completed VIIRS data")
        if gfs_completed_times:
            print(f"Found {len(gfs_completed_times)} time steps with completed GFS data: {gfs_completed_times}")
            print(f"Will process remaining {len(pending_gfs_times)} time steps for GFS")
        if size > 1:
            print(f"Distributing work across {size} MPI processes")
    
    print(f"Rank {rank}: Processing {len(my_time_steps)} time steps: {my_time_steps}")

    # Get GFS Data aligned with existing VIIRS patches
    if rank == 0:
        print(f"Adding GFS data aligned with existing VIIRS patches...")

    for time_idx in my_time_steps:
        # Calculate the base time for this time step
        base_time = start_date + time_idx * gfs_temporal_resolution
        
        print(f"Rank {rank}: Processing GFS for time step {time_idx + 1}/{nr_times}: {base_time}")

        # Get GFS data at configured lead times
        lead_times = gfs_config['lead_times']
        gfs_times = [base_time + datetime.timedelta(hours=lt) for lt in lead_times]
        gfs_data_all = gfs(gfs_times, gfs_config['gfs_variables'])

        # Make tensor
        gfs_data_tensor = torch.from_numpy(gfs_data_all.values)
        gfs_data_tensor = torch.reshape(
            gfs_data_tensor,
            (
                gfs_data_tensor.shape[0]*gfs_data_tensor.shape[1],
                gfs_data_tensor.shape[2],
                gfs_data_tensor.shape[3]
            )
        )
        
        # Regrid to healpix
        print(f"Regrid GFS data to healpix...")
        gfs_data_hpx = lat_lon_to_hpx(gfs_data_tensor)
        print(f"Regrid GFS data to low-res lat/lon...")

        # Regrid to low-res lat/lon
        gfs_data_lr_lat_lon = lat_lon_to_lr_lat_lon(gfs_data_tensor)
        gfs_data_lr_lat_lon = torch.reshape(
            gfs_data_lr_lat_lon,
            (
                len(lead_times), 
                len(gfs_config['gfs_variables']),
                patch_size,
                patch_size
            )
        )

        # Store low-res lat/lon data
        zarr_array["gfs_lr"][time_idx, :, :, :, :] = gfs_data_lr_lat_lon.float().numpy()

        # Get all VIIRS patch IDs for this time step
        viirs_patch_ids = set()
        for sample_idx in range(nr_samples_per_time):
            viirs_patch_id = int(zarr_array["jpss_patch_id"][time_idx, sample_idx, 0])
            viirs_patch_ids.add((viirs_patch_id, sample_idx))
        
        print(f"Found {len(viirs_patch_ids)} unique VIIRS patches to match")

        # Get patches iterator (keep as iterator to save memory)
        patches = healpix_utils.to_patches(
            [gfs_data_hpx.float()],
            patch_size=gfs_patch_size,
            pre_padded_tensors=[],
        )

        # Iterate through GFS patches and store when needed
        patches_stored = 0
        print(f"Iterating through GFS patches...")
        
        with tqdm.tqdm(desc=f"Processing GFS patches for time {time_idx}") as pbar:
            for patch_id, (gfs_patch, _) in enumerate(patches):
                pbar.update(1)

                # Check if any VIIRS samples need this patch
                matching_samples = [sample_idx for (viirs_patch_id, sample_idx) in viirs_patch_ids if viirs_patch_id == patch_id]
                
                if matching_samples:
                    # Reshape GFS patch to [lead_times, vars, y, x] 
                    target_patch_hpx = gfs_patch[0].reshape(len(lead_times), len(gfs_config['gfs_variables']), gfs_patch_size, gfs_patch_size)

                    # Store for all matching VIIRS samples
                    for sample_idx in matching_samples:
                        zarr_array["gfs_patch"][time_idx, sample_idx, :, :, :, :] = target_patch_hpx.float().numpy()
                        patches_stored += 1
                        
                        # Optional: Save debug plots for first few samples (if enabled in config)
                        if (gfs_config['save_debug_plots'] and 
                            time_idx == my_time_steps[0] and  # First time step being processed by this rank
                            patches_stored <= gfs_config['max_debug_samples']):
                            
                            # Get corresponding VIIRS data for comparison
                            viirs_patch = zarr_array["jpss_patch"][time_idx, sample_idx, :, :, :]
                            
                            # Create debug plot showing VIIRS vs GFS
                            fig, axes = plt.subplots(2, len(gfs_config['gfs_variables']), figsize=(4 * len(gfs_config['gfs_variables']), 8))
                            if len(gfs_config['gfs_variables']) == 1:
                                axes = axes.reshape(-1, 1)
                            
                            fig.suptitle(f"Rank {rank}: VIIRS vs GFS Debug - Time {time_idx}, Sample {sample_idx}, Patch {patch_id}", fontsize=14)
                            
                            for var_idx, var_name in enumerate(gfs_config['gfs_variables']):
                                # Plot VIIRS data (if we have corresponding variables)
                                if var_idx < viirs_patch.shape[0]:
                                    axes[0, var_idx].imshow(viirs_patch[var_idx])
                                    axes[0, var_idx].set_title(f"VIIRS: Channel {var_idx}")
                                    axes[0, var_idx].axis('off')
                                else:
                                    axes[0, var_idx].text(0.5, 0.5, 'No VIIRS\ncorrespondence', 
                                                         transform=axes[0, var_idx].transAxes, 
                                                         ha='center', va='center')
                                    axes[0, var_idx].axis('off')
                                
                                # Plot GFS data (HPX version)
                                gfs_patch_data = target_patch_hpx[0, var_idx].numpy()  # First lead time, HPX version
                                im = axes[1, var_idx].imshow(gfs_patch_data, cmap='viridis')
                                axes[1, var_idx].set_title(f"GFS: {var_name}")
                                axes[1, var_idx].axis('off')
                                plt.colorbar(im, ax=axes[1, var_idx], shrink=0.8)
                            
                            plt.tight_layout()
                            debug_filename = f"debug_gfs_rank{rank}_t{time_idx}_s{sample_idx}_p{patch_id}.png"
                            plt.savefig(debug_filename, dpi=150, bbox_inches='tight')
                            plt.close()
                            print(f"    Saved debug plot: {debug_filename}")
                    
                    # Remove processed samples to speed up future lookups
                    viirs_patch_ids = {(pid, sid) for (pid, sid) in viirs_patch_ids if sid not in matching_samples}
                    
                    # Early exit if we've processed all needed patches
                    if not viirs_patch_ids:
                        print(f"  All VIIRS patches matched, stopping early at patch {patch_id}")
                        break
        
        print(f"  Stored GFS data for {patches_stored} patches")
        
        # Mark this time step as GFS completed
        zarr_array["gfs_finished_fetching"][time_idx, 0] = True
        print(f"Rank {rank}: Time step {time_idx} GFS processing completed successfully")
    
    # Wait for all ranks to complete
    if comm is not None:
        comm.Barrier()
    
    # Print final summary (only rank 0)
    if rank == 0:
        print("\nGFS dataset addition complete!")
        
        # Print final summary
        gfs_completed_count = np.sum(zarr_array["gfs_finished_fetching"][:, 0])
        viirs_completed_count = np.sum(zarr_array["jpss_finished_fetching"][:, 0])
        print(f"Final status: {gfs_completed_count}/{viirs_completed_count} VIIRS time steps now have GFS data")
        if gfs_completed_count < viirs_completed_count:
            incomplete_gfs_times = [i for i in range(nr_times) if zarr_array["jpss_finished_fetching"][i, 0] and not zarr_array["gfs_finished_fetching"][i, 0]]
            print(f"Remaining GFS time steps to process: {incomplete_gfs_times}")
            print("You can restart this script to process the remaining time steps")
        else:
            print("🎉 All available VIIRS time steps now have matching GFS data!")
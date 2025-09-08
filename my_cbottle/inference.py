# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import yaml
import sys
from typing import Optional

import cbottle.config.environment as config
import earth2grid
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from cbottle_dataset import create_datasets
from cbottle import models
from cbottle import checkpointing
from cbottle.diffusion_samplers import edm_sampler_from_sigma

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import tqdm
import xarray as xr


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
    required_sections = ['inference']
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required section '{section}' in config file")
            sys.exit(1)
    
    return config


def parse_inference_config(config):
    """Parse inference configuration from config file."""
    inference_config = config['inference']
    
    return {
        'state_path': inference_config.get('state_path', ''),
        'output_path': inference_config.get('output_path', '/tmp/cbottle_inference'),
        'plot_sample': inference_config.get('plot_sample', False),
        'single_sample': inference_config.get('single_sample', True),
        'sample_index': inference_config.get('sample_index', 0),
        'max_samples': inference_config.get('max_samples', 10),
        'num_steps': inference_config.get('num_steps', 18),
        'sigma_max': inference_config.get('sigma_max', 800),
        'save_plots': inference_config.get('save_plots', True),
        'save_predictions': inference_config.get('save_predictions', True),
    }


def inference(
    state_path: str,
    output_path: str,
    plot_sample: bool = False,
    single_sample: bool = True,
    sample_index: int = 0,
    max_samples: int = 10,
    num_steps: int = 18,
    sigma_max: int = 800,
    save_plots: bool = True,
    save_predictions: bool = True,
):
    """
    Run CBottle super-resolution inference on patches using configuration parameters.
    """
    # Setup device
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{LOCAL_RANK}")
        torch.cuda.set_device(LOCAL_RANK)
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load CBottle dataset (same as training)
    dataset, _ = create_datasets()
    
    # Create sampler for single sample or multiple samples
    if single_sample:
        indices = [sample_index] if sample_index < len(dataset) else [0]
        print(f"Processing single sample at index {indices[0]}")
    else:
        max_samples = min(max_samples, len(dataset))
        indices = list(range(max_samples))
        print(f"Processing {len(indices)} samples")
    
    # Load the model using the same approach as training
    model_config = models.ModelConfigV1(
        architecture="unet_hpx1024_patch",
        condition_channels=2*(dataset.n_gfs_channels + dataset.n_gfs_channels),  # local + global GFS
        out_channels=dataset.n_viirs_channels,
    )
    model_config.level = 10
    
    # Load trained network
    net = models.get_model(model_config)
    
    # Load checkpoint
    with checkpointing.Checkpoint(state_path) as checkpoint:
        checkpoint.read_model(net=net)
    
    net.eval().requires_grad_(False).to(device)
    
    print(f"Model loaded from {state_path}")
    print(f"Processing {len(indices)} samples...")
    
    # Process samples
    for i, idx in enumerate(tqdm.tqdm(indices)):
        batch = dataset[idx]
        
        print(f"\nProcessing sample {idx} ({i+1}/{len(indices)})")
        print(f"  Metadata: {batch['metadata']}")
        
        # Prepare data (same as in training)
        viirs_patch = batch['viirs_patch']  # (7, 128, 128)
        viirs_patch_mask = batch['viirs_patch_mask']  # (1, 128, 128)
        viirs_patch_coords = batch['viirs_patch_coords']  # (1, 128, 128)
        gfs_patch = batch['gfs_patch']  # (2, 32, 16, 16)
        gfs_lr = batch['gfs_lr']  # (2, 32, 128, 128)
        
        # Move to device
        viirs_patch = viirs_patch.to(device)
        gfs_patch = gfs_patch.to(device)
        gfs_lr = gfs_lr.to(device)
        viirs_patch_coords = viirs_patch_coords.to(device)
        
        # Add batch dimension
        viirs_patch = viirs_patch.unsqueeze(0)  # (1, 7, 128, 128)
        gfs_patch = gfs_patch.unsqueeze(0)  # (1, 2, 32, 16, 16)
        gfs_lr = gfs_lr.unsqueeze(0)  # (1, 2, 32, 128, 128)
        viirs_patch_coords = viirs_patch_coords.unsqueeze(0)  # (1, 1, 128, 128)
        
        # Prepare LLR (low-res input) - same as training
        gfs_patch_0 = torch.nn.functional.interpolate(
            gfs_patch[:, 0, :, :, :],
            size=(128, 128),
            mode='bilinear',
            align_corners=False,
        )
        gfs_patch_1 = torch.nn.functional.interpolate(
            gfs_patch[:, 1, :, :, :],
            size=(128, 128),
            mode='bilinear',
            align_corners=False,
        )
        llr = torch.concat([
            gfs_patch_0,
            gfs_patch_1,
            gfs_lr[:, 0, :, :, :],
            gfs_lr[:, 1, :, :, :],
        ], dim=1)
        
        # Prepare position embedding - same as training
        lpe = viirs_patch_coords / (12 * (2 ** 13) ** 2.)
        lpe = torch.repeat_interleave(lpe, 20, dim=1)
        
        # Handle NaN values
        viirs_patch = torch.where(torch.isnan(viirs_patch), torch.zeros_like(viirs_patch), viirs_patch)
        llr = torch.where(torch.isnan(llr), torch.zeros_like(llr), llr)
        lpe = torch.where(torch.isnan(lpe), torch.zeros_like(lpe), lpe)
        
        print(f"  Input shapes: viirs_patch={viirs_patch.shape}, llr={llr.shape}, lpe={lpe.shape}")
        
        # Run inference
        with torch.no_grad():
            # Create random noise
            noise = torch.randn_like(viirs_patch) * sigma_max
            
            # Define denoiser
            def denoiser(x_hat, t_hat):
                return net(
                    x_hat,
                    t_hat,
                    class_labels=None,
                    condition=llr,
                    position_embedding=lpe,
                    augment_labels=None,
                )
            
            # Set denoiser attributes (required by EDM sampler)
            denoiser.sigma_max = net.sigma_max
            denoiser.sigma_min = net.sigma_min
            denoiser.round_sigma = net.round_sigma
            
            # Run EDM sampling
            pred = edm_sampler_from_sigma(
                denoiser,
                noise,
                sigma_max=sigma_max,
                num_steps=num_steps,
                randn_like=torch.randn_like,
            )
        
        # Remove batch dimension for saving
        pred = pred.squeeze(0)  # (7, 128, 128)
        viirs_target = viirs_patch.squeeze(0)  # (7, 128, 128)
        
        print(f"  Output shape: {pred.shape}")
        
        # Save results
        if save_predictions:
            pred_path = os.path.join(output_path, f"prediction_sample_{idx}.pt")
            target_path = os.path.join(output_path, f"target_sample_{idx}.pt")
            torch.save(pred.cpu(), pred_path)
            torch.save(viirs_target.cpu(), target_path)
            print(f"  Saved prediction to {pred_path}")
            print(f"  Saved target to {target_path}")
        
        # Create plots
        if save_plots:
            plot_patch_comparison(pred, viirs_target, output_path, idx)
            print(f"  Saved plots to {output_path}")


def plot_patch_comparison(pred, target, output_path, sample_idx):
    """Plot comparison between prediction and target for each channel."""
    import matplotlib.pyplot as plt
    
    n_channels = pred.shape[0]
    
    # Create a large figure with subplots for each channel
    fig, axes = plt.subplots(2, n_channels, figsize=(4*n_channels, 8))
    
    for ch in range(n_channels):
        # Target
        axes[0, ch].imshow(target[ch].cpu().numpy(), cmap='viridis')
        axes[0, ch].set_title(f'Target Ch {ch}')
        axes[0, ch].axis('off')
        
        # Prediction
        axes[1, ch].imshow(pred[ch].cpu().numpy(), cmap='viridis')
        axes[1, ch].set_title(f'Prediction Ch {ch}')
        axes[1, ch].axis('off')
    
    plt.tight_layout()
    plot_path = os.path.join(output_path, f"comparison_sample_{sample_idx}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create individual channel plots with colorbars
    for ch in range(n_channels):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Target
        im1 = ax1.imshow(target[ch].cpu().numpy(), cmap='viridis')
        ax1.set_title(f'Target Channel {ch}')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1)
        
        # Prediction  
        im2 = ax2.imshow(pred[ch].cpu().numpy(), cmap='viridis')
        ax2.set_title(f'Prediction Channel {ch}')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plot_path = os.path.join(output_path, f"channel_{ch}_sample_{sample_idx}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    inference_config = parse_inference_config(config)
    
    print("="*60)
    print("CBottle Super Resolution Patch Inference")
    print("="*60)
    print(f"State path: {inference_config['state_path']}")
    print(f"Output path: {inference_config['output_path']}")
    print(f"Single sample mode: {inference_config['single_sample']}")
    print(f"Sample index: {inference_config['sample_index']}")
    print(f"Max samples: {inference_config['max_samples']}")
    print(f"Num steps: {inference_config['num_steps']}")
    print(f"Sigma max: {inference_config['sigma_max']}")
    print(f"Plot samples: {inference_config['plot_sample']}")
    print(f"Save plots: {inference_config['save_plots']}")
    print(f"Save predictions: {inference_config['save_predictions']}")
    print("="*60)
    
    inference(
        state_path=inference_config['state_path'],
        output_path=inference_config['output_path'],
        plot_sample=inference_config['plot_sample'],
        single_sample=inference_config['single_sample'],
        sample_index=inference_config['sample_index'],
        max_samples=inference_config['max_samples'],
        num_steps=inference_config['num_steps'],
        sigma_max=inference_config['sigma_max'],
        save_plots=inference_config['save_plots'],
        save_predictions=inference_config['save_predictions'],
    )

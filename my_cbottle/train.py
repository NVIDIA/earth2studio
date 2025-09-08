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
import time
import yaml
import sys
from typing import Optional

import earth2grid
import torch
import torch.distributed as dist
import einops

import cbottle.checkpointing
import cbottle.config.environment as config
import cbottle.models
from cbottle import distributed as cbottle_dist

from cbottle_dataset import create_datasets


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
    required_sections = ['training']
    for section in required_sections:
        if section not in config:
            print(f"Error: Missing required section '{section}' in config file")
            sys.exit(1)
    
    return config


def parse_training_config(config):
    """Parse training configuration from config file."""
    training_config = config['training']
    
    return {
        'output_path': training_config.get('output_path', '/tmp/cbottle_sr'),
        'log_freq': training_config.get('log_freq', 100),
        'checkpoint_freq': training_config.get('checkpoint_freq', 100),
        'train_batch_size': training_config.get('train_batch_size', 15),
        'test_batch_size': training_config.get('test_batch_size', 30),
        'dataloader_num_workers': training_config.get('dataloader_num_workers', 3),
        'bf16': training_config.get('bf16', False),
        'valid_min_samples': training_config.get('valid_min_samples', 1),
        'num_steps': training_config.get('num_steps', 40000000),
    }


class EDMLossSR:
    def __init__(
        self,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, pos_embed):
        labels = None
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(img_clean) * sigma
        sigma_lr = None
        D_yn = net(
            img_clean + n,
            sigma,
            class_labels=labels,
            condition=img_lr,
            position_embedding=pos_embed,
            augment_labels=sigma_lr,
        )
        loss = weight * ((D_yn - img_clean) ** 2)
        return loss


def load_checkpoint(path: str, *, network, optimizer, scheduler, map_location) -> int:
    with cbottle.checkpointing.Checkpoint(path) as checkpoint:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            checkpoint.read_model(net=network.module)
        else:
            checkpoint.read_model(net=network)

        with checkpoint.open("loop_state.pth", "r") as f:
            training_state = torch.load(f, weights_only=True, map_location=map_location)
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
            step = training_state["step"]

    return step


def save_checkpoint(path, *, model_config, network, optimizer, scheduler, step, loss):
    with cbottle.checkpointing.Checkpoint(path, "w") as checkpoint:
        if isinstance(network, torch.nn.parallel.DistributedDataParallel):
            checkpoint.write_model(network.module)
        else:
            checkpoint.write_model(network)
        checkpoint.write_model_config(model_config)

        with checkpoint.open("loop_state.pth", "w") as f:
            torch.save(
                {
                    "step": step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": loss,
                },
                f,
            )


def find_latest_checkpoint(output_path: str) -> str:
    max_index_file = " "
    max_index = -1
    for filename in os.listdir(output_path):
        if filename.startswith("cBottle-SR-") and filename.endswith(".zip"):
            index_str = filename.split("-")[-1].split(".")[0]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_index_file = filename
            except ValueError:
                continue
    path = os.path.join(output_path, max_index_file)
    return path


def train(
    output_path: str,
    train_batch_size=64,
    test_batch_size=64,
    valid_min_samples: int = 1,
    num_steps: int = int(4e7),
    log_freq: int = 1000,
    checkpoint_freq: int = 100,
    dataloader_num_workers: int = 3,
    bf16: bool = False,
):
    """
    Train CBottle super-resolution model using VIIRS and GFS data.
    """
    cbottle_dist.init()

    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    WORLD_SIZE = cbottle_dist.get_world_size()
    WORLD_RANK = cbottle_dist.get_rank()

    os.makedirs(output_path, exist_ok=True)

    # dataloader - use our CBottle dataset
    training_dataset, _ = create_datasets()
    # For now, use the same dataset for validation (user will create separate one later)
    test_dataset = training_dataset
    
    # Use regular batch-based loading
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=train_batch_size,
        num_workers=dataloader_num_workers,
        pin_memory=True,
        shuffle=True,
        multiprocessing_context="spawn" if dataloader_num_workers > 0 else None,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
    )

    loss_fn = EDMLossSR()
    
    # Get number of channels from our CBottle dataset
    out_channels = training_dataset.n_viirs_channels

    # the model takes in both local and global lr channels  
    local_lr_channels = training_dataset.n_gfs_channels
    global_lr_channels = training_dataset.n_gfs_channels
    model_config = cbottle.models.ModelConfigV1(
        architecture="unet_hpx1024_patch",
        condition_channels=2*(local_lr_channels + global_lr_channels),
        out_channels=out_channels,
        #position_embed_channels=2,
    )
    img_resolution = model_config.img_resolution
    model_config.level = 10
        
    net = cbottle.models.get_model(model_config)
    net.train().requires_grad_(True).cuda()
    net.cuda(LOCAL_RANK)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[LOCAL_RANK], find_unused_parameters=True
    )

    # optimizer
    params = list(filter(lambda kv: "pos_embed" in kv[0], net.named_parameters()))
    base_params = list(
        filter(lambda kv: "pos_embed" not in kv[0], net.named_parameters())
    )
    params = [i[1] for i in params]
    base_params = [i[1] for i in base_params]
    optimizer = torch.optim.SGD(
        [{"params": base_params}], lr=1e-7, momentum=0.9
        #[{"params": base_params}, {"params": params, "lr": 5e-4}], lr=1e-7, momentum=0.9
    )

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.6)
    tic = time.time()
    step = 0
    train_loss_list = []
    val_loss_list = []

    # load checkpoint
    path = find_latest_checkpoint(output_path)

    try:
        map_location = {
            "cuda:%d" % 0: "cuda:%d" % int(LOCAL_RANK)
        }  # map_location='cuda:{}'.format(self.params.local_rank)
        step = load_checkpoint(
            path,
            network=net,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location,
        )
        step = step + 1
        print(f"Loaded network and optimizer states from {path}")
        if WORLD_RANK == 0:
            for p in optimizer.param_groups:
                print(p["lr"], p["initial_lr"])
    except FileNotFoundError:
        if WORLD_RANK == 0:
            print("Could not load network and optimizer states")

    # training loop
    old_pos = None
    old_pos2 = None
    old_conv = None
    old_conv2 = None
    running_loss = 0

    if WORLD_RANK == 0:
        print("training begin...", flush=True)

    while True:
        for batch in training_loader:

            # Update step
            step += 1

            # Zero grad
            optimizer.zero_grad()

            # Unpack batch
            viirs_patch = batch['viirs_patch'] # (b, 7, 128, 128)
            viirs_patch_mask = batch['viirs_patch_mask'] # (b, 1, 128, 128)
            viirs_patch_coords = batch['viirs_patch_coords'] # (b, 1, 128, 128)
            gfs_patch = batch['gfs_patch'] # (b, 2, 32, 16, 16)
            gfs_lr = batch['gfs_lr'] # (b, 2, 32, 128, 128)
            lead_time = batch['metadata']['lead_time'] # (b, 1)

            # Get ltarget
            ltarget = viirs_patch.cuda()

            # Get llr
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
            llr = torch.concat(
                [
                    gfs_patch_0,
                    gfs_patch_1,
                    gfs_lr[:, 0, :, :, :],
                    gfs_lr[:, 1, :, :, :],
                ],
                dim=1
            ).cuda()

            # Make lpe
            lpe = viirs_patch_coords / (12 * (2 ** 13) ** 2.)
            lpe = torch.repeat_interleave(lpe, 20, dim=1).cuda()

            # Replace NaN with 0
            ltarget = torch.where(torch.isnan(ltarget), torch.zeros_like(ltarget), ltarget)
            llr = torch.where(torch.isnan(llr), torch.zeros_like(llr), llr)
            lpe = torch.where(torch.isnan(lpe), torch.zeros_like(lpe), lpe)

            ## debug plot
            #import matplotlib.pyplot as plt
            #for i in range(ltarget.shape[1]):
            #    plt.imshow(ltarget[0, i, :, :].cpu().numpy())
            #    plt.colorbar()
            #    plt.savefig(f"outputs/debug_ltarget_{i}.png")
            #    plt.close()
            #for i in range(llr.shape[1]):
            #    plt.imshow(llr[0, i, :, :].cpu().numpy())
            #    plt.colorbar()
            #    plt.savefig(f"outputs/debug_llr_{i}.png")
            #    plt.close()
            #for i in range(lpe.shape[1]):
            #    plt.imshow(lpe[0, i, :, :].cpu().numpy())
            #    plt.colorbar()
            #    plt.savefig(f"outputs/debug_lpe_{i}.png")
            #    plt.close()
            #exit()

            #if torch.isnan(ltarget).any() or torch.isnan(llr).any() or torch.isnan(lpe).any():
            #    print("NaN detected in ltarget, llr, or lpe")
            #    continue

            # Compute the loss and its gradients
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                loss = loss_fn(net, img_clean=ltarget, img_lr=llr, pos_embed=lpe)
            loss = loss.sum()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e6)
            optimizer.step()

            # avoid synchronizing gpu
            dist.all_reduce(loss)
            running_loss += loss.item()

            # Check for logging after legacy training step
            if step % log_freq == 0:
                #with torch.no_grad():
                #    val_running_loss = 0
                #    for val_batch in test_loader:
                #        count = 0
                #        for lpe, ltarget, llr in patch_iterator(
                #            val_batch, test_batch_size
                #        ):
                #            with torch.autocast(
                #                "cuda", dtype=torch.bfloat16, enabled=bf16
                #            ):
                #                loss = loss_fn(
                #                    net,
                #                    img_clean=ltarget,
                #                    img_lr=llr,
                #                    pos_embed=lpe,
                #                )
                #            loss = loss.sum()
                #            dist.all_reduce(loss)
                #            count += 1
                #            val_running_loss += loss
                #        break

                # print out (legacy format)
                if WORLD_RANK == 0:
                    train_loss_list.append(
                        running_loss / log_freq / WORLD_SIZE / train_batch_size
                    )
                    #val_loss_list.append(
                    #    val_running_loss
                    #    / len(test_loader)
                    #    / count
                    #    / WORLD_SIZE
                    #    / test_batch_size
                    #)
                    # Continue with legacy logging code (simplified)
                    #print(f"Legacy step {step} | train loss: {train_loss_list[-1]:.2e}, val loss: {val_loss_list[-1]:.2e}")
                    print(f"Legacy step {step} | train loss: {train_loss_list[-1]:.2e}")
                    
                    # Save checkpoint at specified frequency
                    if (step // log_freq) % checkpoint_freq == 0:
                        file_name = "cBottle-SR-{}.zip".format(step)
                        save_checkpoint(
                            os.path.join(output_path, file_name),
                            model_config=model_config,
                            network=net,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            step=step,
                            loss=train_loss_list,
                        )
                        print(f"Saved checkpoint: {file_name}")
                    running_loss = 0.0
                    
                if step >= num_steps:
                    return
                scheduler.step()

            if step >= num_steps:
                print("training finished!")
                return

            scheduler.step()


if __name__ == "__main__":
    # Load configuration
    config = load_config()
    training_config = parse_training_config(config)
    
    print("="*60)
    print("CBottle Super Resolution Training")
    print("="*60)
    print(f"Output path: {training_config['output_path']}")
    print(f"Log frequency: {training_config['log_freq']}")
    print(f"Checkpoint frequency: {training_config['checkpoint_freq']} (every {training_config['log_freq'] * training_config['checkpoint_freq']} steps)")
    print(f"Train batch size: {training_config['train_batch_size']}")
    print(f"Test batch size: {training_config['test_batch_size']}")
    print(f"Dataloader workers: {training_config['dataloader_num_workers']}")
    print(f"BF16 enabled: {training_config['bf16']}")
    print(f"Number of steps: {training_config['num_steps']}")
    print("="*60)
    
    train(
        output_path=training_config['output_path'],
        train_batch_size=training_config['train_batch_size'],
        test_batch_size=training_config['test_batch_size'],
        valid_min_samples=training_config['valid_min_samples'],
        num_steps=training_config['num_steps'],
        log_freq=training_config['log_freq'],
        checkpoint_freq=training_config['checkpoint_freq'],
        dataloader_num_workers=training_config['dataloader_num_workers'],
        bf16=training_config['bf16'],
    )

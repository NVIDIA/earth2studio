# HENS on AWS: setup and run guide

This guide documents the AWS and EC2 workflow used to run the NVIDIA Earth2Studio HENS-SFNO Hurricane Beryl proof of concept.

## 1. AWS resources

Recommended initial instance:

- EC2 `g6e.xlarge` for smoke tests (1 NVIDIA L40S GPU, 48 GB VRAM).
- Use `g6e.2xlarge` or `g6e.4xlarge` for larger multi-checkpoint runs because they provide more system RAM.
- The Deep Learning Base OSS Nvidia Driver GPU AMI for Ubuntu is suitable.
- Allocate approximately 512 GiB of encrypted `gp3` EBS storage for model files, caches, outputs and logs.

The instance needs an IAM role with permissions to read and write the required S3 bucket and, if using the automated runner, stop the EC2 instance. Prefer an EC2 instance role over storing AWS access keys on the instance.

Verify the role after connecting:

```bash
aws sts get-caller-identity
```

## 2. Connect from macOS using VS Code

Add an entry to `~/.ssh/config` on the Mac:

```sshconfig
Host hens-aws
    HostName ec2-YOUR-PUBLIC-DNS
    User ubuntu
    IdentityFile /path/to/hens-aws-key.pem
    IdentitiesOnly yes
```

In VS Code, use **Remote-SSH: Connect to Host... → hens-aws**. The public DNS name can change after stopping and restarting the instance.

## 3. Clone the repository

The repository is cloned once per instance. The HENS setup script is already inside the repository, so it should not clone the repository itself.

```bash
mkdir -p ~/projects
git clone --branch beryl-poc \
  https://github.com/dougrichardson/earth2studio.git \
  ~/projects/earth2studio
```

For HTTPS cloning, GitHub requires a personal access token when prompted for a password. Do not put the token in the URL or commit it to a file.

## 4. Install Earth2Studio and HENS

```bash
cd ~/projects/earth2studio/recipes/hens
bash scripts/setup_hens.sh
```

Set the environment paths:

```bash
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:$PWD/.venv/lib/python3.13/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"
```

## 5. Test the GPU and cuDNN

```bash
cd ~/projects/earth2studio/recipes/hens
uv run python scripts/test_cudnn.py
nvidia-smi
```

The test should report that CUDA is available, identify the GPU, and complete a cuDNN convolution successfully.

## 6. Download model files

Each HENS checkpoint package must be in its own directory containing `config.json`:

```text
hens_model_registry/
├── sfno_seed12/config.json
├── sfno_seed16/config.json
├── sfno_seed17/config.json
└── sfno_seed18/config.json
```

List available packages without downloading them:

```bash
wget -qO- \
  https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/ \
  | sed -n 's/.*href="\\([^"]*\\/\\)".*/\\1/p'
```

Verify the layout:

```bash
find hens_model_registry -name config.json -printf '%h\\n'
```

If the registry root itself contains `config.json`, HENS treats it as one package and ignores subdirectories. The root should contain the checkpoint directories, but not a package-level `config.json`.

## 7. Run HENS

```bash
cd ~/projects/earth2studio/recipes/hens
uv run python main.py --config-name=beryl_poc.yaml
```

For timing and logging:

```bash
/usr/bin/time -v \
  uv run python main.py --config-name=beryl_poc.yaml \
  2>&1 | tee run.log
```

Monitor GPU use in another terminal:

```bash
watch -n 2 nvidia-smi
```

`batch_size` controls how many ensemble members are processed simultaneously and affects GPU memory use.

## 8. Keep a run alive after closing the Mac

Use `tmux` while the EC2 instance remains running:

```bash
tmux new -s hens
cd ~/projects/earth2studio/recipes/hens
uv run python main.py --config-name=beryl_poc.yaml
```

Detach with `Ctrl+B`, then `D`. Reattach later:

```bash
tmux ls
tmux attach -t hens
```

Stopping the EC2 instance stops the running process. EBS data remains available after stopping, but compute charges stop.

## 9. Upload outputs to S3

The current output directory is:

```text
~/projects/earth2studio/recipes/hens/outputs_beryl_poc
```

Upload it with:

```bash
aws s3 cp \
  ~/projects/earth2studio/recipes/hens/outputs_beryl_poc/ \
  s3://hens-beryl/beryl-poc/outputs/ \
  --recursive \
  --region us-east-2
```

## 10. Run and stop automatically

Set instance details in the shell environment rather than committing them:

```bash
export INSTANCE_ID="i-xxxxxxxxxxxxxxxxx"
export AWS_REGION="us-east-2"
```

Then run the script inside `tmux`:

```bash
cd ~/projects/earth2studio/recipes/hens
tmux new -s hens
scripts/run_hens_and_stop_instance.sh
```

Before relying on automatic shutdown, test that the instance role can upload to S3 and stop the instance.

## 11. Resume an interrupted multi-checkpoint run

If a run completed some checkpoints and was interrupted, create a registry containing only the unfinished packages:

```bash
mkdir -p hens_model_registry_remaining
ln -s ../hens_model_registry/sfno_seed16 hens_model_registry_remaining/sfno_seed16
ln -s ../hens_model_registry/sfno_seed17 hens_model_registry_remaining/sfno_seed17
ln -s ../hens_model_registry/sfno_seed18 hens_model_registry_remaining/sfno_seed18
```

Point the YAML file at this directory and set the number of remaining checkpoints:

```yaml
forecast_model:
    registry: /home/ubuntu/projects/earth2studio/recipes/hens/hens_model_registry_remaining
    max_num_checkpoints: 3
```

## 12. Files that should not be committed

Do not commit `.venv/`, model checkpoints, ERA5 caches, NetCDF outputs, AWS credentials, GitHub tokens, private keys, instance-specific DNS names or instance IDs.

Commit scripts, YAML configurations, notebooks, documentation and small diagnostic files to the `beryl-poc` branch.

#!/usr/bin/env bash
set -euo pipefail

PROJECT="$HOME/projects/earth2studio"
HENS_DIR="$PROJECT/recipes/hens"
REGISTRY="$HENS_DIR/hens_model_registry"

sudo apt-get update
sudo apt-get install -y git curl wget build-essential

mkdir -p "$HOME/projects"

git clone https://github.com/dougrichardson/earth2studio.git "$PROJECT"

cd "$HENS_DIR"

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.13
uv venv --python 3.13
source .venv/bin/activate

# Install PyTorch before torch-harmonics is built.
uv pip install torch numpy hatchling editables

# Install Earth2Studio/HENS dependencies.
uv sync --no-build-isolation

mkdir -p "$REGISTRY"

# Download the HENS perturbation skill file.
wget --show-progress \
  --directory-prefix="$REGISTRY" \
  https://portal.nersc.gov/cfs/m4416/hens/d2m_sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16.nc

# Download one HENS forecast model package.
wget --recursive --no-parent --no-host-directories \
  --cut-dirs=5 --show-progress \
  --directory-prefix="$REGISTRY" \
  https://portal.nersc.gov/cfs/m4416/hens/earth2mip_prod_registry/sfno_linear_74chq_sc2_layers8_edim620_wstgl2-epoch70_seed16/

mkdir -p "$HENS_DIR/output/outputs_beryl_poc"

# Make CUDA/cuDNN libraries visible.
CUDA_LIB="$HENS_DIR/.venv/lib/python3.13/site-packages"
export LD_LIBRARY_PATH="$CUDA_LIB/nvidia/cudnn/lib:$CUDA_LIB/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}"

cat >> ~/.bashrc <<'EOF'
export PATH="$HOME/.local/bin:$PATH"
EOF

echo "Environment checks:"
uv run python -c "import earth2studio; print('Earth2Studio imported')"
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

echo "Testing cuDNN:"
uv run python - <<'PY'
import torch

print("cuDNN:", torch.backends.cudnn.version())

x = torch.randn(1, 3, 128, 128, device="cuda")
layer = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
y = layer(x)

print("cuDNN convolution succeeded")
print("Output shape:", tuple(y.shape))
PY

echo "Setup and cuDNN test complete."

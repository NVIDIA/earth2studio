# Origin Datum Earth-2 Research Setup Guide

**Project:** OD-EARTH2-001 / EO-OBSCURA
**Target Platform:** Windows 10/11 with WSL2 (Ubuntu 24.04 LTS)
**Budget Standard:** $0 (Free open-source software, public AWS Open Data NOAA GFS)

---

## 1. Prerequisites & Architecture

NVIDIA Earth2Studio requires scientific C/Cython dependencies (such as `pygrib` and ECMWF `eccodes`) that are notoriously complex to compile natively on Windows. For reproducible scientific execution, **WSL2 (Ubuntu)** provides a native Linux environment while accessing files on Windows storage (`/mnt/d/...`).

### Hardware Profile Tested
- **CPU:** AMD Ryzen 5 3600 (6 Cores / 12 Threads)
- **RAM:** 32 GB Host / 15.6 GiB WSL2
- **GPU:** AMD Radeon RX 9060 XT (Non-CUDA)
- **Execution Mode:** Automatic CPU fallback for PyTorch inference

---

## 2. Setting Up WSL2 & Dependencies

### Step 1: Open PowerShell as Administrator & Ensure WSL2
```powershell
wsl --status
# If WSL2 is not installed:
wsl --install -d Ubuntu
```

### Step 2: Launch Ubuntu Shell
```bash
wsl -d Ubuntu
```

### Step 3: Install `uv` Package Manager in Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

---

## 3. Clone Repository & Setup Virtual Environment

Navigate to the workspace:
```bash
cd "/mnt/d/programming/Blockchain development/Z-space/earth2studio"
git checkout feat/od-earth2-001
```

Create and populate the virtual environment:
```bash
# Create Python 3.12 / 3.13 venv
uv venv --python 3.13 .venv
source .venv/bin/activate

# Install CPU PyTorch wheel
UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu uv pip install torch torchvision --python .venv

# Install Earth2Studio in editable mode
uv pip install -e . --python .venv

# Install scientific and testing tools
uv pip install pytest pyyaml matplotlib --python .venv
```

---

## 4. Origin Datum Research CLI Usage

The isolated research package `od_experiments.earth2_001` provides four core subcommands:

### 1. Environment Doctor
Audits host operating system, CPU, memory, GPU/accelerator, zero-cost cloud verification, and editable Earth2Studio status:
```bash
python -m od_experiments.earth2_001.cli doctor
# For machine-readable JSON:
python -m od_experiments.earth2_001.cli doctor --json
```

### 2. Component Inspection
Inspects supported prognostic models, data sources, and variable schemas without downloading checkpoints:
```bash
python -m od_experiments.earth2_001.cli inspect
```

### 3. Baseline Forecast Execution
Executes the reproducible baseline forecast workflow, outputting Zarr stores, a scientifically labeled PNG plot, and a cryptographic `provenance_manifest.json`:
```bash
python -m od_experiments.earth2_001.cli run
```

### 4. Scientific Validation
Evaluates Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Bias against reference observations:
```bash
python -m od_experiments.earth2_001.cli validate
```

---

## 5. Automated Verification Test Suite

Run the full automated test suite covering configuration, security scrubber, provenance manifests, and metric computations:
```bash
pytest od_experiments/tests -v
```

---

## 6. Zero-Cost Verification & Security

- **No Paid Cloud Infrastructure:** All forecast simulations use either local CPU baselines or anonymous public data buckets (`s3://noaa-gfs-bdp-pds` with `skip_signature=True`).
- **Secret Scrubber:** All manifests and logs automatically redact sensitive API keys, AWS credentials, and tokens.

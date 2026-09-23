<!-- markdownlint-disable MD013 -->
# StormCast CONUS environment

This setup is only for the connector's **StormCast CONUS workflow**. It includes SFNO because SFNO
is the default conditioning model; it is not the standalone SFNO workflow's environment.

The recipe is pinned to a known-good Linux CUDA 13.0 stack:

- Python 3.11–3.13
- PyTorch 2.10.0 and torchvision 0.25.0
- NATTEN 0.21.5
- pinned PhysicsNeMo, Makani, and torch-harmonics revisions

The installer owns the package ordering and repairs Makani's CUDA dependency changes. No NGC
container or Apex build is required.

## Install

From the recipe directory:

```bash
bash scripts/install_stormcast_environment.sh
```

Verify the resolved GPU stack:

```bash
.venv/bin/python -c \
  "import torch, torchvision, natten, makani; \
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Running

```bash
.venv/bin/python main.py --dry-run
.venv/bin/python main.py --once
.venv/bin/python main.py
```

By default, model weights are downloaded from Hugging Face. Set `model.path` to a local checkpoint
directory to use pre-staged StormCast weights; it must contain `metadata.nc`. HRRR and GFS inputs
are still downloaded at runtime.

## Notes

- Apex is optional. When it is absent, `StormCastCONUS.load_model()` uses the compatible
  `torch.nn.LayerNorm` implementation while loading the checkpoint.
- Warp CUDA error 100 ("no CUDA-capable device") prints at import from `warp-lang` (a transitive
  physicsnemo dependency). It is non-fatal for model loading, confirmed by a successful `--dry-run`.
- For a different CUDA or PyTorch version, update the matching PyTorch, torchvision, and NATTEN pins
  together.

#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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
#
# Build the StormCast CONUS + SFNO GPU environment for this recipe.
#
# The stack is NOT fully pip/uv-resolvable across platforms, so it is installed in a
# specific order with exact pins. The rationale for each non-obvious step is kept inline
# (rather than in the user guide) so a future edit does not drop a load-bearing step.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd -- "$RECIPE_DIR/../.." && pwd)"
VENV="$RECIPE_DIR/.venv"

TORCH_PIN="torch==2.10.0+cu130"
TVISION_PIN="torchvision==0.25.0+cu130"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

# --seed installs pip/setuptools/wheel. Required: the torch-harmonics step below uses
# --no-build-isolation, which builds against THIS interpreter and therefore needs
# setuptools already importable here. A plain `uv venv` ships none of them.
if [[ ! -x "$VENV/bin/python" ]]; then
    uv venv --seed --python 3.12 "$VENV"
fi
PY="$VENV/bin/python"

# --seed only takes effect at CREATION, so an existing .venv (e.g. built by an earlier
# revision of this script) can still be missing setuptools and would fail the
# --no-build-isolation build below. Assert the build backend unconditionally.
uv pip install --python "$PY" --quiet setuptools wheel

# uv index semantics: entries given with `--index` take PRIORITY over `--default-index`
# (which merely replaces PyPI). Under uv's default `first-index` strategy only the first index
# carrying a package name is consulted, so the CUDA index must be `--index`, leaving PyPI as
# the default for ordinary deps. Reversing these makes torch==2.10.0+cu130 unresolvable
# ("found on pypi.org, but not at the requested version").
uv_pip=(uv pip install --python "$PY")

# Preflight: torch-harmonics' setup.py compiles its optional DISCO CUDA kernels whenever
# `torch.cuda.is_available() and CUDA_HOME is not None`. If a CUDA toolkit is present but
# its version differs from the one torch was built against, nvcc aborts the build with
# "detected CUDA version (X) mismatches ... PyTorch (Y)" and, under `set -e`, kills this
# script partway. SFNO/StormCast need only the pure-PyTorch SHT/ISHT path, so when the
# toolkit does not match we hide the GPU for that one build and skip the extension.
# Fail-safe: build the extension ONLY when a matching toolkit is positively confirmed.
# Anything else (no nvcc, unreadable nvcc, unparseable or differing version) skips it.
# Skipping is always safe here -- neither SFNO nor StormCast uses the DISCO kernels -- so
# the cost of a false "skip" is nil, while a false "build" aborts the whole install.
TH_ENV=()
toolkit_matches_torch() {
    local nvcc="" want="13.0" have=""
    if command -v nvcc > /dev/null 2>&1; then
        nvcc="$(command -v nvcc)"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        nvcc=/usr/local/cuda/bin/nvcc
    else
        return 1
    fi
    have="$("$nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')" || return 1
    [[ -n "$have" && "$have" == "$want"* ]]
}

echo "==> connector runtime deps"
"${uv_pip[@]}" paho-mqtt==2.1.0 jsonschema==4.26.0 pyyaml==6.0.3

echo "==> torch/torchvision (cu130); NATTEN's +torch2100cu130 tag must match this exactly"
"${uv_pip[@]}" "$TORCH_PIN" "$TVISION_PIN" --index "$TORCH_INDEX"

# No cu130 wheel on PyPI, hence --find-links.
echo "==> natten"
"${uv_pip[@]}" natten==0.21.5+torch2100cu130 --find-links https://whl.natten.org

# Pinned SHA, not main and not PyPI: the StormCast checkpoint uses mdlus_file_version 0.2.0,
# which PyPI's newest (2.1.1) cannot load. Keep in sync with the repo root pyproject.toml
# [tool.uv.sources] pin for the stormcast-conus extra.
# Torch is reasserted here so PhysicsNeMo's resolution cannot move it off cu130 before the
# torch-harmonics build below compiles against it (a wrong-ABI build survives the repair
# step at the end, which reinstalls torch but does NOT rebuild torch-harmonics).
echo "==> nvidia-physicsnemo (pinned SHA)"
"${uv_pip[@]}" "$TORCH_PIN" \
    "nvidia-physicsnemo @ git+https://github.com/NVIDIA/physicsnemo.git@ff4cad2390e412b136e1739476506064d42eefba" \
    --index "$TORCH_INDEX"

if toolkit_matches_torch; then
    echo "==> torch-harmonics (building CUDA extension; toolkit matches torch)"
else
    echo "==> torch-harmonics (CUDA toolkit missing/mismatched -> skipping optional DISCO"
    echo "    extension; SFNO uses only the pure-PyTorch SHT/ISHT path)"
    TH_ENV=(env CUDA_VISIBLE_DEVICES=)
fi
"${TH_ENV[@]}" "${uv_pip[@]}" --no-build-isolation \
    "torch-harmonics @ git+https://github.com/NVIDIA/torch-harmonics.git@a632ca748a12bd9f74dbc1e00653317810991f74"

echo "==> makani (git only, not on PyPI)"
"${uv_pip[@]}" \
    "makani @ git+https://github.com/NVIDIA/makani.git@b38fcb2799d7dbc146fa60459f3f9823394a8bf1"

echo "==> earth2studio (editable, this repo)"
"${uv_pip[@]}" -e "$REPO_ROOT[stormcast-conus,sfno]"

# MANDATORY. Makani constrains torch<2.11 and pulls the cu12 runtime back in, which breaks
# the torchvision::nms op registration and NCCL, and bumps numpy (numba needs numpy < 2.5).
# Do not drop this step: the verification below is what proves it held.
echo "==> repair: restore the tested cu130 runtime after makani"
"${uv_pip[@]}" --reinstall-package torch --reinstall-package torchvision \
    "$TORCH_PIN" "$TVISION_PIN" --index "$TORCH_INDEX"
"${uv_pip[@]}" numpy==2.2.6

# Verify by EXERCISING the ops, not just importing them: `import torchvision` (and even
# `from torchvision.ops import nms`) still succeeds when the compiled extension is
# ABI-mismatched -- the failure only surfaces when the op is actually invoked.
echo "==> verify"
"$PY" - <<'PY'
import sys

import torch
import torchvision
from torchvision.ops import nms

import natten  # noqa: F401
import makani  # noqa: F401

fail = []
print(f"torch            {torch.__version__}")
print(f"torch.version.cuda {torch.version.cuda}")
print(f"torchvision      {torchvision.__version__}")

if not torch.version.cuda or not torch.version.cuda.startswith("13"):
    fail.append(f"expected a CUDA 13.x torch build, got {torch.version.cuda!r}")


def check_nms(dev: str) -> None:
    """Execute the op; importing it is not enough to detect an ABI mismatch."""
    try:
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], device=dev)
        nms(boxes, torch.tensor([0.9, 0.8], device=dev), 0.5)
        print(f"torchvision::nms OK on {dev} (executed, not just imported)")
    except Exception as exc:  # noqa: BLE001
        fail.append(f"torchvision::nms failed on {dev} -- the makani repair did not hold: {exc}")


# CPU always: this catches a broken op registration even on a machine with no visible GPU.
check_nms("cpu")

if not torch.cuda.is_available():
    # StormCast/SFNO are GPU-only (README: "A CUDA GPU is required; CPU is unsupported"),
    # so an unusable CUDA runtime is a failure here, not something to skip past.
    fail.append(
        "torch.cuda.is_available() is False -- this workflow requires a CUDA GPU. "
        "If the hardware is present, the CUDA runtime is broken (check the repair step)."
    )
else:
    check_nms("cuda")
    if not torch.distributed.is_nccl_available():
        fail.append("NCCL unavailable -- the cu12 runtime likely displaced cu130")
    else:
        print("nccl             OK")

if fail:
    print("\nFAILED:")
    for f in fail:
        print(f"  - {f}")
    sys.exit(1)
print("\nenvironment verified")
PY

echo
echo "StormCast environment ready: $VENV"
echo "Run: $PY $RECIPE_DIR/main.py --dry-run"
echo
echo "Note: the first run downloads model weights (SFNO ~6.9 GB, StormCast ~4 GB)."
echo "On a slow link raise the default 300 s package timeout, e.g.:"
echo "  export EARTH2STUDIO_PACKAGE_TIMEOUT=7200"

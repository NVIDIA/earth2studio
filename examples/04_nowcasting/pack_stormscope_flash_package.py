#!/usr/bin/env python3
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

"""Package the selected rotary GOES PDD-5 and MRMS/GLM PDD-7 checkpoints."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from earth2studio.models.auto import Package
from earth2studio.models.px._stormscope_flash.preconditioner import PDDModel
from earth2studio.models.px.stormscope_flash import (
    FLASH_CALLS,
    GOES_VARIABLES,
    _validate_flash_expert,
    load_flash_experts,
)

CHECKPOINTS = {
    "goes": {
        "high": "GOES/high/checkpoint_92800.pt",
        "middle": "GOES/middle/checkpoint_88800.pt",
        "low": "GOES/low/checkpoint_82400.pt",
    },
    "mrms": {
        "high": "MRMS/high/checkpoint_76400.pt",
        "middle": "MRMS/middle/checkpoint_84400.pt",
        "low": "MRMS/low/checkpoint_80000.pt",
    },
}
SOURCE_HASHES = {
    "goes": {
        "high": "3185d14dc34549b52e02d93bf31fdc89b2ae78b325e963786736894028bfda47",
        "middle": "01d1aa63ce440602fd9a56c5692450ed90e943f8e9745760680054bdf66f22e2",
        "low": "a7a1eef05f21ebf64b2eeca6989cbab60da4e471c3c91e8b8cd753730ff37696",
    },
    "mrms": {
        "high": "813d47717b7be8325c36d59292a4895d61a1b152bb1739fba5143de546e4034e",
        "middle": "f9767d0ac01ab61094559b9cc445c4c0e8a312a9ebe059bfc3af650182a9329d",
        "low": "0736cc36330ac87f7f71f0a5b1b7431a094f6f947aebf5063813bf01f4e2e8e1",
    },
}

ASSETS = (
    "lat.npy",
    "lon.npy",
    "topo.npy",
    "nexrad_proximity.npy",
    "mrms_coverage_mask.npy",
    "goes_means.npy",
    "goes_stds.npy",
    "mrms_means.npy",
    "mrms_stds.npy",
)


def sha256_file(path: Path) -> str:
    """Hash a checkpoint or asset without materializing it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def convert_checkpoint(
    source: Path, destination: Path, kind: str, region: str
) -> dict[str, Any]:
    """Convert selected training weights to a strictly verified .mdlus model."""
    checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    raw = checkpoint["pdd_metadata"]
    # Persist inference requirements, not training data paths or optimizer settings.
    meta = {
        key: raw[key]
        for key in (
            "version",
            "coordinate",
            "region",
            "inference_block_alignment",
            "sigma_grid",
            "block_boundaries",
            "training_ranges",
            "model_config",
        )
    }
    meta["config"] = {"student": {"sigma_data": raw["config"]["student"]["sigma_data"]}}
    config = {
        "height": 1024,
        "width": 1792,
        "patch_size": 4,
        "in_chans": 61 if kind == "goes" else 75,
        "base_out_chans": 8 if kind == "goes" else 3,
        "embed_dim": 768,
        "depth": 16,
        "num_heads": 6,
        "attn_kernel": 49,
        "pos_embedding_type": "rotary",
        "rope_theta": 10000.0,
        "use_nan_mask_tokens": kind == "goes",
    }
    model = PDDModel(config, meta)
    state = clean_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state, strict=True)
    _validate_flash_expert(model, kind, region)
    model.save(str(destination))
    # Round-trip every tensor, including the packed heads and schedule buffers.
    restored = PDDModel.from_checkpoint(str(destination), strict=True)
    _validate_flash_expert(restored, kind, region)
    actual = restored.state_dict()
    if state.keys() != actual.keys():
        raise ValueError(f"Checkpoint keys changed during conversion: {source}")
    for key in state:
        if not torch.equal(state[key], actual[key]):
            raise ValueError(f"Checkpoint tensor changed during conversion: {key}")
    return {
        "source_checkpoint": source.name,
        "source_sha256": sha256_file(source),
        "source_size_bytes": source.stat().st_size,
        "deployment_sha256": sha256_file(destination),
        "deployment_size_bytes": destination.stat().st_size,
        "deployment_format": "pdd-mdlus-v1",
        "tensor_parity": "exact",
    }


def clean_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove distributed wrapper prefixes without accepting key collisions."""
    prefixes = ("module.", "_orig_mod.", "_fsdp_wrapped_module.")
    result = {}
    for key, value in state.items():
        while any(key.startswith(prefix) for prefix in prefixes):
            key = next(
                key[len(prefix) :] for prefix in prefixes if key.startswith(prefix)
            )
        if key in result:
            raise ValueError(
                f"Duplicate checkpoint key after stripping wrappers: {key}"
            )
        result[key] = value
    return result


def main() -> None:
    """Build and strictly validate an atomic, optimizer-free deployment package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--assets-package", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.expanduser().resolve()
    if out.exists():
        raise FileExistsError(
            f"Refusing to replace existing package {out}; choose a new --out"
        )
    from earth2studio.models.px.stormscope import StormScopeBase

    assets = (
        Package(str(args.assets_package.expanduser().resolve()))
        if args.assets_package
        else StormScopeBase.load_default_package()
    )
    base = json.loads(Path(assets.resolve("registry.json")).read_text())
    radar = {
        "variables": ["refc", "refc_base", "glm_density"],
        "conditioning_vars": GOES_VARIABLES,
        "image_size": [1024, 1792],
        "spatial_downsample": 1,
        "sliding_window": True,
        "step_interval": 10,
        "n_steps": 6,
        "topo": True,
        "nexrad_proximity": True,
        "mrms_coverage_mask": True,
    }
    goes = {
        **radar,
        "variables": GOES_VARIABLES,
        "conditioning_vars": [],
        "nexrad_proximity": False,
        "mrms_coverage_mask": False,
    }
    registry = {
        "normalization": {k: base["normalization"][k] for k in ("goes", "mrms", "glm")}
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".flash-package-", dir=out.parent))
    try:
        asset_hashes = {}
        for name in ASSETS:
            shutil.copy2(assets.resolve(name), staging / name)
            asset_hashes[name] = sha256_file(staging / name)
        for kind, entry in (("goes", goes), ("mrms", radar)):
            entry.update(
                flash=True,
                region_calls=FLASH_CALLS[kind],
                checkpoints=[],
                description=f"Rotary StormScope {kind} flash PDD",
            )
            for region, filename in CHECKPOINTS[kind].items():
                original = (args.checkpoints_root / filename).resolve()
                if sha256_file(original) != SOURCE_HASHES[kind][region]:
                    raise ValueError(f"Source checkpoint hash mismatch: {original}")
                destination = staging / f"{kind}_{region}.mdlus"
                info = convert_checkpoint(original, destination, kind, region)
                content_path = destination.with_name(
                    f"{kind}_{region}-{info['deployment_sha256'][:16]}.mdlus"
                )
                destination.rename(content_path)
                entry["checkpoints"].append(
                    {
                        "name": region,
                        "path": content_path.name,
                        **info,
                    }
                )
                print(f"Packaged {kind}/{region}: {content_path.name}", flush=True)
            name = "3km_10min"
            registry[kind] = {"models": {name: entry}, "aliases": {}}
            loaded = load_flash_experts(Package(str(staging)), entry, kind)
            del loaded
            print(
                f"Strictly validated {kind} rotary architecture and PDD schedule",
                flush=True,
            )
        registry["flash_provenance"] = {
            "assets_sha256": asset_hashes,
            "glm_preprocessing": "bilinear_raw_counts_then_log1p",
            "coordinate": "edm_linear_t",
        }
        (staging / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
        os.rename(staging, out)
    except BaseException:
        shutil.rmtree(staging)
        raise
    print(f"Flash package ready: {out}")


if __name__ == "__main__":
    main()

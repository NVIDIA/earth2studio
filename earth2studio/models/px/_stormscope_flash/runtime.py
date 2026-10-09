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

import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from physicsnemo import Module

from earth2studio.models.auto import Package
from earth2studio.models.nn.stormscope_flash import (
    FlashModel,
    FlashPrecond,
    sigma_to_flow_time,
)
from earth2studio.models.px._stormscope_flash.region import resolve_region
from earth2studio.models.px._stormscope_flash.sampler import (
    FlashChainPlan,
    FlashExpert,
    build_flash_chain_plan,
    flash_sampler_chain,
    inference_block_boundaries,
    regional_sigma_grids,
)

if TYPE_CHECKING:
    from earth2studio.models.px.stormscope import StormScopeBase

FLASH_CALLS = {
    "goes": {"high": 1, "middle": 1, "low": 3},
    "mrms": {"high": 1, "middle": 1, "low": 5},
}
FLASH_INTERVALS = {
    "goes": {"high": 16, "middle": 32, "low": 80},
    "mrms": {"high": 16, "middle": 32, "low": 80},
}
GOES_VARIABLES = [
    "abi01c",
    "abi02c",
    "abi03c",
    "abi07c",
    "abi08c",
    "abi09c",
    "abi10c",
    "abi13c",
]


def validate_flash_entry(entry: Mapping[str, Any], kind: str) -> None:
    """Validate the physical-input and sampling contract of a flash package."""
    if kind not in FLASH_CALLS:
        raise ValueError("flash kind must be goes or mrms")
    expected = {
        "flash": True,
        "n_steps": 6,
        "image_size": [1024, 1792],
        "step_interval": 10,
        "spatial_downsample": 1,
        "sliding_window": True,
        "topo": True,
        "variables": (
            GOES_VARIABLES if kind == "goes" else ["refc", "refc_base", "glm_density"]
        ),
        "conditioning_vars": [] if kind == "goes" else GOES_VARIABLES,
        "nexrad_proximity": kind == "mrms",
        "region_calls": FLASH_CALLS[kind],
    }
    for key, value in expected.items():
        if entry.get(key) != value:
            raise ValueError(
                f"Invalid flash {kind} package field {key}: expected {value!r}"
            )
    if kind == "mrms" and not entry.get("mrms_coverage_mask"):
        raise ValueError("flash MRMS requires the training coverage mask")


def _validate_flash_expert(model: FlashModel, kind: str, region: str) -> None:
    meta = model.flash_metadata
    config = meta["model_config"]
    expected_grids = regional_sigma_grids(region_intervals=FLASH_INTERVALS[kind])
    expected_model = {
        "architecture": "dit",
        "patch_size": 4,
        "embed_dim": 768,
        "depth": 16,
        "num_heads": 6,
        "attn_kernel": 49,
        "alternate_attn": False,
        "use_transformer_engine": False,
        "use_fused_layernorm": False,
        "qk_norm": True,
        "pos_embed": "rotary",
        "rope_theta": 10000.0,
        "use_nan_mask_tokens": kind == "goes",
        "num_register_tokens": 0,
        "n_goes_channels": 8,
        "n_mrms_channels": 0 if kind == "goes" else 2,
        "n_glm_channels": 0 if kind == "goes" else 1,
        "latlon_input": True,
        "cos_zenith_input": True,
    }
    for key, value in expected_model.items():
        if config.get(key) != value:
            raise ValueError(
                f"{region}: incompatible model_config.{key}: {config.get(key)!r}"
            )
    if config.get("n_nldn_channels", 0) != 0:
        raise ValueError("MRMS Flash requires GLM checkpoints, not NLDN checkpoints")
    if (
        meta.get("version") != 2
        or meta.get("coordinate") != "edm_linear_t"
        or meta.get("region") != region
        or meta.get("inference_block_alignment") != 16
        or meta["config"]["student"]["sigma_data"] != 0.5
    ):
        raise ValueError(f"{region}: incompatible Flash metadata")
    state = model.state_dict()
    sigmas = torch.as_tensor(meta["sigma_grid"], dtype=torch.float32)
    if sigmas.shape != expected_grids[region].shape or not torch.allclose(
        sigmas, expected_grids[region], rtol=2e-5, atol=1e-6
    ):
        raise ValueError(f"{region}: unexpected trained sigma grid")
    for key, expected in (
        ("sigma_grid", sigmas),
        ("time_grid", sigma_to_flow_time(sigmas)),
        ("block_boundaries", torch.tensor(meta["block_boundaries"])),
    ):
        if (
            key not in state
            or state[key].shape != expected.shape
            or not torch.allclose(state[key], expected, rtol=1e-6, atol=1e-7)
        ):
            raise ValueError(f"{region}: metadata and saved {key} disagree")
    boundaries = inference_block_boundaries(
        sigmas.numel() - 1, FLASH_CALLS[kind][region], alignment=16
    ).tolist()
    ranges = {int(k): int(v) for k, v in meta["training_ranges"].items()}
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if start not in ranges or end > ranges[start]:
            raise ValueError(
                f"{region}: flash block {start}:{end} exceeds trained support"
            )
    expected_constructor = {
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
    if model.model_config != expected_constructor:
        raise ValueError(f"{region}: incompatible serialized DiT constructor")


def _validate_checkpoint_file(path: Path, spec: Mapping[str, Any]) -> None:
    if path.suffix != ".mdlus" or spec.get("deployment_format") != "flash-mdlus-v1":
        raise ValueError(
            "Flash requires a .mdlus deployment package; run the package converter"
        )
    if path.stat().st_size != spec["deployment_size_bytes"]:
        raise ValueError(f"Packaged checkpoint size mismatch: {path}")
    expected_hash = spec["deployment_sha256"]
    if os.environ.get("EARTH2STUDIO_VERIFY_CHECKPOINT_HASH", "0").lower() in (
        "1",
        "true",
        "yes",
    ):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected_hash:
            raise ValueError(f"Packaged checkpoint hash mismatch: {path}")


def load_flash_experts(
    package: Package, entry: Mapping[str, Any], kind: str
) -> list[dict[str, Any]]:
    """Load three .mdlus experts and validate architecture and trained block support."""
    validate_flash_entry(entry, kind)
    specs = entry["checkpoints"]
    if len(specs) != 3 or {s["name"] for s in specs} != {"high", "middle", "low"}:
        raise ValueError("flash requires exactly high, middle, and low checkpoints")
    stages = []
    for spec in sorted(
        specs, key=lambda item: ("high", "middle", "low").index(item["name"])
    ):
        region = spec["name"]
        path = Path(package.resolve(spec["path"]))
        _validate_checkpoint_file(path, spec)
        model = Module.from_checkpoint(str(path), strict=True)
        if not isinstance(model, FlashModel):
            raise TypeError("Flash checkpoint must contain a FlashModel")
        _validate_flash_expert(model, kind, region)
        model.eval().requires_grad_(False)
        stages.append(
            {
                "name": region,
                "model": model,
                "sigma_min": float(model.sigma_grid[-1]),
                "sigma_max": float(model.sigma_grid[0]),
            }
        )
    return stages


def configure_region(
    model: "StormScopeBase", region: Mapping[str, Sequence[float]] | None, padding: int
) -> None:
    """Resolve and retain real context on the Flash model input grid."""
    model.region_info = resolve_region(
        model._lat_cpu_copy, model._lon_cpu_copy, region, padding
    )
    y0, y1, x0, x1 = model.region_info.input_bounds
    for name in (
        "latitudes",
        "longitudes",
        "valid_mask",
        "conditioning_valid_mask",
        "topo",
        "nexrad_proximity",
        "mrms_coverage_mask",
    ):
        value = model._buffers.get(name)
        if value is not None:
            model.register_buffer(name, value[y0:y1, x0:x1].clone())
    model.y = model.y[y0:y1].copy()
    model.x = model.x[x0:x1].copy()
    model._lat_cpu_copy = model.latitudes.cpu().numpy()
    model._lon_cpu_copy = model.longitudes.cpu().numpy()
    for expert in model.stage_models:
        cast(FlashPrecond, expert).model.set_image_size(*model.region_info.input_shape)


class FlashRuntime:
    """Private sampling and missing-data policy for existing StormScope classes."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.plan: FlashChainPlan | None = None
        self.region_calls = dict(FLASH_CALLS[kind])
        self.total_nfe = sum(self.region_calls.values())

    def normalize_state(
        self, model: "StormScopeBase", x: torch.Tensor, normalized: torch.Tensor
    ) -> torch.Tensor:
        """Apply trained missing-observation handling after baseline normalization."""
        if self.kind == "goes":
            # The source DiT learned mask tokens for missing observation patches.
            missing = torch.isnan(x).flatten(0, 1).any(dim=(1, 2)) | ~model.valid_mask
            for expert in model.stage_models:
                module = cast(FlashPrecond, getattr(expert, "_orig_mod", expert))
                module.model.set_nan_pixel_mask(missing)
            normalized = torch.where(torch.isnan(normalized), 0.0, normalized)
        return normalized

    def sample(
        self,
        model: "StormScopeBase",
        latents: torch.Tensor,
        condition: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the Flash trajectory with cached fused heads in FP32."""
        experts = {
            name: cast(FlashExpert, expert)
            for name, expert in zip(("high", "middle", "low"), model.stage_models)
        }
        if self.plan is None:
            # Packed heads and flow coordinates must remain FP32 under ambient AMP.
            with torch.autocast(device_type=latents.device.type, enabled=False):
                self.plan = build_flash_chain_plan(
                    experts,
                    total_nfe=self.total_nfe,
                    region_calls=self.region_calls,
                    alignment=16,
                )
        with torch.autocast(
            device_type=latents.device.type, dtype=model.amp_dtype, enabled=model.amp
        ):
            result = flash_sampler_chain(
                experts,
                latents,
                condition,
                total_nfe=self.total_nfe,
                plan=self.plan,
            )
        if not torch.isfinite(result).all():
            raise FloatingPointError("flash forecast contains non-finite values")
        return result

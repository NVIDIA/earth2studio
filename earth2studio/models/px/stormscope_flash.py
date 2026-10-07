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
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from fsspec.implementations.cache_mapper import BasenameCacheMapper

from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.models.auto import Package
from earth2studio.models.px._stormscope_flash.preconditioner import (
    FlashModel,
    FlashPrecond,
    sigma_to_flow_time,
)
from earth2studio.models.px._stormscope_flash.region import RegionInfo, resolve_region
from earth2studio.models.px._stormscope_flash.sampler import (
    FlashChainPlan,
    FlashExpert,
    build_flash_chain_plan,
    flash_sampler_chain,
    inference_block_boundaries,
    regional_sigma_grids,
)
from earth2studio.models.px.stormscope import (
    StormScopeBase,
    StormScopeGOES,
    StormScopeMRMS,
)
from earth2studio.utils.imports import OptionalDependencyFailure
from earth2studio.utils.type import CoordSystem

try:
    from physicsnemo import Module
except ImportError:
    OptionalDependencyFailure("stormscope-flash")
    Module = None


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


class _StormScopeFlash(StormScopeBase):
    _FLASH_KIND: str
    latitudes: torch.Tensor
    longitudes: torch.Tensor
    valid_mask: torch.Tensor
    _lat_cpu_copy: np.ndarray
    _lon_cpu_copy: np.ndarray
    y: np.ndarray
    x: np.ndarray

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._flash_plan: FlashChainPlan | None = None
        self.region_info: RegionInfo
        super().__init__(*args, **kwargs)
        self.sampler_args: dict[str, float | int] = {}
        self.amp_dtype = torch.bfloat16
        self.region_calls = dict(FLASH_CALLS[self._FLASH_KIND])
        self.total_nfe = sum(self.region_calls.values())
        if [s.get("name") for s in self.model_spec] != ["high", "middle", "low"]:
            raise ValueError("flash requires ordered high, middle, low experts")

    @classmethod
    def load_default_package(cls, *, revision: str | None = None) -> Package:
        """Download and cache the selected Flash package from Hugging Face.

        ``revision`` can select a review commit; the released default is pinned
        to the verified checkpoint commit.
        """
        if revision is None:
            revision = "34a61472c7c0eadc914fb73021b11017b97328fa"
        return Package(
            f"hf://nvidia/stormscope-goes-mrms@{revision}",
            cache_options={
                "cache_storage": Package.default_cache(f"stormscope_flash/{revision}"),
                "cache_mapper": BasenameCacheMapper(directory_levels=2),
            },
        )

    @staticmethod
    def _load_registry(package: Package) -> dict[str, Any]:
        """Select Flash variants without changing the shared teacher registry."""
        registry = StormScopeBase._load_registry(package)
        for kind in ("goes", "mrms"):
            if kind not in registry:
                continue
            section = registry[kind]
            models = {
                name: entry
                for name, entry in section["models"].items()
                if entry.get("flash") is True
            }
            section["models"] = models
            section["aliases"] = {
                alias: target
                for alias, target in section.get("aliases", {}).items()
                if target in models
            }
            if "3km_10min_flash" in models:
                section["aliases"]["3km_10min"] = "3km_10min_flash"
        return registry

    @classmethod
    def _load_checkpoints(
        cls, package: Package, entry: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return load_flash_experts(package, entry, cls._FLASH_KIND)

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Any:
        self._flash_plan = None
        # PyTorch's Module._apply has no type annotations in supported runtimes.
        return super()._apply(fn, recurse=recurse)  # type: ignore[no-untyped-call]

    def compile_experts(self, mode: str = "default") -> None:
        """Compile staged experts, invalidating cached fused projection tensors."""
        super().compile_experts(mode)
        self._flash_plan = None

    def _set_region(
        self, region: Mapping[str, Sequence[float]] | None, padding: int
    ) -> None:
        self.region_info = resolve_region(
            self._lat_cpu_copy, self._lon_cpu_copy, region, padding
        )
        y0, y1, x0, x1 = self.region_info.input_bounds
        for name in (
            "latitudes",
            "longitudes",
            "valid_mask",
            "conditioning_valid_mask",
            "topo",
            "nexrad_proximity",
            "mrms_coverage_mask",
        ):
            value = self._buffers.get(name)
            if value is not None:
                self.register_buffer(name, value[y0:y1, x0:x1].clone())
        self.y = self.y[y0:y1].copy()
        self.x = self.x[x0:x1].copy()
        self._lat_cpu_copy = self.latitudes.cpu().numpy()
        self._lon_cpu_copy = self.longitudes.cpu().numpy()
        for expert in self.stage_models:
            cast(FlashPrecond, expert).model.set_image_size(
                *self.region_info.input_shape
            )
        self._flash_plan = None

    def crop_output(
        self, forecast: torch.Tensor, coords: CoordSystem, *, mask: bool = True
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Crop saved/displayed output to the requested region; retain full state for next_input."""
        return self.region_info.crop_output(forecast, coords, mask=mask)

    def _normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        normalized = super()._normalize_state(x)
        if self._FLASH_KIND == "goes":
            # The source DiT learned mask tokens for missing observation patches.
            missing = torch.isnan(x).flatten(0, 1).any(dim=(1, 2)) | ~self.valid_mask
            for expert in self.stage_models:
                module = cast(FlashPrecond, getattr(expert, "_orig_mod", expert))
                module.model.set_nan_pixel_mask(missing)
            normalized = torch.where(torch.isnan(normalized), 0.0, normalized)
        return normalized

    def normalize_conditioning(
        self, conditioning: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Normalize conditioning, preserving the trained missing-GOES convention."""
        normalized = super().normalize_conditioning(conditioning)
        # MRMS has no learned GOES mask tokens: training zero-filled missing GOES
        # conditioning after normalization. Infinities still fail baseline guards.
        if self._FLASH_KIND == "mrms" and normalized is not None:
            normalized = torch.where(torch.isnan(normalized), 0.0, normalized)
        return normalized

    def _edm_sampler(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        experts = {
            name: cast(FlashExpert, expert)
            for name, expert in zip(("high", "middle", "low"), self.stage_models)
        }
        if self._flash_plan is None:
            # Packed heads and flow coordinates must remain FP32 under ambient AMP.
            with torch.autocast(device_type=latents.device.type, enabled=False):
                self._flash_plan = build_flash_chain_plan(
                    experts,
                    total_nfe=self.total_nfe,
                    region_calls=self.region_calls,
                    alignment=16,
                )
        with torch.autocast(
            device_type=latents.device.type, dtype=self.amp_dtype, enabled=self.amp
        ):
            result = flash_sampler_chain(
                experts,
                latents,
                condition,
                total_nfe=self.total_nfe,
                plan=self._flash_plan,
            )
        if not torch.isfinite(result).all():
            raise FloatingPointError("flash forecast contains non-finite values")
        return result


class StormScopeGOESFlash(_StormScopeFlash, StormScopeGOES):
    """StormScope Flash satellite forecasts: five backbone calls per ten-minute lead.

    Uses the baseline physical-unit and six-frame-history interfaces. Download
    the package with :meth:`load_default_package`. A geographic
    ``region`` selects real input context with at least 200 pixels per axis;
    omitted bounds use the full domain. ``region_info`` records the resolved
    geometry. Call :meth:`crop_output` only for output, never for feedback.

    Badges
    ------
    region:na class:nwc product:sat gpu:40gb
    """

    _FLASH_KIND = "goes"

    @classmethod
    def load_model(
        cls,
        package: Package,
        model_name: str = "3km_10min",
        conditioning_data_source: DataSource | ForecastSource | None = None,
        amp: bool = True,
        compile: bool = False,
        *,
        region: Mapping[str, Sequence[float]] | None = None,
        padding: int = 25,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> "StormScopeGOESFlash":
        """Load StormScope GOES Flash on the full grid or a latitude/longitude rectangle.

        Parameters
        ----------
        package : Package
            StormScope Flash package returned by :meth:`load_default_package`.
        model_name : str, optional
            Registry variant, by default ``3km_10min``.
        conditioning_data_source : DataSource | ForecastSource | None, optional
            Optional source, following the baseline API. GOES is observation-only.
        amp : bool, optional
            Enable network autocast with ``amp_dtype``, by default True.
            State and fused heads remain FP32.
        compile : bool, optional
            Compile the region-specific experts, by default False.
        region : Mapping[str, Sequence[float]] | None, optional
            ``{"lat": (south, north), "lon": (west, east)}``, or full grid if None.
            Signed and 0–360 longitudes are accepted within the CONUS footprint.
        padding : int, optional
            Requested real context per side, by default 25. Clipped at edges.
        amp_dtype : torch.dtype, optional
            Network autocast dtype, by default torch.bfloat16. FP16 is also supported;
            state and fused projections stay FP32.

        Returns
        -------
        StormScopeGOESFlash
            Frozen inference model on CPU, with its resolved ``region_info``.
        """
        factory = cast(Callable[..., StormScopeGOESFlash], super().load_model)
        model = factory(
            package,
            model_name=model_name,
            conditioning_data_source=conditioning_data_source,
            amp=amp,
            compile=False,
        )
        if amp_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("amp_dtype must be torch.float16 or torch.bfloat16")
        model.amp_dtype = amp_dtype
        model._set_region(region, padding)
        if compile:
            model.compile_experts()
        return model


class StormScopeMRMSFlash(_StormScopeFlash, StormScopeMRMS):
    """StormScope Flash MRMS/GLM forecasts: seven backbone calls per ten-minute lead.

    State variables are ``refc``, ``refc_base`` and ``glm_density``. GOES history
    provides external conditioning. GLM uses the baseline order: bilinear raw
    counts, then log1p normalization. Coupled rollouts retain predicted GLM.
    Regional input and output behavior follows :class:`StormScopeGOESFlash`.

    Badges
    ------
    region:na class:nwc product:radar gpu:40gb
    """

    _FLASH_KIND = "mrms"

    @classmethod
    def load_model(
        cls,
        package: Package,
        model_name: str = "3km_10min",
        conditioning_data_source: DataSource | ForecastSource | None = None,
        glm_data_source: DataSource | None = None,
        amp: bool = True,
        compile: bool = False,
        *,
        region: Mapping[str, Sequence[float]] | None = None,
        padding: int = 25,
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> "StormScopeMRMSFlash":
        """Load StormScope MRMS/GLM Flash, optionally with real regional context.

        Parameters
        ----------
        package : Package
            StormScope Flash package returned by :meth:`load_default_package`.
        model_name : str, optional
            Registry variant, by default ``3km_10min``.
        conditioning_data_source : DataSource | ForecastSource | None, optional
            GOES source for standalone forecasts; coupling can supply history directly.
        glm_data_source : DataSource | None, optional
            Gridded raw GLM counts, for example ``GOESGLMGrid(satellite="east")``.
        amp : bool, optional
            Enable network autocast with ``amp_dtype``, by default True.
            State and fused heads remain FP32.
        compile : bool, optional
            Compile region-specific experts, by default False.
        region : Mapping[str, Sequence[float]] | None, optional
            Latitude/longitude bounds, matching :meth:`StormScopeGOESFlash.load_model`.
        padding : int, optional
            Requested real context per side, by default 25. Clipped at edges.
        amp_dtype : torch.dtype, optional
            Network autocast dtype, by default torch.bfloat16. FP16 is also supported;
            state and fused projections stay FP32.

        Returns
        -------
        StormScopeMRMSFlash
            Frozen inference model on CPU with its resolved ``region_info``.
        """
        factory = cast(Callable[..., StormScopeMRMSFlash], super().load_model)
        model = factory(
            package,
            model_name=model_name,
            conditioning_data_source=conditioning_data_source,
            glm_data_source=glm_data_source,
            amp=amp,
            compile=False,
        )
        if amp_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("amp_dtype must be torch.float16 or torch.bfloat16")
        model.amp_dtype = amp_dtype
        model._set_region(region, padding)
        if compile:
            model.compile_experts()
        return model

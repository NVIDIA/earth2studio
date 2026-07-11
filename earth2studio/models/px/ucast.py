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

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from loguru import logger

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim, handshake_size
from earth2studio.utils.type import CoordSystem

LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

VARIABLES = [
    "msl",
    "u10m",
    "v10m",
    "t2m",
    "sst",
]
VARIABLES += [f"z{level}" for level in LEVELS]
VARIABLES += [f"q{level}" for level in LEVELS]
VARIABLES += [f"t{level}" for level in LEVELS]
VARIABLES += [f"u{level}" for level in LEVELS]
VARIABLES += [f"v{level}" for level in LEVELS]
VARIABLES += [f"w{level}" for level in LEVELS]

STATIC_FIELDS = ["land_sea_mask", "geopotential_at_surface"]
STATIC_VARIABLES = ["lsm", "z"]
UCAST_CHECKPOINT = "ucast.ckpt"
UCAST_WB2_DATASET = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-6h-240x121_equiangular_with_poles_conservative.zarr"
)
UCAST_STATIC_FIELD_TIME = np.datetime64("2020-01-01T00:00:00")


class _ConvInitKwargs(TypedDict):
    init_weight: float
    init_bias: float


class _BlockKwargs(TypedDict):
    channels_per_head: int
    dropout: float


_SEC_PER_DAY = 86400
_AVG_DAY_PER_YEAR = 365.24219
_WB2_NAMES = {
    "msl": "mean_sea_level_pressure",
    "u10m": "10m_u_component_of_wind",
    "v10m": "10m_v_component_of_wind",
    "t2m": "2m_temperature",
    "sst": "sea_surface_temperature",
}
_WB2_NAMES.update({f"z{level}": f"geopotential_{level}" for level in LEVELS})
_WB2_NAMES.update({f"q{level}": f"specific_humidity_{level}" for level in LEVELS})
_WB2_NAMES.update({f"t{level}": f"temperature_{level}" for level in LEVELS})
_WB2_NAMES.update({f"u{level}": f"u_component_of_wind_{level}" for level in LEVELS})
_WB2_NAMES.update({f"v{level}": f"v_component_of_wind_{level}" for level in LEVELS})
_WB2_NAMES.update({f"w{level}": f"vertical_velocity_{level}" for level in LEVELS})


def _conv2d_circular_height(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    padding: int,
) -> torch.Tensor:
    """Apply circular padding in the first spatial dimension only."""
    if padding == 0:
        return F.conv2d(x, weight, bias=bias)
    x = F.pad(x, (0, 0, padding, padding), mode="circular")
    return F.conv2d(x, weight, bias=bias, padding=(0, padding))


def _weight_init(shape: list[int], fan_in: int) -> torch.Tensor:
    return math.sqrt(1 / fan_in) * (torch.rand(*shape) * 2 - 1)


class Conv2d(torch.nn.Module):
    """U-CAST convolution layer with longitude-circular padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        up: bool = False,
        down: bool = False,
        init_weight: float = 1.0,
        init_bias: float = 0.0,
    ) -> None:
        super().__init__()
        if up and down:
            raise ValueError("Conv2d cannot upsample and downsample simultaneously")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.padding = 0

        if kernel == 0:
            self.weight = None
            self.bias = None
        else:
            fan_in = in_channels * kernel**2
            self.weight = torch.nn.Parameter(
                _weight_init([out_channels, in_channels, kernel, kernel], fan_in)
                * init_weight
            )
            self.bias = torch.nn.Parameter(
                _weight_init([out_channels], fan_in) * init_bias
            )
            self.padding = kernel // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        bias = self.bias

        if self.up:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.down:
            x = F.avg_pool2d(x, kernel_size=2)
        if weight is not None:
            x = _conv2d_circular_height(x, weight, None, self.padding)
        if bias is not None:
            x = x.add_(bias.reshape(1, -1, 1, 1))
        return x


class GroupNorm(torch.nn.Module):
    """Group norm using the same parameter names as the U-CAST checkpoint."""

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        min_channels_per_group: int = 4,
    ) -> None:
        super().__init__()
        num_groups = 32
        while (
            num_channels % num_groups != 0
            or num_channels // num_groups < min_channels_per_group
        ):
            num_groups //= 2
        self.num_groups = num_groups
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(num_channels))
        self.bias = torch.nn.Parameter(torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.group_norm(
            x,
            num_groups=self.num_groups,
            weight=self.weight.to(x.dtype),
            bias=self.bias.to(x.dtype),
            eps=self.eps,
        )


class AttentionOp(torch.autograd.Function):
    """Attention weight computation used by the U-CAST U-Net."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, q: torch.Tensor, k: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        del ctx
        return (
            torch.einsum(
                "ncq,nck->nqk",
                q.to(torch.float32),
                (k / math.sqrt(q.shape[1])).to(torch.float32),
            )
            .softmax(dim=2)
            .to(q.dtype)
        )


class UNetBlock(torch.nn.Module):
    """Residual U-Net block used by U-CAST."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int | None = None,
        channels_per_head: int = 64,
        dropout: float = 0.1,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )

        init: _ConvInitKwargs = {
            "init_weight": math.sqrt(1 / 3),
            "init_bias": math.sqrt(1 / 3),
        }
        init_zero: _ConvInitKwargs = {"init_weight": 0.0, "init_bias": 0.0}

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            **init,
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            **init_zero,
        )
        self.dropout = torch.nn.Dropout(p=dropout)

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **init,
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x
        x = self.conv0(F.silu(self.norm0(x)))
        x = self.conv1(self.dropout(F.silu(self.norm1(x))))
        x = x.add_(self.skip(orig) if self.skip is not None else orig)

        if self.num_heads:
            b, c, h, w = x.shape
            q, k, v = (
                self.qkv(self.norm2(x))
                .reshape(b * self.num_heads, c // self.num_heads, 3, -1)
                .unbind(2)
            )
            attn_w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", attn_w, v)
            a = a.reshape(b, self.num_heads, c // self.num_heads, h, w).reshape(
                b, c, h, w
            )
            x = self.proj(a).add_(x)

        return x


class DhariwalUNet(torch.nn.Module):
    """ADM U-Net architecture used by U-CAST checkpoints."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_channels: int = 320,
        channel_mult: tuple[int, ...] = (1, 2, 3, 4),
        num_blocks: int = 4,
        attn_levels: tuple[int, ...] = (2, 3),
        channels_per_head: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        block_kwargs: _BlockKwargs = {
            "channels_per_head": channels_per_head,
            "dropout": dropout,
        }
        init: _ConvInitKwargs = {
            "init_weight": math.sqrt(1 / 3),
            "init_bias": math.sqrt(1 / 3),
        }
        init_zero: _ConvInitKwargs = {"init_weight": 0.0, "init_bias": 0.0}

        img_resolution = 240
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            level_channels = int(model_channels * mult)
            if level == 0:
                cout = level_channels
                self.enc[f"{res}x{res}_conv"] = Conv2d(
                    in_channels=in_channels,
                    out_channels=cout,
                    kernel=3,
                    **init,
                )
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    down=True,
                    **block_kwargs,
                )
            for idx in range(num_blocks):
                cin = cout
                cout = level_channels
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=level in attn_levels,
                    **block_kwargs,
                )

        skips = [block.out_channels for block in self.enc.values()]
        self.dec = torch.nn.ModuleDict()
        for dec_block_i, mult in enumerate(reversed(channel_mult)):
            level = len(channel_mult) - 1 - dec_block_i
            res = img_resolution >> level
            level_channels = int(model_channels * mult)
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    attention=True,
                    **block_kwargs,
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    **block_kwargs,
                )
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock(
                    in_channels=cout,
                    out_channels=cout,
                    up=True,
                    **block_kwargs,
                )
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = level_channels
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock(
                    in_channels=cin,
                    out_channels=cout,
                    attention=level in attn_levels,
                    **block_kwargs,
                )

        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout,
            out_channels=out_channels,
            kernel=3,
            **init_zero,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        dynamical_condition: torch.Tensor | None = None,
        static_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts = [inputs]
        if dynamical_condition is not None:
            parts.append(dynamical_condition)
        if static_condition is not None:
            parts.append(static_condition)
        x = torch.cat(parts, dim=1) if len(parts) > 1 else inputs

        skips = []
        for block in self.enc.values():
            x = block(x)
            skips.append(x)

        for block in self.dec.values():
            if x.shape[1] != block.in_channels:
                skip = skips.pop()
                if skip.shape[-2:] != x.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear")
                x = torch.cat([x, skip], dim=1)
            x = block(x)

        return self.out_conv(F.silu(self.out_norm(x)))


def _wb2_name(variable: str) -> str:
    return _WB2_NAMES[variable]


def _extract_stat(ds: xr.Dataset, wb2_name: str) -> torch.Tensor:
    if wb2_name in ds:
        return torch.as_tensor(ds[wb2_name].values, dtype=torch.float32)

    var_name, level = wb2_name.rsplit("_", 1)
    if not level.isdigit():
        raise ValueError(f"Could not find statistic for {wb2_name}")
    return torch.as_tensor(
        ds[var_name].sel(level=int(level)).values,
        dtype=torch.float32,
    )


def _load_stat_tensor(package: Package, filename: str) -> torch.Tensor:
    with xr.open_dataset(package.resolve(f"stats/{filename}")) as ds:
        stats = [_extract_stat(ds, _wb2_name(var)) for var in VARIABLES]
    return torch.stack([stat.reshape(()) for stat in stats])


def _load_sst_fill_value(package: Package) -> float:
    with xr.open_dataset(package.resolve("stats/era5_min.nc")) as ds:
        return float(_extract_stat(ds, _wb2_name("sst")).item())


def _load_ema_state_dict(
    model: torch.nn.Module,
    checkpoint_path: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    ema_prefix = "model_ema."
    ema_buffers = {
        key[len(ema_prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(ema_prefix)
    }

    model_param_names = [
        key
        for key in state_dict
        if key.startswith("model.") and not key.startswith("model_ema.")
    ]
    ema_name_to_param = {
        key[len("model.") :].replace(".", ""): key[len("model.") :]
        for key in model_param_names
    }

    model_state = {}
    for ema_key, ema_val in ema_buffers.items():
        if ema_key in {"decay", "num_updates"}:
            continue
        if ema_key in ema_name_to_param:
            model_state[ema_name_to_param[ema_key]] = ema_val

    load_result = model.load_state_dict(model_state, strict=False)
    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys
    if missing or unexpected:
        raise RuntimeError(
            "Failed to load U-CAST EMA checkpoint with "
            f"{len(missing)} missing keys and {len(unexpected)} unexpected keys. "
            f"Missing sample: {missing[:5]}; unexpected sample: {unexpected[:5]}"
        )


def _ensure_latitude_is_ascending(ds: xr.Dataset) -> xr.Dataset:
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    if not (np.diff(ds[lat_name].values) > 0).all():
        return ds.reindex({lat_name: list(reversed(ds[lat_name].values))})
    return ds


def _normalize_static_array(static: np.ndarray) -> np.ndarray:
    mean = static.mean(axis=(-2, -1), keepdims=True)
    std = static.std(axis=(-2, -1), keepdims=True)
    std = np.maximum(std, 1e-6)
    return ((static - mean) / std).astype(np.float32, copy=False)


def _normalize_static_tensor(static: torch.Tensor) -> torch.Tensor:
    # Keep user-provided statics torch-native. The vanilla reference normalizes
    # this with NumPy, which is bit-accurate against the original script:
    # static_array = static.detach().cpu().numpy().astype(np.float32).copy()
    # mean = static_array.mean(axis=(-2, -1), keepdims=True)
    # std = static_array.std(axis=(-2, -1), keepdims=True)
    # normalized = ((static_array - mean) / std).astype(np.float32)
    # Torch reductions can differ by one float32 ulp from NumPy because the
    # reduction order and memory layout handling are not identical.
    mean = static.mean(dim=(-2, -1), keepdim=True)
    std = static.std(dim=(-2, -1), keepdim=True, unbiased=False)
    return (static - mean) / std


def _static_condition_from_input(
    x_static: torch.Tensor,
    batch_size: int,
    time_size: int,
    n_lon: int,
    n_lat: int,
) -> torch.Tensor:
    static = x_static.permute(0, 1, 2, 4, 3).reshape(
        batch_size * time_size, len(STATIC_VARIABLES), n_lon, n_lat
    )
    static = torch.flip(static, dims=(-1,))
    return _normalize_static_tensor(static.float()).to(dtype=x_static.dtype)


def _compute_static_condition(ds: xr.Dataset) -> torch.Tensor:
    ds = _ensure_latitude_is_ascending(ds)
    arrays = []
    for field in STATIC_FIELDS:
        data_array = ds[field]
        if "time" in data_array.dims:
            data_array = data_array.sel(time=UCAST_STATIC_FIELD_TIME)
        data = data_array.compute().values
        if data.ndim > 2:
            data = data[0]

        spatial_dims = [dim for dim in data_array.dims if dim != "time"]
        if spatial_dims[0] in ["latitude", "lat"]:
            data = data.T
        arrays.append(data.astype(np.float32))

    static = _normalize_static_array(np.stack(arrays, axis=0))
    return torch.from_numpy(static.transpose(0, 2, 1)).float()


def _compute_forcings(
    time_vals: np.ndarray,
    lon_vals: np.ndarray,
    n_lat: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    seconds = time_vals.astype("datetime64[s]").astype(np.int64).astype(np.float64)
    year_progress = np.mod(seconds / _SEC_PER_DAY / _AVG_DAY_PER_YEAR, 1.0).astype(
        np.float32
    )
    day_frac = np.mod(seconds, _SEC_PER_DAY) / _SEC_PER_DAY
    lon_offset = np.deg2rad(lon_vals) / (2.0 * np.pi)
    day_progress = np.mod(day_frac[:, None] + lon_offset[None, :], 1.0).astype(
        np.float32
    )

    yp_sin = torch.as_tensor(
        np.sin(2.0 * np.pi * year_progress), device=device, dtype=dtype
    )
    yp_cos = torch.as_tensor(
        np.cos(2.0 * np.pi * year_progress), device=device, dtype=dtype
    )
    dp_sin = torch.as_tensor(
        np.sin(2.0 * np.pi * day_progress), device=device, dtype=dtype
    )
    dp_cos = torch.as_tensor(
        np.cos(2.0 * np.pi * day_progress), device=device, dtype=dtype
    )

    n_time = len(time_vals)
    n_lon = len(lon_vals)
    forcing = torch.stack(
        [
            yp_sin[:, None, None].expand(n_time, n_lon, n_lat),
            yp_cos[:, None, None].expand(n_time, n_lon, n_lat),
            dp_sin[:, :, None].expand(n_time, n_lon, n_lat),
            dp_cos[:, :, None].expand(n_time, n_lon, n_lat),
        ],
        dim=1,
    )
    return forcing


class UCast(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """U-CAST 1.5 degree global probabilistic weather model.

    U-CAST is a 12-hour autoregressive U-Net forecaster trained on WeatherBench2
    ERA5. The model uses two history steps, 83 dynamic variables, four time/solar
    progress forcing channels, and two static fields. Ensemble members can be
    generated by adding an ensemble dimension ahead of ``time``; dropout remains
    active by default during inference so members diverge autoregressively.

    Note
    ----
    For additional information see the following resources:

    - https://arxiv.org/abs/2604.09041
    - https://github.com/Rose-STL-Lab/u-cast
    - https://huggingface.co/salv47/u-cast

    Parameters
    ----------
    model : torch.nn.Module
        U-CAST Dhariwal U-Net core.
    center : torch.Tensor
        Per-variable ERA5 means in ``VARIABLES`` order.
    scale : torch.Tensor
        Per-variable ERA5 standard deviations in ``VARIABLES`` order.
    residual_scale : torch.Tensor
        Per-variable residual standard deviations in ``VARIABLES`` order.
    static_condition : torch.Tensor
        Static fields in ``(field, lat, lon)`` order. Empty if
        ``preload_static_fields=False``.
    sst_fill_value : float
        Fill value used for SST NaNs over land before normalization.
    stochastic : bool
        Keep dropout layers active during inference, by default True.
    preload_static_fields : bool
        If True, static fields are fetched from WeatherBench2 at load time and
        cached. If False, these fields must be provided as input variables
        (``lsm`` and ``z``), allowing use of static fields from alternative
        sources for exact reproducibility, by default True.

    Badges
    ------
    region:global class:mrf product:wind product:temp product:atmos product:ocean year:2026 gpu:40gb
    """

    DT = np.timedelta64(12, "h")

    def __init__(
        self,
        model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
        residual_scale: torch.Tensor,
        static_condition: torch.Tensor,
        sst_fill_value: float,
        stochastic: bool = True,
        preload_static_fields: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.stochastic = stochastic
        self.preload_static_fields = preload_static_fields
        self.sst_index = VARIABLES.index("sst")

        self.register_buffer("center", center.float().view(-1, 1, 1))
        self.register_buffer("scale", scale.float().view(-1, 1, 1))
        self.register_buffer(
            "residual_to_normalized_scale",
            (residual_scale.float() / scale.float()).view(-1, 1, 1),
        )
        if preload_static_fields:
            self.register_buffer("static_condition", static_condition.float())
        else:
            self.register_buffer("static_condition", torch.empty(0))
        self.register_buffer("sst_fill_value", torch.tensor(float(sst_fill_value)))

        input_variables = (
            VARIABLES if preload_static_fields else VARIABLES + STATIC_VARIABLES
        )

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([-self.DT, np.timedelta64(0, "h")]),
                "variable": np.array(input_variables),
                "lat": np.linspace(90, -90, 121),
                "lon": np.linspace(0, 360, 240, endpoint=False),
            }
        )
        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([self.DT]),
                "variable": np.array(VARIABLES),
                "lat": np.linspace(90, -90, 121),
                "lon": np.linspace(0, 360, 240, endpoint=False),
            }
        )

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model."""
        return self._input_coords.copy()

    def _check_input_coords(self, input_coords: CoordSystem) -> None:
        """Validate input coordinates against the public U-CAST coordinate system."""
        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        input_variables = np.asarray(input_coords.get("variable", []))
        if not self.preload_static_fields and input_variables.shape[0] == len(
            VARIABLES
        ):
            target_input_coords["variable"] = np.array(VARIABLES)

        for i, key in enumerate(target_input_coords):
            handshake_dim(test_coords, key, i)
            if key not in ["batch", "time"]:
                handshake_coords(test_coords, target_input_coords, key)

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model."""
        self._check_input_coords(input_coords)
        output_coords = self._output_coords.copy()
        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            input_coords["lead_time"][-1] + output_coords["lead_time"]
        )
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default package for the U-CAST model."""
        package = Package(
            "hf://salv47/u-cast@775f2974b52b5beb8945ac3c60212ca25e0a13f5",
            cache_options={
                "cache_storage": Package.default_cache("ucast"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    def load_model(
        cls, package: Package, preload_static_fields: bool = True
    ) -> PrognosticModel:
        """Load U-CAST from a package.

        Parameters
        ----------
        package : Package
            Model package to load from.
        preload_static_fields : bool, optional
            If True, static fields (lsm and z) are fetched from WeatherBench2 at
            load time and cached. If False, these fields must be provided as
            input variables, allowing use of static fields from alternative
            sources for exact reproducibility with reference implementations,
            by default True.

        Returns
        -------
        PrognosticModel
            Loaded U-CAST model.
        """
        center = _load_stat_tensor(package, "era5_mean.nc")
        scale = _load_stat_tensor(package, "era5_std.nc")
        residual_scale = _load_stat_tensor(package, "era5_residual_std.nc")
        sst_fill_value = _load_sst_fill_value(package)
        # Fetch static fields from WeatherBench2 (only if preloading).
        if preload_static_fields:
            cache_path = Path(package.cache) / "static_condition.pt"
            if cache_path.exists():
                static_condition = torch.load(cache_path, weights_only=True)
            else:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(
                    "Loading U-CAST static fields from WeatherBench2 public ERA5 zarr"
                )
                from earth2studio.data.utils import obstore_zarr_store

                zstore = obstore_zarr_store(
                    UCAST_WB2_DATASET,
                    cache_storage=package.cache,
                    skip_signature=True,
                )
                ds = xr.open_zarr(zstore, zarr_format=2)
                try:
                    static_condition = _compute_static_condition(ds)
                finally:
                    ds.close()
                torch.save(static_condition, cache_path)
        else:
            static_condition = torch.empty(0)

        core_model = DhariwalUNet(
            in_channels=len(VARIABLES) * 2 + len(STATIC_FIELDS) + 4,
            out_channels=len(VARIABLES),
            model_channels=320,
            channel_mult=(1, 2, 3, 4),
            num_blocks=4,
            attn_levels=(2, 3),
            dropout=0.1,
        )
        _load_ema_state_dict(core_model, package.resolve(UCAST_CHECKPOINT))
        core_model.eval()

        return cls(
            core_model,
            center,
            scale,
            residual_scale,
            static_condition,
            sst_fill_value,
            preload_static_fields=preload_static_fields,
        )

    def _enable_inference_dropout(self) -> None:
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train(self.stochastic)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        sst = x[:, :, self.sst_index : self.sst_index + 1]
        sst = torch.where(
            torch.isnan(sst),
            self.sst_fill_value.to(dtype=x.dtype, device=x.device),
            sst,
        )
        x = torch.cat(
            [x[:, :, : self.sst_index], sst, x[:, :, self.sst_index + 1 :]],
            dim=2,
        )
        return (x - self.center.to(dtype=x.dtype)) / self.scale.to(dtype=x.dtype)

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.to(dtype=x.dtype) + self.center.to(dtype=x.dtype)

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        static_condition: torch.Tensor,
        x_norm: torch.Tensor | None = None,
        sst_mask: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._check_input_coords(coords)
        self._enable_inference_dropout()

        batch_size, time_size, history_size, n_variables, n_lat, n_lon = x.shape
        handshake_size(coords, "lead_time", history_size)
        handshake_size(coords, "variable", n_variables)
        handshake_size(coords, "lat", n_lat)
        handshake_size(coords, "lon", n_lon)

        model_shape = (
            batch_size * time_size,
            history_size,
            n_variables,
            n_lon,
            n_lat,
        )
        if x_norm is None:
            # U-CAST operates on (lon, lat) with south-to-north latitude. Keep the
            # public Earth2Studio convention north-to-south and flip only internally.
            x_model = x.permute(0, 1, 2, 3, 5, 4).reshape(model_shape)
            x_model = torch.flip(x_model, dims=(-1,))
            sst_mask = torch.isnan(x_model[:, -1, self.sst_index])
            x_norm = self._normalize(x_model)
        else:
            if x_norm.shape != model_shape:
                raise ValueError(
                    f"Expected normalized U-CAST state shape {model_shape}, got {tuple(x_norm.shape)}"
                )
            if sst_mask is None:
                raise ValueError("sst_mask is required when x_norm is provided")

        model_input = torch.cat([x_norm[:, 0], x_norm[:, 1]], dim=1)

        target_times = coords["time"] + coords["lead_time"][-1] + self.DT
        forcing = _compute_forcings(
            target_times,
            coords["lon"],
            n_lat,
            device=x.device,
            dtype=x.dtype,
        )
        forcing = forcing.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        forcing = forcing.reshape(batch_size * time_size, 4, n_lon, n_lat)

        static = static_condition.to(device=x.device, dtype=x.dtype)

        use_amp = x.device.type == "cuda"
        with torch.autocast(
            device_type=x.device.type, dtype=torch.float16, enabled=use_amp
        ):
            pred_residual = self.model(
                model_input,
                dynamical_condition=forcing,
                static_condition=static,
            )

        pred_norm = (
            pred_residual
            * self.residual_to_normalized_scale.to(
                device=pred_residual.device, dtype=pred_residual.dtype
            )
            + x_norm[:, -1]
        )
        pred = self._denormalize(pred_norm)

        pred_sst = pred[:, self.sst_index]
        pred[:, self.sst_index] = torch.where(
            sst_mask,
            self.sst_fill_value.to(dtype=pred.dtype, device=pred.device),
            pred_sst,
        )

        pred = torch.flip(pred, dims=(-1,))
        out = pred.reshape(batch_size, time_size, n_variables, n_lon, n_lat).permute(
            0, 1, 2, 4, 3
        )[:, :, None]
        if return_state:
            next_x_norm = torch.cat([x_norm[:, 1:], pred_norm[:, None]], dim=1)
            return out, next_x_norm, sst_mask
        return out

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs the 12-hour U-CAST prognostic model one step."""
        out_coords = self.output_coords(coords)
        batch_size, time_size, history_size, n_variables, n_lat, n_lon = x.shape
        handshake_size(coords, "lead_time", history_size)
        handshake_size(coords, "variable", n_variables)
        handshake_size(coords, "lat", n_lat)
        handshake_size(coords, "lon", n_lon)

        if self.preload_static_fields:
            static_condition = (
                self.static_condition.permute(0, 2, 1)
                .unsqueeze(0)
                .expand(batch_size * time_size, -1, -1, -1)
            )
        else:
            static_condition = _static_condition_from_input(
                x[:, :, 0, len(VARIABLES) :],
                batch_size,
                time_size,
                n_lon,
                n_lat,
            )
            x = x[:, :, :, : len(VARIABLES)]
            coords = coords.copy()
            coords["variable"] = self._output_coords["variable"].copy()
        return self._forward(x, coords, static_condition), out_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem]]:
        coords = coords.copy()
        self.output_coords(coords)
        batch_size, time_size, history_size, n_variables, n_lat, n_lon = x.shape
        handshake_size(coords, "lead_time", history_size)
        handshake_size(coords, "variable", n_variables)
        handshake_size(coords, "lat", n_lat)
        handshake_size(coords, "lon", n_lon)

        if self.preload_static_fields:
            static_condition = (
                self.static_condition.permute(0, 2, 1)
                .unsqueeze(0)
                .expand(batch_size * time_size, -1, -1, -1)
            )
        else:
            static_condition = _static_condition_from_input(
                x[:, :, 0, len(VARIABLES) :], batch_size, time_size, n_lon, n_lat
            )
            x = x[:, :, :, : len(VARIABLES)]
            coords["variable"] = self._output_coords["variable"].copy()

        out = x[:, :, 1:]
        out_coords = coords.copy()
        out_coords["lead_time"] = out_coords["lead_time"][1:]
        out_coords["variable"] = self._output_coords["variable"].copy()
        yield out, out_coords

        x_norm = None
        sst_mask = None

        while True:
            x, coords = self.front_hook(x, coords)
            out, x_norm, sst_mask = self._forward(
                x,
                coords,
                x_norm=x_norm,
                sst_mask=sst_mask,
                static_condition=static_condition,
                return_state=True,
            )
            out_coords = self.output_coords(coords)
            out, out_coords = self.rear_hook(out, out_coords)

            x = torch.cat([x[:, :, 1:], out], dim=2)
            coords["lead_time"] = np.array(
                [coords["lead_time"][-1], out_coords["lead_time"][-1]]
            )

            yield out, out_coords.copy()

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates an iterator for autoregressive U-CAST inference."""
        yield from self._default_generator(x, coords)

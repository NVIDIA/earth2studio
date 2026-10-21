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
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from physicsnemo import Module
from physicsnemo.models.dit import DiT as PhysicsNeMoDiT
from physicsnemo.nn.module.conditioning_embedders import DiTConditionEmbedder
from physicsnemo.nn.module.dit_layers import (
    DiTBlock,
    PatchEmbed2DTokenizer,
    ProjReshape2DDetokenizer,
    RopeNatten2DSelfAttention,
)
from physicsnemo.nn.module.rope import build_axial_rope_cos_sin_2d

from earth2studio.models.auto import Package
from earth2studio.models.nn.stormscope_flash import (
    FLASH_CALLS,
    FlashModel,
    build_flash_chain_plan,
    flash_sampler_chain,
    inference_block_boundaries,
    prepare_flash_goes_state,
    regional_sigma_grids,
    resolve_region,
)
from earth2studio.models.nn.stormscope_flash import FlashDiT as DiT
from earth2studio.models.px.stormscope import StormScopeGOES, StormScopeMRMS


def test_rotary_mask_and_state_dict_compatibility():
    dit = DiT(
        height=8,
        width=12,
        patch_size=2,
        in_chans=3,
        base_out_chans=1,
        embed_dim=16,
        depth=2,
        num_heads=2,
        pos_embedding_type="rotary",
        use_nan_mask_tokens=True,
        attn_kernel=3,
    )
    assert "tokenizer.pos_embed" not in dit.state_dict()
    assert all(block.attention.mask_token.shape == (1, 1, 16) for block in dit.blocks)
    mask = torch.zeros(2, 8, 12, dtype=torch.bool)
    mask[0, 1, 2] = True
    dit.set_nan_pixel_mask(mask)
    assert dit.invalid_token_mask_flat.shape == (2, 24)
    assert dit.invalid_token_mask_flat.sum() == 1
    assert dit.invalid_token_mask_flat[0, 1]
    assert "invalid_token_mask_flat" not in dit.state_dict()
    with pytest.raises(ValueError):
        dit.set_nan_pixel_mask(torch.zeros(7, 12))


@pytest.mark.parametrize(
    "device,dtype,amp_dtype",
    [
        ("cpu", torch.float32, torch.bfloat16),
        ("cpu", torch.bfloat16, torch.bfloat16),
        pytest.param(
            "cuda",
            torch.float32,
            torch.float16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "cuda",
            torch.float16,
            torch.float16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_flash_projections_preserve_fp32_under_amp(
    device, dtype, amp_dtype, monkeypatch
):
    dit = (
        DiT(
            height=8,
            width=12,
            patch_size=2,
            in_chans=3,
            base_out_chans=1,
            embed_dim=16,
            depth=1,
            num_heads=2,
            attn_kernel=3,
        )
        .to(device)
        .eval()
    )
    image = torch.randn(2, 3, 8, 12, device=device, dtype=dtype)
    with torch.no_grad(), torch.autocast(device, dtype=amp_dtype):
        actual, shape = dit.prepare_tokens(image)
        with torch.autocast(device, enabled=False):
            expected = (
                torch.nn.functional.conv2d(
                    image.float(),
                    dit.tokenizer.x_embedder.proj.weight,
                    dit.tokenizer.x_embedder.proj.bias,
                    stride=2,
                )
                .to(dtype)
                .flatten(2)
                .transpose(1, 2)
            )
    assert shape == (4, 6)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    features = torch.randn(2, 24, 16, device=device, dtype=dtype)
    monkeypatch.setattr(dit, "forward_features", lambda *args: features)
    with torch.no_grad(), torch.autocast(device, dtype=amp_dtype):
        actual_image = dit(image)
        with torch.autocast(device, enabled=False):
            expected_image = dit.unpatchify(
                torch.nn.functional.linear(
                    features.float(),
                    dit.detokenizer.proj_layer.output_projection.weight,
                    dit.detokenizer.proj_layer.output_projection.bias,
                ).to(dtype),
                8,
                12,
            )
    assert actual_image.dtype == dtype
    torch.testing.assert_close(actual_image, expected_image, rtol=0, atol=0)


def test_flash_native_components_and_checkpoint_names_round_trip():
    config = dict(
        height=8,
        width=12,
        patch_size=2,
        in_chans=3,
        base_out_chans=1,
        embed_dim=16,
        depth=2,
        num_heads=2,
        attn_kernel=3,
        use_nan_mask_tokens=True,
    )
    original = DiT(**config)
    assert isinstance(original, PhysicsNeMoDiT)
    assert type(original.tokenizer) is PatchEmbed2DTokenizer
    assert type(original.conditioning_embedder) is DiTConditionEmbedder
    assert type(original.detokenizer) is ProjReshape2DDetokenizer
    for i, block in enumerate(original.blocks):
        assert isinstance(block, DiTBlock)
        assert type(block.attention) is RopeNatten2DSelfAttention
        for norm in (block.attention.q_norm, block.attention.k_norm):
            assert isinstance(norm, torch.nn.LayerNorm)
            assert norm.elementwise_affine
            assert norm.eps == 1e-5
            assert norm.weight.shape == norm.bias.shape == (8,)
            with torch.no_grad():
                norm.weight.copy_(torch.linspace(0.75, 1.25, 8))
                norm.bias.copy_(torch.linspace(-0.1, 0.1, 8))
        with torch.no_grad():
            block.attention.mask_token.fill_(i + 0.125)
    state = original.state_dict()
    assert {name.split(".", 1)[0] for name in state} == {
        "tokenizer",
        "conditioning_embedder",
        "blocks",
        "detokenizer",
    }
    assert {
        "tokenizer.x_embedder.proj.weight",
        "conditioning_embedder.t_embedder.mlp.0.weight",
        "conditioning_embedder.t_embedder.mlp.2.weight",
        "detokenizer.proj_layer.output_projection.weight",
        "detokenizer.proj_layer.adaptive_modulation.1.weight",
    } <= state.keys()
    for i in range(config["depth"]):
        assert {
            f"blocks.{i}.attention.qkv.weight",
            f"blocks.{i}.attention.proj.weight",
            f"blocks.{i}.attention.q_norm.weight",
            f"blocks.{i}.attention.q_norm.bias",
            f"blocks.{i}.attention.k_norm.weight",
            f"blocks.{i}.attention.k_norm.bias",
            f"blocks.{i}.attention.mask_token",
            f"blocks.{i}.linear.layers.0.weight",
            f"blocks.{i}.linear.layers.2.weight",
            f"blocks.{i}.adaptive_modulation.1.weight",
        } <= state.keys()
    restored = DiT(**config)
    restored.load_state_dict(state, strict=True)
    assert restored.state_dict().keys() == state.keys()
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


def test_flash_mask_tokens_replace_missing_values_before_native_qkv():
    dit = DiT(
        height=8,
        width=12,
        patch_size=2,
        in_chans=3,
        base_out_chans=1,
        embed_dim=16,
        depth=1,
        num_heads=2,
        attn_kernel=3,
        use_nan_mask_tokens=True,
    )
    block = dit.blocks[0]
    with torch.no_grad():
        block.attention.mask_token.copy_(torch.linspace(-1, 1, 16).view(1, 1, 16))
    tokens = torch.randn(2, 24, 16)
    invalid = torch.zeros(2, 24, dtype=torch.bool)
    invalid[0, 3] = invalid[1, 7] = True
    tokens[invalid] = torch.nan
    condition = torch.randn(2, 16)
    captured = []

    def capture_qkv_input(module, inputs):
        captured.append(inputs[0].detach().clone())
        raise RuntimeError("captured native QKV input")

    hook = block.attention.qkv.register_forward_pre_hook(capture_qkv_input)
    cos, sin = build_axial_rope_cos_sin_2d(4, 6, 8)
    try:
        with pytest.raises(RuntimeError, match="captured native QKV input"):
            block(
                tokens,
                condition,
                attn_kwargs={
                    "latent_hw": (4, 6),
                    "rope_cos": cos,
                    "rope_sin": sin,
                    "invalid_token_mask": invalid,
                },
            )
    finally:
        hook.remove()
    assert len(captured) == 1 and torch.isfinite(captured[0]).all()
    torch.testing.assert_close(
        captured[0][invalid], block.attention.mask_token.expand(1, 2, 16)[0]
    )
    shift, scale, *_ = block.adaptive_modulation(condition).chunk(6, dim=1)
    expected = block.modulation(block.pre_attention_norm(tokens), scale, shift)
    torch.testing.assert_close(captured[0][~invalid], expected[~invalid])


def test_flash_patch_projection_rejects_implicit_padding():
    with pytest.raises(ValueError, match="divisible by patch_size"):
        DiT(
            height=9,
            width=12,
            patch_size=2,
            in_chans=3,
            base_out_chans=1,
            embed_dim=16,
            depth=1,
            num_heads=2,
            attn_kernel=3,
        )


@pytest.mark.parametrize("kind", ["goes", "mrms"])
def test_flash_aligned_budget_and_no_noise_reset(kind):
    experts = {name: _FakeExpert(grid) for name, grid in regional_sigma_grids().items()}
    nfe = sum(FLASH_CALLS[kind].values())
    plan = build_flash_chain_plan(
        experts, total_nfe=nfe, region_calls=FLASH_CALLS[kind], alignment=16
    )
    output = flash_sampler_chain(
        experts, torch.ones(1, 1, 2, 2), total_nfe=nfe, plan=plan
    )
    torch.testing.assert_close(output, torch.full_like(output, 800 / 801))
    assert {name: expert.calls for name, expert in experts.items()} == FLASH_CALLS[kind]
    if kind == "goes":
        assert experts["low"].prepared == [
            (0, 32),
            (32, 64),
            (64, 80),
        ]
    assert all(
        start % 16 == 0 for expert in experts.values() for start, _ in expert.prepared
    )
    with pytest.raises(ValueError):
        inference_block_boundaries(80, 6, alignment=16)


@pytest.mark.parametrize("kind", ["goes", "mrms"])
def test_flash_chain_preserves_small_updates_across_expert_handoffs(kind):
    calls = []
    increment = 2.0**-30

    class PrecisionExpert(_FakeExpert):
        def __init__(self, name, grid):
            super().__init__(grid)
            self.name = name

        def forward(self, state, sigma, condition, **kwargs):
            calls.append((self.name, state.clone(), sigma.dtype))
            return state + increment

    experts = {
        name: PrecisionExpert(name, grid)
        for name, grid in regional_sigma_grids().items()
    }
    nfe = sum(FLASH_CALLS[kind].values())
    plan = build_flash_chain_plan(
        experts, total_nfe=nfe, region_calls=FLASH_CALLS[kind], alignment=16
    )
    latents = torch.ones(1, 1, 2, 2, dtype=torch.float64)
    output = flash_sampler_chain(experts, latents, total_nfe=nfe, plan=plan)
    assert [name for name, _, _ in calls] == [
        name for name, count in FLASH_CALLS[kind].items() for _ in range(count)
    ]
    for i, (_, state, sigma_dtype) in enumerate(calls):
        assert state.dtype == sigma_dtype == torch.float64
        torch.testing.assert_close(
            state,
            torch.full_like(latents, 800 / 801 + i * increment),
            rtol=0,
            atol=1e-15,
        )
    torch.testing.assert_close(
        output,
        torch.full_like(latents, 800 / 801 + nfe * increment),
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("layout", [None, "legacy", "physicsnemo-dit-v2"])
def test_flash_requires_native_checkpoint_layout(layout):
    metadata = dict(
        sigma_grid=[3.0, 1.0, 0.0],
        block_boundaries=[0, 2],
        config={"student": {"sigma_data": 0.5}},
    )
    if layout is not None:
        metadata["checkpoint_layout"] = layout
    with pytest.raises(ValueError, match="checkpoint_layout"):
        FlashModel({}, metadata)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_flash_fused_update_matches_individual_heads(monkeypatch, dtype):
    torch.manual_seed(8)
    from earth2studio.models.nn import stormscope_flash

    trunk = _TinyDiT()
    metadata = dict(
        checkpoint_layout="physicsnemo-dit-v1",
        sigma_grid=[3.0, 1.0, 0.01, 0.0],
        block_boundaries=[0, 3],
        config={"student": {"sigma_data": 0.5}},
    )
    with monkeypatch.context() as patch:
        patch.setattr(stormscope_flash, "FlashDiT", lambda **kwargs: trunk)
        expert = FlashModel({}, metadata)
    expert.head_weight.data.normal_()
    expert.head_bias.data.normal_()
    state = torch.randn(2, 1, 3, 4, dtype=dtype)
    for start, end in [(0, 3), (2, 3)]:
        sigma, weight, bias, delta = expert.prepare_inference_block(start, end)
        actual = expert(
            state,
            sigma,
            None,
            inference_weight=weight,
            inference_bias=bias,
            inference_delta=delta,
        )
        assert actual.dtype == dtype
        # Independent EDM-derived velocity expression evaluated in float64.
        sig = sigma.double()
        t = 1 / (1 + sig)
        r = sig * t
        denom = r * r + 0.25 * t * t
        features = state.double() / denom.sqrt()
        residual = torch.zeros_like(features)
        for j in range(start, end):
            dt = (expert.time_grid[j + 1] - expert.time_grid[j]).double()
            residual += dt * (
                expert.head_weight[j, 0, 0].double() * features
                + expert.head_bias[j, 0].double()
            )
        expected = (
            state.double()
            + delta.double() * (0.25 * t - r) / denom * state.double()
            + 0.5 / denom.sqrt() * residual
        )
        torch.testing.assert_close(actual.double(), expected, rtol=3e-5, atol=3e-6)


def test_flash_fp64_update_preserves_small_increments_with_network_amp(monkeypatch):
    from earth2studio.models.nn import stormscope_flash

    trunk = _TinyDiT()
    metadata = dict(
        checkpoint_layout="physicsnemo-dit-v1",
        sigma_grid=[0.25, 0.0],
        block_boundaries=[0, 1],
        config={"student": {"sigma_data": 0.5}},
    )
    with monkeypatch.context() as patch:
        patch.setattr(stormscope_flash, "FlashDiT", lambda **kwargs: trunk)
        expert = FlashModel({}, metadata)
    network_calls = []

    def forward_features(x, time_step_cond, *, training=False):
        features = torch.nn.functional.linear(
            x[:, :1].flatten(2).transpose(1, 2), torch.ones(1, 1)
        )
        network_calls.append((x.dtype, time_step_cond.dtype, features.dtype))
        return features

    monkeypatch.setattr(trunk, "forward_features", forward_features)
    state = torch.full((1, 1, 2, 2), 1 + 2.0**-35, dtype=torch.float64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = expert(
            state,
            torch.tensor(0.25, dtype=torch.float64),
            None,
            inference_weight=torch.zeros(1, 1),
            inference_bias=torch.tensor([2.0**-30]),
            inference_delta=torch.tensor(1.0),
        )

    assert network_calls == [(torch.float32, torch.float32, torch.bfloat16)]
    assert actual.dtype == torch.float64
    assert bool((actual > state).all())
    # At sigma=0.25 the skip velocity vanishes; only the small residual remains.
    expected = state + (0.5 / np.sqrt(0.2)) * 2.0**-30
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-15)


@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
def test_flash_move_invalidates_plan(cls):
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model._is_flash = True
    model._flash_plan = object()
    model.to("cpu")
    assert model._flash_plan is None


@pytest.mark.parametrize("mutation", ["dtype_roundtrip", "weight_update"])
def test_shared_expert_changes_invalidate_other_domain_fused_heads(mutation):
    from copy import copy

    from earth2studio.models.nn.stormscope_flash import sample_flash

    class HeadExpert(torch.nn.Module):
        prepare_inference_block = FlashModel.prepare_inference_block

        def __init__(self, grid):
            super().__init__()
            self.num_intervals = len(grid) - 1
            self.register_buffer("sigma_grid", grid)
            self.register_buffer("time_grid", (1 + grid).reciprocal())
            self.head_weight = torch.nn.Parameter(
                torch.linspace(0.001, 0.019, self.num_intervals).view(-1, 1, 1)
            )
            self.head_bias = torch.nn.Parameter(
                torch.linspace(0.0001, 0.003, self.num_intervals).view(-1, 1)
            )

        def forward(self, state, sigma, condition, **kwargs):
            return (
                state
                + kwargs["inference_weight"].float().sum()
                + kwargs["inference_bias"].float().sum()
            )

    parent = StormScopeGOES.__new__(StormScopeGOES)
    torch.nn.Module.__init__(parent)
    parent._is_flash = True
    parent.amp = False
    parent._flash_plan = None
    parent.stage_models = torch.nn.ModuleList(
        HeadExpert(grid) for grid in regional_sigma_grids().values()
    )
    child = copy(parent)
    child._modules = parent._modules.copy()
    child.stage_models = torch.nn.ModuleList(parent.stage_models)
    latents = torch.zeros(1, 1, 2, 2)
    original = sample_flash(parent, latents)
    cached = parent._flash_plan
    torch.testing.assert_close(sample_flash(parent, latents), original)
    assert parent._flash_plan is cached

    if mutation == "dtype_roundtrip":
        child.half().float()
    else:
        with torch.no_grad():
            child.stage_models[0].head_weight.add_(0.125)
    assert parent._flash_plan is cached
    actual = sample_flash(parent, latents)
    assert parent._flash_plan is not cached
    assert not torch.equal(actual, original)
    experts = dict(zip(("high", "middle", "low"), parent.stage_models))
    expected = flash_sampler_chain(
        experts,
        latents,
        total_nfe=5,
        plan=build_flash_chain_plan(
            experts, total_nfe=5, region_calls=FLASH_CALLS["goes"]
        ),
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("is_flash", [False, True])
def test_variant_configuration_uses_shared_compile_policy(monkeypatch, is_flash):
    from unittest.mock import Mock

    model = StormScopeGOES.__new__(StormScopeGOES)
    torch.nn.Module.__init__(model)
    model._is_flash = not is_flash
    model.compile_experts = Mock()
    model.sampler_args = {"num_steps": 100, "S_churn": 10}
    configure_region = Mock()
    monkeypatch.setattr(
        "earth2studio.models.nn.stormscope_flash.configure_region", configure_region
    )
    model._configure_variant({"flash": True} if is_flash else {}, compile=True)

    assert model._is_flash is is_flash
    model.compile_experts.assert_called_once_with()
    if is_flash:
        configure_region.assert_called_once_with(model, None, 0)
        assert model.sampler_args == {}
    else:
        configure_region.assert_not_called()
        assert model.sampler_args == {"num_steps": 100, "S_churn": 10}


@pytest.mark.parametrize("is_flash, dynamic", [(False, None), (True, False)])
def test_compile_policy_preserves_baseline_and_specializes_flash(
    monkeypatch, is_flash, dynamic
):
    from unittest.mock import Mock, call

    model = StormScopeGOES.__new__(StormScopeGOES)
    torch.nn.Module.__init__(model)
    model._is_flash = is_flash
    model._experts_compiled = False
    experts = [torch.nn.Identity() for _ in range(3)]
    model.stage_models = torch.nn.ModuleList(experts)
    compile = Mock(side_effect=lambda expert, **kwargs: torch.nn.Sequential(expert))
    monkeypatch.setattr(torch, "compile", compile)

    model.compile_experts()
    assert compile.call_args_list == [
        call(expert, mode="reduce-overhead", dynamic=dynamic) for expert in experts
    ]
    assert model._experts_compiled
    assert all(
        wrapped[0] is expert for wrapped, expert in zip(model.stage_models, experts)
    )
    cached_plan, cached_sources = object(), (object(),)
    if is_flash:
        model._flash_plan = cached_plan
        model._flash_plan_sources = cached_sources
    model.compile_experts()
    assert compile.call_count == 3
    if is_flash:
        assert model._flash_plan is cached_plan
        assert model._flash_plan_sources is cached_sources


class _FakeExpert(torch.nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.register_buffer("sigma_grid", grid)
        self.calls = 0
        self.prepared = []

    def prepare_inference_block(self, start, end):
        self.prepared.append((start, end))
        return (
            self.sigma_grid[start],
            torch.ones(1),
            torch.zeros(1),
            torch.tensor(float(end - start)),
        )

    def forward(self, state, sigma, condition, **kwargs):
        self.calls += 1
        return state


class _TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.detokenizer = torch.nn.Module()
        self.detokenizer.proj_layer = torch.nn.Module()
        self.detokenizer.proj_layer.output_projection = torch.nn.Linear(1, 1)
        self.conditioning_embedder = torch.nn.Module()
        self.conditioning_embedder.t_embedder = torch.nn.Module()
        self.conditioning_embedder.t_embedder.register_buffer("freqs", torch.ones(1))
        self._out_chans = 1

    def forward_features(self, x, time_step_cond, *, training=False):
        return x[:, :1].flatten(2).transpose(1, 2)

    def unpatchify(self, x, height, width):
        return x.transpose(1, 2).reshape(x.shape[0], 1, height, width)


def _grid():
    return np.meshgrid(
        np.linspace(30, 50, 400), np.linspace(-120, -80, 600), indexing="ij"
    )


@pytest.mark.parametrize(
    "bounds",
    [
        None,
        {"lat": (30, 34), "lon": (-120, -116)},
        {"lat": (46, 50), "lon": (-84, -80)},
        {"lat": (30, 34), "lon": (-84, -80)},
        {"lat": (46, 50), "lon": (-120, -116)},
        {"lat": (35, 45), "lon": (-110, -90)},
        {"lat": (35, 35.2), "lon": (250, 250.3)},
    ],
)
def test_regional_bounds_use_available_real_context(bounds):
    lat, lon = _grid()
    info = resolve_region(lat, lon, bounds)
    h, w = info.input_shape
    y0, y1, x0, x1 = info.input_bounds
    assert min(h, w) >= 200
    assert all(v % 4 == 0 for v in (y0, y1, x0, x1))
    assert 0 <= y0 < y1 <= 400 and 0 <= x0 < x1 <= 600
    a, b, c, d = info.output_bounds
    assert y0 <= a < b <= y1 and x0 <= c < d <= x1
    if bounds and bounds["lat"][0] == 30:
        assert info.padding[0] == 0
    if bounds is None:
        assert info.input_shape == (400, 600)
        assert info.padding == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "bounds",
    [
        {"lat": (20, 35), "lon": (-110, -100)},
        {"lat": (40, 35), "lon": (-110, -100)},
        {"lat": (35, 40), "lon": (float("nan"), -100)},
        {"lat": (35, 40), "lon": (-190, 170)},
        {"lat": (35, 40)},
    ],
)
def test_invalid_regions_fail(bounds):
    with pytest.raises(ValueError):
        resolve_region(*_grid(), bounds)


def test_region_output_cropping_retains_context_and_coords():
    lat, lon = _grid()
    info = resolve_region(lat, lon, {"lat": (36, 42), "lon": (-110, -92)})
    tensor = torch.ones(1, 1, 1, 3, *info.input_shape)
    y0, y1, x0, x1 = info.input_bounds
    coords = {
        "y": np.arange(y0, y1),
        "x": np.arange(x0, x1),
        "lat": lat[y0:y1, x0:x1],
        "lon": lon[y0:y1, x0:x1],
    }
    output, cropped_coords = info.crop_output(tensor, coords)
    assert output.shape[-2:] == info.requested_mask.shape
    assert cropped_coords["lat"].shape == info.requested_mask.shape
    output.zero_()
    assert tensor.min() == 1
    assert len(coords["y"]) == info.input_shape[0]
    assert coords["y"] is not cropped_coords["y"]


def test_region_signed_and_unsigned_longitudes_match():
    lat, lon = _grid()
    a = resolve_region(lat, lon, {"lat": (35, 45), "lon": (-110, -90)})
    b = resolve_region(lat, lon % 360, {"lat": (35, 45), "lon": (250, 270)})
    assert a.input_bounds == b.input_bounds
    np.testing.assert_array_equal(a.requested_mask, b.requested_mask)


class _DomainTrunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.image_size = None
        self.mask = None

    def set_image_size(self, height, width):
        self.image_size = (height, width)
        self.mask = None

    def set_nan_pixel_mask(self, mask):
        self.mask = mask


def _domain_wrapper(cls, is_flash=True):
    lat, lon = _grid()
    expert = torch.nn.Module()
    expert.model = _DomainTrunk()
    model = cls(
        model_spec=[dict(model=expert, sigma_min=0.0, sigma_max=80.0)],
        means=torch.zeros(1, 1, 1, 1),
        stds=torch.ones(1, 1, 1, 1),
        variables=np.array(["abi13c" if cls is StormScopeGOES else "refc"]),
        conditioning_variables=np.array([], dtype=str),
        latitudes=torch.tensor(lat, dtype=torch.float32),
        longitudes=torch.tensor(lon, dtype=torch.float32),
        topo=torch.arange(400 * 600).view(400, 600).float(),
        nexrad_proximity=torch.ones(400, 600),
        input_times=np.arange(-50, 1, 10).astype("timedelta64[m]"),
        output_times=np.array([np.timedelta64(10, "m")]),
        amp=False,
    )
    if is_flash:
        model._configure_variant({"flash": True}, compile=False)
    return model


@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
def test_baseline_sampling_is_independent_of_flash_instances(cls, monkeypatch):
    from unittest.mock import Mock

    before = _domain_wrapper(cls, is_flash=False)
    flash = _domain_wrapper(cls)
    after = _domain_wrapper(cls, is_flash=False)
    latents = torch.ones(1, 1, 2, 3)
    sample_flash = Mock(
        return_value=torch.zeros_like(latents, dtype=flash._SAMPLER_DTYPE)
    )
    monkeypatch.setattr(
        "earth2studio.models.nn.stormscope_flash.sample_flash", sample_flash
    )

    for baseline in (before, after):
        expert = _FakeExpert(torch.tensor([2.0, 0.1]))
        baseline.stage_models = torch.nn.ModuleList([expert])
        actual = baseline._edm_sampler(
            latents, num_steps=2, sigma_max=2.0, sigma_min=0.1
        )
        assert not baseline._is_flash
        assert expert.calls == 3
        torch.testing.assert_close(actual, (2 * latents).double())
    sample_flash.assert_not_called()

    torch.testing.assert_close(
        flash._edm_sampler(latents),
        torch.zeros_like(latents, dtype=flash._SAMPLER_DTYPE),
    )
    sample_flash.assert_called_once()
    called_model, called_latents, called_condition = sample_flash.call_args.args
    assert called_model is flash
    assert called_condition is None
    torch.testing.assert_close(called_latents, latents.to(flash._SAMPLER_DTYPE))


@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
def test_set_domain_leaves_parent_and_other_views_independent(cls):
    full = _domain_wrapper(cls)
    full.input_interp = object()
    full.conditioning_interp = object()
    if cls is StormScopeMRMS:
        full.glm_interp = object()
    full._flash_plan = object()
    original_buffers = {name: value.clone() for name, value in full._buffers.items()}
    original_y, original_x = full.y.copy(), full.x.copy()
    bounds = {"lat": (36, 38), "lon": (-110, -108)}
    first = full.set_domain(*bounds["lat"], *bounds["lon"])
    second = full.set_domain(35, 45, -110, -90)

    assert first is not full and second is not full and first is not second
    assert first._buffers is not full._buffers and first._buffers is not second._buffers
    assert first.stage_models[0] is full.stage_models[0] is second.stage_models[0]
    assert (
        next(first.parameters()) is next(full.parameters()) is next(second.parameters())
    )
    assert first.input_interp is first.conditioning_interp is first._flash_plan is None
    assert (
        second.input_interp is second.conditioning_interp is second._flash_plan is None
    )
    if cls is StormScopeMRMS:
        assert first.glm_interp is second.glm_interp is None
        assert full.glm_interp is not None
    assert full.input_interp is not None and full.conditioning_interp is not None
    assert full._flash_plan is not None
    assert full.region_info.input_shape == (400, 600)
    assert first.region_info.requested_bounds == tuple(bounds["lat"] + bounds["lon"])
    assert min(first.latitudes.shape) >= 200
    assert all(dimension % 4 == 0 for dimension in first.latitudes.shape)
    assert not np.shares_memory(first.y, full.y)
    assert not np.shares_memory(first.x, full.x)
    assert not np.shares_memory(first._lat_cpu_copy, full._lat_cpu_copy)
    for name in (
        "latitudes",
        "longitudes",
        "valid_mask",
        "conditioning_valid_mask",
        "topo",
        "nexrad_proximity",
    ):
        assert first._buffers[name].data_ptr() != full._buffers[name].data_ptr()
        assert first._buffers[name].data_ptr() != second._buffers[name].data_ptr()
    first.valid_mask.fill_(False)
    first.y[0] = -999
    first._flash_plan = object()
    assert second.valid_mask.all() and second.y[0] != -999
    assert second._flash_plan is None
    np.testing.assert_array_equal(full.y, original_y)
    np.testing.assert_array_equal(full.x, original_x)
    for name, value in original_buffers.items():
        torch.testing.assert_close(full._buffers[name], value)


@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
def test_set_domain_rejects_baseline_and_expansion_beyond_current_view(cls):
    full = _domain_wrapper(cls)
    child = full.set_domain(36, 38, -110, -108)
    with pytest.raises(ValueError, match="outside|footprint|domain"):
        child.set_domain(46, 48, -86, -84)
    full._is_flash = False
    with pytest.raises(ValueError, match="Flash|flash"):
        full.set_domain(36, 38, -110, -108)


@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
def test_set_domain_sequential_calls_rebind_experts_and_keep_forecast_context(
    cls, monkeypatch
):
    full = _domain_wrapper(cls)
    first = full.set_domain(36, 38, -110, -108)
    second = full.set_domain(35, 45, -110, -90)
    visited = []

    def sample(model, latents, condition):
        assert model.stage_models[0].model.image_size == latents.shape[-2:]
        assert latents.shape[-2:] == model.latitudes.shape == condition.shape[-2:]
        if cls is StormScopeGOES:
            mask = model.stage_models[0].model.mask
            assert mask is not None and mask.shape[-2:] == latents.shape[-2:]
        visited.append(tuple(latents.shape[-2:]))
        return torch.ones_like(latents)

    monkeypatch.setattr("earth2studio.models.nn.stormscope_flash.sample_flash", sample)
    for model in (full, first, second, first, full):
        coords = model.input_coords()
        coords["batch"] = np.arange(1)
        coords["time"] = np.array([np.datetime64("2025-04-17T23:30")])
        state = torch.zeros(1, 1, 6, 1, *model.latitudes.shape)
        forecast, forecast_coords = model(state, coords)
        assert forecast.shape == (1, 1, 1, 1, *model.latitudes.shape)
        assert np.array_equal(forecast_coords["y"], model.y)
        assert np.array_equal(forecast_coords["x"], model.x)
        assert forecast_coords["lead_time"][-1] == np.timedelta64(10, "m")
        torch.testing.assert_close(forecast, torch.ones_like(forecast))
        history, history_coords = model.next_input(
            forecast, forecast_coords, state, coords
        )
        assert history.shape == state.shape
        np.testing.assert_array_equal(history_coords["y"], coords["y"])
        np.testing.assert_array_equal(history_coords["x"], coords["x"])
        cropped, cropped_coords = model.crop_output(forecast, forecast_coords)
        assert cropped.shape[-2:] == model.region_info.requested_mask.shape
        assert cropped.shape[-2:] == (
            len(cropped_coords["y"]),
            len(cropped_coords["x"]),
        )
    assert visited == [
        (400, 600),
        first.region_info.input_shape,
        second.region_info.input_shape,
        first.region_info.input_shape,
        (400, 600),
    ]


@pytest.mark.parametrize("is_flash", [False, True])
def test_glm_interpolation_and_normalization_order_is_baseline(is_flash):
    model = StormScopeMRMS.__new__(StormScopeMRMS)
    torch.nn.Module.__init__(model)
    model._is_flash = is_flash
    model.means = torch.tensor([2.0, 4.0, 6.0]).view(1, 3, 1, 1)
    model.stds = torch.tensor([2.0, 4.0, 8.0]).view(1, 3, 1, 1)
    model.glm_mask = torch.tensor([False, False, True])
    model.glm_interp = lambda x: x.mean(-1, keepdim=True)
    counts = torch.tensor([[[[0.0, 8.0]]]])
    regridded = model.interpolate_glm(counts)
    state = torch.cat(
        (torch.zeros_like(regridded), torch.zeros_like(regridded), regridded), dim=1
    )
    actual = model._normalize_state(state)
    torch.testing.assert_close(
        actual[:, :2], (state[:, :2] - model.means[:, :2]) / model.stds[:, :2]
    )
    torch.testing.assert_close(actual[:, 2:], torch.log1p(regridded))
    assert not torch.allclose(actual[:, 2:], torch.log1p(counts).mean(-1, keepdim=True))
    torch.testing.assert_close(model._denormalize_state(actual), state)


@pytest.mark.parametrize(
    "cls,is_flash,fill_nan",
    [
        (StormScopeGOES, False, False),
        (StormScopeGOES, True, False),
        (StormScopeMRMS, False, False),
        (StormScopeMRMS, True, True),
    ],
)
def test_conditioning_nan_fill_is_only_for_flash_mrms(cls, is_flash, fill_nan):
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model._is_flash = is_flash
    model.register_buffer("conditioning_means", torch.full((1, 8, 1, 1), 10.0))
    model.register_buffer("conditioning_stds", torch.full((1, 8, 1, 1), 2.0))
    model.register_buffer("conditioning_glm_mask", torch.zeros(8, dtype=torch.bool))
    conditioning = torch.full((1, 1, 6, 8, 2, 3), 12.0)
    conditioning[0, 0, 2, 3, 0, 0] = torch.nan
    conditioning[0, 0, 2, 3, 0, 1] = torch.inf
    original = conditioning.clone()

    actual = model.normalize_conditioning(conditioning)

    if fill_nan:
        assert actual[0, 0, 2, 3, 0, 0] == 0
    else:
        assert torch.isnan(actual[0, 0, 2, 3, 0, 0])
    assert torch.isposinf(actual[0, 0, 2, 3, 0, 1])
    assert torch.all(actual[torch.isfinite(conditioning)] == 1)
    torch.testing.assert_close(conditioning, original, equal_nan=True)
    assert model.normalize_conditioning(None) is None


@pytest.mark.package
@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
def test_flash_package(cls):
    if not torch.cuda.is_available():
        pytest.skip("NATTEN real-weight test requires CUDA")
    model = (
        cls.load_model(
            cls.load_default_package(),
            model_name="3km_10min_flash",
        )
        .set_domain(38.0, 39.0, -98.0, -97.0)
        .cuda()
    )
    coords = model.input_coords()
    coords["batch"] = np.arange(1)
    coords["time"] = np.array([np.datetime64("2025-04-17T23:30")])
    h, w = model.region_info.input_shape
    state = model.means.view(1, 1, 1, -1, 1, 1).expand(1, 1, 6, -1, h, w).clone()
    if cls is StormScopeMRMS:
        cond = (
            model.conditioning_means.view(1, 1, 1, -1, 1, 1)
            .expand(1, 1, 6, -1, h, w)
            .clone()
        )
        cc = coords.copy()
        cc["variable"] = model.conditioning_variables
    else:
        cond = cc = None
    with torch.inference_mode():
        result = model._forward(state, coords, cond, cc)
    assert result.shape == (1, 1, 1, len(model.variables), h, w)
    assert torch.isfinite(result[..., model.valid_mask]).all()


def test_flash_missing_goes_masks_are_preserved_without_persistence():
    class MaskModel:
        def set_nan_pixel_mask(self, mask):
            self.mask = mask

    model = StormScopeGOES.__new__(StormScopeGOES)
    torch.nn.Module.__init__(model)
    model._is_flash = True
    model.means = torch.full((1, 8, 1, 1), 10.0)
    model.stds = torch.full((1, 8, 1, 1), 2.0)
    model.glm_mask = torch.zeros(8, dtype=torch.bool)
    model.valid_mask = torch.ones(8, 12, dtype=torch.bool)
    expert = torch.nn.Module()
    expert.model = MaskModel()
    model.stage_models = torch.nn.ModuleList([expert])
    x = torch.full((2, 1, 6, 8, 8, 12), 12.0)
    x[1, 0, 2, 3, 4, 5] = torch.nan
    normalized = model._normalize_state(x)
    assert torch.isnan(normalized[1, 0, 2, 3, 4, 5])
    actual = prepare_flash_goes_state(model, x, normalized)
    assert actual[1, 0, 2, 3, 4, 5] == 0
    assert expert.model.mask[1, 4, 5] and not expert.model.mask[0, 4, 5]
    assert torch.isnan(x[1, 0, 2, 3, 4, 5])
    assert actual[1, 0, 1, 3, 4, 5] == 1
    x[0, 0, 0, 0, 0, 0] = torch.inf
    actual = prepare_flash_goes_state(model, x, model._normalize_state(x))
    assert torch.isinf(actual[0, 0, 0, 0, 0, 0])


def test_flash_rejects_mismatched_handoff():
    experts = {name: _FakeExpert(grid) for name, grid in regional_sigma_grids().items()}
    experts["middle"].sigma_grid[0] *= 0.9
    with pytest.raises(ValueError, match="handoff"):
        build_flash_chain_plan(experts, total_nfe=5, region_calls=FLASH_CALLS["goes"])


def test_curvilinear_request_outside_perimeter_is_rejected():
    lat, lon = _grid()
    lon = lon + (lat - 30) * 0.5
    with pytest.raises(ValueError, match="footprint"):
        resolve_region(lat, lon, {"lat": (48, 49), "lon": (-119, -118)})


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NATTEN execution requires CUDA"
)
@pytest.mark.parametrize("hashed_cache_name", [False, True])
def test_mdlus_round_trip_preserves_heads_and_updates(tmp_path, hashed_cache_name):
    torch.manual_seed(8)
    config = dict(
        height=8,
        width=12,
        patch_size=2,
        in_chans=3,
        base_out_chans=1,
        embed_dim=16,
        depth=2,
        num_heads=2,
        pos_embedding_type="rotary",
        attn_kernel=3,
    )
    metadata = dict(
        checkpoint_layout="physicsnemo-dit-v1",
        sigma_grid=[3.0, 1.0, 0.0],
        block_boundaries=[0, 2],
        config={"student": {"sigma_data": 0.5}},
    )
    reference = _initialized_flash_model(config, metadata)
    with torch.no_grad():
        reference.head_weight.normal_(std=0.02)
        reference.head_bias.normal_(std=0.02)
    model = FlashModel(config, metadata)
    model.load_state_dict(reference.state_dict(), strict=True)
    path = tmp_path / "tiny.mdlus"
    model.save(str(path))
    if hashed_cache_name:
        path = path.rename(tmp_path / hashlib.sha256(b"tiny.mdlus").hexdigest())
    restored = Module.from_checkpoint(str(path), strict=True)
    assert isinstance(restored, FlashModel)
    assert isinstance(restored.model, PhysicsNeMoDiT)
    assert restored.model_config == config
    assert restored.flash_metadata == metadata
    assert not any(p.is_meta or p.requires_grad for p in restored.parameters())
    assert not any(b.is_meta for b in restored.buffers())
    assert restored.state_dict().keys() == reference.state_dict().keys()
    for key, tensor in reference.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key]), key
    reference.cuda()
    restored.cuda()
    assigned = FlashModel(config, metadata)
    assigned.load_state_dict(reference.state_dict(), strict=True)
    assert assigned.model.rope.rope_cos.device == reference.head_weight.device
    assert assigned.model.rope.rope_sin.device == reference.head_weight.device
    state = torch.randn(2, 1, 8, 12, device="cuda")
    condition = torch.randn(2, 2, 8, 12, device="cuda")
    sigma, weight, bias, delta = reference.prepare_inference_block(0, 2)
    kwargs = dict(inference_weight=weight, inference_bias=bias, inference_delta=delta)
    with torch.no_grad():
        expected = reference(state, sigma, condition, **kwargs)
        actual = restored(state, sigma, condition, **kwargs)
        assigned_actual = assigned(state, sigma, condition, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(assigned_actual, expected, rtol=0, atol=0)
    with zipfile.ZipFile(path) as archive:
        assert "model.pt" in archive.namelist()
        assert json.loads(archive.read("args.json"))["__name__"] == "FlashModel"
        assert not any("optimizer" in name for name in archive.namelist())
    incomplete = dict(reference.state_dict())
    incomplete.pop("head_weight")
    with pytest.raises(RuntimeError, match="head_weight"):
        FlashModel(config, metadata).load_state_dict(incomplete, strict=True)


@pytest.mark.parametrize("hashed_cache_name", [False, True])
def test_mdlus_file_integrity_guard(tmp_path, monkeypatch, hashed_cache_name):
    from earth2studio.models.nn.stormscope_flash import (
        _validate_checkpoint_file,
    )

    payload = b"checkpoint integrity test"
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / (digest if hashed_cache_name else "expert_0.mdlus")
    path.write_bytes(payload)
    spec = dict(
        path="checkpoints/goes/3km_10min_flash/expert_0.mdlus",
        deployment_format="flash-mdlus-v1",
        deployment_sha256=digest,
        deployment_size_bytes=len(payload),
    )
    monkeypatch.setenv("EARTH2STUDIO_VERIFY_CHECKPOINT_HASH", "1")
    _validate_checkpoint_file(path, spec)
    path.write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        _validate_checkpoint_file(path, spec)
    with pytest.raises(ValueError, match="size mismatch"):
        _validate_checkpoint_file(path, {**spec, "deployment_size_bytes": 1})
    with pytest.raises(ValueError, match="requires a .mdlus"):
        _validate_checkpoint_file(path, {**spec, "deployment_format": "flash-pt-v1"})
    with pytest.raises(ValueError, match="requires a .mdlus"):
        _validate_checkpoint_file(path, {**spec, "path": "expert_0.pt"})


def test_flash_region_uses_fifteen_available_edge_pixels():
    lat, lon = _grid()
    # Output starts 15 pixels inside the grid on both axes. Context reaches the
    # native edge without fabricating the remaining ten requested pixels.
    region = {"lat": (lat[15, 0], lat[249, 0]), "lon": (lon[0, 15], lon[0, 249])}
    info = resolve_region(lat, lon, region, padding=25)
    assert info.input_bounds == (0, 276, 0, 276)
    assert info.padding == (15, 26, 15, 26)
    assert all(n >= 200 and n % 4 == 0 for n in info.input_shape)


@pytest.mark.parametrize(
    "kind,flash_class", [("goes", StormScopeGOES), ("mrms", StormScopeMRMS)]
)
def test_flash_hub_registry_keeps_teacher_selection_separate(
    tmp_path, kind, flash_class
):
    teacher = {"description": "teacher", "checkpoints": [{"path": "teacher.mdlus"}]}
    flash = {
        "description": "flash",
        "flash": True,
        "checkpoints": [{"path": f"checkpoints/{kind}/3km_10min_flash/expert_2.mdlus"}],
    }
    registry = {
        kind: {
            "models": {"3km_10min": teacher, "3km_10min_flash": flash},
            "aliases": {},
        }
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry))
    package = Package(str(tmp_path))
    name, entry = flash_class._resolve_model_entry(package, "3km_10min")
    assert name == "3km_10min" and entry == teacher
    assert flash_class._resolve_model_entry(package, "3km_10min_flash")[1] == flash
    assert set(flash_class.list_available_models(package)) == {
        "3km_10min",
        "3km_10min_flash",
    }
    assert json.loads(path.read_text()) == registry


@pytest.mark.parametrize("flash_class", [StormScopeGOES, StormScopeMRMS])
def test_flash_local_package_remains_compatible(tmp_path, flash_class):
    kind = flash_class._REGISTRY_KEY
    entry = {"flash": True, "description": "local"}
    (tmp_path / "registry.json").write_text(
        json.dumps({kind: {"models": {"3km_10min": entry}, "aliases": {}}})
    )
    assert flash_class._resolve_model_entry(Package(str(tmp_path)), "3km_10min") == (
        "3km_10min",
        entry,
    )


@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
@pytest.mark.parametrize("compile_enabled", [False, True])
def test_load_model_accepts_positional_runtime_options(
    cls, compile_enabled, monkeypatch
):
    from unittest.mock import Mock

    reference = _domain_wrapper(cls, is_flash=False)
    entry = {
        "variables": reference.variables.tolist(),
        "conditioning_vars": [],
        "flash": True,
    }
    monkeypatch.setattr(cls, "_load_registry", Mock(return_value={}))
    monkeypatch.setattr(
        cls, "_resolve_model_entry", Mock(return_value=("3km_10min_flash", entry))
    )
    monkeypatch.setattr(
        cls, "_load_checkpoints", Mock(return_value=reference.model_spec)
    )
    monkeypatch.setattr(
        cls,
        "_build_grid_and_times",
        Mock(
            return_value=(
                reference.latitudes,
                reference.longitudes,
                reference.y,
                reference.x,
                reference.input_times,
                reference.output_times,
                1,
            )
        ),
    )
    monkeypatch.setattr(
        cls,
        "_build_normalization",
        Mock(return_value=(reference.means, reference.stds, reference.glm_mask)),
    )
    compile = Mock(side_effect=lambda expert, **kwargs: expert)
    monkeypatch.setattr(torch, "compile", compile)
    package = Mock()
    package.resolve.side_effect = FileNotFoundError
    conditioning_source, glm_source = object(), object()
    sources = [conditioning_source]
    if cls is StormScopeMRMS:
        sources.append(glm_source)

    model = cls.load_model(package, "3km_10min_flash", *sources, False, compile_enabled)

    assert model.amp is False
    assert model._experts_compiled is compile_enabled
    assert model._is_flash
    assert model.conditioning_data_source is conditioning_source
    if cls is StormScopeMRMS:
        assert model.glm_data_source is glm_source
    if compile_enabled:
        compile.assert_called_once_with(
            reference.stage_models[0], mode="reduce-overhead", dynamic=False
        )
    else:
        compile.assert_not_called()


@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
def test_shared_default_package_has_no_variant_arguments(cls):
    import inspect

    assert not inspect.signature(cls.load_default_package).parameters
    package = cls.load_default_package()
    assert package.root == StormScopeGOES.load_default_package().root
    assert package.cache_options["cache_storage"] == Package.default_cache("stormscope")
    revision = package.root.rsplit("@", 1)[1]
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    assert "amp_dtype" not in inspect.signature(cls.load_model).parameters
    assert "region" not in inspect.signature(cls.load_model).parameters
    assert "padding" not in inspect.signature(cls.load_model).parameters
    assert inspect.signature(cls.set_domain).parameters["padding"].default == 25


def test_flash_cache_separates_numbered_model_experts(tmp_path):
    import fsspec

    cache_options = StormScopeGOES.load_default_package().cache_options
    assert cache_options["same_names"] is False
    assert "cache_mapper" not in cache_options
    fs = fsspec.filesystem("memory")
    root = f"/flash-{tmp_path.name}"
    paths = [
        f"checkpoints/{kind}/{variant}/expert_0.mdlus"
        for kind in ("goes", "mrms")
        for variant in ("3km_10min", "3km_10min_flash")
    ]
    payloads = [path.encode() for path in paths]
    for path, payload in zip(paths, payloads):
        fs.pipe(f"{root}/{path}", payload)
    package = Package(
        root,
        fs=fs,
        cache=True,
        cache_options={
            **cache_options,
            "cache_storage": str(tmp_path / "cache"),
        },
    )
    resolved = [Path(package.resolve(path)) for path in paths]
    assert len(set(resolved)) == 4
    for path, local, payload in zip(paths, resolved, payloads):
        assert local.name != Path(path).name
        assert local.read_bytes() == payload
        assert Path(package.resolve(path)).read_bytes() == payload


@pytest.fixture
def shared_example_cells():
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/04_nowcasting/03_stormscope_goes_example.py"
    )
    # Execute selected repository-owned gallery cells without downloads or GPU loading.
    cells = path.read_text().split("# %%")
    return {
        name: next(cell for cell in cells if cell.lstrip().startswith(start))
        for name, start in (
            ("load", "device ="),
            ("rollout", "y, y_coords ="),
            ("plot", "goes_channel ="),
        )
    }


@pytest.mark.parametrize(
    "model_name,region",
    [
        ("3km_10min", None),
        ("3km_10min_flash", None),
        ("3km_10min_flash", (39.075, 41.075, -97.0, -95.0)),
    ],
)
def test_shared_example_loads_matching_coupled_models(
    shared_example_cells, model_name, region
):
    from unittest.mock import Mock

    namespace = {"torch": torch}
    config, loading = shared_example_cells["load"].split("package =", 1)
    exec(config, namespace)  # noqa: S102
    assert namespace["model_name"] == "3km_10min_flash"
    assert namespace["region"] is None and namespace["padding"] == 25
    namespace.update(model_name=model_name, region=region)
    for name in (
        "StormScopeBase",
        "StormScopeGOES",
        "StormScopeMRMS",
        "GOES",
        "GOESGLMGrid",
    ):
        namespace[name] = Mock()
    for name in ("StormScopeGOES", "StormScopeMRMS"):
        loaded = namespace[name].load_model.return_value
        loaded.to.return_value = loaded
        regional = loaded.set_domain.return_value
        regional.to.return_value = regional
    exec("package =" + loading, namespace)  # noqa: S102
    namespace["StormScopeBase"].load_default_package.assert_called_once_with()
    for name in ("StormScopeGOES", "StormScopeMRMS"):
        kwargs = namespace[name].load_model.call_args.kwargs
        assert kwargs["package"] is namespace["package"]
        assert kwargs["model_name"] == model_name
        assert "region" not in kwargs and "padding" not in kwargs
        assert kwargs["amp"] is True and kwargs["compile"] is True
        loaded = namespace[name].load_model.return_value
        if region is None:
            loaded.set_domain.assert_not_called()
        else:
            loaded.set_domain.assert_called_once_with(*region, padding=25)
        key = "model" if name == "StormScopeGOES" else "model_mrms"
        assert namespace[key] is (
            loaded if region is None else loaded.set_domain.return_value
        )
    assert (
        namespace["StormScopeGOES"].load_model.call_args.kwargs[
            "conditioning_data_source"
        ]
        is None
    )
    namespace["GOESGLMGrid"].assert_called_once_with(satellite="east")


def test_shared_example_keeps_context_and_current_goes_conditioning(
    shared_example_cells,
):
    from unittest.mock import Mock

    def predict(history, coords, **kwargs):
        return history[:, :, -1:] + 1, coords

    def advance(prediction, prediction_coords, history, coords):
        return torch.cat([history[:, :, 1:], prediction], dim=2), coords

    goes = Mock(side_effect=predict)
    mrms = Mock()
    goes.next_input.side_effect = advance
    mrms.next_input.side_effect = advance
    mrms.call_with_conditioning.side_effect = predict
    namespace = dict(
        x=torch.zeros(1, 1, 6, 1, 8, 12),
        x_coords={},
        x_mrms=torch.zeros(1, 1, 6, 3, 8, 12),
        x_coords_mrms={},
        model=goes,
        model_mrms=mrms,
        trange=lambda n, **kwargs: range(n),
    )
    exec(shared_example_cells["rollout"], namespace)  # noqa: S102
    assert goes.call_count == mrms.call_with_conditioning.call_count == 2
    for step, call in enumerate(mrms.call_with_conditioning.call_args_list):
        history = call.kwargs["conditioning"]
        assert history.shape == (1, 1, 6, 1, 8, 12)
        torch.testing.assert_close(
            history[:, :, -1:], torch.full_like(history[:, :, -1:], step)
        )
    assert namespace["y"].shape == namespace["x"].shape
    assert namespace["y_mrms"].shape == namespace["x_mrms"].shape
    goes.crop_output.assert_not_called()
    mrms.crop_output.assert_not_called()


@pytest.mark.parametrize("region", [None, {"lat": (36, 42), "lon": (-110, -92)}])
def test_shared_example_crops_final_fields_and_plot_grid(shared_example_cells, region):
    from types import SimpleNamespace
    from unittest.mock import Mock

    lat, lon = _grid()
    info = resolve_region(lat, lon, region, padding=25)
    y0, y1, x0, x1 = info.input_bounds
    coords = {"y": np.arange(y0, y1), "x": np.arange(x0, x1)}
    mask = torch.ones(info.input_shape, dtype=torch.bool)
    models = [
        SimpleNamespace(
            variables=[variable],
            valid_mask=mask,
            region_info=info,
            crop_output=Mock(side_effect=info.crop_output),
        )
        for variable in ("abi13c", "refc")
    ]
    namespace = dict(
        torch=torch,
        model=models[0],
        model_mrms=models[1],
        region=region,
        y_pred=torch.ones(1, 1, 1, 1, *info.input_shape),
        y_pred_coords=coords,
        y_mrms_pred=torch.ones(1, 1, 1, 1, *info.input_shape),
        y_coords_mrms_pred=coords,
        lat_out=lat[y0:y1, x0:x1],
        lon_out=lon[y0:y1, x0:x1],
    )
    preparation = shared_example_cells["plot"].split("# Prepare HRRR Lambert", 1)[0]
    exec(preparation, namespace)  # noqa: S102
    expected_shape = info.requested_mask.shape
    for name in ("y_pred", "y_mrms_pred", "lat_out", "lon_out", "plot_valid_mask"):
        assert namespace[name].shape[-2:] == expected_shape
    for model in models:
        assert model.valid_mask.shape == info.input_shape
        assert model.crop_output.call_count == int(region is not None)


def _initialized_flash_model(config, metadata):
    trunk = DiT(**config)
    sigmas = torch.tensor(metadata["sigma_grid"], dtype=torch.float32)
    intervals = len(sigmas) - 1
    projection = trunk.detokenizer.proj_layer.output_projection
    state = {f"model.{name}": value for name, value in trunk.state_dict().items()}
    state.update(
        head_weight=projection.weight.detach().unsqueeze(0).repeat(intervals, 1, 1),
        head_bias=projection.bias.detach().unsqueeze(0).repeat(intervals, 1),
        sigma_grid=sigmas,
        time_grid=(1 + sigmas).reciprocal(),
        block_boundaries=torch.tensor(metadata["block_boundaries"], dtype=torch.long),
    )
    model = FlashModel(config, metadata)
    model.load_state_dict(state, strict=True)
    return model


def test_mdlus_explicit_loader_ignores_serialized_module_path(tmp_path, monkeypatch):
    import importlib

    config = dict(
        height=8,
        width=12,
        patch_size=2,
        in_chans=3,
        base_out_chans=1,
        embed_dim=16,
        depth=1,
        num_heads=2,
        attn_kernel=3,
    )
    metadata = dict(
        checkpoint_layout="physicsnemo-dit-v1",
        sigma_grid=[3.0, 1.0, 0.0],
        block_boundaries=[0, 2],
        config={"student": {"sigma_data": 0.5}},
    )
    reference = _initialized_flash_model(config, metadata)
    current = tmp_path / "current.mdlus"
    legacy = tmp_path / "legacy.mdlus"
    reference.save(str(current))
    legacy_module = "earth2studio.models.px._stormscope_flash.preconditioner"
    with zipfile.ZipFile(current) as source, zipfile.ZipFile(legacy, "w") as target:
        for member in source.infolist():
            payload = source.read(member.filename)
            if member.filename == "args.json":
                args = json.loads(payload)
                assert args["__name__"] == "FlashModel"
                args["__module__"] = legacy_module
                payload = json.dumps(args).encode()
            target.writestr(member, payload)

    original_import = importlib.import_module

    def reject_removed_module(name, *args, **kwargs):
        assert not name.startswith("earth2studio.models.px._stormscope_flash")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", reject_removed_module)
    restored = FlashModel.from_checkpoint(str(legacy), strict=True)
    assert isinstance(restored, FlashModel)
    assert isinstance(restored.model, PhysicsNeMoDiT)
    assert restored.model_config == config
    assert restored.flash_metadata == metadata
    assert restored.state_dict().keys() == reference.state_dict().keys()
    assert not any(t.is_meta for t in (*restored.parameters(), *restored.buffers()))
    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(
            restored.state_dict()[name], expected, rtol=0, atol=0
        )

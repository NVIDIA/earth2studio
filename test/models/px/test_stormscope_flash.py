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
import importlib.util
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import torch
from physicsnemo import Module
from physicsnemo.nn.module.rope import apply_rotary_pos_emb, build_axial_rope_cos_sin_2d

from earth2studio.models.auto import Package
from earth2studio.models.nn.stormscope_flash import FlashDiT as DiT
from earth2studio.models.nn.stormscope_flash import (
    FlashModel,
    FlashPrecond,
)
from earth2studio.models.px._stormscope_flash.region import resolve_region
from earth2studio.models.px._stormscope_flash.runtime import (
    FLASH_CALLS,
    FLASH_INTERVALS,
)
from earth2studio.models.px._stormscope_flash.sampler import (
    build_flash_chain_plan,
    flash_sampler_chain,
    inference_block_boundaries,
    regional_sigma_grids,
)
from earth2studio.models.px.stormscope import StormScopeGOES, StormScopeMRMS


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rotary_matches_complex_rotation_and_row_offset(dtype):
    torch.manual_seed(5)
    q = torch.randn(2, 3, 12, 16).to(dtype)
    k = torch.randn_like(q)
    cos, sin = build_axial_rope_cos_sin_2d(10, 4, 16)
    cos, sin = cos[7:].flatten(0, 1), sin[7:].flatten(0, 1)
    actual = (apply_rotary_pos_emb(q, cos, sin), apply_rotary_pos_emb(k, cos, sin))
    row = torch.arange(7, 10).repeat_interleave(4)
    col = torch.arange(4).repeat(3)
    omega = 10000.0 ** (-torch.arange(0, 8, 2).float() / 8)
    angles = torch.cat([row[:, None] * omega, col[:, None] * omega], dim=-1)
    rotation = torch.polar(torch.ones_like(angles), angles)
    for x, rotated in zip((q, k), actual):
        paired = torch.view_as_complex(x.float().reshape(2, 3, 12, 8, 2))
        expected = torch.view_as_real(paired * rotation).reshape_as(x).to(dtype)
        torch.testing.assert_close(rotated, expected)
        assert rotated.dtype == dtype


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
    assert "_pos_emb" not in dit.state_dict()
    assert set(dit._nan_mask_tokens) == {"0", "1"}
    mask = torch.zeros(2, 8, 12, dtype=torch.bool)
    mask[0, 1, 2] = True
    dit.set_nan_pixel_mask(mask)
    assert dit.invalid_token_mask_flat.shape == (2, 24)
    assert dit.invalid_token_mask_flat.sum() == 1
    assert dit.invalid_token_mask_flat[0, 1]
    assert "invalid_token_mask_flat" not in dit.state_dict()
    with pytest.raises(ValueError):
        dit.set_nan_pixel_mask(torch.zeros(7, 12))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_flash_patch_projection_preserves_fp32_under_amp(dtype):
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
        .cuda()
        .eval()
    )
    image = torch.randn(2, 3, 8, 12, device="cuda", dtype=dtype)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        actual, shape = dit.prepare_tokens(image)
        with torch.autocast("cuda", enabled=False):
            expected = (
                torch.nn.functional.conv2d(
                    image.float(),
                    dit._patch_emb.proj.weight,
                    dit._patch_emb.proj.bias,
                    stride=2,
                )
                .to(dtype)
                .flatten(2)
                .transpose(1, 2)
            )
    assert shape == (4, 6)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_flash_legacy_mlp_names_load_without_changing_weights():
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
    )
    original = DiT(**config)
    state = original.state_dict()
    legacy = {
        key.replace(".mlp.layers.0.", ".mlp.fwd.0.").replace(
            ".mlp.layers.2.", ".mlp.fwd.3."
        ): value.clone()
        for key, value in state.items()
    }
    restored = DiT(**config)
    restored.load_state_dict(legacy, strict=True)
    assert restored.state_dict().keys() == state.keys()
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    legacy["_blocks.0.mlp.layers.0.weight"] = state["_blocks.0.mlp.layers.0.weight"]
    with pytest.raises(ValueError, match="Duplicate MLP parameter"):
        restored.load_state_dict(legacy, strict=True)


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
    experts = {
        name: _FakeExpert(grid)
        for name, grid in regional_sigma_grids(
            region_intervals=FLASH_INTERVALS[kind]
        ).items()
    }
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
        assert [(b.start, b.end) for b in plan.experts["low"].blocks] == [
            (0, 32),
            (32, 64),
            (64, 80),
        ]
    assert all(
        b.start % 16 == 0 for expert in plan.experts.values() for b in expert.blocks
    )
    with pytest.raises(ValueError):
        inference_block_boundaries(80, 6, alignment=16)


def test_flash_fused_update_matches_individual_heads():
    torch.manual_seed(8)
    expert = FlashPrecond(_TinyDiT(), [3.0, 1.0, 0.01, 0.0], [0, 3])
    expert.head_weight.data.normal_()
    expert.head_bias.data.normal_()
    state = torch.randn(2, 1, 3, 4)
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


@pytest.mark.parametrize(
    "cls,kind", [(StormScopeGOES, "goes"), (StormScopeMRMS, "mrms")]
)
def test_flash_move_invalidates_plan(cls, kind):
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    from earth2studio.models.px._stormscope_flash.runtime import FlashRuntime

    model._flash_runtime = FlashRuntime(kind)
    model._flash_runtime.plan = object()
    model.to("cpu")
    assert model._flash_runtime.plan is None


class _FakeExpert(torch.nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.register_buffer("sigma_grid", grid)
        self.calls = 0

    def prepare_inference_block(self, start, end):
        return (
            self.sigma_grid[start],
            torch.ones(1),
            torch.zeros(1),
            torch.tensor(float(end - start)),
        )

    def prepare_condition_patch(self, condition):
        return None

    def forward(self, state, sigma, condition, **kwargs):
        self.calls += 1
        return state

    def set_image_size(self, *shape):
        self.shape = shape


class _TinyDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._final_layer = torch.nn.Module()
        self._final_layer.linear = torch.nn.Linear(1, 1)
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


def test_regional_instances_keep_independent_geometry():
    lat, lon = _grid()
    infos = [
        resolve_region(lat, lon, b)
        for b in (None, {"lat": (36, 38), "lon": (-110, -108)})
    ]
    assert infos[0].input_shape != infos[1].input_shape
    assert infos[0].input_shape == (400, 600)


def test_glm_interpolation_and_normalization_order_is_baseline():
    assert StormScopeMRMS.interpolate_glm is StormScopeMRMS.interpolate_glm
    assert StormScopeMRMS.fetch_glm is StormScopeMRMS.fetch_glm
    model = StormScopeMRMS.__new__(StormScopeMRMS)
    torch.nn.Module.__init__(model)
    model.means = torch.zeros(1, 3, 1, 1)
    model.stds = torch.ones(1, 3, 1, 1)
    model.glm_mask = torch.tensor([False, False, True])
    model.glm_interp = lambda x: x.mean(-1, keepdim=True)
    counts = torch.tensor([[[[0.0, 8.0]]]])
    regridded = model.interpolate_glm(counts)
    state = torch.cat(
        (torch.zeros_like(regridded), torch.zeros_like(regridded), regridded), dim=1
    )
    actual = model._normalize_state(state)
    torch.testing.assert_close(actual[:, 2:], torch.log1p(regridded))
    assert not torch.allclose(actual[:, 2:], torch.log1p(counts).mean(-1, keepdim=True))


@pytest.mark.package
@pytest.mark.parametrize("cls", [StormScopeGOES, StormScopeMRMS])
def test_flash_package(cls):
    if not torch.cuda.is_available():
        pytest.skip("NATTEN real-weight test requires CUDA")
    model = cls.load_model(
        cls.load_default_package(model_name="3km_10min_flash"),
        model_name="3km_10min_flash",
        region={"lat": (38.0, 39.0), "lon": (-98.0, -97.0)},
    ).cuda()
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
    from earth2studio.models.px._stormscope_flash.runtime import FlashRuntime

    model._flash_runtime = FlashRuntime("goes")
    model.means = torch.full((1, 8, 1, 1), 10.0)
    model.stds = torch.full((1, 8, 1, 1), 2.0)
    model.glm_mask = torch.zeros(8, dtype=torch.bool)
    model.valid_mask = torch.ones(8, 12, dtype=torch.bool)
    expert = torch.nn.Module()
    expert.model = MaskModel()
    model.stage_models = torch.nn.ModuleList([expert])
    x = torch.full((2, 1, 6, 8, 8, 12), 12.0)
    x[1, 0, 2, 3, 4, 5] = torch.nan
    actual = model._normalize_state(x)
    assert actual[1, 0, 2, 3, 4, 5] == 0
    assert expert.model.mask[1, 4, 5] and not expert.model.mask[0, 4, 5]
    assert torch.isnan(x[1, 0, 2, 3, 4, 5])
    assert actual[1, 0, 1, 3, 4, 5] == 1
    x[0, 0, 0, 0, 0, 0] = torch.inf
    assert torch.isinf(model._normalize_state(x)[0, 0, 0, 0, 0, 0])


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
def test_mdlus_round_trip_preserves_heads_and_updates(tmp_path):
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
        sigma_grid=[3.0, 1.0, 0.0],
        block_boundaries=[0, 2],
        config={"student": {"sigma_data": 0.5}},
    )
    reference = FlashPrecond(DiT(**config), [3.0, 1.0, 0.0], [0, 2]).eval()
    with torch.no_grad():
        reference.head_weight.normal_(std=0.02)
        reference.head_bias.normal_(std=0.02)
    model = FlashModel(config, metadata)
    model.load_state_dict(reference.state_dict(), strict=True)
    path = tmp_path / "tiny.mdlus"
    model.save(str(path))
    restored = Module.from_checkpoint(str(path), strict=True)
    assert isinstance(restored, FlashModel)
    assert restored.model_config == config
    assert restored.flash_metadata == metadata
    assert not any(p.is_meta or p.requires_grad for p in restored.parameters())
    assert not any(b.is_meta for b in restored.buffers())
    assert restored.state_dict().keys() == reference.state_dict().keys()
    for key, tensor in reference.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key]), key
    reference.cuda()
    restored.cuda()
    state = torch.randn(2, 1, 8, 12, device="cuda")
    condition = torch.randn(2, 2, 8, 12, device="cuda")
    sigma, weight, bias, delta = reference.prepare_inference_block(0, 2)
    kwargs = dict(inference_weight=weight, inference_bias=bias, inference_delta=delta)
    with torch.no_grad():
        expected = reference(state, sigma, condition, **kwargs)
        actual = restored(state, sigma, condition, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with zipfile.ZipFile(path) as archive:
        assert "model.pt" in archive.namelist()
        assert json.loads(archive.read("args.json"))["__name__"] == "FlashModel"
        assert not any("optimizer" in name for name in archive.namelist())
    incomplete = dict(reference.state_dict())
    incomplete.pop("head_weight")
    with pytest.raises(RuntimeError, match="head_weight"):
        FlashModel(config, metadata).load_state_dict(incomplete, strict=True)


def test_mdlus_file_integrity_guard(tmp_path, monkeypatch):
    from earth2studio.models.px._stormscope_flash.runtime import (
        _validate_checkpoint_file,
    )

    payload = b"checkpoint integrity test"
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / "expert_0.mdlus"
    path.write_bytes(payload)
    spec = dict(
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


@pytest.fixture
def flash_example():
    path = (
        Path(__file__).resolve().parents[3]
        / "examples/04_nowcasting/03_stormscope_goes_example.py"
    )
    spec = importlib.util.spec_from_file_location("flash_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flash_example_geographic_flags(flash_example):
    full = flash_example.parse_args([])
    assert full.lat is None and full.lon is None
    assert full.model == "3km_10min"
    assert flash_example.padding == 25
    regional = flash_example.parse_args(
        [
            "--model",
            "3km_10min_flash",
            "--lat",
            "38",
            "43",
            "--lon",
            "-99",
            "-93",
        ]
    )
    assert regional.lat == [38, 43] and regional.lon == [-99, -93]
    assert set(vars(regional)) == {"model", "lat", "lon"}


@pytest.mark.parametrize(
    "argv",
    [
        ["--lat", "38", "43"],
        ["--lon", "-99", "-93"],
        ["--satellite", "goes16"],
        ["--no-compile"],
        ["--padding", "15"],
        ["--model", "invalid"],
        ["--lat", "38", "43", "--lon", "-99", "-93"],
    ],
)
def test_flash_example_rejects_invalid_flags(flash_example, argv):
    with pytest.raises(SystemExit) as error:
        flash_example.parse_args(argv)
    assert error.value.code == 2


def test_flash_region_uses_fifteen_available_edge_pixels():
    lat, lon = _grid()
    # Output starts 15 pixels inside the grid on both axes. Context reaches the
    # native edge without fabricating the remaining ten requested pixels.
    region = {"lat": (lat[15, 0], lat[249, 0]), "lon": (lon[0, 15], lon[0, 249])}
    info = resolve_region(lat, lon, region, padding=25)
    assert info.input_bounds == (0, 276, 0, 276)
    assert info.padding == (15, 26, 15, 26)
    assert all(n >= 200 and n % 4 == 0 for n in info.input_shape)


def test_flash_example_selects_east_satellite_at_handover(flash_example):
    from earth2studio.data import GOES

    transition = GOES.GOES_HISTORY_RANGE["goes19"][0]
    assert (
        flash_example.goes_east_satellite(transition - timedelta(minutes=1)) == "goes16"
    )
    assert flash_example.goes_east_satellite(transition) == "goes19"
    assert flash_example.goes_east_satellite(datetime(2024, 3, 13, 23, 30)) == "goes16"
    with pytest.raises(ValueError, match="No operational"):
        flash_example.goes_east_satellite(datetime(2010, 1, 1))


def test_flash_example_preserves_repeated_and_different_region_outputs(
    flash_example, tmp_path
):
    flash_example.output_dir = tmp_path
    region = {"lat": (39.075, 41.075), "lon": (-97.0, -95.0)}
    first = flash_example.create_output_path(region, "3km_10min_flash")
    first.mkdir()
    (first / "sentinel").write_text("keep")
    second = flash_example.create_output_path(region, "3km_10min_flash")
    third = flash_example.create_output_path(
        {"lat": (38.0, 40.0), "lon": (-97.0, -95.0)}, "3km_10min_flash"
    )
    full = flash_example.create_output_path(None, "3km_10min_flash")
    baseline = flash_example.create_output_path(None, "3km_10min")
    assert baseline.parent.parent != full.parent.parent
    assert first.parent.name == "run_001"
    assert second.parent.name == "run_002"
    assert third.parent.parent != first.parent.parent
    assert "full" in full.parent.parent.name
    assert (first / "sentinel").read_text() == "keep"


def test_flash_example_checks_bounds_before_loading_models(
    flash_example, tmp_path, monkeypatch
):
    from earth2studio.models.px import StormScopeGOES

    latitude, longitude = _grid()
    monkeypatch.setattr(
        StormScopeGOES, "_resolve_model_entry", lambda package, name: (name, {})
    )
    monkeypatch.setattr(
        StormScopeGOES,
        "_build_grid_and_times",
        lambda package, entry: (torch.as_tensor(latitude), torch.as_tensor(longitude)),
    )

    def unexpected_load(*args, **kwargs):
        pytest.fail("Checkpoint loading started before geographic validation")

    monkeypatch.setattr(StormScopeGOES, "load_model", unexpected_load)
    monkeypatch.setattr(
        StormScopeGOES, "load_default_package", lambda **kwargs: Package(str(tmp_path))
    )
    flash_example.output_dir = tmp_path / "outputs"
    with pytest.raises(ValueError, match="Invalid geographic request.*outside"):
        flash_example.main(
            [
                "--model",
                "3km_10min_flash",
                "--lat",
                "39.075",
                "81.075",
                "--lon",
                "-97",
                "-95",
            ]
        )
    assert not flash_example.output_dir.exists()


@pytest.mark.parametrize(
    "kind,flash_class", [("goes", StormScopeGOES), ("mrms", StormScopeMRMS)]
)
def test_flash_hub_registry_keeps_teacher_selection_separate(
    tmp_path, kind, flash_class
):
    from earth2studio.models.px.stormscope import StormScopeGOES

    teacher = {"description": "teacher", "checkpoints": [{"path": "teacher.mdlus"}]}
    flash = {
        "description": "flash",
        "flash": True,
        "checkpoints": [{"path": f"checkpoints/{kind}/3km_10min_flash/high.mdlus"}],
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
    baseline = StormScopeGOES if kind == "goes" else StormScopeMRMS
    assert baseline._resolve_model_entry(package, "3km_10min")[1] == teacher
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


@pytest.mark.parametrize("flash_class", [StormScopeGOES, StormScopeMRMS])
def test_flash_hub_package_uses_revision_and_separate_cache(
    monkeypatch, flash_class, tmp_path
):
    from unittest.mock import ANY, Mock

    import earth2studio.models.px.stormscope as module

    constructor = Mock()
    constructor.default_cache.return_value = str(tmp_path / "flash-cache")
    monkeypatch.setattr(module, "Package", constructor)
    revision = "a" * 40
    assert (
        flash_class.load_default_package(
            model_name="3km_10min_flash", revision=revision
        )
        is constructor.return_value
    )
    constructor.assert_called_once_with(
        f"hf://nvidia/stormscope-goes-mrms@{revision}",
        cache_options={
            "cache_storage": str(tmp_path / "flash-cache"),
            "cache_mapper": ANY,
        },
    )
    constructor.default_cache.assert_called_once_with(f"stormscope_flash/{revision}")


def test_flash_example_uses_default_hub_package(flash_example, tmp_path, monkeypatch):
    from earth2studio.models.px import StormScopeGOES

    class ReachedDefaultPackage(Exception):
        pass

    def default_package(**kwargs):
        raise ReachedDefaultPackage()

    assert not hasattr(flash_example, "package_path")
    monkeypatch.delenv("STORMSCOPE_FLASH_PACKAGE", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setattr(StormScopeGOES, "load_default_package", default_package)
    flash_example.output_dir = tmp_path / "outputs"
    with pytest.raises(ReachedDefaultPackage):
        flash_example.main([])
    assert not flash_example.output_dir.exists()


def test_flash_default_package_is_pinned():
    package = StormScopeGOES.load_default_package(model_name="3km_10min_flash")
    revision = package.root.rsplit("@", 1)[1]
    assert len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)
    mapper = package.cache_options["cache_mapper"]
    assert mapper("checkpoints/goes/3km_10min_flash/expert_0.mdlus") != mapper(
        "checkpoints/mrms/3km_10min_flash/expert_0.mdlus"
    )
    assert revision in package.cache_options["cache_storage"]


def test_flash_cache_separates_numbered_model_experts(tmp_path):
    import fsspec

    mapper = StormScopeGOES.load_default_package(
        model_name="3km_10min_flash"
    ).cache_options["cache_mapper"]
    fs = fsspec.filesystem("memory")
    root = f"/flash-{tmp_path.name}"
    paths = [
        f"checkpoints/{kind}/3km_10min_flash/expert_0.mdlus"
        for kind in ("goes", "mrms")
    ]
    for path, payload in zip(paths, (b"goes weights", b"mrms weights")):
        fs.pipe(f"{root}/{path}", payload)
    package = Package(
        root,
        fs=fs,
        cache=True,
        cache_options={
            "cache_storage": str(tmp_path / "cache"),
            "cache_mapper": mapper,
        },
    )
    resolved = [Path(package.resolve(path)) for path in paths]
    assert resolved[0] != resolved[1]
    for path, local, payload in zip(
        paths, resolved, (b"goes weights", b"mrms weights")
    ):
        assert local.suffix == ".mdlus"
        assert local.read_bytes() == payload
        assert Path(package.resolve(path)).read_bytes() == payload


@pytest.mark.parametrize("model_name", ["3km_10min", "3km_10min_flash"])
def test_shared_example_selects_package_and_model(
    flash_example, tmp_path, monkeypatch, model_name
):
    """Both CLI selections reach the matching loader without changing teacher defaults."""
    from earth2studio.models.px import StormScopeGOES

    latitude, longitude = _grid()
    selected = []
    package = Package(str(tmp_path))

    def load_package(**kwargs):
        selected.append(kwargs["model_name"])
        return package

    class ReachedLoader(Exception):
        pass

    def load_model(actual_package, **kwargs):
        assert actual_package is package
        assert kwargs["model_name"] == model_name
        assert kwargs["amp"] is True and kwargs["compile"] is True
        if model_name == "3km_10min":
            assert set(kwargs) == {"model_name", "amp", "compile"}
        else:
            assert kwargs["region"] is None and kwargs["padding"] == 25
            assert kwargs["amp_dtype"] == torch.float16
        raise ReachedLoader

    monkeypatch.setattr(StormScopeGOES, "load_default_package", load_package)
    monkeypatch.setattr(
        StormScopeGOES, "_resolve_model_entry", lambda *a: (model_name, {})
    )
    monkeypatch.setattr(
        StormScopeGOES,
        "_build_grid_and_times",
        lambda *a: (torch.as_tensor(latitude), torch.as_tensor(longitude)),
    )
    monkeypatch.setattr(StormScopeGOES, "load_model", load_model)
    flash_example.output_dir = tmp_path / "outputs"
    with pytest.raises(ReachedLoader):
        flash_example.main(["--model", model_name])
    assert selected == [model_name]

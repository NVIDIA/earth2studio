# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import datetime

import numpy as np
import pytest
import torch

import earth2studio.models.dx.corrdiff_era5_hrrr as corrdiff_module
from earth2studio.grids import Region, resolve_grid
from earth2studio.models.base import DomainAdaptable
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import CorrDiffEra5Hrrr
from earth2studio.models.dx.corrdiff_era5_hrrr import ERA5_VARIABLES, OUTPUT_VARIABLES
from earth2studio.utils import handshake_dim
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch


class PhooNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_t = []

    def forward(self, x, t, condition=None):
        self.seen_t.append(t.detach().clone())
        return torch.zeros_like(x)


@pytest.fixture
def rng_model(monkeypatch):
    p = CorrDiffEra5Hrrr.__new__(CorrDiffEra5Hrrr)
    torch.nn.Module.__init__(p)
    p.register_buffer("lat_input_grid", torch.linspace(30, 27, 4))
    p.lat_input_numpy = p.lat_input_grid.numpy()
    p.lon_input_numpy = np.arange(8) + 260.0
    p.register_buffer("lat_output_grid", torch.ones(2, 3) * 28)
    p._lat_out_cpu = p.lat_output_grid.numpy()
    p._lon_out_cpu = np.ones((2, 3)) * 262
    p.hrrr_y, p.hrrr_x = np.arange(2), np.arange(3)
    p.era5_variables, p.output_variables = np.array(["t2m"]), np.array(["t2m"])
    p.number_of_samples = 2
    p.number_of_steps = 2
    p.solver = "euler"
    p.network_kind = "edm"
    p.distilled = False
    p.sigma_min, p.sigma_max, p.rho = 0.01, 1.0, 7.0
    p.t_max, p.shift = 0.99, 1.0
    p.amp = False
    p.register_buffer("invariants", torch.zeros(1, 2, 3))
    p.register_buffer("out_center", torch.zeros(1, 1, 2, 3))
    p.register_buffer("out_scale", torch.ones(1, 1, 2, 3))
    p.preprocess_input = lambda x, time: None

    class Scheduler:
        def __init__(self, **kwargs):
            pass

        def get_denoiser(self, **kwargs):
            return None

        def sigma(self, t):
            return t

    monkeypatch.setattr(corrdiff_module, "EDMNoiseScheduler", Scheduler)
    monkeypatch.setattr(corrdiff_module, "RectifiedFlowNoiseScheduler", Scheduler)
    monkeypatch.setattr(
        corrdiff_module, "sample", lambda denoiser, latents, scheduler, **kw: latents
    )
    p._rf_time_steps = lambda device, scheduler: torch.tensor([1.0, 0.0], device=device)
    return p


def test_corrdiff_era5_hrrr_conformance(rng_model):
    assert check_diagnostic_contract(rng_model) == []


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("kind", ["edm", "rectified_flow"])
def test_corrdiff_era5_hrrr_rng(rng_model, device, kind):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA missing")
    p = rng_model.to(device)
    p.network_kind = kind
    x = torch.zeros(1, 4, 8, device=device)
    time = datetime(2025, 7, 1)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if x.is_cuda else None
    p.set_rng(42, reset=False)
    first = p._forward(x, time)
    p.set_rng(99, reset=False)
    second = p._forward(x, time)
    assert not torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    p.set_rng(42)
    assert torch.equal(first, p._forward(x, time))
    assert torch.equal(second, p._forward(x, time))
    p.set_rng(43)
    assert not torch.equal(first, p._forward(x, time))
    assert torch.equal(cpu_state, torch.get_rng_state())
    if x.is_cuda:
        assert torch.equal(cuda_state, torch.cuda.get_rng_state(device))


def test_corrdiff_era5_hrrr_unseeded_rng(rng_model):
    state = torch.get_rng_state()
    rng_model._forward(torch.zeros(1, 4, 8), datetime(2025, 7, 1))
    assert not torch.equal(state, torch.get_rng_state())


@pytest.fixture
def model_args():
    lat, lon = torch.meshgrid(
        torch.linspace(27, 29, 4), torch.linspace(261, 264, 6), indexing="ij"
    )
    return dict(
        network=PhooNet(),
        lat_input_grid=torch.arange(30, 26, -0.5),
        lon_input_grid=torch.arange(260, 265, 0.5),
        lat_output_grid=lat + 0.05 * lon.sin(),
        lon_output_grid=lon,
        hrrr_y=torch.arange(4) * 3000,
        hrrr_x=torch.arange(6) * 3000,
        era5_center=torch.zeros(26),
        era5_scale=torch.ones(26),
        out_center=torch.arange(99).float(),
        out_scale=torch.ones(99) * 2,
        invariants=torch.randn(3, 4, 6),
        presence_flags=["tcwv", "sp"],
        number_of_samples=2,
        number_of_steps=3,
        amp=False,
    )


@pytest.mark.parametrize(
    "kind,prediction,batch_size,device",
    [
        ("rectified_flow", "x0", 1, "cpu"),
        ("rectified_flow", "flow", 2, "cpu"),
        ("edm", "x0", 2, "cpu"),
        pytest.param(
            "rectified_flow",
            "x0",
            2,
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_corrdiff_era5_hrrr(model_args, kind, prediction, batch_size, device):
    dx = CorrDiffEra5Hrrr(
        **model_args, network_kind=kind, prediction_type=prediction
    ).to(device)
    dx.set_rng(0)
    if kind == "edm":
        from physicsnemo.diffusion.preconditioners import EDMPreconditioner

        dx.network = EDMPreconditioner(dx.network, sigma_data=0.5).to(device)
    coords = coord_array_like(
        dx.input_coords(),
        {
            "batch": np.arange(batch_size),
            "time": np.array(["2025-07-01T18", "2025-07-02T00"], dtype="datetime64[h]"),
        },
    )
    x = torch.randn(coords.shape, device=device)
    cond = dx.preprocess_input(x[0, 0], datetime(2025, 7, 1, 18))
    assert cond["cond_concat"].shape == (1, 30, 4, 6)
    assert cond["cond_vec"].shape == (1, 4)
    assert torch.all(cond["cond_vec"][0, 2:] == 1)
    assert cond["cond_concat"][0, 26].abs().max() <= 1
    assert torch.isfinite(cond["cond_concat"]).all()
    field = from_torch(x, coords)
    result = dx(field)
    out, out_coords = result.e2s.to_torch()
    assert out.shape == (batch_size, 2, 2, 99, 4, 6)
    assert out.device == torch.device(device) and torch.isfinite(out).all()
    np.testing.assert_array_equal(coords.coords["variable"], ERA5_VARIABLES)
    np.testing.assert_array_equal(out_coords["variable"], OUTPUT_VARIABLES)
    expected = dx.output_coords(coords)
    for index, dim in enumerate(
        ["batch", "sample", "time", "variable", "hrrr_y", "hrrr_x"]
    ):
        handshake_dim(out_coords, dim, index)
        np.testing.assert_array_equal(out_coords[dim], expected.coords[dim])
    dx.set_rng(0)
    torch.testing.assert_close(out, dx(field).e2s.to_torch()[0])
    assert not torch.allclose(out[:, 0], out[:, 1])
    dx.number_of_samples = 1
    dx.set_rng(0)
    single = dx(field).e2s.to_torch()[0]
    torch.testing.assert_close(single[0, 0, 0], out[0, 0, 0])
    if kind == "rectified_flow":
        from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler

        assert 1 < max(float(t.max()) for t in model_args["network"].seen_t) < 999
        scheduler = RectifiedFlowNoiseScheduler(t_max=dx.t_max)
        shifted = dx._rf_time_steps(torch.device(device), scheduler)
        dx.shift = 1
        unshifted = dx._rf_time_steps(torch.device(device), scheduler)
        assert shifted.shape == unshifted.shape and shifted[-1] == 0
        assert torch.all(shifted[1:-1] > unshifted[1:-1])


def test_corrdiff_era5_hrrr_exceptions(model_args):
    dx = CorrDiffEra5Hrrr(**model_args)
    assert dx.network_kind == "rectified_flow"
    coords = coord_array_like(
        dx.input_coords(),
        {"batch": [0], "time": np.array(["2025-07-01"], dtype="datetime64[D]")},
    )
    x = from_torch(torch.zeros(coords.shape), coords)
    for dim in ["time", "variable", "lat", "lon"]:
        wrong = x.copy(deep=True)
        if dim == "time":
            wrong = wrong.rename({dim: "wrong"})
        else:
            values = coords.coords[dim].values
            wrong = wrong.assign_coords(
                {dim: values[::-1] if dim == "variable" else values + 1}
            )
        with pytest.raises((KeyError, ValueError)):
            dx(wrong)
    for key, value in dict(
        network_kind="ddpm",
        prediction_type="epsilon",
        solver="invalid",
        number_of_samples=0,
        number_of_steps=0,
        t_max=1,
        shift=0,
    ).items():
        with pytest.raises(ValueError):
            CorrDiffEra5Hrrr(**(model_args | {key: value}))
    with pytest.raises(ValueError, match="variant"):
        CorrDiffEra5Hrrr.load_model(None, variant="v_pred")


def _tiny_dit_args(input_size: tuple[int, int], backend: str = "timm") -> dict:
    """Constructor arguments of a small DiT backbone. The default timm attention
    runs anywhere (``pos_embed="none"`` keeps it crop-size agnostic); the
    ``natten2d_rope`` variant is the hosted model's backbone with a 3-token kernel."""
    args = dict(
        input_size=input_size,
        in_channels=99 + 30,  # 99 latents + 26 ERA5 + cos zenith + 3 invariants
        out_channels=99,
        patch_size=2,
        hidden_size=16,
        depth=1,
        num_heads=2,
        condition_dim=4,
        attention_backend=backend,
        tokenizer_kwargs={"pos_embed": "none"},
    )
    if backend == "natten2d_rope":
        args["attn_kwargs"] = {"attn_kernel": 3, "qk_norm": True, "norm_layer": "torch"}
    return args


def _tiny_distilled(input_size: tuple[int, int] = (4, 6), backend: str = "timm"):
    """A distilled generator around a small DiT."""
    from earth2studio.models.dx.corrdiff_era5_hrrr import _CorrDiffEra5HrrrDistilled

    torch.manual_seed(0)
    net = _CorrDiffEra5HrrrDistilled(_tiny_dit_args(input_size, backend), n_grid=8)
    # the DiT initialization zeroes the output projection; give it and the
    # heads some signal so the tests see non-trivial fields
    with torch.no_grad():
        for linear in (net.dit.detokenizer.proj_layer.output_projection, *net.heads):
            torch.nn.init.normal_(linear.weight, std=0.02)
            torch.nn.init.normal_(linear.bias, std=0.02)
    return net.eval().requires_grad_(False)


def test_corrdiff_era5_hrrr_distilled(model_args, tmp_path):
    """The distilled generator runs through the wrapper, only in its trained layouts."""
    from earth2studio.models.dx.corrdiff_era5_hrrr import _load_distilled

    student = _tiny_distilled()
    args = model_args | {"network": student, "number_of_steps": 2}
    dx = CorrDiffEra5Hrrr(**args)
    assert dx.distilled and not CorrDiffEra5Hrrr(**model_args).distilled
    coords = coord_array_like(
        dx.input_coords(),
        {
            "batch": np.arange(1),
            "time": np.array(["2025-07-01T18", "2025-07-02T00"], dtype="datetime64[h]"),
        },
    )
    field = from_torch(torch.randn(coords.shape), coords)
    dx.set_rng(0)
    out, out_coords = dx(field).e2s.to_torch()
    assert out.shape == (1, 2, 2, 99, 4, 6) and torch.isfinite(out).all()
    np.testing.assert_array_equal(out_coords["variable"], OUTPUT_VARIABLES)
    dx.set_rng(0)
    torch.testing.assert_close(
        out, dx(field).e2s.to_torch()[0]
    )  # seeded -> reproducible
    assert not torch.allclose(out[:, 0], out[:, 1])  # members differ
    dx.number_of_steps = 1
    dx.set_rng(0)
    out_1 = dx(field).e2s.to_torch()[0]
    assert out_1.shape == out.shape and not torch.allclose(out_1, out)
    # unsupported layouts are rejected at construction and at call time
    with pytest.raises(ValueError, match="1 or 2"):
        CorrDiffEra5Hrrr(**(args | {"number_of_steps": 3}))
    dx.number_of_steps = 4
    with pytest.raises(ValueError, match="1 or 2"):
        dx(field)
    # the generator is rebuilt from a PhysicsNeMo ``.mdlus`` archive (args + state)
    import io
    import json
    import zipfile

    path = tmp_path / "distilled.mdlus"
    buffer = io.BytesIO()
    torch.save(student.state_dict() | {"device_buffer": torch.empty(0)}, buffer)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "args.json",
            json.dumps(
                {
                    "__name__": "Anything",
                    "__module__": "anything",
                    "__args__": dict(dit_args=_tiny_dit_args((4, 6)), n_grid=8),
                }
            ),
        )
        archive.writestr("model.pt", buffer.getvalue())
    dx.network = _load_distilled(str(path))
    dx.number_of_steps = 2
    dx.set_rng(0)
    torch.testing.assert_close(dx(field).e2s.to_torch()[0], out)


@pytest.fixture
def domain_args(model_args):
    """``model_args`` on an 8 x 12 output grid, large enough to cut regional views from."""
    lat, lon = torch.meshgrid(
        torch.linspace(27, 29, 8), torch.linspace(261, 264, 12), indexing="ij"
    )
    hrrr = resolve_grid("hrrr")
    return model_args | dict(
        lat_output_grid=lat + 0.05 * lon.sin(),
        lon_output_grid=lon,
        hrrr_y=torch.from_numpy(hrrr.y[500:508].copy()),  # HRRR rows 500..507
        hrrr_x=torch.from_numpy(hrrr.x[800:812].copy()),  # HRRR columns 800..811
        invariants=torch.randn(3, 8, 12),
    )


def _era5_field(dx):
    coords = coord_array_like(
        dx.input_coords(),
        {
            "batch": np.arange(1),
            "time": np.array(["2025-07-01T18"], dtype="datetime64[h]"),
        },
    )
    return from_torch(torch.randn(coords.shape), coords)


def test_corrdiff_era5_hrrr_with_region(domain_args):
    """with_region returns a view on the block of the grid covering a box, selects
    from the full footprint, and rejects regions the footprint cannot serve."""
    dx = CorrDiffEra5Hrrr(**domain_args)
    dx.set_rng(0)
    assert isinstance(dx, DomainAdaptable)
    assert (dx.hrrr_lat_lim, dx.hrrr_lon_lim) == ((500, 508), (800, 812))
    assert dx._patch_size == (1, 1)
    # cell centers in rows 2..4 (lat 27.52 .. 28.10) and cols 4..7 (lon 262.1 ..
    # 262.9) fall inside the box; one bracketing row / column per side makes the
    # block's cell areas contain it (G1)
    box = Region.box(262.0, 27.5, 263.0, 28.3)
    view = dx.with_region(box)
    assert (view.hrrr_lat_lim, view.hrrr_lon_lim) == ((501, 506), (803, 809))
    assert (dx.hrrr_lat_lim, dx.hrrr_lon_lim) == ((500, 508), (800, 812))
    assert view.network is dx.network and view._rng_generator is not dx._rng_generator
    torch.testing.assert_close(view.lat_output_grid, dx.lat_output_grid[1:6, 3:9])
    torch.testing.assert_close(view.invariants, dx.invariants[:, 1:6, 3:9])
    np.testing.assert_array_equal(view.hrrr_y, dx.hrrr_y[1:6])
    assert view._lat_out_cpu.min() < 27.5 and view._lat_out_cpu.max() > 28.3
    assert view._lon_out_cpu.min() < 262.0 and view._lon_out_cpu.max() > 263.0
    # the input stays the native ERA5 grid; the output is the block
    field = _era5_field(view)
    out, out_coords = view(field).e2s.to_torch()
    assert out.shape == (1, 2, 1, 99, 5, 6) and torch.isfinite(out).all()
    np.testing.assert_array_equal(out_coords["hrrr_y"], dx.hrrr_y[1:6])
    np.testing.assert_array_equal(out_coords["hrrr_x"], dx.hrrr_x[3:9])
    np.testing.assert_array_equal(
        view.output_coords(field).coords["lat"].values, dx._lat_out_cpu[1:6, 3:9]
    )
    # a view selects from the full footprint (G3): the same box gives the same
    # block, and a larger box reaches rows outside the view
    assert view.with_region(box).hrrr_lat_lim == (501, 506)
    outer = view.with_region(Region.box(262.0, 27.2, 263.0, 28.3))
    assert (outer.hrrr_lat_lim, outer.hrrr_lon_lim) == ((500, 506), (803, 809))
    for region, match in (
        (Region.box(262.0, 27.5, 263.0, 35.0), "footprint"),
        (Region.box(262.0, 27.5, 262.0, 28.3), "positive extent"),
        (Region.box(263.0, 27.5, 262.0, 28.3), "positive extent"),
        (Region.box(262.0, 27.6, 263.0, 27.62), "no grid cell"),
        (Region.band(27.5, 28.3), "box regions"),
        (Region.circle(28.0, -97.5, 1e4), "box regions"),
    ):
        with pytest.raises(ValueError, match=match):
            dx.with_region(region)
    dx._min_domain_cells = 7  # a NATTEN kernel that does not fit 5 rows
    with pytest.raises(ValueError, match="minimum"):
        dx.with_region(box)
    # patch snapping grows outward, then falls back to the low edge, and only
    # shrinks when the axis itself is not a multiple of the patch
    from earth2studio.models.dx.corrdiff_era5_hrrr import _match_lon_convention
    from earth2studio.models.nn.corrdiff import snap_to_patch

    assert snap_to_patch(2, 5, 8, 2) == (2, 6)
    assert snap_to_patch(5, 8, 8, 2) == (4, 8)
    assert snap_to_patch(0, 7, 7, 2) == (0, 6)
    assert _match_lon_convention(180.0, np.array([-100.0])) == 180.0
    assert _match_lon_convention(-98.0, np.array([262.0])) == 262.0


def test_corrdiff_era5_hrrr_with_region_dit(domain_args):
    """Real DiT backbones (teacher and distilled) run on a regional view: the DiT is
    rebound to the block (grown to its patch size) and restored to the full grid
    afterwards."""
    from physicsnemo.diffusion.utils import ConcatConditionWrapper
    from physicsnemo.models.dit import DiT
    from tensordict import TensorDict

    from earth2studio.models.nn.corrdiff import dit_bound_to, find_dit

    box = Region.box(262.0, 27.5, 263.0, 28.3)
    torch.manual_seed(0)
    teacher = ConcatConditionWrapper(DiT(**_tiny_dit_args((8, 12)))).eval()
    student = _tiny_distilled((8, 12))
    for network in (teacher, student):
        dx = CorrDiffEra5Hrrr(
            **(
                domain_args
                | dict(network=network, number_of_steps=2, number_of_samples=1)
            )
        )
        assert dx._patch_size == (2, 2) and dx._min_domain_cells == 2
        # the 5 bracketing rows grow to 6 for the patch size, the 6 columns stay
        view = dx.with_region(box)
        assert (view.hrrr_lat_lim, view.hrrr_lon_lim) == ((501, 507), (803, 809))
        field = _era5_field(view)
        out, out_coords = view(field).e2s.to_torch()
        assert out.shape == (1, 1, 1, 99, 6, 6) and torch.isfinite(out).all()
        np.testing.assert_array_equal(out_coords["hrrr_y"], dx.hrrr_y[1:7])
        # the shared backbone is restored to its construction grid after a
        # regional run and still serves the full grid
        dit = find_dit(network)
        assert (dit.detokenizer.h_patches, dit.detokenizer.w_patches) == (4, 6)
        full, _ = dx(field).e2s.to_torch()
        assert full.shape == (1, 1, 1, 99, 8, 12) and torch.isfinite(full).all()
        with pytest.raises(ValueError, match="multiple"):
            with dit_bound_to(dit, 9, 12):
                pass
    # the generator binds itself to the noise it is given
    cond = TensorDict(
        {"cond_concat": torch.randn(1, 30, 4, 6), "cond_vec": torch.randn(1, 4)},
        batch_size=[1],
    )
    assert student.sample(torch.randn(1, 99, 4, 6), cond).shape == (1, 99, 4, 6)
    with pytest.raises(ValueError, match="multiple"):
        student.sample(torch.randn(1, 99, 3, 6), cond)
    # a backbone with a fixed-size positional embedding cannot be cropped
    fixed = ConcatConditionWrapper(
        DiT(**(_tiny_dit_args((8, 12)) | {"tokenizer_kwargs": {}}))
    ).eval()
    with pytest.raises(ValueError, match="positional embedding"):
        CorrDiffEra5Hrrr(**(domain_args | dict(network=fixed))).with_region(box)


@pytest.mark.parametrize("backbone", ["teacher", "distilled"])
def test_corrdiff_era5_hrrr_with_region_natten_rope(model_args, backbone, monkeypatch):
    """The hosted model's NATTEN / RoPE backbone on a regional view: the view runs,
    the construction grid (latent size, RoPE tables) is restored after success and
    after a failure, and a seeded full-domain result is unchanged by a regional run."""
    from physicsnemo.diffusion.utils import ConcatConditionWrapper
    from physicsnemo.models.dit import DiT

    import earth2studio.models.dx.corrdiff_era5_hrrr as corrdiff_module
    from earth2studio.models.nn.corrdiff import find_dit

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # a 16 x 24 grid so a block of 6+ cells (the 3-token kernel times the patch) fits
    lat, lon = torch.meshgrid(
        torch.linspace(27, 29, 16), torch.linspace(261, 264, 24), indexing="ij"
    )
    torch.manual_seed(0)
    if backbone == "teacher":
        network = ConcatConditionWrapper(
            DiT(**_tiny_dit_args((16, 24), "natten2d_rope"))
        ).eval()
    else:
        network = _tiny_distilled((16, 24), "natten2d_rope")
    dx = CorrDiffEra5Hrrr(
        **(
            model_args
            | dict(
                network=network,
                lat_output_grid=lat + 0.05 * lon.sin(),
                lon_output_grid=lon,
                hrrr_y=torch.from_numpy(resolve_grid("hrrr").y[500:516].copy()),
                hrrr_x=torch.from_numpy(resolve_grid("hrrr").x[800:824].copy()),
                invariants=torch.randn(3, 16, 24),
                number_of_steps=2,
                number_of_samples=1,
            )
        )
    ).to(device)
    assert dx._patch_size == (2, 2) and dx._min_domain_cells == 6
    dit = find_dit(network)
    full_latent = (8, 12)
    rope_cos, rope_sin = dit.rope.rope_cos.clone(), dit.rope.rope_sin.clone()

    def grid_state():
        return (
            (dit.detokenizer.h_patches, dit.detokenizer.w_patches),
            dit.attn_kwargs_forward["latent_hw"],
            dit.rope._latent_hw,
            torch.equal(dit.rope.rope_cos, rope_cos)
            and torch.equal(dit.rope.rope_sin, rope_sin),
        )

    field = _era5_field(dx)
    dx.set_rng(0)
    try:
        reference = dx(field).e2s.to_torch()[0]
    except (RuntimeError, NotImplementedError) as err:  # pragma: no cover
        if device == "cpu":
            pytest.skip(f"NATTEN has no CPU kernel here: {err}")
        raise
    # a regional view runs on its block with the NATTEN latent grid rebound
    view = dx.with_region(Region.box(262.0, 27.5, 263.0, 28.3))
    i0, i1 = (lim - dx.hrrr_lat_lim[0] for lim in view.hrrr_lat_lim)
    j0, j1 = (lim - dx.hrrr_lon_lim[0] for lim in view.hrrr_lon_lim)
    assert (i1 - i0) % 2 == 0 and (j1 - j0) % 2 == 0 and min(i1 - i0, j1 - j0) >= 6
    out, out_coords = view(field).e2s.to_torch()
    assert out.shape[-2:] == (i1 - i0, j1 - j0) and torch.isfinite(out).all()
    np.testing.assert_array_equal(out_coords["hrrr_y"], dx.hrrr_y[i0:i1])
    # ... and restores the construction grid and the original RoPE tables
    assert grid_state() == (full_latent, full_latent, full_latent, True)

    # restored after a failure inside the regional run as well
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    if backbone == "teacher":
        monkeypatch.setattr(corrdiff_module, "sample", boom)
    else:
        monkeypatch.setattr(network, "features", boom)
    with pytest.raises(RuntimeError, match="boom"):
        view(field)
    monkeypatch.undo()
    assert grid_state() == (full_latent, full_latent, full_latent, True)
    # the seeded full-domain result does not depend on the regional run before it
    dx.set_rng(0)
    torch.testing.assert_close(dx(field).e2s.to_torch()[0], reference, rtol=0, atol=0)

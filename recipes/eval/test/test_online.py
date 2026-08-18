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

"""Tests for online scoring (src/online.py) and its work partitioning.

The load-bearing property is that the sufficient statistics reproduce the
offline metrics exactly, so most of these tests pin the accumulators
against ``src.metrics`` / ``earth2studio.statistics`` rather than against
hand-computed constants.  Multi-rank grouping invariance lives in
``test_multigpu.py``.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from unittest.mock import patch

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import OmegaConf
from src.metrics import ensemble_variance, mse
from src.online import (
    FieldCache,
    GroupComm,
    OnlineScorer,
    StepContext,
    available_times,
    build_spatial_weights,
    build_statistics,
    check_verification_coverage,
    finalize_stats,
    online_enabled,
    open_stats_store,
    parse_online_settings,
    retain_raw_output,
    stats_array_groups,
)
from src.work import (
    EnsembleGroup,
    WorkItem,
    build_group_work_items,
    build_work_items,
    clear_online_progress,
    ensemble_group_for_rank,
    filter_online_completed,
    online_progress_dir,
    plan_ensemble_groups,
    write_online_marker,
)

from earth2studio.statistics.weights import lat_weight

SMALL_LAT = np.linspace(90, -90, 4)
SMALL_LON = np.linspace(0, 360, 8, endpoint=False)
VARIABLES = ["t2m", "z500"]
LEAD_TIMES = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
IC_TIMES = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _single_rank_comm(ensemble_size: int) -> GroupComm:
    """A one-rank group carrying the whole ensemble, with no collectives.

    Lets the accumulators be exercised without a distributed backend: the
    reductions degrade to no-ops and the single rank is its own root.
    """
    group = EnsembleGroup(
        group_id=0,
        n_groups=1,
        group_rank=0,
        group_size=1,
        members_per_rank=ensemble_size,
        ranks=(0,),
        member_ids=tuple(range(ensemble_size)),
    )
    return GroupComm(group, None)


def _base_cfg(tmp_path, ensemble_size: int = 1, mode: str = "online"):
    return OmegaConf.create(
        {
            "project": "test_online",
            "run_id": "unit",
            "start_times": [str(t) for t in IC_TIMES],
            "nsteps": 2,
            "ensemble_size": ensemble_size,
            "random_seed": 42,
            "resume": False,
            "output": {
                "path": str(tmp_path / "outputs"),
                "variables": list(VARIABLES),
                "overwrite": True,
                "retain": "none",
                "thread_writers": 0,
                "chunks": {"time": 1, "lead_time": 1},
            },
            "scoring": {
                "mode": mode,
                "variables": list(VARIABLES),
                "lat_weights": True,
                "online": {
                    "ensemble_group_size": None,
                    "members_per_rank": 1,
                    "stats_store": "stats.zarr",
                    "moment_comm_dtype": "float64",
                    "verification_cache_size": 4,
                    "validate_coords": True,
                    "nccl_timeout_s": 60,
                    "climatology": None,
                },
                "output": {"store_name": "scores.zarr"},
            },
        }
    )


def _settings(tmp_path, **overrides):
    """Parse online settings from the base cfg with ``scoring.online`` tweaks."""
    cfg = _base_cfg(tmp_path)
    for key, value in overrides.items():
        cfg.scoring.online[key] = value
    return parse_online_settings(cfg)


class _FixedSource:
    """Deterministic gridded source — same values for a given (time, variable).

    ``Random`` re-draws on every call, which would make an unbatched and a
    batched rollout start from different initial conditions and defeat the
    parity checks below.
    """

    def __init__(self, variables=VARIABLES):
        self._variables = list(variables)

    def __call__(self, time, variable) -> xr.DataArray:
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = [str(v) for v in np.atleast_1d(variable)]
        data = np.empty(
            (len(times), len(variables), len(SMALL_LAT), len(SMALL_LON)),
            dtype="float32",
        )
        for i, t in enumerate(times):
            for j, v in enumerate(variables):
                seed = (int(t.astype("int64")) // 10**9) + hash(v) % 9973
                data[i, j] = np.random.default_rng(seed).standard_normal(
                    (len(SMALL_LAT), len(SMALL_LON))
                )
        return xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": times,
                "variable": variables,
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            },
        )


def _fake_dist():
    class _FakeDist:
        rank = 0
        world_size = 1
        distributed = False

    return _FakeDist()


def _weights_2d() -> torch.Tensor:
    """The (lat, lon) weight tensor the offline metrics are given."""
    w = lat_weight(torch.tensor(SMALL_LAT, dtype=torch.float32))
    return torch.ones(len(SMALL_LAT), len(SMALL_LON)) * w.reshape(-1, 1)


def _verification_zarr(path, times, seed: int = 7) -> str:
    rng = np.random.default_rng(seed)
    ds = xr.Dataset()
    for var in VARIABLES:
        ds[var] = xr.DataArray(
            rng.standard_normal(
                (len(times), len(SMALL_LAT), len(SMALL_LON))
            ).astype("float32"),
            dims=["time", "lat", "lon"],
            coords={"time": times, "lat": SMALL_LAT, "lon": SMALL_LON},
        )
    ds.to_zarr(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Ensemble group planning
# ---------------------------------------------------------------------------


class TestEnsembleGroups:
    def test_default_one_member_per_rank(self):
        groups = plan_ensemble_groups(world_size=8, ensemble_size=4)
        assert groups == [(0, 1, 2, 3), (4, 5, 6, 7)]

    def test_leftover_ranks_idle(self):
        groups = plan_ensemble_groups(world_size=7, ensemble_size=4)
        assert groups == [(0, 1, 2, 3)]
        assert ensemble_group_for_rank(6, groups) is None

    def test_deterministic_case_degenerates(self):
        """M = 1 gives one single-rank group per rank — today's per-IC path."""
        groups = plan_ensemble_groups(world_size=4, ensemble_size=1)
        assert groups == [(0,), (1,), (2,), (3,)]
        group = ensemble_group_for_rank(2, groups)
        assert group.group_id == 2
        assert group.n_groups == 4
        assert group.is_root
        assert group.member_ids == (0,)

    def test_member_assignment(self):
        groups = plan_ensemble_groups(
            world_size=4, ensemble_size=8, members_per_rank=2
        )
        assert groups == [(0, 1, 2, 3)]
        group = ensemble_group_for_rank(2, groups, members_per_rank=2)
        assert group.member_ids == (4, 5)
        assert group.ensemble_size == 8
        assert group.root_rank == 0
        assert not group.is_root

    def test_explicit_group_size_must_factor_ensemble(self):
        with pytest.raises(ValueError, match="exact factorization"):
            plan_ensemble_groups(world_size=8, ensemble_size=8, group_size=3)

    def test_world_too_small(self):
        with pytest.raises(ValueError, match="smaller than the ensemble group size"):
            plan_ensemble_groups(world_size=2, ensemble_size=8)

    def test_group_work_items_match_offline_seeds(self):
        """Online work items reproduce the offline per-member seeds exactly."""
        cfg = OmegaConf.create(
            {
                "start_times": [str(t) for t in IC_TIMES],
                "ensemble_size": 4,
                "random_seed": 42,
            }
        )
        offline = {(i.time, i.ensemble_id): i.seed for i in build_work_items(cfg)}

        groups = plan_ensemble_groups(world_size=4, ensemble_size=4)
        for rank in range(4):
            group = ensemble_group_for_rank(rank, groups)
            items = build_group_work_items(list(IC_TIMES), group, cfg)
            assert len(items) == len(IC_TIMES)
            for item in items:
                assert item.seed == offline[(item.time, item.ensemble_id)]


class TestOnlineMarkers:
    def test_marker_roundtrip(self, tmp_path):
        cfg = _base_cfg(tmp_path)
        times = list(IC_TIMES)
        assert filter_online_completed(times, cfg) == times

        write_online_marker(times[0], cfg)
        assert filter_online_completed(times, cfg) == [times[1]]

        clear_online_progress(cfg)
        assert not online_progress_dir(cfg).exists()
        assert filter_online_completed(times, cfg) == times

    def test_namespace_is_distinct_from_offline(self, tmp_path):
        cfg = _base_cfg(tmp_path)
        assert online_progress_dir(cfg).name == ".online_progress"

    def test_clear_is_safe_from_concurrent_ranks(self, tmp_path):
        """Every rank clears on a fresh run, so they race on the same tree.

        Whichever loses sees entries vanish mid-walk; the loser used to
        surface that as FileNotFoundError and abort the whole job.
        """
        from concurrent.futures import ThreadPoolExecutor

        cfg = _base_cfg(tmp_path)
        times = [
            np.datetime64("2024-01-01", "ns") + np.timedelta64(i, "h")
            for i in range(200)
        ]
        for t in times:
            write_online_marker(t, cfg)
        assert len(list(online_progress_dir(cfg).iterdir())) == len(times)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(clear_online_progress, cfg) for _ in range(4)]
            for future in futures:
                future.result()

        assert not online_progress_dir(cfg).exists()

    def test_clear_on_missing_directory_is_a_noop(self, tmp_path):
        clear_online_progress(_base_cfg(tmp_path))


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------


class TestConfig:
    @pytest.mark.parametrize(
        "mode,expected",
        [("offline", False), ("online", True)],
    )
    def test_online_enabled(self, tmp_path, mode, expected):
        assert online_enabled(_base_cfg(tmp_path, mode=mode)) is expected

    def test_retain_none_requires_online(self, tmp_path):
        cfg = _base_cfg(tmp_path, mode="offline")
        with pytest.raises(ValueError, match="scoring.mode=online"):
            retain_raw_output(cfg)

    def test_retain_sample_not_implemented(self, tmp_path):
        cfg = _base_cfg(tmp_path)
        cfg.output.retain = "sample"
        with pytest.raises(NotImplementedError, match="not implemented"):
            retain_raw_output(cfg)

    def test_retain_all(self, tmp_path):
        cfg = _base_cfg(tmp_path)
        cfg.output.retain = "all"
        assert retain_raw_output(cfg) is True

    def test_settings_defaults(self, tmp_path):
        settings = parse_online_settings(_base_cfg(tmp_path))
        assert settings.ensemble_group_size is None
        assert settings.members_per_rank == 1
        assert settings.moment_comm_dtype is torch.float64
        assert settings.stats_store == "stats.zarr"
        assert settings.crps is True
        assert settings.pairwise_comm_dtype is torch.bfloat16
        assert settings.defer_pairwise_one_step is True

    def test_removed_settings_are_rejected_not_ignored(self, tmp_path):
        """A stale config should fail loudly rather than change meaning.

        ``pairwise_exchange=ring`` silently ignored would leave a run that
        asked for the low-memory path taking the all-gather instead, and
        ``mode=both`` silently ignored would fall through to offline and
        write the raw store the campaign was sized to avoid.
        """
        with pytest.raises(ValueError, match="pairwise_exchange"):
            _settings(tmp_path, pairwise_exchange="ring")

        cfg = _base_cfg(tmp_path)
        cfg.scoring.mode = "both"
        with pytest.raises(ValueError, match="both"):
            online_enabled(cfg)

    def test_invalid_mode(self, tmp_path):
        cfg = _base_cfg(tmp_path)
        cfg.scoring.mode = "onlien"
        with pytest.raises(ValueError, match="scoring.mode"):
            online_enabled(cfg)

    def test_invalid_dtypes(self, tmp_path):
        with pytest.raises(ValueError, match="pairwise_comm_dtype"):
            _settings(tmp_path, pairwise_comm_dtype="int8")
        with pytest.raises(ValueError, match="moment_comm_dtype"):
            _settings(tmp_path, moment_comm_dtype="bfloat16")

    def test_members_per_rank_must_be_positive(self, tmp_path):
        with pytest.raises(ValueError, match="members_per_rank"):
            _settings(tmp_path, members_per_rank=0)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


class TestSpatialWeights:
    def test_lat_weights_broadcast_shape(self):
        coords = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        w = build_spatial_weights(coords, lat_weights=True)
        assert w.shape == (len(SMALL_LAT), 1)
        assert w.dtype is torch.float64

    def test_uniform_without_lat(self):
        coords = OrderedDict({"y": np.arange(3), "x": np.arange(5)})
        w = build_spatial_weights(coords, lat_weights=True)
        assert w.shape == (1, 1)
        assert torch.equal(w, torch.ones(1, 1, dtype=torch.float64))

    def test_uniform_when_disabled(self):
        coords = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        w = build_spatial_weights(coords, lat_weights=False)
        assert torch.equal(w, torch.ones(1, 1, dtype=torch.float64))

    def test_matches_offline_weight_sum(self):
        coords = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        w = build_spatial_weights(coords, lat_weights=True)
        total = float(w.expand(len(SMALL_LAT), len(SMALL_LON)).sum())
        assert total == pytest.approx(float(_weights_2d().sum()), rel=1e-6)


# ---------------------------------------------------------------------------
# Accumulator math
# ---------------------------------------------------------------------------


def _run_statistics(
    f: torch.Tensor,
    y: torch.Tensor,
    ensemble_size: int,
    valid: torch.Tensor | None = None,
):
    """Drive one lead step through the statistics and return their state.

    *valid* defaults to an all-True mask (nothing excluded) — pass a
    ``[variable, lat, lon]`` boolean tensor to exercise the mask-aware
    accumulation path (see ``TestMaskAwareAccumulation``).
    """
    comm = _single_rank_comm(ensemble_size)
    weights = build_spatial_weights(
        OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}), lat_weights=True
    )
    w_sum = float(weights.expand(len(SMALL_LAT), len(SMALL_LON)).sum())
    n_variables = y.shape[0]
    if valid is None:
        valid = torch.ones_like(y, dtype=torch.bool)

    statistics = build_statistics(ensemble_size, has_climatology=False)
    for stat in statistics:
        stat.reset(1, n_variables, ensemble_size, torch.device("cpu"))

    ctx = StepContext(
        lead_index=0,
        valid_time=np.datetime64("2024-01-01T06", "ns"),
        y=y,
        clim=None,
        f_local=f,
        d_local=f - y,
        weights=weights,
        valid=valid,
        n_spatial=2,
        ensemble_size=ensemble_size,
        member_ids=tuple(range(ensemble_size)),
        comm=comm,
    )
    # Mirror OnlineScorer._materialize for the products these stats need.
    d = ctx.d_local.double()
    ctx.s1 = d.sum(dim=0)
    ctx.s2 = (d**2).sum(dim=0)
    ctx.below = (ctx.f_local < ctx.y).sum(dim=0, dtype=torch.int32)

    for stat in statistics:
        stat.update(ctx)

    state: dict = {}
    for stat in statistics:
        state.update(stat.state())
    return state, w_sum


class TestAccumulatorMath:
    """Pin the sufficient statistics against the offline metric classes."""

    def test_member_and_ensemble_mse_match_offline(self):
        torch.manual_seed(0)
        m = 5
        f = torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        y = torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))

        state, w_sum = _run_statistics(f, y, ensemble_size=m)

        x_coords = OrderedDict(
            {
                "ensemble": np.arange(m),
                "variable": np.array(VARIABLES),
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            }
        )
        y_coords = OrderedDict(
            {
                "variable": np.array(VARIABLES),
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            }
        )

        ref_member, _ = mse(
            reduction_dimensions=["lat", "lon"], weights=_weights_2d()
        )(f, x_coords, y, y_coords)
        got_member = (state["sse_member"][:, 0, :] / w_sum).float()
        assert torch.allclose(got_member, ref_member, atol=1e-5)

        ref_ens, _ = mse(
            reduction_dimensions=["lat", "lon"],
            weights=_weights_2d(),
            ensemble_dimension="ensemble",
        )(f, x_coords, y, y_coords)
        got_ens = (state["sse_ensmean"][0] / w_sum).float()
        assert torch.allclose(got_ens, ref_ens, atol=1e-5)

    def test_ensemble_variance_matches_offline(self):
        torch.manual_seed(1)
        m = 6
        f = torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        y = torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))

        state, w_sum = _run_statistics(f, y, ensemble_size=m)

        x_coords = OrderedDict(
            {
                "ensemble": np.arange(m),
                "variable": np.array(VARIABLES),
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            }
        )
        ref, _ = ensemble_variance(
            ensemble_dimension="ensemble",
            reduction_dimensions=["lat", "lon"],
            weights=_weights_2d(),
        )(f, x_coords)
        got = (state["var_ens"][0] / w_sum).float()
        assert torch.allclose(got, ref, atol=1e-4)

    def test_deterministic_case_has_no_ensemble_statistics(self):
        torch.manual_seed(2)
        f = torch.randn(1, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        y = torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))

        state, w_sum = _run_statistics(f, y, ensemble_size=1)
        assert "sse_member" not in state
        assert "var_ens" not in state
        assert "rank_counts" not in state

        # The ensemble-mean sum *is* the deterministic squared error.
        expected = (
            (f[0] - y).double() ** 2
            * build_spatial_weights(
                OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}), True
            )
        ).sum(dim=(1, 2))
        assert torch.allclose(state["sse_ensmean"][0], expected, atol=1e-8)

    def test_rank_histogram_bins_and_total_weight(self):
        torch.manual_seed(3)
        m = 4
        f = torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        y = torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))

        state, w_sum = _run_statistics(f, y, ensemble_size=m)
        counts = state["rank_counts"][:, 0, :]
        assert counts.shape == (m + 1, len(VARIABLES))
        # Every gridpoint lands in exactly one bin, weighted by its area.
        assert torch.allclose(
            counts.sum(dim=0), torch.full((len(VARIABLES),), w_sum, dtype=torch.float64)
        )

    def test_rank_histogram_is_flat_for_exchangeable_samples(self):
        """y drawn from the same distribution as the members ranks uniformly."""
        torch.manual_seed(4)
        m = 3
        n_bins = m + 1
        totals = torch.zeros(n_bins, dtype=torch.float64)
        for _ in range(200):
            sample = torch.randn(m + 1, 1, len(SMALL_LAT), len(SMALL_LON))
            state, w_sum = _run_statistics(sample[:m], sample[m], ensemble_size=m)
            totals += state["rank_counts"][:, 0, 0]
        p = totals / totals.sum()
        assert torch.allclose(
            p, torch.full((n_bins,), 1.0 / n_bins, dtype=torch.float64), atol=0.02
        )

    def test_moments_recover_bias_and_correlation(self):
        torch.manual_seed(5)
        m = 3
        f = torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        y = torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))

        state, w_sum = _run_statistics(f, y, ensemble_size=m)
        w = build_spatial_weights(
            OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}), True
        )
        fbar = f.double().mean(dim=0)

        bias = (state["sum_wf"][0] - state["sum_wy"][0]) / w_sum
        expected_bias = ((fbar - y.double()) * w).sum(dim=(1, 2)) / w_sum
        assert torch.allclose(bias, expected_bias, atol=1e-9)

        # Centered Pearson correlation of the ensemble mean against truth.
        cov = state["sum_wfy"][0] - state["sum_wf"][0] * state["sum_wy"][0] / w_sum
        var_f = state["sum_wf2"][0] - state["sum_wf"][0] ** 2 / w_sum
        var_y = state["sum_wy2"][0] - state["sum_wy"][0] ** 2 / w_sum
        corr = cov / torch.sqrt(var_f * var_y)
        assert torch.all(corr <= 1.0 + 1e-9)
        assert torch.all(corr >= -1.0 - 1e-9)


# ---------------------------------------------------------------------------
# Mask-aware accumulation: StepContext.valid must exclude gridpoints a
# pipeline like DLESyMPipeline NaN-masks (ocean variables outside their
# validity cadence) from every weighted sum, rather than either
# propagating NaN through the whole field or, under nan_policy=zero_fill,
# scoring a spurious zero-valued forecast.
# ---------------------------------------------------------------------------


class TestMaskAwareAccumulation:
    """Pin StepContext.valid to actually excluding masked gridpoints,
    not just leaving them to propagate as NaN or landing in rank_counts
    bin 0 (below == False for a NaN comparison, regardless of policy).

    Variable 0 is masked over the right half of the longitude axis in
    every test; variable 1 is left untouched as a control that must come
    out identical to the unmasked case.
    """

    HALF = len(SMALL_LON) // 2

    def _mask_variable_zero(
        self, f_clean: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (f_masked, valid) with variable 0 NaN'd over the right
        half of lon.  The NaN — rather than some other placeholder — is
        deliberate: it's what a pipeline sees under nan_policy=propagate,
        and the accumulators must not let it leak into a sum regardless."""
        valid = torch.ones(
            f_clean.shape[-3], len(SMALL_LAT), len(SMALL_LON), dtype=torch.bool
        )
        valid[0, :, self.HALF :] = False
        f_masked = f_clean.clone()
        f_masked[..., 0, :, self.HALF :] = float("nan")
        return f_masked, valid

    def test_wsum_excludes_masked_gridpoints(self):
        torch.manual_seed(20)
        m = 3
        f_clean = torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        y = torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        f_masked, valid = self._mask_variable_zero(f_clean)

        state, w_sum = _run_statistics(f_masked, y, ensemble_size=m, valid=valid)

        w = build_spatial_weights(
            OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}), True
        ).expand(len(SMALL_LAT), len(SMALL_LON))
        mask = valid[0].double()
        fbar = f_clean.double().mean(dim=0)

        expected_wsum = (w * mask).sum()
        expected_sse = ((fbar[0] - y[0].double()) ** 2 * w * mask).sum()

        assert state["w_sum"][0, 0].item() == pytest.approx(
            expected_wsum.item(), rel=1e-9
        )
        assert state["sse_ensmean"][0, 0].item() == pytest.approx(
            expected_sse.item(), rel=1e-7
        )
        assert not torch.isnan(state["w_sum"]).any()
        assert not torch.isnan(state["sse_ensmean"]).any()

        # The untouched variable keeps the full-grid weight and matches
        # the plain (unmasked) computation exactly.
        assert state["w_sum"][0, 1].item() == pytest.approx(w_sum, rel=1e-9)

    def test_fully_masked_variable_zeros_the_raw_sums(self):
        """A wholly invalid (lead, variable) leaves w_sum and every
        numerator at exactly 0 — finalize_stats' 0/0 division is what
        turns that into the store's NaN, not a raw NaN leaking out of
        accumulation itself."""
        torch.manual_seed(21)
        m = 3
        f = torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        f[:, 0] = float("nan")
        y = torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        valid = torch.ones(
            len(VARIABLES), len(SMALL_LAT), len(SMALL_LON), dtype=torch.bool
        )
        valid[0] = False

        state, _ = _run_statistics(f, y, ensemble_size=m, valid=valid)

        assert state["w_sum"][0, 0].item() == 0.0
        assert state["sse_ensmean"][0, 0].item() == 0.0
        assert not torch.isnan(state["w_sum"]).any()
        assert not torch.isnan(state["sse_ensmean"]).any()
        assert state["w_sum"][0, 1].item() > 0.0

    def test_rank_histogram_excludes_masked_gridpoints(self):
        """Regression test for the exact bug described in the design doc:
        NaN < y is always False, so a masked member used to land in
        rank_counts bin 0 as if it were a genuine low-value forecast."""
        torch.manual_seed(22)
        m = 4
        f_clean = torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        y = torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        f_masked, valid = self._mask_variable_zero(f_clean)

        state, w_sum = _run_statistics(f_masked, y, ensemble_size=m, valid=valid)
        counts = state["rank_counts"][:, 0, :]

        w = build_spatial_weights(
            OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}), True
        ).expand(len(SMALL_LAT), len(SMALL_LON))
        expected_masked_w = float((w * valid[0].double()).sum())

        # Masked variable's total counted weight is the *valid* subgrid's
        # weight only — not the full grid, and critically not inflated by
        # masked points defaulting into bin 0.
        assert counts[:, 0].sum().item() == pytest.approx(expected_masked_w, rel=1e-9)
        assert counts[0, 0].item() < expected_masked_w, (
            "bin 0 alone should not hold every masked gridpoint's weight"
        )
        # The untouched variable is bin-for-bin identical to the unmasked
        # baseline.
        assert counts[:, 1].sum().item() == pytest.approx(w_sum, rel=1e-9)

    def test_crps_terms_match_a_grid_trimmed_to_the_valid_columns(self, tmp_path):
        """The strongest check: masking the right half of lon for every
        lead must reproduce, term for term, running the same statistic on
        a grid that never had those columns.  Weighting here is lat-only,
        so trimming lon columns changes nothing about the per-gridpoint
        weight on what remains — an equivalence that doesn't require
        re-deriving CRPS's fair-estimator formula by hand.  Exercises both
        crps_t1 (via wsum) and crps_t2 (via PairwiseExchange._submit's
        masking) together.
        """
        torch.manual_seed(23)
        m, n_leads = 4, 2
        half = len(SMALL_LON) // 2
        var = VARIABLES[:1]

        full_members = [
            torch.randn(m, 1, len(SMALL_LAT), len(SMALL_LON)) for _ in range(n_leads)
        ]
        full_truths = [
            torch.randn(1, len(SMALL_LAT), len(SMALL_LON)) for _ in range(n_leads)
        ]

        masked_members, masked_truths, valids = [], [], []
        trim_members, trim_truths = [], []
        for f, y in zip(full_members, full_truths):
            valid = torch.zeros(1, len(SMALL_LAT), len(SMALL_LON), dtype=torch.bool)
            valid[:, :, :half] = True
            valids.append(valid)

            f_masked = f.clone()
            f_masked[:, :, :, half:] = float("nan")
            masked_members.append(f_masked)
            masked_truths.append(y)

            trim_members.append(f[:, :, :, :half].clone())
            trim_truths.append(y[:, :, :half].clone())

        settings = _settings(tmp_path, crps=True, pairwise_comm_dtype="float64")
        got_masked = _run_crps(
            settings, masked_members, masked_truths, variables=var, valid=valids
        )
        got_trimmed = _run_crps(settings, trim_members, trim_truths, variables=var)

        assert not torch.isnan(got_masked).any()
        assert torch.allclose(got_masked, got_trimmed, atol=1e-9)


# ---------------------------------------------------------------------------
# Fair CRPS (phase 3)
# ---------------------------------------------------------------------------


def _reference_fair_crps(f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fair CRPS from earth2studio, spatially averaged with lat weights."""
    from earth2studio.statistics import crps as e2s_crps

    x_coords = OrderedDict(
        {
            "ensemble": np.arange(f.shape[0]),
            "variable": np.array(VARIABLES[: f.shape[1]]),
            "lat": SMALL_LAT,
            "lon": SMALL_LON,
        }
    )
    y_coords = OrderedDict(
        {
            "variable": np.array(VARIABLES[: y.shape[0]]),
            "lat": SMALL_LAT,
            "lon": SMALL_LON,
        }
    )
    metric = e2s_crps(
        ensemble_dimension="ensemble",
        reduction_dimensions=["lat", "lon"],
        weights=_weights_2d(),
        fair=True,
    )
    value, _ = metric(f, x_coords, y, y_coords)
    return value


def _run_crps(
    settings,
    members: list[torch.Tensor],
    truths: list[torch.Tensor],
    variables: list[str] | None = None,
    valid: list[torch.Tensor] | None = None,
):
    """Drive FairCRPS over a sequence of lead steps on a single-rank group.

    Returns the finalized ``[lead, variable]`` CRPS, so the caller checks
    what ``finalize_stats`` would write rather than the raw terms — which
    is where a deferral-bookkeeping bug would show up.

    *variables* narrows the variable list ``FairCRPS`` sees (default: the
    full ``VARIABLES``).  *valid* is a per-lead list of ``[variable, lat,
    lon]`` boolean masks (default: all-True — nothing excluded); the
    weighted-mean normalizer is derived mask-aware from the first lead's
    context, matching ``WeightSum``'s per-variable ``w_sum`` rather than a
    single scalar, so a caller that passes a mask gets the same
    normalization the real store would.
    """
    from src.online import FairCRPS

    m = members[0].shape[0]
    comm = _single_rank_comm(m)
    weights = build_spatial_weights(
        OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}), lat_weights=True
    )
    n_leads = len(members)
    all_variables = list(variables) if variables is not None else list(VARIABLES)
    n_variables = truths[0].shape[0]

    stat = FairCRPS(settings, all_variables)
    stat.reset(n_leads, n_variables, m, torch.device("cpu"))

    contexts = []
    for lead, (f, y) in enumerate(zip(members, truths)):
        v = valid[lead] if valid is not None else torch.ones_like(y, dtype=torch.bool)
        ctx = StepContext(
            lead_index=lead,
            valid_time=np.datetime64("2024-01-01T00", "ns"),
            y=y,
            clim=None,
            f_local=f,
            d_local=f - y,
            weights=weights,
            valid=v,
            n_spatial=2,
            ensemble_size=m,
            member_ids=tuple(range(m)),
            comm=comm,
        )
        contexts.append(ctx)
        stat.update(ctx)
    stat.flush(contexts[-1])

    # Assumes a constant mask across leads (true for every caller here) —
    # the first context's mask-aware normalizer applies to all of them.
    w_sum = contexts[0].wsum_valid()
    state = stat.state()
    return state["crps_t1"] / (m * w_sum) - state["crps_t2"] / (
        m * (m - 1) * w_sum
    )


class TestFairCRPS:
    """Pin the two accumulated terms against earth2studio's fair CRPS."""

    @pytest.mark.parametrize("defer", [False, True])
    def test_matches_earth2studio(self, tmp_path, defer):
        torch.manual_seed(7)
        m, n_leads = 5, 3
        members = [
            torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
            for _ in range(n_leads)
        ]
        truths = [
            torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
            for _ in range(n_leads)
        ]
        settings = _settings(
            tmp_path,
            defer_pairwise_one_step=defer,
            pairwise_comm_dtype="float64",
        )

        got = _run_crps(settings, members, truths)
        assert not torch.isnan(got).any(), "a lead step was left unfilled"
        for lead, (f, y) in enumerate(zip(members, truths)):
            expected = _reference_fair_crps(f, y)
            assert torch.allclose(got[lead].float(), expected, atol=1e-5), (
                f"lead {lead}: {got[lead].tolist()} != {expected.tolist()}"
            )

    def test_deferral_places_terms_at_the_right_lead(self, tmp_path):
        """Distinct per-lead ensembles catch an off-by-one in the deferral."""
        torch.manual_seed(8)
        m = 4
        # Scale each lead differently so a shifted term is unmissable.
        members = [
            torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON)) * s
            for s in (1.0, 10.0, 100.0)
        ]
        truths = [
            torch.zeros(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
            for _ in members
        ]
        deferred = _run_crps(
            _settings(
                tmp_path,
                defer_pairwise_one_step=True,
                pairwise_comm_dtype="float64",
            ),
            members,
            truths,
        )
        immediate = _run_crps(
            _settings(
                tmp_path,
                defer_pairwise_one_step=False,
                pairwise_comm_dtype="float64",
            ),
            members,
            truths,
        )
        assert torch.allclose(deferred, immediate, atol=1e-9)

    def test_bfloat16_exchange_stays_close(self, tmp_path):
        """The wire dtype must not move CRPS beyond sampling noise."""
        torch.manual_seed(9)
        m = 6
        members = [torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))]
        truths = [torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))]

        exact = _run_crps(
            _settings(tmp_path, pairwise_comm_dtype="float64"), members, truths
        )
        narrow = _run_crps(
            _settings(tmp_path, pairwise_comm_dtype="bfloat16"), members, truths
        )
        assert torch.allclose(narrow, exact, rtol=2e-2)

    def test_variable_chunking_is_transparent(self, tmp_path):
        torch.manual_seed(10)
        m = 4
        members = [torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))]
        truths = [torch.randn(len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))]

        whole = _run_crps(
            _settings(tmp_path, variable_chunk=0, pairwise_comm_dtype="float64"),
            members,
            truths,
        )
        chunked = _run_crps(
            _settings(tmp_path, variable_chunk=1, pairwise_comm_dtype="float64"),
            members,
            truths,
        )
        assert torch.allclose(whole, chunked, atol=1e-12)

    def test_pairwise_variables_narrows_coverage(self, tmp_path):
        from src.online import FairCRPS

        settings = _settings(tmp_path, pairwise_variables=["z500"])
        stat = FairCRPS(settings, list(VARIABLES))
        assert stat.variables() == ["z500"]

        stats = build_statistics(4, False, settings, VARIABLES)
        _, groups = stats_array_groups(
            stats, VARIABLES, IC_TIMES, LEAD_TIMES, ensemble_size=4
        )
        scalar = next(n for c, n in groups if tuple(c) == ("time", "lead_time"))
        assert "crps_t1__z500" in scalar
        assert "crps_t1__t2m" not in scalar
        # The cheap moment statistics still cover every variable.
        assert "sse_ensmean__t2m" in scalar

    def test_unknown_pairwise_variable_rejected(self, tmp_path):
        from src.online import FairCRPS

        settings = _settings(tmp_path, pairwise_variables=["nope"])
        with pytest.raises(ValueError, match="pairwise_variables"):
            FairCRPS(settings, list(VARIABLES))

    def test_crps_disabled_and_deterministic_runs_skip_it(self, tmp_path):
        settings = _settings(tmp_path, crps=False)
        fields = {f for s in build_statistics(4, False, settings, VARIABLES)
                  for f in s.fields()}
        assert "crps_t1" not in fields

        enabled = _settings(tmp_path)
        fields = {f for s in build_statistics(1, False, enabled, VARIABLES)
                  for f in s.fields()}
        assert "crps_t1" not in fields, "CRPS is undefined for a single member"


# ---------------------------------------------------------------------------
# Store schema
# ---------------------------------------------------------------------------


class TestStatsSchema:
    def test_deterministic_schema_has_no_ensemble_axis(self):
        stats = build_statistics(1, has_climatology=False)
        superset, groups = stats_array_groups(
            stats, VARIABLES, IC_TIMES, LEAD_TIMES, ensemble_size=1
        )
        assert list(superset) == ["time", "lead_time"]
        assert len(groups) == 1
        coords, names = groups[0]
        assert list(coords) == ["time", "lead_time"]
        assert "w_sum__t2m" in names
        assert "sse_ensmean__z500" in names

    def test_ensemble_schema_groups_by_layout(self):
        stats = build_statistics(4, has_climatology=False)
        superset, groups = stats_array_groups(
            stats, VARIABLES, IC_TIMES, LEAD_TIMES, ensemble_size=4
        )
        assert list(superset) == ["time", "ensemble", "rank_bin", "lead_time"]
        by_dims = {tuple(c): n for c, n in groups}
        assert "sse_member__t2m" in by_dims[("time", "ensemble", "lead_time")]
        assert "rank_counts__t2m" in by_dims[("time", "rank_bin", "lead_time")]
        assert "var_ens__t2m" in by_dims[("time", "lead_time")]
        assert len(by_dims[("time", "rank_bin", "lead_time")]) == len(VARIABLES)

    def test_climatology_switches_moment_field_names(self):
        anom = build_statistics(1, has_climatology=True)
        raw = build_statistics(1, has_climatology=False)
        anom_fields = {f for s in anom for f in s.fields()}
        raw_fields = {f for s in raw for f in s.fields()}
        assert "sum_wab" in anom_fields
        assert "sum_wfy" in raw_fields


# ---------------------------------------------------------------------------
# Verification access
# ---------------------------------------------------------------------------


class TestVerification:
    def test_field_cache_normalizes_and_caches(self, tmp_path):
        from src.data import PredownloadedSource

        times = np.array(
            ["2024-01-01T00", "2024-01-01T06"], dtype="datetime64[ns]"
        )
        path = _verification_zarr(tmp_path / "verif.zarr", times)
        source = PredownloadedSource(path)

        cache = FieldCache(
            source, VARIABLES, ("lat", "lon"), torch.device("cpu"), max_size=1
        )
        field = cache.get(times[0])
        assert field.shape == (len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        assert field.dtype is torch.float32
        # Same object back on a hit; a miss evicts under max_size=1.
        assert cache.get(times[0]) is field
        cache.get(times[1])
        assert cache.get(times[0]) is not field

    def test_coverage_check_passes_and_fails(self, tmp_path):
        from src.data import PredownloadedSource

        times = np.array(
            ["2024-01-01T00", "2024-01-01T06", "2024-01-01T12"],
            dtype="datetime64[ns]",
        )
        source = PredownloadedSource(
            _verification_zarr(tmp_path / "verif.zarr", times)
        )
        ics = [np.datetime64("2024-01-01T00", "ns")]

        check_verification_coverage(source, ics, LEAD_TIMES, VARIABLES)

        too_far = np.array([0, 6, 12, 18], dtype="timedelta64[h]").astype(
            "timedelta64[ns]"
        )
        with pytest.raises(ValueError, match="missing 1/4 valid times"):
            check_verification_coverage(source, ics, too_far, VARIABLES)

    def test_coverage_check_tolerates_unknown_sources(self):
        class Opaque:
            def __call__(self, time, variable):  # pragma: no cover - not called
                raise AssertionError

        assert available_times(Opaque()) is None
        # Unknown coverage is a warning, not a failure.
        check_verification_coverage(
            Opaque(), [IC_TIMES[0]], LEAD_TIMES, VARIABLES
        )


# ---------------------------------------------------------------------------
# Member batching (phase 4)
# ---------------------------------------------------------------------------


class _ConstantField:
    """Stand-in FieldCache returning a constant verification field."""

    def __init__(self, value: float) -> None:
        self._value = value

    def get(self, valid_time) -> torch.Tensor:
        return torch.full(
            (len(VARIABLES), len(SMALL_LAT), len(SMALL_LON)),
            self._value,
            dtype=torch.float32,
        )


def _persistence_pipeline(device: torch.device, nsteps: int = 2):
    """A ForecastPipeline wired to Persistence without touching Hydra."""
    from earth2studio.models.px import Persistence
    from src.pipelines import ForecastPipeline

    prognostic = Persistence(
        variable=VARIABLES,
        domain_coords=OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON}),
    )
    pipeline = ForecastPipeline()
    pipeline.prognostic = prognostic.to(device)
    pipeline.diagnostics = []
    pipeline.perturbation = None
    pipeline.nsteps = nsteps
    pipeline._prognostic_ic = prognostic.input_coords()
    pipeline._spatial_ref = prognostic.output_coords(pipeline._prognostic_ic)
    pipeline._dx_input_coords = {}
    return pipeline


class TestMemberBatching:
    def test_forecast_pipeline_supports_batching(self):
        from src.pipelines import Pipeline

        assert _persistence_pipeline(torch.device("cpu")).supports_member_batching()

        class Bare(Pipeline):
            def setup(self, cfg, device):  # pragma: no cover - never called
                ...

            def build_total_coords(self, times, ensemble_size):  # pragma: no cover
                ...

            def run_item(self, item, data_source, device):  # pragma: no cover
                ...

        assert not Bare().supports_member_batching()
        with pytest.raises(NotImplementedError, match="run_item_batched"):
            list(Bare().run_item_batched([], None, torch.device("cpu")))

    def test_batched_matches_unbatched_members(self):
        """Each batched member must reproduce its own K=1 rollout."""
        device = torch.device("cpu")
        pipeline = _persistence_pipeline(device)
        source = _FixedSource()
        items = [
            WorkItem(time=IC_TIMES[0], ensemble_id=m, seed=100 + m) for m in range(3)
        ]

        batched = list(pipeline.run_item_batched(items, source, device))
        per_member = [
            list(pipeline.run_item(item, source, device)) for item in items
        ]

        assert len(batched) == len(per_member[0])
        for step, (x_batch, coords_batch) in enumerate(batched):
            assert list(coords_batch)[0] == "ensemble"
            assert list(coords_batch["ensemble"]) == [0, 1, 2]
            for m in range(len(items)):
                x_solo, _ = per_member[m][step]
                assert torch.allclose(x_batch[m], x_solo, atol=1e-6)

    def test_batch_must_share_one_initial_condition(self):
        device = torch.device("cpu")
        pipeline = _persistence_pipeline(device)
        items = [
            WorkItem(time=IC_TIMES[0], ensemble_id=0, seed=1),
            WorkItem(time=IC_TIMES[1], ensemble_id=1, seed=2),
        ]
        with pytest.raises(ValueError, match="one initial condition per batch"):
            list(pipeline.run_item_batched(items, _FixedSource(), device))

    def test_run_rejects_indivisible_member_batch(self):
        device = torch.device("cpu")
        pipeline = _persistence_pipeline(device)
        items = [
            WorkItem(time=IC_TIMES[0], ensemble_id=m, seed=m) for m in range(3)
        ]

        class _Sink:
            def begin_item(self, item):
                ...

            def update(self, x, coords):
                ...

            def finish_item(self, item):
                ...

        with pytest.raises(ValueError, match="does not divide"):
            pipeline.run(
                items,
                _FixedSource(),
                None,
                list(VARIABLES),
                device,
                scorer=_Sink(),
                member_batch=2,
            )

    def test_scorer_accepts_a_member_block(self, tmp_path):
        """A K>1 chunk must land on the right absolute member indices."""
        m = 4
        comm = _single_rank_comm(m)
        spatial = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        cfg = _base_cfg(tmp_path, ensemble_size=m)

        scorer = OnlineScorer(
            cfg=cfg,
            settings=_settings(tmp_path, crps=False),
            comm=comm,
            verification=_ConstantField(0.0),
            climatology=None,
            variables=list(VARIABLES),
            lead_times=LEAD_TIMES,
            spatial_coords=spatial,
            weights=build_spatial_weights(spatial, True),
            stats_mgr=None,
            device=torch.device("cpu"),
        )

        torch.manual_seed(11)
        block = torch.randn(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        coords = OrderedDict(
            {
                "ensemble": np.arange(m),
                "time": np.array([IC_TIMES[0]]),
                "lead_time": np.array([LEAD_TIMES[1]]),
                "variable": np.array(VARIABLES),
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            }
        )
        scorer.begin_item(WorkItem(time=IC_TIMES[0], ensemble_id=0, seed=0))
        # (member, time, lead_time, variable, lat, lon)
        scorer.update(block[:, None, None], coords)

        # sse_member for member i must equal that member's own weighted SSE.
        state: dict = {}
        for stat in scorer._statistics:
            state.update(stat.state())
        w = build_spatial_weights(spatial, True)
        expected = (block.double() ** 2 * w).sum(dim=(2, 3))
        assert torch.allclose(state["sse_member"][:, 1, :], expected, atol=1e-9)

    def test_scorer_rejects_foreign_members(self, tmp_path):
        m = 4
        spatial = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        scorer = OnlineScorer(
            cfg=_base_cfg(tmp_path, ensemble_size=m),
            settings=_settings(tmp_path, crps=False),
            comm=_single_rank_comm(m),
            verification=_ConstantField(0.0),
            climatology=None,
            variables=list(VARIABLES),
            lead_times=LEAD_TIMES,
            spatial_coords=spatial,
            weights=build_spatial_weights(spatial, True),
            stats_mgr=None,
            device=torch.device("cpu"),
        )
        block = torch.zeros(m, len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
        coords = OrderedDict(
            {
                "ensemble": np.array([9, 8, 7, 6]),
                "lead_time": np.array([LEAD_TIMES[0]]),
                "variable": np.array(VARIABLES),
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            }
        )
        scorer.begin_item(WorkItem(time=IC_TIMES[0], ensemble_id=0, seed=0))
        with pytest.raises(ValueError, match="this rank owns"):
            scorer.update(block.unsqueeze(1), coords)


# ---------------------------------------------------------------------------
# End to end: accumulate -> stats.zarr -> scores.zarr
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """A deterministic online run must match the offline scorer exactly."""

    def _run(self, tmp_path):
        from src.data import PredownloadedSource

        cfg = _base_cfg(tmp_path, ensemble_size=1)
        os.makedirs(cfg.output.path, exist_ok=True)

        valid_times = np.unique(
            np.array(
                [t + lt for t in IC_TIMES for lt in LEAD_TIMES],
                dtype="datetime64[ns]",
            )
        )
        verif_path = _verification_zarr(tmp_path / "verif.zarr", valid_times)
        source = PredownloadedSource(verif_path)

        settings = parse_online_settings(cfg)
        comm = _single_rank_comm(1)
        statistics = build_statistics(1, has_climatology=False)

        rng = np.random.default_rng(11)
        forecasts = {
            (str(t), int(lt.astype("int64"))): rng.standard_normal(
                (len(VARIABLES), len(SMALL_LAT), len(SMALL_LON))
            ).astype("float32")
            for t in IC_TIMES
            for lt in LEAD_TIMES
        }

        spatial = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        with patch(
            "src.output.DistributedManager", return_value=_fake_dist()
        ), patch("src.distributed.DistributedManager", return_value=_fake_dist()):
            stats_mgr = open_stats_store(
                cfg, statistics, VARIABLES, IC_TIMES, LEAD_TIMES, 1
            )
            with stats_mgr:
                scorer = OnlineScorer(
                    cfg=cfg,
                    settings=settings,
                    comm=comm,
                    verification=FieldCache(
                        source, VARIABLES, ("lat", "lon"), torch.device("cpu")
                    ),
                    climatology=None,
                    variables=list(VARIABLES),
                    lead_times=LEAD_TIMES,
                    spatial_coords=spatial,
                    weights=build_spatial_weights(spatial, True),
                    stats_mgr=stats_mgr,
                    device=torch.device("cpu"),
                )
                for item in build_group_work_items(
                    list(IC_TIMES), comm.group, cfg
                ):
                    scorer.begin_item(item)
                    for lt in LEAD_TIMES:
                        x = torch.from_numpy(
                            forecasts[(str(item.time), int(lt.astype("int64")))]
                        )
                        coords = OrderedDict(
                            {
                                "time": np.array([item.time]),
                                "lead_time": np.array([lt]),
                                "variable": np.array(VARIABLES),
                                "lat": SMALL_LAT,
                                "lon": SMALL_LON,
                            }
                        )
                        scorer.update(x[None, None], coords)
                    scorer.finish_item(item)

            finalize_stats(cfg)

        return cfg, forecasts, source

    def test_stats_store_written(self, tmp_path):
        cfg, _, _ = self._run(tmp_path)
        ds = xr.open_zarr(os.path.join(cfg.output.path, "stats.zarr"))
        assert "sse_ensmean__t2m" in ds
        assert "w_sum__z500" in ds
        assert ds["sse_ensmean__t2m"].dims == ("time", "lead_time")
        assert ds["sse_ensmean__t2m"].dtype == np.float64
        assert not np.isnan(ds["sse_ensmean__t2m"].values).any()

    def test_markers_written_per_ic(self, tmp_path):
        cfg, _, _ = self._run(tmp_path)
        markers = sorted(p.name for p in online_progress_dir(cfg).iterdir())
        assert len(markers) == len(IC_TIMES)
        assert filter_online_completed(list(IC_TIMES), cfg) == []

    def test_scores_match_offline_metric(self, tmp_path):
        cfg, forecasts, source = self._run(tmp_path)
        scores = xr.open_zarr(os.path.join(cfg.output.path, "scores.zarr"))

        assert "mse__t2m" in scores
        assert "ensemble_mean_mse__t2m" not in scores  # deterministic run
        assert "ensemble_variance__t2m" not in scores
        assert "bias__z500" in scores
        assert "acc__t2m" not in scores  # no climatology configured

        metric = mse(reduction_dimensions=["lat", "lon"], weights=_weights_2d())
        y_coords = OrderedDict(
            {
                "variable": np.array(VARIABLES),
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            }
        )
        for t in IC_TIMES:
            for lt in LEAD_TIMES:
                x = torch.from_numpy(forecasts[(str(t), int(lt.astype("int64")))])
                y = torch.from_numpy(
                    source([t + lt], VARIABLES)
                    .transpose("time", "variable", "lat", "lon")
                    .values[0]
                    .copy()
                )
                ref, _ = metric(x, y_coords, y, y_coords)
                for j, var in enumerate(VARIABLES):
                    got = float(
                        scores[f"mse__{var}"].sel(time=t, lead_time=lt).values
                    )
                    assert got == pytest.approx(float(ref[j]), rel=1e-5)

    def test_finalize_is_idempotent(self, tmp_path):
        cfg, _, _ = self._run(tmp_path)
        first = xr.open_zarr(os.path.join(cfg.output.path, "scores.zarr")).load()
        finalize_stats(cfg)
        second = xr.open_zarr(os.path.join(cfg.output.path, "scores.zarr")).load()
        xr.testing.assert_allclose(first, second)

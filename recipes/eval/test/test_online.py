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

"""Tests for online scoring on the library Pipeline hooks.

Everything runs on CPU in a single process with analytic fields: a
deterministic source whose value is an affine function of time, latitude,
longitude and variable, so the expected statistics follow by hand.  The
two-rank gloo test at the end launches ``torchrun`` on this file; setting
``E2S_EVAL_SKIP_MULTIRANK`` skips it.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
from omegaconf import DictConfig, open_dict

RECIPE_DIR = Path(__file__).resolve().parents[1]
if str(RECIPE_DIR) not in sys.path:  # torchrun runs this file as a script
    sys.path.insert(0, str(RECIPE_DIR))

from src.campaign import build_items, order_for_groups  # noqa: E402
from src.online import (  # noqa: E402
    EnsembleGroup,
    FieldCache,
    GroupComm,
    MultiRefScorer,
    OnlineRun,
    OnlineScorer,
    build_group_comm,
    build_statistics,
    check_verification_coverage,
    finalize_online_stats,
    finalize_stats,
    group_for_rank,
    named_store_filename,
    online_enabled,
    open_stats_store,
    order_items,
    parse_online_settings,
    plan_groups,
    stats_array_groups,
    verification_source_names,
)
from src.output import build_regridders, progress_path, stream_variables  # noqa: E402
from src.regions import build_spatial_weights  # noqa: E402
from src.runners import build_runner  # noqa: E402

from earth2studio.grids import LatLonGrid  # noqa: E402
from earth2studio.run import Pipeline, ProgressStore, WorkItem  # noqa: E402
from earth2studio.run.work import distribute_work, work_items  # noqa: E402
from test.conftest import SMALL_LAT, SMALL_LON, VARIABLES, compose_cfg  # noqa: E402

T0 = np.datetime64("2024-01-01T00:00:00", "s")
T1 = np.datetime64("2024-01-02T00:00:00", "s")
LEADS = np.array([0, 6, 12], dtype="timedelta64[h]").astype("timedelta64[ns]")
GRID = LatLonGrid(SMALL_LAT, SMALL_LON)
SPATIAL: OrderedDict = OrderedDict([("lat", SMALL_LAT), ("lon", SMALL_LON)])
N_POINTS = SMALL_LAT.size * SMALL_LON.size


# ---------------------------------------------------------------------------
# Analytic data
# ---------------------------------------------------------------------------


def analytic_field(
    time: np.datetime64,
    variables: list[str],
    offset: float = 0.0,
    rate_per_hour: float = 0.0,
) -> np.ndarray:
    """``offset + rate * hours since T0 + 10 * variable index + lat/100 + lon/1000``."""
    hours = float(
        (np.datetime64(time, "ns") - np.datetime64(T0, "ns")) / np.timedelta64(1, "h")
    )
    lat = SMALL_LAT[:, None] / 100.0
    lon = SMALL_LON[None, :] / 1000.0
    return np.stack(
        [
            offset
            + rate_per_hour * hours
            + 10.0 * (VARIABLES.index(v) if v in VARIABLES else 0)
            + lat
            + lon
            for v in variables
        ]
    ).astype(np.float32)


class FixedSource:
    """Deterministic data source on the small grid, importable by Hydra.

    Parameters
    ----------
    offset : float, optional
        Constant added to every value, by default 0
    rate_per_hour : float, optional
        Linear trend per hour since ``T0``, by default 0
    """

    def __init__(self, offset: float = 0.0, rate_per_hour: float = 0.0) -> None:
        self.offset = float(offset)
        self.rate_per_hour = float(rate_per_hour)
        self.calls: list[np.ndarray] = []

    def __call__(self, time, variable) -> xr.DataArray:
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = [str(v) for v in np.atleast_1d(np.asarray(variable))]
        self.calls.append(times)
        data = np.stack(
            [
                analytic_field(t, variables, self.offset, self.rate_per_hour)
                for t in times
            ]
        )
        return xr.DataArray(
            data,
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": times,
                "variable": variables,
                "lat": SMALL_LAT,
                "lon": SMALL_LON,
            },
        )

    async def fetch(self, time, variable) -> xr.DataArray:
        return self(time, variable)


def _single_comm(ensemble_size: int) -> GroupComm:
    """A one-rank group carrying the whole ensemble; collectives are no-ops."""
    group = EnsembleGroup(
        block=0, group_rank=0, group_size=1, members_per_rank=ensemble_size, ranks=(0,)
    )
    return GroupComm(group, None)


def _settings(tmp_path: Path, **online) -> tuple[DictConfig, object]:
    overrides = ["scoring.mode=online", "output.retain=none"] + [
        f"scoring.online.{k}={v}" for k, v in online.items()
    ]
    cfg = compose_cfg(tmp_path, overrides)
    return cfg, parse_online_settings(cfg)


def _field(
    values: np.ndarray,
    lead: np.timedelta64,
    variables: list[str],
    members: tuple[int, ...] | None,
    time: np.datetime64 = T0,
) -> xr.DataArray:
    """A pipeline-shaped field: ``(ensemble?, time, lead_time, variable, lat, lon)``."""
    data = values[None, None] if members is None else values[:, None, None]
    dims = ("time", "lead_time", "variable", "lat", "lon")
    coords = {
        "time": np.array([time], dtype="datetime64[ns]"),
        "lead_time": np.array([lead], dtype="timedelta64[ns]"),
        "variable": np.array(variables),
        "lat": SMALL_LAT,
        "lon": SMALL_LON,
    }
    if members is not None:
        dims = ("ensemble", *dims)
        coords["ensemble"] = np.array(members, dtype=np.int64)
    return xr.DataArray(data, dims=dims, coords=coords)


def _make_scorer(
    tmp_path: Path,
    cfg: DictConfig,
    settings,
    ensemble_size: int,
    source: FixedSource,
    name: str = "stats.zarr",
    lat_weights: bool = True,
) -> tuple[OnlineScorer, str]:
    comm = _single_comm(ensemble_size)
    stats = build_statistics(ensemble_size, False, settings, VARIABLES)
    region_names = None if settings.regions is None else list(settings.regions)
    groups = stats_array_groups(
        stats, VARIABLES, np.array([T0, T1]), LEADS, ensemble_size, region_names
    )
    path = str(tmp_path / "outputs" / name)
    store = open_stats_store(path, groups, region_names, resume=False, overwrite=True)
    scorer = OnlineScorer(
        settings=settings,
        comm=comm,
        verification=FieldCache(source, VARIABLES, GRID, SPATIAL, torch.device("cpu")),
        climatology=None,
        variables=VARIABLES,
        lead_times=LEADS,
        spatial_dims=("lat", "lon"),
        weights=build_spatial_weights(SPATIAL, lat_weights, regions=settings.regions),
        store=store,
        device=torch.device("cpu"),
    )
    return scorer, path


def _score_offsets(
    scorer: OnlineScorer,
    source: FixedSource,
    offsets: list[float],
    time: np.datetime64 = T0,
    streams: dict[str, list[str]] | None = None,
    nan_at: tuple | None = None,
) -> None:
    """Run one item whose members equal verification plus constant offsets."""
    members = tuple(range(len(offsets))) if len(offsets) > 1 else None
    item = WorkItem(
        time=time,
        final_lead_time=np.timedelta64(LEADS[-1], "s"),
        member_ids=members or (),
    )
    streams = streams or {"forecast": VARIABLES}
    scorer.begin_item(item)
    for lead in LEADS:
        valid = np.datetime64(time, "ns") + lead
        truth = source(valid, VARIABLES).values[0]
        for stream, variables in streams.items():
            index = [VARIABLES.index(v) for v in variables]
            truth_sub = truth[index]
            values = np.stack([truth_sub + o for o in offsets])
            if nan_at is not None:
                values[(slice(None), *nan_at)] = np.nan
            if members is None:
                values = values[0]
            scorer.update(item, stream, _field(values, lead, variables, members, time))
    scorer.finish_item(item)


def _finalize(cfg: DictConfig) -> xr.Dataset:
    return xr.open_zarr(finalize_stats(cfg), consolidated=True)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettings:
    def test_defaults(self, tmp_path) -> None:
        cfg, settings = _settings(tmp_path)
        assert online_enabled(cfg)
        assert settings.members_per_rank == 1
        assert settings.ensemble_group_size is None
        assert settings.stats_store == "stats.zarr"
        assert settings.scores_store is None
        assert settings.pairwise_comm_dtype is torch.bfloat16
        assert settings.moment_comm_dtype is torch.float64
        assert settings.crps and not settings.mae and not settings.lsd
        assert settings.defer_pairwise_one_step
        assert settings.variable_chunk == 4
        assert settings.nccl_timeout_s == 1800
        assert settings.regions is None

    def test_offline_mode_and_invalid_mode(self, tmp_path) -> None:
        assert not online_enabled(compose_cfg(tmp_path))
        with pytest.raises(ValueError, match="scoring.mode"):
            online_enabled(compose_cfg(tmp_path, ["scoring.mode=sideways"]))

    @pytest.mark.parametrize(
        "override,match",
        [
            ("scoring.online.members_per_rank=0", "members_per_rank"),
            ("scoring.online.pairwise_comm_dtype=int8", "pairwise_comm_dtype"),
            ("scoring.online.moment_comm_dtype=bfloat16", "moment_comm_dtype"),
            ("scoring.online.variable_chunk=-1", "variable_chunk"),
            ("scoring.online.pairwise_member_tile=0", "pairwise_member_tile"),
            ("+scoring.online.pairwise_exchange=ring", "removed"),
        ],
    )
    def test_invalid_settings_are_rejected(self, tmp_path, override, match) -> None:
        cfg = compose_cfg(tmp_path, ["scoring.mode=online", override])
        with pytest.raises(ValueError, match=match):
            parse_online_settings(cfg)

    def test_regions_come_from_scoring_regions(self, tmp_path) -> None:
        cfg = compose_cfg(
            tmp_path,
            [
                "scoring.mode=online",
                "scoring.regions={global: null, north: {lat: [0, 90]}}",
            ],
        )
        settings = parse_online_settings(cfg)
        assert list(settings.regions) == ["global", "north"]

    def test_lsd_needs_a_whole_grid_region(self, tmp_path) -> None:
        cfg = compose_cfg(
            tmp_path,
            [
                "scoring.mode=online",
                "scoring.online.lsd=true",
                "scoring.regions={north: {lat: [0, 90]}}",
            ],
        )
        with pytest.raises(ValueError, match="whole-grid"):
            parse_online_settings(cfg)


# ---------------------------------------------------------------------------
# Groups and ordering
# ---------------------------------------------------------------------------


class TestGroups:
    def test_plan_groups(self, tmp_path) -> None:
        _, settings = _settings(tmp_path, members_per_rank=2)
        assert plan_groups(settings, 8) == (4, 2)
        assert plan_groups(settings, 2) == (1, 2)
        with pytest.raises(ValueError, match="divide"):
            plan_groups(settings, 3)
        _, explicit = _settings(tmp_path, members_per_rank=2, ensemble_group_size=3)
        with pytest.raises(ValueError, match="ensemble_group_size"):
            plan_groups(explicit, 8)

    def test_block_layout(self) -> None:
        group = group_for_rank(5, 8, group_size=2, members_per_rank=3)
        assert (group.block, group.group_rank, group.ranks) == (2, 1, (4, 5))
        assert group.member_ids == (3, 4, 5)
        assert group.ensemble_size == 6
        assert not group.is_root and group.root_rank == 4
        assert group_for_rank(4, 8, 2, 3).is_root
        with pytest.raises(ValueError, match="multiple"):
            group_for_rank(0, 6, 4, 1)

    def test_single_process_comm_is_a_no_op(self, tmp_path) -> None:
        _, settings = _settings(tmp_path, members_per_rank=2)
        comm = build_group_comm(settings, 0, 1, 2)
        assert not comm.active and comm.is_root
        t = torch.ones(3, dtype=torch.float64)
        assert comm.reduce(t) is t
        buffers, work = comm.all_gather(t)
        assert buffers == [t] and work is None
        assert comm.all_agree(7, torch.device("cpu"))

    def test_deterministic_campaign_is_one_group(self, tmp_path) -> None:
        _, settings = _settings(tmp_path)
        assert plan_groups(settings, 1) == (1, 1)
        assert build_group_comm(settings, 0, 1, 1).group.member_ids == (0,)


class TestOrdering:
    @staticmethod
    def _items(n_times: int, members: int, batch: int) -> list[WorkItem]:
        times = [T0 + np.timedelta64(d, "D") for d in range(n_times)]
        return work_items(
            times, np.timedelta64(12, "h"), members=members, member_batch=batch
        )

    def test_fresh_run_matches_order_for_groups(self, tmp_path) -> None:
        cfg = compose_cfg(tmp_path, ["scoring.mode=online"])
        items = self._items(5, 4, 2)
        assert order_items(cfg, items, world_size=4, group_size=2) == order_for_groups(
            items, 4, 2
        )

    def test_resume_orders_pending_items_first(self, tmp_path) -> None:
        cfg = compose_cfg(tmp_path, ["scoring.mode=online", "resume=true"])
        items = self._items(4, 4, 2)
        progress = ProgressStore(progress_path(cfg))
        done_time = items[2].time
        for item in items:
            if item.time == done_time:
                progress.mark_done(item)
        progress.close()
        ordered = order_items(cfg, items, world_size=2, group_size=2)
        assert sorted(ordered, key=lambda i: (i.time, i.member_ids)) == sorted(
            items, key=lambda i: (i.time, i.member_ids)
        )
        pending = [i for i in ordered if i.time != done_time]
        assert ordered[: len(pending)] == pending
        # What the pipeline runs keeps rank 0 and rank 1 on the same times.
        slices = [distribute_work(pending, rank, 2) for rank in (0, 1)]
        assert [i.time for i in slices[0]] == [i.time for i in slices[1]]
        assert all(i.member_ids == (0, 1) for i in slices[0])
        assert all(i.member_ids == (2, 3) for i in slices[1])

    def test_resume_rejects_a_partly_done_time(self, tmp_path) -> None:
        cfg = compose_cfg(tmp_path, ["scoring.mode=online", "resume=true"])
        items = self._items(2, 4, 2)
        progress = ProgressStore(progress_path(cfg))
        progress.mark_done(items[0])
        progress.close()
        with pytest.raises(ValueError, match="some member groups"):
            order_items(cfg, items, world_size=2, group_size=2)

    def test_world_size_must_fit_groups(self, tmp_path) -> None:
        cfg = compose_cfg(tmp_path, ["scoring.mode=online"])
        with pytest.raises(ValueError, match="multiple"):
            order_items(cfg, self._items(2, 4, 2), world_size=3, group_size=2)


# ---------------------------------------------------------------------------
# Accumulation against an analytic verification field
# ---------------------------------------------------------------------------


class TestAccumulation:
    """Members ``f_i = y + a_i`` with constant offsets have closed-form scores."""

    OFFSETS = [1.0, 3.0]

    def test_ensemble_scores_match_closed_forms(self, tmp_path) -> None:
        cfg, settings = _settings(
            tmp_path, pairwise_comm_dtype="float64", members_per_rank=2
        )
        source = FixedSource(rate_per_hour=0.5)
        scorer, _ = _make_scorer(tmp_path, cfg, settings, 2, source)
        _score_offsets(scorer, source, self.OFFSETS, T0)
        _score_offsets(scorer, source, self.OFFSETS, T1)
        scorer._store.close()

        stats = xr.open_zarr(tmp_path / "outputs" / "stats.zarr", consolidated=False)
        # Cosine weights have mean one, so their sum is the point count.
        np.testing.assert_allclose(stats["w_sum__t2m"].values, N_POINTS)
        scores = _finalize(cfg)
        a, b = self.OFFSETS
        m = (a + b) / 2
        for var in VARIABLES:
            mse = scores[f"mse__{var}"]
            assert dict(mse.sizes) == {"ensemble": 2, "time": 2, "lead_time": 3}
            np.testing.assert_allclose(mse.sel(ensemble=0).values, a**2, rtol=1e-6)
            np.testing.assert_allclose(mse.sel(ensemble=1).values, b**2, rtol=1e-6)
            np.testing.assert_allclose(
                scores[f"ensemble_mean_mse__{var}"].values, m**2, rtol=1e-6
            )
            np.testing.assert_allclose(
                scores[f"ensemble_variance__{var}"].values,
                (a - m) ** 2 + (b - m) ** 2,
                rtol=1e-6,
            )
            # Fair CRPS: mean |d_i| - |a - b| / (M (M - 1)) * (number of pairs)
            np.testing.assert_allclose(
                scores[f"crps__{var}"].values, (a + b) / 2 - abs(a - b) / 2, rtol=1e-6
            )
            # Every member exceeds the truth: all weight in rank bin 0.
            np.testing.assert_allclose(
                scores[f"rank_reliability__{var}"].values,
                (1 - 1 / 3) + 2 / 3,
                rtol=1e-6,
            )
            np.testing.assert_allclose(scores[f"bias__{var}"].values, m, rtol=1e-6)
            np.testing.assert_allclose(scores[f"corr__{var}"].values, 1.0, rtol=1e-6)
            assert f"acc__{var}" not in scores

    def test_deterministic_run_has_no_ensemble_statistics(self, tmp_path) -> None:
        cfg, settings = _settings(tmp_path)
        source = FixedSource()
        scorer, _ = _make_scorer(tmp_path, cfg, settings, 1, source)
        _score_offsets(scorer, source, [2.0], T0)
        scorer._store.close()
        scores = _finalize(cfg)
        assert "ensemble" not in scores.dims
        for var in VARIABLES:
            np.testing.assert_allclose(
                scores[f"mse__{var}"].sel(time=T0).values, 4.0, rtol=1e-6
            )
            assert np.isnan(scores[f"mse__{var}"].sel(time=T1).values).all()
            for metric in (
                "crps",
                "ensemble_variance",
                "rank_reliability",
                "ensemble_mean_mse",
            ):
                assert f"{metric}__{var}" not in scores

    def test_variables_split_over_streams_accumulate_alike(self, tmp_path) -> None:
        cfg, settings = _settings(
            tmp_path, pairwise_comm_dtype="float64", members_per_rank=2
        )
        source = FixedSource(rate_per_hour=0.25)
        one, path_one = _make_scorer(
            tmp_path, cfg, settings, 2, source, name="one.zarr"
        )
        _score_offsets(one, source, self.OFFSETS)
        one._store.close()
        two, path_two = _make_scorer(
            tmp_path, cfg, settings, 2, source, name="two.zarr"
        )
        _score_offsets(
            two, source, self.OFFSETS, streams={"forecast": ["z500"], "diag": ["t2m"]}
        )
        two._store.close()
        ds_one = xr.open_zarr(path_one, consolidated=False).sel(time=T0)
        ds_two = xr.open_zarr(path_two, consolidated=False).sel(time=T0)
        for name in ds_one.data_vars:
            np.testing.assert_allclose(
                ds_one[name].values, ds_two[name].values, rtol=1e-9
            )

    @pytest.mark.parametrize("defer", [True, False])
    def test_pairwise_deferral_and_chunking_are_transparent(
        self, tmp_path, defer
    ) -> None:
        cfg, settings = _settings(
            tmp_path,
            pairwise_comm_dtype="float64",
            members_per_rank=3,
            defer_pairwise_one_step=str(defer).lower(),
            variable_chunk=1,
        )
        source = FixedSource()
        scorer, path = _make_scorer(tmp_path, cfg, settings, 3, source)
        offsets = [0.0, 1.0, 4.0]
        _score_offsets(scorer, source, offsets)
        scorer._store.close()
        ds = xr.open_zarr(path, consolidated=False).sel(time=T0)
        pairs = sum(abs(a - b) for i, a in enumerate(offsets) for b in offsets[i + 1 :])
        # Statistics accumulate in float32 inside the store.
        np.testing.assert_allclose(
            ds["crps_t2__t2m"].values, pairs * N_POINTS, rtol=1e-6
        )
        np.testing.assert_allclose(
            ds["crps_t1__z500"].values, sum(offsets) * N_POINTS, rtol=1e-6
        )

    def test_bfloat16_exchange_stays_close(self, tmp_path) -> None:
        cfg, settings = _settings(tmp_path, members_per_rank=2)
        source = FixedSource()
        scorer, path = _make_scorer(tmp_path, cfg, settings, 2, source)
        _score_offsets(scorer, source, [0.1, 0.35])
        scorer._store.close()
        ds = xr.open_zarr(path, consolidated=False).sel(time=T0)
        np.testing.assert_allclose(
            ds["crps_t2__t2m"].values, 0.25 * N_POINTS, rtol=2e-2
        )

    def test_pairwise_variables_narrow_crps(self, tmp_path) -> None:
        cfg, settings = _settings(
            tmp_path,
            members_per_rank=2,
            pairwise_comm_dtype="float64",
            pairwise_variables="[z500]",
        )
        source = FixedSource()
        scorer, path = _make_scorer(tmp_path, cfg, settings, 2, source)
        _score_offsets(scorer, source, self.OFFSETS)
        scorer._store.close()
        ds = xr.open_zarr(path, consolidated=False)
        assert "crps_t2__z500" in ds and "crps_t2__t2m" not in ds
        scores = _finalize(cfg)
        assert "crps__z500" in scores and "crps__t2m" not in scores
        assert "mse__t2m" in scores

    def test_unknown_pairwise_variable_is_rejected(self, tmp_path) -> None:
        _, settings = _settings(tmp_path, pairwise_variables="[msl]")
        with pytest.raises(ValueError, match="pairwise_variables"):
            build_statistics(2, False, settings, VARIABLES)

    def test_regions_add_a_labelled_axis(self, tmp_path) -> None:
        cfg = compose_cfg(
            tmp_path,
            [
                "scoring.mode=online",
                "scoring.lat_weights=false",
                "scoring.online.members_per_rank=2",
                "scoring.online.pairwise_comm_dtype=float64",
                "scoring.regions={global: null, north: {lat: [0, 90]}}",
            ],
        )
        settings = parse_online_settings(cfg)
        source = FixedSource()
        scorer, path = _make_scorer(
            tmp_path, cfg, settings, 2, source, lat_weights=False
        )
        _score_offsets(scorer, source, self.OFFSETS)
        scorer._store.close()
        ds = xr.open_zarr(path, consolidated=False).sel(time=T0)
        assert ds.attrs["regions"] == ["global", "north"]
        np.testing.assert_allclose(ds["w_sum__t2m"].values[0], N_POINTS)
        np.testing.assert_allclose(ds["w_sum__t2m"].values[1], N_POINTS / 2)
        scores = _finalize(cfg)
        assert list(scores["region"].values) == ["global", "north"]
        a, b = self.OFFSETS
        np.testing.assert_allclose(
            scores["mse__t2m"].sel(time=T0, region="north", ensemble=1).values, b**2
        )
        np.testing.assert_allclose(
            scores["crps__z500"].sel(time=T0).values, (a + b) / 2 - abs(a - b) / 2
        )

    def test_masked_gridpoints_leave_constant_errors_intact(self, tmp_path) -> None:
        cfg, settings = _settings(
            tmp_path, members_per_rank=2, pairwise_comm_dtype="float64"
        )
        source = FixedSource()
        scorer, path = _make_scorer(
            tmp_path, cfg, settings, 2, source, lat_weights=False
        )
        # NaN in every member at one latitude row of the first variable only.
        _score_offsets(scorer, source, self.OFFSETS, nan_at=(0, 1, slice(None)))
        scorer._store.close()
        ds = xr.open_zarr(path, consolidated=False).sel(time=T0)
        np.testing.assert_allclose(ds["w_sum__t2m"].values, N_POINTS - SMALL_LON.size)
        np.testing.assert_allclose(ds["w_sum__z500"].values, N_POINTS)
        scores = _finalize(cfg)
        a, b = self.OFFSETS
        np.testing.assert_allclose(
            scores["mse__t2m"].sel(time=T0, ensemble=1).values, b**2
        )
        np.testing.assert_allclose(
            scores["crps__t2m"].sel(time=T0).values, (a + b) / 2 - abs(a - b) / 2
        )

    def test_foreign_members_and_bad_layouts_are_rejected(self, tmp_path) -> None:
        cfg, settings = _settings(tmp_path, members_per_rank=2)
        source = FixedSource()
        scorer, _ = _make_scorer(tmp_path, cfg, settings, 2, source)
        item = WorkItem(T0, np.timedelta64(12, "h"), (0, 1))
        truth = source(T0, VARIABLES).values[0]
        scorer.begin_item(item)
        with pytest.raises(ValueError, match="carries members"):
            scorer.update(
                item,
                "forecast",
                _field(np.stack([truth, truth]), LEADS[0], VARIABLES, (2, 3)),
            )
        with pytest.raises(ValueError, match="lead_time"):
            scorer.update(
                item,
                "forecast",
                _field(np.stack([truth, truth]), LEADS[0], VARIABLES, (0, 1)).isel(
                    lead_time=0, drop=True
                ),
            )
        fresh, _ = _make_scorer(tmp_path, cfg, settings, 2, source, name="fresh.zarr")
        with pytest.raises(RuntimeError, match="begin_item"):
            fresh.update(
                item,
                "forecast",
                _field(np.stack([truth, truth]), LEADS[0], VARIABLES, (0, 1)),
            )

    def test_unscored_variables_are_ignored(self, tmp_path) -> None:
        cfg, settings = _settings(tmp_path)
        source = FixedSource()
        scorer, path = _make_scorer(tmp_path, cfg, settings, 1, source)
        item = WorkItem(T0, np.timedelta64(12, "h"))
        scorer.begin_item(item)
        extra = np.zeros((1, SMALL_LAT.size, SMALL_LON.size), dtype=np.float32)
        scorer.update(item, "diag", _field(extra, LEADS[0], ["unscored"], None))
        scorer.finish_item(item)
        scorer._store.close()
        ds = xr.open_zarr(path, consolidated=False)
        assert np.isnan(ds["w_sum__t2m"].values).all()


# ---------------------------------------------------------------------------
# Verification access and store schema
# ---------------------------------------------------------------------------


class TestVerification:
    def test_field_cache_caches_and_orders(self) -> None:
        source = FixedSource()
        cache = FieldCache(
            source, ["z500", "t2m"], GRID, SPATIAL, torch.device("cpu"), max_size=1
        )
        first = cache.get(T0)
        assert first.shape == (2, SMALL_LAT.size, SMALL_LON.size)
        np.testing.assert_allclose(first[0].numpy(), analytic_field(T0, VARIABLES)[1])
        assert cache.get(T0) is first and len(source.calls) == 1
        cache.get(T1)
        assert len(source.calls) == 2 and cache.get(T0) is not first

    def test_field_cache_crops_onto_the_scored_grid(self) -> None:
        other = OrderedDict([("lat", SMALL_LAT[:2]), ("lon", SMALL_LON)])
        cache = FieldCache(
            FixedSource(),
            VARIABLES,
            LatLonGrid(SMALL_LAT[:2], SMALL_LON),
            other,
            torch.device("cpu"),
        )
        # The library crops the source onto the smaller target grid.
        assert cache.get(T0).shape == (2, 2, SMALL_LON.size)

    def test_valid_ranges_and_zero_fill(self) -> None:
        source = FixedSource(offset=100.0)
        cache = FieldCache(
            source,
            VARIABLES,
            GRID,
            SPATIAL,
            torch.device("cpu"),
            valid_ranges={"t2m": {"max": 50.0}},
            nan_policy="zero_fill",
        )
        field = cache.get(T0)
        assert float(field[0].max()) == 50.0 and float(field[1].max()) > 50.0

    def test_coverage_check(self, tmp_path) -> None:
        class _Store:
            _da = xr.DataArray(
                np.zeros(3),
                dims=("time",),
                coords={
                    "time": np.array(
                        [T0, T0 + np.timedelta64(6, "h"), T0 + np.timedelta64(12, "h")],
                        dtype="datetime64[ns]",
                    )
                },
            )

        check_verification_coverage(_Store(), [T0], LEADS, VARIABLES)
        with pytest.raises(ValueError, match="missing"):
            check_verification_coverage(_Store(), [T0, T1], LEADS, VARIABLES)
        check_verification_coverage(
            FixedSource(), [T0, T1], LEADS, VARIABLES
        )  # unknown coverage


class TestStoreSchema:
    def test_deterministic_schema_has_no_member_axes(self, tmp_path) -> None:
        _, settings = _settings(tmp_path)
        groups = stats_array_groups(
            build_statistics(1, False, settings, VARIABLES),
            VARIABLES,
            np.array([T0]),
            LEADS,
            1,
        )
        assert len(groups) == 1
        coords, names = groups[0]
        assert list(coords) == ["time", "lead_time"]
        assert "w_sum__t2m" in names and "sse_ensmean__z500" in names
        assert not any(n.startswith("sse_member") for n in names)

    def test_ensemble_schema_groups_by_layout(self, tmp_path) -> None:
        _, settings = _settings(tmp_path, pairwise_variables="[t2m]")
        groups = stats_array_groups(
            build_statistics(2, True, settings, VARIABLES),
            VARIABLES,
            np.array([T0]),
            LEADS,
            2,
            ["global"],
        )
        layouts = {tuple(coords): names for coords, names in groups}
        assert ("time", "region", "lead_time") in layouts
        assert ("time", "region", "ensemble", "lead_time") in layouts
        assert ("time", "region", "rank_bin", "lead_time") in layouts
        scalar = layouts[("time", "region", "lead_time")]
        assert (
            "sum_wa__t2m" in scalar
            and "crps_t1__t2m" in scalar
            and "crps_t1__z500" not in scalar
        )

    def test_reopen_skips_existing_arrays_and_checks_regions(self, tmp_path) -> None:
        _, settings = _settings(tmp_path)
        groups = stats_array_groups(
            build_statistics(1, False, settings, VARIABLES),
            VARIABLES,
            np.array([T0]),
            LEADS,
            1,
            ["global"],
        )
        path = str(tmp_path / "stats.zarr")
        store = open_stats_store(path, groups, ["global"], resume=False, overwrite=True)
        store.close()
        again = open_stats_store(path, groups, ["global"], resume=True, overwrite=False)
        assert "w_sum__t2m" in again
        with pytest.raises(ValueError, match="regions"):
            open_stats_store(path, groups, ["europe"], resume=True, overwrite=False)
        with pytest.raises(FileExistsError):
            open_stats_store(path, groups, ["global"], resume=False, overwrite=False)


# ---------------------------------------------------------------------------
# Multi-reference naming
# ---------------------------------------------------------------------------


class TestMultiReference:
    def test_unset_sources_keep_plain_filenames(self, tmp_path) -> None:
        cfg = compose_cfg(tmp_path, ["scoring.mode=online"])
        assert verification_source_names(cfg) == [None]
        assert named_store_filename("stats.zarr", None) == "stats.zarr"
        assert named_store_filename("stats.zarr", "era5") == "stats__era5.zarr"
        assert named_store_filename("scores", "ifs") == "scores__ifs.zarr"

    @staticmethod
    def _with_sources(tmp_path, sources) -> DictConfig:
        cfg = compose_cfg(tmp_path, ["scoring.mode=online"])
        with open_dict(cfg):
            cfg.scoring.online.verification_sources = sources
        return cfg

    def test_named_sources_are_validated(self, tmp_path) -> None:
        cfg = self._with_sources(tmp_path, {"era5": None, "ifs-hres": None})
        assert verification_source_names(cfg) == ["era5", "ifs-hres"]
        with pytest.raises(ValueError, match="empty"):
            verification_source_names(self._with_sources(tmp_path, {}))
        with pytest.raises(ValueError, match="Invalid verification source name"):
            verification_source_names(self._with_sources(tmp_path, {"../x": None}))

    def test_fan_out_hands_every_field_to_every_reference(self, tmp_path) -> None:
        cfg, settings = _settings(
            tmp_path, members_per_rank=2, pairwise_comm_dtype="float64"
        )
        truth = FixedSource()
        shifted = FixedSource(offset=1.0)
        a, path_a = _make_scorer(
            tmp_path, cfg, settings, 2, truth, name="stats__a.zarr"
        )
        b, path_b = _make_scorer(
            tmp_path, cfg, settings, 2, shifted, name="stats__b.zarr"
        )
        _score_offsets(MultiRefScorer({"a": a, "b": b}), truth, [1.0, 3.0])
        a._store.close()
        b._store.close()
        ds_a = xr.open_zarr(path_a, consolidated=False).sel(time=T0)
        ds_b = xr.open_zarr(path_b, consolidated=False).sel(time=T0)
        np.testing.assert_allclose(ds_a["sse_member__t2m"].values[0], 1.0 * N_POINTS)
        np.testing.assert_allclose(ds_b["sse_member__t2m"].values[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(ds_b["sse_member__t2m"].values[1], 4.0 * N_POINTS)
        with pytest.raises(ValueError, match="at least one"):
            MultiRefScorer({})


# ---------------------------------------------------------------------------
# End to end through the library Pipeline
# ---------------------------------------------------------------------------


def _online_cfg(
    tmp_path: Path, members_per_rank: int, extra: list[str] | None = None
) -> DictConfig:
    cfg = compose_cfg(
        tmp_path,
        [
            "scoring.mode=online",
            "output.retain=none",
            "nsteps=2",
            "ensemble_size=2",
            f"scoring.online.members_per_rank={members_per_rank}",
            "scoring.online.pairwise_comm_dtype=float64",
            f"start_times=['{T0}', '{T1}']",
            *(extra or []),
        ],
    )
    with open_dict(cfg):
        cfg.model = {
            "_target_": "earth2studio.models.px.Persistence",
            "variable": list(VARIABLES),
            "domain_coords": "latlon-small",
        }
        cfg.ic_source = {
            "_target_": "test.test_online.FixedSource",
            "offset": 0.0,
            "rate_per_hour": 0.5,
        }
        cfg.verification_source = {
            "_target_": "test.test_online.FixedSource",
            "offset": 0.0,
            "rate_per_hour": 0.5,
        }
        cfg.output.variables = list(VARIABLES)
    return cfg


def run_online(cfg: DictConfig) -> OnlineRun:
    """The online branch of ``main.run``, without the raw forecast store."""
    settings = parse_online_settings(cfg)
    runner = build_runner(cfg)
    items = build_items(cfg, runner, member_batch=settings.members_per_rank)
    schemas = runner.output_coords(max(item.final_lead_time for item in items))
    regridders = build_regridders(cfg, schemas)
    variables = stream_variables(schemas, list(cfg.output.variables))
    run = OnlineRun.build(
        cfg, settings, runner, items, schemas, regridders, variables, "cpu"
    )
    resume = bool(cfg.get("resume", False))
    progress = ProgressStore(progress_path(cfg), overwrite=not resume)
    Pipeline(
        runner,
        output=None,
        scorer=run.scorer,
        output_variables=variables,
        regridder=regridders,
        progress=progress,
        device="cpu",
        sync=run.sync,
    ).run(run.items)
    run.finalize()
    return run


class TestEndToEnd:
    def test_persistence_ensemble_scores_its_trend(self, tmp_path) -> None:
        cfg = _online_cfg(tmp_path, members_per_rank=2)
        run = run_online(cfg)
        assert all(len(item.member_ids) == 2 for item in run.items)
        out = Path(cfg.output.path)
        assert (out / "stats.zarr").is_dir() and (out / "scores.zarr").is_dir()
        assert not (out / "forecast.zarr").exists()
        scores = xr.open_zarr(out / "scores.zarr", consolidated=True)
        assert dict(scores["mse__t2m"].sizes) == {
            "ensemble": 2,
            "time": 2,
            "lead_time": 3,
        }
        # Persistence holds the IC while the truth drifts 0.5 per hour.
        hours = LEADS.astype("timedelta64[h]").astype(float)
        for var in VARIABLES:
            mse = (
                scores[f"mse__{var}"].transpose("time", "ensemble", "lead_time").values
            )
            np.testing.assert_allclose(
                mse, np.broadcast_to((0.5 * hours) ** 2, mse.shape), atol=1e-6
            )
            np.testing.assert_allclose(
                scores[f"ensemble_variance__{var}"].values, 0.0, atol=1e-9
            )
            np.testing.assert_allclose(
                scores[f"bias__{var}"].transpose("time", "lead_time").values,
                np.broadcast_to(-0.5 * hours, (2, 3)),
                atol=1e-6,
            )
        # Deriving the scores twice gives the same store.
        finalize_online_stats(cfg)
        again = xr.open_zarr(out / "scores.zarr", consolidated=True)
        np.testing.assert_array_equal(
            again["mse__z500"].values, scores["mse__z500"].values
        )

    def test_resume_skips_finished_items_and_keeps_stats(self, tmp_path) -> None:
        cfg = _online_cfg(tmp_path, members_per_rank=2)
        run_online(cfg)
        stats_path = Path(cfg.output.path) / "stats.zarr"
        before = xr.open_zarr(stats_path, consolidated=False).load()
        with open_dict(cfg):
            cfg.resume = True
        run_online(cfg)
        after = xr.open_zarr(stats_path, consolidated=False).load()
        for name in before.data_vars:
            np.testing.assert_array_equal(before[name].values, after[name].values)

    def test_deterministic_campaign(self, tmp_path) -> None:
        cfg = _online_cfg(tmp_path, members_per_rank=1, extra=["ensemble_size=1"])
        run = run_online(cfg)
        assert all(item.member_ids == () for item in run.items)
        scores = xr.open_zarr(Path(cfg.output.path) / "scores.zarr", consolidated=True)
        assert "ensemble" not in scores.dims and "crps__t2m" not in scores
        np.testing.assert_allclose(
            scores["mse__t2m"].sel(lead_time=LEADS[1]).values, 9.0, atol=1e-6
        )

    def test_named_references_write_their_own_stores(self, tmp_path) -> None:
        cfg = _online_cfg(tmp_path, members_per_rank=2)
        with open_dict(cfg):
            cfg.scoring.online.verification_sources = {
                "truth": None,
                "shifted": {
                    "_target_": "test.test_online.FixedSource",
                    "offset": 1.0,
                    "rate_per_hour": 0.5,
                },
            }
        run_online(cfg)
        out = Path(cfg.output.path)
        for name in ("truth", "shifted"):
            assert (out / f"stats__{name}.zarr").is_dir()
            assert (out / f"scores__{name}.zarr").is_dir()
        assert not (out / "stats.zarr").exists()
        truth = xr.open_zarr(out / "scores__truth.zarr", consolidated=True)
        shifted = xr.open_zarr(out / "scores__shifted.zarr", consolidated=True)
        np.testing.assert_allclose(
            truth["mse__t2m"].sel(lead_time=LEADS[0]).values, 0.0, atol=1e-9
        )
        np.testing.assert_allclose(
            shifted["mse__t2m"].sel(lead_time=LEADS[0]).values, 1.0, atol=1e-6
        )

    def test_world_size_must_be_a_multiple_of_the_group_size(self, tmp_path) -> None:
        cfg = _online_cfg(tmp_path, members_per_rank=1)  # G = 2 on one rank
        with pytest.raises(ValueError, match="multiple"):
            run_online(cfg)

    def test_scored_variable_outside_output_is_rejected(self, tmp_path) -> None:
        cfg = _online_cfg(
            tmp_path, members_per_rank=2, extra=["scoring.variables=[t2m, msl]"]
        )
        with pytest.raises(ValueError, match="scoring.variables"):
            run_online(cfg)


# ---------------------------------------------------------------------------
# Two ranks on gloo
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.skipif(
    os.environ.get("E2S_EVAL_SKIP_MULTIRANK") == "1", reason="multi-rank test disabled"
)
def test_two_rank_group_matches_single_process(tmp_path) -> None:
    """One member per rank over two ranks equals two members on one rank."""
    single = _online_cfg(tmp_path / "single", members_per_rank=2)
    run_online(single)
    reference = xr.open_zarr(
        Path(single.output.path) / "stats.zarr", consolidated=False
    ).load()

    out = tmp_path / "pair"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    env["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (
            str(RECIPE_DIR),
            str(RECIPE_DIR.parents[1]),
            env.get("PYTHONPATH", ""),
        )
        if p
    )
    result = subprocess.run(  # noqa: S603 - fixed interpreter and script
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={_free_port()}",
            __file__,
            str(out),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    pair = xr.open_zarr(out / "outputs" / "stats.zarr", consolidated=False).load()
    for name in reference.data_vars:
        np.testing.assert_allclose(
            pair[name].values, reference[name].values, rtol=1e-9, atol=1e-9
        )


def _torchrun_main() -> int:
    """One rank of the two-rank run: ``torchrun ... test_online.py OUT``."""
    import torch.distributed as dist

    from earth2studio.grids import list_grids, register_grid

    if "latlon-small" not in list_grids():
        register_grid("latlon-small", GRID)
    out = Path(sys.argv[1])
    dist.init_process_group("gloo")
    try:
        run_online(_online_cfg(out, members_per_rank=1))
    finally:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(_torchrun_main())

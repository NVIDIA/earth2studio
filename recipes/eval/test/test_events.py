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

"""Events: a time window paired with a region (``scoring.events``).

Covers the config parser and its region merge, the initial conditions an
event adds to a campaign, the wiring into both scoring pathways, and the
scorecard exporter's event windows on a synthetic score store.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from omegaconf import OmegaConf
from src.online import parse_online_settings
from src.regions import (
    event_region_name,
    events_from_attrs,
    events_to_attrs,
    parse_events,
    scoring_regions,
)
from src.scoring import instantiate_metrics
from src.work import build_work_items, event_initial_times

from .conftest import SMALL_LAT, SMALL_LON

BOX = {"lat": [48, 62], "lon": [-15, 5]}
NORTH = {"lat": [0, 90]}  # lands on the small test grid (90, 30)


def _event(**overrides):
    spec = {
        "start": "2025-01-23 12:00:00",
        "end": "2025-01-25 00:00:00",
        "region": BOX,
        "ics": {"step_hours": 12, "lookback_hours": 48},
    }
    spec.update(overrides)
    return spec


# ---------------------------------------------------------------------------
# parse_events
# ---------------------------------------------------------------------------


class TestParseEvents:
    def test_absent(self):
        assert parse_events(None) is None

    def test_defaults_and_normalization(self):
        events = parse_events({"storm": _event()})
        assert list(events) == ["storm"]
        ev = events["storm"]
        assert ev["label"] == "storm"
        assert ev["window"] == "valid"
        assert ev["start"] == np.datetime64("2025-01-23T12:00:00")
        assert ev["end"] == np.datetime64("2025-01-25T00:00:00")
        # A single box normalizes to a one-element list of float bounds,
        # exactly as scoring.regions does.
        assert ev["region"] == [{"lat": [48.0, 62.0], "lon": [-15.0, 5.0]}]
        assert ev["ics"] == {"step_hours": 12, "lookback_hours": 48}
        assert event_region_name("storm", ev) == "storm"

    def test_omegaconf_input_and_label(self):
        cfg = OmegaConf.create({"storm": _event(label="Storm Éowyn")})
        events = parse_events(cfg)
        assert events["storm"]["label"] == "Storm Éowyn"

    def test_region_by_name_and_whole_grid(self):
        events = parse_events(
            {
                "named": _event(region="europe"),
                "whole": _event(region=None),
                "absent": {k: v for k, v in _event().items() if k != "region"},
            }
        )
        assert events["named"]["region"] == "europe"
        assert event_region_name("named", events["named"]) == "europe"
        assert events["whole"]["region"] is None
        assert events["absent"]["region"] is None
        assert event_region_name("whole", events["whole"]) == "whole"

    def test_init_window_defaults_lookback_to_zero(self):
        events = parse_events({"storm": _event(window="init", ics={"step_hours": 6})})
        assert events["storm"]["ics"] == {"step_hours": 6, "lookback_hours": 0}

    def test_no_ics_block(self):
        events = parse_events({"storm": _event(ics=None)})
        assert events["storm"]["ics"] is None

    @pytest.mark.parametrize(
        "spec, match",
        [
            (_event(strat="2025-01-01"), "unknown key"),
            ({k: v for k, v in _event().items() if k != "end"}, "needs"),
            (_event(start="2025-02-01 00:00:00"), "must not be after"),
            (_event(window="lead"), "window must be one of"),
            (_event(start="not a time"), "ISO timestamp"),
            (_event(ics={"lookback_hours": 48}), "step_hours is required"),
            (_event(ics={"step_hours": 12}), "lookback_hours"),
            (_event(ics={"step_hours": 0, "lookback_hours": 1}), "positive"),
            (_event(ics={"step_hours": 6, "lookback_hours": -1}), ">= 0"),
            (_event(ics={"step_hours": 6, "lookbck_hours": 1}), "unknown ics key"),
            (_event(ics=12), "'ics' must be a mapping"),
            (_event(region={"lat": [70, 10]}), "lat bounds"),
            ("not a mapping", "must be a mapping"),
        ],
    )
    def test_rejects(self, spec, match):
        with pytest.raises(ValueError, match=match):
            parse_events({"storm": spec})

    def test_rejects_empty_block(self):
        with pytest.raises(ValueError, match="non-empty mapping"):
            parse_events({})

    def test_attrs_round_trip(self):
        events = parse_events(
            {"storm": _event(label="Storm"), "named": _event(region="europe", ics=None)}
        )
        attrs = events_to_attrs(events)
        assert attrs["storm"]["start"] == "2025-01-23T12:00:00"
        assert attrs["named"]["region"] == "europe"
        assert attrs["named"]["ics"] is None
        assert events_from_attrs(attrs) == events
        assert events_to_attrs(None) is None
        assert events_from_attrs(None) is None


# ---------------------------------------------------------------------------
# scoring_regions: events merged into the region set
# ---------------------------------------------------------------------------


class TestScoringRegions:
    def test_no_events_is_plain_regions(self):
        assert scoring_regions(None) is None
        assert scoring_regions({"regions": None}) is None
        plain = scoring_regions({"regions": {"global": None, "north": NORTH}})
        assert list(plain) == ["global", "north"]

    def test_event_box_joins_regions(self):
        merged = scoring_regions(
            {
                "regions": {"global": None, "north": NORTH},
                "events": {"storm": _event()},
            }
        )
        assert list(merged) == ["global", "north", "storm"]
        assert merged["storm"] == [{"lat": [48.0, 62.0], "lon": [-15.0, 5.0]}]

    def test_event_by_name_adds_nothing(self):
        merged = scoring_regions(
            {
                "regions": {"global": None, "north": NORTH},
                "events": {"storm": _event(region="north")},
            }
        )
        assert list(merged) == ["global", "north"]

    def test_first_regions_get_a_global_split(self):
        merged = scoring_regions({"events": {"storm": _event()}})
        assert list(merged) == ["global", "storm"]
        assert merged["global"] is None
        # A whole-grid event needs no extra global entry.
        whole = scoring_regions({"events": {"storm": _event(region=None)}})
        assert list(whole) == ["storm"]
        assert whole["storm"] is None

    def test_unknown_region_name_rejected(self):
        with pytest.raises(ValueError, match="does not define"):
            scoring_regions({"events": {"storm": _event(region="europe")}})

    def test_name_clash_rejected(self):
        with pytest.raises(ValueError, match="already has a region named"):
            scoring_regions(
                {"regions": {"storm": NORTH}, "events": {"storm": _event()}}
            )

    def test_omegaconf_block(self):
        cfg = OmegaConf.create(
            {"scoring": {"regions": {"global": None}, "events": {"storm": _event()}}}
        )
        assert list(scoring_regions(cfg.scoring)) == ["global", "storm"]


# ---------------------------------------------------------------------------
# Initial conditions from events (work.py)
# ---------------------------------------------------------------------------

EXPECTED_ICS = [
    np.datetime64(t)
    for t in (
        "2025-01-21T12:00:00",
        "2025-01-22T00:00:00",
        "2025-01-22T12:00:00",
        "2025-01-23T00:00:00",
        "2025-01-23T12:00:00",
        "2025-01-24T00:00:00",
        "2025-01-24T12:00:00",
        "2025-01-25T00:00:00",
    )
]


class TestEventInitialTimes:
    def test_grid_from_lookback_to_end(self):
        times = event_initial_times(parse_events({"storm": _event()}))
        assert times == EXPECTED_ICS
        assert all(t.dtype == np.dtype("datetime64[s]") for t in times)

    def test_end_not_on_grid_is_not_overshot(self):
        # 48 h lookback at a 5 h step: the last IC stays at or before `end`.
        events = parse_events(
            {"storm": _event(ics={"step_hours": 5, "lookback_hours": 48})}
        )
        times = event_initial_times(events)
        assert times[0] == np.datetime64("2025-01-21T12:00:00")
        assert times[-1] <= np.datetime64("2025-01-25T00:00:00")
        assert times[-1] > np.datetime64("2025-01-24T19:00:00")

    def test_events_without_ics_add_nothing(self):
        assert event_initial_times(None) == []
        assert event_initial_times(parse_events({"storm": _event(ics=None)})) == []

    def test_union_across_events_is_sorted_and_unique(self):
        events = parse_events(
            {
                "a": _event(),
                "b": _event(
                    start="2025-01-24 00:00:00",
                    end="2025-01-26 00:00:00",
                    ics={"step_hours": 24, "lookback_hours": 24},
                ),
            }
        )
        times = event_initial_times(events)
        assert times == sorted(set(times))
        assert np.datetime64("2025-01-26T00:00:00") in times
        assert times.count(np.datetime64("2025-01-24T00:00:00")) == 1


class TestBuildWorkItemsWithEvents:
    def test_events_only(self):
        cfg = OmegaConf.create(
            {
                "start_times": None,
                "ensemble_size": 1,
                "random_seed": 0,
                "scoring": {"events": {"storm": _event()}},
            }
        )
        items = build_work_items(cfg)
        assert [i.time for i in items] == EXPECTED_ICS

    def test_union_with_start_times(self):
        cfg = OmegaConf.create(
            {
                # One duplicate of an event IC and one IC far away.
                "start_times": ["2025-03-01 00:00:00", "2025-01-22 00:00:00"],
                "ensemble_size": 2,
                "random_seed": 0,
                "scoring": {"events": {"storm": _event()}},
            }
        )
        items = build_work_items(cfg)
        times = sorted({i.time for i in items})
        assert len(times) == len(EXPECTED_ICS) + 1
        assert times[-1] == np.datetime64("2025-03-01T00:00:00")
        assert len(items) == 2 * len(times)
        # Work items stay time-major, member-minor.
        assert [i.ensemble_id for i in items[:4]] == [0, 1, 0, 1]

    def test_union_with_ic_block(self):
        cfg = OmegaConf.create(
            {
                "start_times": None,
                "ic_block_start": "2025-02-01 00:00:00",
                "ic_block_end": "2025-02-02 00:00:00",
                "ic_block_step": 24,
                "scoring": {"events": {"storm": _event()}},
            }
        )
        times = [i.time for i in build_work_items(cfg)]
        assert len(times) == len(EXPECTED_ICS) + 2

    def test_no_source_at_all_raises(self):
        cfg = OmegaConf.create(
            {"start_times": None, "scoring": {"events": {"storm": _event(ics=None)}}}
        )
        with pytest.raises(ValueError, match="scoring.events"):
            build_work_items(cfg)

    def test_no_scoring_block_still_works(self):
        cfg = OmegaConf.create({"start_times": ["2025-01-01 00:00:00"]})
        assert len(build_work_items(cfg)) == 1


# ---------------------------------------------------------------------------
# Both scoring pathways see the event's box as a region
# ---------------------------------------------------------------------------


class TestPathwayWiring:
    def test_online_settings(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path)},
                "scoring": {
                    "lat_weights": True,
                    "regions": {"global": None, "north": NORTH},
                    "events": {"storm": _event(), "named": _event(region="north")},
                    "online": {"lsd": True},
                    "output": {"store_name": "scores.zarr"},
                },
            }
        )
        settings = parse_online_settings(cfg)
        assert list(settings.regions) == ["global", "north", "storm"]

    def test_online_settings_events_without_regions(self, tmp_path):
        cfg = OmegaConf.create(
            {
                "output": {"path": str(tmp_path)},
                "scoring": {
                    "events": {"storm": _event()},
                    # LSD needs a whole-grid region; the merge supplies it.
                    "online": {"lsd": True},
                    "output": {"store_name": "scores.zarr"},
                },
            }
        )
        assert list(parse_online_settings(cfg).regions) == ["global", "storm"]

    def test_offline_metrics(self):
        cfg = OmegaConf.create(
            {
                "scoring": {
                    "lat_weights": False,
                    "events": {"north_event": _event(region=NORTH)},
                    "metrics": {
                        "rmse": {
                            "_target_": "earth2studio.statistics.rmse",
                            "reduction_dimensions": ["lat", "lon"],
                        }
                    },
                }
            }
        )
        coords = OrderedDict({"lat": SMALL_LAT, "lon": SMALL_LON})
        metrics = instantiate_metrics(cfg, coords)
        assert list(metrics["rmse"]._per_region) == ["global", "north_event"]


# ---------------------------------------------------------------------------
# Exporter: event windows on a synthetic score store
# ---------------------------------------------------------------------------

SCORECARD = Path(__file__).resolve().parents[1] / "scorecard"


@pytest.fixture(scope="module")
def export_mod():
    """The scorecard exporter, loaded from its script path."""
    spec = importlib.util.spec_from_file_location(
        "export_scores", SCORECARD / "export_scores.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_scores"] = module
    spec.loader.exec_module(module)
    return module


IC_TIMES = np.array(
    [
        "2025-01-01T00:00",
        "2025-01-01T06:00",
        "2025-01-01T12:00",
        "2025-01-01T18:00",
        "2025-01-02T00:00",
    ],
    dtype="datetime64[ns]",
)
LEADS = np.array([0, 6, 12, 18, 24], dtype="timedelta64[h]").astype("timedelta64[ns]")
REGIONS = ["global", "storm"]


def _write_scores(run: Path, events: dict | None, region_offset: float = 10.0):
    """A store whose per-IC MSE is ``ic_index + 1`` on the whole grid and
    ``ic_index + 1 + region_offset`` on the event box, constant over lead
    time, so windowed means have closed forms."""
    n_t, n_r, n_l = len(IC_TIMES), len(REGIONS), len(LEADS)
    base = (np.arange(n_t, dtype=float) + 1.0)[:, None, None] * np.ones((1, n_r, n_l))
    base[:, 1, :] += region_offset
    mse = np.stack([base, base * 4.0], axis=2)  # ensemble axis: member 1 differs
    ds = xr.Dataset(
        {
            "mse__t2m": (("time", "region", "ensemble", "lead_time"), mse),
            "crps__t2m": (("time", "region", "lead_time"), base / 2.0),
        },
        coords={
            "time": IC_TIMES,
            "region": np.array(REGIONS),
            "ensemble": np.arange(2),
            "lead_time": LEADS,
        },
        attrs={"regions": REGIONS},
    )
    if events is not None:
        ds.attrs["events"] = events_to_attrs(events)
    (run).mkdir(parents=True, exist_ok=True)
    ds.to_zarr(run / "scores.zarr", mode="w", consolidated=True)


def _valid_event(**overrides):
    spec = {
        "label": "Storm",
        "start": "2025-01-01 12:00:00",
        "end": "2025-01-01 18:00:00",
        "region": BOX,
    }
    spec.update(overrides)
    return spec


class TestExportEvents:
    def test_curve_valid_window(self, export_mod, tmp_path):
        run = tmp_path / "toy_run"
        _write_scores(run, None)
        ds = xr.open_zarr(run / "scores.zarr")
        window = (
            np.datetime64("2025-01-01T12:00"),
            np.datetime64("2025-01-01T18:00"),
            "valid",
        )
        got = export_mod.curve(
            ds, ("mse",), "t2m", single=True, region="storm", window=window
        )
        # Lead L keeps the ICs with IC + L inside [12Z, 18Z]: leads 0/6/12 h
        # average two ICs, 18 h one IC, 24 h none.
        expected = [
            np.sqrt(np.mean([13.0, 14.0])),
            np.sqrt(np.mean([12.0, 13.0])),
            np.sqrt(np.mean([11.0, 12.0])),
            np.sqrt(11.0),
            np.nan,
        ]
        np.testing.assert_allclose(got[:4], expected[:4])
        assert np.isnan(got[4])

    def test_curve_init_window(self, export_mod, tmp_path):
        run = tmp_path / "toy_run"
        _write_scores(run, None)
        ds = xr.open_zarr(run / "scores.zarr")
        window = (
            np.datetime64("2025-01-01T06:00"),
            np.datetime64("2025-01-01T12:00"),
            "init",
        )
        got = export_mod.curve(
            ds, ("crps",), "t2m", single=False, region=None, window=window
        )
        np.testing.assert_allclose(got, np.full(len(LEADS), np.mean([2.0, 3.0]) / 2))

    def test_curve_window_outside_run_is_none(self, export_mod, tmp_path):
        run = tmp_path / "toy_run"
        _write_scores(run, None)
        ds = xr.open_zarr(run / "scores.zarr")
        window = (np.datetime64("2025-03-01"), np.datetime64("2025-03-02"), "valid")
        assert export_mod.curve(ds, ("mse",), "t2m", True, window=window) is None

    def test_export_documents(self, export_mod, tmp_path):
        run = tmp_path / "toy_2025_scorecard"
        events = parse_events(
            {
                "storm": _valid_event(),
                "early": _valid_event(
                    label="Early ICs",
                    start="2025-01-01 06:00:00",
                    end="2025-01-01 12:00:00",
                    region="global",
                    window="init",
                ),
            }
        )
        _write_scores(run, events)
        doc, splits = export_mod.export("toy", run)

        assert doc["has_events"] is True
        assert list(doc["events"]) == ["storm", "early"]
        assert doc["events"]["storm"] == {
            "label": "Storm",
            "start": "2025-01-01 12:00",
            "end": "2025-01-01 18:00",
            "window": "valid",
            "region": "storm",
        }
        # The event's box is not a year-round regional split; only the
        # whole-grid entry survives, as for any store without other boxes.
        assert doc.get("regions", ["global"]) == ["global"]
        assert not any(k.startswith("region_") for k in splits)

        ev = splits["events"]
        assert ev["events"] == ["storm", "early"]
        storm = ev["metrics_by_event"]["storm"]
        assert storm["run"] == run.name
        assert storm["initial_conditions"] == 4
        assert storm["region_box"] == [{"lat": [48.0, 62.0], "lon": [-15.0, 5.0]}]
        # The export drops lead 0 like everywhere else: leads 6/12/18/24 h remain.
        rmse = storm["metrics"]["rmse"]["values"]["t2m"]
        assert rmse[:3] == pytest.approx(
            [np.sqrt(12.5), np.sqrt(11.5), np.sqrt(11.0)], rel=1e-4
        )
        assert rmse[3] is None
        early = ev["metrics_by_event"]["early"]
        assert early["region"] == "global"
        assert early["region_box"] is None
        assert early["initial_conditions"] == 2
        assert early["metrics"]["crps"]["values"]["t2m"] == pytest.approx(
            [1.25] * 4, rel=1e-4
        )
        assert run.name in ev["runs"]

    def test_export_merges_events_run(self, export_mod, tmp_path):
        main_run = tmp_path / "toy_2025_scorecard"
        events_run = tmp_path / "toy_2025_events"
        _write_scores(main_run, None)
        _write_scores(
            events_run,
            parse_events({"storm": _valid_event()}),
            region_offset=20.0,
        )
        doc, splits = export_mod.export("toy", main_run, events_run=events_run)
        assert doc["has_events"] is True
        storm = splits["events"]["metrics_by_event"]["storm"]
        assert storm["run"] == "toy_2025_events"
        # The sibling run's numbers, not the main run's: offset 20 on the box.
        rmse = storm["metrics"]["rmse"]["values"]["t2m"]
        assert rmse[0] == pytest.approx(np.sqrt(np.mean([22.0, 23.0])), rel=1e-4)
        assert list(splits["events"]["runs"]) == ["toy_2025_events"]

    def test_export_without_events_is_unchanged(self, export_mod, tmp_path):
        run = tmp_path / "toy_2025_scorecard"
        _write_scores(run, None)
        doc, splits = export_mod.export("toy", run)
        assert "has_events" not in doc
        assert "events" not in doc
        assert "events" not in splits
        assert doc["regions"] == REGIONS
        assert "region_storm" in splits

    def test_run_events_falls_back_to_campaign_file(self, export_mod, tmp_path):
        # A run directory named after the bundled events campaign, with no
        # attribute stamp, resolves its events from cfg/campaign/.
        run = tmp_path / "stormcast_2025_events"
        _write_scores(run, None)
        ds = xr.open_zarr(run / "scores.zarr")
        events = export_mod.run_events(run, ds)
        assert list(events) == ["mar14_missouri_arkansas", "apr02_mid_south"]
        assert events["apr02_mid_south"]["window"] == "valid"
        assert events["apr02_mid_south"]["region"] == "global"
        assert events["apr02_mid_south"]["ics"] == {
            "step_hours": 3,
            "lookback_hours": 12,
        }


# ---------------------------------------------------------------------------
# Exporter: publishing to a Hugging Face dataset
# ---------------------------------------------------------------------------


class _FakeHfApi:
    """Records the calls ``upload_exports`` makes instead of hitting the hub."""

    def __init__(self, user: str | None = "someone"):
        self._user = user
        self.created: list = []
        self.uploads: list = []

    def whoami(self):
        if self._user is None:
            raise OSError("Token is required")
        return {"name": self._user}

    def create_repo(self, repo_id, repo_type=None, exist_ok=False):
        self.created.append((repo_id, repo_type, exist_ok))

    def upload_folder(self, **kwargs):
        self.uploads.append(kwargs)


class TestUploadExports:
    def test_defaults_to_the_logged_in_users_dataset(self, export_mod, monkeypatch):
        monkeypatch.delenv("SCORECARD_DATA_REPO", raising=False)
        api = _FakeHfApi(user="dibyajyoti43")
        repo = export_mod.upload_exports("fcn3", api=api)
        assert repo == "dibyajyoti43/earth2studio-assets"
        assert api.created == [(repo, "dataset", True)]
        (call,) = api.uploads
        assert call["repo_id"] == repo
        assert call["repo_type"] == "dataset"
        assert call["path_in_repo"] == "scorecard/fcn3"
        assert call["folder_path"] == str(export_mod.EXPORTS)
        # Only this model's files, and stale ones of the same model go away.
        patterns = ["eval_scores_fcn3.json", "eval_scores_fcn3_*.json"]
        assert call["allow_patterns"] == patterns
        assert call["delete_patterns"] == patterns

    def test_env_and_explicit_repo_win_over_login(self, export_mod, monkeypatch):
        monkeypatch.setenv("SCORECARD_DATA_REPO", "org/from-env")
        api = _FakeHfApi(user=None)  # not logged in: whoami would fail
        assert export_mod.upload_exports("fcn3", api=api) == "org/from-env"
        assert export_mod.upload_exports("fcn3", "org/explicit", api=api) == (
            "org/explicit"
        )

    def test_missing_login_is_a_clear_error(self, export_mod, monkeypatch):
        monkeypatch.delenv("SCORECARD_DATA_REPO", raising=False)
        with pytest.raises(RuntimeError, match="hf auth login"):
            export_mod.upload_exports("fcn3", api=_FakeHfApi(user=None))


class TestDropUnscoredTimes:
    def test_never_run_initial_conditions_are_left_out(self, export_mod, tmp_path):
        run = tmp_path / "toy_2025_scorecard"
        _write_scores(run, None)
        ds = xr.open_zarr(run / "scores.zarr")
        # Blank the last two initial conditions in every array: never run.
        blanked = ds.copy()
        for name in blanked.data_vars:
            values = blanked[name].values.copy()
            values[-2:] = np.nan
            blanked[name] = (blanked[name].dims, values)
        kept = export_mod.drop_unscored_times(blanked)
        assert kept.sizes["time"] == len(IC_TIMES) - 2
        np.testing.assert_array_equal(kept.time.values, IC_TIMES[:-2])
        # A partially scored initial condition stays.
        partial = ds.copy()
        values = partial["crps__t2m"].values.copy()
        values[0, :, 1:] = np.nan
        partial["crps__t2m"] = (partial["crps__t2m"].dims, values)
        assert export_mod.drop_unscored_times(partial).sizes["time"] == len(IC_TIMES)

    def test_export_reports_only_scored_initial_conditions(self, export_mod, tmp_path):
        run = tmp_path / "toy_2025_scorecard"
        _write_scores(run, None)
        ds = xr.open_zarr(run / "scores.zarr").load()
        for name in ds.data_vars:
            values = ds[name].values.copy()
            values[-1] = np.nan
            ds[name] = (ds[name].dims, values)
        ds.to_zarr(run / "scores.zarr", mode="w", consolidated=True)
        doc, splits = export_mod.export("toy", run)
        assert len(doc["initial_conditions"]) == len(IC_TIMES) - 1
        assert len(splits["heatmap"]["initial_conditions"]) == len(IC_TIMES) - 1

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

"""SFNO collector, workflow, and catalog tests."""

from __future__ import annotations

import sys
import threading
import types
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.dsx.schema import WeatherSchema
from src.sfno.collector import SFNOCollector
from src.sfno.variables import VARIABLES
from src.sfno.workflow import (
    _build_perturbation,
    _CycleNotReady,
    _forecast_loop,
    _publish_newest_loadable,
    _validate_ensemble,
)
from src.sfno.workflow import (
    run as run_workflow,
)
from src.shared.site_extraction import RegularLatLonSiteExtractor

_H6 = timedelta(hours=6)


# --- SFNOCollector --------------------------------------------------------------

_LAT5 = np.linspace(2.0, -2.0, 5)  # [2,1,0,-1,-2]
_LON8 = np.linspace(0.0, 360.0, 8, endpoint=False)  # [0,45,90,...]


def _collector_one_site() -> SFNOCollector:
    ex = RegularLatLonSiteExtractor(
        _LAT5, _LON8, [{"id": "s", "lat": 0.0, "lon": 90.0}]
    )
    return SFNOCollector(ex, "sfno", init_ms=1000)


def _field(value: float) -> np.ndarray:
    """A 5x8 field whose site cell (lat 0 -> idx 2, lon 90 -> idx 2) holds ``value``."""
    f = np.full((5, 8), -999.0)
    f[2, 2] = value
    return f


def _ens_field(values: list[float]) -> np.ndarray:
    """An (N, 5, 8) ensemble batch; member i's site cell holds ``values[i]``."""
    f = np.full((len(values), 5, 8), -999.0)
    for i, v in enumerate(values):
        f[i, 2, 2] = v
    return f


def test_collector_maps_sources_to_wire_vars() -> None:
    c = _collector_one_site()
    # Two leads (6 h, 12 h), each with the three surface source vars.
    for lead, base in ((np.timedelta64(6, "h"), 0.0), (np.timedelta64(12, "h"), 100.0)):
        coords = {"lead_time": lead}
        c.write(_field(base + 1), coords, "t2m")
        c.write(_field(base + 2), coords, "u10m")
        c.write(_field(base + 3), coords, "v10m")
    series = {(s.site_id, s.variable): s for s in c.collect()}
    assert {v for (_s, v) in series} == {"Temperature", "WindU", "WindV"}
    temp = series[("s", "Temperature")]
    assert temp.lead_seconds == [21600, 43200]  # sorted, exact seconds
    assert temp.member_values == [[1.0, 101.0]]  # single (deterministic) member
    assert series[("s", "WindU")].member_values == [[2.0, 102.0]]
    assert series[("s", "WindV")].member_values == [[3.0, 103.0]]
    assert temp.model == "sfno"


def test_collector_ignores_unrequested_source_vars() -> None:
    c = _collector_one_site()
    c.write(_field(5.0), {"lead_time": np.timedelta64(6, "h")}, "t2m")
    c.write(
        _field(9.0), {"lead_time": np.timedelta64(6, "h")}, "sp"
    )  # not a source var
    series = {(s.site_id, s.variable): s for s in c.collect()}
    assert {v for (_s, v) in series} == {"Temperature"}  # sp dropped


@pytest.mark.parametrize(
    ("field", "message"),
    [
        pytest.param(np.zeros((2, 5, 8)), "member", id="unexpected-leading-dimension"),
        pytest.param(
            np.zeros(8), "at least 2 dimensions", id="missing-spatial-dimension"
        ),
    ],
)
def test_collector_rejects_invalid_field_shape(field: np.ndarray, message: str) -> None:
    collector = _collector_one_site()
    collector.begin_cycle(1000, member_count=1)

    with pytest.raises(ValueError, match=message):
        collector.write(
            field,
            {"lead_time": np.timedelta64(6, "h")},
            "t2m",
        )


def test_collector_ensemble_multibatch_ordered() -> None:
    # Two batch_size writes (members [0,1] then [2]) assemble into ordered member_values.
    c = _collector_one_site()
    c.begin_cycle(1000, member_count=3)
    lead = {"lead_time": np.timedelta64(6, "h")}
    c.write(_ens_field([10.0, 11.0]), {"ensemble": np.array([0, 1]), **lead}, "t2m")
    c.write(_ens_field([12.0]), {"ensemble": np.array([2]), **lead}, "t2m")
    series = {(s.site_id, s.variable): s for s in c.collect()}
    temp = series[("s", "Temperature")]
    assert temp.member_values == [[10.0], [11.0], [12.0]]  # ordered by member index


def test_collector_ensemble_rejects_out_of_range_member() -> None:
    c = _collector_one_site()
    c.begin_cycle(1000, member_count=2)
    with pytest.raises(ValueError, match="out of range"):
        c.write(
            _ens_field([1.0]),
            {"ensemble": np.array([2]), "lead_time": np.timedelta64(6, "h")},
            "t2m",
        )


@pytest.mark.parametrize(
    "member_ids",
    [
        np.array([0.5]),
        np.array([], dtype=int),
        np.array([0, 0]),
    ],
)
def test_collector_rejects_invalid_member_coordinates(
    member_ids: np.ndarray,
) -> None:
    collector = _collector_one_site()
    collector.begin_cycle(1000, member_count=2)

    with pytest.raises(ValueError, match="ensemble coordinates"):
        collector.write(
            _ens_field([1.0] * len(member_ids)),
            {
                "ensemble": member_ids,
                "lead_time": np.timedelta64(6, "h"),
            },
            "t2m",
        )

    assert not collector._buffer


def test_collector_rejects_field_name_count_before_buffering() -> None:
    collector = _collector_one_site()

    with pytest.raises(ValueError, match="2 field.*1 variable"):
        collector.write(
            [_field(1.0), _field(2.0)],
            {"lead_time": np.timedelta64(6, "h")},
            np.array(["t2m"]),
        )

    assert not collector._buffer


def test_collector_ensemble_rejects_duplicate_member_write() -> None:
    c = _collector_one_site()
    c.begin_cycle(1000, member_count=2)
    coords = {"ensemble": np.array([0]), "lead_time": np.timedelta64(6, "h")}
    c.write(_ens_field([1.0]), coords, "t2m")
    with pytest.raises(ValueError, match="duplicate"):
        c.write(_ens_field([2.0]), coords, "t2m")  # same member/lead/var again


def test_collector_ensemble_collect_rejects_incomplete_ensemble() -> None:
    # member_count 3 but only 0,1 written -> collect refuses (won't fabricate a shorter ensemble).
    c = _collector_one_site()
    c.begin_cycle(1000, member_count=3)
    c.write(
        _ens_field([1.0, 2.0]),
        {"ensemble": np.array([0, 1]), "lead_time": np.timedelta64(6, "h")},
        "t2m",
    )
    with pytest.raises(ValueError, match="expected members 0..2"):
        c.collect()


def test_collector_ensemble_collect_rejects_inconsistent_lead_times() -> None:
    collector = _collector_one_site()
    collector.begin_cycle(1000, member_count=2)
    for member_id in (0, 1):
        collector.write(
            _ens_field([float(member_id)]),
            {
                "ensemble": np.array([member_id]),
                "lead_time": np.timedelta64(6, "h"),
            },
            "t2m",
        )
    collector.write(
        _ens_field([2.0]),
        {
            "ensemble": np.array([0]),
            "lead_time": np.timedelta64(12, "h"),
        },
        "t2m",
    )

    with pytest.raises(ValueError, match="different lead times"):
        collector.collect()


def test_collector_begin_cycle_resets() -> None:
    c = _collector_one_site()
    c.write(_field(1.0), {"lead_time": np.timedelta64(6, "h")}, "t2m")
    c.begin_cycle(2000)
    assert c.init_ms == 2000
    assert dict(c._buffer) == {}


@pytest.mark.parametrize(
    ("init_ms", "member_count"),
    [(True, 1), (1000, 0), (1000, True), (1000, 1.5)],
)
def test_collector_begin_cycle_rejects_invalid_values_without_clearing(
    init_ms: object, member_count: object
) -> None:
    collector = _collector_one_site()
    collector.write(_field(1.0), {"lead_time": np.timedelta64(6, "h")}, "t2m")
    buffered = {key: dict(values) for key, values in collector._buffer.items()}

    with pytest.raises(ValueError):
        collector.begin_cycle(init_ms, member_count)  # type: ignore[arg-type]

    assert collector.init_ms == 1000
    assert collector._buffer == buffered


# --- catalog vs contract --------------------------------------------------------


def test_sfno_catalog_is_subset_of_governed_enum() -> None:
    schema = WeatherSchema.load()
    enum = set(schema.channel_enum("forecast", "variable"))
    assert set(VARIABLES) <= enum, f"sfno vars not in enum: {set(VARIABLES) - enum}"
    assert set(VARIABLES) == {"Temperature", "WindU", "WindV"}
    assert all("unit" in spec for spec in VARIABLES.values())


# --- GFS walk-back fallback (_publish_newest_loadable) --------------------------

_INIT = datetime(2026, 1, 2, 12, 0, 0)


def _attempt_failing_for(not_ready: set[datetime]):
    """Build (attempt, calls): attempt raises _CycleNotReady for cycles in ``not_ready``."""
    calls: list[datetime] = []

    def attempt(cyc: datetime) -> None:
        calls.append(cyc)
        if cyc in not_ready:
            raise _CycleNotReady("f000 not uploaded")

    return attempt, calls


def test_fallback_newest_incomplete_uses_previous_cycle() -> None:
    attempt, calls = _attempt_failing_for({_INIT})
    used = _publish_newest_loadable(
        _INIT, _INIT - timedelta(hours=24), False, None, attempt
    )
    assert used == _INIT - _H6  # walked back one 6h cycle to the loadable one
    assert calls == [_INIT, _INIT - _H6]


def test_fallback_exhausts_lookback_window_returns_none() -> None:
    # Everything unready; earliest = init-12h, so it tries init, -6h, -12h then gives up.
    attempt, calls = _attempt_failing_for({_INIT, _INIT - _H6, _INIT - 2 * _H6})
    used = _publish_newest_loadable(
        _INIT, _INIT - timedelta(hours=12), False, None, attempt
    )
    assert used is None
    assert calls == [_INIT, _INIT - _H6, _INIT - 2 * _H6]


def test_fallback_never_walks_to_or_before_last_init() -> None:
    # last_init is one cycle back, so a not-ready newest cycle must NOT re-publish last_init.
    attempt, calls = _attempt_failing_for({_INIT})
    used = _publish_newest_loadable(
        _INIT, _INIT - timedelta(hours=24), False, _INIT - _H6, attempt
    )
    assert used is None
    assert calls == [_INIT]  # never attempted last_init (init-6h)


def test_fallback_fixed_init_never_walks_back() -> None:
    attempt, calls = _attempt_failing_for({_INIT})
    used = _publish_newest_loadable(
        _INIT, _INIT - timedelta(hours=24), True, None, attempt
    )
    assert used is None
    assert calls == [_INIT]  # pinned: no walk-back


def _install_fake_earth2studio(monkeypatch, deterministic) -> None:
    """Inject fake earth2studio.run / utils.time so _forecast_loop runs without the model stack."""
    run_mod = types.ModuleType("earth2studio.run")
    run_mod.deterministic = deterministic
    time_mod = types.ModuleType("earth2studio.utils.time")
    time_mod.to_time_array = lambda lst: np.array([np.datetime64(lst[0])])
    utils = types.ModuleType("earth2studio.utils")
    utils.time = time_mod
    e2s = types.ModuleType("earth2studio")
    e2s.run = run_mod
    e2s.utils = utils
    for name, mod in [
        ("earth2studio", e2s),
        ("earth2studio.run", run_mod),
        ("earth2studio.utils", utils),
        ("earth2studio.utils.time", time_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def _loop_args(deterministic, monkeypatch):
    _install_fake_earth2studio(monkeypatch, deterministic)
    return dict(
        init_cfg="2026-01-02T12:00:00",  # fixed init: no resolve_latest / data.available needed
        max_consecutive_failures=5,
        max_lookback=24,
        model=MagicMock(),
        data=MagicMock(),
        collector=MagicMock(),
        coordinator=MagicMock(),
        publisher=MagicMock(),
        transport=None,
        device="cpu",
        nsteps=8,
        ensemble=dict(members=1, include_members=False),  # deterministic path
        poll_interval_s=900,
        one_shot=True,
        stop=threading.Event(),
    )


def test_one_shot_exhaustion_exits_nonzero(monkeypatch) -> None:
    def deterministic(*a, **k):
        raise FileNotFoundError("f000 not uploaded")

    kwargs = _loop_args(deterministic, monkeypatch)
    # Fixed init that never loads -> nothing published -> --once must exit nonzero, not silently 0.
    with pytest.raises(SystemExit):
        _forecast_loop(**kwargs)


def test_one_shot_with_forecast_does_not_exit(monkeypatch) -> None:
    import src.sfno.workflow as sfno_workflow

    def deterministic(*a, **k):
        return None  # succeeds

    pruned = []
    monkeypatch.setattr(
        sfno_workflow,
        "prune_data_cache",
        lambda hours, subdirs: pruned.append((hours, subdirs)),
    )
    kwargs = _loop_args(deterministic, monkeypatch)
    kwargs["cache_retention_hours"] = 48
    kwargs["publisher"].publish_pending.return_value = (
        True  # publish_with_reconnect -> True
    )
    _forecast_loop(**kwargs)  # no SystemExit
    kwargs["publisher"].publish_pending.assert_called()
    assert pruned == [(48, ("gfs",))]


def test_staging_file_error_propagates_not_treated_as_not_ready(monkeypatch) -> None:
    def deterministic(*a, **k):
        return None  # the fetch succeeds

    kwargs = _loop_args(deterministic, monkeypatch)
    # A FileNotFoundError AFTER the fetch (staging) is a real error, not "cycle not ready": it must
    # propagate, not trigger a walk-back.
    kwargs["coordinator"].stage_forecasts.side_effect = FileNotFoundError(
        "schema missing"
    )
    with pytest.raises(FileNotFoundError):
        _forecast_loop(**kwargs)


def test_forecast_loop_honors_preset_stop(monkeypatch) -> None:
    ran: list[int] = []

    def deterministic(*a, **k):
        ran.append(1)

    kwargs = _loop_args(deterministic, monkeypatch)
    kwargs["one_shot"] = False  # persistent mode
    stop = threading.Event()
    stop.set()
    kwargs["stop"] = stop
    _forecast_loop(**kwargs)  # returns cleanly (no SystemExit), no cycle run
    assert ran == []


# --- workflow config validation ------------------------------------------------
@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("run", "init_time", "not-a-time"),
        ("run", "max_lookback_hours", 1.5),
        ("run", "max_consecutive_failures", True),
        ("run", "nsteps", True),
        ("run", "poll_interval_seconds", 0),
        ("bus", "heartbeat_seconds", 101),
        ("bus", "heartbeat_seconds", 0),
        ("bus", "metadata_heartbeat_seconds", 90),
    ],
)
def test_run_rejects_invalid_config_before_model_import(
    section: str, key: str, value: object
) -> None:
    cfg = {
        "sites": [{"id": "site", "lat": 0.0, "lon": 0.0}],
        "run": {"init_time": "latest", "nsteps": 1},
        "bus": {},
    }
    cfg[section][key] = value
    args = types.SimpleNamespace(once=True, dry_run=True)

    with pytest.raises(ValueError, match=key):
        run_workflow(cfg, args, threading.Event())


# --- ensemble config validation -------------------------------------------------
@pytest.mark.parametrize(
    ("config", "expected"),
    [
        pytest.param({}, {"members": 1, "include_members": False}, id="defaults"),
        pytest.param(
            {"ensemble": {"members": 1}},
            {"members": 1, "include_members": False},
            id="explicit-deterministic",
        ),
        pytest.param(
            {
                "ensemble": {
                    "members": 4,
                    "batch_size": 2,
                    "include_members": True,
                    "seed": 7,
                }
            },
            {
                "members": 4,
                "batch_size": 2,
                "include_members": True,
                "seed": 7,
            },
            id="ensemble-options",
        ),
    ],
)
def test_validate_ensemble_accepts_valid_config(
    config: dict[str, object], expected: dict[str, object]
) -> None:
    assert _validate_ensemble(config) == expected


@pytest.mark.parametrize(
    ("config", "message"),
    [
        *[
            pytest.param(
                {"ensemble": {"members": bad}},
                "members",
                id=f"members-{bad!r}",
            )
            for bad in (0, -1, 2.0, True, "3")
        ],
        pytest.param(
            {"ensemble": {"members": 3, "batch_size": 5}},
            "batch_size",
            id="batch-size-out-of-range",
        ),
        pytest.param(
            {"ensemble": {"members": 2, "seed": 1.5}},
            "seed",
            id="seed-not-integer",
        ),
        *[
            pytest.param(
                {"ensemble": {"members": 2, "include_members": bad}},
                "include_members",
                id=f"include-members-{bad!r}",
            )
            for bad in ("yes", 1, 0, None)
        ],
        pytest.param(
            {"ensemble": {"member": 8}},
            "unknown ensemble config key",
            id="unknown-key",
        ),
        pytest.param({"ensemble": [1, 2]}, "mapping", id="non-mapping"),
    ],
)
def test_validate_ensemble_rejects_invalid_config(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_ensemble(config)


def test_build_perturbation_requires_target_variable() -> None:
    # If the model's inputs lack z500, the perturbation must fail loud, not perturb nothing.
    model = MagicMock()
    model.input_coords.return_value = {"variable": np.array(["t2m", "u10m"])}
    with pytest.raises(ValueError, match="z500"):
        _build_perturbation(model)


# --- ensemble dispatch (mocked run.ensemble) ------------------------------------
def test_ensemble_dispatch_calls_run_ensemble(monkeypatch) -> None:
    # members > 1 must call run.ensemble (not run.deterministic) with the configured member count
    # and batch size, and tell the collector the member count for the cycle.
    det = MagicMock(name="deterministic")
    ens = MagicMock(name="ensemble")
    _install_fake_earth2studio(monkeypatch, det)
    sys.modules["earth2studio.run"].ensemble = ens
    csg = MagicMock(name="CSG")
    pert = types.ModuleType("earth2studio.perturbation")
    pert.CorrelatedSphericalGaussian = csg
    monkeypatch.setitem(sys.modules, "earth2studio.perturbation", pert)
    sys.modules["earth2studio"].perturbation = pert

    # Fake torch so _build_perturbation can construct the per-variable amplitude without the stack.
    class _FakeAmp:
        def __init__(self, data: object) -> None:
            self.data = list(data)  # type: ignore[arg-type]

        def reshape(self, *shape: int) -> _FakeAmp:
            self.shape = shape
            return self

    faketorch = types.ModuleType("torch")
    faketorch.tensor = lambda data, dtype=None: _FakeAmp(data)
    faketorch.float32 = "f32"
    faketorch.manual_seed = lambda s: None
    monkeypatch.setitem(sys.modules, "torch", faketorch)

    model = MagicMock()
    model.input_coords.return_value = {"variable": np.array(["z500", "t2m", "u10m"])}
    collector = MagicMock()
    publisher = MagicMock()
    publisher.publish_pending.return_value = True
    _forecast_loop(
        init_cfg="2026-01-02T12:00:00",
        max_consecutive_failures=5,
        max_lookback=24,
        model=model,
        data=MagicMock(),
        collector=collector,
        coordinator=MagicMock(),
        publisher=publisher,
        transport=None,
        device="cpu",
        nsteps=8,
        ensemble=dict(members=3, batch_size=1, seed=None, include_members=False),
        poll_interval_s=900,
        one_shot=True,
        stop=threading.Event(),
    )
    ens.assert_called_once()
    det.assert_not_called()
    assert collector.begin_cycle.call_args.kwargs.get("member_count") == 3
    args, ekw = ens.call_args
    assert args[2] == 3  # nensemble (positional)
    assert ekw.get("batch_size") == 1
    # The perturbation amplitude targets z500 only (39.27), zero for the other input variables.
    amp = csg.call_args.kwargs["noise_amplitude"]
    assert amp.data == [39.27, 0.0, 0.0]


# --- ensemble pipeline: collector -> coordinator -> schema-valid payload ---------
def test_ensemble_pipeline_to_schema_valid_payload() -> None:
    from src.dsx.coordinator import DSXCoordinator
    from src.dsx.publisher import DSXPublisher

    schema = WeatherSchema.load()
    site = {"id": "s", "lat": 0.0, "lon": 90.0}
    collector = SFNOCollector(RegularLatLonSiteExtractor(_LAT5, _LON8, [site]), "sfno")
    publisher = DSXPublisher(None, schema, dry_run=True)
    coord = DSXCoordinator(
        publisher,
        schema,
        "Weather/v1/PUB",
        "global-medium-range-weather",
        {"s": site},
        "sfno",
        horizon_seconds=6 * 21600,
        cadence_seconds=21600,
        inputs=[{"source": "GFS", "role": "initialCondition"}],
        variables=VARIABLES,
        include_members=True,
    )
    collector.begin_cycle(1000, member_count=2)
    collector.write(
        _ens_field([300.0, 302.0]),
        {
            "ensemble": np.array([0, 1]),
            "lead_time": np.timedelta64(6, "h"),
        },
        "t2m",
    )
    coord.stage_forecasts(collector.collect())

    staged = dict(publisher._pending)
    temp = staged["Weather/v1/PUB/Forecast/global-medium-range-weather/s/Temperature"]
    assert temp["memberCount"] == 2
    assert temp["values"] == [301.0]
    assert temp["members"] == [[300.0], [302.0]]
    assert temp["model"] == "sfno"

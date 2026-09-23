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

"""Run SFNO global medium-range forecasts and publish results for each site.

This workflow initializes SFNO forecasts from GFS and advances the global
state in fixed 6-hour steps. A 14-day forecast uses 56 steps and produces 57
lead times, including the initial state.

``run(cfg, args, stop)`` is the entry point. ``main.py`` selects this workflow
from ``cfg["workflow"]``. The workflow reuses the shared DSX publishing and
reconnection code.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
from loguru import logger

from ..dsx.bus import BusTransport, build_transport
from ..dsx.contract_adapter import TOPIC_PREFIX, check_topics_config
from ..dsx.coordinator import DSXCoordinator
from ..dsx.publish_loop import publish_with_reconnect, republish_on_heartbeat
from ..dsx.publisher import DSXPublisher
from ..dsx.schema import WeatherSchema
from ..shared.cycle_availability import floor_to_cycle, resolve_latest
from ..shared.data_cache import prune_data_cache, resolve_cache_retention_hours
from ..shared.site_extraction import (
    RegularLatLonSiteExtractor,
    validate_and_normalize_sites,
)
from .collector import SOURCE_VARS, SFNOCollector
from .variables import VARIABLES


class _CycleNotReady(Exception):
    """Signal that the workflow should try an older GFS cycle.

    If the forecast run cannot find an input file, the workflow raises this
    exception and tries an older GFS cycle. Missing files during staging or
    publishing still fail normally.
    """


# SFNO advances the forecast by a fixed six hours per step.
_NATIVE_STEP_SECONDS = 21600

# GFS analysis cycles start every six hours.
_GFS_CYCLE_HOURS = 6

# Experimental z500-only ensemble perturbation.
# This is not calibrated probabilistic guidance.
_PERTURB_VARIABLE = "z500"
_PERTURB_AMPLITUDE = 39.27


def _epoch_ms(dt64: np.datetime64) -> int:
    """Convert a NumPy datetime to milliseconds since the Unix epoch."""
    return int(np.asarray(dt64).astype("datetime64[ms]").astype("int64"))


def _resolve_package(
    model_cfg: dict[str, Any], package_cls: Any, default_loader: Any
) -> Any:
    """Return the configured SFNO model package.

    Use ``model.path`` when it is set. Otherwise, load the default package
    from NGC. The package class and default loader are passed in to keep this
    function easy to test.
    """
    model_path = model_cfg.get("path")
    if model_path:
        return package_cls(
            model_path,
            cache_options={
                "cache_storage": package_cls.default_cache("sfno"),
                "same_names": True,
            },
        )
    return default_loader()


def _publish_newest_loadable(
    init: datetime,
    earliest: datetime,
    is_fixed: bool,
    last_init: datetime | None,
    attempt: Callable[[datetime], None],
) -> datetime | None:
    """Try to publish the newest available GFS forecast.

    First try the requested cycle. If its data is not ready, try the cycle
    from six hours earlier. Keep moving backward until a cycle publishes, the
    allowed time range is exhausted, or an already published cycle is reached.
    If the user requested an exact cycle, do not try older ones.

    Return the published cycle, or ``None`` if none could be published.
    Unexpected errors are passed to the caller.
    """
    cyc = init
    while True:
        try:
            attempt(cyc)
            return cyc
        except _CycleNotReady as exc:
            prev = cyc - timedelta(hours=_GFS_CYCLE_HOURS)
            # Log the underlying error so a real fault masquerading as "not ready" stays visible.
            if (
                is_fixed
                or prev < earliest
                or (last_init is not None and prev <= last_init)
            ):
                logger.warning("GFS IC {} not ready ({}); will retry", cyc, exc)
                return None
            logger.warning(
                "GFS cycle {} not ready ({}); trying previous {}", cyc, exc, prev
            )
            cyc = prev


def _validate_ensemble(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate ensemble settings and return normalized values.

    Missing settings or ``members: 1`` select a deterministic forecast.
    More than one member enables the fixed experimental perturbation.
    """
    ens = cfg.get("ensemble")
    if ens is None:
        ens = {}
    elif not isinstance(ens, dict):
        raise ValueError(f"ensemble must be a mapping, got {type(ens).__name__}")
    # Reject unknown keys so a typo such as `member:` raises an error instead
    # of silently selecting a deterministic forecast.
    allowed = {"members", "batch_size", "seed", "include_members"}
    unknown = set(ens) - allowed
    if unknown:
        raise ValueError(
            f"unknown ensemble config key(s) {sorted(unknown)}; allowed: {sorted(allowed)}"
        )
    members = ens.get("members", 1)
    if not isinstance(members, int) or isinstance(members, bool) or members < 1:
        raise ValueError(f"ensemble.members must be an integer >= 1, got {members!r}")
    include_members = ens.get("include_members", False)
    if not isinstance(include_members, bool):
        raise ValueError(
            f"ensemble.include_members must be true or false, got {include_members!r}"
        )
    # Validate seed / batch_size whenever supplied, even for members == 1, so an invalid value is
    # never silently ignored (batch_size only makes sense in [1, members]).
    seed = ens.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError(f"ensemble.seed must be an integer or omitted, got {seed!r}")
    batch_size = ens.get("batch_size", members)
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or not (1 <= batch_size <= members)
    ):
        raise ValueError(
            f"ensemble.batch_size must be an integer between 1 and {members}, "
            f"got {batch_size!r}"
        )
    if members == 1:
        return {"members": 1, "include_members": include_members}
    return {
        "members": members,
        "include_members": include_members,
        "batch_size": batch_size,
        "seed": seed,
    }


def _build_perturbation(model: Any) -> Any:
    """Build the experimental z500 perturbation used for ensemble forecasts.

    The amplitude 39.27 is borrowed from
    ``examples/02_medium_range/03_huge_ensembles.py``, where it represents
    0.35 times the z500 skill. This recipe does not implement HENS or provide
    calibrated probabilistic guidance.

    Only z500 receives a nonzero amplitude. The library defaults provide the
    correlation scales. Raise ``ValueError`` if the model does not use z500.
    """
    vars_in = [str(v) for v in model.input_coords()["variable"]]
    if _PERTURB_VARIABLE not in vars_in:
        raise ValueError(
            f"the experimental perturbation targets {_PERTURB_VARIABLE!r}, which is not among "
            f"the model's input variables; edit _build_perturbation for this model"
        )
    # Import lazily because only ensemble forecasts need this perturbation code.
    import torch

    from earth2studio.perturbation import CorrelatedSphericalGaussian

    amplitude = torch.tensor(
        [_PERTURB_AMPLITUDE if v == _PERTURB_VARIABLE else 0.0 for v in vars_in],
        dtype=torch.float32,
    ).reshape(len(vars_in), 1, 1)
    return CorrelatedSphericalGaussian(noise_amplitude=amplitude)


def _forecast_loop(
    init_cfg: str,
    max_consecutive_failures: int,
    max_lookback: int,
    model: Any,
    data: Any,
    collector: SFNOCollector,
    coordinator: DSXCoordinator,
    publisher: DSXPublisher,
    transport: BusTransport | None,
    device: Any,
    nsteps: int,
    ensemble: dict[str, Any],
    poll_interval_s: float,
    one_shot: bool,
    stop: threading.Event,
    cache_retention_hours: float = 0,
) -> None:
    """Repeatedly check for new GFS cycles and publish them."""
    # Import here so tests can replace these Earth2Studio modules.
    import earth2studio.run as run
    from earth2studio.utils.time import to_time_array

    members = ensemble["members"]
    # Perturbation is built once (members > 1); deterministic runs need none.
    perturbation = _build_perturbation(model) if members > 1 else None
    # Set the seed once before processing cycles. Setting it for every cycle
    # would repeat the same perturbations.
    if members > 1 and ensemble["seed"] is not None:
        import torch

        torch.manual_seed(ensemble["seed"])

    fixed_init: datetime | None = None
    if init_cfg != "latest":
        fixed_init = datetime.fromisoformat(init_cfg)
        # Convert the requested time to UTC. Treat a time without a timezone as UTC.
        fixed_init = (
            fixed_init.replace(tzinfo=timezone.utc)
            if fixed_init.tzinfo is None
            else fixed_init.astimezone(timezone.utc)
        )

    def _attempt(cyc: datetime) -> None:
        """Run, stage, and publish one GFS cycle."""
        times = to_time_array([cyc.strftime("%Y-%m-%dT%H:%M:%S")])
        # Start a fresh cycle and discard values left by the previous attempt.
        collector.begin_cycle(_epoch_ms(times[0]), member_count=members)
        logger.info("sfno: init={} nsteps={} members={}", cyc, nsteps, members)
        try:
            if members == 1:
                run.deterministic(
                    times,
                    nsteps,
                    model,
                    data,
                    collector,
                    output_coords=OrderedDict({"variable": np.array(SOURCE_VARS)}),
                    device=device,
                )
            else:
                if perturbation is None:  # built whenever members > 1
                    raise RuntimeError("ensemble perturbation was not built")
                run.ensemble(
                    times,
                    nsteps,
                    members,
                    model,
                    data,
                    collector,
                    perturbation=perturbation,
                    batch_size=ensemble["batch_size"],
                    output_coords=OrderedDict({"variable": np.array(SOURCE_VARS)}),
                    device=device,
                )
        except FileNotFoundError as exc:
            # The newest GFS cycle may be listed before all its files are uploaded.
            # We cannot distinguish that from other missing files during inference,
            # so any missing file triggers an attempt with an older cycle. Missing
            # files during staging or publishing remain normal errors.
            raise _CycleNotReady(str(exc)) from exc
        coordinator.stage_forecasts(collector.collect())
        collector.clear_buffer()
        if not publish_with_reconnect(
            publisher, coordinator, transport, one_shot, stop
        ):
            raise SystemExit(
                "bus publish failed after reconnect retries; exiting for supervisor restart"
            )

    last_init: datetime | None = None
    idle_logged = False
    consecutive_failures = 0
    published_any = False
    while True:
        # Finish any active forecast, but do not start another after shutdown.
        if stop.is_set():
            break
        try:
            now = datetime.now(timezone.utc)
            if fixed_init is not None:
                init: datetime | None = fixed_init
            else:
                init = resolve_latest(
                    data.available, now, _GFS_CYCLE_HOURS, max_lookback
                )

            if init is None:
                logger.warning(
                    "no GFS cycle available within {}h; retrying", max_lookback
                )
            elif last_init is None or init > last_init:
                # Use the same lower time limit as resolve_latest. Basing it on
                # init could search too far back when init is already old.
                earliest = floor_to_cycle(now, _GFS_CYCLE_HOURS) - timedelta(
                    hours=max_lookback
                )
                used = _publish_newest_loadable(
                    init, earliest, fixed_init is not None, last_init, _attempt
                )
                if used is not None:
                    last_init = used  # Update only after successful publication.
                    published_any = True
                    logger.info("cycle published (initTime={})", collector.init_ms)
                    prune_data_cache(cache_retention_hours, ("gfs",))
            elif fixed_init is not None and not idle_logged:
                logger.info(
                    "fixed init {} already published; idling (use --once to exit)",
                    fixed_init,
                )
                idle_logged = True
        except Exception:
            # Retry temporary errors. One-shot mode raises immediately; repeated
            # failures exit so the supervisor can restart the process.
            logger.exception("forecast cycle failed")
            if one_shot:
                raise
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                raise SystemExit(
                    f"{consecutive_failures} consecutive cycle failures; exiting for "
                    "supervisor restart"
                )
        else:
            consecutive_failures = 0
        if one_shot or stop.wait(poll_interval_s):
            break

    # A one-shot run must fail if it did not publish a forecast.
    if one_shot and not published_any:
        raise SystemExit(
            "no forecast produced (no loadable init within lookback); exiting nonzero"
        )


def run(cfg: dict[str, Any], args: Any, stop: threading.Event) -> None:
    """Load SFNO and publish forecasts and metadata to DSX.

    Shutdown is cooperative: an active model load or forecast finishes before
    the function stops.
    """
    one_shot = args.once or args.dry_run
    sites = validate_and_normalize_sites(cfg["sites"])
    run_cfg = cfg["run"]

    # Validate all settings before connecting to the broker or loading the model.
    init_cfg = run_cfg["init_time"]  # "latest" or an ISO timestamp
    if not isinstance(init_cfg, str):
        raise ValueError(
            "run.init_time must be 'latest' or an ISO timestamp, " f"got {init_cfg!r}"
        )
    if init_cfg != "latest":
        try:
            datetime.fromisoformat(init_cfg)
        except ValueError as exc:
            raise ValueError(
                "run.init_time must be 'latest' or a valid ISO timestamp, "
                f"got {init_cfg!r}"
            ) from exc
    max_lookback = run_cfg.get("max_lookback_hours", 24)
    if (
        not isinstance(max_lookback, int)
        or isinstance(max_lookback, bool)
        or max_lookback < 0
    ):
        raise ValueError(
            "run.max_lookback_hours must be a non-negative integer, "
            f"got {max_lookback!r}"
        )
    max_consecutive_failures = run_cfg.get("max_consecutive_failures", 5)
    if (
        not isinstance(max_consecutive_failures, int)
        or isinstance(max_consecutive_failures, bool)
        or max_consecutive_failures < 1
    ):
        raise ValueError(
            "run.max_consecutive_failures must be a positive integer, "
            f"got {max_consecutive_failures!r}"
        )
    nsteps = run_cfg["nsteps"]
    if not isinstance(nsteps, int) or isinstance(nsteps, bool) or nsteps < 0:
        raise ValueError(f"run.nsteps must be a non-negative integer, got {nsteps!r}")
    ensemble = _validate_ensemble(cfg)
    cache_retention_hours = resolve_cache_retention_hours(
        run_cfg, float(max_lookback + _GFS_CYCLE_HOURS)
    )

    # This controls how often the workflow checks for a new GFS cycle.
    poll_interval_s = run_cfg.get("poll_interval_seconds", 900)
    if (
        not isinstance(poll_interval_s, (int, float))
        or isinstance(poll_interval_s, bool)
        or not math.isfinite(poll_interval_s)
        or poll_interval_s <= 0
    ):
        raise ValueError(
            f"run.poll_interval_seconds must be a positive, finite number, got {poll_interval_s!r}"
        )

    # Republish about every 100 seconds, as in the BMS contract. The 90-second default
    # leaves some room for scheduling and network delays.
    check_topics_config(cfg.get("topics") or {})
    if "metadata_heartbeat_seconds" in cfg["bus"]:
        raise ValueError(
            "bus.metadata_heartbeat_seconds was renamed to bus.heartbeat_seconds"
        )
    heartbeat_s = cfg["bus"].get("heartbeat_seconds", 90)
    if (
        not isinstance(heartbeat_s, (int, float))
        or isinstance(heartbeat_s, bool)
        or not (0 < heartbeat_s <= 100)
    ):
        raise ValueError("bus.heartbeat_seconds must be a number in (0, 100]")

    # Import model libraries only after all lightweight configuration checks.
    import torch

    from earth2studio.data import GFS
    from earth2studio.models.auto import Package
    from earth2studio.models.px import SFNO

    # Connect before the slow model load so broker errors fail quickly.
    transport = build_transport(cfg["bus"], args.dry_run)
    heartbeat_thread: threading.Thread | None = None
    try:
        pkg = _resolve_package(cfg["model"], Package, SFNO.load_default_package)
        logger.info(
            "SFNO package: {}", cfg["model"].get("path") or f"default ({pkg.root})"
        )
        logger.info("loading SFNO...")
        model = SFNO.load_model(pkg)

        schema = WeatherSchema.load()
        # Use SFNO's global latitude/longitude grid to find the nearest site cells.
        ic = model.input_coords()
        extractor = RegularLatLonSiteExtractor(ic["lat"], ic["lon"], sites)

        # Advertise the latest forecast time that this fixed-step model produces.
        horizon_seconds = nsteps * _NATIVE_STEP_SECONDS
        model_id = cfg["model"].get("id", "sfno")

        collector = SFNOCollector(extractor, model_id, init_ms=0)
        publisher = DSXPublisher(transport, schema, dry_run=args.dry_run)
        coordinator = DSXCoordinator(
            publisher,
            schema,
            TOPIC_PREFIX,
            cfg["topics"].get("product", "global-medium-range-weather"),
            {s["id"]: s for s in sites},
            model_id,
            horizon_seconds=horizon_seconds,
            # GFS normally provides a new forecast every six hours.
            cadence_seconds=_GFS_CYCLE_HOURS * 60 * 60,
            inputs=cfg.get("inputs", []),
            variables=VARIABLES,
            include_members=ensemble["include_members"],
        )

        # Publish metadata first so consumers know the units and site locations.
        # Do not publish forecasts when their metadata could not be published.
        if not coordinator.publish_metadata():
            raise SystemExit(
                "initial retained metadata publish failed; exiting for supervisor restart"
            )

        if transport is not None and not one_shot:

            def _heartbeat() -> None:
                """Republish metadata and the latest forecasts every heartbeat_s."""
                while not stop.wait(heartbeat_s):
                    try:
                        republish_on_heartbeat(coordinator, publisher, transport, stop)
                    except Exception:
                        logger.exception("heartbeat failed")

            heartbeat_thread = threading.Thread(
                target=_heartbeat, name="dsx-heartbeat", daemon=True
            )
            heartbeat_thread.start()
            logger.info("heartbeat every {}s", heartbeat_s)

        # Reuse one GFS source so all cycles share its downloaded-file cache.
        data = GFS()
        device = torch.device(cfg["model"].get("device", "cuda"))
        _forecast_loop(
            init_cfg,
            max_consecutive_failures,
            max_lookback,
            model,
            data,
            collector,
            coordinator,
            publisher,
            transport,
            device,
            nsteps,
            ensemble,
            poll_interval_s,
            one_shot,
            stop,
            cache_retention_hours,
        )
    finally:
        stop.set()
        if heartbeat_thread is not None:
            # Wait for the heartbeat to finish before closing its connection.
            heartbeat_thread.join()
        if transport is not None:
            # Give QoS-0 messages a brief, best-effort chance to leave the client.
            time.sleep(0.5)
            transport.close()
    logger.info("done.")

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

"""Run StormCast CONUS forecasts and publish results to DSX.

StormCast starts from hourly HRRR data. For each six-hour GFS cycle, this
workflow runs a global conditioning model—SFNO by default or FCN3 when
configured—and interpolates its output to hourly steps. That saved conditioning
is reused across the cycle's hourly HRRR initializations, and is rebuilt only
when a newer GFS cycle becomes available.

``run(cfg, args, stop)`` is the entry point. ``run.init_time: latest`` selects
the newest available HRRR cycle, while a specific date and time selects an
exact cycle. StormCast-specific setup lives here, while publishing and
reconnection use the shared DSX components.
"""

from __future__ import annotations

import math
import pathlib
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

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
    CurvilinearSiteExtractor,
    validate_and_normalize_sites,
)
from .collector import SOURCE_VARS, StormCastCollector
from .coverage import conditioning_covers, conditioning_hours
from .variables import VARIABLES

if TYPE_CHECKING:
    import torch

    from earth2studio.data import DataSource, InferenceOutputSource
    from earth2studio.models.px import InterpModAFNO, StormCastCONUS

# Import conditioning helpers inside functions so lightweight tests can load
# this module without importing torch or xarray.

# StormCast advances the forecast by one hour per step.
_NATIVE_STEP_SECONDS = 3600

# StormCast normally publishes once per hourly HRRR analysis. This advertised
# schedule is separate from polling and actual delivery times.
_ISSUE_CADENCE_SECONDS = 3600

# Keep one extra six-hour GFS cycle when cleaning the input cache.
_CACHE_SAFETY_HOURS = 6
_GFS_CYCLE_HOURS = 6

# StormCast model-grid limits used to validate crops before loading the model.
# These values mirror StormCastCONUS and must remain in sync with it.
_HRRR_LAT_RANGE = (17, 1041)
_HRRR_LON_RANGE = (3, 1795)
_PATCH = 4
# The full model grid has 1024 x 1792 cells, with one token per patch.
_POS_TOKENS = (1024 // _PATCH, 1792 // _PATCH)
_MIN_CROP = 128  # Connector safety limit, not a model requirement.


@dataclass(frozen=True)
class _WorkflowSettings:
    """Validated settings used to run the StormCast workflow."""

    fixed_init: datetime | None
    conditioning_output_path: str
    conditioning_kind: str
    model_device: str | torch.device
    conditioning_device: str | torch.device
    max_lookback: int
    max_consecutive_failures: int
    cache_retention_hours: float
    nsteps: int
    poll_interval_s: float
    heartbeat_s: float
    one_shot: bool


def _epoch_ms(dt64: np.datetime64) -> int:
    """Convert a NumPy datetime to milliseconds since the Unix epoch."""
    return int(np.asarray(dt64).astype("datetime64[ms]").astype("int64"))


def _resolve_num_diffusion_steps(perf: dict[str, Any]) -> int | None:
    """Validate the optional number of diffusion steps.

    Return ``None`` to use StormCast's default of 18. Otherwise, require an
    integer of at least 2. The scheduler calculates intervals between steps,
    so one step is not valid.
    """
    value = perf.get("num_diffusion_steps")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"performance.num_diffusion_steps must be an integer, got {value!r}"
        )
    if value < 2:
        raise ValueError(
            f"performance.num_diffusion_steps must be at least 2, got {value}"
        )
    return value


def _parse_workflow_settings(cfg: dict[str, Any], args: Any) -> _WorkflowSettings:
    """Validate lightweight configuration before importing or loading models."""
    conditioning_value = cfg.get("conditioning")
    if conditioning_value is None:
        conditioning_cfg: dict[str, Any] = {}
    elif isinstance(conditioning_value, dict):
        conditioning_cfg = conditioning_value
    else:
        raise ValueError("conditioning must be a mapping")

    conditioning_kind = conditioning_cfg.get("model", "sfno")
    if conditioning_kind not in {"sfno", "fcn3"}:
        raise ValueError(
            "conditioning.model must be 'sfno' or 'fcn3', " f"got {conditioning_kind!r}"
        )

    model_device = cfg["model"].get("device", "cuda")
    if not isinstance(model_device, str) or not model_device.strip():
        raise ValueError(
            f"model.device must be a non-empty string, got {model_device!r}"
        )
    conditioning_device = conditioning_cfg.get("device", model_device)
    if not isinstance(conditioning_device, str) or not conditioning_device.strip():
        raise ValueError(
            "conditioning.device must be a non-empty string, "
            f"got {conditioning_device!r}"
        )

    max_lookback = conditioning_cfg.get("max_lookback_hours", 12)
    if (
        not isinstance(max_lookback, int)
        or isinstance(max_lookback, bool)
        or max_lookback < 0
    ):
        raise ValueError(
            "conditioning.max_lookback_hours must be a non-negative integer, "
            f"got {max_lookback!r}"
        )

    scratch_dir_value = conditioning_cfg.get("scratch_dir")
    if scratch_dir_value is None:
        scratch_dir = pathlib.Path(tempfile.gettempdir())
    elif isinstance(scratch_dir_value, str) and scratch_dir_value.strip():
        scratch_dir = pathlib.Path(scratch_dir_value)
    else:
        raise ValueError(
            "conditioning.scratch_dir must be a non-empty path string, "
            f"got {scratch_dir_value!r}"
        )
    conditioning_output_path = str(scratch_dir / "dsx_conditioning.nc")

    run_cfg = cfg["run"]
    init_cfg = run_cfg["init_time"]
    if not isinstance(init_cfg, str):
        raise ValueError(
            f"run.init_time must be 'latest' or a date and time, got {init_cfg!r}"
        )
    fixed_init: datetime | None = None
    if init_cfg != "latest":
        try:
            fixed_init = datetime.fromisoformat(init_cfg)
        except ValueError as exc:
            raise ValueError(
                "run.init_time must be 'latest' or a valid date and time, "
                f"got {init_cfg!r}"
            ) from exc
        fixed_init = (
            fixed_init.replace(tzinfo=timezone.utc)
            if fixed_init.tzinfo is None
            else fixed_init.astimezone(timezone.utc)
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

    poll_interval_s = run_cfg.get("poll_interval_seconds", 600)
    if (
        not isinstance(poll_interval_s, (int, float))
        or isinstance(poll_interval_s, bool)
        or not math.isfinite(poll_interval_s)
        or poll_interval_s <= 0
    ):
        raise ValueError(
            "run.poll_interval_seconds must be a positive, finite number, "
            f"got {poll_interval_s!r}"
        )

    check_topics_config(cfg.get("topics") or {})
    if "metadata_heartbeat_seconds" in cfg["bus"]:
        raise ValueError(
            "bus.metadata_heartbeat_seconds was renamed to bus.heartbeat_seconds"
        )
    heartbeat_s = cfg["bus"].get("heartbeat_seconds", 90)
    if (
        not isinstance(heartbeat_s, (int, float))
        or isinstance(heartbeat_s, bool)
        or not math.isfinite(heartbeat_s)
        or not (0 < heartbeat_s <= 100)
    ):
        raise ValueError("bus.heartbeat_seconds must be a finite number in (0, 100]")

    cache_retention_hours = resolve_cache_retention_hours(
        run_cfg, float(max_lookback + _CACHE_SAFETY_HOURS)
    )
    return _WorkflowSettings(
        fixed_init=fixed_init,
        conditioning_output_path=conditioning_output_path,
        conditioning_kind=conditioning_kind,
        model_device=model_device,
        conditioning_device=conditioning_device,
        max_lookback=max_lookback,
        max_consecutive_failures=max_consecutive_failures,
        cache_retention_hours=cache_retention_hours,
        nsteps=nsteps,
        poll_interval_s=float(poll_interval_s),
        heartbeat_s=float(heartbeat_s),
        one_shot=bool(args.once or args.dry_run),
    )


def _validate_subregion_box(
    lat_lim: tuple[int, int], lon_lim: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Validate that a crop fits the StormCast grid and model requirements.

    Each axis must stay inside the HRRR domain, align with the model's patch
    grid, meet the minimum crop size, and fit inside the position-embedding
    grid. Return the crop unchanged when it is valid.
    """
    for axis, (start, end), (domain_start, domain_end), token_count in (
        ("lat", lat_lim, _HRRR_LAT_RANGE, _POS_TOKENS[0]),
        ("lon", lon_lim, _HRRR_LON_RANGE, _POS_TOKENS[1]),
    ):
        extent = end - start
        if not (domain_start <= start < end <= domain_end):
            raise ValueError(
                f"subregion {axis} {(start, end)} must satisfy "
                f"{domain_start} <= start < end <= {domain_end}"
            )
        if (start - domain_start) % _PATCH or extent % _PATCH:
            raise ValueError(
                f"subregion {axis} offset-from-origin and extent must be "
                f"divisible by {_PATCH}"
            )
        if extent < _MIN_CROP:
            raise ValueError(
                f"subregion {axis} extent {extent} is below the "
                f"{_MIN_CROP}-cell minimum"
            )
        if (end - domain_start) // _PATCH > token_count:
            raise ValueError(
                f"subregion {axis} upper index {end} is too close to the domain "
                "edge for this model's cropping (pos-embed overflow); move the "
                "site inward or use full CONUS"
            )
    return lat_lim, lon_lim


def _derive_subregion_box(
    y: int, x: int, size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Create a valid StormCast crop around a grid cell.

    The requested crop is centered as closely as possible on ``(y, x)``.
    Its boundaries are aligned with the model's patch grid. If it would
    extend beyond the HRRR domain, it is moved inward while keeping the
    requested size.

    Return the latitude and longitude index ranges for the crop. Raise
    ``ValueError`` if the center or requested size cannot produce a valid crop.
    """

    def _axis_bounds(
        center: int,
        extent: int,
        domain_start: int,
        domain_end: int,
        axis: str,
    ) -> tuple[int, int]:
        domain_size = domain_end - domain_start
        if extent <= 0 or extent % _PATCH or extent > domain_size:
            raise ValueError(
                f"subregion size ({axis}) must be a positive multiple of "
                f"{_PATCH} and no greater than {domain_size}, got {extent}"
            )
        if not domain_start <= center < domain_end:
            raise ValueError(
                f"subregion center ({axis}={center}) is outside the model "
                f"domain [{domain_start}, {domain_end})"
            )

        crop_start = center - extent // 2
        # Align the crop's starting point with the model's patch grid.
        crop_start -= (crop_start - domain_start) % _PATCH
        # Move the crop inside the domain if it crosses an edge.
        crop_start = max(domain_start, min(crop_start, domain_end - extent))
        return crop_start, crop_start + extent

    lat_lim = _axis_bounds(y, size[0], *_HRRR_LAT_RANGE, "lat")
    lon_lim = _axis_bounds(x, size[1], *_HRRR_LON_RANGE, "lon")
    return _validate_subregion_box(lat_lim, lon_lim)


def _nearest_hrrr_index(lat: float, lon: float) -> tuple[int, int]:
    """Find the HRRR grid cell nearest to a latitude and longitude.

    Return the cell as ``(y, x)`` grid indices. Longitude may use either the
    -180 to 180 or 0 to 360 convention.
    """
    if not (np.isfinite(lat) and np.isfinite(lon)):
        raise ValueError(f"coordinates must be finite, got ({lat}, {lon})")
    if not -90.0 <= lat <= 90.0:
        raise ValueError(f"latitude must be between -90 and 90, got {lat}")

    # Import lazily so lightweight validation tests do not require Earth2Studio.
    from earth2studio.data import HRRR

    grid_lat, grid_lon = HRRR.grid()
    grid_lat = np.deg2rad(np.asarray(grid_lat, dtype=float))
    grid_lon = np.deg2rad(np.asarray(grid_lon, dtype=float) % 360.0)
    query_lat = np.deg2rad(lat)
    query_lon = np.deg2rad(lon % 360.0)

    # Convert both points to a unit sphere. The shortest straight-line
    # distance between them identifies the nearest point on the surface.
    distance_squared = (
        (np.cos(grid_lat) * np.cos(grid_lon) - np.cos(query_lat) * np.cos(query_lon))
        ** 2
        + (np.cos(grid_lat) * np.sin(grid_lon) - np.cos(query_lat) * np.sin(query_lon))
        ** 2
        + (np.sin(grid_lat) - np.sin(query_lat)) ** 2
    )
    y, x = np.unravel_index(int(np.argmin(distance_squared)), grid_lat.shape)
    return int(y), int(x)


def _resolve_conditioning(
    cond_model: InterpModAFNO,
    gfs_source: DataSource,
    target_cycle: datetime,
    current_cycle: datetime | None,
    nsteps: int,
    out_path: str,
    device: str | torch.device,
    max_lookback: int,
    kind: str,
) -> tuple[InferenceOutputSource, datetime] | None:
    """Build conditioning from the newest usable GFS cycle.

    Try ``target_cycle`` first. If its required GFS files are not ready, try
    older six-hour cycles until reaching ``current_cycle`` or the lookback
    limit.

    Parameters
    ----------
    cond_model : InterpModAFNO
        Hourly SFNO or FCN3 conditioning model.
    gfs_source : DataSource
        GFS source used to initialize the conditioning model.
    target_cycle : datetime
        Newest six-hour GFS cycle to try.
    current_cycle : datetime or None
        Cycle currently in use, if one has already been loaded.
    nsteps : int
        Number of StormCast forecast steps.
    out_path : str
        Path where the conditioning forecast is written.
    device : str or torch.device
        Device used to run the conditioning model.
    max_lookback : int
        Maximum number of hours to search backward.
    kind : str
        Conditioning model name used in log messages.

    Returns
    -------
    tuple[InferenceOutputSource, datetime] or None
        The new conditioning source paired with its GFS cycle, or ``None`` when
        no newer usable cycle is found.
    """
    from .conditioning import (
        _ConditioningDataNotReady,
        run_conditioning,
    )

    candidate_cycle = target_cycle
    earliest = target_cycle - timedelta(hours=max_lookback)
    hours = conditioning_hours(nsteps)
    while candidate_cycle >= earliest:
        if current_cycle is not None and candidate_cycle <= current_cycle:
            # No newer cycle could be loaded, so keep the current conditioning.
            return None
        logger.info("conditioning {}: cycle={} hours={}", kind, candidate_cycle, hours)
        try:
            source = run_conditioning(
                cond_model, gfs_source, candidate_cycle, hours, out_path, device
            )
            return source, candidate_cycle
        except _ConditioningDataNotReady:
            logger.warning("GFS cycle {} not ready; trying previous", candidate_cycle)
            candidate_cycle -= timedelta(hours=_GFS_CYCLE_HOURS)
    return None


class _ConditioningSourceManager:
    """Close conditioning data sources when they are replaced or no longer needed."""

    def __init__(self) -> None:
        self.current: InferenceOutputSource | None = None

    def replace(self, source: InferenceOutputSource) -> None:
        """Store a new source and close the previous one."""
        previous = self.current
        self.current = source
        if previous is not None:
            try:
                previous.da.close()
            except Exception:
                logger.opt(exception=True).debug("closing previous conditioning failed")

    def close(self) -> None:
        """Close the current source."""
        if self.current is None:
            return
        try:
            self.current.da.close()
        except Exception:
            logger.opt(exception=True).debug("closing conditioning at shutdown failed")
        finally:
            self.current = None


@dataclass(frozen=True)
class _CycleOutcome:
    """State returned after processing one StormCast initialization."""

    conditioning_cycle: datetime | None
    published: bool


def _run(
    settings: _WorkflowSettings,
    model: StormCastCONUS,
    hrrr: DataSource,
    collector: StormCastCollector,
    coordinator: DSXCoordinator,
    publisher: DSXPublisher,
    transport: BusTransport | None,
    conditioning_sources: _ConditioningSourceManager,
    stop: threading.Event,
) -> None:
    """Run StormCast when a new hourly HRRR analysis becomes available.

    When new hourly HRRR data is available, find the related six-hour GFS
    cycle. If conditioning has not already been created from that GFS cycle,
    create it before running StormCast.

    A fixed initialization processes one historical time. ``"latest"`` keeps
    checking for new HRRR data and processes each new time once.

    Parameters
    ----------
    settings : _WorkflowSettings
        Validated workflow settings.
    model : StormCastCONUS
        Loaded StormCast model. Its conditioning source is updated here.
    hrrr : DataSource
        HRRR data used to initialize StormCast.
    collector : StormCastCollector
        Collects model output for the current forecast cycle.
    coordinator : DSXCoordinator
        Validates and stages forecasts for publishing.
    publisher : DSXPublisher
        Queues and publishes forecast and metadata messages.
    transport : BusTransport or None
        DSX bus connection, or ``None`` in dry-run mode.
    conditioning_sources : _ConditioningSourceManager
        Owns the open conditioning source and closes replaced sources.
    stop : threading.Event
        Signals that the workflow should stop.
    """
    import earth2studio.run as run
    from earth2studio.data import GFS, HRRR
    from earth2studio.utils.time import to_time_array

    from .conditioning import build_conditioning_model  # deferred: imports torch/xarray

    logger.info(
        "building {} conditioning model on {}...",
        settings.conditioning_kind,
        settings.conditioning_device,
    )
    cond_model = build_conditioning_model(
        settings.conditioning_device, settings.conditioning_kind
    )
    gfs_source = GFS()
    fixed_init = settings.fixed_init
    max_lookback = settings.max_lookback
    nsteps = settings.nsteps
    one_shot = settings.one_shot

    def _process_init(
        stormcast_init: datetime,
        conditioning_cycle: datetime | None,
    ) -> _CycleOutcome:
        """Prepare conditioning, run StormCast, and publish one initialization."""
        target_cycle = floor_to_cycle(stormcast_init, _GFS_CYCLE_HOURS)
        if conditioning_cycle is None or target_cycle > conditioning_cycle:
            resolved = _resolve_conditioning(
                cond_model,
                gfs_source,
                target_cycle,
                conditioning_cycle,
                nsteps,
                settings.conditioning_output_path,
                settings.conditioning_device,
                max_lookback,
                settings.conditioning_kind,
            )
            # A returned source is always paired with its cycle (or the whole result is None),
            # so a source can never be adopted without being registered for cleanup.
            if resolved is not None:
                conditioning_source, resolved_cycle = resolved
                model.conditioning_data_source = conditioning_source
                conditioning_sources.replace(conditioning_source)
                conditioning_cycle = resolved_cycle

        if conditioning_cycle is None:
            logger.warning(
                "no GFS cycle for init {} within {}h; skipping",
                stormcast_init,
                max_lookback,
            )
            return _CycleOutcome(None, False)

        if not conditioning_covers(conditioning_cycle, stormcast_init, nsteps):
            logger.warning(
                "conditioning from {} does not cover init {} +{}h; "
                "waiting for a newer GFS cycle",
                conditioning_cycle,
                stormcast_init,
                nsteps,
            )
            return _CycleOutcome(conditioning_cycle, False)

        stormcast_times = to_time_array([stormcast_init.strftime("%Y-%m-%dT%H:%M:%S")])
        # Set the initialization time and discard output left by a previous
        # failed attempt.
        collector.begin_cycle(_epoch_ms(stormcast_times[0]))
        logger.info(
            "stormcast: init={} nsteps={} cond={}",
            stormcast_init,
            nsteps,
            conditioning_cycle,
        )
        try:
            run.deterministic(
                stormcast_times,
                nsteps,
                model,
                hrrr,
                collector,
                output_coords=OrderedDict({"variable": np.array(SOURCE_VARS)}),
                device=settings.model_device,
            )
        except FileNotFoundError as exc:
            # A cycle may be listed before all input files are available. Any
            # missing file during inference waits for the next attempt.
            logger.warning(
                "input for StormCast init {} is not ready ({}); will retry",
                stormcast_init,
                exc,
            )
            return _CycleOutcome(conditioning_cycle, False)

        # Keep collected output until staging succeeds.
        coordinator.stage_forecasts(collector.collect())
        collector.clear_buffer()
        if not publish_with_reconnect(
            publisher, coordinator, transport, one_shot, stop
        ):
            # The cycle is not marked as published, so it runs again after
            # the process supervisor restarts the connector.
            raise SystemExit(
                "bus publish failed after reconnect retries; "
                "exiting for supervisor restart"
            )

        logger.info("cycle published (initTime={})", collector.init_ms)
        # Clean the input cache only after successful publication.
        prune_data_cache(settings.cache_retention_hours)
        return _CycleOutcome(conditioning_cycle, True)

    current_conditioning_cycle: datetime | None = None
    last_stormcast_init: datetime | None = None
    idle_logged = False
    consecutive_failures = 0
    published_any = False
    while True:
        # Cooperative shutdown: honor a stop set during the previous cycle's inference before
        # starting another. An in-flight rollout is not interrupted mid-cycle.
        if stop.is_set():
            break
        try:
            now = datetime.now(timezone.utc)
            # StormCast init: fixed (backfill) or the latest available hourly HRRR.
            if fixed_init is not None:
                stormcast_init: datetime | None = fixed_init
            else:
                stormcast_init = resolve_latest(HRRR.available, now, 1, max_lookback)

            if stormcast_init is None:
                logger.warning(
                    "no HRRR analysis available within {}h; retrying", max_lookback
                )
            elif last_stormcast_init is None or stormcast_init > last_stormcast_init:
                outcome = _process_init(stormcast_init, current_conditioning_cycle)
                current_conditioning_cycle = outcome.conditioning_cycle
                if outcome.published:
                    last_stormcast_init = stormcast_init
                    published_any = True
            elif fixed_init is not None and not idle_logged:
                logger.info(
                    "fixed init {} already published; idling (use --once to exit)",
                    fixed_init,
                )
                idle_logged = True
        except Exception:
            # Retry temporary cycle failures. Stop after repeated failures so
            # the process supervisor can restart the connector.
            logger.exception("forecast cycle failed")
            if one_shot:
                raise
            consecutive_failures += 1
            if consecutive_failures >= settings.max_consecutive_failures:
                raise SystemExit(
                    f"{consecutive_failures} consecutive cycle failures; exiting for "
                    "supervisor restart"
                )
        else:
            consecutive_failures = 0
        if one_shot or stop.wait(settings.poll_interval_s):
            break

    # A one-shot run (--once / --dry-run) that produced no forecast (no HRRR/GFS within lookback)
    # exits nonzero: an exit 0 with nothing published would look like success.
    if one_shot and not published_any:
        raise SystemExit(
            "no forecast produced (no usable init within lookback); exiting nonzero"
        )


def _resolve_model_package(
    model_cfg: dict, package_cls: Any, default_loader: Any
) -> Any:
    """Select the StormCast package: a local checkpoint (``model.path``) if set, else the default.

    An explicit ``model.path`` is honored as-is (an air-gapped / mounted checkpoint dir); a missing
    dir then fails loud in the loader rather than being silently swapped for the default. With no
    path, the published default package (Hugging Face) is used — the zero-config path. ``package_cls``
    and ``default_loader`` are injected so this stays torch-free and unit-testable.
    """
    model_path = model_cfg.get("path")
    if model_path:
        return package_cls(
            model_path,
            cache_options={
                "cache_storage": package_cls.default_cache("stormcast-conus"),
                "same_names": True,
            },
        )
    return default_loader()


def run(cfg: dict, args: Any, stop: threading.Event) -> None:
    """Load StormCast CONUS, then publish forecasts + retained metadata to the DSX bus.

    ``stop`` is the shared shutdown event (set by ``main.py``'s SIGTERM/SIGINT handler). Shutdown is
    COOPERATIVE: the loop checks ``stop`` at each cycle boundary (and the poll wait / heartbeat wake
    on it), so an in-flight model load or rollout runs to completion first rather than being
    interrupted; a SIGTERM received mid-inference takes effect at the next cycle boundary.
    """
    settings = _parse_workflow_settings(cfg, args)
    sites = validate_and_normalize_sites(cfg["sites"])

    import torch

    from earth2studio.data import HRRR
    from earth2studio.models.auto import Package
    from earth2studio.models.px import StormCastCONUS

    try:
        model_device = torch.device(settings.model_device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid model.device {settings.model_device!r}") from exc
    try:
        conditioning_device = torch.device(settings.conditioning_device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid conditioning.device {settings.conditioning_device!r}"
        ) from exc
    settings = replace(
        settings,
        model_device=model_device,
        conditioning_device=conditioning_device,
    )

    # Connect to the broker BEFORE the (multi-minute) model load, so a down broker fails
    # in seconds instead of after paying full load cost.
    transport = build_transport(cfg["bus"], args.dry_run)

    # A local checkpoint (model.path) if set, else the published default package (Hugging Face).
    pkg = _resolve_model_package(
        cfg["model"], Package, StormCastCONUS.load_default_package
    )
    logger.info(
        "StormCast CONUS package: {}",
        cfg["model"].get("path") or f"default ({pkg.root})",
    )
    load_kwargs: dict[str, Any] = {}
    sub = cfg.get("subregion") or {}
    if sub.get("enabled"):
        has_lat, has_lon = "hrrr_lat_lim" in sub, "hrrr_lon_lim" in sub
        if has_lat or has_lon:
            # Advanced override: an explicit grid-index box. Require BOTH (a lone one is a mistake).
            if not (has_lat and has_lon):
                raise ValueError(
                    "subregion: set BOTH hrrr_lat_lim and hrrr_lon_lim, or neither (center mode)"
                )
            for key in ("hrrr_lat_lim", "hrrr_lon_lim"):
                lim = sub[key]
                if not (
                    isinstance(lim, (list, tuple))
                    and len(lim) == 2
                    and all(isinstance(v, int) and not isinstance(v, bool) for v in lim)
                ):
                    raise ValueError(
                        f"subregion.{key} must be [low, high] integer grid indices, got {lim!r}"
                    )
            lat_lim, lon_lim = _validate_subregion_box(
                tuple(sub["hrrr_lat_lim"]), tuple(sub["hrrr_lon_lim"])
            )
            logger.info("subregion: explicit box lat={} lon={}", lat_lim, lon_lim)
        else:
            # Derive from a lat/lon center (defaults to the site centroid) + a size in grid cells.
            size = sub.get("size_cells", [512, 640])
            if not (
                isinstance(size, (list, tuple))
                and len(size) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in size)
            ):
                raise ValueError(
                    f"subregion.size_cells must be [height, width] integers, got {size!r}"
                )
            center = sub.get("center") or {}
            c_lat = float(center.get("lat", sum(s["lat"] for s in sites) / len(sites)))
            c_lon = float(center.get("lon", sum(s["lon"] for s in sites) / len(sites)))
            if not (np.isfinite(c_lat) and np.isfinite(c_lon)):
                raise ValueError(
                    f"subregion center must be finite, got ({c_lat}, {c_lon})"
                )
            y, x = _nearest_hrrr_index(c_lat, c_lon)
            lat_lim, lon_lim = _derive_subregion_box(y, x, (size[0], size[1]))
            logger.info(
                "subregion: center=({:.3f}, {:.3f}) -> cell ({}, {}); box lat={} lon={}",
                c_lat,
                c_lon,
                y,
                x,
                lat_lim,
                lon_lim,
            )
        # Every configured site must fall inside the crop (derived OR explicit): fail loud
        # before loading rather than silently sampling the crop edge for an out-of-box site.
        for s in sites:
            sy, sx = _nearest_hrrr_index(s["lat"], s["lon"])
            if not (lat_lim[0] <= sy < lat_lim[1] and lon_lim[0] <= sx < lon_lim[1]):
                raise ValueError(
                    f"site {s.get('id')!r} at cell ({sy}, {sx}) is outside the subregion crop "
                    f"lat={lat_lim} lon={lon_lim}; enlarge size_cells, set center, use an "
                    f"explicit box that contains it, or use full CONUS"
                )
        load_kwargs.update(hrrr_lat_lim=lat_lim, hrrr_lon_lim=lon_lim)
    # Opt-in speed knob (default off): forward num_diffusion_steps into the model. TF32 and
    # cudnn.benchmark showed no measurable gain in the short crop benchmark (the denoiser already
    # runs BF16 AMP), so they are not exposed. See cfg `performance:` for the fidelity trade-off.
    perf = cfg.get("performance") or {}
    steps = _resolve_num_diffusion_steps(perf)
    if steps is not None:
        load_kwargs["num_diffusion_steps"] = steps
        logger.info("performance: num_diffusion_steps={}", steps)
    logger.info("loading StormCast CONUS...")
    model = StormCastCONUS.load_model(pkg, **load_kwargs)

    schema = WeatherSchema.load()
    extractor = CurvilinearSiteExtractor(model.lat, model.lon, sites)
    # horizonSeconds advertises the maximum forecast lead. Keeping the native step in the workflow
    # (not config) prevents a config value from advertising a step different from the model's.
    # Each bundle's real leadSeconds come from its write; the coordinator checks every lead <= horizon.
    horizon_seconds = settings.nsteps * _NATIVE_STEP_SECONDS
    model_id = cfg["model"].get("id", "stormcast-conus")
    # StormCast collector (IOBackend) -> coordinator (convert/validate/stage + metadata) ->
    # the single publisher (queue/replay). init_ms is set per cycle from the HRRR init time.
    collector = StormCastCollector(extractor, model_id, init_ms=0)
    publisher = DSXPublisher(transport, schema, dry_run=args.dry_run)
    coordinator = DSXCoordinator(
        publisher,
        schema,
        TOPIC_PREFIX,
        cfg["topics"].get("product", "conus-site-weather"),
        {s["id"]: s for s in sites},
        model_id,
        horizon_seconds=horizon_seconds,
        # Nominal product issue cadence (workflow-owned), NOT the poll interval.
        cadence_seconds=_ISSUE_CADENCE_SECONDS,
        inputs=cfg.get("inputs", []),
        variables=VARIABLES,
    )

    # Publish retained metadata up front so a consumer connecting mid-inference already has the
    # units and locations (the heartbeat keeps it fresh). Forecasts depend on it: if metadata was
    # rejected, consumers can't interpret them, so exit nonzero for a supervisor restart.
    # (Dry-run prints the messages and returns True.)
    if not coordinator.publish_metadata():
        raise SystemExit(
            "initial retained metadata publish failed; exiting for supervisor restart"
        )

    hb_thread = None
    if transport and not settings.one_shot:

        def _heartbeat() -> None:
            """Republish metadata and the latest forecasts every heartbeat_s."""
            while not stop.wait(settings.heartbeat_s):
                try:
                    republish_on_heartbeat(coordinator, publisher, transport, stop)
                except Exception:
                    logger.exception("heartbeat failed")

        hb_thread = threading.Thread(
            target=_heartbeat, name="dsx-heartbeat", daemon=True
        )
        hb_thread.start()
        logger.info("heartbeat every {}s", settings.heartbeat_s)

    hrrr = HRRR()  # Construct once; the on-disk cache is shared across cycles.
    conditioning_sources = _ConditioningSourceManager()
    try:
        _run(
            settings,
            model,
            hrrr,
            collector,
            coordinator,
            publisher,
            transport,
            conditioning_sources,
            stop,
        )
    finally:
        stop.set()
        conditioning_sources.close()
        if hb_thread:
            # Join with no timeout so the heartbeat fully stops before close(): stop.set() ends it
            # after its current (publish-timeout-bounded) iteration. A timeout could let it publish
            # or reconnect after close(), on an already torn-down client.
            hb_thread.join()
        if transport:
            # Give queued QoS-0 messages time to be sent before close (QoS >= 1 is already
            # broker-confirmed).
            time.sleep(0.5)
            transport.close()
    logger.info("done.")

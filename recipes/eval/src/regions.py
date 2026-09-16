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

"""Named regional splits, shared by both scoring pathways.

The ``scoring.regions`` config block maps region names to rectangular
boxes on the scored grid.  A box maps spatial dimension names to
``[min, max]`` ranges in that dimension's coordinate values — ``lat``/
``lon`` degrees on a global grid, or projection/index coordinates on a
limited-area grid.  Dimensions left out of a box cover their full
extent.  A region may also be ``null`` (the whole grid) or a LIST of
boxes scored as their union (e.g. the extra-tropics as both
``|lat| >= 20`` bands).

Two special cases apply to geographic dimension names:

* ``lon`` compares on the [0, 360) circle — negative bounds mean degrees
  west, a box whose normalized min exceeds its max wraps across the
  dateline, and a span of 360 degrees or more means every longitude.
* ``lat`` bounds must lie in [-90, 90].

Events (``scoring.events``) pair such a region with a time window, so a
campaign can report skill for one named episode.  :func:`parse_events`
checks the block and :func:`scoring_regions` merges the event boxes into
the region set that both scoring pathways use.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from earth2studio.statistics.weights import lat_weight
from earth2studio.utils.type import CoordSystem

# Dimensions that never count as spatial when scanning a grid's
# coordinate system for its spatial axes.
NON_SPATIAL = frozenset({"batch", "time", "lead_time", "variable", "ensemble"})


def spatial_dims(spatial_coords: CoordSystem) -> list[str]:
    """Return the spatial dimension names of a coordinate system."""
    return [d for d in spatial_coords if d not in NON_SPATIAL]


def parse_regions(value: Any) -> dict[str, list[dict] | None] | None:
    """Check and normalize the ``scoring.regions`` block.

    Each region is ``null`` (whole grid), one box, or a LIST of boxes
    whose union defines the region.  A box maps spatial dimension names
    to ``[min, max]`` coordinate ranges; dimensions left out cover their
    full extent.  The parser turns single boxes into one-element lists so
    the mask builder handles one shape.  :func:`region_masks` checks box
    dimensions against the scored grid later, once the grid exists.
    """
    if value is None:
        return None
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "scoring.regions must be a non-empty mapping of "
            "name -> null | {<dim>: [min, max], ...} | list of such boxes "
            "(their union)."
        )

    def _one_box(name: str, spec: Any) -> dict:
        if not isinstance(spec, dict) or not spec:
            raise ValueError(
                f"Region '{name}' boxes must map spatial dimension names to "
                f"[min, max] ranges; got {spec!r}."
            )
        box: dict[str, list[float]] = {}
        for key, bounds_in in spec.items():
            try:
                bounds = [float(b) for b in bounds_in]
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Region '{name}': '{key}' must be [min, max]; "
                    f"got {bounds_in!r}."
                ) from err
            if len(bounds) != 2:
                raise ValueError(
                    f"Region '{name}': '{key}' must be [min, max]; "
                    f"got {bounds_in!r}."
                )
            box[str(key)] = bounds
        if "lat" in box:
            lat_lo, lat_hi = box["lat"]
            if not (-90.0 <= lat_lo < lat_hi <= 90.0):
                raise ValueError(
                    f"Region '{name}': lat bounds must satisfy "
                    f"-90 <= min < max <= 90; got {box['lat']}."
                )
        for key, (lo, hi) in box.items():
            # lon may wrap (min > max means crossing the dateline); every
            # other dimension is an ordinary interval.
            if key != "lon" and lo >= hi:
                raise ValueError(
                    f"Region '{name}': '{key}' must satisfy min < max; "
                    f"got [{lo}, {hi}]."
                )
        return box

    out: dict[str, list[dict] | None] = {}
    for name, spec in value.items():
        if spec is None:
            out[str(name)] = None
        elif isinstance(spec, list):
            if not spec:
                raise ValueError(f"Region '{name}': box list must be non-empty.")
            out[str(name)] = [_one_box(str(name), b) for b in spec]
        else:
            out[str(name)] = [_one_box(str(name), spec)]
    return out


def region_masks(
    spatial_coords: CoordSystem,
    regions: dict[str, list[dict] | None],
) -> OrderedDict[str, torch.Tensor]:
    """Compute each region's {0, 1} mask on the scored grid.

    Parameters
    ----------
    spatial_coords : CoordSystem
        Spatial coordinate arrays of the scored grid (1D per dimension).
    regions : dict[str, list[dict] | None]
        Parsed ``scoring.regions`` (see :func:`parse_regions`).

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Float64 mask of the full spatial shape per region, in config
        order.

    Raises
    ------
    ValueError
        If a box names a dimension the grid does not have, or a region
        selects no gridpoints.
    """
    dims = spatial_dims(spatial_coords)
    full_shape = [len(np.asarray(spatial_coords[d])) for d in dims]
    axes = {
        d: torch.tensor(np.asarray(spatial_coords[d]), dtype=torch.float64)
        for d in dims
    }

    def _box_mask(name: str, spec: dict) -> torch.Tensor:
        unknown = sorted(set(spec) - set(dims))
        if unknown:
            raise ValueError(
                f"Region '{name}' uses dimensions {unknown} that are not "
                f"spatial dimensions of the scored grid; got {dims}."
            )
        mask = torch.ones(full_shape, dtype=torch.float64)
        for key, (lo, hi) in spec.items():
            vals = axes[key]
            if key == "lon":
                # Longitudes compare on [0, 360); a box whose normalized
                # min exceeds its max wraps across the dateline/meridian,
                # and a span of >= 360 degrees means every longitude (a
                # [0, 360] bound must not normalize into an empty span).
                vals_n = vals % 360.0
                if hi - lo >= 360.0:
                    axis_mask = torch.ones_like(vals_n, dtype=torch.bool)
                else:
                    lon_lo, lon_hi = lo % 360.0, hi % 360.0
                    if lon_lo <= lon_hi:
                        axis_mask = (vals_n >= lon_lo) & (vals_n <= lon_hi)
                    else:
                        axis_mask = (vals_n >= lon_lo) | (vals_n <= lon_hi)
            else:
                axis_mask = (vals >= lo) & (vals <= hi)
            view = [1] * len(dims)
            view[dims.index(key)] = -1
            mask = mask * axis_mask.double().reshape(view)
        return mask

    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, spec in regions.items():
        if spec is None:
            mask = torch.ones(full_shape, dtype=torch.float64)
        else:
            # A region is the union of its boxes.  parse_regions
            # normalizes single boxes to one-element lists; accept a bare
            # box here too so direct callers keep working.
            boxes = spec if isinstance(spec, (list, tuple)) else [spec]
            mask = torch.zeros(full_shape, dtype=torch.float64)
            for box in boxes:
                mask = torch.maximum(mask, _box_mask(name, box))
        if spec is not None and not mask.any():
            raise ValueError(
                f"Region '{name}' selects no gridpoints on the scored "
                "grid — check its boxes."
            )
        out[name] = mask
    return out


def build_spatial_weights(
    spatial_coords: CoordSystem,
    lat_weights: bool,
    regions: dict[str, list[dict] | None] | None = None,
) -> torch.Tensor:
    """Build the spatial weight tensor for online reductions.

    Mirrors the offline scorer's weighting: cosine-latitude weights when
    ``scoring.lat_weights`` is true and the grid has a ``lat`` dimension,
    uniform weights otherwise.  Region-free, the returned tensor has one
    axis per spatial dimension so it broadcasts against
    ``[..., <spatial...>]`` tensors.  With ``regions`` configured the
    tensor becomes ``[region, <spatial...>]``: the same weights multiplied
    by each region's {0, 1} mask, evaluated on the actual scored grid so
    box edges land exactly on gridpoints.

    Parameters
    ----------
    spatial_coords : CoordSystem
        Spatial coordinate arrays of the scored grid.
    lat_weights : bool
        Whether to apply cosine-latitude weighting.
    regions : dict[str, list[dict] | None] | None
        Parsed ``scoring.regions`` (see :func:`parse_regions`).

    Returns
    -------
    torch.Tensor
        Float64 weights (broadcastable, or full-shaped per region).
    """
    dims = spatial_dims(spatial_coords)
    shape = [1] * len(dims)

    if lat_weights and "lat" in dims:
        lat_vals = np.asarray(spatial_coords["lat"])
        w = lat_weight(torch.tensor(lat_vals, dtype=torch.float64))
        shape[dims.index("lat")] = len(lat_vals)
        base = w.reshape(shape)
    else:
        base = torch.ones(shape, dtype=torch.float64)

    if regions is None:
        return base

    full_shape = [len(np.asarray(spatial_coords[d])) for d in dims]
    masks = region_masks(spatial_coords, regions)
    return torch.stack(
        [base.expand(full_shape) * mask for mask in masks.values()], dim=0
    )


# ---------------------------------------------------------------------------
# Events: a time window paired with a region
# ---------------------------------------------------------------------------

# Keys an event may carry.  The parser rejects anything else, so a
# misspelled key fails at config time instead of silently scoring the
# whole year.
_EVENT_KEYS = frozenset({"label", "start", "end", "region", "window", "ics"})
_EVENT_IC_KEYS = frozenset({"step_hours", "lookback_hours"})
# Which timestamp an event window filters: the valid time of each
# (initial condition, lead) pair, or the initial time of whole forecasts.
EVENT_WINDOWS = ("valid", "init")


def _event_time(name: str, key: str, value: Any) -> np.datetime64:
    """Parse one event timestamp, accepting ISO strings and datetimes."""
    try:
        return np.datetime64(value.strip() if isinstance(value, str) else value)
    except (TypeError, ValueError) as err:
        raise ValueError(
            f"Event '{name}': '{key}' must be an ISO timestamp such as "
            f"'2025-01-23 12:00:00'; got {value!r}."
        ) from err


def parse_events(value: Any) -> dict[str, dict[str, Any]] | None:
    """Check and normalize the ``scoring.events`` block.

    An event pairs a time window with a region, so that a campaign can
    report skill for one named episode (a windstorm, a heat wave, a
    hurricane) instead of the whole year::

        scoring:
            events:
                storm_eowyn:
                    label: "Storm Éowyn"              # optional display name
                    start: "2025-01-23 12:00:00"      # window, inclusive
                    end: "2025-01-25 00:00:00"
                    region: {lat: [48, 62], lon: [-15, 5]}
                    window: valid                     # or init
                    ics: {step_hours: 12, lookback_hours: 336}

    ``region`` takes the same forms as a ``scoring.regions`` entry (one
    box, a list of boxes scored as their union, or ``null`` for the whole
    grid) or the NAME of a ``scoring.regions`` entry.  A missing region
    means the whole grid.  ``window`` selects which timestamp the window
    filters: ``valid`` keeps the (initial condition, lead) pairs that
    verify inside it, ``init`` keeps whole forecasts by their initial
    time.  The optional ``ics`` block adds initial conditions to
    the campaign, every ``step_hours`` from ``start - lookback_hours`` up
    to ``end`` (see :func:`src.work.event_initial_times`).  A ``valid``
    window must state ``lookback_hours``: it decides the longest lead
    time that still has valid times inside the window.

    Parameters
    ----------
    value : Any
        The ``scoring.events`` block (``DictConfig``, ``dict`` or ``None``).

    Returns
    -------
    dict[str, dict] | None
        Event name to ``{"label", "start", "end", "region", "window",
        "ics"}`` in config order.  ``start`` and ``end`` are
        ``np.datetime64``; ``region`` is a region name, a list of boxes or
        ``None``; ``ics`` is ``{"step_hours", "lookback_hours"}`` or
        ``None``.  ``None`` when the block is absent.

    Raises
    ------
    ValueError
        On an unknown key, a missing or unordered window, an unknown
        window kind, a malformed box, or an ``ics`` block without the
        entries its window needs.
    """
    if value is None:
        return None
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "scoring.events must be a non-empty mapping of "
            "event name -> {start, end, region, window, ics}."
        )

    out: dict[str, dict[str, Any]] = {}
    for raw_name, spec in value.items():
        name = str(raw_name)
        if not isinstance(spec, dict):
            raise ValueError(
                f"Event '{name}' must be a mapping with 'start', 'end' and "
                f"'region'; got {spec!r}."
            )
        unknown = sorted(set(spec) - _EVENT_KEYS)
        if unknown:
            raise ValueError(
                f"Event '{name}' has unknown key(s) {unknown}; "
                f"allowed keys are {sorted(_EVENT_KEYS)}."
            )
        missing = [k for k in ("start", "end") if spec.get(k) is None]
        if missing:
            raise ValueError(f"Event '{name}' needs {missing}.")
        start = _event_time(name, "start", spec["start"])
        end = _event_time(name, "end", spec["end"])
        if not start <= end:
            raise ValueError(
                f"Event '{name}': 'start' must not be after 'end'; "
                f"got {start} > {end}."
            )
        window = str(spec.get("window") or "valid").lower()
        if window not in EVENT_WINDOWS:
            raise ValueError(
                f"Event '{name}': window must be one of {EVENT_WINDOWS}; "
                f"got {spec.get('window')!r}."
            )
        region_spec = spec.get("region", None)
        region: str | list[dict] | None
        if isinstance(region_spec, str):
            region = region_spec
        else:
            parsed = parse_regions({name: region_spec})
            region = parsed[name] if parsed is not None else None

        ics_spec = spec.get("ics", None)
        ics: dict[str, int] | None = None
        if ics_spec is not None:
            if not isinstance(ics_spec, dict):
                raise ValueError(
                    f"Event '{name}': 'ics' must be a mapping with "
                    f"'step_hours' and 'lookback_hours'; got {ics_spec!r}."
                )
            unknown_ic = sorted(set(ics_spec) - _EVENT_IC_KEYS)
            if unknown_ic:
                raise ValueError(
                    f"Event '{name}': unknown ics key(s) {unknown_ic}; "
                    f"allowed keys are {sorted(_EVENT_IC_KEYS)}."
                )
            if ics_spec.get("step_hours") is None:
                raise ValueError(f"Event '{name}': ics.step_hours is required.")
            step_hours = int(ics_spec["step_hours"])
            if step_hours <= 0:
                raise ValueError(
                    f"Event '{name}': ics.step_hours must be positive; "
                    f"got {step_hours}."
                )
            lookback = ics_spec.get("lookback_hours", None)
            if lookback is None:
                if window == "valid":
                    raise ValueError(
                        f"Event '{name}': a 'valid' window needs "
                        "ics.lookback_hours, the hours before 'start' from "
                        "which initial conditions are added.  Set it to the "
                        "forecast horizon (nsteps x model step) so every "
                        "lead time has valid times inside the window."
                    )
                lookback = 0
            lookback_hours = int(lookback)
            if lookback_hours < 0:
                raise ValueError(
                    f"Event '{name}': ics.lookback_hours must be >= 0; "
                    f"got {lookback_hours}."
                )
            ics = {"step_hours": step_hours, "lookback_hours": lookback_hours}

        out[name] = {
            "label": str(spec.get("label") or name),
            "start": start,
            "end": end,
            "region": region,
            "window": window,
            "ics": ics,
        }
    return out


def event_region_name(name: str, event: dict[str, Any]) -> str:
    """Name of the score-store region that carries an event's scores.

    An event that points at a ``scoring.regions`` entry by name scores on
    that region; one with its own box (or the whole grid) scores on a
    region named after the event.

    Parameters
    ----------
    name : str
        Event name.
    event : dict
        One parsed event (see :func:`parse_events`).

    Returns
    -------
    str
        Region label as it appears on the ``region`` axis.
    """
    region = event.get("region")
    return region if isinstance(region, str) else name


def events_to_attrs(events: dict[str, dict[str, Any]] | None) -> dict | None:
    """JSON-friendly form of parsed events, for zarr store attributes.

    The exporter reads events back from the score store (see
    :func:`events_from_attrs`), so the definitions travel with the numbers
    they describe rather than depending on the campaign file still being
    around.

    Parameters
    ----------
    events : dict[str, dict] | None
        Parsed events (see :func:`parse_events`).

    Returns
    -------
    dict | None
        Same structure with timestamps as ``YYYY-MM-DDTHH:MM:SS`` strings,
        or ``None`` when there are no events.
    """
    if not events:
        return None
    out: dict[str, dict[str, Any]] = {}
    for name, event in events.items():
        out[name] = {
            "label": event["label"],
            "start": str(np.datetime64(event["start"], "s")),
            "end": str(np.datetime64(event["end"], "s")),
            "region": event["region"],
            "window": event["window"],
            "ics": event["ics"],
        }
    return out


def events_from_attrs(attrs: Any) -> dict[str, dict[str, Any]] | None:
    """Parsed events from a score store's ``events`` attribute.

    Parameters
    ----------
    attrs : Any
        The value stored by :func:`events_to_attrs`, or ``None``.

    Returns
    -------
    dict[str, dict] | None
        Parsed events (see :func:`parse_events`), or ``None``.
    """
    if not attrs:
        return None
    return parse_events(dict(attrs))


def scoring_regions(scoring_cfg: Any) -> dict[str, list[dict] | None] | None:
    """The regions a run scores: ``scoring.regions`` plus the event boxes.

    Both scoring pathways call this instead of :func:`parse_regions`
    directly, so an event's box becomes a region of the score stores with
    no further wiring: the offline path masks it like any other region and
    the online path folds it into the accumulated sums.  That is also why
    an online run needs its event regions in the config before it starts,
    unlike event time windows.  An event whose ``region`` names a
    ``scoring.regions`` entry adds nothing.  When events introduce the
    first regions of a run, a whole-grid ``global`` split goes in front,
    so the headline curves and the spectral metrics keep a place to land.

    Parameters
    ----------
    scoring_cfg : DictConfig | dict | None
        The ``scoring`` config block.

    Returns
    -------
    dict[str, list[dict] | None] | None
        Parsed regions in store order, or ``None`` when the run has neither
        regions nor events with boxes.

    Raises
    ------
    ValueError
        If an event names a region that ``scoring.regions`` does not
        define, or an event's own box collides with a region of the same
        name.
    """
    scoring_cfg = scoring_cfg or {}
    regions = parse_regions(scoring_cfg.get("regions", None))
    events = parse_events(scoring_cfg.get("events", None))
    if events is None:
        return regions

    merged: OrderedDict[str, list[dict] | None] = OrderedDict(regions or {})
    added = False
    for name, event in events.items():
        region = event["region"]
        if isinstance(region, str):
            if region not in merged:
                raise ValueError(
                    f"Event '{name}' refers to region '{region}', which "
                    "scoring.regions does not define."
                )
            continue
        if name in merged:
            raise ValueError(
                f"Event '{name}' defines its own box, but scoring.regions "
                f"already has a region named '{name}'.  Rename one of them, "
                "or point the event at the region by name."
            )
        merged[name] = region
        added = True

    if added and regions is None and not any(v is None for v in merged.values()):
        merged = OrderedDict([("global", None), *merged.items()])
    return merged or None

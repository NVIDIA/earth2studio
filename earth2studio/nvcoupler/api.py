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

"""User-facing API: auto-wiring, one-call runs, and plan inspection.

The NUOPC analogy stops at the door here — NUOPC makes you write the runSeq;
:func:`couple` writes it for you. Matching is by field-dictionary standard
name: every advertised import is paired with its unique exporter, derived
fields (entries carrying a :class:`~.dictionary.CellMethod`) get an
:class:`~.mediator.AccumulationMediator` synthesized between cadences, and
the run sequence is laid out slot-by-slot in the canonical lagged shape.
:func:`describe` renders the resulting plan (terraform-plan style) before a
single tensor moves, and :func:`coupled` is the notebook one-liner from
initial conditions to xarray Datasets.
"""

import math
from collections.abc import Sequence as SequenceABC

import numpy as np

from .clock import Clock, DeltaLike, TimeLike, as_timedelta, fmt_timedelta
from .component import Component
from .connector import Connector
from .driver import Driver
from .errors import AmbiguousCouplingError, UnmatchedImportError
from .mediator import AccumulationMediator, Mediator
from .sequence import (
    Action,
    ConnectAction,
    MediateAction,
    RunAction,
    RunSequence,
    Slot,
)

__all__ = ["couple", "coupled", "describe", "describe_html"]


# ---------------------------------------------------------------------------
# couple(): auto-wiring
# ---------------------------------------------------------------------------
def _short_name(component: Component, standard_name: str) -> str:
    """Shortest registered alias of a standard name (else the name itself),
    sanitized to the run-sequence name grammar."""
    entry = component.dictionary.resolve(standard_name)
    candidates = [entry.standard_name, *entry.aliases]
    short = min(candidates, key=len)
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in short)


def _gcd_timestep(components: SequenceABC[Component]) -> np.timedelta64:
    ns = [c.timestep.astype("timedelta64[ns]").astype(np.int64) for c in components]
    return np.timedelta64(int(math.gcd(*(int(n) for n in ns))), "ns")


def _synthesize_mediators(
    components: list[Component],
) -> tuple[list[Mediator], list[tuple[str, str]]]:
    """Create AccumulationMediators for derived imports nobody exports.

    Returns the new mediators plus the extra (src, dst) connections wiring
    each base-field exporter into its mediator and each mediator into the
    importer.
    """
    mediators: dict[str, Mediator] = {}  # derived std name -> mediator
    connections: list[tuple[str, str]] = []
    exports_by_comp = {c.name: list(c.export_names) for c in components}

    for comp in components:
        for imp in comp.import_names:
            exporters = [
                c.name
                for c in components
                if c.name != comp.name and imp in c.export_names
            ]
            if len(exporters) > 1:
                raise AmbiguousCouplingError(imp, comp.name, exporters)
            if len(exporters) == 1:
                continue
            # zero exporters: a derived field with a CellMethod can be
            # produced by a synthesized mediator, if someone exports the base
            entry = comp.dictionary.resolve(imp)
            cell_method = entry.cell_method
            base_exporters = (
                [
                    c
                    for c in components
                    if cell_method is not None and cell_method.base in c.export_names
                ]
                if cell_method is not None
                else []
            )
            if cell_method is None or not base_exporters:
                raise UnmatchedImportError(comp.name, imp, exports_by_comp)
            if len(base_exporters) > 1:
                raise AmbiguousCouplingError(
                    cell_method.base, comp.name, [c.name for c in base_exporters]
                )
            if imp not in mediators:
                med = AccumulationMediator(
                    f"med_{_short_name(comp, imp)}",
                    [imp],
                    dictionary=comp.dictionary,
                )
                mediators[imp] = med
                connections.append((base_exporters[0].name, med.name))
            connections.append((mediators[imp].name, comp.name))
    return list(mediators.values()), connections


def _build_sequence(
    components: list[Component],
    connections: list[tuple[str, str]],
) -> RunSequence:
    """Lay out the canonical lagged run sequence.

    One slot per distinct cadence, fast to slow. A connection runs at the
    faster of its endpoints' cadences; within a slot the order is: plain
    connects (lagged: destinations see the sources' previous exports), then
    each mediator's compute followed by its outgoing connects, then the
    component runs.
    """
    by_name = {c.name: c for c in components}
    order = {c.name: i for i, c in enumerate(components)}
    cadences = sorted(
        {c.timestep for c in components},
        key=lambda ts: ts.astype("timedelta64[ns]").astype(np.int64),
    )

    def cadence_of(name: str) -> np.timedelta64:
        return by_name[name].timestep

    slots: list[Slot] = []
    for interval in cadences:
        actions: list[Action] = []
        # 1. lagged connects: sources are not mediators (mediator exports
        #    only exist after compute), delivered at the faster endpoint
        for src, dst in connections:
            if isinstance(by_name[src], Mediator):
                continue
            if min(cadence_of(src), cadence_of(dst)) == interval:
                actions.append(ConnectAction(src, dst))
        # 2. mediators of this cadence: compute, then deliver
        for comp in components:
            if isinstance(comp, Mediator) and comp.timestep == interval:
                actions.append(MediateAction(comp.name))
                actions.extend(
                    ConnectAction(src, dst)
                    for src, dst in connections
                    if src == comp.name
                )
        # 3. component runs of this cadence
        actions.extend(
            RunAction(comp.name)
            for comp in components
            if not isinstance(comp, Mediator) and comp.timestep == interval
        )
        if actions:
            slots.append(Slot(interval, actions))
    # deterministic connect order inside stage 1: source order as given
    for slot in slots:
        split = len(slot.actions)
        for i, a in enumerate(slot.actions):
            if not isinstance(a, ConnectAction):
                split = i
                break
        lagged = slot.actions[:split]
        lagged.sort(key=lambda a: (order[a.src], order[a.dst]))
        slot.actions = lagged + slot.actions[split:]
    return RunSequence(slots)


def couple(
    *components: Component,
    start: TimeLike,
    stop: TimeLike,
    dt: DeltaLike | None = None,
    connectors: list[Connector] | None = None,
    collect: bool = True,
) -> Driver:
    """Auto-wire components into a ready-to-initialize :class:`Driver`.

    Every advertised import is matched to its exporter by standard name;
    derived imports (dictionary entries with a CellMethod) whose base field
    is exported get an AccumulationMediator synthesized between the two
    cadences. The generated run sequence uses lagged (NUOPC-explicit)
    coupling: connects precede the runs of each slot.

    Parameters
    ----------
    *components : Component
        Participants (mediators may be included explicitly, or synthesized).
    start, stop : TimeLike
        Clock span.
    dt : DeltaLike, optional
        Coupling interval; defaults to the GCD of all component timesteps.
    connectors : list[Connector], optional
        Pre-built connectors overriding the defaults for their (src, dst).
    collect : bool
        Keep per-ring exports in memory for ``to_xarray()``.

    Returns
    -------
    Driver
        Not yet initialized — call ``driver.initialize(ics)``.
    """
    comps = list(components)
    mediators, med_connections = _synthesize_mediators(comps)
    comps.extend(mediators)
    by_name = {c.name: c for c in comps}

    # direct (unique-exporter) connections
    connections: list[tuple[str, str]] = []
    for comp in comps:
        for imp in comp.import_names:
            for c in comps:
                if c.name != comp.name and imp in c.export_names:
                    if (c.name, comp.name) not in connections:
                        connections.append((c.name, comp.name))
    for pair in med_connections:
        if pair not in connections:
            connections.append(pair)

    if dt is None:
        dt = _gcd_timestep(comps)
    sequence = _build_sequence(comps, connections)
    clock = Clock(start, stop, dt)
    return Driver(by_name, sequence, clock, connectors=connectors, collect=collect)


# ---------------------------------------------------------------------------
# coupled(): one-call convenience entry point
# ---------------------------------------------------------------------------
def coupled(
    time: TimeLike,
    stop_or_nsteps: TimeLike | int,
    components: "SequenceABC[Component] | dict[str, Component]",
    ics: dict[str, tuple],
    dt: DeltaLike | None = None,
    collect: bool = True,
    verbose: bool = True,
) -> dict:
    """Build, initialize, and run a coupled system in one call.

    Parameters
    ----------
    time : TimeLike
        Start time.
    stop_or_nsteps : TimeLike | int
        Stop time, or a number of driver (dt) steps.
    components : list[Component] | dict[str, Component]
        Participants; a dict's values are used (keys are cosmetic).
    ics : dict[str, tuple[torch.Tensor, CoordSystem]]
        Initial condition per non-mediator component name.
    dt : DeltaLike, optional
        Coupling interval; defaults to the GCD of the component timesteps.
    verbose : bool
        Show a tqdm progress bar over driver steps.

    Returns
    -------
    dict[str, xarray.Dataset]
        One Dataset of collected exports per component.
    """
    from tqdm import tqdm

    comps = list(components.values()) if isinstance(components, dict) else list(components)
    dt_td = as_timedelta(dt) if dt is not None else _gcd_timestep(comps)
    if isinstance(stop_or_nsteps, (int, np.integer)):
        stop = np.datetime64(time) + int(stop_or_nsteps) * dt_td
    else:
        stop = stop_or_nsteps
    driver = couple(*comps, start=time, stop=stop, dt=dt_td, collect=collect)
    driver.initialize(ics)

    counts = {n: c.run_count for n, c in driver.components.items()}
    with tqdm(
        total=driver.clock.n_steps,
        desc="Running coupled inference",
        disable=(not verbose),
    ) as pbar:
        for step_time, _ in driver.steps():
            ran = [
                n
                for n, c in driver.components.items()
                if c.run_count != counts[n]
            ]
            counts = {n: c.run_count for n, c in driver.components.items()}
            pbar.set_postfix_str(f"{np.datetime_as_string(step_time, unit='h')} ran {'+'.join(ran)}")
            pbar.update(1)
    return driver.to_xarray()


# ---------------------------------------------------------------------------
# describe(): the coupling plan, before anything runs
# ---------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*row) for row in rows)
    return lines


def _connector_rows(driver: Driver) -> list[dict]:
    """One row per ConnectAction: fields, policies, and coupling mode derived
    from the action's position relative to the destination's run/mediate."""
    rows = []
    prebuilt = {
        (c.src.name, c.dst.name): c for c in driver._connectors.values()
    }
    for slot in driver.sequence.slots:
        dst_ran: set[str] = set()
        for action in slot.actions:
            if isinstance(action, RunAction):
                dst_ran.add(action.component)
            elif isinstance(action, MediateAction):
                dst_ran.add(action.mediator)
            elif isinstance(action, ConnectAction):
                conn = prebuilt.get((action.src, action.dst))
                if conn is not None:
                    fields = conn.match()
                    time_policy, fill = conn.time_policy, conn.fill
                else:
                    src = driver.components[action.src]
                    dst = driver.components[action.dst]
                    _, exports = src.advertise()
                    imports, _ = dst.advertise()
                    fields = [n for n in imports if n in exports]
                    time_policy, fill = "constant", "none"
                dst_runs_here = any(
                    (isinstance(a, RunAction) and a.component == action.dst)
                    or (isinstance(a, MediateAction) and a.mediator == action.dst)
                    for a in slot.actions
                )
                if not dst_runs_here:
                    mode = "lagged"
                else:
                    mode = "sequential" if action.dst in dst_ran else "lagged"
                rows.append(
                    {
                        "name": f"{action.src} -> {action.dst}",
                        "fields": fields,
                        "time_policy": time_policy,
                        "fill": fill,
                        "mode": mode,
                        "slot": fmt_timedelta(slot.interval),
                    }
                )
    return rows


def describe(driver: Driver) -> str:
    """Terraform-plan-style text summary of a coupled system.

    Works before ``initialize()``: only advertised imports/exports, the run
    sequence, and the clock are consulted (grids are unknown until realize).
    """
    lines = [f"Coupled system: {driver.clock!r}", "", "Components:"]
    comp_rows = []
    for name, comp in driver.components.items():
        imports, exports = comp.advertise()
        comp_rows.append(
            [
                name,
                type(comp).__name__,
                fmt_timedelta(comp.timestep),
                ", ".join(imports) or "-",
                ", ".join(exports) or "-",
            ]
        )
    lines.extend(
        "  " + row
        for row in _table(
            ["name", "type", "cadence", "imports", "exports"], comp_rows
        )
    )
    lines.extend(["", "Connectors:"])
    conn_rows = [
        [
            r["name"],
            ", ".join(r["fields"]) or "-",
            r["time_policy"],
            r["fill"],
            r["mode"],
            r["slot"],
        ]
        for r in _connector_rows(driver)
    ]
    if conn_rows:
        lines.extend(
            "  " + row
            for row in _table(
                ["connector", "fields", "time_policy", "fill", "mode", "slot"],
                conn_rows,
            )
        )
    else:
        lines.append("  (none)")
    lines.extend(["", "Run sequence:"])
    lines.extend("  " + line for line in str(driver.sequence).splitlines())
    return "\n".join(lines)


_HTML_STYLE = """
.nvc-plan { font-family: -apple-system, Segoe UI, sans-serif; color: #1a1a1a; }
.nvc-plan h4 { margin: 0.6em 0 0.3em; }
.nvc-boxes { display: flex; flex-wrap: wrap; gap: 10px; }
.nvc-box { border: 1px solid #888; border-radius: 6px; padding: 8px 12px;
           background: #f7f7f7; min-width: 180px; }
.nvc-box .nvc-name { font-weight: 600; }
.nvc-box .nvc-meta { font-size: 0.85em; color: #444; }
.nvc-arrows { list-style: none; padding-left: 0; }
.nvc-arrows li { padding: 2px 0; font-size: 0.9em; }
.nvc-arrow { color: #0a6; font-weight: 600; }
.nvc-seq { background: #f0f0f0; border-radius: 6px; padding: 8px 12px;
           font-size: 0.9em; }
"""


def describe_html(driver: Driver) -> str:
    """Self-contained HTML rendering of the coupling plan (for Jupyter)."""
    import html

    def esc(s: str) -> str:
        return html.escape(str(s))

    boxes = []
    for name, comp in driver.components.items():
        imports, exports = comp.advertise()
        boxes.append(
            f'<div class="nvc-box"><div class="nvc-name">{esc(name)}</div>'
            f'<div class="nvc-meta">{esc(type(comp).__name__)} @ '
            f"{esc(fmt_timedelta(comp.timestep))}<br>"
            f"imports: {esc(', '.join(imports) or '-')}<br>"
            f"exports: {esc(', '.join(exports) or '-')}</div></div>"
        )
    arrows = []
    for r in _connector_rows(driver):
        src, dst = r["name"].split(" -> ")
        arrows.append(
            f'<li>{esc(src)} <span class="nvc-arrow">&rarr;</span> {esc(dst)}: '
            f"{esc(', '.join(r['fields']) or '-')} "
            f"[{esc(r['time_policy'])}, fill={esc(r['fill'])}, {esc(r['mode'])}, "
            f"@{esc(r['slot'])}]</li>"
        )
    seq = "<br>".join(esc(line) for line in str(driver.sequence).splitlines())
    return (
        f"<style>{_HTML_STYLE}</style>"
        '<div class="nvc-plan">'
        f"<h4>Coupled system</h4><div>{esc(repr(driver.clock))}</div>"
        f'<h4>Components</h4><div class="nvc-boxes">{"".join(boxes)}</div>'
        f'<h4>Connectors</h4><ul class="nvc-arrows">{"".join(arrows)}</ul>'
        f'<h4>Run sequence</h4><div class="nvc-seq">{seq}</div>'
        "</div>"
    )

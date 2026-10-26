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

"""Render a coupler's plan as text or HTML, before anything runs."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

from earth2studio.nvcoupler._time import fmt_timedelta

if TYPE_CHECKING:
    from earth2studio.nvcoupler.coupler import Coupler


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(header), *(len(row[i]) for row in rows)) if rows else len(header)
        for i, header in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines.extend(fmt.format(*row) for row in rows)
    return lines


def _model_rows(coupler: Coupler) -> list[list[str]]:
    rows = []
    for name, spec in coupler.specs.items():
        rows.append(
            [
                name,
                spec.kind,
                fmt_timedelta(coupler.cadences[name]),
                ", ".join(spec.imports) or "-",
                ", ".join(spec.exports) or "-",
            ]
        )
    return rows


def _exchange_rows(coupler: Coupler) -> list[list[str]]:
    rows = []
    same_step = set(coupler._before)
    for connector in coupler.connectors:
        labels, kinds, regrid = [], set(), "-"
        for (kind, _), providers in coupler._providers[connector.dst].items():
            for label, provider in providers.items():
                if provider.connector is connector:
                    labels.append(label)
                    kinds.add(kind)
                    if provider.regridder is not None:
                        regrid = type(provider.regridder).__name__
        mode = "same step" if (connector.src, connector.dst) in same_step else "lagged"
        rows.append(
            [
                connector.name,
                ", ".join(labels) or "-",
                "/".join(sorted(kinds)) or "-",
                regrid,
                mode,
            ]
        )
    return rows


def describe(coupler: Coupler) -> str:
    """Plain-text plan of a coupled system."""
    lines = [
        f"Coupled system: dt={fmt_timedelta(coupler.dt)}",
        "",
        "Models:",
        *(
            "  " + row
            for row in _table(
                ["name", "kind", "cadence", "imports", "exports"], _model_rows(coupler)
            )
        ),
        "",
        "Exchanges:",
    ]
    rows = _exchange_rows(coupler)
    if rows:
        lines.extend(
            "  " + row
            for row in _table(["connector", "labels", "into", "regrid", "mode"], rows)
        )
    else:
        lines.append("  (none)")
    lines.extend(["", "Run sequence:"])
    lines.extend("  " + line for line in str(coupler.sequence).splitlines())
    return "\n".join(lines)


_HTML_STYLE = """
.nvc-plan { font-family: -apple-system, Segoe UI, sans-serif; color: #1a1a1a; }
.nvc-plan h4 { margin: 0.6em 0 0.3em; }
.nvc-plan table { border-collapse: collapse; font-size: 0.9em; }
.nvc-plan th, .nvc-plan td { border: 1px solid #bbb; padding: 2px 8px; text-align: left; }
.nvc-seq { background: #f0f0f0; border-radius: 6px; padding: 8px 12px; font-size: 0.9em; }
"""


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def describe_html(coupler: Coupler) -> str:
    """Self-contained HTML plan of a coupled system, for notebooks."""
    sequence = "<br>".join(
        html.escape(line) for line in str(coupler.sequence).splitlines()
    )
    return (
        f"<style>{_HTML_STYLE}</style>"
        '<div class="nvc-plan">'
        f"<h4>Coupled system</h4><div>dt={html.escape(fmt_timedelta(coupler.dt))}</div>"
        "<h4>Models</h4>"
        + _html_table(
            ["name", "kind", "cadence", "imports", "exports"], _model_rows(coupler)
        )
        + "<h4>Exchanges</h4>"
        + _html_table(
            ["connector", "labels", "into", "regrid", "mode"], _exchange_rows(coupler)
        )
        + f'<h4>Run sequence</h4><div class="nvc-seq">{sequence}</div>'
        "</div>"
    )

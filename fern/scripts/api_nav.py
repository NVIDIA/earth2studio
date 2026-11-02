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

"""Build the API Reference sidebar so it is grouped like the MkDocs ``API`` nav.

MkDocs groups the API by kind (Prognostic Models, Data Sources, ...), while Fern's
generated pages are one per Python module. This module assigns every generated page to
one of the MkDocs groups and emits:

* the navigation block, spliced into ``fern/pages/index.yml`` between the
  ``BEGIN/END GENERATED API NAV`` markers, and
* ``fern/scripts/api_urls.json``, mapping each page (``models/px/fcn``) to its public URL,
  used by ``convert_examples.py`` to link to API pages.

Group membership comes from the directory a module lives in, and for data sources from
the ``autosummary`` lists in ``docs/modules/datasources_*.md``.
"""

import json
import re
from pathlib import Path

SITE = "/earth2studio"
BEGIN = "      # BEGIN GENERATED API NAV (fern/scripts/generate_api.py) - do not edit by hand"
END = "      # END GENERATED API NAV"

# (slug, title, directory holding the group's index page or None)
GROUPS = [
    ("models_px", "Prognostic Models", "models/px"),
    ("models_dx", "Diagnostic Models", "models/dx"),
    ("models_da", "Data Assimilation", "models/da"),
    ("datasources_analysis", "Data Sources", "data"),
    ("datasources_forecast", "Forecast Data Sources", None),
    ("datasources_dataframe", "DataFrame Sources", None),
    ("io", "IO Backends", "io"),
    ("perturbation", "Perturbations", "perturbation"),
    ("statistics", "Statistics", "statistics"),
    ("utils_all", "Utilities", "utils"),
    ("workflows", "Workflows", None),
    ("other", "Additional Modules", None),
]
DIR_GROUP = {
    "models/px": "models_px",
    "models/dx": "models_dx",
    "models/da": "models_da",
    "io": "io",
    "perturbation": "perturbation",
    "statistics": "statistics",
    "utils": "utils_all",
}
DATA_LISTS = {
    "datasources_forecast": "datasources_forecast.md",
    "datasources_dataframe": "datasources_dataframe.md",
    "datasources_analysis": "datasources_analysis.md",
}


def autosummary_names(md: Path) -> list[str]:
    if not md.exists():
        return []
    return re.findall(r"^(earth2studio\.[\w.]+)\s*$", md.read_text(), flags=re.M)


def assign_groups(rels: list[str], find_page, modules_dir: Path) -> dict[str, str]:
    """Return {page rel path (no suffix): group slug}."""
    # Which data-source lists mention each data module.
    data_kinds: dict[str, set[str]] = {}
    for kind, fname in DATA_LISTS.items():
        for name in autosummary_names(modules_dir / fname):
            page = find_page(name)
            if page:
                data_kinds.setdefault(page, set()).add(kind)

    out: dict[str, str] = {}
    for rel in rels:
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        if rel == "run":
            out[rel] = "workflows"
        elif parent in DIR_GROUP:
            out[rel] = DIR_GROUP[parent]
        elif rel == "data/index":
            out[rel] = "datasources_analysis"
        elif parent == "data":
            kinds = data_kinds.get(rel, set())
            if "datasources_analysis" in kinds or not kinds:
                out[rel] = "datasources_analysis"
            elif "datasources_forecast" in kinds:
                out[rel] = "datasources_forecast"
            else:
                out[rel] = "datasources_dataframe"
        elif rel == "index":
            out[rel] = "overview"
        else:
            out[rel] = "other"
    return out


def label_for(rel: str, text: str) -> str:
    classes = re.findall(r"^class earth2studio\.[\w.]*\.(\w+)\(", text, flags=re.M)
    classes = [c for c in classes if not c.startswith("_")]
    if len(classes) == 1:
        return classes[0]
    stem = rel.rsplit("/", 1)[-1]
    if stem == "index":
        stem = rel.rsplit("/", 2)[-2] if "/" in rel else "api"
    return stem


def build(
    pages: dict[str, str],
    find_page,
    root: Path,
    index_yml: Path,
    urls_json: Path,
) -> int:
    """``pages`` maps rel path (no suffix, posix) to cleaned MDX text."""
    assigned = assign_groups(sorted(pages), find_page, root / "docs" / "modules")
    by_group: dict[str, list[str]] = {g[0]: [] for g in GROUPS}
    for rel, group in assigned.items():
        if group != "overview":
            by_group[group].append(rel)

    urls: dict[str, str] = {"index": f"{SITE}/api/overview"}
    lines: list[str] = []

    def page_entry(rel: str, label: str, slug: str, indent: str) -> list[str]:
        return [
            f"{indent}- page: {label}",
            f"{indent}  path: ../../docs/api/{rel}.mdx",
            f"{indent}  slug: {slug}",
        ]

    lines += page_entry("index", "Overview", "overview", "      ")

    for gslug, title, index_dir in GROUPS:
        members = by_group[gslug]
        if not members:
            continue
        index_rel = f"{index_dir}/index" if index_dir else None
        has_index = index_rel in members
        if gslug == "workflows":
            # Single page, shown as a plain entry like the MkDocs "Workflows" page.
            lines += page_entry("run", "Workflows", "workflows", "      ")
            urls["run"] = f"{SITE}/api/workflows"
            continue
        lines.append(f"      - section: {title}")
        if has_index:
            lines.append(f"        path: ../../docs/api/{index_rel}.mdx")
            urls[index_rel] = f"{SITE}/api/{gslug}"
        lines.append(f"        slug: {gslug}")
        lines.append("        contents:")
        entries = []
        used: set[str] = set()
        for rel in members:
            if rel == index_rel:
                continue
            stem = rel.rsplit("/", 1)[-1]
            if gslug == "other" or stem in used:
                slug = rel.replace("/", "_")
            else:
                slug = stem
            if slug in used:
                raise SystemExit(f"duplicate slug {slug!r} in group {gslug}")
            used.add(slug)
            label = (
                rel.replace("/", ".")
                if gslug == "other"
                else label_for(rel, pages[rel])
            )
            if gslug == "other" and rel.endswith("/index"):
                label = rel[: -len("/index")].replace("/", ".") + " (package)"
            entries.append((label.lower(), rel, label, slug))
            urls[rel] = f"{SITE}/api/{gslug}/{slug}"
        for _, rel, label, slug in sorted(entries):
            lines += page_entry(rel, label, slug, "          ")

    block = "\n".join([BEGIN, *lines, END])
    text = index_yml.read_text()
    pat = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    if not pat.search(text):
        raise SystemExit(f"{index_yml}: GENERATED API NAV markers not found")
    index_yml.write_text(pat.sub(lambda _: block, text))
    urls_json.write_text(json.dumps(urls, indent=2, sort_keys=True) + "\n")

    missing = set(pages) - set(urls)
    if missing:
        raise SystemExit(f"pages not in nav: {sorted(missing)[:5]}")
    return len(urls)

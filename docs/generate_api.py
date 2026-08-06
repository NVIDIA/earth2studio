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
"""Generate MkDocs API pages from the legacy Sphinx autosummary sources."""

from __future__ import annotations

import ast
import os
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODULES = DOCS / "modules"
GENERATED = MODULES / "generated"
DOC_VERSION = os.environ.get("DOC_VERSION", "main")

BADGE_RE = re.compile(
    r"\b(?:region|class|dataclass|year|product|gpu):[A-Za-z0-9_.-]+\b"
)
RST_ROLE_RE = re.compile(r":(?:mod|class|func|py:class|py:func|py:obj):`~?([^`]+)`")


@dataclass(frozen=True)
class ObjectPage:
    """Metadata used to render one generated API object page."""

    display: str
    full_name: str
    source: Path | None
    kind: str
    summary: str
    docstring: str
    badges: tuple[str, ...]
    line_start: int | None
    line_end: int | None
    output: Path


def clean_rst_roles(text: str) -> str:
    """Convert simple reStructuredText roles to Markdown code or links."""
    text = RST_ROLE_RE.sub(lambda match: f"`{match.group(1)}`", text)
    text = re.sub(r"`([^`<]+)\s*<([^`>]+)>`_", r"[\1](\2)", text)
    return text.replace("``", "`")


def rst_title(line: str) -> str:
    """Normalize an RST title line for Markdown output."""
    return clean_rst_roles(line.strip())


def parse_import_map(package_dir: Path) -> dict[str, tuple[Path, str]]:
    """Map package re-exports to their source files."""
    init = package_dir / "__init__.py"
    if not init.exists():
        return {}
    tree = ast.parse(init.read_text(encoding="utf-8"))
    result: dict[str, tuple[Path, str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.level:
            base = package_dir
            for _ in range(node.level - 1):
                base = base.parent
            module_path = base / Path(*node.module.split(".")).with_suffix(".py")
        elif node.module.startswith("earth2studio."):
            module_path = ROOT / Path(*node.module.split(".")).with_suffix(".py")
        else:
            continue
        for alias in node.names:
            result[alias.asname or alias.name] = (module_path, alias.name)
    return result


IMPORT_MAPS: dict[Path, dict[str, tuple[Path, str]]] = {}


def package_import_map(package_dir: Path) -> dict[str, tuple[Path, str]]:
    """Return a cached import map for a package directory."""
    package_dir = package_dir.resolve()
    if package_dir not in IMPORT_MAPS:
        IMPORT_MAPS[package_dir] = parse_import_map(package_dir)
    return IMPORT_MAPS[package_dir]


def find_node(source: Path, name: str) -> ast.AST | None:
    """Find a top-level class or function node in a source file."""
    if not source.exists():
        return None
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def module_docstring(source: Path) -> str:
    """Read the module docstring from a Python source file."""
    if not source.exists():
        return ""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    return ast.get_docstring(tree) or ""


def resolve(full_name: str) -> tuple[Path | None, ast.AST | None, str]:
    """Resolve an import path to source file, AST node, and kind."""
    parts = full_name.split(".")
    if len(parts) < 2 or parts[0] != "earth2studio":
        return None, None, "object"

    for split in range(len(parts) - 1, 1, -1):
        package_dir = ROOT / Path(*parts[:split])
        obj_name = parts[split]
        if not (package_dir / "__init__.py").exists():
            continue
        mapped = package_import_map(package_dir).get(obj_name)
        if mapped:
            source, actual = mapped
            node = find_node(source, actual)
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            return source, node, kind

    candidate_module = ROOT / Path(*parts[:-1]).with_suffix(".py")
    obj_name = parts[-1]
    if candidate_module.exists():
        node = find_node(candidate_module, obj_name)
        if node is not None:
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            return candidate_module, node, kind
        return candidate_module, None, "module"

    candidate_module = ROOT / Path(*parts).with_suffix(".py")
    if candidate_module.exists():
        return candidate_module, None, "module"

    return None, None, "object"


def first_paragraph(docstring: str) -> str:
    """Extract the first prose paragraph from a docstring."""
    lines = [line.strip() for line in docstring.splitlines()]
    kept: list[str] = []
    for line in lines:
        if not line:
            if kept:
                break
            continue
        if BADGE_RE.search(line) or re.fullmatch(r"[-=~^]+", line):
            continue
        kept.append(line)
    return " ".join(kept) or "API reference page."


def source_link(source: Path | None, start: int | None, end: int | None) -> str:
    """Build a GitHub source URL for a generated API page."""
    if source is None:
        return ""
    rel = source.relative_to(ROOT).as_posix()
    lines = f"#L{start}-L{end}" if start and end else ""
    return f"https://github.com/NVIDIA/earth2studio/blob/{DOC_VERSION}/{rel}{lines}"


def yaml_list(values: Iterable[str]) -> str:
    """Format scalar strings as a compact YAML list."""
    values = list(values)
    if not values:
        return "[]"
    return "[" + ", ".join(values) + "]"


def object_page(display: str, full_name: str, output_dir: Path) -> ObjectPage:
    """Collect source, summary, and badge metadata for one API object."""
    source, node, kind = resolve(full_name)
    if node is not None:
        docstring = ast.get_docstring(node) or ""
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
    elif source is not None:
        docstring = module_docstring(source)
        start = end = None
    else:
        docstring = ""
        start = end = None
    badges = tuple(dict.fromkeys(BADGE_RE.findall(docstring)))
    summary = clean_rst_roles(first_paragraph(docstring))
    filename = display.replace(".", "_").replace("~", "") + ".md"
    return ObjectPage(
        display=display,
        full_name=full_name,
        source=source,
        kind=kind,
        summary=summary,
        docstring=clean_rst_roles(docstring),
        badges=badges,
        line_start=start,
        line_end=end,
        output=output_dir / filename,
    )


def strip_badges_section(docstring: str) -> str:
    """Remove the Sphinx badge section from copied docstring text."""
    lines = docstring.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip() == "Badges":
            index += 1
            if index < len(lines) and re.fullmatch(r"[-=~^]+", lines[index].strip()):
                index += 1
            while index < len(lines) and (
                BADGE_RE.search(lines[index]) or not lines[index].strip()
            ):
                index += 1
            continue
        output.append(lines[index])
        index += 1
    return "\n".join(output).strip()


def write_object(page: ObjectPage) -> None:
    """Write one generated API object page."""
    page.output.parent.mkdir(parents=True, exist_ok=True)
    link = source_link(page.source, page.line_start, page.line_end)
    body = [
        "---",
        f"title: {page.display}",
        f"summary: {page.summary!r}",
        (
            f"signature: {page.display}(...)"
            if page.kind != "module"
            else f"signature: {page.display}"
        ),
        f"badges: {yaml_list(page.badges)}",
        "---",
        "",
        f"# `{page.display}`",
        "",
    ]
    if page.badges:
        body.extend(["{% badges " + " ".join(page.badges) + " %}", ""])
    body.extend([f"**Import path:** `{page.full_name}`", ""])
    if link:
        body.extend([f"[Source]({link})", ""])
    if page.docstring:
        doc = strip_badges_section(page.docstring)
        body.extend(["## Documentation", "", doc, ""])
    page.output.write_text("\n".join(body), encoding="utf-8")


def collect_autosummaries(
    path: Path,
) -> list[tuple[str, list[str], dict[str, str], list[str]]]:
    """Collect autosummary entries and badge-filter options from RST."""
    lines = path.read_text(encoding="utf-8").splitlines()
    current_module = "earth2studio"
    result: list[tuple[str, list[str], dict[str, str], list[str]]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(".. currentmodule::"):
            current_module = stripped.split("::", 1)[1].strip()
        if not stripped.startswith(".. autosummary::"):
            continue

        badges: list[str] = []
        options: dict[str, str] = {}
        for prev in range(index - 1, -1, -1):
            previous = lines[prev]
            pstrip = previous.strip()
            if pstrip.startswith(".. badge-filter::"):
                badges.extend(BADGE_RE.findall(pstrip))
                scan = prev + 1
                while scan < index:
                    s = lines[scan].strip()
                    if s.startswith(":filter-mode:"):
                        options["mode"] = s.split(":", 2)[2].strip()
                    elif s.startswith(":badge-order-fixed:"):
                        options["order"] = "fixed"
                    elif s.startswith(":group-visibility-toggle:"):
                        options["toggle"] = "true"
                    elif s.startswith(":group-hidden:"):
                        options["hidden"] = s.split(":", 2)[2].strip()
                    else:
                        badges.extend(BADGE_RE.findall(s))
                    scan += 1
                break
            if pstrip and not previous.startswith(" ") and not pstrip.startswith(":"):
                break

        objects: list[str] = []
        scan = index + 1
        while scan < len(lines):
            raw = lines[scan]
            s = raw.strip()
            if not s:
                scan += 1
                continue
            if not raw.startswith(" "):
                break
            if s.startswith(":") or s.startswith(".."):
                scan += 1
                continue
            objects.append(s)
            scan += 1
        result.append((current_module, list(dict.fromkeys(badges)), options, objects))
    return result


def intro_markdown(path: Path) -> str:
    """Convert introductory RST content to Markdown."""
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    skip_directive = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(".. badge-filter::") or stripped.startswith(
            ".. autosummary::"
        ):
            break
        if (
            stripped.startswith(".. _")
            or stripped.startswith(".. automodule::")
            or stripped.startswith(".. currentmodule::")
        ):
            skip_directive = True
            continue
        if skip_directive:
            if line.startswith(" ") or not stripped:
                continue
            skip_directive = False
        if index + 1 < len(lines) and set(lines[index + 1].strip()) in (
            {"-"},
            {"~"},
        ):
            heading = rst_title(stripped)
            level = "#" if set(lines[index + 1].strip()) == {"-"} else "##"
            output.append(f"{level} {heading}")
            continue
        if index > 0 and set(stripped) in ({"-"}, {"~"}):
            continue
        if stripped.startswith(".. warning"):
            output.append("!!! warning")
            continue
        if line.startswith("   ") and output and output[-1].startswith("!!!"):
            output.append("    " + clean_rst_roles(line.strip()))
            continue
        output.append(clean_rst_roles(line))
    text = "\n".join(output).strip()
    return text or f"# {path.stem.replace('_', ' ').title()}"


def full_name(current_module: str, name: str) -> str:
    """Expand an autosummary name relative to its current module."""
    if name.startswith("earth2studio."):
        return name
    return f"{current_module}.{name}"


def filter_options(options: dict[str, str]) -> str:
    """Format badge-filter options for the MkDocs marker."""
    parts = []
    if mode := options.get("mode"):
        parts.append(f"mode={mode}")
    if order := options.get("order"):
        parts.append(f"order={order}")
    if toggle := options.get("toggle"):
        parts.append(f"toggle={toggle}")
    if hidden := options.get("hidden"):
        parts.append(f'hidden="{hidden}"')
    return " ".join(parts)


def generated_dir_for(path: Path, group_index: int) -> Path:
    """Return the output directory for generated API object pages."""
    name = path.stem
    if name.startswith("models_"):
        return GENERATED / "models" / name.removeprefix("models_")
    if name.startswith("datasources_"):
        return GENERATED / "data" / name.removeprefix("datasources_")
    if name == "statistics":
        return GENERATED / "statistics" / str(group_index)
    if name == "utils_all":
        return GENERATED / "utils" / str(group_index)
    return GENERATED / name / str(group_index)


def build_module_page(path: Path) -> None:
    """Generate one MkDocs API summary page from a Sphinx page."""
    groups = collect_autosummaries(path)
    md_path = path.with_suffix(".md")
    body = [intro_markdown(path), ""]
    for group_index, (current_module, badges, options, objects) in enumerate(
        groups, start=1
    ):
        if not objects:
            continue
        out_dir = generated_dir_for(path, group_index)
        pages = [
            object_page(obj, full_name(current_module, obj), out_dir) for obj in objects
        ]
        for page in pages:
            write_object(page)
        if len(groups) > 1:
            body.extend([f"## {path.stem.replace('_', ' ').title()} {group_index}", ""])
        option_text = filter_options(options)
        if badges:
            body.append(
                "<!-- mkdocs-badges:filter "
                + " ".join(badges)
                + (f" {option_text}" if option_text else "")
                + " -->"
            )
            body.append("")
        body.append("{% autosummary %}")
        body.extend(page.output.relative_to(DOCS).as_posix() for page in pages)
        body.append("{% endautosummary %}")
        if badges:
            body.extend(["", "<!-- mkdocs-badges:end -->"])
        body.append("")
    md_path.write_text("\n".join(body), encoding="utf-8")


def write_index() -> None:
    """Write the API reference landing page."""
    pages = [
        ("Prognostic Models", "models_px.md"),
        ("Diagnostic Models", "models_dx.md"),
        ("Data Assimilation", "models_da.md"),
        ("Analysis Data Sources", "datasources_analysis.md"),
        ("Forecast Data Sources", "datasources_forecast.md"),
        ("DataFrame Sources", "datasources_dataframe.md"),
        ("IO Backends", "io.md"),
        ("Perturbations", "perturbation.md"),
        ("Statistics", "statistics.md"),
        ("Utilities", "utils_all.md"),
        ("Workflows", "workflows.md"),
    ]
    lines = [
        "# API Reference",
        "",
        "The API reference is generated from the legacy Sphinx autosummary source",
        "files and rendered with MkDocs badges for filtering.",
        "",
    ]
    for title, target in pages:
        lines.append(f"- [{title}]({target})")
    lines.append("")
    (MODULES / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Generate all MkDocs API pages."""
    if GENERATED.exists():
        shutil.rmtree(GENERATED)
    for path in sorted(MODULES.glob("*.rst")):
        build_module_page(path)
    write_index()


if __name__ == "__main__":
    main()

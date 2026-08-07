# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MkDocs compatibility hooks for legacy Sphinx/MyST documentation blocks."""

from __future__ import annotations

import re
from html import escape
from posixpath import dirname, relpath

OPEN_RE = re.compile(
    r"^(?P<indent>\s*)(?P<fence>:{3,}|`{3,})"
    r"\{(?P<kind>[A-Za-z0-9_-]+)\}(?:\s+(?P<title>.*?))?\s*$"
)
OPT_RE = re.compile(r"^:([A-Za-z0-9_-]+):\s*(.*)$")
RST_ADM_RE = re.compile(
    r"^(?P<indent>\s*)\.\.\s+"
    r"(?P<kind>note|warning|tip|important|seealso|danger|caution)::"
    r"\s*(?P<title>.*?)\s*$",
    re.I,
)
RST_CODE_RE = re.compile(
    r"^(?P<indent>\s*)\.\.\s+code-block::\s*(?P<lang>\S*)\s*$",
    re.I,
)
RST_HIGHLIGHT_RE = re.compile(r"^(?P<indent>\s*)\.\.\s+highlight::\s*\S*\s*$", re.I)
RST_LIST_TABLE_RE = re.compile(
    r"^(?P<indent>\s*)\.\.\s+list-table::\s*(?P<title>.*?)\s*$",
    re.I,
)
MYST_TARGET_RE = re.compile(r"^\((?P<label>[A-Za-z0-9_.:-]+)\)=\s*$")
REF_RE = re.compile(r"(?:\{ref\}|:ref:)`(?P<target>[^`]+)`")
DOC_RE = re.compile(r":doc:`(?P<target>[^`]+)`")
PY_ROLE_RE = re.compile(
    r":(?:py:)?(?P<role>class|func|meth|mod|obj|attr|exc|data|const):`(?P<target>[^`]+)`"
)
MYST_ROLE_RE = re.compile(
    r"\{(?P<role>class|func|meth|mod|obj|attr|exc|data|const)\}`(?P<target>[^`]+)`"
)
CUSTOM_TARGET_RE = re.compile(
    r"^(?P<title>.+?)\s*(?:<|&lt;)(?P<target>[^>&]+)(?:>|&gt;)$"
)

CALLOUT_KINDS = {
    "note",
    "warning",
    "tip",
    "important",
    "seealso",
    "danger",
    "caution",
    "admonition",
    "dropdown",
}
TYPE_ALIASES = {"warn": "warning", "error": "danger", "hint": "tip", "dropdown": "note"}
KNOWN_TYPES = {
    "note",
    "abstract",
    "info",
    "tip",
    "success",
    "question",
    "warning",
    "failure",
    "danger",
    "bug",
    "example",
    "quote",
    "seealso",
    "important",
    "caution",
}
REF_TARGETS = {
    "automodel_userguide": (
        "AutoModels",
        "userguide/advanced/auto/#automodel_userguide",
    ),
    "batch_function_userguide": (
        "Batch Dimension",
        "userguide/advanced/batch/#batch_function_userguide",
    ),
    "building_documentation": (
        "Building Documentation",
        "userguide/developer/documentation/#building-documentation",
    ),
    "configuration_userguide": (
        "Configuration",
        "userguide/about/install/#configuration_userguide",
    ),
    "coordinates_userguide": (
        "Coordinate Systems",
        "userguide/about/overview/#coordinates_userguide",
    ),
    "data_userguide": ("Data Movement", "userguide/about/overview/#data_userguide"),
    "developer_overview": (
        "Developer Overview",
        "userguide/developer/overview/#developer_overview",
    ),
    "diagnostic_model_userguide": (
        "Diagnostic Models",
        "userguide/components/diagnostic/#diagnostic_model_userguide",
    ),
    "earth2studio.data.analysis": (
        "earth2studio.data.analysis",
        "modules/datasources_analysis/",
    ),
    "earth2studio.models.dx": ("earth2studio.models.dx", "modules/models_dx/"),
    "earth2studio.models.px": ("earth2studio.models.px", "modules/models_px/"),
    "earth2studio.perturbation": (
        "earth2studio.perturbation",
        "modules/perturbation/",
    ),
    "examples_userguide": ("Examples", "examples/"),
    "extension_examples": ("extension examples", "examples/08_extend/"),
    "install_guide": ("Install", "userguide/about/install/#install_guide"),
    "lexicon_userguide": (
        "Lexicon",
        "userguide/advanced/lexicon/#lexicon_userguide",
    ),
    "model_dependencies": (
        "Model Dependencies",
        "userguide/about/install/#model_dependencies",
    ),
    "optional_dependencies": (
        "Optional Dependencies",
        "userguide/about/install/#optional_dependencies",
    ),
    "prognostic_model_userguide": (
        "Prognostic Models",
        "userguide/components/prognostic/#prognostic_model_userguide",
    ),
    "pytorch_container_environment": (
        "Docker Container",
        "userguide/about/install/#pytorch_container_environment",
    ),
    "sphx_glr_examples_01_getting_started_03_ensemble_workflow.py": (
        "getting started ensemble workflow example",
        "examples/01_getting_started/03_ensemble_workflow/",
    ),
    "userguide": ("User Guide", "userguide/"),
}


def on_page_markdown(markdown: str, **kwargs: object) -> str:
    """Convert legacy Sphinx/MyST blocks before Python-Markdown renders pages."""
    return _convert_legacy_blocks(markdown, kwargs.get("page"))


def _relative_url(target: str, page: object | None) -> str:
    target = target.strip().lstrip("/")
    if not target or target.startswith(("#", "http://", "https://")):
        return target

    if target.endswith(".md"):
        target = target[:-3] + "/"
    elif (
        "#" not in target
        and not target.endswith("/")
        and "." not in target.rsplit("/", 1)[-1]
    ):
        target += "/"

    page_url = str(getattr(page, "url", "") or "")
    if not page_url:
        return target

    base = page_url if page_url.endswith("/") else dirname(page_url) + "/"
    relative = relpath(target, start=base)
    return "" if relative == "." else relative


def _split_role_target(value: str) -> tuple[str | None, str]:
    value = value.strip()
    custom = CUSTOM_TARGET_RE.match(value)
    if custom:
        return custom.group("title").strip(), custom.group("target").strip()
    return None, value


def _xref(label: str, page: object | None) -> str:
    title, target = _split_role_target(label)
    default_title, url = REF_TARGETS.get(target, (target, ""))
    if not url:
        return f"<code>{escape(title or default_title)}</code>"
    return (
        f'<a href="{escape(_relative_url(url, page), quote=True)}">'
        f"{escape(title or default_title)}</a>"
    )


def _doc_link(label: str, page: object | None) -> str:
    title, target = _split_role_target(label)
    if not title:
        title = target.strip("/").rsplit("/", 1)[-1].replace("-", " ").title()
    return (
        f'<a href="{escape(_relative_url(target, page), quote=True)}">'
        f"{escape(title)}</a>"
    )


def _python_role(label: str) -> str:
    title, target = _split_role_target(label)
    display = title or target.lstrip("~")
    display = escape(display).replace("_", "&#95;")
    return f"<code>{display}</code>"


def _convert_legacy_roles(line: str, page: object | None) -> str:
    line = REF_RE.sub(lambda match: _xref(match.group("target"), page), line)
    line = DOC_RE.sub(lambda match: _doc_link(match.group("target"), page), line)
    line = PY_ROLE_RE.sub(lambda match: _python_role(match.group("target")), line)
    return MYST_ROLE_RE.sub(lambda match: _python_role(match.group("target")), line)


def _active_indent(stack: list[dict[str, object]]) -> str:
    return " " * sum(int(item.get("indent", 0)) for item in stack)


def _clean_title(title: str | None) -> str:
    return (title or "").strip().strip('"')


def _parse_options(lines: list[str], start: int) -> tuple[dict[str, str], int]:
    options: dict[str, str] = {}
    i = start
    while i < len(lines):
        match = OPT_RE.match(lines[i].strip())
        if not match:
            break
        options[match.group(1).lower()] = match.group(2).strip()
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return options, i


def _admonition_type(
    kind: str, title: str, options: dict[str, str]
) -> tuple[str, str, str]:
    classes = {
        TYPE_ALIASES.get(class_name.lower(), class_name.lower())
        for class_name in options.get("class", "").split()
    }
    color = TYPE_ALIASES.get(
        options.get("color", "").lower(), options.get("color", "").lower()
    )

    if kind == "dropdown":
        return "???", color if color in KNOWN_TYPES else "note", title

    if kind == "admonition":
        marker = "???" if "dropdown" in classes else "!!!"
        typ = next(
            (
                class_name
                for class_name in classes
                if class_name in KNOWN_TYPES and class_name != "dropdown"
            ),
            "",
        )
        if not typ and title.lower() in KNOWN_TYPES:
            typ = title.lower()
        return marker, typ or "note", title

    return "!!!", TYPE_ALIASES.get(kind, kind), title


def _directive_line(marker: str, typ: str, title: str, prefix: str) -> str:
    if not title:
        return f"{prefix}{marker} {typ}"
    return f'{prefix}{marker} {typ} "{title.replace(chr(34), chr(92) + chr(34))}"'


def _consume_indented(
    lines: list[str], start: int, out: list[str], prefix: str, page: object | None
) -> int:
    i = start
    while i < len(lines) and not lines[i].strip():
        i += 1
    while i < len(lines):
        line = lines[i]
        if line.strip() and not line.startswith((" ", "\t")):
            break
        if line.strip():
            content = line[3:] if line.startswith("   ") else line.lstrip()
            out.append(prefix + "    " + _convert_legacy_roles(content, page))
        else:
            out.append("")
        i += 1
    return i


def _table_cell(text: str, page: object | None) -> str:
    text = _convert_legacy_roles(text.strip(), page)
    return text.replace("|", r"\|") or " "


def _render_list_table(
    title: str, table_lines: list[str], out: list[str], prefix: str, page: object | None
) -> None:
    rows: list[list[str]] = []
    for raw in table_lines:
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("* - "):
            rows.append([stripped[4:].strip()])
        elif stripped.startswith("- ") and rows:
            rows[-1].append(stripped[2:].strip())
        elif rows and rows[-1]:
            rows[-1][-1] = (rows[-1][-1] + " " + stripped).strip()

    if not rows:
        return

    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    if title:
        out.extend([prefix + f"**{title}**", ""])
    header = normalized[0]
    out.append(
        prefix + "| " + " | ".join(_table_cell(cell, page) for cell in header) + " |"
    )
    out.append(prefix + "| " + " | ".join("---" for _ in header) + " |")
    out.extend(
        (prefix + "| " + " | ".join(_table_cell(cell, page) for cell in row) + " |")
        for row in normalized[1:]
    )
    out.append("")


def _convert_rst_list_table(
    lines: list[str], i: int, out: list[str], prefix: str, page: object | None
) -> int:
    match = RST_LIST_TABLE_RE.match(lines[i])
    if match is None:
        return i

    title = _clean_title(match.group("title"))
    i += 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith(":"):
            i += 1
            continue
        break

    table_lines: list[str] = []
    while i < len(lines):
        raw = lines[i]
        if raw.strip() and raw[0] not in {" ", "\t"}:
            break
        table_lines.append(raw)
        i += 1

    _render_list_table(title, table_lines, out, prefix, page)
    return i


def _convert_rst_directive(
    lines: list[str], i: int, out: list[str], prefix: str, page: object | None
) -> int | None:
    if RST_HIGHLIGHT_RE.match(lines[i]):
        return i + 1

    if RST_LIST_TABLE_RE.match(lines[i]):
        return _convert_rst_list_table(lines, i, out, prefix, page)

    admonition = RST_ADM_RE.match(lines[i])
    if admonition:
        kind = admonition.group("kind").lower()
        title = _clean_title(admonition.group("title"))
        out.append(_directive_line("!!!", kind, title, prefix))
        return _consume_indented(lines, i + 1, out, prefix, page)

    code = RST_CODE_RE.match(lines[i])
    if code:
        lang = code.group("lang") or ""
        out.append(f"{prefix}```{lang}")
        i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1
        while i < len(lines):
            line = lines[i]
            if line.strip() and not line.startswith((" ", "\t")):
                break
            content = line[3:] if line.startswith("   ") else line.lstrip()
            out.append(prefix + content)
            i += 1
        out.append(f"{prefix}```")
        return i

    return None


def _convert_legacy_blocks(markdown: str, page: object | None = None) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    stack: list[dict[str, object]] = []
    in_code_fence = False
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        prefix = _active_indent(stack)

        if stack and stripped == stack[-1]["close"]:
            stack.pop()
            i += 1
            continue

        block = OPEN_RE.match(lines[i])
        if not in_code_fence and block:
            kind = block.group("kind").lower()
            fence = block.group("fence")
            close = fence[0] * len(fence)
            title = _clean_title(block.group("title"))

            if kind == "tab-set":
                stack.append({"kind": kind, "close": close, "indent": 0})
                i += 1
                continue

            if kind == "tab-item":
                out.append(f'{prefix}=== "{title}"')
                stack.append({"kind": kind, "close": close, "indent": 4})
                i += 1
                continue

            if kind == "list-table":
                _options, i = _parse_options(lines, i + 1)
                table_lines: list[str] = []
                while i < len(lines) and lines[i].strip() != close:
                    table_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    i += 1
                _render_list_table(title, table_lines, out, prefix, page)
                continue

            if kind in CALLOUT_KINDS:
                options, i = _parse_options(lines, i + 1)
                marker, typ, final_title = _admonition_type(kind, title, options)
                out.append(_directive_line(marker, typ, final_title, prefix))
                stack.append({"kind": kind, "close": close, "indent": 4})
                continue

        if not stack and stripped.startswith("```"):
            in_code_fence = not in_code_fence
            out.append(lines[i])
            i += 1
            continue

        if not in_code_fence:
            target = MYST_TARGET_RE.match(stripped)
            if target:
                out.append(f'<a id="{escape(target.group("label"), quote=True)}"></a>')
                i += 1
                continue

            next_i = _convert_rst_directive(lines, i, out, prefix, page)
            if next_i is not None:
                i = next_i
                continue

        line = lines[i] if in_code_fence else _convert_legacy_roles(lines[i], page)
        out.append(prefix + line)
        i += 1

    return "\n".join(out) + ("\n" if markdown.endswith("\n") else "")

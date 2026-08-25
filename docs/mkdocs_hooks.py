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

"""MkDocs hooks for Earth2Studio documentation components."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from html import escape
from pathlib import Path
from posixpath import dirname, relpath

from mkdocs_badges.render import DEFAULT_COLOR, resolve_badge

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
RST_INLINE_ROLE_RE = re.compile(
    r":(?:(?P<domain>[A-Za-z][\w-]*):)?"
    r"(?P<role>func|meth|class|attr|mod|data|obj|exc):`(?P<target>[^`]+)`"
)
MARKDOWN_CODE_XREF_RE = re.compile(r"\[`([^`]+)`\]\[([^\]]+)\]")
DOCS_ROOT = Path(__file__).resolve().parent
GENERATED_API_ROOT = DOCS_ROOT / "modules" / "generated"
INSTALL_SELECTOR_MARKER = "<!-- e2s-install-selector -->"
INSTALL_SELECTOR_CONFIG = DOCS_ROOT / "userguide" / "about" / "install_options.yml"
CATALOG_MARKER = "<!-- e2s-catalog -->"
EXAMPLES_GALLERY_DESCRIPTION = (
    "Runnable examples, grouped by topic. Each card opens the complete source, "
    "output, and captured figures."
)
SCORECARD_PLOT_IFRAME = (
    '<iframe data-e2s-scorecard-plot src="../../_static/scorecard/plot.html?'
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


def on_page_markdown(markdown: str, **kwargs: object) -> str:
    """Convert legacy Sphinx/MyST blocks before Python-Markdown renders pages."""
    page = kwargs.get("page")
    markdown = _rewrite_scorecard_plot_url(markdown)
    markdown = _render_install_selector(markdown)
    markdown = _render_catalog(markdown, page)
    markdown = _remove_examples_gallery_description(markdown, page)
    return _convert_legacy_blocks(markdown, page)


def _rewrite_scorecard_plot_url(markdown: str) -> str:
    """Adjust the marked scorecard iframe path for a Mike build."""
    if not os.getenv("MIKE_DOCS_VERSION"):
        return markdown
    return markdown.replace(
        SCORECARD_PLOT_IFRAME,
        "<iframe data-e2s-scorecard-plot " 'src="../../../_static/scorecard/plot.html?',
    )


def _remove_examples_gallery_description(markdown: str, page: object | None) -> str:
    """Remove the generated examples index description."""
    if str(getattr(page, "url", "") or "") != "examples/":
        return markdown
    return markdown.replace(f"\n{EXAMPLES_GALLERY_DESCRIPTION}\n", "\n")


def _render_install_selector(markdown: str) -> str:
    if INSTALL_SELECTOR_MARKER not in markdown:
        return markdown

    import yaml

    data = yaml.safe_load(INSTALL_SELECTOR_CONFIG.read_text(encoding="utf-8"))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = (
        '<section class="e2s-install-selector" data-e2s-install-selector>'
        '<script type="application/json" data-e2s-install-data>'
        f"{payload}"
        "</script>"
        '<div class="e2s-install-selector__layout">'
        '<div class="e2s-install-selector__controls" data-e2s-install-controls></div>'
        '<div class="e2s-install-selector__output" data-e2s-install-output></div>'
        "</div>"
        "</section>"
    )
    return markdown.replace(INSTALL_SELECTOR_MARKER, html)


def _render_catalog(markdown: str, page: object | None) -> str:
    if CATALOG_MARKER not in markdown:
        return markdown

    records = _catalog_records(page)
    payload = json.dumps(records, separators=(",", ":")).replace("</", "<\\/")
    html = (
        '<section class="e2s-catalog" data-e2s-catalog>'
        '<script type="application/json" data-e2s-catalog-data>'
        f"{payload}"
        "</script>" + _catalog_fallback(records) + "</section>"
    )
    return markdown.replace(CATALOG_MARKER, html)


def _catalog_fallback(records: list[dict[str, object]]) -> str:
    models = [record for record in records if record.get("kind") == "model"][:24]
    data_sources = [record for record in records if record.get("kind") == "data"][:24]
    return (
        '<div class="e2s-catalog-fallback">'
        '<div class="e2s-catalog-fallback__group">'
        "<h2>Models</h2>" + _catalog_fallback_cards(models) + "</div>"
        '<div class="e2s-catalog-fallback__group">'
        "<h2>Data Sources</h2>" + _catalog_fallback_cards(data_sources) + "</div>"
        "</div>"
    )


def _catalog_fallback_cards(records: list[dict[str, object]]) -> str:
    cards = []
    for record in records:
        title = escape(str(record.get("title") or "Catalog entry"))
        summary = escape(str(record.get("summary") or ""))
        url = escape(str(record.get("url") or "#"), quote=True)
        chips = "".join(
            f"<span>{escape(str(chip))}</span>"
            for chip in list(record.get("chips") or ())[:4]
        )
        cards.append(
            '<article class="e2s-catalog-card" data-kind="'
            + escape(str(record.get("kind") or "model"), quote=True)
            + '" data-tone="'
            + escape(str(record.get("tone") or "model"), quote=True)
            + '">'
            '<div class="e2s-catalog-card__art" aria-hidden="true"><span></span></div>'
            '<div class="e2s-catalog-card__body">'
            f'<h3><a href="{url}">{title}</a></h3>'
            f"<p>{summary}</p>"
            f'<div class="e2s-catalog-card__chips">{chips}</div>'
            "</div>"
            "</article>"
        )
    return '<div class="e2s-catalog-list">' + "".join(cards) + "</div>"


def _front_matter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}
    try:
        import yaml

        _start, raw, _body = text.split("---", 2)
        return yaml.safe_load(raw) or {}
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _mkdocs_badges_config() -> dict[str, object]:
    """Return the configured mkdocs-badges plugin options."""
    try:
        import yaml

        config_text = (DOCS_ROOT.parent / "mkdocs.yml").read_text(encoding="utf-8")
        config = yaml.safe_load(re.sub(r"!!python/name:\S+", "null", config_text)) or {}
    except (OSError, ValueError, yaml.YAMLError):
        return {}

    plugins = config.get("plugins", [])
    if not isinstance(plugins, list):
        return {}
    for plugin in plugins:
        if isinstance(plugin, dict) and "badges" in plugin:
            badges = plugin.get("badges")
            return badges if isinstance(badges, dict) else {}
    return {}


def _mkdocs_badge_definitions() -> dict[str, dict[str, object]]:
    """Return badge definitions using the mkdocs-badges config schema."""
    definitions = _mkdocs_badges_config().get("definitions", {})
    return definitions if isinstance(definitions, dict) else {}


def _mkdocs_badge_default_color() -> str:
    """Return the configured mkdocs-badges fallback color."""
    default_color = _mkdocs_badges_config().get("default_color", DEFAULT_COLOR)
    return str(default_color)


def _catalog_records(page: object | None) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(GENERATED_API_ROOT.rglob("*.md")):
        parts = path.relative_to(GENERATED_API_ROOT).parts
        if len(parts) < 2 or parts[0] not in {"models", "data"}:
            continue

        meta = _front_matter(path)
        title = str(meta.get("title") or path.stem)
        badges = [str(item) for item in meta.get("badges", []) if isinstance(item, str)]
        kind, group = _catalog_kind_group(parts)
        if kind is None:
            continue

        records.append(
            {
                "kind": kind,
                "group": group,
                "title": title.removeprefix("data."),
                "summary": str(meta.get("summary") or "API reference page."),
                "signature": str(meta.get("signature") or ""),
                "url": _relative_url(
                    path.relative_to(DOCS_ROOT).with_suffix("").as_posix() + "/",
                    page,
                ),
                "badges": badges,
                "chips": _catalog_chips(kind, group, badges),
                "filters": _catalog_filters(kind, badges),
                "tone": _catalog_tone(badges, kind),
            }
        )
    return records


def _catalog_kind_group(parts: tuple[str, ...]) -> tuple[str | None, str]:
    if parts[0] == "models":
        return "model", {
            "px": "Prognostic",
            "dx": "Diagnostic",
            "da": "Data Assimilation",
        }.get(parts[1], parts[1].replace("_", " ").title())
    if parts[0] == "data":
        return "data", {
            "analysis": "Data Source",
            "forecast": "Forecast Data Source",
            "dataframe": "DataFrame Source",
        }.get(parts[1], parts[1].replace("_", " ").title())
    return None, ""


def _catalog_filters(kind: str, badges: list[str]) -> dict[str, list[str]]:
    prefixes = (
        (
            ("class", "class"),
            ("provider", "provider"),
            ("backend", "backend"),
            ("product", "product"),
            ("region", "region"),
            ("gpu", "gpu"),
            ("year", "year"),
        )
        if kind == "model"
        else (
            ("dataclass", "data class"),
            ("product", "product"),
            ("region", "region"),
            ("gpu", "gpu"),
            ("year", "year"),
        )
    )
    filters = {
        label: [
            _catalog_filter_label(label, badge)
            for badge in badges
            if badge.startswith(f"{prefix}:")
        ]
        for prefix, label in prefixes
    }
    return {
        key: [value for value in values if value] for key, values in filters.items()
    }


def _catalog_chips(kind: str, group: str, badges: list[str]) -> list[str]:
    prefixes = (
        ("class", "provider", "backend", "product")
        if kind == "model"
        else (
            "dataclass",
            "product",
        )
    )
    chips = [group]
    for prefix in prefixes:
        chips.extend(
            _catalog_filter_label(prefix, badge)
            for badge in badges
            if badge.startswith(f"{prefix}:")
        )
    return list(dict.fromkeys(chip for chip in chips if chip))[:6]


def _catalog_filter_label(group: str, badge: str) -> str:
    """Return display label for a catalog filter value."""
    return _badge_label(badge)


def _badge_label(badge: str) -> str:
    """Return the configured display label for a badge ID."""
    resolved = resolve_badge(
        badge,
        _mkdocs_badge_definitions(),
        _mkdocs_badge_default_color(),
    )
    name = getattr(resolved, "name", "")
    if name:
        return name
    if resolved.label:
        return resolved.label
    if resolved.tooltip:
        return resolved.tooltip
    return resolve_badge(badge, {}, _mkdocs_badge_default_color()).label


def _catalog_tone(badges: list[str], kind: str) -> str:
    tones = (
        ("product:radar", "radar"),
        ("product:sat", "satellite"),
        ("product:solar", "solar"),
        ("product:ocean", "ocean"),
        ("product:precip", "precip"),
        ("class:data-assimilation", "assimilation"),
        ("class:downscaling", "downscaling"),
        ("dataclass:observation", "observation"),
        ("dataclass:reanalysis", "reanalysis"),
        ("dataclass:simulation", "simulation"),
    )
    for badge, tone in tones:
        if badge in badges:
            return tone
    return "model" if kind == "model" else "data"


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
            out.append(prefix + "    " + content)
        else:
            out.append("")
        i += 1
    return i


def _table_cell(text: str, page: object | None) -> str:
    text = text.strip()
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


def _is_class_like_xref(label: str) -> bool:
    """Return whether a Markdown xref label looks like a class/type."""
    name = label.split(".")[-1].lstrip("~")
    stripped = name.lstrip("_")
    return bool(stripped) and stripped[0].isupper()


def _legacy_role_label(target: str) -> str:
    """Return the displayed text for a Sphinx inline role as inline code."""
    target = target.strip()
    explicit = re.match(r"(?P<label>.+?)\s+<(?P<target>[^<>]+)>", target)
    if explicit:
        return explicit.group("label").strip()
    if target.startswith("~"):
        return target[1:].rsplit(".", 1)[-1]
    return target


def _inline_code(label: str) -> str:
    """Return Markdown inline code for label text."""
    escaped = label.replace("`", r"\`")
    return f"`{escaped}`"


def _convert_legacy_inline_roles(markdown: str) -> str:
    """Render unsupported Sphinx inline roles as Markdown inline code."""

    def replace(match: re.Match[str]) -> str:
        return _inline_code(_legacy_role_label(match.group("target")))

    return RST_INLINE_ROLE_RE.sub(replace, markdown)


def _convert_markdown_code_xrefs(markdown: str) -> str:
    """Keep class code xrefs, but render method/function xrefs as inline code."""

    def replace(match: re.Match[str]) -> str:
        label = match.group(1)
        return match.group(0) if _is_class_like_xref(label) else _inline_code(label)

    return MARKDOWN_CODE_XREF_RE.sub(replace, markdown)


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

        line = prefix + lines[i]
        if not in_code_fence:
            line = _convert_legacy_inline_roles(line)
            line = _convert_markdown_code_xrefs(line)
        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if markdown.endswith("\n") else "")

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
"""Griffe extensions for Earth2Studio API documentation."""

from __future__ import annotations

import re

import griffe

BADGE_RE = re.compile(
    r"\b(?:region|class|dataclass|year|product|gpu):[A-Za-z0-9_.-]+\b"
)
RST_ROLE_RE = re.compile(
    r":(?:py:)?(?P<role>mod|class|func|obj|meth|attr|exc|data|const):"
    r"`(?P<target>[^`]+)`"
)
RST_REF_RE = re.compile(r":ref:`(?P<target>[^`]+)`")
RST_LINK_RE = re.compile(r"`(?P<title>[^`<]+?)\s*<(?P<url>[^`>]+)>`_")
DIRECTIVE_RE = re.compile(r"(?P<indent>\s*)\.\. (?P<name>[a-z-]+)::\s*(?P<arg>.*)")

ADMONITIONS = {
    "attention",
    "caution",
    "danger",
    "error",
    "hint",
    "important",
    "note",
    "tip",
    "warning",
}
EXTERNAL_REF_PREFIXES = (
    "contextlib.",
    "fsspec.",
    "numpy.",
    "obstore.",
    "pandas.",
    "torch.",
    "xarray.",
)


def _visible_name(target: str) -> str:
    """Return the compact name to show for a cross-reference target."""
    if target.startswith("~"):
        target = target[1:]
    return target.rsplit(".", 1)[-1]


def _split_role_target(target: str) -> tuple[str, str]:
    """Split Sphinx role text into display text and reference target."""
    match = re.fullmatch(r"(?P<title>.+?)\s*<(?P<target>.+)>", target.strip())
    if match:
        return match.group("title"), match.group("target")
    if target.startswith("~"):
        return _visible_name(target), target[1:]
    return _visible_name(target), target


def _format_role(match: re.Match[str]) -> str:
    """Convert a simple Sphinx role into Markdown."""
    title, target = _split_role_target(match.group("target"))
    target = target.strip()
    if target.startswith("earth2studio."):
        return f"[{title}][{target}]"
    if target.startswith(EXTERNAL_REF_PREFIXES) and " " not in target:
        return f"[`{title}`][{target}]"
    return f"`{title}`"


def _format_ref(match: re.Match[str]) -> str:
    """Convert a simple Sphinx ref role into readable Markdown text."""
    title, target = _split_role_target(match.group("target"))
    if title != target:
        return title
    return target.replace("_", " ")


def _convert_roles(docstring: str) -> str:
    """Convert common Sphinx inline markup to Markdown."""
    docstring = RST_LINK_RE.sub(r"[\g<title>](\g<url>)", docstring)
    docstring = RST_ROLE_RE.sub(_format_role, docstring)
    return RST_REF_RE.sub(_format_ref, docstring)


def _indented_after(line: str, indent: int) -> bool:
    """Return True when a line belongs to an indented directive body."""
    return not line.strip() or len(line) - len(line.lstrip()) > indent


def _collect_indented_block(
    lines: list[str], index: int, indent: int
) -> tuple[list[str], int]:
    """Collect and dedent a directive body."""
    while index < len(lines) and not lines[index].strip():
        index += 1

    block: list[str] = []
    while index < len(lines) and _indented_after(lines[index], indent):
        block.append(lines[index])
        index += 1

    nonempty = [line for line in block if line.strip()]
    if not nonempty:
        return [], index
    base = min(len(line) - len(line.lstrip()) for line in nonempty)
    return [line[base:] if len(line) >= base else line for line in block], index


def _convert_directives(docstring: str) -> str:
    """Convert common block-level Sphinx directives to Markdown."""
    lines = docstring.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        match = DIRECTIVE_RE.fullmatch(line)
        if match is None:
            output.append(line)
            index += 1
            continue

        indent_text = match.group("indent")
        indent = len(indent_text)
        name = match.group("name")
        arg = match.group("arg").strip()

        if name == "highlight":
            index += 1
            continue

        if name in {"code", "code-block"}:
            block, index = _collect_indented_block(lines, index + 1, indent)
            language = arg or "python"
            output.append(f"{indent_text}```{language}")
            output.extend(indent_text + item if item else "" for item in block)
            output.append(f"{indent_text}```")
            continue

        if name in ADMONITIONS:
            block, index = _collect_indented_block(lines, index + 1, indent)
            output.append(f"{indent_text}!!! {name}")
            if arg:
                output.append(f"{indent_text}    {arg}")
            output.extend(f"{indent_text}    {item}" if item else "" for item in block)
            continue

        output.append(line)
        index += 1
    return "\n".join(output)


def _is_doctest_line(line: str) -> bool:
    """Return True when a line starts a doctest prompt."""
    stripped = line.lstrip()
    return stripped.startswith(">>>") or stripped.startswith("...")


def _convert_doctest_blocks(docstring: str) -> str:
    """Fence doctest examples so Markdown does not parse them as prose."""
    lines = docstring.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not _is_doctest_line(line):
            output.append(line)
            index += 1
            continue

        indent = line[: len(line) - len(line.lstrip())]
        output.append(f"{indent}```pycon")
        while index < len(lines):
            current = lines[index]
            if _is_doctest_line(current):
                output.append(current)
                index += 1
                continue
            if not current.strip() and index + 1 < len(lines):
                if _is_doctest_line(lines[index + 1]):
                    output.append(current)
                    index += 1
                    continue
            break
        output.append(f"{indent}```")
    return "\n".join(output)


def _strip_badges_section(docstring: str) -> str:
    """Remove Earth2Studio badge metadata before docstring rendering."""
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


class StripBadgesSection(griffe.Extension):
    """Strip the package badge section from rendered docstrings."""

    def on_object(self, *, obj: griffe.Object, **kwargs: object) -> None:
        """Update collected docstrings in place before mkdocstrings renders them."""
        if not obj.docstring:
            return
        docstring = _strip_badges_section(obj.docstring.value)
        docstring = _convert_roles(docstring)
        docstring = _convert_directives(docstring)
        obj.docstring.value = _convert_doctest_blocks(docstring)

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
RST_INLINE_ROLE_RE = re.compile(
    r":(?:(?P<domain>[A-Za-z][\w-]*):)?"
    r"(?P<role>func|meth|class|attr|mod|data|obj|exc):`(?P<target>[^`]+)`"
)


def _legacy_role_label(target: str) -> tuple[str, str]:
    """Return the display label and target for a Sphinx inline role."""
    target = target.strip()
    explicit = re.match(r"(?P<label>.+?)\s+<(?P<target>[^<>]+)>", target)
    if explicit:
        return explicit.group("label").strip(), explicit.group("target").strip().lstrip(
            "~"
        )
    if target.startswith("~"):
        normalized = target[1:]
        return normalized.rsplit(".", 1)[-1], normalized
    return target, target


def _inline_code(label: str) -> str:
    """Return Markdown inline code for label text."""
    escaped = label.replace("`", r"\`")
    return f"`{escaped}`"


def _convert_inline_roles(docstring: str) -> str:
    """Render supported Sphinx inline roles for Markdown API docs."""

    def replace(match: re.Match[str]) -> str:
        raw_target = match.group("target").strip()
        label, target = _legacy_role_label(raw_target)
        if raw_target.startswith("~earth2studio."):
            return f"[`{label}`][{target}]"
        return _inline_code(label)

    return RST_INLINE_ROLE_RE.sub(replace, docstring)


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
        docstring = _convert_inline_roles(docstring)
        obj.docstring.value = _convert_doctest_blocks(docstring)

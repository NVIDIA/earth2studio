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
"""Repair API cross-references left literal by the Zensical mkdocstrings path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mkdocs_hooks import _convert_docstring_xrefs

DOCS_ROOT = Path(__file__).resolve().parent
SITE_ROOT = DOCS_ROOT.parent / "site"


@dataclass(frozen=True)
class _Page:
    """Minimal page object needed by the shared URL helper."""

    url: str


def _page_url(path: Path) -> str:
    """Return the MkDocs-style URL for a generated HTML page."""
    relative = path.relative_to(SITE_ROOT)
    if relative.name == "index.html":
        return relative.parent.as_posix().rstrip("/") + "/"
    return relative.with_suffix("").as_posix() + "/"


def main() -> None:
    """Rewrite leftover docstring cross-reference syntax in generated HTML."""
    for path in SITE_ROOT.rglob("*.html"):
        html = path.read_text(encoding="utf-8")
        updated = _convert_docstring_xrefs(html, _Page(_page_url(path)))
        if updated != html:
            path.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()

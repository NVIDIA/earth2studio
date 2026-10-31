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

"""Convert gallery examples (``# %%`` percent-format scripts) to Fern MDX.

Each ``examples/<section>/<name>.py`` becomes ``examples/<section>/<name>.mdx`` next to
the script; each ``README.rst`` becomes ``README.mdx``; ``examples/index.mdx`` is a
card index of every section. Only the sections listed in ``SECTIONS`` (the pilot) are
converted, the others link to the legacy site.

Script conventions handled
--------------------------
* First cell: docstring whose first line + ``====`` underline is the page title.
* ``# /// script`` (PEP 723) block: shown in a "Script dependencies" accordion.
* Comment-only cells are markdown; ``# Title`` / ``# -----`` becomes a ``##`` heading.
* Other cells are Python code blocks.
* MkDocs ``!!! note`` admonitions become Fern callouts.
* ``[text][earth2studio.x.Y]`` autorefs and legacy ``/modules/...#earth2studio.x.Y``
  links are resolved against the generated API reference in ``docs/api``.

Run ``fern/scripts/generate_api.py`` first so ``docs/api`` exists, then::

    python fern/scripts/convert_examples.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_api import Resolver  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
API = ROOT / "docs" / "api"

SITE = "/earth2studio"
LEGACY = "https://nvidia.github.io/earth2studio/main"
GITHUB = "https://github.com/NVIDIA/earth2studio/blob/main"

# Sections converted so far. Everything else in examples/ links to the legacy site.
SECTIONS = ["01_getting_started", "08_extend"]

CALLOUT = {
    "note": "Note",
    "info": "Info",
    "tip": "Tip",
    "hint": "Tip",
    "example": "Info",
    "abstract": "Info",
    "warning": "Warning",
    "caution": "Warning",
    "danger": "Warning",
    "important": "Warning",
}

unresolved: list[str] = []


# --------------------------------------------------------------------------- #
# API link resolution
# --------------------------------------------------------------------------- #
def build_resolver() -> Resolver:
    if not API.exists():
        sys.exit("docs/api not found; run fern/scripts/generate_api.py first.")
    files = {}
    for mdx in API.rglob("*.mdx"):
        files[mdx] = re.findall(r'<Anchor id="([^"]+)"', mdx.read_text())
    return Resolver(files)


def api_url(resolver: Resolver, target: str) -> str | None:
    hit = resolver.find(target)
    if not hit:
        return None
    fqn, path = hit
    rel = path.relative_to(API).with_suffix("").as_posix()
    url = api_urls().get(rel)
    if not url:
        return None
    return f"{url}#{fqn.replace('.', '-')}"


_API_URLS: dict[str, str] = {}


def api_urls() -> dict[str, str]:
    """Page -> URL map written by generate_api.py (the nav groups pages like MkDocs)."""
    if not _API_URLS:
        f = Path(__file__).resolve().parent / "api_urls.json"
        if not f.exists():
            sys.exit("fern/scripts/api_urls.json not found; run generate_api.py first.")
        _API_URLS.update(json.loads(f.read_text()))
    return _API_URLS


# --------------------------------------------------------------------------- #
# Markdown -> MDX helpers
# --------------------------------------------------------------------------- #
CODESPAN = re.compile(r"(`[^`\n]*`)")
AUTOREF = re.compile(r"\[([^\]\n]+)\]\[([^\]\n]*)\]")
LEGACY_API_LINK = re.compile(
    r"\[([^\]\n]+)\]\(https://nvidia\.github\.io/earth2studio/(?:main|latest)/modules/"
    r"[^)#\s]*#([A-Za-z0-9_.]+)\)"
)
LEGACY_ANY = re.compile(r"\(https://nvidia\.github\.io/earth2studio/(?:main|latest)/([^)\s]*)\)")


_HEADING_IDS: dict[str, str] = {}


def heading_ids() -> dict[str, str]:
    """Map MkDocs custom heading ids (``## Title { #my_id }``) to Fern's heading slugs."""
    if not _HEADING_IDS:
        for md in (ROOT / "docs" / "userguide").rglob("*.md"):
            for m in re.finditer(r"^#+\s+(.*?)\s*\{\s*#([\w-]+)\s*\}\s*$", md.read_text(), re.M):
                slug = re.sub(r"[^a-z0-9]+", "-", m.group(1).lower()).strip("-")
                _HEADING_IDS[m.group(2)] = slug
    return _HEADING_IDS


def convert_links(text: str, resolver: Resolver) -> str:
    def autoref(m: re.Match) -> str:
        label, target = m.group(1), m.group(2).strip() or m.group(1).strip("`")
        url = api_url(resolver, target)
        if not url:
            unresolved.append(target)
            return label if label.startswith("`") else f"`{label}`"
        return f"[{label}]({url})"

    def legacy_api(m: re.Match) -> str:
        label, target = m.group(1), m.group(2)
        url = api_url(resolver, target)
        if not url:
            unresolved.append(target)
            return f"[{label}]({LEGACY}/modules/)"
        return f"[{label}]({url})"

    def userguide(m: re.Match) -> str:
        page = m.group(1)
        # about/install and about/overview were promoted to top-level pages.
        page = re.sub(r"^about/(install|overview)$", r"\1", page)
        frag = (m.group(2) or "").lstrip("#")
        frag = "#" + heading_ids().get(frag, frag) if frag else ""
        return f"]({SITE}/{page}{frag})"

    text = re.sub(r"\]\((?:\.\./)+userguide/([^)#\s]+?)\.md(#[^)\s]*)?\)", userguide, text)
    text = LEGACY_API_LINK.sub(legacy_api, text)
    text = AUTOREF.sub(autoref, text)
    return text


def escape_prose(line: str) -> str:
    parts = CODESPAN.split(line)
    for i in range(0, len(parts), 2):  # even parts are outside code spans
        p = parts[i]
        p = re.sub(r"<(https?://[^>\s]+)>", r"[\1](\1)", p)
        p = p.replace("{", "\\{").replace("}", "\\}")
        p = re.sub(r"<(?!/?(?:Note|Info|Tip|Warning|Accordion|Card|CardGroup|Tabs|Tab)\b)", "&lt;", p)
        parts[i] = p
    return "".join(parts)


def markdown_to_mdx(lines: list[str], resolver: Resolver) -> list[str]:
    """Convert a block of mkdocs-flavoured markdown lines to MDX."""
    out: list[str] = []
    i = 0
    in_fence = False
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            i += 1
            continue
        if in_fence:
            out.append(line)
            i += 1
            continue
        # Admonition: "!!! note ["Title"]" followed by 4-space indented body.
        m = re.match(r'^!!!\s+(\w+)(?:\s+"([^"]*)")?\s*$', line)
        if m:
            kind = CALLOUT.get(m.group(1).lower(), "Note")
            title = m.group(2)
            body: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith("    ") or not lines[i].strip()):
                body.append(lines[i][4:] if lines[i].startswith("    ") else "")
                i += 1
            while body and not body[-1].strip():
                body.pop()
            out += ["", f"<{kind}>", ""]
            if title:
                out += [f"**{escape_prose(title)}**", ""]
            out += markdown_to_mdx(body, resolver)
            out += ["", f"</{kind}>", ""]
            continue
        # Setext heading: "Title" underlined with ---- / ====.
        if (
            i + 1 < len(lines)
            and line.strip()
            and re.fullmatch(r"-{3,}|={3,}|~{3,}", lines[i + 1].strip() or "x")
            and not line.lstrip().startswith(("-", "*", "|"))
        ):
            level = "##" if lines[i + 1].strip()[0] in "-=" else "###"
            out += ["", f"{level} {escape_prose(line.strip())}", ""]
            i += 2
            continue
        out.append(escape_prose(convert_links(line, resolver)))
        i += 1
    return out


def tidy(lines: list[str]) -> list[str]:
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n").split("\n")


# --------------------------------------------------------------------------- #
# Script parsing
# --------------------------------------------------------------------------- #
CELL = re.compile(r"^# %%")


def split_cells(src: str) -> list[list[str]]:
    cells: list[list[str]] = []
    cur: list[str] | None = None
    for line in src.split("\n"):
        if CELL.match(line):
            if cur is not None:
                cells.append(cur)
            cur = []
        elif cur is not None:
            cur.append(line)
    if cur is not None:
        cells.append(cur)
    return cells


def is_comment_cell(cell: list[str]) -> bool:
    body = [l for l in cell if l.strip()]
    return bool(body) and all(l.lstrip().startswith("#") for l in body)


def uncomment(cell: list[str]) -> list[str]:
    return [re.sub(r"^# ?", "", l) for l in cell]


def convert_script(path: Path, resolver: Resolver) -> str:
    src = path.read_text()
    cells = split_cells(src)
    if not cells:
        raise ValueError(f"{path}: no '# %%' cells")

    # First cell: docstring (title + intro) and the PEP 723 block.
    first = "\n".join(cells[0])
    m = re.search(r'"""(.*?)"""', first, flags=re.S)
    if not m:
        raise ValueError(f"{path}: first cell has no docstring")
    doc = m.group(1).strip("\n").split("\n")
    title = doc[0].strip()
    intro = doc[2:] if len(doc) > 1 and re.fullmatch(r"=+", doc[1].strip()) else doc[1:]
    pep = re.search(r"^# /// script\n(.*?)^# ///\s*$", first, flags=re.S | re.M)

    rel = path.relative_to(ROOT).as_posix()
    out: list[str] = ["---", f"title: {title}", "---", ""]
    out += markdown_to_mdx(intro, resolver)
    out += [
        "",
        "<Info>",
        "",
        f"Source: [`{rel}`]({GITHUB}/{rel})",
        "",
        "</Info>",
        "",
    ]
    if pep:
        block = "# /// script\n" + pep.group(1) + "# ///"
        out += [
            '<Accordion title="Script dependencies">',
            "",
            "This script declares its dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)), "
            "so it can be run with `uv run`.",
            "",
            "```python",
            *block.rstrip().split("\n"),
            "```",
            "",
            "</Accordion>",
            "",
        ]

    for cell in cells[1:]:
        while cell and not cell[-1].strip():
            cell = cell[:-1]
        while cell and not cell[0].strip():
            cell = cell[1:]
        if not cell:
            continue
        if is_comment_cell(cell):
            out += [""] + markdown_to_mdx(uncomment(cell), resolver) + [""]
        else:
            out += ["", "```python", *cell, "```", ""]
    return "\n".join(tidy(out)) + "\n"


# --------------------------------------------------------------------------- #
# README.rst + index
# --------------------------------------------------------------------------- #
def read_readme(path: Path) -> tuple[str, str]:
    lines = [l for l in path.read_text().split("\n") if not l.startswith(".. _")]
    lines = [l for l in lines]
    while lines and not lines[0].strip():
        lines.pop(0)
    title = lines[0].strip()
    body = [l for l in lines[2:]]
    desc = " ".join(" ".join(body).split())
    return title, desc


def convert_readme(path: Path) -> str:
    title, desc = read_readme(path)
    return f"---\ntitle: {title}\n---\n\n{escape_prose(desc)}\n"


def build_index(resolver: Resolver) -> str:
    cards = []
    for sec in sorted(p.parent.name for p in EXAMPLES.glob("*/README.rst")):
        title, desc = read_readme(EXAMPLES / sec / "README.rst")
        if sec in SECTIONS:
            href = f"{SITE}/examples/{sec}"
            extra = ""
        else:
            href = f"{LEGACY}/examples/"
            extra = " (on the legacy site)"
        cards.append(
            f'  <Card title="{title}" href="{href}">\n    {escape_prose(desc)}{extra}\n  </Card>'
        )
    return (
        "---\ntitle: Examples\n---\n\n"
        "Runnable examples that show how to use Earth2Studio, from a first forecast to custom "
        "models and data sources. Every example is a Python script that can be found in the "
        f"[`examples/`]({GITHUB}/examples) folder of the repository.\n\n"
        "<CardGroup cols={2}>\n" + "\n".join(cards) + "\n</CardGroup>\n"
    )


def main() -> None:
    resolver = build_resolver()
    written = []
    for sec in SECTIONS:
        d = EXAMPLES / sec
        (d / "README.mdx").write_text(convert_readme(d / "README.rst"))
        written.append(d / "README.mdx")
        for py in sorted(d.glob("[0-9]*.py")):
            out = py.with_suffix(".mdx")
            out.write_text(convert_script(py, resolver))
            written.append(out)
    idx = EXAMPLES / "index.mdx"
    idx.write_text(build_index(resolver))
    written.append(idx)
    for w in written:
        print("wrote", w.relative_to(ROOT))
    if unresolved:
        print("UNRESOLVED API targets:", sorted(set(unresolved)))


if __name__ == "__main__":
    main()

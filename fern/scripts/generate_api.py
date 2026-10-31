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

"""Generate the Fern API reference (docs/api/*.mdx) from the earth2studio package.

Pipeline
--------
1. Stage a copy of ``earth2studio/`` in ``.api-build/src`` and add an empty
   ``models/__init__.py``. The real package has none (``models`` is a namespace
   package), which makes Fern's Python parser skip ``earth2studio.models`` entirely.
2. Run ``fern docs md generate --local`` in a throw-away Fern project
   (``.api-build/fern``) so the real ``fern/docs.yml`` needs no ``libraries:`` block.
3. Clean the generated MDX (see ``clean_page``) and write it to ``docs/api``.

Requirements: Node >= 20 with ``fern`` on PATH (``npm i -g fern-api``) and Docker.
The snap build of Docker cannot read ``/tmp``, so everything is kept under the repo.

Usage (from the repo root)::

    python fern/scripts/generate_api.py            # regenerate docs/api
    python fern/scripts/generate_api.py --no-run   # re-clean the last raw output only
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUILD = ROOT / ".api-build"
RAW = BUILD / "fern" / "static" / "api" / "earth2studio" / "earth2studio"
OUT = ROOT / "docs" / "api"

FERN_VERSION = "5.168.0"

DOCS_YML = """\
instances:
  - url: e2s-api-build.docs.buildwithfern.com
title: Earth2Studio API build
libraries:
  earth2studio:
    input:
      path: ../src
    output:
      path: ./static/api
    lang: python
navigation:
  - section: API Reference
    contents:
      - folder: ./static/api
"""


# --------------------------------------------------------------------------- #
# Step 1 + 2: stage and generate
# --------------------------------------------------------------------------- #
def run_generator() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    src = BUILD / "src"
    shutil.copytree(
        ROOT / "earth2studio",
        src / "earth2studio",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (src / "earth2studio" / "models" / "__init__.py").touch()
    fern_dir = BUILD / "fern"
    fern_dir.mkdir(parents=True)
    (fern_dir / "fern.config.json").write_text(
        f'{{\n  "organization": "nvidia",\n  "version": "{FERN_VERSION}"\n}}\n'
    )
    (fern_dir / "docs.yml").write_text(DOCS_YML)

    env = dict(os.environ)
    tmp = BUILD / "tmp"
    tmp.mkdir()
    env["TMPDIR"] = str(tmp)  # Docker (snap) must be able to see the temp dir
    subprocess.run(
        ["fern", "docs", "md", "generate", "--local"],
        cwd=BUILD,
        env=env,
        check=True,
    )


# --------------------------------------------------------------------------- #
# Step 3: clean the generated pages
# --------------------------------------------------------------------------- #
ANCHOR_OPEN = re.compile(r'^<Anchor id="([^"]+)">\s*$')
PARAM_OPEN = re.compile(r"^<ParamField\b[^>]*>\s*$")
AUTOREF = re.compile(r"\[([^\]\n]+)\]\[([^\]\n]*)\]")
FENCE = re.compile(r"^\s*```")


class Block:
    """One documented object: ``<Anchor>`` header plus optional ``<Indent>`` body."""

    def __init__(self, anchor_id: str, header: list[str], body: list | None):
        self.id = anchor_id
        self.header = header
        self.body = body  # None when the object has no <Indent> section

    @property
    def name(self) -> str:
        return self.id.rsplit("-", 1)[-1]

    @property
    def signature(self) -> str:
        for line in self.header:
            if line.startswith(("class ", "earth2studio.")):
                return line
        return ""

    def has_doc(self) -> bool:
        if self.body is None:
            return False
        return any(
            isinstance(i, Block) or (isinstance(i, str) and i.strip())
            for i in self.body
        )


def parse(lines: list[str], i: int = 0, until_close: bool = False):
    """Parse lines into a list of ``str`` / ``Block`` items. Returns (items, next_i)."""
    items: list = []
    in_fence = False
    while i < len(lines):
        line = lines[i]
        if FENCE.match(line):
            in_fence = not in_fence
        if not in_fence:
            if until_close and line.strip() == "</Indent>":
                return items, i + 1
            m = ANCHOR_OPEN.match(line)
            if m:
                header = [line]
                i += 1
                while i < len(lines) and lines[i].strip() != "</Anchor>":
                    header.append(lines[i])
                    i += 1
                header.append(lines[i])
                i += 1
                j = i
                while j < len(lines) and not lines[j].strip():
                    j += 1
                body = None
                if j < len(lines) and lines[j].strip() == "<Indent>":
                    body, i = parse(lines, j + 1, until_close=True)
                items.append(Block(m.group(1), header, body))
                continue
        items.append(line)
        i += 1
    return items, i


def keep_block(b: Block) -> bool:
    name = b.name
    sig = b.signature
    is_class = sig.startswith("class ")
    paren, eq = sig.find("("), sig.find(" = ")
    is_callable = not is_class and paren != -1 and (eq == -1 or paren < eq)
    if name.startswith("_"):
        # Keep documented dunders such as __call__; drop private and empty ones.
        return name.startswith("__") and name.endswith("__") and b.has_doc()
    if is_class:
        return True
    if is_callable:
        return True
    return False  # module data / constants


def drop_stray_params(items: list) -> list:
    """Remove empty ``ParamField``s that are not part of a Parameters/Attributes list."""
    out: list = []
    in_params = False
    i = 0
    while i < len(items):
        it = items[i]
        if isinstance(it, Block):
            out.append(it)
            in_params = False
            i += 1
            continue
        s = it.strip()
        if s in ("**Parameters:**", "**Attributes:**", "**Raises:**"):
            in_params = True
        elif PARAM_OPEN.match(it):
            # Collect up to the closing tag.
            j = i
            chunk = [it]
            while "</ParamField>" not in items[j]:
                j += 1
                chunk.append(items[j])
            inner = "".join(c for c in chunk if isinstance(c, str))
            inner = re.sub(r"</?ParamField[^>]*>", "", inner).strip()
            if not inner and not in_params:
                i = j + 1
                # eat one following blank line
                if (
                    i < len(items)
                    and isinstance(items[i], str)
                    and not items[i].strip()
                ):
                    i += 1
                continue
            out.extend(chunk)
            i = j + 1
            continue
        elif s:
            in_params = False
        out.append(it)
        i += 1
    return out


def filter_items(items: list) -> list:
    out: list = []
    for it in items:
        if isinstance(it, Block):
            if not keep_block(it):
                continue
            if it.body is not None:
                it.body = filter_items(it.body)
        out.append(it)
    return drop_stray_params(out)


def render(items: list) -> list[str]:
    out: list[str] = []
    for it in items:
        if isinstance(it, Block):
            out.extend(it.header)
            if it.body is not None:
                out.append("")
                out.append("<Indent>")
                out.extend(render(it.body))
                out.append("</Indent>")
        else:
            out.append(it)
    return out


def collect_ids(items: list, acc: list[str]) -> None:
    for it in items:
        if isinstance(it, Block):
            acc.append(it.id)
            if it.body:
                collect_ids(it.body, acc)


def fix_summary(text: str, kept_names: set[str]) -> str:
    """Drop the 'Data' summary and rows of removed objects from the module header."""
    text = re.sub(r"### Data\n.*?(?=\n### |\Z)", "", text, flags=re.S)

    def row_ok(line: str) -> bool:
        m = re.match(r"\| \[`([^`]+)`\]", line)
        return not m or m.group(1) in kept_names

    return "\n".join(l for l in text.split("\n") if row_ok(l))


def fix_badges(text: str) -> str:
    """Turn the free-text 'Badges' paragraph into a row of inline-code tags."""

    def repl(m: re.Match) -> str:
        tags = m.group(1).split()
        return "**Badges:** " + " ".join(f"`{t}`" for t in tags) + "\n"

    return re.sub(r"\*\*Badges\*\*\n\n((?:[^\n]+\n?)+)", lambda m: repl(m), text)


class Resolver:
    """Resolves MkDocs autoref targets (``earth2studio.data.GFS``) to MDX anchors."""

    def __init__(self, files: dict[Path, list[str]]):
        self.fqn: dict[str, Path] = {}
        for path, ids in files.items():
            for anchor in ids:
                self.fqn[anchor.replace("-", ".")] = path

    def find(self, target: str) -> tuple[str, Path] | None:
        if target in self.fqn:
            return target, self.fqn[target]
        parts = target.split(".")
        for n in (1, 2):
            if len(parts) <= n:
                continue
            prefix = ".".join(parts[:-n]) + "."
            tail = parts[-n:]
            cands = [
                f
                for f in self.fqn
                if f.startswith(prefix) and f.split(".")[-n:] == tail
            ]
            if cands:
                best = min(cands, key=len)
                return best, self.fqn[best]
        return None

    def link(self, text: str, target: str, here: Path) -> str:
        label = text
        target = target.strip() or text.strip("`")
        hit = self.find(target)
        if not hit:
            return label if label.startswith("`") else f"`{label}`"
        fqn, path = hit
        rel = os.path.relpath(path, here.parent).replace(os.sep, "/")
        if not rel.startswith("."):
            rel = "./" + rel
        anchor = fqn.replace(".", "-")
        if not label.startswith("`"):
            label = f"`{label}`"
        return f"[{label}]({rel}#{anchor})"


def fix_autorefs(text: str, resolver: Resolver, here: Path) -> str:
    out: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        if not in_fence and "][" in line:
            line = AUTOREF.sub(
                lambda m: resolver.link(m.group(1), m.group(2), here), line
            )
        out.append(line)
    return "\n".join(out)


def split_frontmatter(text: str) -> tuple[str, str]:
    m = re.match(r"---\n(.*?)\n---\n", text, flags=re.S)
    if not m:
        return "", text
    return m.group(1), text[m.end() :]


CODEBLOCK = re.compile(r"<CodeBlock\b[^>]*>\n(.*?)\n</CodeBlock>", re.S)
PARAM_TAG = re.compile(
    r'^<ParamField path="(?P<path>.*?)"(?= type=| default=|>$)'
    r'(?: type="(?P<type>.*?)")?(?= default="|>$)'
    r'(?: default="(?P<default>.*)")?>$'
)


def fix_nested_fences(body: str) -> str:
    """Unwrap docstring ``Examples`` that already contain markdown fences.

    Fern wraps the whole Examples section in a single fenced ``CodeBlock``; when the
    docstring itself has ```` ```python ```` fences the nesting breaks MDX. Such blocks
    have an odd number of fence lines, so drop the wrapper and the outer opening fence.
    """

    def repl(m: re.Match) -> str:
        inner = m.group(1)
        fences = [l for l in inner.split("\n") if FENCE.match(l)]
        if len(fences) % 2 == 0:
            return m.group(0)
        lines = inner.split("\n")
        first = next(i for i, l in enumerate(lines) if FENCE.match(l))
        del lines[first]
        return "\n".join(lines)

    return CODEBLOCK.sub(repl, body)


def fix_param_tags(body: str) -> str:
    """Escape embedded double quotes in ``ParamField`` attribute values."""
    out = []
    for line in body.split("\n"):
        if line.startswith("<ParamField "):
            m = PARAM_TAG.match(line)
            if m:
                parts = [f'path="{m["path"].replace(chr(34), chr(39))}"']
                if m["type"] is not None:
                    parts.append(f'type="{m["type"].replace(chr(34), chr(39))}"')
                if m["default"] is not None:
                    parts.append(f'default="{m["default"].replace(chr(34), chr(39))}"')
                line = "<ParamField " + " ".join(parts) + ">"
        out.append(line)
    return "\n".join(out)


def clean_page(path: Path) -> tuple[str, list[str], set[str]]:
    text = path.read_text()
    fm, body = split_frontmatter(text)
    body = fix_nested_fences(body)
    body = fix_param_tags(body)
    # Fern derives slugs from the folder layout; the generated absolute slugs
    # (earth2studio/earth2studio/...) would duplicate the section prefix.
    fm = "\n".join(l for l in fm.split("\n") if not l.startswith("slug:"))
    items, _ = parse(body.split("\n"))
    items = filter_items(items)
    ids: list[str] = []
    collect_ids(items, ids)
    top_names = {i.rsplit("-", 1)[-1] for i in ids}
    body = "\n".join(render(items))
    body = fix_summary(body, top_names)
    body = fix_badges(body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return f"---\n{fm}\n---\n{body.strip()}\n", ids, top_names


def clean_all() -> int:
    if not RAW.exists():
        sys.exit(f"No raw output at {RAW}; run without --no-run first.")
    pages: dict[Path, tuple[str, list[str]]] = {}
    for mdx in sorted(RAW.rglob("*.mdx")):
        rel = mdx.relative_to(RAW)
        text, ids, _ = clean_page(mdx)
        pages[rel] = (text, ids)
    resolver = Resolver({OUT / rel: ids for rel, (_, ids) in pages.items()})
    if OUT.exists():
        shutil.rmtree(OUT)
    final: dict[str, str] = {}
    for rel, (text, _) in pages.items():
        dest = OUT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = fix_autorefs(text, resolver, dest)
        dest.write_text(text)
        final[rel.with_suffix("").as_posix()] = text

    def find_page(name: str) -> str | None:
        hit = resolver.find(name)
        if not hit:
            return None
        return hit[1].relative_to(OUT).with_suffix("").as_posix()

    import api_nav

    api_nav.build(
        final,
        find_page,
        ROOT,
        ROOT / "fern" / "pages" / "index.yml",
        ROOT / "fern" / "scripts" / "api_urls.json",
    )
    return len(pages)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--no-run", action="store_true", help="skip fern; re-clean the last raw output"
    )
    args = ap.parse_args()
    if not args.no_run:
        run_generator()
    n = clean_all()
    print(f"Wrote {n} pages to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

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
import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODULES = DOCS / "modules"
GENERATED = MODULES / "generated"
SOURCE_REF = "main"

BADGE_RE = re.compile(
    r"\b(?:region|class|dataclass|year|product|gpu):[A-Za-z0-9_.-]+\b"
)
RST_ROLE_RE = re.compile(
    r":(?:py:)?(?:mod|class|func|obj|meth|attr|exc|data|const):`~?([^`]+)`"
)
RST_LINK_RE = re.compile(r"`([^`<]+)\s*<([^`>]+)>`_")

METHODS_BY_TEMPLATE = {
    "dataassim.rst": (
        "__call__",
        "create_generator",
        "load_default_package",
        "load_model",
    ),
    "datasource.rst": ("__call__", "fetch", "available"),
    "diagnostic.rst": ("__call__", "load_default_package", "load_model"),
    "io.rst": ("add_array", "write"),
    "perturbation.rst": ("__call__",),
    "prognostic.rst": (
        "__call__",
        "create_iterator",
        "load_default_package",
        "load_model",
    ),
    "statistics.rst": ("__call__",),
}

GROUP_TITLES = {
    "datasources_analysis": ("Data Sources", "AI Data Sources"),
    "utils_all": (
        "Coordinate Utilities",
        "Grid Interpolation",
        "Observation Utilities",
        "Time Utilities",
        "Checkpoint Classes",
        "Checkpoint Helpers",
        "Data Utilities",
        "Model Utilities",
    ),
}

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class ApiMember:
    """Metadata for one rendered class member."""

    name: str
    owner: str
    source: Path | None
    node: FunctionNode
    signature: str
    docstring: str
    line_start: int | None
    line_end: int | None


@dataclass(frozen=True)
class ObjectPage:
    """Metadata used to render one generated API object page."""

    display: str
    full_name: str
    source: Path | None
    node: ast.AST | None
    kind: str
    signature: str
    summary: str
    docstring: str
    badges: tuple[str, ...]
    members: tuple[ApiMember, ...]
    line_start: int | None
    line_end: int | None
    output: Path
    template: str


def clean_rst_roles(text: str) -> str:
    """Convert simple reStructuredText roles to Markdown code or links."""
    text = RST_ROLE_RE.sub(lambda match: f"`{match.group(1)}`", text)
    text = RST_LINK_RE.sub(r"[\1](\2)", text)
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
SOURCE_IMPORTS: dict[Path, dict[str, str]] = {}
TREES: dict[Path, ast.Module] = {}


def package_import_map(package_dir: Path) -> dict[str, tuple[Path, str]]:
    """Return a cached import map for a package directory."""
    package_dir = package_dir.resolve()
    if package_dir not in IMPORT_MAPS:
        IMPORT_MAPS[package_dir] = parse_import_map(package_dir)
    return IMPORT_MAPS[package_dir]


def module_tree(source: Path) -> ast.Module | None:
    """Parse a Python module once and cache the AST."""
    source = source.resolve()
    if not source.exists():
        return None
    if source not in TREES:
        TREES[source] = ast.parse(source.read_text(encoding="utf-8"))
    return TREES[source]


def find_node(source: Path, name: str) -> ast.AST | None:
    """Find a top-level class or function node in a source file."""
    tree = module_tree(source)
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return node
    return None


def module_docstring(source: Path) -> str:
    """Read the module docstring from a Python source file."""
    tree = module_tree(source)
    if tree is None:
        return ""
    return ast.get_docstring(tree) or ""


def module_name(source: Path) -> str:
    """Return the importable module name for a source file."""
    rel = source.resolve().relative_to(ROOT).with_suffix("")
    parts = rel.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def source_imports(source: Path) -> dict[str, str]:
    """Return a map from local imported names to fully-qualified import paths."""
    source = source.resolve()
    if source in SOURCE_IMPORTS:
        return SOURCE_IMPORTS[source]

    tree = module_tree(source)
    if tree is None:
        SOURCE_IMPORTS[source] = {}
        return {}

    result: dict[str, str] = {}
    current_package = module_name(source.parent / "__init__.py").split(".")
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.level:
                module_parts = current_package[: len(current_package) - node.level + 1]
                if node.module:
                    module_parts.extend(node.module.split("."))
                module = ".".join(module_parts)
            else:
                module = node.module or ""
            if not module.startswith("earth2studio"):
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                result[local] = f"{module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not alias.name.startswith("earth2studio"):
                    continue
                result[alias.asname or alias.name.split(".", 1)[0]] = alias.name

    SOURCE_IMPORTS[source] = result
    return result


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


def annotation(node: ast.AST | None) -> str:
    """Format an AST annotation or return an empty string."""
    return ast.unparse(node) if node is not None else ""


def argument(arg: ast.arg, default: ast.expr | None = None, prefix: str = "") -> str:
    """Format one function argument from AST."""
    text = prefix + arg.arg
    if arg.annotation is not None:
        text += f": {annotation(arg.annotation)}"
    if default is not None:
        text += f" = {ast.unparse(default)}"
    return text


def function_signature(
    name: str,
    node: FunctionNode,
    *,
    skip_bound_arg: bool = False,
    include_return: bool = True,
    include_def: bool = False,
) -> str:
    """Format a function or method signature from AST."""
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    defaults: list[ast.expr | None] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)
    pairs = list(zip(positional, defaults))
    if skip_bound_arg and pairs and pairs[0][0].arg in {"self", "cls"}:
        pairs = pairs[1:]

    parts = [argument(arg, default) for arg, default in pairs]
    if args.posonlyargs:
        posonly = len(args.posonlyargs) - (1 if skip_bound_arg else 0)
        if posonly > 0:
            parts.insert(posonly, "/")
    if args.vararg is not None:
        parts.append(argument(args.vararg, prefix="*"))
    elif args.kwonlyargs:
        parts.append("*")
    parts.extend(
        argument(arg, default)
        for arg, default in zip(args.kwonlyargs, args.kw_defaults)
    )
    if args.kwarg is not None:
        parts.append(argument(args.kwarg, prefix="**"))

    signature = f"{name}({', '.join(parts)})"
    if include_return and node.returns is not None:
        signature += f" -> {annotation(node.returns)}"
    if include_def:
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        signature = prefix + signature
    elif isinstance(node, ast.AsyncFunctionDef):
        signature = "async " + signature
    return signature


def decorator_name(decorator: ast.expr) -> str:
    """Return a best-effort dotted decorator name."""
    if isinstance(decorator, ast.Call):
        decorator = decorator.func
    return ast.unparse(decorator)


def is_staticmethod(node: FunctionNode) -> bool:
    """Return True if the function node is decorated as a staticmethod."""
    return any(decorator_name(item) == "staticmethod" for item in node.decorator_list)


def class_signature(display: str, node: ast.AST | None) -> str:
    """Return a useful class constructor signature."""
    if not isinstance(node, ast.ClassDef):
        return f"{display}(...)"
    init = next(
        (
            item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == "__init__"
        ),
        None,
    )
    if init is None:
        return f"{display}()"
    return function_signature(
        display,
        init,
        skip_bound_arg=True,
        include_return=False,
    )


def page_signature(display: str, node: ast.AST | None, kind: str) -> str:
    """Return the top-level signature for an API page."""
    if kind == "module":
        return display
    if isinstance(node, ast.ClassDef):
        return class_signature(display, node)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return function_signature(display, node)
    return f"{display}(...)"


def mkdocstrings_block(page: ObjectPage) -> list[str]:
    """Render the object documentation using mkdocstrings."""
    members = [member.name for member in page.members]
    render_path = page.full_name
    if page.source is not None and isinstance(
        page.node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
        rel = page.source.relative_to(ROOT).with_suffix("")
        if rel.parts[:2] == ("earth2studio", "models"):
            render_path = ".".join((*rel.parts[2:], page.node.name))
    block = [
        f"::: {render_path}",
        "    handler: python",
        "    options:",
        "      heading_level: 3",
        "      show_root_heading: false",
        "      show_root_toc_entry: false",
        "      show_root_full_path: false",
        "      show_root_members_full_path: false",
        "      show_object_full_path: false",
        "      show_source: false",
        "      show_signature: true",
        "      show_signature_annotations: true",
        "      separate_signature: true",
        "      signature_crossrefs: true",
        "      show_symbol_type_toc: true",
        "      docstring_style: numpy",
        "      docstring_section_style: list",
        "      merge_init_into_class: true",
        "      group_by_category: false",
        "      members_order: source",
    ]
    if members:
        block.extend(["      members:"])
        block.extend(f"      - {member}" for member in members)
        block.extend(["      inherited_members:"])
        block.extend(f"      - {member}" for member in members)
    else:
        block.extend(["      members: []", "      inherited_members: []"])
    return block


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
    return f"https://github.com/NVIDIA/earth2studio/blob/{SOURCE_REF}/{rel}{lines}"


def source_markdown(source: Path | None, start: int | None, end: int | None) -> str:
    """Render a compact source link for a generated API page."""
    link = source_link(source, start, end)
    if not link or source is None:
        return ""
    rel = source.relative_to(ROOT).as_posix()
    if start:
        rel += f":{start}"
    return f'[View source on GitHub]({link}){{ .e2s-source-link title="{rel}" }}'


def yaml_scalar(value: str) -> str:
    """Format a string as a YAML-safe quoted scalar."""
    return json.dumps(value)


def yaml_list(values: Iterable[str]) -> str:
    """Format scalar strings as a compact YAML list."""
    values = list(values)
    if not values:
        return "[]"
    return "[" + ", ".join(yaml_scalar(value) for value in values) + "]"


def base_name(base: ast.expr) -> str:
    """Return a best-effort base class name from an AST expression."""
    if isinstance(base, ast.Subscript):
        return base_name(base.value)
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ast.unparse(base).split(".")[-1]


def resolve_base(
    source: Path, base: ast.expr
) -> tuple[Path | None, ast.AST | None, str]:
    """Resolve a class base to a source file and AST node when possible."""
    name = base_name(base)
    local = find_node(source, name)
    if local is not None:
        return source, local, f"{module_name(source)}.{name}"
    imported = source_imports(source).get(name)
    if imported:
        base_source, base_node, _ = resolve(imported)
        return base_source, base_node, imported
    return None, None, name


def find_class_member(
    source: Path | None,
    node: ast.AST | None,
    name: str,
    *,
    seen: set[tuple[Path, str]] | None = None,
) -> tuple[Path | None, ast.ClassDef | None, FunctionNode | None]:
    """Find a class member in the class or local Earth2Studio bases."""
    if source is None or not isinstance(node, ast.ClassDef):
        return None, None, None
    seen = seen or set()
    key = (source.resolve(), node.name)
    if key in seen:
        return None, None, None
    seen.add(key)

    for item in node.body:
        if (
            isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == name
        ):
            return source, node, item

    for base in node.bases:
        base_source, base_node, _ = resolve_base(source, base)
        found_source, found_class, found_node = find_class_member(
            base_source, base_node, name, seen=seen
        )
        if found_node is not None:
            return found_source, found_class, found_node

    return None, None, None


def public_class_members(source: Path | None, node: ast.AST | None) -> tuple[str, ...]:
    """Return public members defined directly on a class."""
    if source is None or not isinstance(node, ast.ClassDef):
        return ()
    members = [
        item.name
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not item.name.startswith("_")
    ]
    return tuple(dict.fromkeys(members))


def template_members(
    template: str, source: Path | None, node: ast.AST | None
) -> tuple[str, ...]:
    """Return the class members requested by an autosummary template."""
    if template == "class.rst":
        return public_class_members(source, node)
    return METHODS_BY_TEMPLATE.get(template, ())


def collect_members(
    full_name: str,
    source: Path | None,
    node: ast.AST | None,
    template: str,
) -> tuple[ApiMember, ...]:
    """Collect rendered member metadata for a class API page."""
    if not isinstance(node, ast.ClassDef):
        return ()
    members: list[ApiMember] = []
    for name in template_members(template, source, node):
        member_source, member_class, member_node = find_class_member(source, node, name)
        if member_node is None:
            continue
        owner = (
            full_name
            if member_class is node
            else (member_class.name if member_class else "")
        )
        skip_bound_arg = not is_staticmethod(member_node)
        members.append(
            ApiMember(
                name=name,
                owner=owner,
                source=member_source,
                node=member_node,
                signature=function_signature(
                    name,
                    member_node,
                    skip_bound_arg=skip_bound_arg,
                    include_def=True,
                ),
                docstring=ast.get_docstring(member_node) or "",
                line_start=getattr(member_node, "lineno", None),
                line_end=getattr(member_node, "end_lineno", None),
            )
        )
    return tuple(members)


def object_page(
    display: str, full_name: str, output_dir: Path, template: str
) -> ObjectPage:
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
        node=node,
        kind=kind,
        signature=page_signature(display, node, kind),
        summary=summary,
        docstring=docstring,
        badges=badges,
        members=collect_members(full_name, source, node, template),
        line_start=start,
        line_end=end,
        output=output_dir / filename,
        template=template,
    )


def write_object(page: ObjectPage) -> None:
    """Write one generated API object page."""
    page.output.parent.mkdir(parents=True, exist_ok=True)
    body = [
        "---",
        f"title: {yaml_scalar(page.display)}",
        f"summary: {yaml_scalar(page.summary)}",
        f"signature: {yaml_scalar(page.signature)}",
        f"badges: {yaml_list(page.badges)}",
        "---",
        "",
        f"# `{page.display}`",
        "",
    ]
    if page.badges:
        body.extend(["{% badges " + " ".join(page.badges) + " %}", ""])
    body.extend([f"**Import path:** `{page.full_name}`", ""])
    if source := source_markdown(page.source, page.line_start, page.line_end):
        body.extend([source, ""])
    if page.docstring:
        body.extend(["## Documentation", "", *mkdocstrings_block(page), ""])
    page.output.write_text("\n".join(body), encoding="utf-8")


def collect_autosummaries(
    path: Path,
) -> list[tuple[str, list[str], dict[str, str], str, list[str]]]:
    """Collect autosummary entries and badge-filter options from RST."""
    lines = path.read_text(encoding="utf-8").splitlines()
    current_module = "earth2studio"
    result: list[tuple[str, list[str], dict[str, str], str, list[str]]] = []
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
        template = "class.rst"
        scan = index + 1
        while scan < len(lines):
            raw = lines[scan]
            s = raw.strip()
            if not s:
                scan += 1
                continue
            if not raw.startswith(" "):
                break
            if s.startswith(":template:"):
                template = s.split(":", 2)[2].strip()
                scan += 1
                continue
            if s.startswith(":") or s.startswith(".."):
                scan += 1
                continue
            objects.append(s)
            scan += 1
        result.append(
            (current_module, list(dict.fromkeys(badges)), options, template, objects)
        )
    return result


def intro_markdown(path: Path) -> str:
    """Convert introductory RST content to Markdown."""
    lines = path.read_text(encoding="utf-8").splitlines()
    output: list[str] = []
    skip_directive = False
    index = 0
    while index < len(lines):
        line = lines[index]
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
            index += 1
            continue
        if skip_directive:
            if line.startswith(" ") or not stripped:
                index += 1
                continue
            skip_directive = False
        if index + 1 < len(lines) and set(lines[index + 1].strip()) in (
            {"-"},
            {"~"},
        ):
            heading = rst_title(stripped)
            level = "#" if set(lines[index + 1].strip()) == {"-"} else "##"
            output.append(f"{level} {heading}")
            index += 1
            continue
        if index > 0 and set(stripped) in ({"-"}, {"~"}):
            index += 1
            continue
        output.append(clean_rst_roles(line))
        index += 1
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


def group_title(path: Path, group_index: int) -> str:
    """Return a human-readable section title for an autosummary group."""
    titles = GROUP_TITLES.get(path.stem)
    if titles and group_index <= len(titles):
        return titles[group_index - 1]
    return f"{path.stem.replace('_', ' ').title()} {group_index}"


def build_module_page(path: Path) -> None:
    """Generate one MkDocs API summary page from a Sphinx page."""
    groups = collect_autosummaries(path)
    md_path = path.with_suffix(".md")
    body = [intro_markdown(path), ""]
    for group_index, (current_module, badges, options, template, objects) in enumerate(
        groups, start=1
    ):
        if not objects:
            continue
        out_dir = generated_dir_for(path, group_index)
        pages = [
            object_page(obj, full_name(current_module, obj), out_dir, template)
            for obj in objects
        ]
        for page in pages:
            write_object(page)
        if len(groups) > 1:
            body.extend([f"## {group_title(path, group_index)}", ""])
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
        ("Data Sources", "datasources_analysis.md"),
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

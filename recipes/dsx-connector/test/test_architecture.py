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

"""Architecture boundaries between shared code, the DSX core, and model producers.

``src/dsx/`` is the model-independent publishing path; ``src/stormcast/`` and ``src/sfno/`` are
swappable producers; and ``src/shared/`` contains neutral utilities. Three rules are enforced:

1. ``dsx`` never imports a producer (``stormcast`` or ``sfno``), so the core stays copyable into a
   future producer's recipe unchanged.
2. A producer never imports another producer, so each is a self-contained example.
3. ``shared`` never imports ``dsx`` or a producer, so its utilities remain independent.

The DSX core and producers may import ``shared``. Producers may also import the DSX core. Imports
are read statically with ``ast``—no module is imported, so this test does not require model
dependencies or a GPU.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
_DSX = _SRC / "dsx"
_SHARED = _SRC / "shared"
_PRODUCERS = ("stormcast", "sfno")


def _imported_modules(py_file: Path) -> set[str]:
    """Return every module path referenced by an import statement in ``py_file``."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    package = ("src", *py_file.relative_to(_SRC).parent.parts)
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                ancestor_count = node.level - 1
                base_parts = package[: len(package) - ancestor_count]
                base_parts += tuple((node.module or "").split("."))
                base = ".".join(part for part in base_parts if part)
            else:
                base = node.module or ""
            if base:
                mods.add(base)
            for alias in node.names:
                mods.add(f"{base}.{alias.name}" if base else alias.name)
    return mods


def _assert_no_forbidden_imports(
    package: Path, forbidden: tuple[str, ...], boundary: str
) -> None:
    files = sorted(package.rglob("*.py"))
    assert files, f"no src/{package.name} modules found — check the package layout"
    offenders = {
        str(py.relative_to(package)): sorted(
            module
            for module in _imported_modules(py)
            if any(name in module.split(".") for name in forbidden)
        )
        for py in files
    }
    offenders = {name: modules for name, modules in offenders.items() if modules}
    assert not offenders, f"{boundary}; offending imports: {offenders}"


def test_dsx_core_never_imports_a_producer() -> None:
    _assert_no_forbidden_imports(
        _DSX,
        _PRODUCERS,
        "src/dsx (the model-independent publishing core) must not import a "
        f"producer ({', '.join(_PRODUCERS)})",
    )


def test_shared_never_imports_dsx_or_a_producer() -> None:
    _assert_no_forbidden_imports(
        _SHARED,
        ("dsx", *_PRODUCERS),
        "src/shared must not import the DSX core or a producer",
    )


@pytest.mark.parametrize("producer", _PRODUCERS)
def test_producers_never_import_each_other(producer: str) -> None:
    others = tuple(name for name in _PRODUCERS if name != producer)
    _assert_no_forbidden_imports(
        _SRC / producer,
        others,
        f"src/{producer} must not import another producer ({', '.join(others)})",
    )

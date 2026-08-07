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

"""Prepare a persistent GitHub Pages directory for MkDocs output."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

PRESERVED_ROOT_ITEMS = {"v", "versions.json", ".nojekyll"}


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _copy_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for child in source.iterdir():
        target = destination / child.name
        if target.exists() or target.is_symlink():
            _remove(target)
        if child.is_dir() and not child.is_symlink():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _refresh_root(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    for child in destination.iterdir():
        if child.name not in PRESERVED_ROOT_ITEMS:
            _remove(child)

    _copy_contents(source, destination)


def _load_versions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    versions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(versions, list):
        raise TypeError(f"{path} must contain a JSON list")

    return [
        entry for entry in versions if isinstance(entry, dict) and "version" in entry
    ]


def _version_sort_key(entry: dict[str, Any]) -> tuple[int, tuple[int, ...], str]:
    version = str(entry["version"])
    numbers = tuple(int(match) for match in re.findall(r"\d+", version))
    is_release = int(bool(numbers))
    return (is_release, numbers, version)


def _update_versions(path: Path, doc_version: str, doc_title: str) -> None:
    versions = {
        str(entry["version"]): {
            "version": str(entry["version"]),
            "title": str(entry.get("title", entry["version"])),
            "aliases": list(entry.get("aliases", [])),
        }
        for entry in _load_versions(path)
    }
    versions[doc_version] = {
        "version": doc_version,
        "title": doc_title,
        "aliases": versions.get(doc_version, {}).get("aliases", []),
    }

    ordered = sorted(versions.values(), key=_version_sort_key, reverse=True)
    path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    source = _env_path("DOCS_BUILD_DIR", "docs/_build/html")
    destination = _env_path("DOCS_PAGES_STATE", "docs-pages-state")
    publish_path = os.environ.get("PUBLISH_PATH", ".")
    doc_version = os.environ.get("DOC_VERSION", "main")
    doc_title = os.environ.get("DOC_TITLE", doc_version)

    if not source.is_dir():
        raise FileNotFoundError(f"Docs build directory does not exist: {source}")

    if publish_path in ("", "."):
        _refresh_root(source, destination)
    else:
        target = destination / publish_path
        if target.exists() or target.is_symlink():
            _remove(target)
        _copy_contents(source, target)
        _update_versions(destination / "versions.json", doc_version, doc_title)

    (destination / ".nojekyll").touch()
    print(f"[docs-pages] prepared {destination} from {source} at {publish_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

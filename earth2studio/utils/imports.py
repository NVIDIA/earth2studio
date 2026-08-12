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
import inspect
import re
import sys
import tomllib
from collections.abc import Callable
from functools import lru_cache, wraps
from importlib.metadata import PackageNotFoundError, requires, version
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar, cast

from packaging.requirements import Requirement
from rich.console import Console
from rich.table import Table
from rich.traceback import Traceback

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type[Any])


class OptionalDependencyError(ImportError):
    """Custom error for missing optional dependencies."""

    def __init__(
        self,
        group_name: str,
        object_name: str,
        error_value: BaseException | None = None,
        error_traceback: TracebackType | None = None,
        doc_url: str | None = None,
    ):
        if doc_url is None:
            doc_url = "https://nvidia.github.io/earth2studio/userguide/about/install.html#optional-dependencies"

        console = Console()
        table = Table(show_header=False, show_lines=True)
        table.add_row(
            "[blue]Earth2Studio Extra Dependency Error\n"
            + "This error typically indicates an extra dependency group is needed.\n"
            + "Don't panic, this is usually an easy fix.[/blue]"
        )
        table.add_row(
            f"[yellow]This feature ('{object_name}') is marked needing optional dependency group '{group_name}'.\n\n"
            + f"uv install with: `uv add earth2studio --extra {group_name}`\n\n"
            + "For more information (such as pip install instructions), visit the install documentation: \n"
            + f"{doc_url}[/yellow]"
        )
        if error_value:
            table.add_row(
                Traceback(
                    Traceback.extract(
                        ImportError, error_value, error_traceback, show_locals=False
                    )
                )
            )
        console.print(table)
        super().__init__("Optional dependency import error")


class OptionalDependencyFailure:
    """Optional dependency group failure.
    Instantiate this in the catch of an import failure for a group of optional
    dependencies. This will add the failure the class singleton dictionary for
    later look up.

    Parameters
    ----------
    group_name : str
        Name of the optional dependency group imports are present in. For example
        this would be "data" if the install command is:
        `pip install earth2studio[data]`
    key : str | None, optional
        Unique dependency group key to id the import. This is used with the
        `check_optional_dependencies` to check if needed optional dependencies are
        installed. Setting this manually is typically not needed. If None, the key
        will be the filename of the caller object, by default None
    doc_url : str | None, optional
        Install doc url override of the install doc link present in error
        message if import error occurs, by default None

    Raises
    ------
    ValueError
        If key is not unique
    """

    failures: dict[str, "OptionalDependencyFailure"] = {}

    def __init__(
        self,
        group_name: str,
        key: str | None = None,
        doc_url: str | None = None,
    ):
        if key is None:
            stack = inspect.stack()
            caller_frame = stack[1]
            key = caller_frame.filename

        if key in self.failures:
            # Raising error here actually causes problems when theres two imports from
            # same file, should be fine to just return None if already registered.
            # raise ValueError(f"Extra dependency key already defined: {key}")
            return
        if group_name is None:
            group_name = key

        # Get current exception and traceback
        _, exc_value, tb_value = sys.exc_info()

        self.import_exception = exc_value
        self.import_traceback = tb_value
        self.doc_url = doc_url
        self.group_name = group_name
        self.failures[key] = self


def check_optional_dependencies(
    key: str | None = None,
) -> Callable[[Any], Any]:
    """Decorator to handle validating optional dependencies are installed for both
    functions and classes.

    Parameters
    ----------
    key : str | None, optional
        Optional dependency group key to check for any import errors. This typically
        does not need to be set. If None will use the filename of the caller object, by
        default None
    """
    if key is None:
        stack = inspect.stack()
        caller_frame = stack[1]
        key = caller_frame.filename

    def _check_deps(obj_name: str) -> None:
        # No error in singleton = no import issues
        if key not in OptionalDependencyFailure.failures:
            return

        group = OptionalDependencyFailure.failures[key]
        if group.import_exception is not None:
            raise OptionalDependencyError(
                group.group_name,
                obj_name,
                group.import_exception,
                group.import_traceback,
                group.doc_url,
            )

    def _decorator(obj: F | C) -> F | C:
        if isinstance(obj, type):
            original_init = obj.__init__  # type: ignore

            @wraps(original_init)
            def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                _check_deps(obj.__name__)
                original_init(self, *args, **kwargs)

            obj.__init__ = _wrapped_init  # type: ignore
            return cast(C, obj)
        else:
            # Handle function decoration
            @wraps(obj)
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                _check_deps(f"{obj.__module__}.{obj.__name__}")
                return obj(*args, **kwargs)

            return cast(F, _wrapper)

    return _decorator


def _find_pyproject_toml() -> Path:
    """Locate pyproject.toml relative to this module.

    This only works for editable/source installs. For wheel installs,
    use :func:`_parse_optional_dependencies` which prefers
    ``importlib.metadata``.

    Returns
    -------
    Path
        Path to pyproject.toml

    Raises
    ------
    FileNotFoundError
        If pyproject.toml cannot be found within 5 parent directories
    """
    current = Path(__file__).resolve().parent
    for _ in range(5):  # Max 5 levels up
        candidate = current / "pyproject.toml"
        if candidate.exists():
            return candidate
        current = current.parent
    raise FileNotFoundError(
        "Could not locate pyproject.toml from earth2studio/utils/imports.py"
    )


_EXTRA_RE = re.compile(r"""extra\s*==\s*['"]([^'"]+)['"]""")


@lru_cache(maxsize=1)
def _parse_optional_dependencies_from_metadata() -> dict[str, list[str]]:
    """Parse optional-dependencies from installed package metadata.

    Uses ``importlib.metadata.requires()`` which reads the ``.dist-info/METADATA``
    file present in both editable and wheel installs.

    Returns
    -------
    dict[str, list[str]]
        Mapping of group name to list of package specs
    """
    reqs = requires("earth2studio") or []
    groups: dict[str, list[str]] = {}
    for req_str in reqs:
        # Each entry looks like: "package_spec; extra == 'group'"
        # or just "package_spec" for core dependencies
        parts = req_str.split(";", 1)
        spec = parts[0].strip()
        if len(parts) == 2:
            marker = parts[1].strip()
            m = _EXTRA_RE.search(marker)
            if m:
                group = m.group(1)
                groups.setdefault(group, []).append(spec)
        # Core deps (no extra marker) are not included in optional-dependencies
    return groups


@lru_cache(maxsize=1)
def _parse_optional_dependencies() -> dict[str, list[str]]:
    """Parse and cache optional-dependencies for the earth2studio package.

    Prefers ``importlib.metadata`` (works for both wheel and editable installs).
    Falls back to reading ``pyproject.toml`` directly if metadata is unavailable.

    Returns
    -------
    dict[str, list[str]]
        Mapping of group name to list of package specs
    """
    try:
        deps = _parse_optional_dependencies_from_metadata()
        if deps:
            return deps
    except PackageNotFoundError:
        pass

    # Fallback: read pyproject.toml (editable/source installs only)
    pyproject_path = _find_pyproject_toml()
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("project", {}).get("optional-dependencies", {})


def _check_package(spec: str) -> tuple[bool, str | None]:
    """Check if a package spec is satisfied.

    Parameters
    ----------
    spec : str
        Package specification (e.g., "scipy>=1.15", "torch", "redis>=5.0,<6.0")

    Returns
    -------
    tuple[bool, str | None]
        (True, None) if satisfied, (False, reason) if not satisfied
    """
    req = Requirement(spec)
    try:
        installed_version = version(req.name)
    except PackageNotFoundError:
        return False, f"{req.name} not installed"

    if req.specifier and not req.specifier.contains(installed_version):
        return (
            False,
            f"{req.name}=={installed_version} does not satisfy {req.specifier}",
        )

    return True, None


def _get_group_packages(group: str, seen: set[str] | None = None) -> list[str]:
    """Get package specs for a dependency group, recursively expanding earth2studio refs.

    Parameters
    ----------
    group : str
        Group name from pyproject.toml optional-dependencies
    seen : set[str] | None
        Groups already visited (prevents infinite recursion)

    Returns
    -------
    list[str]
        Package specs (excluding earth2studio[...] self-references)

    Raises
    ------
    ValueError
        If the group name is not found in pyproject.toml
    """
    if seen is None:
        seen = set()
    if group in seen:
        return []
    seen.add(group)

    optional_deps = _parse_optional_dependencies()
    if group not in optional_deps:
        raise ValueError(f"Unknown dependency group: {group!r}")

    packages: list[str] = []
    for spec in optional_deps[group]:
        req = Requirement(spec)
        if req.name == "earth2studio":
            # Recursively expand earth2studio[extra] references
            for extra in req.extras:
                packages.extend(_get_group_packages(extra, seen))
        else:
            # Strip markers from spec, just keep package name and version
            clean_spec = req.name
            if req.specifier:
                clean_spec = f"{req.name}{req.specifier}"
            packages.append(clean_spec)

    return packages


def pytest_require(
    *packages: str,
    groups: list[str] | None = None,
) -> Any:
    """Return a pytest skipif marker if required packages or groups are missing.

    Use this to skip entire test modules when optional dependencies are not installed.

    Parameters
    ----------
    *packages : str
        Package specs, optionally with version constraints (e.g., "scipy>=1.15", "torch").
    groups : list[str] | None
        Dependency group names from pyproject.toml (e.g., ["serve", "data"]).

    Returns
    -------
    pytest.MarkDecorator
        A skipif marker that skips if any requirement is not satisfied.

    Examples
    --------
    Skip entire module if 'serve' group dependencies are not installed:

    >>> pytestmark = pytest_require(groups=["serve"])

    Skip if specific packages are missing or wrong version:

    >>> pytestmark = pytest_require("fme", "scipy>=1.15")

    Combine packages and groups:

    >>> pytestmark = pytest_require("redis>=5.0", groups=["serve"])
    """
    import pytest  # Lazy import - pytest is a dev dependency

    missing: list[str] = []

    # Check explicit packages
    for spec in packages:
        ok, reason = _check_package(spec)
        if not ok:
            missing.append(reason or spec)

    # Check group packages
    if groups:
        for group in groups:
            for spec in _get_group_packages(group):
                ok, reason = _check_package(spec)
                if not ok:
                    missing.append(f"group '{group}': {reason or spec}")

    reason = f"Missing: {', '.join(missing)}" if missing else ""
    return pytest.mark.skipif(bool(missing), reason=reason)

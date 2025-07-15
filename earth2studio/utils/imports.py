# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
import sys
from collections.abc import Callable
from functools import wraps
from types import TracebackType
from typing import Any, TypeVar, cast

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
            raise ValueError(f"Extra dependency key already defined: {key}")
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

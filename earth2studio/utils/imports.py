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
import importlib
from collections.abc import Callable
from functools import wraps
from importlib.util import find_spec
from typing import Any, TypeVar, cast

from loguru import logger

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type[Any])


class ExtraDependencyError(ImportError):
    """Custom error for missing extra dependencies."""

    def __init__(self, extra_name: str, obj_name: str):
        super().__init__(
            f"Extra dependency group '{extra_name}' is required for {obj_name}.\n"
            f"Install with: uv pip install earth2studio --extra {extra_name} \n"
            f"or refer to the install documentation."
        )


def check_extra_imports(
    extra_name: str, package_obj: Any | list[Any]
) -> Callable[[Any], Any]:
    """Decorator to handle validating optional dependencies are installed for both
    functions and classes.

    Parameters
    ----------
    extra_name : str
        Name of the extra/optional dependency group present in the pyproject.toml
        E.g. 'corrdiff', 'fcn'
    package_obj : Any | list[Any]
        Object(s) that should be instantiated / imported. If it is a string, it will be
        treated as a module path and checked if it is available. Otherwise this will
        check if the object is instantiated.
    """

    def _check_dependencies(obj_name: Any) -> None:
        package_objs = package_obj if isinstance(package_obj, list) else [package_obj]
        for pkg in package_objs:
            # If package is a string (module path like module.submodule), check if the
            #  module is available.
            if isinstance(pkg, str):
                if find_spec(pkg) is None:
                    logger.error(f"Could not find spec for module {pkg}")
                    raise ExtraDependencyError(extra_name, obj_name)
                else:
                    # Check to make sure that we can actually import the package
                    try:
                        importlib.import_module(pkg)
                    except ImportError as e:
                        raise ImportError(
                            f"Unexpected extra dependency import error of package {pkg}:\n{e}"
                        )
            elif pkg is None:
                raise ExtraDependencyError(extra_name, obj_name)

    def _decorator(obj: F | C) -> F | C:
        if isinstance(obj, type):
            original_init = obj.__init__  # type: ignore

            @wraps(original_init)
            def _wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                _check_dependencies(obj.__name__)
                original_init(self, *args, **kwargs)

            obj.__init__ = _wrapped_init  # type: ignore
            return cast(C, obj)
        else:
            # Handle function decoration
            @wraps(obj)
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                _check_dependencies(f"{obj.__module__}.{obj.__name__}")
                return obj(*args, **kwargs)

            return cast(F, _wrapper)

    return _decorator

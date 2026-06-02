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

import pytest

from earth2studio.utils.imports import (
    OptionalDependencyError,
    OptionalDependencyFailure,
    _check_package,
    _get_group_packages,
    _parse_optional_dependencies,
    check_optional_dependencies,
    pytest_require,
)


@pytest.mark.parametrize(
    "package_obj,should_raise",
    [
        ("json", False),
        ("os", False),
        ("nonexistent_package", True),
    ],
)
def test_extra_check_function(package_obj, should_raise):
    OptionalDependencyFailure.failures = {}
    try:
        __import__(package_obj)
    except ImportError:
        OptionalDependencyFailure("test-group")

    @check_optional_dependencies()
    def func():
        return 1993

    if should_raise:
        with pytest.raises(OptionalDependencyError) as exc_info:
            func()
            assert "test-group" in str(exc_info.value)
            assert "func" in str(exc_info.value)
    else:
        assert func() == 1993


@pytest.mark.parametrize(
    "package_obj,should_raise",
    [
        ("json", False),
        ("os", False),
        ("nonexistent_package", True),
        ("other_package", True),
    ],
)
def test_extra_check_class(package_obj, should_raise):
    OptionalDependencyFailure.failures = {}
    try:
        __import__(package_obj)
    except ImportError:
        OptionalDependencyFailure("test-group", doc_url="https://www.nvidia.com/en-us/")

    @check_optional_dependencies()
    class TestClass:
        def __init__(self):
            self.value = 1993

    if should_raise:
        with pytest.raises(OptionalDependencyError) as exc_info:
            TestClass()
            assert "test-group" in str(exc_info.value)
            assert "TestClass" in str(exc_info.value)
            assert "https://www.nvidia.com/en-us/" in str(exc_info.value)
    else:
        obj = TestClass()
        assert obj.value == 1993


@pytest.mark.parametrize(
    "package_objs,should_raise",
    [
        (["json", "os"], False),
        (["json", "nonexistent_package"], True),
        (["nonexistent1", "nonexistent2"], True),
    ],
)
def test_extra_check_multiple(package_objs, should_raise):
    OptionalDependencyFailure.failures = {}
    try:
        for obj in package_objs:
            __import__(obj)
    except ImportError:
        OptionalDependencyFailure("test-group")

    @check_optional_dependencies()
    def func():
        return 1993

    if should_raise:
        with pytest.raises(OptionalDependencyError) as exc_info:
            func()
            assert "test-group" in str(exc_info.value)
            assert "func" in str(exc_info.value)
    else:
        assert func() == 1993


@pytest.mark.parametrize(
    "obj_type",
    ["function", "class"],
)
def test_extra_check_preserves_metadata(obj_type):
    OptionalDependencyFailure.failures = {}
    try:
        import json  # noqa: F401
    except ImportError:
        OptionalDependencyFailure("test-group")

    if obj_type == "function":

        @check_optional_dependencies()
        def my_func():
            """Test function docstring."""
            return 1993

        obj = my_func
        name = "my_func"
        doc = "Test function docstring."
    else:

        @check_optional_dependencies()
        class MyClass:
            """Test class docstring."""

            pass

        obj = MyClass
        name = "MyClass"
        doc = "Test class docstring."

    assert obj.__name__ == name
    assert obj.__doc__ == doc


def test_extra_check_class_inheritance():
    OptionalDependencyFailure.failures = {}
    try:
        import json  # noqa: F401
    except ImportError:
        OptionalDependencyFailure("test-group", "key")

    class BaseClass:
        def base_method(self):
            return "base"

    @check_optional_dependencies("key")
    class DerivedClass(BaseClass):
        def derived_method(self, input: str):
            return input

    obj = DerivedClass()
    assert obj.base_method() == "base"
    assert obj.derived_method("nvidia") == "nvidia"


# =============================================================================
# Tests for pytest_require() and helper functions
# =============================================================================


class TestCheckPackage:
    """Tests for _check_package() helper."""

    def test_installed_package_no_version(self):
        """Test checking an installed package without version constraint."""
        ok, reason = _check_package("torch")
        assert ok is True
        assert reason is None

    def test_installed_package_version_satisfied(self):
        """Test checking an installed package with satisfied version constraint."""
        ok, reason = _check_package("torch>=2.0")
        assert ok is True
        assert reason is None

    def test_installed_package_version_not_satisfied(self):
        """Test checking an installed package with unsatisfied version constraint."""
        ok, reason = _check_package("torch>=999.0")
        assert ok is False
        assert reason is not None
        assert "does not satisfy" in reason

    def test_missing_package(self):
        """Test checking a package that is not installed."""
        ok, reason = _check_package("nonexistent_package_xyz_12345")
        assert ok is False
        assert reason is not None
        assert "not installed" in reason


class TestParseOptionalDependencies:
    """Tests for _parse_optional_dependencies() helper."""

    def test_returns_dict(self):
        """Test that pyproject.toml is parsed and returns a dict."""
        deps = _parse_optional_dependencies()
        assert isinstance(deps, dict)

    def test_contains_known_groups(self):
        """Test that known dependency groups are present."""
        deps = _parse_optional_dependencies()
        assert "serve" in deps
        assert "data" in deps
        assert "ace2" in deps


class TestGetGroupPackages:
    """Tests for _get_group_packages() helper."""

    def test_known_group(self):
        """Test getting packages for a known group."""
        packages = _get_group_packages("ace2")
        assert isinstance(packages, list)
        assert len(packages) > 0
        # ace2 should include fme
        assert any("fme" in p for p in packages)

    def test_unknown_group_raises(self):
        """Test that unknown group raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dependency group"):
            _get_group_packages("nonexistent_group_xyz_12345")

    def test_recursive_expansion(self):
        """Test that earth2studio[extra] refs are expanded recursively."""
        # stormscope includes earth2studio[utils]
        packages = _get_group_packages("stormscope")
        # Should not contain earth2studio[...] self-references
        assert not any("earth2studio" in p for p in packages)
        # Should contain packages from utils group (e.g., earth2grid)
        # Note: may or may not be present depending on group definition

    def test_no_infinite_recursion(self):
        """Test that circular references don't cause infinite recursion."""
        # 'all' group references many other groups
        packages = _get_group_packages("all")
        assert isinstance(packages, list)


class TestPytestRequire:
    """Tests for pytest_require() function."""

    def test_installed_package_no_skip(self):
        """Test that installed packages don't cause skip."""
        marker = pytest_require("torch")
        # marker.args[0] is the skip condition (bool)
        assert marker.args[0] is False

    def test_missing_package_skips(self):
        """Test that missing packages cause skip."""
        marker = pytest_require("nonexistent_package_xyz_12345")
        assert marker.args[0] is True
        assert "nonexistent_package_xyz_12345" in marker.kwargs["reason"]

    def test_version_satisfied_no_skip(self):
        """Test package with satisfied version constraint doesn't skip."""
        marker = pytest_require("torch>=2.0")
        assert marker.args[0] is False

    def test_version_not_satisfied_skips(self):
        """Test package with unsatisfied version constraint skips."""
        marker = pytest_require("torch>=999.0")
        assert marker.args[0] is True
        assert "does not satisfy" in marker.kwargs["reason"]

    def test_group_unknown_raises(self):
        """Test that unknown group raises ValueError."""
        with pytest.raises(ValueError, match="Unknown dependency group"):
            pytest_require(groups=["nonexistent_group_xyz_12345"])

    def test_group_check_runs(self):
        """Test that group check executes without error."""
        # May or may not skip depending on what's installed
        marker = pytest_require(groups=["ace2"])
        assert hasattr(marker, "args")
        assert hasattr(marker, "kwargs")

    def test_combined_packages_and_groups(self):
        """Test combining explicit packages with groups."""
        marker = pytest_require("torch", groups=["utils"])
        assert hasattr(marker, "args")

    def test_multiple_packages(self):
        """Test checking multiple packages."""
        marker = pytest_require("torch", "numpy")
        assert marker.args[0] is False  # Both should be installed

    def test_multiple_groups(self):
        """Test checking multiple groups."""
        marker = pytest_require(groups=["utils", "perturbation"])
        assert hasattr(marker, "args")

    def test_empty_args_no_skip(self):
        """Test that no requirements means no skip."""
        marker = pytest_require()
        assert marker.args[0] is False

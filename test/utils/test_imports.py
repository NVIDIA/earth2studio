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

import json as js

import pytest

from earth2studio.utils.imports import ExtraDependencyError, check_extra_imports


@pytest.mark.parametrize(
    "package_obj,should_raise",
    [
        ("json", False),
        (js, False),
        ("nonexistent_package", True),
        (None, True),
    ],
)
def test_extra_check_function(package_obj, should_raise):
    @check_extra_imports("test-group", package_obj)
    def func():
        return 1993

    if should_raise:
        with pytest.raises(ExtraDependencyError) as exc_info:
            func()
            assert "test-group" in str(exc_info.value)
            assert "func" in str(exc_info.value)
    else:
        assert func() == 1993


@pytest.mark.parametrize(
    "package_obj,should_raise",
    [
        ("json", False),
        (js, False),
        ("nonexistent_package", True),
        (None, True),
    ],
)
def test_extra_check_class(package_obj, should_raise):
    @check_extra_imports("test-group", package_obj)
    class TestClass:
        def __init__(self):
            self.value = 1993

    if should_raise:
        with pytest.raises(ExtraDependencyError) as exc_info:
            TestClass()
            assert "test-group" in str(exc_info.value)
            assert "TestClass" in str(exc_info.value)
    else:
        obj = TestClass()
        assert obj.value == 1993


@pytest.mark.parametrize(
    "package_objs,should_raise",
    [
        (["json", "os"], False),
        (["json", "nonexistent_package"], True),
        ([js, None], True),
        (["nonexistent1", "nonexistent2"], True),
    ],
)
def test_extra_check_multiple(package_objs, should_raise):
    @check_extra_imports("test-group", package_objs)
    def func():
        return 1993

    if should_raise:
        with pytest.raises(ExtraDependencyError) as exc_info:
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
    if obj_type == "function":

        @check_extra_imports("test-group", "json")
        def my_func():
            """Test function docstring."""
            return 1993

        obj = my_func
        name = "my_func"
        doc = "Test function docstring."
    else:

        @check_extra_imports("test-group", "json")
        class MyClass:
            """Test class docstring."""

            pass

        obj = MyClass
        name = "MyClass"
        doc = "Test class docstring."

    assert obj.__name__ == name
    assert obj.__doc__ == doc


def test_extra_check_class_inheritance():
    class BaseClass:
        def base_method(self):
            return "base"

    @check_extra_imports("test-group", "json")
    class DerivedClass(BaseClass):
        def derived_method(self, input: str):
            return input

    obj = DerivedClass()
    assert obj.base_method() == "base"
    assert obj.derived_method("nvidia") == "nvidia"

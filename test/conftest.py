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

import os

import pytest


@pytest.fixture(scope="session")
def model_cache_context():
    class EnvContextManager:
        def __init__(self, **kwargs):
            # Set default to CI cache
            self.env_vars = {}
            # Over rider with inputs
            for key, value in kwargs.items():
                self.env_vars[key] = value

        def __enter__(self):
            self.old_values = {
                key: os.environ.get(key)
                for key in self.env_vars.keys()
                if os.environ.get(key) is not None
            }
            os.environ.update(self.env_vars)

        def __exit__(self, exc_type, exc_value, exc_traceback):
            os.environ.update(self.old_values)

    return EnvContextManager


def pytest_addoption(parser):
    parser.addoption(
        "--package-download",
        action="store_true",
        default=False,
        help="test auto model package downloads and generate cache",
    )
    parser.addoption(
        "--package",
        action="store_true",
        default=False,
        help="test model packages using CI cache ",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "package: mark test as requiring model package cache"
    )
    config.addinivalue_line(
        "markers",
        "package_download: mark test as requiring a model download from external store",
    )


def pytest_collection_modifyitems(config, items):

    enable_packages = config.getoption("--package") or os.getenv(
        "EARTH2STUDIO_TEST_PACKAGES", ""
    ).strip().lower() in ("1", "true", "yes", "on")

    if not enable_packages:
        skip_model_package = pytest.mark.skip(
            reason="need --package option to run model package tests"
        )
        for item in items:
            if "package" in item.keywords:
                item.add_marker(skip_model_package)

    enable_download = config.getoption("--package-download") or os.getenv(
        "EARTH2STUDIO_DOWNLOAD_PACKAGES", ""
    ).strip().lower() in ("1", "true", "yes", "on")

    if not enable_download:
        skip_download = pytest.mark.skip(
            reason="need --package-download option to run package download tests"
        )
        for item in items:
            if "package_download" in item.keywords:
                item.add_marker(skip_download)

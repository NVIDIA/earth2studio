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

import os
from pathlib import Path

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
        "--model-download",
        action="store_true",
        default=False,
        help="test auto model package downloads and generate cache",
    )
    parser.addoption(
        "--ci-cache",
        action="store_true",
        default=False,
        help="test model packages using CI cache ",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "ci_cache: mark test as requiring model package cache"
    )
    config.addinivalue_line(
        "markers",
        "model_download: mark test as requiring a model download from external store",
    )


def pytest_collection_modifyitems(config, items):

    if (
        not config.getoption("--ci-cache")
        or not Path("/data/nfs/earth2studio-cache").is_dir()
    ):
        skip_ci_cache = pytest.mark.skip(
            reason="need --ci-cache option to model packages from CI cache and /data/nfs/earth2studio-cache must exist"
        )
        for item in items:
            if "ci_cache" in item.keywords:
                item.add_marker(skip_ci_cache)

    if not config.getoption("--model-download"):
        skip_model_download = pytest.mark.skip(
            reason="need --model-download option to run automodel download tests"
        )
        for item in items:
            if "model_download" in item.keywords:
                item.add_marker(skip_model_download)

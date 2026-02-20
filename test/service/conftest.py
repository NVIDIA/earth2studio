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

"""
Shared pytest configuration for service tests.

This conftest.py ensures that the service/inferenceserver directory is in the
Python path so that api_server modules can be imported.
"""

import sys
from pathlib import Path

# Add service/inferenceserver to Python path for imports
_inferenceserver_path = (
    Path(__file__).parent.parent.parent / "service" / "inferenceserver"
)
if str(_inferenceserver_path) not in sys.path:
    sys.path.insert(0, str(_inferenceserver_path))

# Store original api_server.config module to restore it if it gets mocked
_original_config_module = None


def pytest_configure(config):
    """Store the original api_server.config module before any tests run"""
    global _original_config_module
    if "api_server.config" in sys.modules:
        _original_config_module = sys.modules["api_server.config"]


def pytest_runtest_setup(item):
    """Restore the real api_server.config module before test_config.py tests"""
    # Only restore for test_config.py tests to avoid interfering with other tests
    if "test_config" in str(item.fspath):
        import importlib

        # Remove any mock that might have been set by other test files
        if "api_server.config" in sys.modules:
            # Check if it's a mock (Mock objects have _mock_name attribute or are Mock instances)
            config_module = sys.modules["api_server.config"]
            is_mock = (
                hasattr(config_module, "_mock_name")
                or str(type(config_module)) == "<class 'unittest.mock.Mock'>"
                or (
                    hasattr(config_module, "__class__")
                    and "Mock" in str(config_module.__class__)
                )
            )
            if is_mock:
                # It's a mock, restore the real module
                # First, clear any references in parent modules
                if "api_server" in sys.modules:
                    api_server_module = sys.modules["api_server"]
                    if (
                        hasattr(api_server_module, "config")
                        and api_server_module.config is config_module
                    ):
                        delattr(api_server_module, "config")
                # Remove the mock from sys.modules
                del sys.modules["api_server.config"]
                # Also clear any submodules that might reference the mock
                modules_to_remove = [
                    key
                    for key in sys.modules.keys()
                    if key.startswith("api_server.config.")
                ]
                for key in modules_to_remove:
                    del sys.modules[key]
        # Force reload the real module to ensure fresh state
        if "api_server.config" not in sys.modules:
            importlib.import_module("api_server.config")
        else:
            # Only reload if it was a mock, otherwise just ensure state is correct
            config_module = sys.modules["api_server.config"]
            is_mock = (
                hasattr(config_module, "_mock_name")
                or str(type(config_module)) == "<class 'unittest.mock.Mock'>"
            )
            if is_mock:
                importlib.reload(sys.modules["api_server.config"])
        # Ensure _config_manager and ConfigManager._instance are properly initialized
        real_config = sys.modules["api_server.config"]
        # Check if _config_manager is a mock and reset if needed
        cm = getattr(real_config, "_config_manager", None)
        if cm is not None and (
            hasattr(cm, "_mock_name") or str(type(cm)) == "<class 'unittest.mock.Mock'>"
        ):
            real_config._config_manager = None
        # Check if ConfigManager._instance is a mock and reset if needed
        if hasattr(real_config, "ConfigManager") and hasattr(
            real_config.ConfigManager, "_instance"
        ):
            inst = getattr(real_config.ConfigManager, "_instance", None)
            if inst is not None and (
                hasattr(inst, "_mock_name")
                or str(type(inst)) == "<class 'unittest.mock.Mock'>"
            ):
                real_config.ConfigManager._instance = None

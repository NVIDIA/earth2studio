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

This conftest.py ensures the Python path allows earth2studio.serve.server
modules to be imported (earth2studio is the project package).
"""

import sys
from pathlib import Path

# Repo root is parent of test/serve/server; ensure project is importable
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Store original earth2studio.serve.server.config module to restore if mocked
_original_config_module = None

_CONFIG_MODULE = "earth2studio.serve.server.config"


def pytest_configure(config):
    """Store the original config module before any tests run"""
    global _original_config_module
    if _CONFIG_MODULE in sys.modules:
        _original_config_module = sys.modules[_CONFIG_MODULE]


def pytest_runtest_setup(item):
    """Restore the real config module before test_config.py tests"""
    if "test_config" not in str(item.fspath):
        return
    import importlib

    if _CONFIG_MODULE in sys.modules:
        config_module = sys.modules[_CONFIG_MODULE]
        is_mock = (
            hasattr(config_module, "_mock_name")
            or str(type(config_module)) == "<class 'unittest.mock.Mock'>"
            or (
                hasattr(config_module, "__class__")
                and "Mock" in str(config_module.__class__)
            )
        )
        if is_mock:
            if "earth2studio.serve.server" in sys.modules:
                server_module = sys.modules["earth2studio.serve.server"]
                if (
                    hasattr(server_module, "config")
                    and server_module.config is config_module
                ):
                    delattr(server_module, "config")
            del sys.modules[_CONFIG_MODULE]
            modules_to_remove = [
                k for k in sys.modules if k.startswith(_CONFIG_MODULE + ".")
            ]
            for k in modules_to_remove:
                del sys.modules[k]
    if _CONFIG_MODULE not in sys.modules:
        importlib.import_module(_CONFIG_MODULE)
    else:
        config_module = sys.modules[_CONFIG_MODULE]
        is_mock = (
            hasattr(config_module, "_mock_name")
            or str(type(config_module)) == "<class 'unittest.mock.Mock'>"
        )
        if is_mock:
            importlib.reload(sys.modules[_CONFIG_MODULE])
    real_config = sys.modules[_CONFIG_MODULE]
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

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

"""Model-package selection: local model.path vs the published default (Hugging Face)."""

from __future__ import annotations

import pytest
from src.stormcast import workflow


class _FakePackage:
    """Stand-in for earth2studio's Package: records the path it was built from."""

    def __init__(self, path: str, cache_options: dict | None = None) -> None:
        self.path = path
        self.cache_options = cache_options

    @staticmethod
    def default_cache(name: str) -> str:
        return f"/cache/{name}"


def test_resolve_model_package_uses_local_path_when_set() -> None:
    def default_loader() -> object:
        raise AssertionError("default loader must not be used for a local path")

    pkg = workflow._resolve_model_package(
        {"path": "/ckpt"}, _FakePackage, default_loader
    )

    assert pkg.path == "/ckpt"  # honored as-is
    assert pkg.cache_options == {
        "cache_storage": "/cache/stormcast-conus",
        "same_names": True,
    }


@pytest.mark.parametrize("model_cfg", [{}, {"path": ""}, {"path": None}])
def test_resolve_model_package_falls_back_to_default_when_no_path(
    model_cfg: dict[str, object],
) -> None:
    def default_loader() -> object:
        return "DEFAULT_PACKAGE"

    assert (
        workflow._resolve_model_package(model_cfg, _FakePackage, default_loader)
        == "DEFAULT_PACKAGE"
    )

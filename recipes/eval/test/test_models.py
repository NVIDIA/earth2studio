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

"""Model loading: every load, including the package download, runs rank 0 first,
through the collective-safe ``rank0_first_checked``."""

from __future__ import annotations

import src.models as models

from earth2studio.models.px import Persistence


class _FakeAuto:
    """Stand-in for an ``AutoModelMixin`` class recording its load steps."""

    calls: list[str] = []

    @classmethod
    def load_default_package(cls) -> str:
        cls.calls.append("package")
        return "pkg"

    @classmethod
    def load_model(cls, package: str, **kwargs) -> _FakeAuto:
        cls.calls.append(f"load:{package}:{sorted(kwargs)}")
        return cls()


def _spy(seen: list) -> object:
    def rank0_first_checked(fn):
        result = fn()
        seen.append((type(result).__name__, list(_FakeAuto.calls)))
        return result

    return rank0_first_checked


def test_target_nodes_instantiate_inside_rank0_first(base_cfg, monkeypatch) -> None:
    seen: list = []
    monkeypatch.setattr(models, "rank0_first_checked", _spy(seen))
    model = models.load_from_node(base_cfg.model, "prognostic model")
    assert isinstance(model, Persistence)
    assert [name for name, _ in seen] == ["Persistence"]


def test_architecture_nodes_download_and_load_inside_rank0_first(monkeypatch) -> None:
    seen: list = []
    _FakeAuto.calls.clear()
    monkeypatch.setattr(models, "rank0_first_checked", _spy(seen))
    node = {"architecture": "test.test_models._FakeAuto", "load_args": {"amp": True}}
    model = models.load_from_node(node)
    assert isinstance(model, _FakeAuto)
    # Both the package lookup and load_model had run when the helper returned.
    assert seen == [("_FakeAuto", ["package", "load:pkg:['amp']"])]

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

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

icechunk = pytest.importorskip("icechunk", reason="icechunk not installed")

import earth2studio.io.arraylake as arraylake_io  # noqa: E402
from earth2studio.io import ArraylakeBackend  # noqa: E402
from earth2studio.utils.imports import OptionalDependencyFailure  # noqa: E402


class FakeClient:
    """Stand-in for arraylake.Client that vends real in-memory Icechunk repos.

    Arraylake hands back a genuine ``icechunk.Repository``, so serving one backed by
    in-memory storage exercises the whole backend without network or credentials.
    """

    def __init__(self, service_uri: str | None = None, token: str | None = None):
        self.token = token
        self.calls: list[tuple[str, str, dict]] = []
        self._repos: dict[str, icechunk.Repository] = {}

    def _repo(self, name: str) -> icechunk.Repository:
        if name not in self._repos:
            self._repos[name] = icechunk.Repository.open_or_create(
                icechunk.in_memory_storage()
            )
        return self._repos[name]

    def get_or_create_repo(self, name: str, **kwargs: object) -> icechunk.Repository:
        self.calls.append(("get_or_create_repo", name, dict(kwargs)))
        return self._repo(name)


@pytest.fixture
def patch_arraylake(monkeypatch):
    """Patch the arraylake module in the backend and return the fake client class."""
    # Required so check_optional_dependencies does not raise when arraylake is absent
    monkeypatch.delitem(
        OptionalDependencyFailure.failures, arraylake_io.__file__, raising=False
    )
    monkeypatch.delenv(arraylake_io._API_KEY_ENV_VAR, raising=False)
    created: list[FakeClient] = []

    def _client_factory(*args: object, **kwargs: object) -> FakeClient:
        client = FakeClient(*args, **kwargs)  # type: ignore[arg-type]
        created.append(client)
        return client

    monkeypatch.setattr(
        arraylake_io, "arraylake", SimpleNamespace(Client=_client_factory)
    )
    return created


def test_arraylake_repo_resolution(patch_arraylake: list[FakeClient]) -> None:
    """The repo name is resolved through the client, with repo_kwargs passed on."""

    client = FakeClient()

    io = ArraylakeBackend(
        "test-org/new-repo",
        client=client,
        repo_kwargs={"bucket_config_nickname": "default"},
    )
    assert io.repo_name == "test-org/new-repo"
    assert io.repo is client._repos["test-org/new-repo"]
    assert client.calls == [
        (
            "get_or_create_repo",
            "test-org/new-repo",
            {"bucket_config_nickname": "default"},
        )
    ]
    # An injected client is used as is, so no client is built from the environment
    assert patch_arraylake == []

    # No repo_kwargs means none are invented
    client.calls.clear()
    ArraylakeBackend("test-org/new-repo", client=client)
    assert client.calls == [("get_or_create_repo", "test-org/new-repo", {})]


def test_arraylake_forwards_backend_options(patch_arraylake: list[FakeClient]) -> None:
    """Backend options reach IceChunkBackend, and writes land on the vended repo.

    Everything below construction is inherited and covered by test_icechunk.py; what
    is specific here is that __init__ forwards each option to super() correctly and
    that the parent machinery drives a client-vended repository end to end.
    """

    total_coords = OrderedDict(
        {
            "time": np.asarray([np.datetime64("2021-01-01")]),
            "variable": np.asarray(["t2m"]),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    shape = tuple(len(dim) for dim in total_coords.values())

    io = ArraylakeBackend(
        "test-org/test-repo",
        branch="experiment",
        client=FakeClient(),
        chunks={"time": 1, "variable": 1, "lat": 4, "lon": 16},
        blocking=True,
    )
    assert io.branch == "experiment"
    assert "experiment" in io.repo.list_branches()
    assert io._blocking is True
    assert io._executor is None

    io.add_array(total_coords, "fields")
    assert io["fields"].chunks == (1, 1, 4, 16)

    x = torch.randn(shape, dtype=torch.float32)
    io.write(x, total_coords, "fields")
    io.commit("write fields")

    readonly = io.repo.readonly_session("experiment")
    root = zarr.open_group(readonly.store, mode="r")
    assert np.allclose(root["fields"][:], x.numpy())


def test_arraylake_token_precedence(
    monkeypatch, patch_arraylake: list[FakeClient]
) -> None:

    explicit, from_env = "explicit-token", "env-key"

    # Explicit token wins over the environment
    monkeypatch.setenv(arraylake_io._API_KEY_ENV_VAR, from_env)
    ArraylakeBackend("test-org/test-repo", token=explicit)
    assert patch_arraylake[-1].token == explicit

    # Environment variable is used when no token is passed
    ArraylakeBackend("test-org/test-repo")
    assert patch_arraylake[-1].token == from_env

    # With neither, a bare Client() is built so arraylake resolves its own auth
    monkeypatch.delenv(arraylake_io._API_KEY_ENV_VAR)
    ArraylakeBackend("test-org/test-repo")
    assert patch_arraylake[-1].token is None


def test_arraylake_uncommitted_warning(patch_arraylake: list[FakeClient]) -> None:
    from loguru import logger

    messages: list[str] = []
    handler_id = logger.add(messages.append, level="WARNING")

    total_coords = OrderedDict(
        {
            "time": np.asarray([np.datetime64("2021-01-01")]),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    try:
        io = ArraylakeBackend("test-org/test-repo")
        io.add_array(total_coords, "fields")
        io.__del__()
    finally:
        logger.remove(handler_id)

    # Warning names the concrete subclass, not IceChunkBackend
    assert any(
        "ArraylakeBackend deleted with uncommitted changes" in m for m in messages
    )

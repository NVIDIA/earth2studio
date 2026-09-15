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

    def get_repo(self, name: str, **kwargs: object) -> icechunk.Repository:
        self.calls.append(("get_repo", name, dict(kwargs)))
        if name not in self._repos:
            raise ValueError(f"repo {name} does not exist")
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


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize("variable", [["t2m"], ["t2m", "tcwv"]])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_arraylake_field(
    time: list[np.datetime64],
    variable: list[str],
    device: str,
    patch_arraylake: list[FakeClient],
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    chunks = OrderedDict({"time": 1, "variable": 1, "lat": 180, "lon": 180})

    io = ArraylakeBackend("test-org/test-repo", chunks=chunks)
    assert io.repo_name == "test-org/test-repo"
    assert isinstance(io.repo, icechunk.Repository)
    assert isinstance(io.root, zarr.Group)

    array_name = "fields"
    io.add_array(total_coords, array_name)

    for dim in total_coords:
        assert dim in io
        assert dim in io.coords
        assert io[dim].shape == total_coords[dim].shape

    assert array_name in io

    shape = tuple(len(dim) for dim in total_coords.values())
    assert io[array_name].shape == shape

    x = torch.randn(shape, device=device, dtype=torch.float32)
    io.write(x, total_coords, array_name)
    assert np.allclose(io[array_name][:], x.to("cpu").numpy())

    xx, _ = io.read(total_coords, array_name, device=device)
    assert torch.allclose(x, xx)

    # Commit and confirm the write survives a fresh readonly session
    snapshot_id = io.commit("write fields")
    assert isinstance(snapshot_id, str)

    readonly = io.repo.readonly_session(io.branch)
    root = zarr.open_group(readonly.store, mode="r")
    assert np.allclose(root[array_name][:], x.to("cpu").numpy())


def test_arraylake_create_flag(patch_arraylake: list[FakeClient]) -> None:

    client = FakeClient()

    # create=True resolves through get_or_create_repo and passes repo_kwargs on
    io = ArraylakeBackend(
        "test-org/new-repo",
        client=client,
        repo_kwargs={"bucket_config_nickname": "default"},
    )
    assert client.calls == [
        (
            "get_or_create_repo",
            "test-org/new-repo",
            {"bucket_config_nickname": "default"},
        )
    ]
    assert isinstance(io.repo, icechunk.Repository)

    # create=False resolves through get_repo, which fails on a missing repo
    client.calls.clear()
    io2 = ArraylakeBackend("test-org/new-repo", client=client, create=False)
    assert client.calls == [("get_repo", "test-org/new-repo", {})]
    assert isinstance(io2.repo, icechunk.Repository)

    with pytest.raises(ValueError):
        ArraylakeBackend("test-org/missing", client=client, create=False)


def test_arraylake_branch(patch_arraylake: list[FakeClient]) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray([np.datetime64("2021-01-01")]),
            "variable": np.asarray(["t2m"]),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    array_name = "fields"
    shape = tuple(len(dim) for dim in total_coords.values())
    client = FakeClient()

    io = ArraylakeBackend("test-org/test-repo", branch="experiment", client=client)
    assert io.branch == "experiment"
    assert "experiment" in io.repo.list_branches()

    io.add_array(total_coords, array_name)
    x = torch.randn(shape, dtype=torch.float32)
    io.write(x, total_coords, array_name)
    io.commit("write fields")

    # Reopening the same repo on the same branch sees the committed data
    io2 = ArraylakeBackend("test-org/test-repo", branch="experiment", client=client)
    assert array_name in io2
    xx, _ = io2.read(total_coords, array_name)
    assert torch.allclose(x, xx)

    # The untouched main branch does not
    io3 = ArraylakeBackend("test-org/test-repo", client=client)
    assert array_name not in io3


def test_arraylake_injected_client_skips_auth(
    monkeypatch, patch_arraylake: list[FakeClient]
) -> None:

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("_make_client should not be called with client=")

    monkeypatch.setattr(ArraylakeBackend, "_make_client", staticmethod(_boom))

    client = FakeClient()
    io = ArraylakeBackend("test-org/test-repo", client=client)
    assert io.repo is client._repos["test-org/test-repo"]


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

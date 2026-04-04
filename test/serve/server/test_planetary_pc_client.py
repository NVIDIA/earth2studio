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

import sys
from pathlib import Path

# Top-level package azure_planetary_computer lives under <repo>/serve/server (see start_api_server.sh PYTHONPATH).
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
_serve_server = _repo_root / "serve" / "server"
if str(_serve_server) not in sys.path:
    sys.path.insert(0, str(_serve_server))

from datetime import datetime, timedelta, timezone  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("earth2studio.serve.server")

from azure_planetary_computer.pc_client import PlanetaryComputerClient  # noqa: E402


@pytest.fixture
def mock_azure_credential():
    with patch("azure.identity.DefaultAzureCredential") as m:
        cred = MagicMock()
        token = MagicMock()
        token.token = "mock-token"  # noqa: S105
        cred.get_token.return_value = token
        m.return_value = cred
        yield m


@pytest.fixture
def mock_requests_pc():
    with patch("azure_planetary_computer.pc_client.requests") as m:
        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = {"status": "Succeeded"}
        get_resp.headers = {}
        m.get.return_value = get_resp

        post_resp = MagicMock()
        post_resp.status_code = 201
        post_resp.headers = {"location": "https://geocatalog.example/status/123"}
        m.post.return_value = post_resp

        put_resp = MagicMock()
        put_resp.status_code = 200
        m.put.return_value = put_resp

        yield m


def test_pc_client_resolve_start_time_iso_string(mock_azure_credential):
    client = PlanetaryComputerClient("foundry_fcn3_workflow")
    params = {"start_time": "2025-01-15T12:00:00Z"}
    result = client._resolve_start_time(params)
    assert result == datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_pc_client_resolve_start_time_stormscope_key(mock_azure_credential):
    client = PlanetaryComputerClient("foundry_fcn3_stormscope_goes_workflow")
    params = {"start_time_stormscope": "2025-01-15T12:00:00Z"}
    result = client._resolve_start_time(params)
    assert result == datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_pc_client_get_feature_json_formats_times(mock_azure_credential):
    client = PlanetaryComputerClient("foundry_fcn3_workflow")
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=6)
    out = client._get_feature_json(
        start_time=start,
        end_time=end,
        blob_url="https://storage.example/container/blob.nc",
    )
    assert out["properties"]["datetime"] == start.isoformat()
    assert out["assets"]["data"]["href"] == "https://storage.example/container/blob.nc"
    assert out["id"].startswith("fcn3-")


def test_pc_client_create_feature_new_collection(
    mock_azure_credential, mock_requests_pc
):
    mock_requests_pc.get.return_value.json.return_value = {"status": "Succeeded"}
    client = PlanetaryComputerClient("foundry_fcn3_workflow")
    collection_id, feature_id = client.create_feature(
        geocatalog_url="https://geocatalog.example/",
        collection_id=None,
        parameters={"start_time": "2025-01-01T00:00:00Z"},
        blob_url="https://storage.example/blob.nc",
    )
    assert collection_id is not None
    assert feature_id is not None
    assert feature_id.startswith("fcn3-")

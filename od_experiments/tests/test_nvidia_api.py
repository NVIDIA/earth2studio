# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from od_experiments.earth2_001.nvidia_api import (
    NvidiaAPIError,
    NvidiaAPIKeyMissingError,
    NvidiaAuthenticationError,
    NvidiaClientConfig,
    NvidiaCloudClient,
    NvidiaConnectionError,
    NvidiaRateLimitError,
    NvidiaTimeoutError,
    find_and_load_api_key,
)


def test_api_key_loading_from_env_vars(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-testkey-env-1234567890")
    key = find_and_load_api_key()
    assert key == "nvapi-testkey-env-1234567890"

    # Test precedence with NVIDIA_API
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API", "nvapi-alternate-9876543210")
    assert find_and_load_api_key() == "nvapi-alternate-9876543210"

    # Test KEY= prefix stripping
    monkeypatch.setenv("NVIDIA_API", "KEY=nvapi-with-prefix-1234567890")
    assert find_and_load_api_key() == "nvapi-with-prefix-1234567890"


def test_api_key_loading_from_dotenv_file(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    env_file = tmp_path / ".env.local"
    env_file.write_text(
        "NVIDIA_API_KEY=nvapi-file-loaded-123456789\nSOME_OTHER_KEY=hello\n"
    )

    loaded_key = find_and_load_api_key(env_file=env_file)
    assert loaded_key == "nvapi-file-loaded-123456789"


def test_missing_key_handling(monkeypatch):
    """Test 1: Missing key handling when no environment or local file key is present."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)

    client = NvidiaCloudClient(
        api_key=None,
        config=NvidiaClientConfig(mock_mode=False, search_env_files=False),
    )
    assert not client.is_authenticated
    assert client.get_masked_key() == "[NO_KEY_CONFIGURED]"

    with pytest.raises(NvidiaAPIKeyMissingError) as exc_info:
        client.request("GET", "/models")
    assert "missing" in str(exc_info.value).lower()


@patch("requests.Session.request")
def test_invalid_key_authentication_rejection(mock_request):
    """Test 2: Invalid key authentication rejection (HTTP 401 & 403)."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": "Invalid API key provided"}
    mock_resp.text = '{"error": "Invalid API key provided"}'
    mock_request.return_value = mock_resp

    client = NvidiaCloudClient(
        api_key="nvapi-invalid-key-123456", config=NvidiaClientConfig(mock_mode=False)
    )
    with pytest.raises(NvidiaAuthenticationError) as exc_info:
        client.request("GET", "models")
    assert exc_info.value.status_code == 401

    # Verify HTTP 403 Forbidden rejection as well
    mock_resp.status_code = 403
    mock_resp.json.return_value = {"error": "Permission denied"}
    with pytest.raises(NvidiaAuthenticationError) as exc_info:
        client.request("GET", "models")
    assert exc_info.value.status_code == 403


def test_header_construction():
    """Test 3: Header construction (Authorization Bearer, User-Agent, Content-Type, custom headers)."""
    client = NvidiaCloudClient(
        api_key="nvapi-test-secret-header-12345",
        config=NvidiaClientConfig(mock_mode=False),
    )
    headers = client._get_headers(custom_headers={"X-Custom-Trace": "od-test-123"})
    assert headers["Authorization"] == "Bearer nvapi-test-secret-header-12345"
    assert headers["Content-Type"] == "application/json"
    assert headers["Accept"] == "application/json"
    assert "Earth2Studio" in headers["User-Agent"]
    assert headers["X-Custom-Trace"] == "od-test-123"

    # In mock mode without key, Authorization header should not be added
    mock_client = NvidiaCloudClient(config=NvidiaClientConfig(mock_mode=True))
    mock_headers = mock_client._get_headers()
    assert "Authorization" not in mock_headers


@patch("requests.Session.request")
def test_timeout_and_connection_handling(mock_request):
    """Test 4: Timeout handling and connection error mapping."""
    client = NvidiaCloudClient(
        api_key="nvapi-fake-key-123456", config=NvidiaClientConfig(mock_mode=False)
    )

    mock_request.side_effect = requests.exceptions.Timeout(
        "Connection timed out after 30s"
    )
    with pytest.raises(NvidiaTimeoutError) as exc_info:
        client.request("GET", "status")
    assert "timed out" in str(exc_info.value).lower()

    mock_request.side_effect = requests.exceptions.ConnectionError(
        "Failed to resolve host"
    )
    with pytest.raises(NvidiaConnectionError) as exc_info:
        client.request("GET", "status")
    assert "failed to connect" in str(exc_info.value).lower()


@patch("requests.Session.request")
def test_rate_limit_error_and_backoff_handling(mock_request):
    """Test 5: Rate limit (429) backoff / Retry-After header parsing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.headers = {"Retry-After": "7.5"}
    mock_resp.json.return_value = {
        "error": "Too Many Requests",
        "message": "Quota limit reached",
    }
    mock_request.return_value = mock_resp

    client = NvidiaCloudClient(
        api_key="nvapi-fake-key-123456", config=NvidiaClientConfig(mock_mode=False)
    )
    with pytest.raises(NvidiaRateLimitError) as exc_info:
        client.request("POST", "infer")
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == 7.5
    assert "rate limit exceeded" in str(exc_info.value).lower()


@patch("requests.Session.request")
def test_response_payload_parsing(mock_request):
    """Test 6: Response payload parsing for JSON objects and fallback formats."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": "success",
        "predictions": [0.12, 0.45, 0.78],
        "model": "nvidia/fourcastnet",
    }
    mock_request.return_value = mock_resp

    client = NvidiaCloudClient(
        api_key="nvapi-fake-key-123456", config=NvidiaClientConfig(mock_mode=False)
    )
    parsed = client.request("POST", "predict", json_data={"lead_time": 6})
    assert parsed["status"] == "success"
    assert parsed["model"] == "nvidia/fourcastnet"
    assert len(parsed["predictions"]) == 3

    # Non-dict text response fallback
    mock_resp.json.side_effect = ValueError("Non-JSON text")
    mock_resp.text = "RAW_STRING_RESPONSE"
    parsed_text = client.request("GET", "text-endpoint")
    assert parsed_text["status"] == "success"
    assert parsed_text["data"] == "RAW_STRING_RESPONSE"


def test_secret_redaction_in_exceptions_and_repr():
    """Test 7: Secret redaction in exceptions, repr, and logs."""
    sensitive_token = "nvapi-SECRETTOKEN1234567890ABC"
    client = NvidiaCloudClient(
        api_key=sensitive_token,
        config=NvidiaClientConfig(mock_mode=False),
    )

    # 1. repr check
    client_repr = repr(client)
    assert sensitive_token not in client_repr
    assert "SECRETTOKEN" not in client_repr
    assert client.get_masked_key() == "nvap****0ABC"

    # 2. NvidiaAPIError exception scrubber check
    err = NvidiaAPIError(
        f"Failed query with key={sensitive_token} and auth Bearer {sensitive_token}",
        status_code=500,
        details={"token": sensitive_token, "nested": {"auth_key": sensitive_token}},
    )
    err_str = str(err)
    assert sensitive_token not in err_str
    assert "[REDACTED]" in err_str
    assert err.details["token"] == "[REDACTED]"
    assert err.details["nested"]["auth_key"] == "[REDACTED]"


def test_offline_fallback_mode():
    """Test 8: Offline fallback / mock mode without credit consumption or network requests."""
    client = NvidiaCloudClient(config=NvidiaClientConfig(mock_mode=True))
    assert client.is_authenticated
    assert client.get_masked_key() == "[MOCK_MODE]"

    # Probe health in offline mock mode
    health = client.probe_health()
    assert health["status"] == "ok"
    assert health["mode"] == "mock"

    # Verify mock models query
    models = client.request("GET", "models")
    assert "data" in models
    assert any(m["id"] == "nvidia/fourcastnet" for m in models["data"])

    # Verify mock inference
    inference = client.request("POST", "infer", json_data={"lead_time_hours": 12})
    assert inference["status"] == "completed"
    assert inference["result"]["lead_time_hours"] == 12

    # Verify custom mock handler
    def custom_handler(method, endpoint, payload):
        return {"custom_handled": True, "method": method, "endpoint": endpoint}

    client_custom = NvidiaCloudClient(
        config=NvidiaClientConfig(mock_mode=True), mock_handler=custom_handler
    )
    res = client_custom.request("POST", "custom/endpoint")
    assert res["custom_handled"] is True
    assert res["method"] == "POST"
    assert res["endpoint"] == "custom/endpoint"


@patch("requests.Session.request")
def test_live_probe_health(mock_request):
    """Test 9: Live probe health check when authenticated."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"id": "nvidia/fourcastnet"}]}
    mock_request.return_value = mock_resp

    client = NvidiaCloudClient(
        api_key="nvapi-live-test-12345678", config=NvidiaClientConfig(mock_mode=False)
    )
    health = client.probe_health()
    assert health["status"] == "ok"
    assert health["mode"] == "live"
    assert health["authenticated"] is True
    assert health["models_count"] == 1
    assert "latency_ms" in health

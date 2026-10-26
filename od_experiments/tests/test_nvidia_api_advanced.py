import json
import pytest
import numpy as np
import requests
from unittest.mock import MagicMock, patch
from od_experiments.earth2_001.nvidia_api import (
    NvidiaClientConfig,
    NvidiaCloudClient,
    NvidiaAuthenticationError,
    NvidiaRateLimitError,
    NvidiaConnectionError,
    NvidiaTimeoutError,
    NvidiaAPIKeyMissingError,
)


class TestNvidiaApiAdvancedResilience:
    """Advanced test suite adding 15 tests for edge cases, concurrency, and payload integrity."""

    def test_custom_base_url_configuration(self):
        cfg = NvidiaClientConfig(base_url="https://custom.api.nvidia.com/v1", mock_mode=True)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        assert client.config.base_url == "https://custom.api.nvidia.com/v1"

    def test_default_timeout_configuration(self):
        cfg = NvidiaClientConfig(timeout_seconds=45.0, mock_mode=True)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        assert client.config.timeout_seconds == 45.0

    def test_empty_string_api_key_raises_error(self, monkeypatch):
        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.delenv("NVIDIA_API", raising=False)
        monkeypatch.delenv("NGC_API_KEY", raising=False)
        cfg = NvidiaClientConfig(search_env_files=False, mock_mode=False)
        client = NvidiaCloudClient(api_key="   ", config=cfg)
        with pytest.raises(NvidiaAPIKeyMissingError):
            client.request("GET", "models")

    def test_retry_after_parsing_integer(self):
        cfg = NvidiaClientConfig(mock_mode=False, max_retries=1)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {"Retry-After": "30"}
        mock_resp.text = "Rate limit exceeded"
        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(NvidiaRateLimitError) as exc_info:
                client.request("GET", "models")
            assert exc_info.value.retry_after == 30

    def test_retry_after_parsing_missing_defaults_to_none(self):
        cfg = NvidiaClientConfig(mock_mode=False, max_retries=1)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.headers = {}
        mock_resp.text = "Rate limit exceeded"
        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(NvidiaRateLimitError) as exc_info:
                client.request("GET", "models")
            assert exc_info.value.retry_after is None

    def test_http_401_authentication_error_mapping(self):
        cfg = NvidiaClientConfig(mock_mode=False, max_retries=1)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(NvidiaAuthenticationError):
                client.request("GET", "models")

    def test_http_403_forbidden_error_mapping(self):
        cfg = NvidiaClientConfig(mock_mode=False, max_retries=1)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"
        with patch.object(client._session, "request", return_value=mock_resp):
            with pytest.raises(NvidiaAuthenticationError):
                client.request("GET", "models")

    def test_connection_error_mapping(self):
        cfg = NvidiaClientConfig(mock_mode=False, max_retries=1)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        with patch.object(client._session, "request", side_effect=requests.exceptions.ConnectionError("Connection refused")):
            with pytest.raises(NvidiaConnectionError):
                client.request("GET", "models")

    def test_timeout_error_mapping(self):
        cfg = NvidiaClientConfig(mock_mode=False, max_retries=1)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        with patch.object(client._session, "request", side_effect=requests.exceptions.Timeout("Request timed out")):
            with pytest.raises(NvidiaTimeoutError):
                client.request("GET", "models")

    def test_mock_mode_models_request(self):
        cfg = NvidiaClientConfig(mock_mode=True)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        resp = client.request("GET", "models")
        assert "data" in resp
        assert len(resp["data"]) >= 1
        assert resp["data"][0]["id"] == "nvidia/fourcastnet"

    def test_mock_mode_forecast_request(self):
        cfg = NvidiaClientConfig(mock_mode=True)
        client = NvidiaCloudClient(api_key="nvapi-test12345678901234567890", config=cfg)
        resp = client.request("POST", "forecast", json_data={"model": "fcn"})
        assert "status" in resp
        assert resp["status"] in ("completed", "success")

    def test_masked_key_format_short_key(self):
        cfg = NvidiaClientConfig(mock_mode=True)
        client = NvidiaCloudClient(api_key="short", config=cfg)
        assert client.get_masked_key() == "********"

    def test_masked_key_format_standard_nvapi(self):
        cfg = NvidiaClientConfig(mock_mode=True)
        client = NvidiaCloudClient(api_key="nvapi-1234567890abcdef1234567890abcdef", config=cfg)
        masked = client.get_masked_key()
        assert masked.startswith("nvap****")
        assert masked.endswith("cdef")

    def test_live_probe_offline_simulation(self):
        cfg = NvidiaClientConfig(mock_mode=True)
        client = NvidiaCloudClient(api_key="nvapi-1234567890abcdef", config=cfg)
        report = client.probe_health()
        assert report["status"] == "ok"
        assert report["mode"] == "mock"
        assert report["authenticated"] is True

    def test_custom_mock_handler_exception(self):
        def failing_handler(method, url, kwargs):
            raise ValueError("Simulated handler crash")

        cfg = NvidiaClientConfig(mock_mode=True)
        client = NvidiaCloudClient(api_key="nvapi-1234567890abcdef", config=cfg, mock_handler=failing_handler)
        with pytest.raises(ValueError, match="Simulated handler crash"):
            client.request("GET", "/models")

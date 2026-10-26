# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Robust, isolated client for NVIDIA Cloud / NIM APIs.

Features:
- Safe, non-hardcoded API key loading from .env.local, .env, or environment
  variables (NVIDIA_API_KEY, NVIDIA_API, NGC_API_KEY).
- Automatic scrubbing and redacting of sensitive credentials in logs and string representations.
- Graceful handling of network timeouts, connection errors, HTTP errors (401, 403, 429),
  with exponential backoff and rate limit parsing.
- Mock / offline execution mode avoiding real credit consumption in CI/test.
- Health check / probe functions to verify authentication, latency, and connectivity safely.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from od_experiments.earth2_001.security import scrub_string, scrub_dict


class NvidiaAPIError(Exception):
    """Base exception for NVIDIA API operations."""

    def __init__(self, message: str, status_code: Optional[int] = None, details: Optional[Any] = None) -> None:
        safe_msg = scrub_string(message)
        super().__init__(safe_msg)
        self.status_code = status_code
        self.details = scrub_dict(details) if isinstance(details, dict) else scrub_string(str(details)) if details else None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(status_code={self.status_code}, message={str(self)!r})"


class NvidiaAuthenticationError(NvidiaAPIError):
    """Raised when authentication fails (e.g. invalid or expired API key, HTTP 401/403)."""
    pass


class NvidiaRateLimitError(NvidiaAPIError):
    """Raised when rate limits are exceeded (HTTP 429)."""

    def __init__(self, message: str, retry_after: Optional[float] = None, details: Optional[Any] = None) -> None:
        super().__init__(message=message, status_code=429, details=details)
        self.retry_after = retry_after


class NvidiaConnectionError(NvidiaAPIError):
    """Raised when network connectivity or DNS resolution fails."""
    pass


class NvidiaTimeoutError(NvidiaAPIError):
    """Raised when an API request times out."""
    pass


class NvidiaAPIKeyMissingError(NvidiaAuthenticationError):
    """Raised when no NVIDIA API key can be found in environment or local .env files."""
    pass


def find_and_load_api_key(
    env_file: Optional[Union[str, Path]] = None,
    env_keys: Optional[List[str]] = None,
    search_env_files: bool = True,
) -> Optional[str]:
    """Safely search and retrieve the NVIDIA API key without logging or hardcoding it.

    Search precedence:
    1. Direct environment variables (NVIDIA_API_KEY, NVIDIA_API, NGC_API_KEY).
    2. Explicitly passed env_file if provided.
    3. .env.local file in current working directory or repo root (if search_env_files is True).
    4. .env file in current working directory or repo root (if search_env_files is True).

    Returns the stripped API key if found, or None.
    """
    if env_keys is None:
        env_keys = ["NVIDIA_API_KEY", "NVIDIA_API", "NGC_API_KEY"]

    # 1. Check existing environment variables first
    for k in env_keys:
        val = os.environ.get(k)
        if val and val.strip():
            raw_key = val.strip().strip("'\"")
            if raw_key.startswith("KEY="):
                raw_key = raw_key[4:]
            return raw_key

    if not search_env_files and not env_file:
        return None

    # Candidate file locations
    candidate_paths: List[Path] = []
    if env_file:
        candidate_paths.append(Path(env_file))
    elif search_env_files:
        cwd = Path.cwd()
        candidate_paths.extend([
            cwd / ".env.local",
            cwd / ".env",
            cwd.parent / ".env.local",
            cwd.parent / ".env",
        ])

    for p in candidate_paths:
        if p.exists() and p.is_file():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            var_name, var_val = line.split("=", 1)
                            var_name = var_name.strip()
                            var_val = var_val.strip().strip("'\"")
                            if var_name in env_keys and var_val:
                                if var_val.startswith("KEY="):
                                    var_val = var_val[4:]
                                return var_val
            except Exception:
                continue

    return None


@dataclass
class NvidiaClientConfig:
    """Configuration settings for NvidiaCloudClient."""
    base_url: str = "https://integrate.api.nvidia.com/v1"
    nvcf_url: str = "https://api.nvcf.nvidia.com/v2"
    climate_url: str = "https://climate.api.nvidia.com/v1"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    backoff_factor: float = 0.5
    retry_status_codes: tuple[int, ...] = (429, 500, 502, 503, 504)
    mock_mode: bool = False
    rate_limit_pause: float = 1.0
    search_env_files: bool = True
    env_file: Optional[Union[str, Path]] = None


class NvidiaCloudClient:
    """Isolated, robust client for NVIDIA Cloud / NIM APIs.

    Guarantees:
    - Never leaks API keys in __repr__, __str__, or error logs.
    - Handles timeouts, connection errors, HTTP 401/403 auth failures, and HTTP 429 rate limits.
    - Supports mock execution mode for deterministic offline testing and CI workflows without credit usage.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[NvidiaClientConfig] = None,
        mock_handler: Optional[Callable[[str, str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        self.config = config or NvidiaClientConfig()
        self._mock_handler = mock_handler

        # Resolve API key
        if api_key:
            self._api_key: Optional[str] = api_key.strip()
        elif self.config.mock_mode:
            # In mock mode without explicit key, remain in pure mock
            self._api_key = None
        else:
            self._api_key = find_and_load_api_key(
                env_file=self.config.env_file,
                search_env_files=self.config.search_env_files,
            )

        self._session: Optional[requests.Session] = None
        if not self.config.mock_mode:
            self._init_session()

    def _init_session(self) -> None:
        """Configures requests session with robust retry policies."""
        self._session = requests.Session()
        retries = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.backoff_factor,
            status_forcelist=list(self.config.retry_status_codes),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    @property
    def is_authenticated(self) -> bool:
        """Returns True if client has an active API key or is running in mock mode."""
        if self.config.mock_mode:
            return True
        return bool(self._api_key)

    def get_masked_key(self) -> str:
        """Returns safe masked preview of key (e.g. nvapi-****1234), or placeholder."""
        if self.config.mock_mode and not self._api_key:
            return "[MOCK_MODE]"
        if not self._api_key:
            return "[NO_KEY_CONFIGURED]"
        if len(self._api_key) <= 8:
            return "********"
        return f"{self._api_key[:4]}****{self._api_key[-4:]}"

    def __repr__(self) -> str:
        return (
            f"NvidiaCloudClient(base_url={self.config.base_url!r}, "
            f"mock_mode={self.config.mock_mode}, "
            f"key_status={self.get_masked_key()!r})"
        )

    def _get_headers(self, custom_headers: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
        """Construct secure authorization and request headers."""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "Earth2Studio-OD-Client/1.0",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if custom_headers:
            headers.update(custom_headers)
        return headers

    def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Execute an API request with full error handling and retry mechanics."""
        method_upper = method.upper()
        clean_endpoint = endpoint.lstrip("/")

        # Handle Mock Mode
        if self.config.mock_mode:
            if self._mock_handler:
                return self._mock_handler(method_upper, clean_endpoint, {"params": params, "json": json_data})
            return self._default_mock_response(method_upper, clean_endpoint, json_data)

        # Enforce API Key presence before real external network request
        if not self._api_key:
            raise NvidiaAPIKeyMissingError(
                "NVIDIA API Key is missing. Please set NVIDIA_API_KEY in environment or .env.local, "
                "or initialize NvidiaCloudClient with mock_mode=True for offline execution."
            )

        target_base = base_url or self.config.base_url
        url = f"{target_base.rstrip('/')}/{clean_endpoint}" if clean_endpoint else target_base
        req_headers = self._get_headers(headers)
        req_timeout = timeout or self.config.timeout_seconds

        try:
            assert self._session is not None
            response = self._session.request(
                method=method_upper,
                url=url,
                params=params,
                json=json_data,
                headers=req_headers,
                timeout=req_timeout,
            )
        except requests.exceptions.Timeout as exc:
            raise NvidiaTimeoutError(f"Request to NVIDIA API timed out after {req_timeout}s: {exc}") from exc
        except (requests.exceptions.ConnectionError, requests.exceptions.SSLError) as exc:
            raise NvidiaConnectionError(f"Failed to connect to NVIDIA Cloud API endpoint: {exc}") from exc
        except requests.exceptions.RequestException as exc:
            raise NvidiaAPIError(f"Unexpected transport error during NVIDIA API request: {exc}") from exc

        # Handle status codes
        return self._handle_response(response)

    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Inspect and parse HTTP response status and payloads."""
        status = response.status_code

        # Attempt to parse response body as JSON
        body: Any = None
        try:
            body = response.json()
        except Exception:
            body = response.text

        if 200 <= status < 300:
            if isinstance(body, dict):
                return body
            return {"status": "success", "data": body, "status_code": status}

        # Authentication Errors
        if status in (401, 403):
            raise NvidiaAuthenticationError(
                f"NVIDIA API authentication failed (HTTP {status}). Verify API key permissions.",
                status_code=status,
                details=body,
            )

        # Rate Limiting
        if status == 429:
            retry_after_header = response.headers.get("Retry-After")
            retry_after: Optional[float] = None
            if retry_after_header:
                try:
                    retry_after = float(retry_after_header)
                except ValueError:
                    retry_after = self.config.rate_limit_pause
            raise NvidiaRateLimitError(
                f"NVIDIA API rate limit exceeded (HTTP 429).",
                retry_after=retry_after,
                details=body,
            )

        # Other API Errors
        raise NvidiaAPIError(
            f"NVIDIA API returned error status HTTP {status}.",
            status_code=status,
            details=body,
        )

    def _default_mock_response(self, method: str, endpoint: str, json_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Returns deterministic offline responses for testing without calling remote services."""
        if "health" in endpoint or "status" in endpoint or endpoint == "":
            return {
                "status": "healthy",
                "mock": True,
                "service": "nvidia-nim-cloud-mock",
                "timestamp": time.time(),
            }
        if "models" in endpoint:
            return {
                "data": [
                    {"id": "nvidia/fourcastnet", "status": "active", "type": "weather_forecast"},
                    {"id": "nvidia/corrdiff", "status": "active", "type": "downscaling"},
                    {"id": "nvidia/dlwp", "status": "active", "type": "global_medium_range"},
                ]
            }
        if "nvcf" in endpoint or "functions" in endpoint:
            return {
                "functions": [
                    {"id": "mock-fn-fourcastnet", "name": "fourcastnet", "status": "ACTIVE"},
                    {"id": "mock-fn-corrdiff", "name": "corrdiff", "status": "ACTIVE"},
                ]
            }
        if "infer" in endpoint or "predict" in endpoint or "fourcastnet" in endpoint:
            return {
                "id": "mock-inference-task-12345",
                "status": "completed",
                "result": {
                    "lead_time_hours": (json_data or {}).get("lead_time_hours", 6),
                    "variables": (json_data or {}).get("variables", ["t2m", "u10m", "v10m"]),
                    "output_shape": [1, 3, 721, 1440],
                    "simulated": True,
                },
            }

        return {
            "status": "success",
            "mock": True,
            "method": method,
            "endpoint": endpoint,
            "received_payload": json_data,
        }

    def list_models(self) -> List[Dict[str, Any]]:
        """List available models from the NVIDIA Build Cloud catalog."""
        res = self.request("GET", "models")
        if isinstance(res, dict) and "data" in res and isinstance(res["data"], list):
            return res["data"]
        return []

    def probe_health(self) -> Dict[str, Any]:
        """Probes API health status or mock capability without executing heavy workloads."""
        if self.config.mock_mode:
            return {"status": "ok", "mode": "mock", "authenticated": True, "masked_key": self.get_masked_key()}
        if not self._api_key:
            return {"status": "unauthenticated", "mode": "live", "authenticated": False, "masked_key": self.get_masked_key()}

        start = time.perf_counter()
        try:
            res = self.request("GET", "models")
            latency_ms = (time.perf_counter() - start) * 1000.0
            return {
                "status": "ok",
                "mode": "live",
                "authenticated": True,
                "latency_ms": round(latency_ms, 2),
                "masked_key": self.get_masked_key(),
                "models_count": len(res.get("data", [])) if isinstance(res, dict) else 0,
            }
        except NvidiaAuthenticationError as e:
            return {"status": "auth_error", "mode": "live", "authenticated": False, "error": str(e)}
        except NvidiaAPIError as e:
            return {"status": "error", "mode": "live", "authenticated": False, "error": str(e)}

    def close(self) -> None:
        """Close active session connections."""
        if self._session:
            self._session.close()

    def __enter__(self) -> NvidiaCloudClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

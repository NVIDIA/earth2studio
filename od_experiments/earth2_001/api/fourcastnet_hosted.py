"""Client implementation for NVIDIA Hosted FourCastNet API.

Endpoint: https://climate.api.nvidia.com/v1/nvidia/fourcastnet
Supports documented input parameters:
- input_id: 0..4
- variables: comma-separated list of w10m, t2m, msl, tcwv, z500
- simulation_length: 1..40
- ensemble_size: 1..4
- noise_amplitude: 0.0..0.1
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from od_experiments.earth2_001.security import SecretScrubber, get_nvidia_api_key

DEFAULT_ENDPOINT = "https://climate.api.nvidia.com/v1/nvidia/fourcastnet"
ALLOWED_VARIABLES: set[str] = {"w10m", "t2m", "msl", "tcwv", "z500"}


class NvidiaApiError(Exception):
    """Base exception for NVIDIA API errors."""

    def __init__(
        self, message: str, status_code: int | None = None, response_body: str = ""
    ):
        super().__init__(SecretScrubber.scrub_text(message))
        self.status_code = status_code
        self.response_body = SecretScrubber.scrub_text(response_body)


class NvidiaAuthenticationError(NvidiaApiError):
    """Raised when authentication fails (HTTP 401/403)."""


class NvidiaValidationError(NvidiaApiError):
    """Raised when request parameters fail client or server validation (HTTP 400)."""


class NvidiaRateLimitError(NvidiaApiError):
    """Raised when rate limits are exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str,
        status_code: int | None = 429,
        response_body: str = "",
        retry_after: int | None = None,
    ):
        super().__init__(message, status_code=status_code, response_body=response_body)
        self.retry_after = retry_after


class NvidiaGatewayTimeoutError(NvidiaApiError):
    """Raised when the server or gateway times out (HTTP 504)."""


@dataclass
class ForecastRequest:
    """Validated FourCastNet forecast request parameters."""

    input_id: int
    variables: str
    simulation_length: int = 1
    ensemble_size: int = 1
    noise_amplitude: float = 0.0
    accept_format: str = "application/json"

    def validate(self) -> None:
        """Validate parameter ranges and types."""
        if not isinstance(self.input_id, int) or isinstance(self.input_id, bool):
            raise NvidiaValidationError(
                f"input_id must be an integer, got {type(self.input_id).__name__}"
            )
        if self.input_id < 0 or self.input_id > 4:
            raise NvidiaValidationError(
                f"input_id must be in range 0..4, got {self.input_id}"
            )

        if not isinstance(self.variables, str) or not self.variables.strip():
            raise NvidiaValidationError(
                "variables must be a non-empty comma-separated string"
            )
        var_list = [v.strip().lower() for v in self.variables.split(",") if v.strip()]
        if not var_list:
            raise NvidiaValidationError("No valid variables specified")
        for v in var_list:
            if v not in ALLOWED_VARIABLES:
                raise NvidiaValidationError(
                    f"Invalid variable '{v}'. Allowed: {sorted(ALLOWED_VARIABLES)}"
                )

        if not (1 <= self.simulation_length <= 40):
            raise NvidiaValidationError(
                f"simulation_length must be between 1 and 40, got {self.simulation_length}"
            )

        if not (1 <= self.ensemble_size <= 4):
            raise NvidiaValidationError(
                f"ensemble_size must be between 1 and 4, got {self.ensemble_size}"
            )

        if not (0.0 <= self.noise_amplitude <= 0.1):
            raise NvidiaValidationError(
                f"noise_amplitude must be between 0.0 and 0.1, got {self.noise_amplitude}"
            )

        if self.accept_format not in {"application/json", "application/x-tar", "*/*"}:
            raise NvidiaValidationError(
                f"Unsupported accept_format '{self.accept_format}'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_id": self.input_id,
            "variables": self.variables,
            "simulation_length": self.simulation_length,
            "ensemble_size": self.ensemble_size,
            "noise_amplitude": self.noise_amplitude,
        }


@dataclass
class ForecastResponse:
    """Encapsulates raw and parsed response data from NVIDIA FourCastNet service."""

    status_code: int
    req_id: str | None
    nvcf_status: str | None
    content_type: str
    raw_data: bytes
    json_payload: dict[str, Any] | None = None
    duration_seconds: float = 0.0
    retry_count: int = 0
    endpoint: str = DEFAULT_ENDPOINT


class FourCastNetHostedClient:
    """Reusable API client for NVIDIA-hosted FourCastNet model."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = 90.0,
        max_retries: int = 2,
        backoff_factor: float = 2.0,
    ):
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._api_key = api_key or get_nvidia_api_key()

    def get_masked_key(self) -> str:
        if not self._api_key:
            return "<MISSING>"
        if len(self._api_key) <= 8:
            return "***"
        return f"{self._api_key[:6]}...{self._api_key[-4:]}"

    def infer(
        self, req: ForecastRequest, bypass_auth_for_testing: bool = False
    ) -> ForecastResponse:
        """Submit a single inference request to the hosted service."""
        req.validate()

        if not self._api_key and not bypass_auth_for_testing:
            raise NvidiaAuthenticationError(
                "NVIDIA_API_KEY is not configured in environment or .env.local"
            )

        headers = {
            "Content-Type": "application/json",
            "Accept": req.accept_format,
            "User-Agent": "OriginDatum-Earth2/1.0",
        }
        if not bypass_auth_for_testing and self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload_bytes = json.dumps(req.to_dict()).encode("utf-8")

        attempts = 0
        current_delay = 1.0

        while True:
            attempts += 1
            t_start = time.perf_counter()
            http_req = urllib.request.Request(
                self.endpoint, data=payload_bytes, headers=headers
            )

            try:
                with urllib.request.urlopen(http_req, timeout=self.timeout) as resp:
                    duration = time.perf_counter() - t_start
                    status = resp.status
                    headers_dict = dict(resp.headers)
                    body_bytes = resp.read()

                    req_id = headers_dict.get("Nvcf-Reqid") or headers_dict.get(
                        "nvcf-reqid"
                    )
                    nvcf_status = headers_dict.get("Nvcf-Status") or headers_dict.get(
                        "nvcf-status"
                    )
                    ct = (
                        headers_dict.get("Content-Type")
                        or headers_dict.get("content-type")
                        or ""
                    )

                    json_data = None
                    if "application/json" in ct:
                        try:
                            json_data = json.loads(body_bytes.decode("utf-8"))
                        except Exception:
                            json_data = None

                    return ForecastResponse(
                        status_code=status,
                        req_id=req_id,
                        nvcf_status=nvcf_status,
                        content_type=ct,
                        raw_data=body_bytes,
                        json_payload=json_data,
                        duration_seconds=duration,
                        retry_count=attempts - 1,
                        endpoint=self.endpoint,
                    )

            except urllib.error.HTTPError as e:
                duration = time.perf_counter() - t_start
                body = e.read().decode("utf-8", errors="replace")

                if e.code in (401, 403):
                    raise NvidiaAuthenticationError(
                        f"HTTP {e.code}: {body}", status_code=e.code, response_body=body
                    )
                elif e.code == 400:
                    raise NvidiaValidationError(
                        f"HTTP 400: {body}", status_code=400, response_body=body
                    )
                elif e.code == 429:
                    retry_header = e.headers.get("Retry-After")
                    retry_seconds = (
                        int(retry_header)
                        if (retry_header and retry_header.isdigit())
                        else int(current_delay)
                    )
                    if attempts <= self.max_retries:
                        time.sleep(retry_seconds)
                        current_delay *= self.backoff_factor
                        continue
                    raise NvidiaRateLimitError(
                        f"HTTP 429 Rate limit exceeded: {body}",
                        status_code=429,
                        response_body=body,
                        retry_after=retry_seconds,
                    )
                elif e.code == 504:
                    if attempts <= self.max_retries:
                        time.sleep(current_delay)
                        current_delay *= self.backoff_factor
                        continue
                    raise NvidiaGatewayTimeoutError(
                        f"HTTP 504 Gateway Timeout: {body}",
                        status_code=504,
                        response_body=body,
                    )
                else:
                    raise NvidiaApiError(
                        f"HTTP {e.code}: {body}", status_code=e.code, response_body=body
                    )

            except urllib.error.URLError as e:
                if attempts <= self.max_retries:
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor
                    continue
                raise NvidiaApiError(f"Connection failure: {e.reason}")

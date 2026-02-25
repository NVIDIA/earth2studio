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
Main client class for Earth2Studio API.
"""

import io
import json
import time
from typing import cast
from urllib.parse import urljoin

import requests  # type: ignore[import-untyped]
from requests.adapters import HTTPAdapter  # type: ignore[import-untyped]
from requests.packages.urllib3.util.retry import Retry  # type: ignore[import-untyped]

from api_client.exceptions import (
    BadRequestError,
    Earth2StudioAPIError,
    InferenceRequestNotFoundError,
    InternalServerError,
    RequestTimeoutError,
)
from api_client.exceptions import (
    ConnectionError as ClientConnectionError,
)
from api_client.models import (
    HealthStatus,
    InferenceRequest,
    InferenceRequestResponse,
    InferenceRequestResults,
    InferenceRequestStatus,
    RequestStatus,
    StorageType,
)


class Earth2StudioClient:
    """
    Python client for Earth2Studio REST API.

    Args:
        base_url: Base URL of the Earth2Studio API server (default: "http://localhost:8000")
        workflow_name: Name of the workflow to use (default: "deterministic_earth2_workflow")
        timeout: Request timeout in seconds (default: 30.0)
        max_retries: Maximum number of retries for failed requests (default: 3)
        retry_backoff_factor: Backoff factor for retries (default: 0.3)
        token: Optional authentication token (sent as Bearer token in Authorization header)

    Example:
        >>> client = Earth2StudioClient(base_url="http://localhost:8000")
        >>> health = client.health_check()
        >>> print(health.status)
        healthy

        # With authentication
        >>> client = Earth2StudioClient(base_url="http://localhost:8000", token="your-token")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        workflow_name: str = "deterministic_earth2_workflow",
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.3,
        token: str | None = None,
    ):
        """Initialize the client with base URL, workflow name, and optional settings."""
        self.base_url = base_url.rstrip("/")
        self.workflow_name = workflow_name
        self.timeout = timeout
        self.token = token

        # Create session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set default headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # Add authorization header if token is provided
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.session.headers.update(headers)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: dict | None = None,
        params: dict | None = None,
        return_response: bool = False,
        stream: bool = False,
        timeout: float | None = None,
    ) -> dict | requests.Response:
        """Make an HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json_data: Optional JSON data for request body
            params: Optional query parameters
            return_response: If True, return raw Response object instead of parsed JSON
            stream: If True, enable streaming for large responses
            timeout: Optional timeout in seconds for the download request. If None, uses client default.

        Returns:
            dict: Parsed JSON response (if return_response=False)
            requests.Response: Raw response object (if return_response=True)
        """
        url = urljoin(self.base_url + "/", endpoint.lstrip("/"))
        timeout = timeout or self.timeout
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=json_data,
                params=params,
                timeout=timeout,
                stream=stream,
            )
        except requests.exceptions.Timeout:
            raise RequestTimeoutError(
                f"Request to {url} timed out after {timeout} seconds"
            )
        except requests.exceptions.ConnectionError as e:
            raise ClientConnectionError(f"Failed to connect to {url}: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise Earth2StudioAPIError(f"Request failed: {str(e)}")

        # Handle HTTP error status codes
        if response.status_code == 400:
            error_data = self._parse_error_response(response)
            raise BadRequestError(
                error_data.get("error", "Bad request"),
                details=error_data.get("details"),
            )
        elif response.status_code == 404:
            error_data = self._parse_error_response(response)
            raise InferenceRequestNotFoundError(
                error_data.get("error", "Not found"), details=error_data.get("details")
            )
        elif response.status_code == 500:
            error_data = self._parse_error_response(response)
            raise InternalServerError(
                error_data.get("error", "Internal server error"),
                details=error_data.get("details"),
            )
        elif not response.ok:
            error_data = self._parse_error_response(response)
            raise Earth2StudioAPIError(
                error_data.get("error", f"HTTP {response.status_code} error"),
                status_code=response.status_code,
                details=error_data.get("details"),
            )

        # Return raw response if requested (for binary content like zip files)
        if return_response:
            return response

        # Otherwise parse as JSON (default behavior)
        try:
            return response.json()
        except json.JSONDecodeError:
            raise Earth2StudioAPIError("Invalid JSON response from server")

    def _parse_error_response(self, response: requests.Response) -> dict:
        """Parse error response JSON, fallback to empty dict."""
        try:
            return response.json()
        except json.JSONDecodeError:
            return {"error": response.text or f"HTTP {response.status_code} error"}

    def health_check(self) -> HealthStatus:
        """
        Check if the API is running and healthy.

        Returns:
            HealthStatus object with status and timestamp

        Raises:
            Earth2StudioAPIError: If the API is not accessible
        """
        response_data = self._make_request("GET", "/health")
        return HealthStatus.from_dict(response_data)

    def submit_inference_request(
        self, request: InferenceRequest
    ) -> InferenceRequestResponse:
        """
        Submit an inference request to be processed.

        Args:
            request: InferenceRequest object with workflow configuration

        Returns:
            InferenceRequestResponse with request ID and initial status

        Raises:
            BadRequestError: If the request parameters are invalid
            Earth2StudioAPIError: For other API errors
        """
        response_data = self._make_request(
            "POST", f"/v1/infer/{self.workflow_name}", json_data=request.to_dict()
        )
        return InferenceRequestResponse.from_dict(response_data)

    def get_request_status(self, request_id: str) -> InferenceRequestStatus:
        """
        Get the current status of an inference request.

        Args:
            request_id: Unique identifier for the inference request

        Returns:
            InferenceRequestStatus with current progress and status

        Raises:
            InferenceRequestNotFoundError: If the request ID is not found
            Earth2StudioAPIError: For other API errors
        """
        response_data = self._make_request(
            "GET", f"/v1/infer/{self.workflow_name}/{request_id}/status"
        )
        return InferenceRequestStatus.from_dict(response_data)

    def get_request_results(
        self,
        request_id: str,
        timeout: float | None = None,
    ) -> InferenceRequestResults:
        """
        Get the results of a completed inference request by downloading the results zip file.

        Args:
            request_id: Unique identifier for the inference request
            timeout: Optional timeout in seconds for the download request. If None, uses client default.

        Returns:
            InferenceRequestResults with output file paths.
            Use its result_paths() method to find result objects.

        Raises:
            InferenceRequestNotFoundError: If the request ID is not found
            Earth2StudioAPIError: For other API errors

        Note:
            This method will return a 202 status if the request is still processing.
            Use get_request_status() to check if the request is completed first.
        """
        # Use _make_request with return_response=True to get raw response for zip file
        response = cast(
            requests.Response,
            self._make_request(
                method="GET",
                endpoint=f"/v1/infer/{self.workflow_name}/{request_id}/results",
                return_response=True,
                timeout=timeout,
            ),
        )

        # Handle 202 status specially (request still processing)
        if response.status_code == 202:
            try:
                error_data = response.json()
                raise Earth2StudioAPIError(
                    error_data.get("message", "Request is still processing"),
                    status_code=202,
                    details=error_data.get("status"),
                )
            except json.JSONDecodeError:
                raise Earth2StudioAPIError(
                    "Request is still processing", status_code=202
                )

        response_data = response.json()
        return InferenceRequestResults.from_dict(response_data)

    def wait_for_completion(
        self,
        request_id: str,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> InferenceRequestResults:
        """
        Wait for an inference request to complete and return results.

        Args:
            request_id: Unique identifier for the inference request
            poll_interval: How often to check status (in seconds)
            timeout: Maximum time to wait (in seconds). None for no timeout.

        Returns:
            InferenceRequestResults when the request completes successfully.

        Raises:
            RequestTimeoutError: If timeout is reached
            Earth2StudioAPIError: If the request fails or other API errors occur
        """
        start_time = time.time()

        while True:
            status = self.get_request_status(request_id)

            if status.status == RequestStatus.COMPLETED:
                return self.get_request_results(request_id, timeout)
            elif status.status == RequestStatus.FAILED:
                raise Earth2StudioAPIError(
                    f"Inference request {request_id} failed: {status.error_message}"
                )
            elif status.status == RequestStatus.CANCELLED:
                raise Earth2StudioAPIError("Inference request was cancelled")

            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                raise RequestTimeoutError(
                    f"Request {request_id} did not complete within {timeout} seconds"
                )

            time.sleep(poll_interval)

    def run_inference_sync(
        self,
        request: InferenceRequest,
        poll_interval: float = 5.0,
        timeout: float | None = None,
    ) -> InferenceRequestResults:
        """
        Submit an inference request and wait for it to complete.

        This is a convenience method that combines submit_inference_request()
        and wait_for_completion().

        Args:
            request: InferenceRequest object with workflow configuration
            poll_interval: How often to check status (in seconds)
            timeout: Maximum time to wait (in seconds). None for no timeout.

        Returns:
            InferenceRequestResults when the request completes successfully.
        """
        response = self.submit_inference_request(request)
        return self.wait_for_completion(response.execution_id, poll_interval, timeout)

    def result_root_path(self, result: InferenceRequestResults) -> str:
        """
        Get the root URL path for accessing result files.

        Args:
            result: InferenceRequestResults object returned by the request

        Returns:
            str: Root path URL for the result files (e.g., "/v1/infer/workflow_name/exec_1766159252_5f779460/results/")
        """
        return f"/v1/infer/{self.workflow_name}/{result.request_id}/results/"

    def download_result(
        self,
        result: InferenceRequestResults,
        path: str,
        timeout: float | None = None,
    ) -> io.BytesIO:
        """
        Download a result file from the server or S3.

        Args:
            result: InferenceRequestResults object containing request metadata
            path: Relative path to the result file (as returned in InferenceRequestResults)
            timeout: Optional timeout in seconds for the download request. If None, uses client default.

        Returns:
            io.BytesIO: In-memory bytes buffer containing the downloaded file content

        Raises:
            Earth2StudioAPIError: If the download fails or other API errors occur
        """
        if result.storage_type == StorageType.S3:
            if not result.signed_url:
                raise Earth2StudioAPIError("S3 storage type requires a signed URL")

            from api_client.object_storage import create_cloudfront_mapper

            mapper = create_cloudfront_mapper(result.signed_url, zarr_path="")
            # strip out the prefix execution_id from the path
            path = "/".join(path.split("/")[1:])
            content = mapper.fs.cat_file(path)
            return io.BytesIO(content)

        # Default: download from server
        response = cast(
            requests.Response,
            self._make_request(
                method="GET",
                endpoint=f"{self.result_root_path(result)}{path}",
                return_response=True,
                stream=True,
                timeout=timeout,
            ),
        )
        # TODO: add support for downloading to tmp file to avoid OOM
        return io.BytesIO(response.content)

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self.session.close()  # type: ignore[no-untyped-call]

    def __enter__(self) -> "Earth2StudioClient":
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit."""
        self.close()

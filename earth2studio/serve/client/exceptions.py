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


class Earth2StudioAPIError(Exception):
    """Base exception for Earth2Studio API errors.

    Parameters
    ----------
    message : str
        Human-readable error message.
    status_code : int, optional
        HTTP status code if the error came from an API response.
    details : str, optional
        Additional error details.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        details: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details

    def __str__(self) -> str:
        """
        Return a human-readable error string.

        Returns
        -------
        str
            Error string including message, status code (if set), and details (if set).
        """
        error_msg = f"Earth2Studio API Error: {self.message}"
        if self.status_code:
            error_msg += f" (HTTP {self.status_code})"
        if self.details:
            error_msg += f" - {self.details}"
        return error_msg


class BadRequestError(Earth2StudioAPIError):
    """Exception raised for HTTP 400 Bad Request errors."""

    def __init__(self, message: str, details: str | None = None):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Human-readable error message.
        details : str, optional
            Additional error details.
        """
        super().__init__(message, status_code=400, details=details)


class InferenceRequestNotFoundError(Earth2StudioAPIError):
    """Exception raised for HTTP 404 Not Found errors when inference request is not found."""

    def __init__(self, message: str, details: str | None = None):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Human-readable error message.
        details : str, optional
            Additional error details.
        """
        super().__init__(message, status_code=404, details=details)


class InternalServerError(Earth2StudioAPIError):
    """Exception raised for HTTP 500 Internal Server errors."""

    def __init__(self, message: str, details: str | None = None):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str
            Human-readable error message.
        details : str, optional
            Additional error details.
        """
        super().__init__(message, status_code=500, details=details)


class RequestTimeoutError(Earth2StudioAPIError):
    """Exception raised when a request times out."""

    def __init__(self, message: str = "Request timed out"):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str, optional
            Human-readable error message. Default is "Request timed out".
        """
        super().__init__(message)


class APIConnectionError(Earth2StudioAPIError):
    """Exception raised when connection to the API fails."""

    def __init__(self, message: str = "Failed to connect to Earth2Studio API"):
        """
        Initialize the exception.

        Parameters
        ----------
        message : str, optional
            Human-readable error message. Default is "Failed to connect to Earth2Studio API".
        """
        super().__init__(message)

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
FSSpec mapper for signed URLs with Zarr.

The purpose of this module is to create an fsspec mapper for an AWS wildcard signed URL.
This mapper allows access to the files in the directory encapsulated by the wildcard signed URL.

"""

import os
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import fsspec  # type: ignore[import-untyped]

from api_client.models import InferenceRequestResults, StorageType


class SignedURLFileSystem(fsspec.AbstractFileSystem):
    """Wrapper that appends signed params to all requests and converts 403 to FileNotFoundError."""

    def __init__(
        self, base_fs: Any, query_params: dict[str, str], base_url: str
    ) -> None:
        """Wrap a base filesystem and URL with query params for signed requests."""
        super().__init__()
        self._fs = base_fs
        self._query_params = query_params
        self._base_url = base_url
        self._query_string = urlencode(query_params, safe="~")

    def _make_signed_path(self, path: str) -> str:
        """Return the path with signed query parameters appended."""
        if path.startswith("http"):
            full_url = path
        else:
            clean_path = path.lstrip("/")
            full_url = (
                f"{self._base_url}/{clean_path}" if clean_path else self._base_url
            )
        separator = "&" if "?" in full_url else "?"
        return f"{full_url}{separator}{self._query_string}"

    def _handle_403(self, e: BaseException, path: str) -> None:
        """Convert 403 errors to FileNotFoundError."""
        error_str = str(e).lower()
        if "403" in str(e) or "forbidden" in error_str:
            raise FileNotFoundError(f"File not found: {path}") from None
        raise

    def _open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        """Open a path with signed URL; 403 is converted to FileNotFoundError."""
        try:
            return self._fs._open(self._make_signed_path(path), mode=mode, **kwargs)
        except Exception as e:
            self._handle_403(e, path)

    def cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> Any:
        """Read file contents with signed URL; 403 is converted to FileNotFoundError."""
        try:
            return self._fs.cat_file(
                self._make_signed_path(path), start=start, end=end, **kwargs
            )
        except Exception as e:
            self._handle_403(e, path)

    def _cat_file(
        self, path: str, start: int | None = None, end: int | None = None, **kwargs: Any
    ) -> Any:
        """Async version used by zarr."""
        return self.cat_file(path, start=start, end=end, **kwargs)

    def info(self, path: str, **kwargs: Any) -> Any:
        """Return metadata for path with signed URL; 403 becomes FileNotFoundError."""
        try:
            return self._fs.info(self._make_signed_path(path), **kwargs)
        except Exception as e:
            self._handle_403(e, path)

    def exists(self, path: str, **kwargs: Any) -> bool:
        """Return True if path exists; 403 is treated as not found."""
        try:
            return self._fs.exists(self._make_signed_path(path), **kwargs)
        except Exception as e:
            try:
                self._handle_403(e, path)
            except FileNotFoundError:
                return False
        return False


def create_cloudfront_mapper(signed_url: str, zarr_path: str = "") -> Any:
    """
    Create an fsspec mapper for a CloudFront signed Zarr URL.

    Parameters
    ----------
    signed_url : str
        CloudFront signed URL with Policy, Signature, and Key-Pair-Id query params.

    Returns
    -------
    mapper : fsspec.mapping.FSMap
        A mapper suitable for use with xarray.open_zarr()
    """
    # Parse the URL
    parsed = urlparse(signed_url)

    # Extract query parameters
    query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

    # Get base path (remove trailing /* if present)
    base_path = parsed.path.rstrip("/*").rstrip("*")

    # Reconstruct base URL
    if zarr_path != "":
        base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}/{zarr_path}"
    else:
        base_url = f"{parsed.scheme}://{parsed.netloc}{base_path}"

    # Create HTTP filesystem
    fs = fsspec.filesystem("https")

    signed_fs = SignedURLFileSystem(fs, query_params, base_url)
    mapper = fsspec.mapping.FSMap(root="", fs=signed_fs, check=False, create=False)

    return mapper


def get_mapper(
    request_result: InferenceRequestResults, zarr_path: str = ""
) -> Any | None:
    """Return an fsspec mapper for the result, or None for SERVER storage."""
    if request_result.storage_type == StorageType.S3:
        if request_result.signed_url is None:
            raise ValueError("S3 storage type requires a signed URL")
        return create_cloudfront_mapper(request_result.signed_url, zarr_path)
    elif request_result.storage_type == StorageType.SERVER:
        return None
    else:
        raise ValueError(f"Unsupported storage type: {request_result.storage_type}")


# Usage with xarray
if __name__ == "__main__":
    import xarray as xr

    url = os.getenv("SIGNED_URL")
    if not url:
        raise ValueError("SIGNED_URL environment variable is not set")
    mapper = create_cloudfront_mapper(url, zarr_path="results.zarr")

    # Try consolidated first, fall back to non-consolidated
    try:
        ds = xr.open_zarr(mapper, consolidated=True)
    except (FileNotFoundError, KeyError):
        print("Consolidated metadata not found, trying non-consolidated...")
        ds = xr.open_zarr(mapper, consolidated=False)

    print(ds)

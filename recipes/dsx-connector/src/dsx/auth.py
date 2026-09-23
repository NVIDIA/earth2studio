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

"""Access-token providers for the DSX MQTT connection.

For OAuth2 authentication, the broker expects username ``oauthtoken`` and a bearer access token as
the password. This module supplies an existing token; obtaining and refreshing it remains the
deployment operator's responsibility.

The bus requests a token whenever it connects or reconnects. A mounted token file can therefore be
rotated without restarting the connector. Environment-based tokens require a process restart.

This module uses only the Python standard library.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class TokenProvider(Protocol):
    """Supplies the current bearer token for the DSX bus."""

    def token(self) -> str:
        """Return the bearer token to use for the next connection."""
        ...


class StaticTokenProvider:
    """A fixed, pre-issued bearer token with no refresh.

    Parameters
    ----------
    token : str
        The bearer token.
    """

    def __init__(self, token: str) -> None:
        token = token.strip()
        if not token:
            raise ValueError("empty bearer token")
        self._token = token

    def token(self) -> str:
        """Return the fixed token."""
        return self._token


class EnvTokenProvider:
    """Read the bearer token from an environment variable.

    Environment variables supplied at process startup cannot be rotated externally while the
    process is running. Restart the connector to use a changed value, or use ``FileTokenProvider``
    for live rotation.

    Parameters
    ----------
    env_var : str, optional
        Environment variable holding the token, by default "DSX_OAUTH_TOKEN".
    """

    def __init__(self, env_var: str = "DSX_OAUTH_TOKEN") -> None:
        self._env_var = env_var

    def token(self) -> str:
        """Return the token from the environment; raise if unset or empty.

        Returns
        -------
        str
            The token value.

        Raises
        ------
        ValueError
            If the environment variable is unset, empty, or contains only whitespace.
        """
        value = os.environ.get(self._env_var)
        if value is None or not value.strip():
            raise ValueError(f"{self._env_var} is unset or empty")
        return value.strip()


class FileTokenProvider:
    """Read the bearer token from a file on every call.

    The connector does not read the token only at startup or keep it in memory. It opens the file
    for every broker connection attempt, including reconnects. A running connector therefore sees
    updated file contents on its next connection attempt without needing a restart.

    Parameters
    ----------
    path : str
        Path to the file containing the bearer token (surrounding whitespace stripped).
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def token(self) -> str:
        """Return the current token read from the file; raise if missing or empty.

        Returns
        -------
        str
            The token value.

        Raises
        ------
        ValueError
            If the file cannot be read, is empty, or contains only whitespace.
        """
        try:
            value = Path(self._path).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(f"cannot read token file {self._path}: {exc}") from exc
        if not value:
            raise ValueError(f"token file {self._path} is empty")
        return value

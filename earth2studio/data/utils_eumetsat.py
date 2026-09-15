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

from __future__ import annotations

import os
import pathlib

from loguru import logger


def eumetsat_credentials() -> tuple[str, str]:
    """Resolve EUMETSAT Data Store API credentials.

    Credentials are read from the ``EUMETSAT_CONSUMER_KEY`` and
    ``EUMETSAT_CONSUMER_SECRET`` environment variables. When either is unset,
    the ``eumdac`` CLI credentials file is used as a fallback (by default
    ``~/.eumdac/credentials``, overridable with ``EUMDAC_CONFIG_DIR``).

    Register at https://eoportal.eumetsat.int/ to obtain credentials.

    Returns
    -------
    tuple[str, str]
        ``(consumer_key, consumer_secret)``. Both are empty strings when no
        credentials could be resolved; a warning is logged in that case and
        any subsequent data fetch will fail.

    Note
    ----
    The ``eumdac`` credentials file layout is documented at
    https://gitlab.eumetsat.int/eumetlab/data-services/eumdac
    """
    consumer_key = os.environ.get("EUMETSAT_CONSUMER_KEY", "")
    consumer_secret = os.environ.get("EUMETSAT_CONSUMER_SECRET", "")
    if consumer_key and consumer_secret:
        return consumer_key, consumer_secret

    credentials_file = (
        pathlib.Path(os.getenv("EUMDAC_CONFIG_DIR", (pathlib.Path.home() / ".eumdac")))
        / "credentials"
    )
    try:
        with open(credentials_file) as f:
            credentials = f.read().strip()
        key, secret = credentials.split(",", 1)
        return key.strip(), secret.strip()
    except (OSError, ValueError):
        logger.warning(
            "EUMETSAT_CONSUMER_KEY and/or EUMETSAT_CONSUMER_SECRET not set. "
            "Data fetching will fail."
        )
        return consumer_key, consumer_secret

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

"""HTTP client for Azure Planetary Computer Pro GeoCatalog STAC ingestion."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping
from datetime import datetime, timedelta
from time import perf_counter
from typing import Any, Final
from uuid import uuid4

import requests

logger = logging.getLogger("azure_planetary_computer")
logger.setLevel(logging.INFO)

# Workflows with templates and GeoCatalog behavior in this package (single source of truth).
PLANETARY_COMPUTER_CLIENT_WORKFLOWS: Final[frozenset[str]] = frozenset(
    {
        "foundry_fcn3_workflow",
        "foundry_fcn3_stormscope_goes_workflow",
    }
)


class PlanetaryComputerClient:
    """Client for STAC collection/feature creation against Planetary Computer Pro.

    Templates and render options are loaded from JSON files alongside this module.

    Parameters
    ----------
    workflow_name : {'foundry_fcn3_workflow', 'foundry_fcn3_stormscope_goes_workflow'}
        Earth2Studio workflow identifier selecting templates and step sizing.
    """

    APPLICATION_URL = "https://geocatalog.spatio.azure.com/"
    REQUESTS_TIMEOUT = 30
    CREATION_TIMEOUT = 300

    def __init__(self, workflow_name: str) -> None:
        if workflow_name not in PLANETARY_COMPUTER_CLIENT_WORKFLOWS:
            raise ValueError(
                f"Unsupported workflow_name for PlanetaryComputerClient: {workflow_name!r}. "
                f"Supported: {sorted(PLANETARY_COMPUTER_CLIENT_WORKFLOWS)}"
            )
        self.workflow_name = workflow_name
        self.headers: dict[str, str] | None = None

    def update_headers(self) -> None:
        """Refresh the Authorization header using a new Azure credential token."""
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as e:
            raise ImportError(
                "PlanetaryComputerClient requires 'azure-identity'. "
                "Install with the serve extra or pip install azure-identity."
            ) from e
        credential = DefaultAzureCredential()
        token = credential.get_token(self.APPLICATION_URL)
        self.headers = {"Authorization": f"Bearer {token.token}"}

    def _require_headers(self) -> dict[str, str]:
        if self.headers is None:
            raise RuntimeError(
                "Authorization headers are not set; call update_headers() first."
            )
        return self.headers

    def _get(self, url: str) -> requests.Response:
        headers = self._require_headers()
        return requests.get(
            url,
            headers=headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _post(self, url: str, body: dict | None = None) -> requests.Response:
        headers = self._require_headers()
        return requests.post(
            url,
            json=body,
            headers=headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _put(self, url: str, body: dict | None = None) -> requests.Response:
        headers = self._require_headers()
        return requests.put(
            url,
            json=body,
            headers=headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _create_element(self, url: str, stac_config: dict) -> None:
        """Create a STAC collection or feature and wait for it to finish."""
        response = self._post(
            url,
            body=stac_config,
        )
        location = response.headers["location"]

        logger.info("Creating '%s'...", stac_config["id"])
        start = perf_counter()
        while True:
            if (perf_counter() - start) > self.CREATION_TIMEOUT:
                logger.error("Creation of '%s' timed out", stac_config["id"])
                return

            response = self._get(location)
            status = response.json()["status"]
            logger.info(status)
            if status not in {"Pending", "Running"}:
                break
            time.sleep(5)

        if status == "Succeeded":
            logger.info("Successfully created '%s'", stac_config["id"])
        else:
            logger.error("Failed to create '%s': %s", stac_config["id"], response.text)

    def _get_collection_json(self, collection_id: str | None) -> dict:
        """Load the STAC collection template and set the collection ID."""
        template_fns = {
            "foundry_fcn3_workflow": "template-collection-fcn3.json",
            "foundry_fcn3_stormscope_goes_workflow": "template-collection-fcn3-stormscope-goes.json",
        }
        template_fn = os.path.join(
            os.path.dirname(__file__), template_fns[self.workflow_name]
        )
        with open(template_fn) as f:
            stac_config = json.load(f)

        if collection_id is None:
            stac_config["id"] = stac_config["id"].format(uuid=uuid4())
        else:
            stac_config["id"] = collection_id
        return stac_config

    def _get_feature_json(
        self,
        start_time: datetime,
        end_time: datetime,
        blob_url: str,
    ) -> dict:
        """Load the STAC feature template and set the workflow parameters."""
        template_fns = {
            "foundry_fcn3_workflow": "template-feature-fcn3.json",
            "foundry_fcn3_stormscope_goes_workflow": "template-feature-fcn3-stormscope-goes.json",
        }
        template_fn = os.path.join(
            os.path.dirname(__file__), template_fns[self.workflow_name]
        )
        with open(template_fn) as f:
            stac_config = json.load(f)

        iso_start = start_time.isoformat()
        iso_end = end_time.isoformat()

        stac_config["id"] = stac_config["id"].format(
            start_time=iso_start[:13], uuid=uuid4()
        )
        stac_config["properties"]["datetime"] = iso_start
        stac_config["properties"]["start_datetime"] = iso_start
        stac_config["properties"]["end_datetime"] = iso_end
        stac_config["assets"]["data"]["href"] = blob_url
        stac_config["assets"]["data"]["description"] = stac_config["assets"]["data"][
            "description"
        ].format(start_time=iso_start, end_time=iso_end)
        return stac_config

    def _update_tile_settings(self, geocatalog_url: str, collection_id: str) -> None:
        """Update 'minZoom' of the tile settings so user can zoom out."""
        tile_settings = {
            "minZoom": 0,
            "maxItemsPerTile": 35,
        }
        response = self._put(
            f"{geocatalog_url}/stac/collections/{collection_id}/configurations/tile-settings",
            body=tile_settings,
        )
        status = response.status_code
        if status not in {200, 201}:
            logger.error(
                "Could not update tile settings: Error %s - %s",
                status,
                response.text,
            )

    def _update_render_options(
        self,
        geocatalog_url: str,
        collection_id: str,
    ) -> None:
        """Add example render options for a new collection."""
        if self.workflow_name == "foundry_fcn3_workflow":
            render_params = [
                {
                    "id": "t2m",
                    "scale": [263, 313],
                    "cmap": "balance",
                },
                {
                    "id": "t850",
                    "scale": [263, 313],
                    "cmap": "balance",
                },
                {
                    "id": "u10m",
                    "scale": [-20, 20],
                    "cmap": "prgn",
                },
                {
                    "id": "v10m",
                    "scale": [-20, 20],
                    "cmap": "prgn",
                },
                {
                    "id": "z500",
                    "scale": [45000, 60000],
                    "cmap": "viridis",
                },
            ]
        elif self.workflow_name == "foundry_fcn3_stormscope_goes_workflow":
            render_params = [
                {
                    "id": f"abi{aid:02}c",
                    "scale": [0, 1],
                    "cmap": "plasma",
                }
                for aid in [1, 2, 3, 7, 8, 9, 10, 13]
            ]
        else:
            render_params = []

        for params in render_params:
            render_option = {
                "id": f"auto-{params['id']}",
                "name": params["id"],
                "type": "raster-tile",
                "options": (
                    f"assets=data&subdataset_name={params['id']}"
                    "&sel=time=2100-01-01&sel=ensemble=0&sel_method=nearest"
                    f"&rescale={params['scale'][0]},{params['scale'][1]}"
                    f"&colormap_name={params['cmap']}"
                ),
                "minZoom": 0,
            }
            response = self._post(
                f"{geocatalog_url}/stac/collections/{collection_id}/configurations/render-options",
                body=render_option,
            )
            status = response.status_code
            if status not in {200, 201}:
                logger.error(
                    "Could not update render options: Error %s - %s",
                    status,
                    response.text,
                )

    def _create_collection(
        self,
        geocatalog_url: str,
        collection_id: str | None,
    ) -> str:
        """Create a new STAC collection."""
        stac_config = self._get_collection_json(collection_id)
        self._create_element(
            url=f"{geocatalog_url}/stac/collections",
            stac_config=stac_config,
        )
        self._update_tile_settings(geocatalog_url, stac_config["id"])
        self._update_render_options(geocatalog_url, stac_config["id"])
        return stac_config["id"]

    def _ensure_collection_exists(
        self,
        geocatalog_url: str,
        collection_id: str,
    ) -> str:
        """Return collection ID, creating the collection if it does not exist."""
        response = self._get(f"{geocatalog_url}/stac/collections/{collection_id}")
        status = response.status_code
        if status == 200:
            return collection_id
        if status != 404:
            raise RuntimeError(
                f"Failed to retrieve collection: Error {status} - {response.text}"
            )

        return self._create_collection(geocatalog_url, collection_id)

    def _resolve_start_time(
        self, parameters: Mapping[str, Any] | dict[str, Any]
    ) -> datetime:
        """Resolve start_time from workflow parameters (workflow-specific keys)."""
        raw: datetime | str | None = None
        if self.workflow_name == "foundry_fcn3_workflow":
            raw = parameters.get("start_time")
        elif self.workflow_name == "foundry_fcn3_stormscope_goes_workflow":
            raw = parameters.get("start_time_stormscope")
        else:
            raise ValueError(f"Unsupported workflow name: {self.workflow_name}")
        if raw is None:
            raise ValueError(
                f"Missing start time in parameters for workflow {self.workflow_name}. "
                "Expected 'start_time' or (for stormscope) 'start_time_stormscope'."
            )
        if isinstance(raw, str):
            normalized = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(normalized)
        if isinstance(raw, datetime):
            return raw
        if hasattr(raw, "isoformat"):
            return datetime.fromisoformat(raw.isoformat())
        raise TypeError(f"start_time must be str or datetime, got {type(raw)}")

    def create_feature(
        self,
        geocatalog_url: str,
        collection_id: str | None,
        parameters: Mapping[str, Any] | dict[str, Any],
        blob_url: str,
    ) -> tuple[str, str]:
        """Ingest a new STAC feature into the collection.

        Parameters
        ----------
        geocatalog_url : str
            URL to the Planetary Computer Pro catalog
        collection_id : str | None
            Existing collection ID, or None to create a new collection
        parameters : Mapping[str, Any] | dict[str, Any]
            Workflow parameters including start time (and optional collection_id override
            is passed separately)
        blob_url : str
            Blob location on Azure Blob Storage

        Returns
        -------
        tuple[str, str]
            Collection ID and feature ID
        """
        self.update_headers()

        if collection_id is None:
            collection_id = self._create_collection(geocatalog_url, None)
        else:
            self._ensure_collection_exists(geocatalog_url, collection_id)

        start_time = self._resolve_start_time(parameters)
        step_sizes = {
            "foundry_fcn3_workflow": 6,
            "foundry_fcn3_stormscope_goes_workflow": 1,
        }
        end_time = start_time + timedelta(hours=step_sizes[self.workflow_name])
        stac_config = self._get_feature_json(start_time, end_time, blob_url)

        self._create_element(
            url=f"{geocatalog_url}/stac/collections/{collection_id}/items",
            stac_config=stac_config,
        )

        return collection_id, stac_config["id"]

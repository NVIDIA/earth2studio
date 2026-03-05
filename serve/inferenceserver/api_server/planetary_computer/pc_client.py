import json
import logging
import os
import time
from datetime import datetime, timedelta
from time import perf_counter
from typing import Literal
from uuid import uuid4

import requests
from api_server.workflow import WorkflowParameters
from azure.identity import DefaultAzureCredential

logger = logging.getLogger("planetary_computer")
logger.setLevel(logging.INFO)


class PlanetaryComputerClient:
    """
    Simple client for the interaction with Planetary Computer.

    The official SDK does not work as documented, hence this implementation.

    Args:
        workflow_name: Name of the Earth2Studio workflow
    """

    APPLICATION_URL = "https://geocatalog.spatio.azure.com/"
    REQUESTS_TIMEOUT = 30
    CREATION_TIMEOUT = 120

    def __init__(
        self,
        workflow_name: Literal[
            "foundry_fcn3_workflow", "foundry_fcn3_stormscope_goes_workflow"
        ],
    ):
        self.workflow_name = workflow_name
        self.headers: dict | None = None

    def update_headers(self) -> None:
        """
        Get the Azure authorization headers.
        """
        credential = DefaultAzureCredential()
        token = credential.get_token(self.APPLICATION_URL)
        self.headers = {"Authorization": f"Bearer {token.token}"}

    def _get(self, url: str) -> requests.Response:
        return requests.get(
            url,
            headers=self.headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _post(self, url: str, body: dict | None = None) -> requests.Response:
        return requests.post(
            url,
            json=body,
            headers=self.headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _put(self, url: str, body: dict | None = None) -> requests.Response:
        return requests.put(
            url,
            json=body,
            headers=self.headers,
            params={"api-version": "2025-04-30-preview"},
            timeout=self.REQUESTS_TIMEOUT,
        )

    def _create_element(self, url: str, stac_config: dict) -> None:
        """
        Create a STAC collection or feature and wait for it to finish.
        """
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
        """
        Load the STAC collection template and set the collection ID.

        Args:
            collection_id: STAC collection ID

        Returns:
            STAC config as dict
        """
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
        """
        Load the STAC feature template and set the workflow parameters.

        Args:
            start_time: Forecast start time
            end_time: Forecast end time
            blob_url: Blob location on Azure Blob Storage

        Returns:
            STAC config as dict
        """
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
        stac_config["properties"]["start_datetime"] = iso_end
        stac_config["properties"]["end_datetime"] = iso_end
        stac_config["assets"]["data"]["href"] = blob_url
        return stac_config

    def _update_tile_settings(self, geocatalog_url: str, collection_id: str) -> None:
        """
        Update 'minZoom' of the tile settings so user can zoom out.

        Args:
            geocatalog_url: URL to the Planetary Computer Pro catalog
            collection_id: STAC collection ID
        """
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
                f"Could not update tile settings: Error {status} - {response.text}"
            )

    def _update_render_options(
        self,
        geocatalog_url: str,
        collection_id: str,
    ) -> None:
        """
        Add some example render options for a new collection.

        Args:
            geocatalog_url: URL to the Planetary Computer Pro catalog
            collection_id: STAC collection ID
        """
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
                    f"Could not update render options: Error {status} - {response.text}"
                )

    def _create_collection(
        self,
        geocatalog_url: str,
        collection_id: str | None,
    ) -> str:
        """
        Create a new STAC collection.

        Args:
            geocatalog_url: URL to the Planetary Computer Pro catalog
            collection_id: STAC collection ID

        Returns:
            ID of the created collection
        """
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
        """
        Check whether a collection already exists and if not create it.

        Args:
            geocatalog_url: URL to the Planetary Computer Pro catalog
            collection_id: STAC collection ID

        Returns:
            ID of the collection
        """
        # Check whether collection already exists
        response = self._get(f"{geocatalog_url}/stac/collections/{collection_id}")
        status = response.status_code
        if status == 200:
            return collection_id
        if status != 404:
            raise RuntimeError(
                f"Failed to retrieve collection: Error {status} - {response.text}"
            )

        # Create new collection
        return self._create_collection(geocatalog_url, collection_id)

    def create_feature(
        self,
        geocatalog_url: str,
        collection_id: str | None,
        parameters: dict | WorkflowParameters,
        blob_url: str,
    ) -> tuple[str, str]:
        """
        Ingest a new STAC feature into the collection.

        Args:
            workflow_name: Name of the workflow
            geocatalog_url: URL to the Planetary Computer Pro catalog
            collection_id: STAC collection ID
            parameters: Workflow parameters with start time and number of steps
            blob_url: Blob location on Azure Blob Storage

        Returns:
            Tuple of collection ID and feature ID
        """
        self.update_headers()

        # Make sure the target collection exists
        if collection_id is None:
            collection_id = self._create_collection(geocatalog_url, None)
        else:
            self._ensure_collection_exists(geocatalog_url, collection_id)

        start_time = parameters["start_time"]
        step_sizes = {
            # Forecast step size in hours
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

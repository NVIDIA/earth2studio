import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Literal
from uuid import uuid4

import requests
from api_server.workflow import WorkflowParameters
from azure.identity import DefaultAzureCredential

logger = logging.getLogger("planetary_computer")
logger.setLevel(logging.INFO)

APPLICATION_URL = "https://geocatalog.spatio.azure.com/"
REQUESTS_TIMEOUT = 30


def get_headers() -> dict:
    """Get the Azure authorization headers."""
    credential = DefaultAzureCredential()
    token = credential.get_token(APPLICATION_URL)
    return {"Authorization": f"Bearer {token.token}"}


def _create_element(url: str, stac_config: dict, headers: dict) -> None:
    """Create a STAC collection or feature and wait for it to finish."""
    response = requests.post(
        url,
        json=stac_config,
        headers=headers,
        params={"api-version": "2025-04-30-preview"},
        timeout=REQUESTS_TIMEOUT,
    )
    location = response.headers["location"]

    logger.info("Creating '%s'...", stac_config["id"])
    while True:
        response = requests.get(location, headers=headers, timeout=REQUESTS_TIMEOUT)
        status = response.json()["status"]
        logger.info(status)
        if status not in {"Pending", "Running"}:
            break
        time.sleep(5)

    if status == "Finished":
        logger.info("Successfully created '%s'", stac_config["id"])
    else:
        logger.error("Failed to create '%s': %s", stac_config["id"], response.text)


def get_collection_json(
    workflow_name: Literal[
        "foundry_fcn3_workflow", "foundry_fcn3_stormscope_goes_workflow"
    ],
    collection_id: str | None,
) -> dict:
    """
    Load the STAC collection template and set the collection ID.

    Args:
        workflow_name: Name of the workflow
        collection_id: STAC collection ID

    Returns:
        STAC config as dict
    """
    template_fns = {
        "foundry_fcn3_workflow": "template-collection-fcn3.json",
        "foundry_fcn3_stormscope_goes_workflow": "template-collection-fcn3-stormscope-goes.json",
    }
    with open(
        os.path.join(os.path.dirname(__file__), template_fns[workflow_name])
    ) as f:
        stac_config = json.load(f)
    if collection_id is None:
        stac_config["id"] = stac_config["id"].format(uuid=uuid4())
    else:
        stac_config["id"] = collection_id
    return stac_config


def get_feature_json(
    workflow_name: Literal[
        "foundry_fcn3_workflow", "foundry_fcn3_stormscope_goes_workflow"
    ],
    start_time: datetime,
    end_time: datetime,
    blob_url: str,
) -> dict:
    """
    Load the STAC feature template and set the workflow parameters.

    Args:
        workflow_name: Name of the workflow
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
    with open(
        os.path.join(os.path.dirname(__file__), template_fns[workflow_name])
    ) as f:
        stac_config = json.load(f)

    iso_start = start_time.isoformat()
    iso_end = end_time.isoformat()

    stac_config["id"] = stac_config["id"].format(
        start_time=iso_start[:13], uuid=uuid4()
    )
    stac_config["properties"]["datetime"] = iso_start
    stac_config["properties"]["start_datetime"] = iso_end
    stac_config["properties"]["end_datetime"] = end_time
    stac_config["assets"]["data"]["href"] = blob_url
    return stac_config


def create_collection(
    workflow_name: Literal[
        "foundry_fcn3_workflow", "foundry_fcn3_stormscope_goes_workflow"
    ],
    geocatalog_url: str,
    collection_id: str | None,
    headers: dict,
) -> str:
    """
    Create a new STAC collection.

    Args:
        workflow_name: Name of the workflow
        geocatalog_url: URL to the Planetary Computer Pro catalog
        collection_id: STAC collection ID
        headers: Azure authorization headers

    Returns:
        ID of the created collection
    """
    stac_config = get_collection_json(workflow_name, collection_id)
    _create_element(
        url=f"{geocatalog_url}/stac/collections",
        stac_config=stac_config,
        headers=headers,
    )
    return stac_config["id"]


def ensure_collection_exists(
    workflow_name: Literal[
        "foundry_fcn3_workflow", "foundry_fcn3_stormscope_goes_workflow"
    ],
    geocatalog_url: str,
    collection_id: str,
    headers: dict,
) -> str:
    """
    Check whether a collection already exists and if not create it.

    Args:
        workflow_name: Name of the workflow
        geocatalog_url: URL to the Planetary Computer Pro catalog
        collection_id: STAC collection ID
        headers: Azure authorization headers

    Returns:
        ID of the collection
    """
    # Check whether collection already exists
    response = requests.get(
        f"{geocatalog_url}/stac/collections/{collection_id}",
        headers=headers,
        params={"api-version": "2025-04-30-preview"},
        timeout=REQUESTS_TIMEOUT,
    )
    status = response.status_code
    if status == 200:
        return collection_id
    if status != 404:
        raise RuntimeError(
            f"Failed to retrieve collection: Error {status} - {response.text}"
        )

    # Create new collection
    return create_collection(workflow_name, geocatalog_url, collection_id, headers)


def create_feature(
    workflow_name: Literal[
        "foundry_fcn3_workflow", "foundry_fcn3_stormscope_goes_workflow"
    ],
    geocatalog_url: str,
    collection_id: str | None,
    parameters: dict | WorkflowParameters,
    blob_url: str,
) -> None:
    """
    Ingest a new STAC feature into the collection.

    Args:
        workflow_name: Name of the workflow
        geocatalog_url: URL to the Planetary Computer Pro catalog
        collection_id: STAC collection ID
        parameters: Workflow parameters with start time and number of steps
        blob_url: Blob location on Azure Blob Storage
    """
    headers = get_headers()

    # Make sure the target collection exists
    if collection_id is None:
        collection_id = create_collection(workflow_name, geocatalog_url, None, headers)
    else:
        ensure_collection_exists(workflow_name, geocatalog_url, collection_id, headers)

    start_time = parameters["start_time"]
    step_sizes = {
        # Forecast step size in hours
        "foundry_fcn3_workflow": 6,
        "foundry_fcn3_stormscope_goes_workflow": 1,
    }
    end_time = start_time + timedelta(hours=step_sizes[workflow_name])
    stac_config = get_feature_json(workflow_name, start_time, end_time, blob_url)

    _create_element(
        url=f"{geocatalog_url}/stac/collections/{collection_id}/items",
        stac_config=stac_config,
        headers=headers,
    )

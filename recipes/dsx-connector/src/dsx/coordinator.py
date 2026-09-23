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

"""Coordinate DSX forecast and metadata publishing.

This module is model-independent: it accepts ``ForecastSeries`` objects from any producer. For
forecasts, it converts and validates the complete batch before adding it to the publisher queue.
A conversion or validation error therefore leaves the queue unchanged.

For metadata, it builds and validates messages for each configured site and variable. The
publisher then sends these as retained MQTT messages.

The caller is responsible for clearing its forecast buffer after ``stage_forecasts`` succeeds.
"""

from __future__ import annotations

from typing import Any

from ..shared.forecast import ForecastSeries
from .contract_adapter import series_to_bundle, site_metadata_message
from .publisher import DSXPublisher
from .schema import WeatherSchema


class DSXCoordinator:
    """Prepare and validate forecasts and metadata before publishing.

    Parameters
    ----------
    publisher : DSXPublisher
        Shared publisher that queues forecasts and sends metadata.
    schema : WeatherSchema
        Loaded DSX weather contract used to validate topics and payloads.
    topic_prefix : str
        Beginning of each MQTT topic, such as ``Weather/v1/PUB``.
    product : str
        Product identifier included in each topic.
    sites : dict[str, dict[str, Any]]
        Site configurations keyed by site identifier. Each configuration contains ``id``, ``lat``,
        and ``lon``.
    model_id : str
        Identifier of the model that produced the forecasts.
    horizon_seconds : int
        Maximum supported forecast lead time in seconds. Forecasts exceeding this value are
        rejected.
    cadence_seconds : int
        Expected number of seconds between forecast cycles.
    inputs : list[dict[str, str]]
        Input data sources and the role of each source in producing the forecast.
    variables : dict[str, dict[str, Any]]
        Metadata for each forecast variable. Metadata messages are prepared for every configured
        combination of site and variable.
    include_members : bool, optional
        Whether ensemble forecast payloads include the values of individual members in addition
        to summary statistics. Default is ``False``.
    """

    def __init__(
        self,
        publisher: DSXPublisher,
        schema: WeatherSchema,
        topic_prefix: str,
        product: str,
        sites: dict[str, dict[str, Any]],
        model_id: str,
        horizon_seconds: int,
        cadence_seconds: int,
        inputs: list[dict[str, str]],
        variables: dict[str, dict[str, Any]],
        include_members: bool = False,
    ) -> None:
        self.publisher = publisher
        self.schema = schema
        self.topic_prefix = topic_prefix
        self.product = product
        self.sites = sites
        self.model_id = model_id
        self.horizon_seconds = horizon_seconds
        self.cadence_seconds = cadence_seconds
        self.inputs = inputs
        self.variables = variables
        # Only meaningful for ensemble producers: emit raw member curves alongside the summaries.
        self.include_members = include_members

    def stage_forecasts(self, series_list: list[ForecastSeries]) -> None:
        """Convert and validate a complete forecast batch before adding it to the queue.

        Every message is prepared in a local list. The list is added to the publisher queue only
        after all messages pass validation, so a failure leaves the queue unchanged.

        Parameters
        ----------
        series_list : list[ForecastSeries]
            Forecast series to prepare as one batch.

        Raises
        ------
        ValueError
            If a forecast cannot be converted, exceeds the configured horizon, or produces a topic
            that does not match the DSX contract.
        jsonschema.exceptions.ValidationError
            If a forecast payload does not match the DSX contract.
        """
        # The publisher replaces this placeholder with the actual send time.
        publication_time_ms = 0
        staged: list[tuple[str, dict[str, Any]]] = []
        for series in series_list:
            topic, bundle = series_to_bundle(
                series,
                self.topic_prefix,
                self.product,
                publication_time_ms,
                self.include_members,
            )
            lead_seconds = bundle["leadSeconds"]
            max_lead_seconds = max(lead_seconds, default=0)
            if max_lead_seconds > self.horizon_seconds:
                raise ValueError(
                    f"leadSeconds {max_lead_seconds} exceeds advertised horizonSeconds "
                    f"{self.horizon_seconds} for {topic}"
                )
            # Payload validation does not cover identifiers contained in the topic.
            if self.schema.match_topic("forecast", topic) is None:
                raise ValueError(f"forecast topic does not match the contract: {topic}")
            self.schema.validate_payload("ForecastBundleMessage", bundle)
            staged.append((topic, bundle))
        self.publisher.stage_batch(staged)

    def publish_metadata(self) -> bool:
        """Build and validate metadata for every configured site and variable, then publish it.

        Workflows call this method at startup and during heartbeats. The publisher sends
        the metadata as retained messages but does not add failed metadata messages to its replay
        queue.

        All topics and payloads are validated before publishing begins. Invalid configuration
        therefore cannot cause only part of the metadata set to be published.

        Returns
        -------
        bool
            ``True`` if every message was accepted, otherwise ``False``. Rejected messages are
            logged by the publisher, while exceptions from validation or publishing propagate.

        Raises
        ------
        ValueError
            If variable metadata does not provide a unit or a metadata topic does not match the DSX
            contract.
        jsonschema.exceptions.ValidationError
            If a metadata payload does not match the DSX contract.
        """
        messages: list[tuple[str, dict[str, Any]]] = []
        for site in self.sites.values():
            for variable, spec in self.variables.items():
                topic, metadata = site_metadata_message(
                    site,
                    variable,
                    spec,
                    self.topic_prefix,
                    self.product,
                    self.model_id,
                    self.horizon_seconds,
                    self.cadence_seconds,
                    self.inputs,
                )
                # Payload validation does not cover identifiers contained in the topic.
                if self.schema.match_topic("metadata", topic) is None:
                    raise ValueError(
                        f"metadata topic does not match the contract: {topic}"
                    )
                self.schema.validate_payload("ForecastMetadataMessage", metadata)
                messages.append((topic, metadata))
        return self.publisher.publish_metadata(messages)

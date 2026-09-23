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

"""Load the DSX weather rules and check outgoing MQTT messages.

An MQTT topic identifies where a message is published, for example
``Weather/v1/PUB/Forecast/my-product/omaha/Temperature``. A payload is the data inside that
message, represented here as a Python dictionary.

The ``data/weather.yaml`` AsyncAPI contract defines the allowed topics and payload contents. This
module checks payload fields and types using JSON Schema, and checks each topic's structure and
placeholder values before publication.

It does not depend on Earth2Studio, PyTorch, or model-specific code.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

DEFAULT_SCHEMA = Path(__file__).resolve().parents[2] / "data" / "weather.yaml"


class WeatherSchema:
    """Provide validation methods for a loaded DSX weather contract.

    Parameters
    ----------
    doc : dict[str, Any]
        Contents of an AsyncAPI YAML document after it has been read into a dictionary.
    """

    def __init__(self, doc: dict[str, Any]) -> None:
        """Store the contract's message and channel definitions."""
        self.doc = doc
        self._messages: dict[str, Any] = doc.get("components", {}).get("messages", {})
        self._channels: dict[str, Any] = doc.get("channels", {})
        self._validators: dict[str, Draft202012Validator] = {}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SCHEMA) -> WeatherSchema:
        """Load the DSX weather rules from an AsyncAPI YAML file.

        Parameters
        ----------
        path : str | Path, optional
            Path to the contract YAML. Defaults to the local ``data/weather.yaml`` file.

        Returns
        -------
        WeatherSchema
            Loaded weather contract.

        Raises
        ------
        ValueError
            If the YAML does not contain a dictionary at its top level.
        """
        schema_path = Path(path)
        with schema_path.open(encoding="utf-8") as file:
            document = yaml.safe_load(file)
        if not isinstance(document, dict):
            raise ValueError(
                f"AsyncAPI YAML must contain a dictionary at its top level: {schema_path}"
            )
        return cls(document)

    def payload_schema(self, message_name: str) -> dict[str, Any]:
        """Return the payload validation rules for a named message.

        Parameters
        ----------
        message_name : str
            Name under ``components.messages`` in the AsyncAPI contract, such as
            ``ForecastBundleMessage``.

        Returns
        -------
        dict[str, Any]
            JSON Schema that defines the message's allowed payload fields and values.

        Raises
        ------
        KeyError
            If no payload schema exists for ``message_name``.
        """
        try:
            return self._messages[message_name]["payload"]
        except KeyError as exc:
            raise KeyError(f"no payload schema for message '{message_name}'") from exc

    def validate_payload(self, message_name: str, payload: dict[str, Any]) -> None:
        """Check that a message payload follows its validation rules.

        Parameters
        ----------
        message_name : str
            Name under ``components.messages`` in the AsyncAPI contract.
        payload : dict[str, Any]
            Data inside the message.

        Raises
        ------
        KeyError
            If no payload schema exists for ``message_name``.
        jsonschema.exceptions.ValidationError
            If the payload does not follow the message's validation rules.
        """
        validator = self._validators.get(message_name)
        if validator is None:
            validator = Draft202012Validator(self.payload_schema(message_name))
            self._validators[message_name] = validator
        validator.validate(payload)

    @staticmethod
    def _address_regex(address: str) -> re.Pattern[str]:
        """Convert an MQTT topic template into a pattern that extracts placeholder values."""
        pattern_parts: list[str] = []
        for segment in address.split("/"):
            placeholder_match = re.fullmatch(r"\{(\w+)\}", segment)
            if placeholder_match:
                placeholder_name = placeholder_match.group(1)
                pattern_parts.append(f"(?P<{placeholder_name}>[^/]+)")
            else:
                pattern_parts.append(re.escape(segment))
        return re.compile("/".join(pattern_parts))

    def _channel_parameter(
        self, channel_name: str, parameter_name: str
    ) -> dict[str, Any]:
        """Return the validation rules for one value in an MQTT topic template.

        The rules may be written directly in the channel or referenced from the shared
        ``components.parameters`` section of ``weather.yaml``. This method handles both forms.
        """
        parameters = self._channels.get(channel_name, {}).get("parameters", {})
        parameter_definition = parameters.get(parameter_name, {})
        reference = parameter_definition.get("$ref")
        if reference is None:
            return parameter_definition

        prefix = "#/components/parameters/"
        if not reference.startswith(prefix):
            raise ValueError(f"unsupported parameter reference: {reference}")
        component_name = reference.removeprefix(prefix)
        return self.doc["components"]["parameters"][component_name]

    def channel_enum(self, channel_name: str, parameter_name: str) -> list[str] | None:
        """Return the fixed list of allowed values for a topic placeholder, if defined.

        Parameters
        ----------
        channel_name : str
            Topic definition under ``channels`` in ``weather.yaml``, such as ``forecast``.
        parameter_name : str
            Topic placeholder name, such as ``variable``.

        Returns
        -------
        list[str] | None
            Fixed list of allowed values, or ``None`` if no fixed list is defined.
        """
        return self._channel_parameter(channel_name, parameter_name).get("enum")

    def match_topic(self, channel_name: str, topic: str) -> dict[str, str] | None:
        """Check an MQTT topic's structure and placeholder values.

        A topic template such as ``.../{site}/{variable}`` can match a concrete topic ending in
        ``.../omaha/Temperature``. The extracted values must also satisfy any fixed allowed values,
        required format, and maximum length defined in ``weather.yaml``.

        Parameters
        ----------
        channel_name : str
            Topic definition under ``channels`` in ``weather.yaml``, such as ``forecast``.
        topic : str
            Complete MQTT topic to check.

        Returns
        -------
        dict[str, str] | None
            Extracted values for placeholders such as ``site`` and ``variable``, or ``None`` if the
            topic does not satisfy the channel rules.

        Raises
        ------
        KeyError
            If ``channel_name`` is unknown or its MQTT topic template is missing.
        """
        topic_template = self._channels[channel_name]["address"]
        topic_match = self._address_regex(topic_template).fullmatch(topic)
        if topic_match is None:
            return None
        parameters = topic_match.groupdict()
        for parameter_name, value in parameters.items():
            rules = self._channel_parameter(channel_name, parameter_name)
            allowed_values = rules.get("enum")
            if allowed_values is not None and value not in allowed_values:
                return None
            # x-pattern checks the required format; x-maxLength limits the number of characters.
            value_pattern = rules.get("x-pattern")
            if value_pattern is not None and not re.fullmatch(value_pattern, value):
                return None
            maximum_length = rules.get("x-maxLength")
            if maximum_length is not None and len(value) > int(maximum_length):
                return None
        return parameters

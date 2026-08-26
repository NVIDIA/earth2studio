# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python-Markdown extension for generated public API aliases."""

from __future__ import annotations

from re import Match
from xml.etree.ElementTree import Element

from markdown import Markdown
from markdown.extensions import Extension
from markdown.inlinepatterns import InlineProcessor

API_ALIAS_PATTERN = r'<span data-e2s-api-alias="([A-Za-z_][A-Za-z0-9_.]*)"></span>'


class _ApiAliasInlineProcessor(InlineProcessor):
    """Convert an API alias marker into an anchor element."""

    def handleMatch(  # noqa: N802
        self, match: Match[str], _data: str
    ) -> tuple[Element, int, int]:
        """Return an empty anchor containing the literal API identifier."""
        anchor = Element("a", {"href": "", "id": match.group(1)})
        return anchor, match.start(0), match.end(0)


class _ApiAliasExtension(Extension):
    """Register the API alias marker with Python-Markdown."""

    def extendMarkdown(self, md: Markdown) -> None:  # noqa: N802
        """Register alias processing before raw HTML and emphasis handling."""
        md.inlinePatterns.register(
            _ApiAliasInlineProcessor(API_ALIAS_PATTERN, md),
            "e2s-api-alias",
            175,
        )


def makeExtension(**_kwargs: object) -> _ApiAliasExtension:  # noqa: N802
    """Return the API alias Markdown extension."""
    return _ApiAliasExtension()

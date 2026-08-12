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

"""Generate catalog metadata consumed by the docs catalog page."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mkdocs_hooks import DOCS_ROOT, _catalog_records


@dataclass(frozen=True)
class _CatalogPage:
    url: str = "userguide/about/catalog/"


def main() -> None:
    """Write catalog records derived from generated API front matter."""
    output = Path(DOCS_ROOT) / "assets" / "data" / "e2s-catalog.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = _catalog_records(_CatalogPage())
    output.write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

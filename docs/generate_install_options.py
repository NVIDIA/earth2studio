# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Generate static install selector data for the docs site."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

DOCS_ROOT = Path(__file__).resolve().parent
SOURCE = DOCS_ROOT / "userguide" / "about" / "install_options.yml"
TARGET = DOCS_ROOT / "assets" / "data" / "install-options.json"


def main() -> None:
    """Write the install selector JSON asset."""
    data = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

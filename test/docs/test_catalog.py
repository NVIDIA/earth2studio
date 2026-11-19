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

import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs"))

import generate_api
import generate_catalog
import mkdocs_hooks as hooks


@pytest.mark.parametrize(
    "badges, expected",
    [
        (["gpu:24gb"], 24),
        (["gpu:80gb"], 80),
        (["gpu:96gb"], 96),
        (["gpu:0gb"], 0),
        (["gpu:12.5gb"], 12.5),
        (["gpu:80gb", "gpu:80gb"], 80),
        ([], None),
        (["gpu:unknown"], None),
        (["gpu:cpu"], None),
        (["gpu:-1gb"], None),
        (["gpu:40-80gb"], None),
        (["gpu:2x40gb"], None),
        (["gpu:80gb-per-gpu"], None),
        (["gpu:80gb", "gpu:40gb"], None),
        (["gpu:80gb", "gpu:unknown"], None),
        (["gpu:NaNgb"], None),
        (["gpu:Infinitygb"], None),
        (["gpu:80"], None),
        (["gpu:80gib"], None),
        (["gpu:" + "9" * 400 + "gb"], None),
    ],
)
def test_recommended_vram_metadata(badges: list[str], expected: float | None) -> None:
    assert hooks._catalog_vram_gb(badges) == expected


def test_configured_badge_label_does_not_determine_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hooks,
        "_mkdocs_badge_definitions",
        lambda: {"gpu:80gb": {"label": "80 GB recommended (H100)"}},
    )
    chips = hooks._catalog_chips("model", "Prognostic", ["gpu:80gb"])
    assert "Recommended VRAM: 80 GB recommended (H100)" in chips
    assert hooks._catalog_vram_gb(["gpu:80gb"]) == 80


def test_vram_chip_survives_chip_limit_and_unknown_is_explicit() -> None:
    badges = ["class:mrf", "provider:nvidia", "backend:pytorch"] + [
        f"product:{value}" for value in ("wind", "temp", "atmos", "radar")
    ]
    assert "Recommended VRAM: 80 GB" in hooks._catalog_chips(
        "model", "Prognostic", badges + ["gpu:80gb"]
    )
    assert "Recommended VRAM: unknown" in hooks._catalog_chips(
        "model", "Diagnostic", []
    )
    assert "Recommended VRAM: unknown" in " ".join(
        hooks._catalog_chips("model", "Diagnostic", ["gpu:2x40gb"])
    )
    assert hooks._catalog_chips("data", "Data Source", []) == ["Data Source"]


def test_source_badges_reach_generated_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    generated = docs / "modules" / "generated"
    page = generate_api.object_page(
        "FCN",
        "earth2studio.models.px.fcn.FCN",
        generated / "models" / "px",
        "prognostic",
    )
    assert "gpu:40gb" in page.badges
    generate_api.write_object(page)
    unknown = generated / "models" / "dx" / "unknown.md"
    unknown.parent.mkdir(parents=True)
    unknown.write_text("---\ntitle: Unknown\nbadges: []\n---\n")
    data = generated / "data" / "analysis" / "data.md"
    data.parent.mkdir(parents=True)
    data.write_text("---\ntitle: data.Example\nbadges: []\n---\n")
    definitions = hooks._mkdocs_badge_definitions()
    monkeypatch.setattr(hooks, "_mkdocs_badge_definitions", lambda: definitions)
    monkeypatch.setattr(hooks, "DOCS_ROOT", docs)
    monkeypatch.setattr(hooks, "GENERATED_API_ROOT", generated)
    monkeypatch.setattr(generate_catalog, "DOCS_ROOT", docs)
    generate_catalog.main()
    records = json.loads((docs / "assets/data/e2s-catalog.json").read_text())
    by_title = {item["title"]: item for item in records}
    assert by_title["FCN"]["recommended_vram_gb"] == 40
    assert by_title["Unknown"]["recommended_vram_gb"] is None
    assert "recommended_vram_gb" not in by_title["Example"]
    assert by_title["FCN"]["url"] == "../../../modules/generated/models/px/FCN"
    assert "Recommended VRAM: 40 GB" in hooks._catalog_fallback(records)
    assert "Recommended VRAM: unknown" in hooks._catalog_fallback(records)


@pytest.mark.parametrize(
    "source_badges, expected",
    [
        ("gpu:40gb/80gb", None),
        ("gpu:80gb+40gb", None),
        ("gpu:80gb gpu:?", None),
        ("gpu:80gb gpu:", None),
        ("gpu:80gb", 80),
        ("gpu:12.5gb", 12.5),
    ],
)
def test_complete_source_tokens_reach_catalog(
    source_badges: str,
    expected: float | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = ast.parse(
        f'class Example:\n    """Example model.\n\n    {source_badges}\n    """\n'
    ).body[0]
    monkeypatch.setattr(generate_api, "resolve", lambda _: (None, node, "class"))
    docs = tmp_path / "docs"
    generated = docs / "modules" / "generated"
    page = generate_api.object_page(
        "Example", "synthetic.Example", generated / "models" / "px", "prognostic"
    )
    generate_api.write_object(page)
    definitions = hooks._mkdocs_badge_definitions()
    monkeypatch.setattr(hooks, "_mkdocs_badge_definitions", lambda: definitions)
    monkeypatch.setattr(hooks, "DOCS_ROOT", docs)
    monkeypatch.setattr(hooks, "GENERATED_API_ROOT", generated)
    monkeypatch.setattr(generate_catalog, "DOCS_ROOT", docs)
    generate_catalog.main()
    records = json.loads((docs / "assets/data/e2s-catalog.json").read_text())
    assert len(records) == 1
    assert records[0]["recommended_vram_gb"] == expected
    if expected is None:
        assert "Recommended VRAM: unknown" in hooks._catalog_fallback(records)

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

import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import pytest


@pytest.mark.skipif(
    os.environ.get("E2S_CATALOG_BROWSER_TESTS") != "1",
    reason="Set E2S_CATALOG_BROWSER_TESTS=1 to run the optional Playwright check",
)
def test_generated_catalog_in_browser() -> None:
    from playwright.sync_api import Route, expect, sync_playwright

    docs = Path(__file__).resolve().parents[2] / "docs"
    catalog = docs / "assets/data/e2s-catalog.json"
    assert (
        catalog.exists()
    ), "Run docs/generate_api.py and docs/generate_catalog.py first"
    records = json.loads(catalog.read_text())
    expected = [
        item
        for item in records
        if item["kind"] == "model"
        and item["recommended_vram_gb"] is not None
        and item["recommended_vram_gb"] <= 80
    ]
    assert expected
    html = """<!doctype html><html lang="en"><head><title>Catalog</title>
    <link rel="stylesheet" href="/assets/stylesheets/nvidia-material.css">
    </head><body><main class="md-typeset">
    <section class="e2s-catalog" data-e2s-catalog></section></main>
    <script src="/assets/javascripts/e2s-catalog.js"></script></body></html>"""

    def serve(route: Route) -> None:
        path = urlsplit(route.request.url).path
        if path == "/catalog/":
            route.fulfill(body=html, content_type="text/html")
        elif path in {
            "/assets/stylesheets/nvidia-material.css",
            "/assets/javascripts/e2s-catalog.js",
            "/assets/data/e2s-catalog.json",
        }:
            route.fulfill(path=docs / path.lstrip("/"))
        else:
            route.abort()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=os.environ.get("E2S_CHROMIUM_EXECUTABLE")
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/*", serve)
        page.goto("http://catalog.test/catalog/")
        select = page.get_by_label("Available GPU VRAM")
        select.focus()
        select.select_option("80")
        expect(select).to_be_focused()
        expect(page.locator(".e2s-catalog-count")).to_have_text(
            f"Showing 1-{min(10, len(expected))} of {len(expected)} models"
        )
        expect(page.locator(".e2s-catalog-card h3")).to_have_text(
            [item["title"] for item in expected[:10]]
        )
        assert "vram=80" in page.url
        page.reload()
        expect(select).to_have_value("80")
        expect(page.locator(".e2s-catalog-card__chips").first).to_contain_text(
            "Recommended VRAM:"
        )
        if screenshot := os.environ.get("E2S_CATALOG_SCREENSHOT"):
            page.screenshot(path=screenshot, full_page=True)
        page.get_by_role("button", name="Clear filters").click()
        expect(select).to_have_value("")
        assert "vram=" not in page.url
        page.get_by_role("button", name="Data Sources", exact=True).click()
        expect(select).to_have_count(0)
        assert not errors
        browser.close()

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

"""Generate CSS-based HTML artboard graphics for the Earth2Studio README.

This script produces a set of HTML artboards (hero banner, quickstart video,
agent setup banner, data sources diagram, model zoo overview, and composability
pipeline graphic) along with a shared CSS stylesheet, a review page, and a JSON
manifest.

Optionally, it can export high-resolution PNG screenshots of each artboard
using a headless Chromium-based browser (Chrome, Chromium, or Edge).

Usage
-----
Generate HTML artboards only (no browser required)::

    python docs/readme_graphics.py

Generate HTML artboards and export 2x PNG screenshots::

    python docs/readme_graphics.py --export-png

Output directory: ``outputs/earth2studio-readme-graphics-css/``
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "earth2studio-readme-graphics-css"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1600, 460
ASSET_VERSION = "agent-setup-v98"
EXPORT_SCALE = 2

# Browser executable names to search on PATH (cross-platform)
BROWSER_PATH_NAMES = [
    "google-chrome",
    "google-chrome-stable",
    "chromium-browser",
    "chromium",
    "chrome",
    "msedge",
]

# Platform-specific browser install paths
CHROME_CANDIDATES = [
    # Linux
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/google-chrome-stable"),
    Path("/usr/bin/chromium-browser"),
    Path("/usr/bin/chromium"),
    Path("/snap/bin/chromium"),
    # macOS
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
    Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    # Windows
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
]

CSS = r"""
:root {
  --lightningcss-dark: initial;
  color-scheme: dark;
  --background: #0c0f0b;
  --foreground: #f4f7ef;
  --card: #131711;
  --card-foreground: #f4f7ef;
  --popover: #151914;
  --popover-foreground: #f4f7ef;
  --primary: #76b900;
  --primary-foreground: #10140d;
  --secondary: #1d241a;
  --secondary-foreground: #f4f7ef;
  --muted: #1b2118;
  --muted-foreground: #a5ae9e;
  --artboard-surface: rgba(8, 18, 9, .92);
  --artboard-surface-strong: rgba(19, 38, 12, .96);
  --card-glass: rgba(19, 23, 17, .86);
  --secondary-glass: rgba(29, 36, 26, .78);
  --muted-glass: rgba(27, 33, 24, .76);
  --box-border: color-mix(in srgb, var(--signal-green) 60%, var(--border));
  --box-border-soft: color-mix(in srgb, var(--signal-green) 44%, rgba(41, 48, 36, .7));
  --box-glow: inset 0 0 0 1px rgba(118, 185, 0, .18), 0 10px 28px rgba(118, 185, 0, .08);
  --box-glow-soft: inset 0 0 0 1px rgba(118, 185, 0, .12), 0 7px 18px rgba(118, 185, 0, .055);
  --radius-artboard: 18px;
  --radius-card: 12px;
  --accent: #2d3a23;
  --accent-foreground: #e8f7c1;
  --destructive: #ef4444;
  --border: #293024;
  --border-strong: #293024;
  --input: #293024;
  --ring: #8ad12f;
  --sidebar: #10140f;
  --sidebar-foreground: #f4f7ef;
  --sidebar-accent: #1c2518;
  --sidebar-accent-foreground: #f4f7ef;
  --sidebar-border: #293024;
  --ink-soft: #c7d0c0;
  --signal-green: var(--primary);
  --signal-cyan: #0ea5a4;
  --signal-blue: #38bdf8;
  --signal-purple: #c084fc;
  --signal-orbit: #8b5cf6;
  --signal-gold: #facc15;
  --data-spectrum-c1: #38bdf8;
  --data-spectrum-c2: #c084fc;
  --data-spectrum-c3: #8b5cf6;
  --data-spectrum-buoy: #d946ef;
  --data-spectrum-c5: #5abf65;
  --data-spectrum-c6: #0ea5a4;
  --data-spectrum-c7: #38bdf8;
  --data-spectrum-c8: #facc15;
  --data-source-spectrum: linear-gradient(90deg, var(--data-spectrum-c1) 0%, var(--data-spectrum-c2) 18%, var(--data-spectrum-c3) 33%, var(--data-spectrum-c5) 52%, var(--data-spectrum-c6) 66%, var(--data-spectrum-c7) 80%, var(--data-spectrum-c8) 100%);
  --green-wash: rgba(118, 185, 0, .035);
  --cyan-wash: rgba(14, 165, 164, .025);
  --font-sans: "NVIDIA Sans", Inter, Arial, Helvetica, sans-serif;
  --font-mono: "Cascadia Code", "SFMono-Regular", Consolas, monospace;
}

* { box-sizing: border-box; }

html,
body {
  width: 1600px;
  height: 460px;
  margin: 0;
  overflow: hidden;
  background: transparent;
  font-family: var(--font-sans);
  color: var(--foreground);
}

.artboard {
  position: relative;
  width: calc(100% - 8px);
  height: calc(100% - 8px);
  margin: 4px;
  overflow: hidden;
  padding: 30px 44px;
  background:
    radial-gradient(circle at 0% 0%, rgba(14, 165, 164, .32), rgba(91, 191, 121, .22) 22%, transparent 48%),
    radial-gradient(circle at 22% 8%, rgba(118, 185, 0, .16), transparent 38%),
    linear-gradient(135deg, rgba(16, 58, 42, .97) 0%, rgba(8, 42, 33, .95) 42%, rgba(5, 23, 19, .93) 72%, rgba(3, 11, 8, .92) 100%);
  border: 1px solid var(--box-border);
  border-radius: var(--radius-artboard);
  box-shadow: inset 0 0 0 1px rgba(118, 185, 0, .12);
}

.artboard::before {
  content: "";
  pointer-events: none;
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(color-mix(in srgb, var(--border) 62%, transparent) 1px, transparent 1px),
    linear-gradient(90deg, color-mix(in srgb, var(--border) 62%, transparent) 1px, transparent 1px);
  background-size: 28px 28px;
  mask-image: linear-gradient(rgba(0, 0, 0, .48), transparent 78%);
}

.artboard,
.artboard * {
  text-shadow: 0 1px 3px rgba(0, 0, 0, .68);
}

.artboard > * { position: relative; z-index: 1; }

.graphic-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  column-gap: 32px;
  margin-bottom: 18px;
  padding-bottom: 12px;
  border-bottom: 1px solid color-mix(in srgb, var(--border) 72%, transparent);
}

.kicker,
.zone-kicker,
.sensor-label,
.lane-label {
  color: var(--signal-green);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.1;
  font-weight: 820;
  letter-spacing: 0;
  text-transform: uppercase;
}

.graphic-title {
  margin: 7px 0 0;
  font-size: 38px;
  line-height: 1.04;
  font-weight: 800;
}

.graphic-subtitle {
  margin: 7px 0 0;
  max-width: 1040px;
  color: var(--ink-soft);
  font-size: 20px;
  line-height: 1.34;
}

.kicker {
  font-size: 16px;
  line-height: 1.05;
  font-weight: 860;
}

.header-pill {
  display: inline-grid;
  min-width: 270px;
  height: 44px;
  align-items: center;
  justify-content: center;
  padding: 0 20px;
  border: 1px solid var(--box-border);
  border-radius: 999px;
  background: var(--secondary-glass);
  color: var(--ink-soft);
  font-size: 17px;
  line-height: 1;
  font-weight: 860;
  box-shadow: var(--box-glow-soft);
}

.arch-grid {
  display: grid;
  align-items: stretch;
  gap: 12px;
}

.arch-grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.arch-grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.arch-grid.cols-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }

.zone {
  min-width: 0;
  min-height: 184px;
  padding: 16px;
  border: 1px solid var(--box-border-soft);
  border-top: 4px solid var(--accent, var(--signal-green));
  border-radius: 8px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent, var(--signal-green)) 7%, transparent), transparent 42%),
    var(--card-glass);
  box-shadow: var(--box-glow-soft);
}

.zone h3 {
  margin: 9px 0 0;
  color: var(--foreground);
  font-size: 21px;
  line-height: 1.15;
  font-weight: 760;
}

.zone p {
  margin: 8px 0 0;
  color: var(--muted-foreground);
  font-size: 14px;
  line-height: 1.35;
}

.component-grid {
  display: grid;
  gap: 8px;
  margin-top: 12px;
}

.component-grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }

.component {
  min-width: 0;
  padding: 10px 12px;
  border: 1px solid var(--box-border-soft);
  border-radius: 8px;
  background: var(--muted-glass);
  box-shadow: var(--box-glow-soft);
}

.component strong,
.component small {
  display: block;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.component strong {
  color: var(--foreground);
  font-size: 15px;
  line-height: 1.25;
  font-weight: 740;
}

.component small {
  margin-top: 5px;
  color: var(--muted-foreground);
  font-size: 12px;
  line-height: 1.25;
  font-weight: 620;
}

.pill-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 14px;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 9px;
  height: 30px;
  padding: 0 13px;
  border: 1px solid color-mix(in srgb, var(--accent, var(--signal-green)) 72%, var(--box-border));
  border-radius: 999px;
  background: var(--secondary-glass);
  color: var(--muted-foreground);
  font-size: 13px;
  line-height: 1;
  font-weight: 720;
  box-shadow: var(--box-glow-soft);
}

.pill::before {
  content: "";
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: var(--accent, var(--signal-green));
}

.pipeline {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.node {
  min-width: 0;
  min-height: 58px;
  display: grid;
  align-content: center;
  justify-items: center;
  padding: 9px 10px;
  border: 1px solid color-mix(in srgb, var(--accent, var(--signal-green)) 72%, var(--box-border));
  border-radius: 8px;
  background: color-mix(in srgb, var(--card-glass) 88%, var(--accent, var(--signal-green)) 12%);
  box-shadow: var(--box-glow-soft);
}

.node strong,
.node small {
  display: block;
  max-width: 100%;
  overflow: hidden;
  text-align: center;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.node strong {
  font-size: 18px;
  line-height: 1.15;
  font-weight: 780;
}

.node small {
  margin-top: 6px;
  color: var(--muted-foreground);
  font-size: 12px;
  font-weight: 650;
}

.flow-band {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 18px;
}

.flow-note {
  min-height: 34px;
  display: grid;
  place-items: center;
  padding: 0 14px;
  border: 1px solid color-mix(in srgb, var(--accent, var(--signal-green)) 72%, var(--box-border));
  border-radius: 999px;
  background: var(--secondary-glass);
  color: var(--muted-foreground);
  font-size: 13px;
  line-height: 1;
  font-weight: 720;
  box-shadow: var(--box-glow-soft);
}

.workflow-stack {
  display: grid;
  gap: 9px;
  margin-top: 6px;
}

.workflow-row {
  display: grid;
  grid-template-columns: 205px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
}

.workflow-run {
  height: 64px;
  display: grid;
  align-content: center;
  padding: 8px 12px;
  border: 1px solid var(--box-border);
  border-left: 4px solid var(--signal-green);
  border-radius: 8px;
  background:
    linear-gradient(90deg, color-mix(in srgb, var(--signal-green) 12%, transparent), transparent 72%),
    var(--secondary-glass);
  box-shadow: var(--box-glow-soft);
}

.workflow-run strong,
.workflow-run small,
.workflow-step strong,
.workflow-step small {
  display: block;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.workflow-run strong {
  color: var(--foreground);
  font-family: var(--font-mono);
  font-size: 15px;
  line-height: 1.1;
  font-weight: 820;
}

.workflow-run small {
  margin-top: 5px;
  color: var(--muted-foreground);
  font-size: 12px;
  line-height: 1.1;
  font-weight: 650;
}

.workflow-chain {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.workflow-step {
  position: relative;
  min-width: 0;
  height: 64px;
  flex: 1 1 0;
  display: grid;
  align-content: center;
  padding: 8px 14px;
  border: 1px solid color-mix(in srgb, var(--accent, var(--signal-green)) 58%, var(--box-border));
  border-radius: 8px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent, var(--signal-green)) 10%, transparent), transparent 62%),
    color-mix(in srgb, var(--card-glass) 86%, var(--accent, var(--signal-green)) 14%);
  box-shadow: var(--box-glow-soft);
}

.workflow-link {
  position: relative;
  flex: 0 0 30px;
  height: 2px;
  border-radius: 999px;
  background: color-mix(in srgb, var(--accent, var(--signal-green)) 82%, transparent);
}

.workflow-link::after {
  content: "";
  position: absolute;
  top: 50%;
  right: 0;
  width: 8px;
  height: 8px;
  border-top: 2px solid color-mix(in srgb, var(--accent, var(--signal-green)) 90%, transparent);
  border-right: 2px solid color-mix(in srgb, var(--accent, var(--signal-green)) 90%, transparent);
  transform: translateY(-50%) rotate(45deg);
}

.workflow-step strong {
  color: var(--foreground);
  font-size: 16px;
  line-height: 1.15;
  font-weight: 800;
}

.workflow-step small {
  margin-top: 5px;
  color: var(--muted-foreground);
  font-size: 12px;
  line-height: 1.15;
  font-weight: 650;
}

.workflow-notes {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.hero-map {
  display: grid;
  grid-template-columns: 270px 42px 300px 42px 300px 42px 350px;
  align-items: stretch;
  gap: 10px;
}

.core-strip {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
}

.core-card {
  min-width: 0;
  min-height: 168px;
  padding: 16px;
  border: 1px solid var(--box-border-soft);
  border-top: 4px solid var(--accent, var(--signal-green));
  border-radius: 8px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent, var(--signal-green)) 8%, transparent), transparent 44%),
    var(--card-glass);
  box-shadow: var(--box-glow);
}

.core-card h3 {
  margin: 9px 0 0;
  color: var(--foreground);
  font-size: 22px;
  line-height: 1.08;
  font-weight: 800;
}

.core-card p {
  margin: 8px 0 0;
  min-height: 30px;
  color: var(--muted-foreground);
  font-size: 14px;
  line-height: 1.38;
  font-weight: 650;
}

.core-card .phase {
  margin-top: 12px;
  min-height: 30px;
  display: flex;
  align-items: center;
  padding: 0 10px;
  border: 1px solid color-mix(in srgb, var(--accent, var(--signal-green)) 42%, var(--border));
  border-radius: 999px;
  background: var(--secondary-glass);
  color: var(--muted-foreground);
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1;
  font-weight: 760;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.flow-arrow {
  align-self: center;
  height: 3px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--signal-green), var(--signal-cyan));
  position: relative;
}

.flow-arrow::after {
  content: "";
  position: absolute;
  top: 50%;
  right: -1px;
  width: 9px;
  height: 9px;
  border-top: 2px solid var(--signal-cyan);
  border-right: 2px solid var(--signal-cyan);
  transform: translateY(-50%) rotate(45deg);
}

.quickstart-flow {
  display: grid;
  grid-template-columns: 1fr 52px 1fr 52px 1fr 34px 280px;
  gap: 14px;
  align-items: stretch;
  margin-top: 18px;
}

.quick-spacer {
  min-width: 0;
}

.quick-node,
.quick-video {
  min-width: 0;
  min-height: 220px;
  border: 1px solid color-mix(in srgb, var(--accent, var(--signal-green)) 68%, var(--box-border));
  border-top: 4px solid var(--accent, var(--signal-green));
  border-radius: var(--radius-card);
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent, var(--signal-green)) 10%, transparent), transparent 58%),
    color-mix(in srgb, var(--card-glass) 88%, var(--accent, var(--signal-green)) 12%);
  box-shadow: var(--box-glow);
}

.quick-node {
  display: grid;
  grid-template-columns: 86px minmax(0, 1fr);
  gap: 16px;
  align-items: center;
  padding: 22px;
}

.quick-node .icon {
  width: 78px;
  height: 78px;
  color: var(--accent, var(--signal-green));
  filter: drop-shadow(0 8px 18px color-mix(in srgb, var(--accent, var(--signal-green)) 30%, transparent));
}

.quick-copy {
  min-width: 0;
}

.quick-copy h3 {
  margin: 9px 0 0;
  color: var(--foreground);
  font-size: 28px;
  line-height: 1.06;
  font-weight: 820;
  white-space: nowrap;
}

.quick-copy p {
  margin: 8px 0 0;
  color: var(--muted-foreground);
  font-size: 14px;
  line-height: 1.35;
  font-weight: 650;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.quick-chip {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  margin-top: 15px;
  padding: 0 10px;
  border: 1px solid color-mix(in srgb, var(--accent, var(--signal-green)) 58%, var(--border));
  border-radius: 999px;
  background: var(--secondary-glass);
  color: var(--muted-foreground);
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1;
  font-weight: 780;
  white-space: nowrap;
}

.quick-arrow {
  align-self: center;
  height: 3px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--signal-green), var(--signal-cyan));
  position: relative;
}

.quick-arrow::after {
  content: "";
  position: absolute;
  top: 50%;
  right: -1px;
  width: 11px;
  height: 11px;
  border-top: 2px solid var(--signal-cyan);
  border-right: 2px solid var(--signal-cyan);
  transform: translateY(-50%) rotate(45deg);
}

.quick-video {
  display: grid;
  align-content: center;
  justify-items: start;
  padding: 22px;
  border-color: color-mix(in srgb, var(--signal-green) 48%, var(--box-border));
  border-top-color: color-mix(in srgb, var(--signal-green) 58%, var(--box-border));
  background:
    radial-gradient(circle at 18% 18%, rgba(118, 185, 0, .12), transparent 34%),
    linear-gradient(135deg, rgba(118, 185, 0, .07), rgba(14, 165, 164, .05) 58%, transparent),
    var(--card-glass);
}

.play-lockup {
  display: flex;
  align-items: center;
  gap: 13px;
  color: var(--ink-soft);
  font-family: var(--font-mono);
  font-size: 15px;
  font-weight: 820;
  text-transform: uppercase;
}

.play-button {
  width: 50px;
  height: 50px;
  display: grid;
  place-items: center;
  border: 1px solid color-mix(in srgb, var(--signal-green) 54%, var(--box-border));
  border-radius: 999px;
  background: color-mix(in srgb, var(--signal-green) 8%, transparent);
  box-shadow: var(--box-glow-soft);
}

.play-button svg {
  width: 23px;
  height: 23px;
  margin-left: 2px;
}

.quick-video strong {
  display: block;
  margin-top: 18px;
  color: var(--foreground);
  font-size: 27px;
  line-height: 1.08;
  font-weight: 840;
}

.quick-video small {
  display: block;
  margin-top: 10px;
  color: var(--muted-foreground);
  font-size: 14px;
  line-height: 1.35;
  font-weight: 650;
}

.quick-command {
  margin-top: 14px;
  min-height: 34px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid color-mix(in srgb, var(--signal-green) 64%, var(--box-border));
  border-radius: 999px;
  background: var(--secondary-glass);
  color: var(--ink-soft);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 760;
  box-shadow: var(--box-glow-soft);
}

.agent-setup-flow {
  position: relative;
  display: grid;
  grid-template-columns: 690px minmax(0, 1fr);
  gap: 28px;
  margin-top: 22px;
}

.agent-terminal {
  min-height: 232px;
  padding: 18px 20px;
  border: 1px solid color-mix(in srgb, var(--signal-cyan) 86%, var(--box-border));
  border-radius: var(--radius-card);
  background: linear-gradient(180deg, rgba(5, 18, 16, .96), rgba(7, 15, 10, .96));
  box-shadow: var(--box-glow), inset 0 0 0 1px rgba(22, 199, 199, .12);
}

.agent-terminal-top {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 14px;
}

.agent-dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  background: var(--signal-green);
}

.agent-dot:nth-child(2) { background: var(--signal-cyan); }
.agent-dot:nth-child(3) { background: var(--signal-gold); }

.agent-terminal-title {
  margin-left: 8px;
  color: var(--ink-soft);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1;
  font-weight: 800;
}

.agent-line {
  --chars: 60;
  display: block;
  width: 100%;
  overflow: hidden;
  white-space: nowrap;
  color: var(--foreground);
  font-family: var(--font-mono);
  font-size: 13.8px;
  line-height: 1.68;
  font-weight: 700;
}

.agent-line .cmd { color: var(--signal-green); }
.agent-line .ready { color: var(--signal-gold); }

.agent-cursor {
  display: inline-block;
  width: 9px;
  height: 19px;
  margin-left: 4px;
  vertical-align: -4px;
  background: var(--signal-green);
}

.agent-skill-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 18px;
}

.agent-skill-card {
  min-width: 0;
  min-height: 232px;
  padding: 18px;
  border: 1px solid var(--accent);
  border-top: 5px solid var(--accent);
  border-radius: var(--radius-card);
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent) 13%, transparent), transparent 56%),
    var(--card-glass);
  box-shadow: var(--box-glow);
}

.agent-skill-card .label {
  color: var(--accent);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.1;
  font-weight: 850;
  text-transform: uppercase;
}

.agent-skill-card h3 {
  margin: 14px 0 0;
  color: var(--foreground);
  font-size: 25px;
  line-height: 1.08;
  font-weight: 820;
}

.agent-skill-card p {
  margin: 9px 0 0;
  color: var(--ink-soft);
  font-size: 15.5px;
  line-height: 1.28;
  font-weight: 660;
}

.agent-examples {
  display: grid;
  gap: 6px;
  margin: 12px 0 0;
  padding: 0;
  list-style: none;
}

.agent-examples li {
  min-width: 0;
  padding: 6px 9px;
  overflow: hidden;
  border: 1px solid color-mix(in srgb, var(--accent) 42%, transparent);
  border-radius: 999px;
  background: var(--secondary-glass);
  color: var(--foreground);
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 800;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.observing-system {
  position: relative;
  height: 184px;
  margin-top: -5px;
}

.earth-horizon {
  position: absolute;
  left: 34px;
  right: 34px;
  bottom: -164px;
  height: 242px;
  border: 1px solid color-mix(in srgb, var(--signal-cyan) 46%, var(--border));
  border-radius: 50% 50% 0 0 / 100% 100% 0 0;
  background:
    radial-gradient(ellipse at 32% 8%, rgba(118, 185, 0, .24), transparent 34%),
    linear-gradient(90deg, rgba(16, 37, 34, .74) 0 42%, rgba(22, 41, 20, .74) 42% 100%);
  overflow: hidden;
}

.earth-horizon::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse at 50% 112%, transparent 30%, rgba(14, 165, 164, .22) 30.4%, transparent 31%),
    radial-gradient(ellipse at 50% 112%, transparent 48%, rgba(90, 191, 101, .18) 48.4%, transparent 49%),
    radial-gradient(ellipse at 50% 112%, transparent 64%, rgba(14, 165, 164, .14) 64.4%, transparent 65%),
    radial-gradient(ellipse at 18% 112%, transparent 42%, rgba(14, 165, 164, .12) 42.4%, transparent 43%),
    radial-gradient(ellipse at 82% 112%, transparent 42%, rgba(14, 165, 164, .12) 42.4%, transparent 43%);
  opacity: .72;
  mask-image: linear-gradient(to top, rgba(0, 0, 0, .7), rgba(0, 0, 0, .35) 58%, transparent 96%);
}

.earth-horizon::after {
  content: "";
  position: absolute;
  left: -2%;
  right: -2%;
  top: 56px;
  height: 70px;
  background: rgba(22, 41, 20, .72);
  clip-path: polygon(0 64%, 9% 48%, 18% 52%, 28% 34%, 39% 45%, 50% 26%, 61% 42%, 72% 32%, 86% 48%, 100% 38%, 100% 100%, 0 100%);
}

.source-grid {
  position: absolute;
  inset: 6px 0 0;
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  grid-template-rows: 78px 92px;
  gap: 8px 10px;
}

.sensor {
  position: relative;
  display: grid;
  justify-items: center;
  align-content: start;
  min-width: 0;
}

.sensor .icon {
  width: 64px;
  height: 46px;
  color: var(--accent, var(--signal-green));
}

.sensor .sensor-label {
  margin-top: 1px;
  color: var(--accent, var(--signal-green));
  text-align: center;
  font-size: 10px;
  line-height: 1.15;
}

.sensor small {
  display: block;
  margin-top: 2px;
  color: var(--muted-foreground);
  text-align: center;
  font-size: 10px;
  line-height: 1.2;
  font-weight: 650;
}

.wire-layer {
  position: absolute;
  inset: 0;
  pointer-events: none;
}

.data-api {
  position: absolute;
  left: 595px;
  bottom: 6px;
  width: 314px;
  height: 58px;
  display: grid;
  align-content: center;
  justify-items: center;
  border: 1px solid var(--signal-green);
  border-top-width: 4px;
  border-radius: 8px;
  background: var(--card-glass);
  box-shadow: var(--box-glow);
}

.data-api strong {
  font-size: 20px;
  line-height: 1.1;
}

.data-api small {
  margin-top: 5px;
  color: var(--muted-foreground);
  font-size: 13px;
  font-weight: 650;
}

.source-lanes {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 7px;
}

.source-lane {
  min-width: 0;
  padding: 10px 12px;
  border: 1px dashed color-mix(in srgb, var(--accent, var(--signal-green)) 56%, var(--box-border));
  border-radius: 8px;
  background: color-mix(in srgb, var(--muted-glass) 78%, transparent);
  box-shadow: var(--box-glow-soft);
}

.source-lane strong {
  display: block;
  margin-top: 5px;
  overflow: hidden;
  color: var(--foreground);
  font-size: 15px;
  line-height: 1.1;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.source-lane small {
  display: block;
  margin-top: 4px;
  overflow: hidden;
  color: var(--muted-foreground);
  font-size: 11px;
  line-height: 1.25;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.model-zoo {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 0;
}

.model-section {
  min-width: 0;
  min-height: 198px;
  padding: 14px;
  border: 1px solid var(--box-border-soft);
  border-top: 4px solid var(--accent);
  border-radius: 8px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent) 8%, transparent), transparent 44%),
    var(--card-glass);
  box-shadow: var(--box-glow);
}

.model-section h3 {
  margin: 7px 0 0;
  font-size: 22px;
  line-height: 1.15;
}

.model-section p {
  height: 36px;
  margin: 5px 0 0;
  color: var(--muted-foreground);
  font-size: 13px;
  line-height: 1.32;
}

.model-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 9px;
  margin-top: 8px;
}

.model-list .component {
  padding: 9px 12px;
}

.model-list .component small {
  margin-top: 4px;
}

.footer-axis {
  position: absolute;
  left: 48px;
  right: 48px;
  bottom: 18px;
  margin-top: 0;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 16px;
  color: var(--muted-foreground);
  font-size: 13px;
  line-height: 1;
  font-weight: 700;
}

.footer-axis::before,
.footer-axis::after {
  content: "";
  height: 1px;
  background: color-mix(in srgb, var(--border-strong) 80%, transparent);
}

.icon svg {
  width: 100%;
  height: 100%;
  overflow: visible;
}

.icon path,
.icon line,
.icon rect,
.icon circle,
.icon ellipse,
.icon polygon {
  vector-effect: non-scaling-stroke;
}

.zone,
.component,
.node,
.workflow-run,
.workflow-step,
.core-card,
.data-api,
.source-lane,
.model-section {
  border-radius: var(--radius-card);
}
"""


ICONS = {
    "satellite": """<svg viewBox="0 0 100 70" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="39" y="27" width="22" height="16" rx="3" fill="currentColor" fill-opacity=".12"/><path d="M39 35 23 24M39 35 23 47M61 35 77 24M61 35 77 47"/><rect x="4" y="15" width="20" height="14" rx="2" fill="currentColor" fill-opacity=".12"/><rect x="4" y="42" width="20" height="14" rx="2" fill="currentColor" fill-opacity=".12"/><rect x="76" y="15" width="20" height="14" rx="2" fill="currentColor" fill-opacity=".12"/><rect x="76" y="42" width="20" height="14" rx="2" fill="currentColor" fill-opacity=".12"/><path d="M44 43 36 62M56 43 64 62"/></svg>""",
    "aircraft": """<svg viewBox="0 0 112 70" fill="none" stroke="currentColor" stroke-width="2.35" stroke-linecap="round" stroke-linejoin="round"><path d="M56 4c4 0 7 4 7 10v18l42 14c5 2 7 6 5 10L63 49v11l16 6v4L56 65 33 70v-4l16-6V49L2 56c-2-4 0-8 5-10l42-14V14c0-6 3-10 7-10Z" fill="currentColor" fill-opacity=".12"/><path d="M56 4v61M49 32 7 46M63 32l42 14M49 49l-16 17M63 49l16 17"/></svg>""",
    "radar": """<svg viewBox="0 0 90 70" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M19 26a26 26 0 0 1 52 0"/><path d="M9 19a40 40 0 0 1 72 0" opacity=".55"/><path d="M0 12a54 54 0 0 1 90 0" opacity=".32"/><circle cx="45" cy="34" r="7" fill="currentColor" fill-opacity=".15"/><path d="M45 41 33 66h24L45 41ZM25 66h40"/></svg>""",
    "balloon": """<svg viewBox="0 0 70 84" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M35 4c18 0 27 15 24 31-3 17-14 28-24 28S14 52 11 35C8 19 17 4 35 4Z" fill="currentColor" fill-opacity=".12"/><path d="M24 62 35 76l11-14M35 76v8M24 84h22"/></svg>""",
    "buoy": """<svg viewBox="0 0 86 80" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M22 58 43 13l21 45Z" fill="currentColor" fill-opacity=".12"/><circle cx="43" cy="8" r="7" fill="currentColor" fill-opacity=".18"/><path d="M30 38h26M26 52h34M4 68c16-11 30 11 44 0s22 2 34 0" opacity=".72"/></svg>""",
    "supercomputer": """<svg viewBox="0 0 92 70" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><rect x="8" y="8" width="76" height="52" rx="5" fill="currentColor" fill-opacity=".10"/><path d="M27 8v52M46 8v52M65 8v52M8 25h76M8 43h76"/><circle cx="18" cy="17" r="2.5" fill="currentColor"/><circle cx="37" cy="17" r="2.5" fill="currentColor"/><circle cx="56" cy="17" r="2.5" fill="currentColor"/><circle cx="75" cy="17" r="2.5" fill="currentColor"/></svg>""",
    "station": """<svg viewBox="0 0 90 70" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><rect x="16" y="36" width="48" height="25" rx="3" fill="currentColor" fill-opacity=".10"/><path d="M16 36 40 18l24 18M67 17v44M58 17h18M62 8h10M45 61V46H34v15"/></svg>""",
    "reanalysis": """<svg viewBox="0 0 92 70" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><rect x="24" y="16" width="44" height="34" rx="4" fill="currentColor" fill-opacity=".10"/><path d="M24 27h44M24 39h44M35 16v34M46 16v34M57 16v34"/><path d="M30 11h32M18 55h56M23 60h46" opacity=".68"/><circle cx="46" cy="33" r="4" fill="currentColor" fill-opacity=".22"/></svg>""",
    "globe": """<svg viewBox="0 0 86 86" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><circle cx="43" cy="43" r="34" fill="currentColor" fill-opacity=".10"/><ellipse cx="43" cy="43" rx="16" ry="34"/><path d="M43 9v68M9 43h68M15 27h58M15 59h58"/></svg>""",
    "network": """<svg viewBox="0 0 96 86" fill="none" stroke="currentColor" stroke-width="2.35" stroke-linecap="round" stroke-linejoin="round"><path d="M18 18 48 18M18 18 48 43M18 18 48 68M18 43 48 18M18 43 48 43M18 43 48 68M18 68 48 18M18 68 48 43M18 68 48 68M48 18 78 18M48 18 78 43M48 18 78 68M48 43 78 18M48 43 78 43M48 43 78 68M48 68 78 18M48 68 78 43M48 68 78 68" opacity=".54"/><circle cx="18" cy="18" r="7" fill="currentColor" fill-opacity=".12"/><circle cx="18" cy="43" r="7" fill="currentColor" fill-opacity=".12"/><circle cx="18" cy="68" r="7" fill="currentColor" fill-opacity=".12"/><circle cx="48" cy="18" r="7" fill="currentColor" fill-opacity=".12"/><circle cx="48" cy="43" r="7" fill="currentColor" fill-opacity=".12"/><circle cx="48" cy="68" r="7" fill="currentColor" fill-opacity=".12"/><circle cx="78" cy="18" r="7" fill="currentColor" fill-opacity=".12"/><circle cx="78" cy="43" r="7" fill="currentColor" fill-opacity=".12"/><circle cx="78" cy="68" r="7" fill="currentColor" fill-opacity=".12"/></svg>""",
    "zarr": """<svg viewBox="0 0 92 86" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><rect x="14" y="14" width="64" height="48" rx="5" fill="currentColor" fill-opacity=".10"/><path d="M14 30h64M14 46h64M30 14v48M46 14v48M62 14v48"/><path d="M24 70h52M32 76h44" opacity=".68"/><rect x="30" y="30" width="16" height="16" fill="currentColor" fill-opacity=".12"/><rect x="46" y="46" width="16" height="16" fill="currentColor" fill-opacity=".12"/></svg>""",
    "play": """<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8 5v14l11-7Z"/></svg>""",
}


def esc(value: str) -> str:
    """HTML-escape a string for safe embedding in HTML attributes and content."""
    return html.escape(value, quote=True)


def icon(name: str) -> str:
    """Return the SVG markup for a named icon from the ICONS dictionary."""
    return ICONS[name]


def component(name: str, sub: str) -> str:
    """Render a component card with a bold name and subtitle."""
    return f'<div class="component"><strong>{esc(name)}</strong><small>{esc(sub)}</small></div>'


def zone(
    kicker: str,
    title: str,
    body: str,
    items: list[tuple[str, str]],
    accent: str,
    two: bool = False,
) -> str:
    """Render a themed zone section with kicker, title, body, and a component grid."""
    grid_class = "component-grid two" if two else "component-grid"
    return f"""
      <section class="zone" style="--accent: var({accent});">
        <span class="zone-kicker">{esc(kicker)}</span>
        <h3>{esc(title)}</h3>
        <p>{esc(body)}</p>
        <div class="{grid_class}">
          {"".join(component(a, b) for a, b in items)}
        </div>
      </section>"""


def header(kicker: str, title: str, subtitle: str, pill: str) -> str:
    """Render the graphic header with kicker, title, subtitle, and a pill badge."""
    return f"""
    <header class="graphic-header">
      <div>
        <span class="kicker">{esc(kicker)}</span>
        <h1 class="graphic-title">{esc(title)}</h1>
        <p class="graphic-subtitle">{esc(subtitle)}</p>
      </div>
      <div class="header-pill">{esc(pill)}</div>
    </header>"""


def layout_page(title: str, body: str) -> str:
    """Wrap body content in a full HTML page with the shared artboard stylesheet."""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=1600, initial-scale=1" />
  <title>{esc(title)}</title>
  <link rel="stylesheet" href="./earth2studio-graphics.css?v={ASSET_VERSION}" />
</head>
<body>
  <main class="artboard" role="img" aria-label="{esc(title)}">
{body}
  </main>
</body>
</html>
"""


def hero() -> str:
    """Build the hero banner artboard showcasing Earth2Studio core capabilities."""
    core = [
        (
            "Models",
            "Pre-trained model zoo",
            "Forecast, diagnose, and assimilate with built-in AI models.",
            "phase: model",
        ),
        (
            "Data",
            "AI-ready data sources",
            "PyData-native loaders for weather, climate, and observations.",
            "phase: fetch",
        ),
        (
            "APIs",
            "Composable APIs",
            "Chain data, models, perturbations, statistics, and IO.",
            "phase: compose",
        ),
        (
            "Compute",
            "GPU accelerated",
            "Run inference and evaluation on NVIDIA-optimized workflows.",
            "phase: run",
        ),
        (
            "Agents",
            "Agent ready",
            "Structured interfaces for recipes, skills, automation, and AI assistants.",
            "phase: automate",
        ),
    ]
    accents = [
        "--signal-green",
        "--signal-cyan",
        "--signal-gold",
        "--signal-blue",
        "--signal-purple",
    ]
    cards = "\n".join(
        f"""
      <section class="core-card" style="--accent: var({accent});">
        <span class="zone-kicker">{esc(kicker)}</span>
        <h3>{esc(title)}</h3>
        <p>{esc(body)}</p>
        <div class="phase">{esc(phase)}</div>
      </section>"""
        for (kicker, title, body, phase), accent in zip(core, accents)
    )
    return layout_page(
        "Earth2Studio README hero",
        header(
            "NVIDIA",
            "Earth2Studio",
            "A Python package for building, researching, and exploring AI-driven Earth system models.",
            "Earth-2",
        )
        + f"""
    <section class="core-strip">
      {cards}
    </section>
    <div class="pill-row">
      <span class="pill" style="--accent: var(--signal-green)">Pre-trained models</span>
      <span class="pill" style="--accent: var(--signal-cyan)">PyData native</span>
      <span class="pill" style="--accent: var(--signal-gold)">Composable workflows</span>
      <span class="pill" style="--accent: var(--signal-blue)">GPU ready</span>
      <span class="pill" style="--accent: var(--signal-purple)">AI agents</span>
    </div>""",
    )


def quickstart_video() -> str:
    """Build the quickstart video artboard showing a simple forecast workflow."""
    steps = [
        ("globe", "Data Source", "GFS", "initialization data", "fetch"),
        (
            "network",
            "Prognostic Model",
            "FourCastNet3",
            "AI medium-range model",
            "model",
        ),
        ("zarr", "IO Backend", "Zarr store", "chunked output", "write"),
    ]
    accents = ["--signal-green", "--signal-cyan", "--signal-gold"]
    nodes = []
    for index, ((icon_name, kicker, title, body, chip), accent) in enumerate(
        zip(steps, accents)
    ):
        nodes.append(
            f"""
      <section class="quick-node" style="--accent: var({accent});">
        <div class="icon">{icon(icon_name)}</div>
        <div class="quick-copy">
          <span class="zone-kicker">{esc(kicker)}</span>
          <h3>{esc(title)}</h3>
          <p>{esc(body)}</p>
          <span class="quick-chip">{esc(chip)}</span>
        </div>
      </section>"""
        )
        if index < len(steps) - 1:
            nodes.append('<div class="quick-arrow"></div>')

    return layout_page(
        "Earth2Studio quick start video",
        header(
            "Quick start",
            "Get started with Earth2Studio in 5 minutes",
            "Run a simple FourCastNet3 workflow: fetch GFS, run inference, and write a Zarr store.",
            "watch video",
        )
        + f"""
    <section class="quickstart-flow">
      {"".join(nodes)}
      <div class="quick-spacer"></div>
      <aside class="quick-video">
        <div class="play-lockup">
          <span class="play-button">{icon("play")}</span>
          <span>click to watch</span>
        </div>
        <strong>Forecast workflow tutorial</strong>
      </aside>
    </section>""",
    )


def agent_setup() -> str:
    """Build the agentic setup artboard showing install and workflow skills."""
    commands = [
        ("earth2studio-install", "58", ".15s"),
        ("earth2studio-discover", "59", "1.55s"),
        ("earth2studio-data-fetch", "59", "2.95s"),
        ("earth2studio-deterministic-forecast", "72", "4.35s"),
    ]
    command_lines = "\n".join(
        f"""
        <span class="agent-line" style="--chars: {chars}; animation-delay: {delay}">
          <span class="cmd">$</span> npx skills add NVIDIA/skills --skill {esc(skill)}
        </span>"""
        for skill, chars, delay in commands
    )
    skill_cards = [
        (
            "earth2studio-discover",
            "Discover",
            "Ask an agent to recommend data, models, IO, and docs for your workflow.",
            ["find a forecast recipe", "compare data sources"],
            "--signal-cyan",
        ),
        (
            "earth2studio-install",
            "Install",
            "Automate environment setup and model-specific package guidance.",
            ["setup Earth2Studio", "install model deps"],
            "--signal-green",
        ),
        (
            "deterministic forecast",
            "Run",
            "Run a first forecast with a data source, model, and Zarr store.",
            ["GFS -> FourCastNet3", "write Zarr output"],
            "--signal-gold",
        ),
    ]
    cards = []
    for label, title, body, examples, accent in skill_cards:
        chips = "".join(f"<li>{esc(example)}</li>" for example in examples)
        cards.append(
            f"""
        <article class="agent-skill-card" style="--accent: var({accent})">
          <div class="label">{esc(label)}</div>
          <h3>{esc(title)}</h3>
          <p>{esc(body)}</p>
          <ul class="agent-examples">{chips}</ul>
        </article>"""
        )

    return layout_page(
        "Earth2Studio agentic setup",
        header(
            "Agent setup",
            "Agentic Earth2Studio setup",
            "Use NVIDIA skills to automate setup, discover workflows, and launch a first forecast.",
            "agent-ready setup",
        )
        + f"""
    <section class="agent-setup-flow">
      <div class="agent-terminal">
        <div class="agent-terminal-top">
          <span class="agent-dot"></span><span class="agent-dot"></span><span class="agent-dot"></span>
          <span class="agent-terminal-title">install commands</span>
        </div>
{command_lines}
        <span class="agent-line" style="--chars: 39; animation-delay: 5.85s">
          <span class="ready">ready&gt;</span> Earth2Studio skills installed<span class="agent-cursor"></span>
        </span>
      </div>
      <div class="agent-skill-grid">
        {"".join(cards)}
      </div>
    </section>""",
    )


def datasource() -> str:
    """Build the data sources artboard illustrating observing systems and data APIs."""
    source_lanes = [
        (
            "Data sources",
            "ARCO, CDS, CMIP6, GOES, MRMS",
            "GFS, HRRR, IFS, JPSS, WB2ERA5",
            "--signal-green",
        ),
        (
            "Forecast sources",
            "AIFS_FX, GFS_FX, GEFS_FX",
            "HRRR_FX, IFS_FX, CFS_FX, CAMS_FX",
            "--signal-cyan",
        ),
        (
            "DataFrame sources",
            "UFS, NNJA, JPSS, MetOp",
            "GHCNDaily, GOESGLM, ISD, IBTrACS",
            "--signal-gold",
        ),
    ]
    lanes = "\n".join(
        f"""<div class="source-lane" style="--accent: var({accent});">
          <span class="lane-label">{esc(kicker)}</span>
          <strong>{esc(title)}</strong>
          <small>{esc(sub)}</small>
        </div>"""
        for kicker, title, sub, accent in source_lanes
    )
    sensors = [
        (
            "satellite",
            "Low-Earth Orbit",
            "JPSS / MetOp",
            "--data-spectrum-c3",
            "grid-column: 3; grid-row: 1;",
        ),
        (
            "reanalysis",
            "Re-analysis systems",
            "ERA5 CDS / ARCO / NCAR",
            "--data-spectrum-c5",
            "grid-column: 5; grid-row: 1;",
        ),
        (
            "satellite",
            "Geostationary",
            "GOES / Himawari / MTG",
            "--data-spectrum-c6",
            "grid-column: 6; grid-row: 1;",
        ),
        (
            "aircraft",
            "Aircraft",
            "NNJA / GDAS",
            "--data-spectrum-c1",
            "grid-column: 1; grid-row: 1; align-self:end;",
        ),
        (
            "balloon",
            "Weather balloon",
            "NNJA / GDAS",
            "--data-spectrum-c2",
            "grid-column: 2; grid-row: 2; transform: translate(-24px, -12px);",
        ),
        (
            "buoy",
            "Ocean buoy",
            "NNJA / GDAS",
            "--data-spectrum-buoy",
            "grid-column: 3; grid-row: 2; transform: translateX(-68px);",
        ),
        (
            "radar",
            "Weather radar",
            "MRMS",
            "--data-spectrum-c7",
            "grid-column: 7; grid-row: 2;",
        ),
        (
            "supercomputer",
            "Forecast systems",
            "GFS / IFS / AIFS",
            "--data-spectrum-c8",
            "grid-column: 8; grid-row: 2; transform: translateY(-24px);",
        ),
    ]
    sensor_html = "\n".join(
        f"""<div class="sensor" style="{style} --accent: var({accent});">
          <div class="icon">{icon(icon_name)}</div>
          <span class="sensor-label">{esc(label)}</span>
          <small>{esc(sub)}</small>
        </div>"""
        for icon_name, label, sub, accent, style in sensors
    )
    return layout_page(
        "Earth2Studio comprehensive data sources",
        header(
            "Data access",
            "AI-ready Earth system data sources",
            "Massive international collection of PyData-native weather, climate, and observation feeds.",
            "cloud data sources",
        )
        + f"""
    <section class="observing-system">
      <div class="earth-horizon"></div>
      <svg class="wire-layer" viewBox="0 0 1504 154" preserveAspectRatio="none" aria-hidden="true">
        <defs>
          <marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#76b900"/>
          </marker>
          <marker id="arrow-cyan" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#0ea5a4"/>
          </marker>
          <marker id="arrow-orbit" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b5cf6"/>
          </marker>
        </defs>
        <path d="M 468 30 C 530 56 642 82 752 119" stroke="var(--data-spectrum-c3)" stroke-width="1.35" stroke-opacity=".74" fill="none" stroke-dasharray="5 8"/>
        <path d="M 836 30 C 812 54 778 86 752 119" stroke="var(--data-spectrum-c5)" stroke-width="1.25" stroke-opacity=".66" fill="none" stroke-dasharray="4 8"/>
        <path d="M 1038 30 C 950 62 842 91 752 119" stroke="var(--data-spectrum-c6)" stroke-width="1.35" stroke-opacity=".74" fill="none" stroke-dasharray="5 8"/>
        <path d="M 90 35 C 260 65 538 93 752 119" stroke="var(--data-spectrum-c1)" stroke-width="1.15" stroke-opacity=".62" fill="none" stroke-dasharray="5 9"/>
        <path d="M 260 89 C 380 103 590 112 752 119" stroke="var(--data-spectrum-c2)" stroke-width="1.05" stroke-opacity=".58" fill="none" stroke-dasharray="5 9"/>
        <path d="M 392 101 C 494 112 632 118 752 119" stroke="var(--data-spectrum-buoy)" stroke-width="1.05" stroke-opacity=".58" fill="none" stroke-dasharray="5 9"/>
        <path d="M 1225 107 C 1088 110 900 116 752 119" stroke="var(--data-spectrum-c7)" stroke-width="1.15" stroke-opacity=".62" fill="none" stroke-dasharray="5 9"/>
        <path d="M 1414 83 C 1208 104 946 116 752 119" stroke="var(--data-spectrum-c8)" stroke-width="1.05" stroke-opacity=".58" fill="none" stroke-dasharray="5 9"/>
      </svg>
      <div class="source-grid">
        {sensor_html}
      </div>
      <div class="data-api">
        <strong>Earth2Studio Data API</strong>
        <small>fetch · cache · consume</small>
      </div>
    </section>
    <section class="source-lanes">
      {lanes}
    </section>""",
    )


def model_zoo() -> str:
    """Build the model zoo artboard showing prognostic, diagnostic, and DA models."""
    sections = [
        (
            "earth2studio.models.px",
            "Prognostics",
            "Time-series forecasting models grouped by forecast horizon, from nowcasting to climate.",
            [
                ("FCN3 / Atlas / StormScope", "NVIDIA"),
                ("AIFS2 / AIFS2-ENS", "ECMWF"),
                ("GraphCast / GenCast", "Google"),
                ("ACE2 / Pangu / Aurora", "third-party models"),
            ],
            "--signal-green",
        ),
        (
            "earth2studio.models.dx",
            "Diagnostics",
            "Instantaneous models for derived quantities, downscaling, precipitation, hazards, and analysis fields.",
            [
                ("CorrDiff / Orbit2", "downscaling"),
                ("CBottle", "climate modeling"),
                ("Precip / Solar Variables", "downstream product fields"),
                ("Tropical Cyclone Utils", "TC trackers / CBottle guidance"),
            ],
            "--signal-cyan",
        ),
        (
            "earth2studio.models.da",
            "Data Assimilation",
            "Assimilate sparse observations and dense fields into grids for guided forecasts and downstream models.",
            [("HealDA", "global analysis"), ("StormCast SDA", "regional DA")],
            "--signal-gold",
        ),
    ]
    cards = "\n".join(
        f"""
      <section class="model-section" style="--accent: var({accent});">
        <span class="zone-kicker">{esc(kicker)}</span>
        <h3>{esc(title)}</h3>
        <p>{esc(desc)}</p>
        <div class="model-list">
          {"".join(component(a, b) for a, b in items)}
        </div>
      </section>"""
        for kicker, title, desc, items, accent in sections
    )
    return layout_page(
        "Earth2Studio model zoo",
        header(
            "Model zoo",
            "Largest model zoo across the Earth system AI community",
            "NVIDIA and community models for forecasting, diagnostics, and data assimilation.",
            "pre-trained models",
        )
        + f"""
    <section class="model-zoo">
      {cards}
    </section>""",
    )


def composability() -> str:
    """Build the composability artboard showing built-in, custom, and agent workflows."""
    workflows = [
        (
            "Built-in workflow",
            "run.deterministic",
            [
                ("DataSource", "GFS / HRRR / IFS", "--signal-green"),
                ("PrognosticModel", "FCN3 / GraphCast", "--signal-green"),
                ("IOBackend", "ZarrBackend / XarrayBackend", "--signal-gold"),
            ],
        ),
        (
            "Custom workflow",
            "diagnostic / downscaling",
            [
                ("DataSource", "HRRR / GFS / IFS", "--signal-green"),
                ("PrognosticModel", "AIFS2 / Atlas", "--signal-green"),
                ("DiagnosticModel", "CorrDiff / Precip-Solar", "--signal-cyan"),
                ("IOBackend", "ZarrBackend / NetCDF4Backend", "--signal-gold"),
            ],
        ),
        (
            "Agent-built workflow",
            "assembled recipe",
            [
                ("DataSource", "ARCO / CDS / WB2ERA5", "--signal-green"),
                ("Perturbation", "BredVector / LaggedEnsemble", "--signal-purple"),
                ("PrognosticModel", "GenCast / AIFS2ENS", "--signal-green"),
                ("IOBackend", "ZarrBackend / XarrayBackend", "--signal-gold"),
            ],
        ),
    ]

    def workflow_row(name: str, sub: str, nodes: list[tuple[str, str, str]]) -> str:
        parts = []
        for index, (title, detail, accent) in enumerate(nodes):
            parts.append(
                f"""          <div class="workflow-step" style="--accent: var({accent});">
            <strong>{esc(title)}</strong>
            <small>{esc(detail)}</small>
          </div>"""
            )
            if index < len(nodes) - 1:
                parts.append(
                    f"""          <div class="workflow-link" style="--accent: var({accent});"></div>"""
                )
        steps = "\n".join(parts)
        return f"""      <section class="workflow-row">
        <div class="workflow-run">
          <strong>{esc(name)}</strong>
          <small>{esc(sub)}</small>
        </div>
        <div class="workflow-chain">
{steps}
        </div>
      </section>"""

    rows = "\n".join(workflow_row(name, sub, nodes) for name, sub, nodes in workflows)

    return layout_page(
        "Earth2Studio composable pipelines",
        header(
            "Composability",
            "Connect data and models into inference workflows",
            "Start with built-in workflows, compose custom pipelines, or let agents assemble reusable recipes.",
            "built-in · custom · agent-built",
        )
        + f"""
    <section class="workflow-stack">
{rows}
    </section>
    <section class="workflow-notes">
      <div class="flow-note" style="--accent: var(--signal-green)">swap data sources without changing the runner</div>
      <div class="flow-note" style="--accent: var(--signal-cyan)">chain prognostic and diagnostic models</div>
      <div class="flow-note" style="--accent: var(--signal-gold)">write outputs through standard IO backends</div>
    </section>""",
    )


PAGES = {
    "earth2studio-readme-hero": hero,
    "earth2studio-readme-quickstart-video": quickstart_video,
    "earth2studio-readme-agent-setup": agent_setup,
    "earth2studio-readme-data-sources": datasource,
    "earth2studio-readme-model-zoo": model_zoo,
    "earth2studio-readme-composability": composability,
}


def find_browser() -> Path:
    """Locate a Chromium-based browser for headless PNG export.

    Searches PATH first using ``shutil.which``, then falls back to
    platform-specific install paths for Chrome, Chromium, and Edge.

    Returns
    -------
    Path
        Resolved path to the browser executable.

    Raises
    ------
    FileNotFoundError
        If no supported browser is found.
    """
    # First, try to find browser on PATH (most reliable cross-platform)
    for name in BROWSER_PATH_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    # Fall back to platform-specific install paths
    for candidate in CHROME_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Chrome, Chromium, or Edge for HTML artboard export. "
        "Install one of: google-chrome, chromium-browser, chromium, or msedge."
    )


def export_pngs() -> None:
    """Export each HTML artboard to a 2x PNG screenshot using a headless browser."""
    browser = find_browser()
    for slug in PAGES:
        html_path = OUT_DIR / f"{slug}.html"
        png_path = OUT_DIR / f"{slug}.png"
        subprocess.run(  # noqa: S603
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                "--default-background-color=00000000",
                "--virtual-time-budget=9000",
                f"--force-device-scale-factor={EXPORT_SCALE}",
                f"--window-size={W},{H}",
                f"--screenshot={png_path}",
                html_path.resolve().as_uri(),
            ],
            check=True,
            cwd=ROOT,
        )


def write_png_bundle() -> None:
    """Package all exported PNG screenshots into a ZIP archive."""
    bundle_path = OUT_DIR / "earth2studio-readme-graphics-png.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for slug in PAGES:
            png_path = OUT_DIR / f"{slug}.png"
            if png_path.exists():
                bundle.write(png_path, png_path.name)


def write_review() -> None:
    """Generate an HTML review page with side-by-side iframe previews of all artboards."""
    cards = []
    for slug in PAGES:
        title = slug.replace("earth2studio-readme-", "").replace("-", " ").title()
        cards.append(
            f"""      <section>
        <header><h2>{esc(title)}</h2><a href="./{slug}.html?v={ASSET_VERSION}" target="_blank">Open HTML</a><a href="./{slug}.png?v={ASSET_VERSION}" target="_blank">Open PNG</a><a href="./{slug}.png?v={ASSET_VERSION}" download="{slug}.png">Download PNG</a></header>
        <div class="viewport"><iframe src="./{slug}.html?v={ASSET_VERSION}" title="{esc(title)}"></iframe></div>
      </section>"""
        )
    review_css = """
    :root { color-scheme: dark; --green:#76b900; --link:#9cff2e; }
    body[data-preview-theme="dark"] { color-scheme: dark; --bg:#0c0f0b; --panel:#10140f; --card:#131711; --text:#f4f7ef; --muted:#a5ae9e; --border:#293024; --secondary:#1d241a; --preview-bg:#0c0f0b; --preview-grid:rgba(244,247,239,.045); --shadow:rgba(0,0,0,.28); }
    body[data-preview-theme="light"] { color-scheme: light; --bg:#f4f7ef; --panel:#e8eee0; --card:#ffffff; --text:#10140d; --muted:#53604b; --border:#cdd8c2; --secondary:#e8eee0; --preview-bg:#f4f7ef; --preview-grid:rgba(41,48,36,.12); --shadow:rgba(16,20,13,.16); }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: "NVIDIA Sans", Arial, Helvetica, sans-serif; transition: background .18s ease, color .18s ease; }
    main { width: min(1660px, calc(100vw - 48px)); margin: 32px auto 56px; }
    h1 { margin: 0 0 8px; font-size: 30px; line-height: 1.1; }
    p { margin: 0 0 26px; color: var(--muted); }
    .top-row { display: flex; align-items: flex-end; gap: 18px; margin: 0 0 26px; }
    .intro { flex: 1; min-width: 0; }
    .intro p { margin: 0; }
    .controls { flex: 0 0 auto; display: flex; align-items: center; gap: 12px; }
    .theme-toggle { display: inline-flex; gap: 3px; padding: 3px; border: 1px solid var(--border); border-radius: 999px; background: var(--secondary); }
    .theme-toggle button { min-width: 72px; height: 32px; border: 0; border-radius: 999px; background: transparent; color: var(--muted); font: 700 13px/1 "NVIDIA Sans", Arial, Helvetica, sans-serif; cursor: pointer; }
    .theme-toggle button[aria-pressed="true"] { background: var(--green); color: #10140d; }
    .download-all { flex: 0 0 auto; padding: 9px 14px; border: 1px solid var(--green); border-radius: 999px; background: var(--secondary); }
    section { margin: 0 0 28px; padding: 14px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 18px 42px var(--shadow); }
    header { display: flex; align-items: center; gap: 12px; margin: 0 0 12px; }
    h2 { flex: 1; margin: 0; font-size: 18px; font-weight: 700; }
    a { color: var(--link); text-decoration: none; font-weight: 700; font-size: 13px; }
    body[data-preview-theme="light"] a { color: #4f8500; }
    .viewport {
      width: 100%;
      overflow: auto;
      border-radius: 8px;
      background-color: var(--preview-bg);
      background-image:
        linear-gradient(var(--preview-grid) 1px, transparent 1px),
        linear-gradient(90deg, var(--preview-grid) 1px, transparent 1px);
      background-size: 40px 40px;
    }
    iframe { width: 1600px; height: 460px; display: block; border: 0; background: var(--preview-bg); }
    """
    (OUT_DIR / "review.html").write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Earth2Studio README Graphics Review</title>
  <style>{review_css}</style>
</head>
<body data-preview-theme="dark">
  <main>
    <div class="top-row">
      <div class="intro">
        <h1>Earth2Studio README Graphics</h1>
        <p>CSS-structured 1600x460 artboards based on the NVIDIA docs arch-product-diagram style.</p>
      </div>
      <div class="controls">
        <div class="theme-toggle" role="group" aria-label="Preview theme">
          <button type="button" data-theme="dark" aria-pressed="true">Dark</button>
          <button type="button" data-theme="light" aria-pressed="false">Light</button>
        </div>
        <a class="download-all" href="./earth2studio-readme-graphics-png.zip?v={ASSET_VERSION}" download="earth2studio-readme-graphics-png.zip">Download PNG bundle</a>
      </div>
    </div>
{"".join(cards)}
  </main>
  <script>
    const buttons = Array.from(document.querySelectorAll("[data-theme]"));
    const setTheme = (theme) => {{
      document.body.dataset.previewTheme = theme;
      buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.theme === theme)));
      localStorage.setItem("earth2studio-readme-preview-theme", theme);
    }};
    buttons.forEach((button) => button.addEventListener("click", () => setTheme(button.dataset.theme)));
    setTheme(localStorage.getItem("earth2studio-readme-preview-theme") || "dark");
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    """CLI entry point: build HTML artboards and optionally export PNGs."""
    parser = argparse.ArgumentParser(description="Build Earth2Studio README graphics.")
    parser.add_argument(
        "--export-png",
        action="store_true",
        help="Export PNGs at 2x resolution and create the PNG bundle.",
    )
    args = parser.parse_args()

    (OUT_DIR / "earth2studio-graphics.css").write_text(CSS, encoding="utf-8")
    for slug, builder in PAGES.items():
        (OUT_DIR / f"{slug}.html").write_text(builder(), encoding="utf-8")
    write_review()
    manifest = {
        "project": "Earth2Studio README graphics",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dimensions": {"width": W, "height": H, "aspectRatio": "1600:460"},
        "pngExportScale": EXPORT_SCALE,
        "pngDimensions": {"width": W * EXPORT_SCALE, "height": H * EXPORT_SCALE},
        "source": "CSS structured HTML artboards",
        "styleReference": "NVIDIA docs arch-product-diagram",
        "assets": [
            {
                "title": slug.replace("earth2studio-readme-", "")
                .replace("-", " ")
                .title(),
                "html": f"{slug}.html",
                "png": f"{slug}.png",
            }
            for slug in PAGES
        ],
        "references": [
            "https://skill-eval-ci-7fdbb4.gitlab-master-pages.nvidia.com/docs/architecture/",
            "https://nvidia.github.io/earth2studio/modules/datasources_analysis.html",
            "https://nvidia.github.io/earth2studio/modules/datasources_forecast.html",
            "https://nvidia.github.io/earth2studio/modules/datasources_dataframe.html",
            "https://nvidia.github.io/earth2studio/userguide/components/prognostic.html",
            "https://nvidia.github.io/earth2studio/userguide/components/diagnostic.html",
            "https://nvidia.github.io/earth2studio/modules/models_da.html",
        ],
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    if args.export_png:
        export_pngs()
        write_png_bundle()


if __name__ == "__main__":
    main()

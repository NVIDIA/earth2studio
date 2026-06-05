from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "earth2studio-readme-graphics-css"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1600, 400
ASSET_VERSION = "dark-v65"


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
  height: 400px;
  margin: 0;
  overflow: hidden;
  background: var(--background);
  font-family: var(--font-sans);
  color: var(--foreground);
}

.artboard {
  position: relative;
  width: 1600px;
  height: 400px;
  overflow: hidden;
  padding: 30px 48px;
  background:
    radial-gradient(circle at 10% 8%, var(--green-wash), transparent 34%),
    radial-gradient(circle at 82% 16%, var(--cyan-wash), transparent 30%),
    var(--background);
  border: 1px solid color-mix(in srgb, var(--signal-green) 28%, var(--border));
  border-radius: 8px;
  box-shadow: 0 24px 70px rgba(0, 0, 0, .16);
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

.artboard > * { position: relative; z-index: 1; }

.graphic-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  column-gap: 32px;
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid color-mix(in srgb, var(--border) 72%, transparent);
}

.kicker,
.zone-kicker,
.sensor-label,
.lane-label {
  color: var(--signal-green);
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.1;
  font-weight: 820;
  letter-spacing: 0;
  text-transform: uppercase;
}

.graphic-title {
  margin: 7px 0 0;
  font-size: 32px;
  line-height: 1.04;
  font-weight: 800;
}

.graphic-subtitle {
  margin: 7px 0 0;
  max-width: 760px;
  color: var(--ink-soft);
  font-size: 15px;
  line-height: 1.45;
}

.header-pill {
  display: inline-grid;
  min-width: 250px;
  height: 36px;
  align-items: center;
  justify-content: center;
  padding: 0 20px;
  border: 1px solid var(--signal-green);
  border-radius: 999px;
  background: var(--secondary);
  color: var(--ink-soft);
  font-size: 14px;
  line-height: 1;
  font-weight: 700;
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
  min-height: 168px;
  padding: 14px;
  border: 1px solid color-mix(in srgb, var(--border) 76%, transparent);
  border-top: 4px solid var(--accent, var(--signal-green));
  border-radius: 8px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent, var(--signal-green)) 7%, transparent), transparent 42%),
    var(--card);
}

.zone h3 {
  margin: 9px 0 0;
  color: var(--foreground);
  font-size: 18px;
  line-height: 1.15;
  font-weight: 760;
}

.zone p {
  margin: 8px 0 0;
  color: var(--muted-foreground);
  font-size: 12px;
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
  padding: 9px 10px;
  border: 1px solid color-mix(in srgb, var(--border) 76%, transparent);
  border-radius: 8px;
  background: var(--muted);
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
  font-size: 13px;
  line-height: 1.25;
  font-weight: 740;
}

.component small {
  margin-top: 5px;
  color: var(--muted-foreground);
  font-size: 11px;
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
  border: 1px solid var(--accent, var(--signal-green));
  border-radius: 999px;
  background: var(--secondary);
  color: var(--muted-foreground);
  font-size: 12px;
  line-height: 1;
  font-weight: 720;
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
  border: 1px solid var(--accent, var(--signal-green));
  border-radius: 8px;
  background: color-mix(in srgb, var(--card) 88%, var(--accent, var(--signal-green)) 12%);
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
  font-size: 16px;
  line-height: 1.15;
  font-weight: 780;
}

.node small {
  margin-top: 6px;
  color: var(--muted-foreground);
  font-size: 11px;
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
  border: 1px solid var(--accent, var(--signal-green));
  border-radius: 999px;
  background: var(--secondary);
  color: var(--muted-foreground);
  font-size: 13px;
  line-height: 1;
  font-weight: 720;
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
  height: 55px;
  display: grid;
  align-content: center;
  padding: 8px 12px;
  border: 1px solid color-mix(in srgb, var(--signal-green) 56%, var(--border));
  border-left: 4px solid var(--signal-green);
  border-radius: 8px;
  background:
    linear-gradient(90deg, color-mix(in srgb, var(--signal-green) 12%, transparent), transparent 72%),
    var(--secondary);
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
  font-size: 13px;
  line-height: 1.1;
  font-weight: 820;
}

.workflow-run small {
  margin-top: 5px;
  color: var(--muted-foreground);
  font-size: 10.5px;
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
  height: 55px;
  flex: 1 1 0;
  display: grid;
  align-content: center;
  padding: 8px 14px;
  border: 1px solid color-mix(in srgb, var(--accent, var(--signal-green)) 58%, var(--border));
  border-radius: 8px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent, var(--signal-green)) 10%, transparent), transparent 62%),
    color-mix(in srgb, var(--card) 86%, var(--accent, var(--signal-green)) 14%);
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
  font-size: 14px;
  line-height: 1.15;
  font-weight: 800;
}

.workflow-step small {
  margin-top: 5px;
  color: var(--muted-foreground);
  font-size: 10.5px;
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
  min-height: 150px;
  padding: 14px;
  border: 1px solid color-mix(in srgb, var(--border) 76%, transparent);
  border-top: 4px solid var(--accent, var(--signal-green));
  border-radius: 8px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent, var(--signal-green)) 8%, transparent), transparent 44%),
    var(--card);
}

.core-card h3 {
  margin: 9px 0 0;
  color: var(--foreground);
  font-size: 18px;
  line-height: 1.08;
  font-weight: 800;
}

.core-card p {
  margin: 8px 0 0;
  min-height: 30px;
  color: var(--muted-foreground);
  font-size: 12px;
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
  background: var(--secondary);
  color: var(--muted-foreground);
  font-family: var(--font-mono);
  font-size: 11px;
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

.observing-system {
  position: relative;
  height: 154px;
  margin-top: -5px;
}

.earth-horizon {
  position: absolute;
  left: 34px;
  right: 34px;
  bottom: -92px;
  height: 168px;
  border: 1px solid color-mix(in srgb, var(--signal-cyan) 46%, var(--border));
  border-radius: 50% 50% 0 0 / 100% 100% 0 0;
  background:
    radial-gradient(ellipse at 32% 8%, rgba(118, 185, 0, .24), transparent 34%),
    linear-gradient(90deg, #102522 0 42%, #162914 42% 100%);
  overflow: hidden;
}

.earth-horizon::before {
  content: "";
  position: absolute;
  inset: 36px 0 auto 0;
  height: 1px;
  background: color-mix(in srgb, var(--signal-green) 48%, transparent);
  transform: rotate(-2deg);
}

.earth-horizon::after {
  content: "";
  position: absolute;
  left: -2%;
  right: -2%;
  top: 56px;
  height: 70px;
  background: #162914;
  clip-path: polygon(0 64%, 9% 48%, 18% 52%, 28% 34%, 39% 45%, 50% 26%, 61% 42%, 72% 32%, 86% 48%, 100% 38%, 100% 100%, 0 100%);
}

.source-grid {
  position: absolute;
  inset: 6px 0 0;
  display: grid;
  grid-template-columns: repeat(8, minmax(0, 1fr));
  grid-template-rows: 70px 78px;
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
  font-size: 9px;
  line-height: 1.15;
}

.sensor small {
  display: block;
  margin-top: 2px;
  color: var(--muted-foreground);
  text-align: center;
  font-size: 9px;
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
  background: var(--card);
}

.data-api strong {
  font-size: 17px;
  line-height: 1.1;
}

.data-api small {
  margin-top: 5px;
  color: var(--muted-foreground);
  font-size: 12px;
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
  padding: 8px 10px;
  border: 1px dashed color-mix(in srgb, var(--accent, var(--signal-green)) 42%, var(--border));
  border-radius: 8px;
  background: color-mix(in srgb, var(--muted) 78%, transparent);
}

.source-lane strong {
  display: block;
  margin-top: 5px;
  overflow: hidden;
  color: var(--foreground);
  font-size: 13px;
  line-height: 1.1;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.source-lane small {
  display: block;
  margin-top: 4px;
  overflow: hidden;
  color: var(--muted-foreground);
  font-size: 9.5px;
  line-height: 1.25;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.model-zoo {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}

.model-section {
  min-width: 0;
  min-height: 190px;
  padding: 14px;
  border: 1px solid color-mix(in srgb, var(--border) 76%, transparent);
  border-top: 4px solid var(--accent);
  border-radius: 8px;
  background:
    linear-gradient(180deg, color-mix(in srgb, var(--accent) 8%, transparent), transparent 44%),
    var(--card);
}

.model-section h3 {
  margin: 8px 0 0;
  font-size: 18px;
  line-height: 1.15;
}

.model-section p {
  height: 31px;
  margin: 7px 0 0;
  color: var(--muted-foreground);
  font-size: 12px;
  line-height: 1.38;
}

.model-list {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 8px;
  margin-top: 9px;
}

.footer-axis {
  margin-top: 15px;
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  align-items: center;
  gap: 16px;
  color: var(--muted-foreground);
  font-size: 13px;
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
"""


ICONS = {
    "satellite": """<svg viewBox="0 0 100 70" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><rect x="39" y="27" width="22" height="16" rx="3" fill="currentColor" fill-opacity=".12"/><path d="M39 35 23 24M39 35 23 47M61 35 77 24M61 35 77 47"/><rect x="4" y="15" width="20" height="14" rx="2" fill="currentColor" fill-opacity=".12"/><rect x="4" y="42" width="20" height="14" rx="2" fill="currentColor" fill-opacity=".12"/><rect x="76" y="15" width="20" height="14" rx="2" fill="currentColor" fill-opacity=".12"/><rect x="76" y="42" width="20" height="14" rx="2" fill="currentColor" fill-opacity=".12"/><path d="M44 43 36 62M56 43 64 62"/></svg>""",
    "aircraft": """<svg viewBox="0 0 110 70" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linejoin="round"><path d="M6 37 82 14c12-4 23 3 22 10-1 5-6 8-15 10l-25 6 17 21-14 4-25-20-28 7-12 14-10-4 11-18-17-4Z" fill="currentColor" fill-opacity=".12"/><path d="m42 43 38-10"/></svg>""",
    "radar": """<svg viewBox="0 0 90 70" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M19 26a26 26 0 0 1 52 0"/><path d="M9 19a40 40 0 0 1 72 0" opacity=".55"/><path d="M0 12a54 54 0 0 1 90 0" opacity=".32"/><circle cx="45" cy="34" r="7" fill="currentColor" fill-opacity=".15"/><path d="M45 41 33 66h24L45 41ZM25 66h40"/></svg>""",
    "balloon": """<svg viewBox="0 0 70 84" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><path d="M35 4c18 0 27 15 24 31-3 17-14 28-24 28S14 52 11 35C8 19 17 4 35 4Z" fill="currentColor" fill-opacity=".12"/><path d="M24 62 35 76l11-14M35 76v8M24 84h22"/></svg>""",
    "buoy": """<svg viewBox="0 0 86 80" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M22 58 43 13l21 45Z" fill="currentColor" fill-opacity=".12"/><circle cx="43" cy="8" r="7" fill="currentColor" fill-opacity=".18"/><path d="M30 38h26M26 52h34M4 68c16-11 30 11 44 0s22 2 34 0" opacity=".72"/></svg>""",
    "supercomputer": """<svg viewBox="0 0 92 70" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><rect x="8" y="8" width="76" height="52" rx="5" fill="currentColor" fill-opacity=".10"/><path d="M27 8v52M46 8v52M65 8v52M8 25h76M8 43h76"/><circle cx="18" cy="17" r="2.5" fill="currentColor"/><circle cx="37" cy="17" r="2.5" fill="currentColor"/><circle cx="56" cy="17" r="2.5" fill="currentColor"/><circle cx="75" cy="17" r="2.5" fill="currentColor"/></svg>""",
    "station": """<svg viewBox="0 0 90 70" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><rect x="16" y="36" width="48" height="25" rx="3" fill="currentColor" fill-opacity=".10"/><path d="M16 36 40 18l24 18M67 17v44M58 17h18M62 8h10M45 61V46H34v15"/></svg>""",
    "reanalysis": """<svg viewBox="0 0 92 70" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"><rect x="24" y="16" width="44" height="34" rx="4" fill="currentColor" fill-opacity=".10"/><path d="M24 27h44M24 39h44M35 16v34M46 16v34M57 16v34"/><path d="M30 11h32M18 55h56M23 60h46" opacity=".68"/><circle cx="46" cy="33" r="4" fill="currentColor" fill-opacity=".22"/></svg>""",
}


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def icon(name: str) -> str:
    return ICONS[name]


def component(name: str, sub: str) -> str:
    return f'<div class="component"><strong>{esc(name)}</strong><small>{esc(sub)}</small></div>'


def zone(kicker: str, title: str, body: str, items: list[tuple[str, str]], accent: str, two: bool = False) -> str:
    grid_class = "component-grid two" if two else "component-grid"
    return f"""
      <section class="zone" style="--accent: var({accent});">
        <span class="zone-kicker">{esc(kicker)}</span>
        <h3>{esc(title)}</h3>
        <p>{esc(body)}</p>
        <div class="{grid_class}">
          {''.join(component(a, b) for a, b in items)}
        </div>
      </section>"""


def header(kicker: str, title: str, subtitle: str, pill: str) -> str:
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
    core = [
        ("Models", "Pre-trained model zoo", "Forecast, diagnose, and assimilate with built-in AI models.", "phase: model"),
        ("Data", "AI-ready remote data sources", "PyData-native loaders for weather, climate, and observations.", "phase: fetch"),
        ("APIs", "Composable APIs", "Chain data, models, perturbations, statistics, and IO.", "phase: compose"),
        ("Compute", "GPU accelerated", "Run inference and evaluation on NVIDIA-optimized workflows.", "phase: run"),
        ("Agents", "Agent ready", "Structured interfaces for recipes, skills, automation, and AI assistants.", "phase: automate"),
    ]
    accents = ["--signal-green", "--signal-cyan", "--signal-gold", "--signal-blue", "--signal-purple"]
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
        header("NVIDIA", "Earth2Studio", "A Python package for building, researching, and exploring AI-driven Earth system models.", "Earth-2")
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


def datasource() -> str:
    source_lanes = [
        ("Data sources", "ARCO, CDS, CMIP6, GOES, MRMS", "GFS, HRRR, IFS, JPSS, WB2ERA5", "--signal-green"),
        ("Forecast sources", "AIFS_FX, GFS_FX, GEFS_FX", "HRRR_FX, IFS_FX, CFS_FX, CAMS_FX", "--signal-cyan"),
        ("DataFrame sources", "UFS, NNJA, JPSS, MetOp", "GHCNDaily, GOESGLM, ISD, IBTrACS", "--signal-gold"),
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
        ("satellite", "Low-Earth Orbit", "JPSS / MetOp", "--data-spectrum-c3", "grid-column: 3; grid-row: 1;"),
        ("reanalysis", "Re-analysis systems", "ERA5 CDS / ARCO / NCAR", "--data-spectrum-c5", "grid-column: 5; grid-row: 1;"),
        ("satellite", "Geostationary", "GOES / Himawari / MTG", "--data-spectrum-c6", "grid-column: 6; grid-row: 1;"),
        ("aircraft", "Aircraft", "NNJA / GDAS", "--data-spectrum-c1", "grid-column: 1; grid-row: 1; align-self:end;"),
        ("balloon", "Weather balloon", "NNJA / GDAS", "--data-spectrum-c2", "grid-column: 2; grid-row: 2;"),
        ("buoy", "Ocean buoy", "NNJA / GDAS", "--data-spectrum-c3", "grid-column: 3; grid-row: 2;"),
        ("radar", "Weather radar", "MRMS", "--data-spectrum-c7", "grid-column: 7; grid-row: 2;"),
        ("supercomputer", "Forecast systems", "GFS / IFS / AIFS", "--data-spectrum-c8", "grid-column: 8; grid-row: 2;"),
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
            "DataArray / DataFrame native",
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
        <path d="M 468 30 C 526 58 586 80 656 99" stroke="var(--data-spectrum-c3)" stroke-width="1.35" stroke-opacity=".74" fill="none" stroke-dasharray="5 8" marker-end="url(#arrow-orbit)"/>
        <path d="M 836 30 C 808 58 778 78 752 97" stroke="var(--data-spectrum-c5)" stroke-width="1.25" stroke-opacity=".66" fill="none" stroke-dasharray="4 8"/>
        <path d="M 1038 30 C 984 56 916 80 852 99" stroke="var(--data-spectrum-c6)" stroke-width="1.35" stroke-opacity=".74" fill="none" stroke-dasharray="5 8" marker-end="url(#arrow-cyan)"/>
        <path d="M 90 35 C 252 68 430 96 596 111" stroke="var(--data-spectrum-c1)" stroke-width="1.15" stroke-opacity=".62" fill="none" stroke-dasharray="5 9"/>
        <path d="M 284 101 C 392 113 500 119 596 125" stroke="var(--data-spectrum-c2)" stroke-width="1.05" stroke-opacity=".58" fill="none" stroke-dasharray="5 9"/>
        <path d="M 470 101 C 520 112 568 126 634 138" stroke="var(--data-spectrum-c3)" stroke-width="1.05" stroke-opacity=".58" fill="none" stroke-dasharray="5 9"/>
        <path d="M 1225 107 C 1108 108 1002 113 910 118" stroke="var(--data-spectrum-c7)" stroke-width="1.15" stroke-opacity=".62" fill="none" stroke-dasharray="5 9"/>
        <path d="M 1414 107 C 1220 118 1058 129 910 136" stroke="var(--data-spectrum-c8)" stroke-width="1.05" stroke-opacity=".58" fill="none" stroke-dasharray="5 9"/>
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
    sections = [
        (
            "earth2studio.models.px",
            "Prognostics",
            "Time-series forecasting models grouped by forecast horizon, from nowcasting to climate.",
            [("FCN3 / Atlas / StormScope", "NVIDIA"), ("AIFS2 / AIFS2-ENS", "ECMWF"), ("GraphCast / GenCast", "Google"), ("ACE2 / Pangu / Aurora", "third-party models")],
            "--signal-green",
        ),
        (
            "earth2studio.models.dx",
            "Diagnostics",
            "Instantaneous models for derived quantities, downscaling, precipitation, hazards, and analysis fields.",
            [("CorrDiff / Orbit2", "downscaling"), ("CBottle", "climate modeling"), ("Precip / Solar Variables", "downstream product fields"), ("Tropical Cyclone Utils", "TC trackers / CBottle guidance")],
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
          {''.join(component(a, b) for a, b in items)}
        </div>
      </section>"""
        for kicker, title, desc, items, accent in sections
    )
    return layout_page(
        "Earth2Studio model zoo",
        header(
            "Model zoo",
            "Largest model zoo across the Earth system AI community",
            "NVIDIA and community models for forecasting, diagnostics, and data simulation.",
            "Pre-Trained Models",
        )
        + f"""
    <section class="model-zoo">
      {cards}
    </section>
    <div class="footer-axis">one model interface across forecast, derived-field, and observation-constrained workflows</div>""",
    )


def composability() -> str:
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
                parts.append(f"""          <div class="workflow-link" style="--accent: var({accent});"></div>""")
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
    "earth2studio-readme-data-sources": datasource,
    "earth2studio-readme-model-zoo": model_zoo,
    "earth2studio-readme-composability": composability,
}


def write_review() -> None:
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
    :root { color-scheme: dark; --bg:#0c0f0b; --panel:#10140f; --card:#131711; --text:#f4f7ef; --muted:#a5ae9e; --border:#293024; --green:#76b900; --secondary:#1d241a; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: "NVIDIA Sans", Arial, Helvetica, sans-serif; }
    main { width: min(1660px, calc(100vw - 48px)); margin: 32px auto 56px; }
    h1 { margin: 0 0 8px; font-size: 30px; line-height: 1.1; }
    p { margin: 0 0 26px; color: var(--muted); }
    .top-row { display: flex; align-items: flex-end; gap: 18px; margin: 0 0 26px; }
    .intro { flex: 1; min-width: 0; }
    .intro p { margin: 0; }
    .download-all { flex: 0 0 auto; padding: 9px 14px; border: 1px solid var(--green); border-radius: 999px; background: var(--secondary); }
    section { margin: 0 0 28px; padding: 14px; background: var(--card); border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 18px 42px rgba(0,0,0,.28); }
    header { display: flex; align-items: center; gap: 12px; margin: 0 0 12px; }
    h2 { flex: 1; margin: 0; font-size: 18px; font-weight: 700; }
    a { color: #9cff2e; text-decoration: none; font-weight: 700; font-size: 13px; }
    .viewport { width: 100%; overflow: auto; border-radius: 8px; background: var(--panel); }
    iframe { width: 1600px; height: 400px; display: block; border: 0; background: var(--panel); }
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
<body>
  <main>
    <div class="top-row">
      <div class="intro">
        <h1>Earth2Studio README Graphics</h1>
        <p>CSS-structured 1600x400 artboards based on the NVIDIA docs arch-product-diagram style.</p>
      </div>
      <a class="download-all" href="./earth2studio-readme-graphics-png.zip?v={ASSET_VERSION}" download="earth2studio-readme-graphics-png.zip">Download PNG bundle</a>
    </div>
{''.join(cards)}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    (OUT_DIR / "earth2studio-graphics.css").write_text(CSS, encoding="utf-8")
    for slug, builder in PAGES.items():
        (OUT_DIR / f"{slug}.html").write_text(builder(), encoding="utf-8")
    write_review()
    manifest = {
        "project": "Earth2Studio README graphics",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "dimensions": {"width": W, "height": H, "aspectRatio": "16:4"},
        "source": "CSS structured HTML artboards",
        "styleReference": "NVIDIA docs arch-product-diagram",
        "assets": [
            {"title": slug.replace("earth2studio-readme-", "").replace("-", " ").title(), "html": f"{slug}.html", "png": f"{slug}.png"}
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
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
